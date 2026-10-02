"""Gates in fusion_screen.py."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fusion_screen as screen

COPY = "at::native::elementwise_kernel<direct_copy_kernel_cuda>"
QK_CATALOG = {"kernels": [
    {"name": "fused_qk_norm_mrope_3d_cache_pts_quant_shuffle",
     "op_tags": ["kv_cache", "layout", "norm", "quant", "rope"], "dtype_tags": [], "sources": ["aiter"]},
    {"name": "fused_qk_norm_rope_cache_quant_shuffle",
     "op_tags": ["kv_cache", "layout", "norm", "quant", "rope"], "dtype_tags": [], "sources": ["aiter"]},
]}


def _row(row_id, name, duration, **extra):
    item = {
        "row_id": row_id,
        "raw_name": name,
        "short_name": name,
        "duration_us": duration,
        "phase": "decode",
        "pattern_id": "p0",
        "pattern_layer_count": 40,
        "stream": 0,
        "layer_id": 1,
        "shape": {"input_dims": [], "input_types": ["bf16"]},
    }
    item.update(extra)
    return item


def _table(rows, layer_total):
    return {
        "tables": [{
            "phase": "decode",
            "pattern_id": "p0",
            "pattern_layer_count": 40,
            "layer_total_us": layer_total,
            "rows": rows,
        }]
    }


class ScreenTest(unittest.TestCase):
    def test_small_chain_passes_and_large_epilogue_needs_115(self):
        # One layer: four 40us quant kernels (7% of the step) plus a long GEMM
        # whose epilogue is too small to clear 1.15x.
        quants = [_row("q%d" % i, "quant_cast", 40, device_seq_index=i) for i in range(4)]
        gemm = _row("g", "gemm_bf16", 2000, device_seq_index=4, shape={
            "input_dims": [[8192, 5120, 4096]], "input_types": ["bf16"]})
        epi = _row("e", "quant_cast", 80, device_seq_index=5)
        rows = screen.load_rows(_table(quants + [gemm, epi], 2000 + 160 + 80))
        result = screen.screen(rows, gfx="gfx942")
        names = [item["kernel_names"] for item in result["initial_topk"]]
        self.assertIn(["quant_cast", "quant_cast", "quant_cast", "quant_cast"], names)
        self.assertNotIn(["gemm_bf16", "quant_cast"], names)
        self.assertTrue(result["kernel_class"]["g"]["large"])
        self.assertFalse(result["kernel_class"]["q0"]["large"])

    def test_under_3_percent_is_dropped(self):
        rows = screen.load_rows(_table([
            _row("a", "silu", 10, device_seq_index=0),
            _row("b", "quant_cast", 10, device_seq_index=1),
            _row("rest", "gemm_bf16", 5000, device_seq_index=2, shape={
                "input_dims": [[8192, 5120, 4096]], "input_types": ["bf16"]}),
        ], 5020))
        result = screen.screen(rows)
        self.assertEqual(result["initial_topk"], [])

    def test_conflict_keeps_higher_score_and_caps_at_8(self):
        rows = []
        # First four kernels are one adjacent run, so the 4-kernel window shares
        # rows with its shorter subsets. The rest are disjoint pairs.
        for index in range(4):
            rows.append(_row("h%d" % index, "quant_cast" if index % 2 == 0 else "silu",
                             40, device_seq_index=index, timestamp=index * 40))
        for index in range(8):
            base = 10000 + index * 1000
            rows.append(_row("a%d" % index, "quant_cast", 40,
                             device_seq_index=10 + index * 2, timestamp=base))
            rows.append(_row("b%d" % index, "silu", 40,
                             device_seq_index=11 + index * 2, timestamp=base + 40))
        layer_total = sum(row["duration_us"] for row in rows)
        result = screen.screen(screen.load_rows(_table(rows, layer_total)), top_k=8)
        self.assertEqual(len(result["execution_list"]), 8)
        taken = []
        for item in result["execution_list"]:
            self.assertFalse(set(item["row_ids"]) & set(taken))
            taken.extend(item["row_ids"])
        kept_heads = [item["row_ids"] for item in result["execution_list"]
                      if "h0" in item["row_ids"]]
        self.assertEqual(kept_heads, [["h0", "h1", "h2", "h3"]])
        self.assertTrue(all(item["difficulty"] == 1.3 for item in result["execution_list"]))
        self.assertTrue(all(item["pass_ranks"] == [item["pass_ranks"][0]] * 2
                            for item in result["execution_list"]))

    def test_existing_kernel_uses_difficulty_1(self):
        rows = screen.load_rows(_table([
            _row("a", "rmsnorm", 40, device_seq_index=0, timestamp=0),
            _row("b", "quant_fp8", 40, device_seq_index=1, timestamp=40),
            _row("pad", "copy_layout", 2000, device_seq_index=2, timestamp=5000),
        ], 2080))
        catalog = {"kernels": [{"name": "fused_rmsnorm_quant_fp8"}]}
        result = screen.screen(rows, catalog=catalog)
        chain = next(item for item in result["initial_topk"] if item["kernel_names"] == ["rmsnorm", "quant_fp8"])
        self.assertEqual(chain["difficulty"], 1.0)
        self.assertTrue(chain["has_existing_kernel"])
        final = next(item for item in result["execution_list"] if item["kernel_names"] == ["rmsnorm", "quant_fp8"])
        self.assertTrue(final["compare_existing"])
        self.assertTrue(final["self_author"])

    def test_merge_prefers_both_passes_then_rank_sum(self):
        def chain(name, score, row_ids):
            return {"candidate_id": name, "score": score, "row_ids": row_ids}
        a = chain("a", 9, ["r1"])
        b = chain("b", 8, ["r2"])
        c = chain("c", 7, ["r3"])
        d = chain("d", 6, ["r3", "r4"])
        merged = screen.merge_passes([[a, c, b], [d, b]], top_k=3)
        # b is in both passes (3+2). Single-pass rank sums, missing = K+1 = 4:
        # a 1+4, d 4+1, c 2+4. c then shares r3 with d and is dropped.
        self.assertEqual([item["candidate_id"] for item in merged], ["b", "a", "d"])
        self.assertEqual(merged[0]["pass_ranks"], [3, 2])
        self.assertEqual(merged[1]["pass_ranks"], [1, 4])

    def test_parallel_window_requires_distinct_streams(self):
        rows = screen.load_rows(_table([
            _row("a", "quant_cast", 30, stream=1, timestamp=0, device_seq_index=0),
            _row("b", "silu", 30, stream=2, timestamp=5, device_seq_index=1),
            _row("c", "quant_cast", 30, stream=1, timestamp=1000, device_seq_index=2),
            _row("pad", "copy_layout", 800, stream=0, timestamp=2000, device_seq_index=3),
        ], 890))
        result = screen.screen(rows)
        kinds = {tuple(item["kernel_names"]): item["kind"] for item in result["initial_topk"]}
        self.assertEqual(kinds.get(("quant_cast", "silu")), "parallel")

    def test_qk_norm_rope_grows_to_kv_write_with_existing_kernel(self):
        # Qwen3 decode attention prologue: the covering aiter kernel also writes the
        # KV cache, so the chain grows past 4 compute kernels to store_kvcache and
        # never swallows the paged-attention barrier behind it.
        names = [("c0", COPY, 4.5), ("qn", "aiter::add_rmsnorm_quant_kernel", 4.2),
                 ("c1", COPY, 4.3), ("kn", "aiter::add_rmsnorm_quant_kernel", 4.4),
                 ("rope", "rotary_embedding_kernel<bf16>", 5.0), ("kv", "store_kvcache<2048>", 4.4),
                 ("pa", "paged_attention_ll4mi_qkv_kernel", 24.0),
                 ("g", "gemm_a8w8_blockscale_kernel", 583.2)]
        rows = screen.load_rows(_table(
            [_row(rid, name, dur, device_seq_index=i) for i, (rid, name, dur) in enumerate(names)], 634.0))
        result = screen.screen(rows, catalog=QK_CATALOG)
        top = result["execution_list"][0]
        self.assertIn("kv", top["row_ids"])
        self.assertIn("rope", top["row_ids"])
        self.assertTrue(top["has_existing_kernel"])
        self.assertEqual(top["difficulty"], 1.0)
        self.assertEqual(top["matched_existing_kernels"][0]["name"],
                         "fused_qk_norm_rope_cache_quant_shuffle")
        for item in result["initial_topk"]:
            self.assertNotIn("pa", item["row_ids"])
            self.assertNotIn("g", item["row_ids"])

    def test_barrier_never_inside_a_chain(self):
        rows = screen.load_rows(_table([
            _row("n", "rmsnorm", 40, device_seq_index=0),
            _row("pa", "paged_attention_kernel", 40, device_seq_index=1),
            _row("q", "quant_cast", 40, device_seq_index=2),
            _row("r", "rotary_embedding_kernel", 40, device_seq_index=3),
            _row("pad", "copy_layout", 500, device_seq_index=4, timestamp=9000),
        ], 660))
        result = screen.screen(rows)
        for item in result["initial_topk"]:
            self.assertNotIn("pa", item["row_ids"])
        self.assertIn(["q", "r"], [item["row_ids"] for item in result["initial_topk"]])
        self.assertGreater(result["rejected"].get("barrier_without_existing_kernel", 0), 0)

    def test_existing_kernel_lowers_time_gate_to_1_percent(self):
        # SiLU*Mul + fp8 group quant: ~2% of the step, below 3% but above 1%.
        def rows():
            return screen.load_rows(_table([
                _row("act", "sgl_hip::activation::act_and_mul_kernel<silu>", 8.6, device_seq_index=0),
                _row("q", "aiter::dynamic_per_group_scaled_quant_kernel", 4.6, device_seq_index=1),
                _row("g", "gemm_a8w8_blockscale_kernel", 620.8, device_seq_index=2),
            ], 634.0))
        self.assertEqual(screen.screen(rows())["initial_topk"], [])
        catalog = {"kernels": [
            {"name": "is_activation_quantization_format", "op_tags": ["activation", "quant"],
             "dtype_tags": [], "sources": ["sglang"]},
            {"name": "act_mul_and_fp8_group_quant", "op_tags": ["activation", "quant"],
             "dtype_tags": ["fp8", "fp8_blockscale"], "sources": ["aiter"]},
        ]}
        result = screen.screen(rows(), catalog=catalog)
        chain = next(item for item in result["execution_list"] if item["row_ids"] == ["act", "q"])
        self.assertEqual(chain["gate_min"], screen.EXISTING_CHAIN_E2E_MIN)
        self.assertEqual([m["name"] for m in chain["matched_existing_kernels"]],
                         ["act_mul_and_fp8_group_quant"])

    def test_same_seam_in_prefill_and_decode_is_one_entry(self):
        def phase_rows(prefix, phase, extra):
            return [_row(prefix + rid, name, dur, phase=phase, device_seq_index=i, **extra)
                    for i, (rid, name, dur) in enumerate([
                        ("qn", "rmsnorm", 4.0), ("kn", "rmsnorm", 4.0),
                        ("r", "rotary_embedding_kernel", 5.0), ("g", "gemm_bf16", 500.0)])]
        table = {"tables": [
            {"phase": "prefill", "pattern_id": "p0", "pattern_layer_count": 40, "layer_total_us": 513.0,
             "rows": phase_rows("p", "prefill", {"step_input_tokens": 2048})},
            {"phase": "decode", "pattern_id": "p0", "pattern_layer_count": 40, "layer_total_us": 634.0,
             "rows": phase_rows("d", "decode", {"step_batch_size": 8})},
        ]}
        catalog = {"kernels": [{"name": "fused_qk_norm_rope_2way", "op_tags": ["norm", "rope"],
                                "dtype_tags": [], "sources": ["aiter"]}]}
        result = screen.screen(screen.load_rows(table), catalog=catalog,
                               workload={"isl": 1024, "osl": 1024, "conc": 64})
        self.assertEqual(len(result["execution_list"]), 1)
        entry = result["execution_list"][0]
        self.assertEqual(sorted(entry["phases"]), ["decode", "prefill"])
        self.assertEqual(len(entry["candidate_ids"]), 2)
        self.assertEqual({entry["row_ids"][0][0], entry["partner_row_ids"][0][0]}, {"p", "d"})

    def test_rope_is_elementwise_and_copies_do_not_set_difficulty(self):
        rows = screen.load_rows(_table([
            _row("c", COPY, 40, device_seq_index=0),
            _row("r", "rotary_embedding_kernel", 40, device_seq_index=1),
            _row("q", "quant_cast", 40, device_seq_index=2),
            _row("pad", "copy_layout", 500, device_seq_index=3, timestamp=9000),
        ], 620))
        result = screen.screen(rows)
        chain = next(item for item in result["initial_topk"] if item["row_ids"] == ["c", "r", "q"])
        self.assertEqual(chain["difficulty"], 1.3)
        self.assertEqual(chain["data_move_row_ids"], ["c"])
        self.assertEqual(chain["unknown_kernels"], [])

    def test_baseline_fused_kernel_with_follow_up_quant_is_flagged(self):
        rows = screen.load_rows(_table([
            _row("n", "aiter::add_rmsnorm_quant_kernel", 4.7, device_seq_index=0),
            _row("q", "aiter::dynamic_per_group_scaled_quant_kernel", 4.6, device_seq_index=1),
            _row("g", "gemm_a8w8_blockscale_kernel", 624.7, device_seq_index=2),
        ], 634.0))
        catalog = {"kernels": [{"name": "add_rmsnorm_quant",
                                "op_tags": ["add_residual", "norm", "quant"],
                                "dtype_tags": [], "sources": ["aiter"]}]}
        result = screen.screen(rows, catalog=catalog)
        fused = result["baseline_fused"]
        self.assertEqual([item["catalog_kernel"] for item in fused], ["add_rmsnorm_quant"])
        self.assertTrue(fused[0]["still_in_candidates"])

    def test_cli_writes_both_boards(self):
        import tempfile
        table = _table([
            _row("a", "quant_cast", 40, device_seq_index=0, timestamp=0),
            _row("b", "silu", 40, device_seq_index=1, timestamp=40),
            _row("pad", "copy_layout", 2000, device_seq_index=2, timestamp=5000),
        ], 2080)
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "table.json")
            with open(src, "w") as handle:
                json.dump(table, handle)
            out = os.path.join(tmp, "fusion")
            rc = screen.main(["--semantic-table", src, "--out-dir", out,
                              "--md", os.path.join(tmp, "03_FUSION_TOPK.md")])
            self.assertEqual(rc, 0)
            with open(os.path.join(out, "fusion_topk.json")) as handle:
                topk = json.load(handle)
            self.assertEqual(len(topk["execution_list"]), 1)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "03_FUSION_TOPK.md")))


if __name__ == "__main__":
    unittest.main()
