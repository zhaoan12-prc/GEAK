import os
import sys
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_two_trace_mapping as two_trace


def _row(row_id, pos, name, level="external_id", dims=None, op="aten::mm"):
    return {
        "row_id": row_id,
        "pos": pos,
        "raw_name": name,
        "parent_operator": {
            "canonical_op": op,
            "op_instance_id": "ext-%s" % pos,
            "mapping_level": level,
            "mapping_cardinality": "1:1",
            "device_launch_count": 1,
        },
        "shape": {
            "source": "kernel_exact" if dims else "unresolved",
            "input_dims": dims,
            "input_types": ["c10::BFloat16"] if dims else [],
        },
    }


def _table(rows, pattern="p0", phase="decode", layer=59):
    return {
        "pattern_id": pattern,
        "phase": phase,
        "representative_layer_id": layer,
        "rows": rows,
    }


class TwoTraceMappingTest(unittest.TestCase):
    def test_repeated_kernel_names_bind_by_ordered_position(self):
        """The k-th occurrence of a name must bind to the k-th, not the first.

        This is the case pure name matching cannot express, and the one that
        produced non_unique_native_kernel_name unavailable rows.
        """
        formal = _table([
            _row("f0", 0, "gemm", level="unresolved"),
            _row("f1", 1, "gemm", level="unresolved"),
            _row("f2", 2, "gemm", level="unresolved"),
        ])
        mapping = _table([
            _row("m0", 0, "gemm", dims=[[1, 1]], op="qkv_proj"),
            _row("m1", 1, "gemm", dims=[[2, 2]], op="o_proj"),
            _row("m2", 2, "gemm", dims=[[3, 3]], op="down_proj"),
        ])
        doc = two_trace.build({"tables": [formal]}, {"tables": [mapping]})
        entries = doc["entries"]
        self.assertEqual(entries["f0"]["op"]["canonical_op"], "qkv_proj")
        self.assertEqual(entries["f1"]["op"]["canonical_op"], "o_proj")
        self.assertEqual(entries["f2"]["op"]["canonical_op"], "down_proj")
        self.assertEqual(entries["f1"]["shape"]["input_dims"], [[2, 2]])
        for entry in entries.values():
            self.assertEqual(entry["match"]["position_delta"], 0)
            self.assertEqual(entry["match"]["confidence"], "high")

    def test_extra_mapping_row_shifts_alignment_without_crossing(self):
        """A graph-off-only kernel must shift later bindings, not mis-bind them."""
        formal = _table([
            _row("f0", 0, "norm", level="unresolved"),
            _row("f1", 1, "gemm", level="unresolved"),
        ])
        mapping = _table([
            _row("m0", 0, "norm", dims=[[9, 9]], op="input_layernorm"),
            _row("m1", 1, "extra_eager_only", dims=[[7, 7]], op="scratch"),
            _row("m2", 2, "gemm", dims=[[4, 4]], op="qkv_proj"),
        ])
        doc = two_trace.build({"tables": [formal]}, {"tables": [mapping]})
        entries = doc["entries"]
        self.assertEqual(entries["f0"]["op"]["canonical_op"], "input_layernorm")
        self.assertEqual(entries["f1"]["op"]["canonical_op"], "qkv_proj")
        self.assertEqual(entries["f1"]["match"]["position_delta"], 1)
        self.assertEqual(entries["f1"]["match"]["confidence"], "medium")

    def test_leading_mapping_kernels_do_not_mis_anchor_a_repeated_name(self):
        """The real 20260812v3 Qwen decode failure, reduced.

        The mapping layer instance starts with four kernels the formal window
        does not have, and the first of them repeats the name of formal row 0.
        Two maximum-length alignments exist -- bind row 0 to the leading MoE
        quant (offset 0, everything else offset 4) or to the attention input
        quant (offset 4 throughout) -- and a greedy backtrack takes the first,
        putting MoE shapes on the attention row.  Every pair must agree on one
        offset instead.
        """
        formal = _table([
            _row("f0", 0, "quant", level="unresolved"),
            _row("f1", 1, "gemm", level="unresolved"),
            _row("f2", 2, "copy", level="unresolved"),
        ])
        mapping = _table([
            _row("m0", 0, "quant", dims=[[4, 11, 128]], op="moe_quant"),
            _row("m1", 1, "moe_gemm", dims=[[1, 1]], op="moe_gemm"),
            _row("m2", 2, "memcpy", dims=[[1, 1]], op="memcpy"),
            _row("m3", 3, "allreduce", dims=[[1, 1]], op="all_reduce"),
            _row("m4", 4, "quant", dims=[[4, 4096]], op="attn_input_quant"),
            _row("m5", 5, "gemm", dims=[[4, 4096]], op="qkv_proj"),
            _row("m6", 6, "copy", dims=[[4, 4, 256]], op="split_qkv"),
        ])
        doc = two_trace.build({"tables": [formal]}, {"tables": [mapping]})
        entries = doc["entries"]
        self.assertEqual(
            entries["f0"]["op"]["canonical_op"], "attn_input_quant")
        self.assertEqual(entries["f0"]["shape"]["input_dims"], [[4, 4096]])
        self.assertEqual(entries["f1"]["op"]["canonical_op"], "qkv_proj")
        self.assertEqual(entries["f2"]["op"]["canonical_op"], "split_qkv")
        self.assertEqual(
            [entry["match"]["position_delta"] for entry in entries.values()],
            [4, 4, 4])

    def test_preferring_an_offset_never_costs_a_matched_row(self):
        """A tidier offset must never be bought with a shorter alignment.

        Only one alignment here is maximum length (three pairs) and its offsets
        are unavoidably mixed, so aiming at any offset must leave it alone.
        """
        formal = ["a", "b", "c"]
        mapping = ["a", "x", "b", "y", "y", "c"]
        plain = two_trace._lcs_pairs(formal, mapping)
        self.assertEqual(plain, [(0, 0), (1, 2), (2, 5)])
        for target in (-2, 0, 1, 3, 7):
            aimed = two_trace._lcs_pairs(formal, mapping, target)
            self.assertEqual(len(aimed), len(plain))
            self.assertEqual(aimed, plain)

    def test_alignment_length_is_invariant_under_any_preferred_offset(self):
        """Deferring a pair is gated on the DP table, so length cannot drop."""
        formal = ["q", "g", "q", "g", "n", "q"]
        mapping = ["q", "z", "q", "g", "q", "g", "n", "q", "q"]
        best = len(two_trace._lcs_pairs(formal, mapping))
        for target in range(-4, 6):
            aimed = two_trace._lcs_pairs(formal, mapping, target)
            self.assertEqual(len(aimed), best)
            self.assertEqual(aimed, sorted(aimed))
            self.assertEqual(
                len({i for i, _ in aimed}), len(aimed))
            self.assertEqual(
                len({j for _, j in aimed}), len(aimed))
            for (i0, j0), (i1, j1) in zip(aimed, aimed[1:]):
                self.assertLess(i0, i1)
                self.assertLess(j0, j1)

    def test_name_mismatch_is_never_bridged(self):
        """No fuzzy fallback: an unmatched row must stay unmatched."""
        formal = _table([_row("f0", 0, "totally_different", level="unresolved")])
        mapping = _table([_row("m0", 0, "gemm", dims=[[1, 1]])])
        doc = two_trace.build({"tables": [formal]}, {"tables": [mapping]})
        self.assertEqual(doc["entries"], {})
        self.assertEqual(doc["matched_row_count"], 0)

    def test_missing_mapping_table_is_reported_not_guessed(self):
        formal = _table([_row("f0", 0, "gemm", level="unresolved")])
        doc = two_trace.build({"tables": [formal]}, {"tables": []})
        self.assertEqual(doc["entries"], {})
        self.assertEqual(doc["tables"][0]["status"], "no_mapping_table")

    def test_unresolved_mapping_row_is_not_counted_as_recovery(self):
        formal = _table([_row("f0", 0, "gemm", level="unresolved")])
        mapping = _table([_row("m0", 0, "gemm", level="unresolved")])
        doc = two_trace.build({"tables": [formal]}, {"tables": [mapping]})
        self.assertEqual(doc["matched_row_count"], 1)
        self.assertEqual(doc["op_recovered_row_count"], 0)
        self.assertEqual(doc["shape_recovered_row_count"], 0)
        self.assertFalse(doc["entries"]["f0"]["op"]["recovered"])

    def test_different_representative_layer_lowers_confidence(self):
        formal = _table([_row("f0", 0, "gemm", level="unresolved")], layer=59)
        mapping = _table([_row("m0", 0, "gemm", dims=[[1, 1]])], layer=42)
        doc = two_trace.build({"tables": [formal]}, {"tables": [mapping]})
        match = doc["entries"]["f0"]["match"]
        self.assertEqual(
            match["representative_layer_match"],
            "different_layer_same_pattern")
        self.assertEqual(match["confidence"], "medium")

    def test_shape_scope_follows_the_mapping_rows_own_cardinality(self):
        # kernel_exact means one cpu_op launched one kernel, so the dims are
        # this kernel's. parent_context means a 1:N wrapper handed the same
        # dims to every kernel it launched.
        formal = _table([
            _row("f0", 0, "one_to_one", level="unresolved"),
            _row("f1", 1, "one_of_many", level="unresolved"),
        ])
        exact = _row("m0", 0, "one_to_one", dims=[[4, 16]])
        shared = _row("m1", 1, "one_of_many", dims=[[4, 16]])
        shared["shape"]["source"] = "parent_context"
        shared["parent_operator"]["mapping_cardinality"] = "1:N"
        shared["parent_operator"]["device_launch_count"] = 3
        doc = two_trace.build({"tables": [formal]},
                              {"tables": [_table([exact, shared])]})
        self.assertEqual(doc["entries"]["f0"]["shape"]["scope"], "kernel")
        self.assertEqual(doc["entries"]["f1"]["shape"]["scope"], "wrapper")
        # Both are still recovered -- wrapper dims are weaker, not useless.
        self.assertEqual(doc["shape_recovered_row_count"], 2)
        self.assertEqual(doc["kernel_scope_shape_row_count"], 1)
        self.assertEqual(doc["wrapper_scope_shape_row_count"], 1)

    def test_drop_tables_removes_only_the_skipped_windows_bindings(self):
        formal_doc = {"tables": [
            _table([_row("d0", 0, "gemm", level="unresolved")], phase="decode"),
            _table([_row("p0", 0, "gemm", level="unresolved")], phase="prefill"),
        ]}
        mapping_doc = {"tables": [
            _table([_row("m0", 0, "gemm", dims=[[1, 1]], op="decode_op")],
                   phase="decode"),
            _table([_row("m1", 0, "gemm", dims=[[2, 2]], op="prefill_op")],
                   phase="prefill"),
        ]}
        doc = two_trace.build(formal_doc, mapping_doc)
        self.assertEqual(doc["matched_row_count"], 2)
        two_trace.drop_tables(doc, ["p0|decode"])
        self.assertNotIn("d0", doc["entries"])
        self.assertIn("p0", doc["entries"])
        self.assertEqual(doc["dropped_entry_count"], 1)
        self.assertEqual(doc["matched_row_count"], 1)
        self.assertEqual(doc["shape_recovered_row_count"], 1)
        skipped = [item for item in doc["tables"]
                   if item["table"] == "p0|decode"]
        self.assertEqual(
            skipped[0]["status"], "skipped_not_workload_identical")
        self.assertEqual(skipped[0]["matched_rows"], 0)

    def test_drop_tables_with_nothing_to_drop_is_a_no_op(self):
        formal = _table([_row("f0", 0, "gemm", level="unresolved")])
        mapping = _table([_row("m0", 0, "gemm", dims=[[1, 1]])])
        doc = two_trace.build({"tables": [formal]}, {"tables": [mapping]})
        two_trace.drop_tables(doc, [])
        self.assertIn("f0", doc["entries"])
        self.assertNotIn("dropped_entry_count", doc)

    def test_tables_are_matched_per_pattern_and_phase(self):
        formal_doc = {"tables": [
            _table([_row("d0", 0, "gemm", level="unresolved")], phase="decode"),
            _table([_row("p0", 0, "gemm", level="unresolved")], phase="prefill"),
        ]}
        mapping_doc = {"tables": [
            _table([_row("m0", 0, "gemm", dims=[[1, 1]], op="decode_op")],
                   phase="decode"),
            _table([_row("m1", 0, "gemm", dims=[[2, 2]], op="prefill_op")],
                   phase="prefill"),
        ]}
        doc = two_trace.build(formal_doc, mapping_doc)
        self.assertEqual(
            doc["entries"]["d0"]["op"]["canonical_op"], "decode_op")
        self.assertEqual(
            doc["entries"]["p0"]["op"]["canonical_op"], "prefill_op")


if __name__ == "__main__":
    unittest.main()
