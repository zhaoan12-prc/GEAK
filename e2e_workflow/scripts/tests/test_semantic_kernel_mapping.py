import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_kernel_mapping as mapping


class SemanticKernelMappingTest(unittest.TestCase):
    def test_stage_precedence_keeps_norm_and_gemm_semantics(self):
        self.assertEqual(
            mapping._stage("add_rmsnorm_quant_kernel", "kernel"), "norm")
        self.assertEqual(
            mapping._stage("_gemm_a8w8_blockscale_kernel_cache_hint", "kernel"),
            "gemm")
        self.assertEqual(
            mapping._stage("_causal_conv1d_fwd_kernel", "kernel"), "linear_attn")
        self.assertEqual(
            mapping._stage(
                "fused_recurrent_gated_delta_rule_packed_decode_kernel", "kernel"),
            "linear_attn")
        self.assertEqual(mapping._stage("l2norm_fwd_kernel", "kernel"), "norm")

    def _patterns(self, root):
        path = os.path.join(root, "patterns.json")
        with open(path, "w") as fh:
            json.dump({
                "schema_version": 1,
                "patterns": [{
                    "pattern_id": "P_DENSE",
                    "pattern_display_name": "Dense",
                    "layer_ids": [0, 1],
                }],
                "coverage_check": {
                    "total_main_layers": 2, "covered": 2,
                    "mutually_exclusive": True, "full_coverage": True,
                },
                "quality": {"status": "pass"},
            }, fh)
        return path

    def _trace(self, root, annotated=True):
        events = []
        if annotated:
            events.extend([
                {"cat": "gpu_user_annotation",
                 "name": "execute_context_1(8)_generation_0(0)",
                 "ts": 0, "dur": 100},
                {"cat": "gpu_user_annotation",
                 "name": "execute_context_0(0)_generation_2(16)",
                 "ts": 200, "dur": 100},
            ])
        phases = [("prefill", 10 if annotated else 10, [10, 20]),
                  ("decode", 210 if annotated else 110, [4, 6])]
        ext = 0
        device = []
        for _, base, durations in phases:
            cursor = base
            for layer_id, duration in enumerate(durations):
                ext += 1
                events.append({
                    "cat": "cpu_op", "name": "model.layers.%d.mlp" % layer_id,
                    "ts": cursor - 1, "dur": duration + 2,
                    "args": {"External id": ext, "Input Dims": [[2, 4]],
                             "Input type": ["Half"]},
                })
                device.append({
                    "cat": "kernel", "name": "fused_mlp_kernel",
                    "ts": cursor, "dur": duration,
                    "args": {"External id": ext, "stream": 1},
                })
                cursor += duration + 2
        events.extend(device)
        path = os.path.join(root, "trace.json")
        with open(path, "w") as fh:
            json.dump({"traceEvents": events}, fh)
        return path

    def test_conservation_representative_and_exact_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = mapping.build(
                self._trace(tmp), self._patterns(tmp), os.path.join(tmp, "out"))
            self.assertEqual(result["status"], "pass")
            with open(result["quality_json"]) as fh:
                quality = json.load(fh)
            gate = quality["gates"]["analysis_window_conservation"]
            self.assertEqual(gate["input_event_count"], 4)
            self.assertEqual(gate["assigned_event_count"], 4)
            self.assertEqual(gate["status"], "pass")
            integrity = quality["gates"]["representative_layer_integrity"]
            self.assertEqual(integrity["status"], "pass")
            self.assertEqual(integrity["table_count"], 2)
            self.assertTrue(all(
                item["interval_complete"] and item["duration_matches"]
                for item in integrity["tables"]))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            self.assertIn(audit["representatives"]["P_DENSE"]["layer_id"], [0, 1])
            with open(result["semantic_table_json"]) as fh:
                tables = json.load(fh)["tables"]
            self.assertEqual({table["phase"] for table in tables},
                             {"prefill", "decode"})
            self.assertTrue(all(
                table["pattern_layer_ids"] == [0, 1]
                and table["pattern_layer_count"] == 2
                and table["representative_layer_id"]
                in table["pattern_layer_ids"]
                for table in tables))
            self.assertTrue(all(row["shape"]["source"] == "kernel_exact"
                                for table in tables for row in table["rows"]))
            self.assertTrue(all(
                "batch_size" in table["selected_bucket"]
                for table in tables))

    def test_representative_integrity_rejects_a_truncated_table(self):
        rows = [
            {"row_id": "event-1", "device_seq_index": 1, "duration_us": 1.0},
            {"row_id": "event-2", "device_seq_index": 2, "duration_us": 2.0},
        ]
        representatives = {"P": {"selected_instances": {
            "decode": {
                "first_device_seq_index": 1,
                "last_device_seq_index": 2,
            }}}}
        tables = [{
            "phase": "decode", "pattern_id": "P",
            "representative_layer_id": 0, "event_count": 1,
            "layer_total_us": 1.0,
            "rows": [dict(rows[0], pos=0)],
        }]
        gate = mapping._representative_integrity(
            rows, tables, representatives)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["tables"][0]["dropped_row_ids"], ["event-2"])

    def test_non_dominant_metadata_prefix_is_demoted_losslessly(self):
        rows = []
        sequence = 0
        for layer_id, stages in (
                (0, ["elementwise", "norm", "gemm"]),
                (1, ["norm", "gemm"]),
                (2, ["norm", "gemm"])):
            for index, stage in enumerate(stages):
                rows.append({
                    "row_id": "event-%d" % sequence,
                    "device_seq_index": sequence,
                    "phase": "prefill",
                    "step_id": "step-1",
                    "assignment": "layer_body",
                    "layer_id": layer_id,
                    "layer_instance_id": "instance-%d" % layer_id,
                    "pattern_id": "P_DENSE",
                    "stage": stage,
                    "layer_evidence": "module_span_sequence_medoid",
                    "layer_region": "layer_body",
                    "boundary_role": (
                        "body_start_kernel" if index == 0 else None),
                })
                sequence += 1
        diagnostics = [{
            "step_id": "step-1",
            "mapped_event_count": len(rows),
            "layer_boundaries": [
                {"layer_id": 0, "body_start_event": "event-0"},
                {"layer_id": 1, "body_start_event": "event-3"},
                {"layer_id": 2, "body_start_event": "event-5"},
            ],
        }]
        demotions = mapping._demote_non_dominant_prefixes(
            rows, diagnostics)
        self.assertEqual(len(demotions), 1)
        self.assertEqual(rows[0]["assignment"], "transition_global")
        self.assertEqual(
            rows[0]["layer_evidence"],
            "pattern_variant_prefix_demoted")
        self.assertEqual(rows[1]["boundary_role"], "body_start_kernel")
        self.assertEqual(
            diagnostics[0]["layer_boundaries"][0]["body_start_event"],
            "event-1")
        self.assertEqual(diagnostics[0]["mapped_event_count"], 6)

    def test_shared_external_id_is_parent_context_not_kernel_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = self._patterns(tmp)
            events = [
                {"cat": "gpu_user_annotation",
                 "name": "step[EXTEND bs=1 toks=8]", "ts": 0, "dur": 100},
                {"cat": "cpu_op", "name": "aiter::wrapper",
                 "ts": 5, "dur": 10,
                 "args": {"External id": 7, "Input Dims": [[8, 4]],
                          "Input type": ["BFloat16"],
                          "Module Hierarchy": "model.layers.0.mlp"}},
                {"cat": "kernel", "name": "child_kernel_a", "ts": 20,
                 "dur": 1, "args": {"External id": 7}},
                {"cat": "kernel", "name": "child_kernel_b", "ts": 22,
                 "dur": 1, "args": {"External id": 7}},
            ]
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["semantic_table_json"]) as fh:
                rows = json.load(fh)["tables"][0]["rows"]
            self.assertEqual(
                [row["shape"]["source"] for row in rows],
                ["parent_context", "parent_context"])
            self.assertTrue(all(
                row["parent_operator"]["mapping_cardinality"] == "1:N"
                for row in rows))
            with open(result["shape_capture_plan_json"]) as fh:
                plan = json.load(fh)
            self.assertEqual(plan["target_count"], 2)

    def test_missing_annotations_degrades_phase_without_losing_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = mapping.build(
                self._trace(tmp, annotated=False), self._patterns(tmp),
                os.path.join(tmp, "out"))
            self.assertEqual(result["status"], "partial")
            with open(result["quality_json"]) as fh:
                quality = json.load(fh)
            self.assertEqual(quality["gates"]["phase"]["status"], "partial")
            gate = quality["gates"]["analysis_window_conservation"]
            self.assertEqual(gate["input_event_count"], gate["assigned_event_count"])

    def test_sglang_module_spans_are_global_ordered_and_stream_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1,
                    "num_hidden_layers_main": 2,
                    "patterns": [
                        {"pattern_id": "P_LINEAR_ATTENTION",
                         "pattern_display_name": "Linear",
                         "attention_type": "linear_attention", "layer_ids": [0]},
                        {"pattern_id": "P_FULL_ATTENTION",
                         "pattern_display_name": "Full",
                         "attention_type": "full_attention", "layer_ids": [1]},
                    ],
                    "coverage_check": {"total_main_layers": 2, "covered": 2,
                                       "mutually_exclusive": True, "full_coverage": True},
                    "quality": {"status": "pass"},
                }, fh)
            events = [
                {"cat": "gpu_user_annotation", "name": "step[EXTEND bs=1 toks=8]",
                 "ts": 0, "dur": 100},
                {"cat": "python_function",
                 "name": "nn.Module: Qwen3_5LinearDecoderLayer_0",
                 "ts": 10, "dur": 20},
                {"cat": "python_function",
                 "name": "nn.Module: Qwen3_5AttentionDecoderLayer_0",
                 "ts": 40, "dur": 20},
                {"cat": "cpu_op", "name": "aten::mm", "ts": 12, "dur": 2,
                 "args": {"External id": 1, "Input Dims": [[2, 2]]}},
                {"cat": "cpu_op", "name": "aten::mm", "ts": 42, "dur": 2,
                 "args": {"External id": 2, "Input Dims": [[2, 2]]}},
                {"cat": "kernel", "name": "linear_attention_kernel", "ts": 70, "dur": 2,
                 "args": {"External id": 1, "stream": 7}},
                {"cat": "kernel", "name": "comm_interleaved", "ts": 72, "dur": 1,
                 "args": {"stream": 8}},
                {"cat": "kernel", "name": "paged_attention_kernel", "ts": 74, "dur": 2,
                 "args": {"External id": 2, "stream": 7}},
            ]
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            self.assertEqual(audit["module_scope_count"], 2)
            self.assertEqual(len(audit["instances"]), 2)
            self.assertTrue(all(item["boundary_complete"] for item in audit["instances"]))
            self.assertEqual({item["layer_id"] for item in audit["instances"]}, {0, 1})

    def test_cpu_module_spans_use_the_cpu_step_window_not_the_gpu_one(self):
        """Eager decode: host wall-clock >> device time, so the GPU-side
        `step[...]` window covers only a fraction of the CPU-side
        `nn.Module: DecoderLayer_N` spans.  Locating CPU spans against the GPU
        window dropped most layers and silently degraded decode boundaries to
        anchor segmentation; they must be located against the CPU window."""
        events = [
            # CPU-side step annotation spans the whole host-side step.
            {"ph": "X", "cat": "user_annotation", "name": "step[DECODE bs=4]",
             "ts": 1000, "dur": 10000},
            # GPU-side annotation covers only the tail: the device work.
            {"ph": "X", "cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
             "ts": 10500, "dur": 400},
        ]
        # Three DecoderLayer spans, all inside the CPU window, only the last
        # inside the GPU window.
        for layer_id, ts in enumerate((1500, 5000, 10600)):
            events.append({
                "ph": "X", "cat": "python_function", "ts": ts, "dur": 100,
                "name": "nn.Module: DeepseekV2AttentionDecoderLayer_%d" % layer_id,
            })
        spans = mapping._collect_step_spans(events)
        self.assertTrue(spans, "step span not recognised")
        self.assertEqual(len(spans[0]), 9, "CPU window not carried on the span")
        self.assertEqual((spans[0][7], spans[0][8]), (1000, 11000))
        pattern_doc = {
            "num_hidden_layers_main": 3,
            "patterns": [{"pattern_id": "P_FULL", "attention_type": "full",
                          "layer_ids": [0, 1, 2]}],
        }
        scopes, diagnostics = mapping._module_layer_scopes(
            events, spans, pattern_doc)
        self.assertEqual(len(scopes), 3)
        self.assertEqual(diagnostics[0]["full_passes"], 1)
        self.assertEqual([s["layer_id"] for s in scopes], [0, 1, 2])

    def test_module_medoid_partitions_moduleless_decode_without_named_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1,
                    "num_hidden_layers_main": 2,
                    "patterns": [
                        {"pattern_id": "P_LINEAR_ATTENTION",
                         "pattern_display_name": "Linear",
                         "attention_type": "linear_attention", "layer_ids": [0]},
                        {"pattern_id": "P_FULL_ATTENTION",
                         "pattern_display_name": "Full",
                         "attention_type": "full_attention", "layer_ids": [1]},
                    ],
                    "coverage_check": {"total_main_layers": 2, "covered": 2,
                                       "mutually_exclusive": True, "full_coverage": True},
                    "quality": {"status": "pass"},
                }, fh)
            events = [
                {"cat": "gpu_user_annotation", "name": "step[EXTEND bs=1 toks=8]",
                 "ts": 0, "dur": 100},
                {"cat": "python_function",
                 "name": "nn.Module: HybridLinearDecoderLayer_0",
                 "ts": 10, "dur": 20},
                {"cat": "python_function",
                 "name": "nn.Module: HybridAttentionDecoderLayer_0",
                 "ts": 40, "dur": 20},
                {"cat": "cpu_op", "name": "linear_layer", "ts": 12, "dur": 2,
                 "args": {"External id": 1}},
                {"cat": "cpu_op", "name": "full_layer", "ts": 42, "dur": 2,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "rmsnorm_kernel", "ts": 70, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "gated_delta_kernel", "ts": 72, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "quant_kernel", "ts": 74, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "rmsnorm_kernel", "ts": 76, "dur": 1,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "paged_attention_kernel", "ts": 78, "dur": 1,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "quant_kernel", "ts": 80, "dur": 1,
                 "args": {"External id": 2}},
                {"cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
                 "ts": 200, "dur": 100},
                {"cat": "kernel", "name": "quant_kernel", "ts": 210, "dur": 1, "args": {}},
                {"cat": "kernel", "name": "gated_delta_kernel", "ts": 212, "dur": 1,
                 "args": {}},
                {"cat": "kernel", "name": "quant_kernel", "ts": 214, "dur": 1, "args": {}},
                {"cat": "kernel", "name": "paged_attention_kernel", "ts": 216, "dur": 1,
                 "args": {}},
                {"cat": "kernel", "name": "quant_kernel", "ts": 218, "dur": 1, "args": {}},
            ]
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            diag = next(
                item for item in audit["boundary_partition_diagnostics"]
                if item["partition_method"] != "module_span_sequence_medoid")
            self.assertEqual(diag["status"], "mapped")
            self.assertIn(
                diag["partition_method"],
                {"repeated_sequence_medoid", "forced_best_alignment"})
            self.assertEqual({item["layer_id"] for item in audit["instances"]}, {0, 1})

    def test_module_span_is_not_overridden_by_sequence_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1,
                    "num_hidden_layers_main": 2,
                    "patterns": [
                        {"pattern_id": "P_LINEAR_ATTENTION",
                         "pattern_display_name": "Linear",
                         "attention_type": "linear_attention", "ffn_type": "moe",
                         "structural_signature": {"is_moe": True},
                         "layer_ids": [0]},
                        {"pattern_id": "P_FULL_ATTENTION",
                         "pattern_display_name": "Full",
                         "attention_type": "full_attention", "ffn_type": "moe",
                         "structural_signature": {"is_moe": True},
                         "layer_ids": [1]},
                    ],
                    "coverage_check": {"total_main_layers": 2, "covered": 2,
                                       "mutually_exclusive": True, "full_coverage": True},
                    "quality": {"status": "pass"},
                }, fh)
            events = [
                {"cat": "gpu_user_annotation", "name": "step[EXTEND bs=1 toks=8]",
                 "ts": 0, "dur": 100},
                {"cat": "python_function",
                 "name": "nn.Module: HybridLinearDecoderLayer_0",
                 "ts": 10, "dur": 20},
                {"cat": "python_function",
                 "name": "nn.Module: HybridAttentionDecoderLayer_0",
                 "ts": 40, "dur": 20},
                {"cat": "cpu_op", "name": "layer_zero", "ts": 12, "dur": 2,
                 "args": {"External id": 1}},
                {"cat": "cpu_op", "name": "layer_one", "ts": 42, "dur": 2,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "topk_kernel", "ts": 70, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "kernel_moe_gemm", "ts": 72, "dur": 1,
                 "args": {"External id": 1}},
                {"cat": "kernel", "name": "topk_kernel", "ts": 74, "dur": 1,
                 "args": {"External id": 2}},
                {"cat": "kernel", "name": "kernel_moe_gemm", "ts": 76, "dur": 1,
                 "args": {"External id": 2}},
            ]
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            self.assertEqual(len(audit["instances"]), 2)
            self.assertTrue(all(item["boundary_complete"]
                                for item in audit["instances"]))
            self.assertTrue(all(
                item["boundary_evidence"]["end_anchor_valid"]
                for item in audit["instances"]))
            self.assertEqual(
                audit["boundary_partition_diagnostics"][0]["partition_method"],
                "module_span_sequence_medoid")
            self.assertTrue(all(
                any(source == "module_span_sequence_medoid"
                    for source in item["boundary_evidence"]["sources"])
                for item in audit["instances"]))

    def test_sequence_partition_counts_fused_boundary_kernel_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            patterns = os.path.join(tmp, "patterns.json")
            with open(patterns, "w") as fh:
                json.dump({
                    "schema_version": 1,
                    "model_type": "unregistered_model",
                    "num_hidden_layers_main": 2,
                    "patterns": [
                        {
                            "pattern_id": "P_ATTN_DENSE",
                            "pattern_display_name": "Dense",
                            "attention_type": "attention",
                            "ffn_type": "dense",
                            "layer_ids": [0],
                        },
                        {
                            "pattern_id": "P_ATTN_MOE",
                            "pattern_display_name": "MoE",
                            "attention_type": "attention",
                            "ffn_type": "moe",
                            "layer_ids": [1],
                        },
                    ],
                    "coverage_check": {"total_main_layers": 2, "covered": 2,
                                       "mutually_exclusive": True, "full_coverage": True},
                    "quality": {"status": "pass"},
                }, fh)
            fusion = "opaque_fused_boundary_kernel"
            names = [
                "quant_kernel", "gemm_kernel", fusion,
                "quant_kernel", "gemm_kernel", fusion,
            ]
            events = [{"cat": "gpu_user_annotation", "name": "step[DECODE bs=4]",
                       "ts": 0, "dur": 100}]
            events.extend(
                {"cat": "kernel", "name": name, "ts": 10 + index * 5,
                 "dur": 1, "args": {}}
                for index, name in enumerate(names))
            trace = os.path.join(tmp, "trace.json")
            with open(trace, "w") as fh:
                json.dump({"traceEvents": events}, fh)
            result = mapping.build(trace, patterns, os.path.join(tmp, "out"))
            with open(result["layer_instance_audit_json"]) as fh:
                audit = json.load(fh)
            self.assertEqual(len(audit["instances"]), 2)
            self.assertTrue(all(
                item["boundary_complete"] for item in audit["instances"]))
            with open(result["semantic_event_audit_jsonl"]) as fh:
                rows = [json.loads(line) for line in fh]
            fused_rows = [row for row in rows if row["raw_name"] == fusion]
            self.assertEqual(len(fused_rows), 2)
            self.assertEqual(
                len({row["layer_instance_id"] for row in fused_rows}), 2)
            self.assertEqual(
                sum(item["event_count"] for item in audit["instances"]), 6)


class PhaseCoverageTest(unittest.TestCase):
    """Regression tests for the decode-coverage defects (docs/decode-coverage-bugs.md)."""

    def test_phase_tag_and_sibling_discovery(self):
        self.assertEqual(
            mapping._phase_tag("x/1787.0-TP-0-EXTEND.trace.json.gz"), "EXTEND")
        self.assertEqual(
            mapping._phase_tag("x/1787.0-TP-3-DECODE.trace.json.gz"), "DECODE")
        self.assertIsNone(mapping._phase_tag("x/plain.trace.json.gz"))

    def test_sibling_discovery_finds_the_other_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = ["1787.0-TP-0-EXTEND.trace.json.gz",
                     "1787.0-TP-0-DECODE.trace.json.gz",
                     "1787.0-TP-1-DECODE.trace.json.gz"]
            for name in names:
                open(os.path.join(tmp, name), "w").close()
            found = mapping._sibling_phase_traces(
                os.path.join(tmp, names[0]))
            self.assertEqual(sorted(found), ["DECODE", "EXTEND"])
            # rank 1 must not be adopted into a rank 0 analysis
            self.assertNotIn("TP-1", found["DECODE"])

    def test_b1_table_phases_never_reports_all(self):
        """`table_phases` must name observed phases, not the word 'all'."""
        coverage = mapping._phase_coverage(
            instances=[{"phase": "extend"}],
            tables=[{"phase": "prefill", "rows": [
                {"shape": {"source": "kernel_exact"}}]}],
            trace_paths=["a-TP-0-EXTEND.trace.json.gz"],
            adopted_siblings=[], table_phases=None, require_phases=None)
        self.assertEqual(coverage["phases_in_tables"], ["prefill"])
        self.assertNotIn("all", coverage["phases_in_tables"])
        self.assertTrue(coverage["single_phase"])
        self.assertFalse(coverage["decode_sequence_covered"])
        self.assertEqual(coverage["decode_evidence"], "no_decode_trace_analysed")

    # ------------------------------------------------------------------ #
    # declared dispatch-op layer anchors (the torch.compile / hybrid-model path)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hybrid_pattern_doc(n=8, linear_branch="vllm::qwen_gdn_attention_core",
                            full_branch="vllm::unified_attention_with_output"):
        """n layers, every 4th one full-attention -- the Qwen3.5 hybrid shape."""
        full = [i for i in range(n) if i % 4 == 3]
        lin = [i for i in range(n) if i % 4 != 3]
        def pat(pid, attn, layers, branch):
            sig = {"attention_type": attn, "runtime_dispatch_branch": branch}
            return {"pattern_id": pid, "attention_type": attn,
                    "layer_ids": layers, "structural_signature": sig}
        return {"num_hidden_layers_main": n,
                "patterns": [pat("P_lin", "linear", lin, linear_branch),
                             pat("P_full", "full", full, full_branch)]}

    @staticmethod
    def _step_and_anchors(doc, n=8, base=1000, step=10, swap=None, drop=0):
        """One annotated step plus one cpu_op anchor per layer, in layer order."""
        events = [{"cat": "gpu_user_annotation", "name": "step[EXTEND bs=1 toks=64]",
                   "ts": base, "dur": step * (n + 2)},
                  {"cat": "user_annotation", "name": "step[EXTEND bs=1 toks=64]",
                   "ts": base, "dur": step * (n + 2)}]
        by_layer = {}
        for pattern in doc["patterns"]:
            for layer in pattern["layer_ids"]:
                by_layer[layer] = pattern["structural_signature"][
                    "runtime_dispatch_branch"]
        for layer in range(n - drop):
            name = by_layer[layer]
            if swap and layer in swap:
                name = swap[layer]
            events.append({"cat": "cpu_op", "name": name,
                           "ts": base + step * (layer + 1), "dur": 1})
        return events

    def test_dispatch_anchors_resolve_every_layer_when_modules_are_compiled_away(self):
        """The vllm case: no nn.Module frames, one splitting op per layer."""
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(doc)
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._dispatch_anchor_scopes(events, spans, doc)

        self.assertEqual(diag["status"], "mapped")
        self.assertEqual(len(scopes), 8)
        self.assertEqual([s["layer_id"] for s in scopes], list(range(8)))
        # Pattern assignment must follow the declared layout, not the anchor order alone.
        self.assertEqual([s["pattern_id"] for s in scopes],
                         ["P_lin", "P_lin", "P_lin", "P_full"] * 2)
        # Contiguous, non-overlapping partition.
        for earlier, later in zip(scopes, scopes[1:]):
            self.assertEqual(earlier["end"], later["ts"])
        self.assertTrue(all(s["scope_source"] == "declared_dispatch_op" for s in scopes))

    def test_dispatch_scoped_rows_are_not_re_cut_by_the_medoid(self):
        """An exact op boundary must survive the refinement meant for module spans.

        `_module_guided_segments` re-cuts a step against a stage-sequence medoid because an
        nn.Module span opens at the module's python entry, ahead of the layer's first
        kernel. A dispatch op has no such slack. Refining it anyway CORRUPTS the result:
        the medoid prefers prefill sequences, and on Qwen3.5-2B applying a prefill template
        to a decode step pushed the attention run into the previous segment -- the decode
        P_full_attn representative came back with no `attn` kernel at all.
        """
        rows = []
        for layer in range(2):
            for pos in range(3):
                rows.append({
                    "row_id": "r%d-%d" % (layer, pos),
                    "device_seq_index": layer * 3 + pos,
                    "layer_id": layer,
                    "layer_instance_id": "step-0:pass-0:layer-%d" % layer,
                    "layer_evidence": "declared_dispatch_op_span",
                    "stage": ("attn", "gemm", "norm")[pos],
                    "assignment": None, "phase": "decode",
                })
        patterns = {0: {"pattern_id": "P0"}, 1: {"pattern_id": "P1"}}
        result = mapping._direct_scope_segments(rows, patterns)

        self.assertEqual(result["mapped"], 6)
        self.assertEqual([c["layer_id"] for c in result["cuts"]], [0, 1])
        # The first row of each layer keeps its attention kernel -- the exact defect.
        self.assertEqual(rows[0]["stage"], "attn")
        self.assertEqual(rows[0]["boundary_role"], "body_start_kernel")
        self.assertEqual(rows[2]["boundary_role"], "end_kernel")
        self.assertTrue(all(r["assignment"] == "layer_body" for r in rows))
        self.assertEqual([r["pattern_id"] for r in rows], ["P0"] * 3 + ["P1"] * 3)

    def test_direct_segments_leave_unclaimed_rows_outside_any_layer(self):
        """An event no scope claimed is evidence, not something to fold into a neighbour.

        Folding it in would inflate that layer's measured cost, which is the number the
        fusion ranking is built on.
        """
        rows = [
            {"row_id": "pre", "device_seq_index": 0, "layer_id": None,
             "layer_instance_id": None, "layer_evidence": "unresolved",
             "stage": "memory", "assignment": None},
            {"row_id": "in", "device_seq_index": 1, "layer_id": 0,
             "layer_instance_id": "step-0:pass-0:layer-0",
             "layer_evidence": "declared_dispatch_op_span", "stage": "attn",
             "assignment": None},
        ]
        result = mapping._direct_scope_segments(rows, {0: {"pattern_id": "P0"}})
        self.assertEqual(result["mapped"], 1)
        self.assertEqual(rows[0]["assignment"], "transition_global")
        self.assertEqual(rows[0]["layer_evidence"], "sequence_outside_layer")
        self.assertIsNone(rows[0]["layer_id"])
        self.assertEqual(rows[1]["assignment"], "layer_body")

    def test_direct_segments_decline_when_no_dispatch_scope_is_present(self):
        rows = [{"row_id": "a", "device_seq_index": 0, "layer_id": 0,
                 "layer_instance_id": "s:pass-0:layer-0",
                 "layer_evidence": "python_module_span_external_id",
                 "stage": "attn", "assignment": None}]
        self.assertIsNone(mapping._direct_scope_segments(rows, {}))

    def test_dispatch_anchors_decline_when_a_pattern_declares_no_branch(self):
        """A layer kind with no anchor is the 6-of-24 defect; refuse, do not part-map.

        Half-anchoring is worse than not anchoring: it segments the layers it can see
        and silently swallows the rest into whatever segment happens to be open.
        """
        doc = self._hybrid_pattern_doc()
        doc["patterns"][1]["structural_signature"]["runtime_dispatch_branch"] = ""
        events = self._step_and_anchors(doc)
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._dispatch_anchor_scopes(events, spans, doc)
        self.assertEqual(scopes, [])
        self.assertEqual(diag["status"], "patterns_without_dispatch_branch")
        self.assertEqual(diag["patterns_missing_branch"], ["P_full"])

    def test_dispatch_anchors_decline_a_step_that_is_missing_anchors(self):
        """CUDA-graph decode emits no per-layer cpu_op; that step must not be mapped."""
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(doc, drop=3)
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._dispatch_anchor_scopes(events, spans, doc)
        self.assertEqual(scopes, [])
        self.assertEqual(diag["steps"][0]["status"], "anchor_count_mismatch")
        self.assertEqual(diag["steps"][0]["anchor_count"], 5)

    def test_dispatch_anchors_decline_when_order_disagrees_with_the_patterns(self):
        """The check that makes this evidence: anchor kind must match the layer's Pattern."""
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(
            doc, swap={0: "vllm::unified_attention_with_output"})
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._dispatch_anchor_scopes(events, spans, doc)
        self.assertEqual(scopes, [])
        self.assertEqual(diag["steps"][0]["status"],
                         "anchor_order_disagrees_with_patterns")
        self.assertEqual(diag["steps"][0]["first_mismatch_layer"], 0)

    def test_module_spans_still_win_over_dispatch_anchors(self):
        """The fallback must not displace real module evidence when it exists."""
        doc = self._hybrid_pattern_doc()
        events = self._step_and_anchors(doc)
        for layer in range(8):
            events.append({"cat": "python_function",
                           "name": "nn.Module: SomeDecoderLayer_%d" % layer,
                           "ts": 1000 + 10 * (layer + 1) - 2, "dur": 8})
        spans = mapping._collect_step_spans(events)
        scopes, diag = mapping._module_layer_scopes(events, spans, doc)
        self.assertEqual(len(scopes), 8)
        self.assertNotIn("scope_source", scopes[0])
        self.assertFalse(any(d.get("source", "").startswith("declared_dispatch")
                             for d in diag))

    # ------------------------------------------------------------------ #
    # required stages per declared attention type
    # ------------------------------------------------------------------ #
    def test_generically_named_attention_kernel_resolves_from_its_parent_op(self):
        """vLLM's TRITON_ATTN launches `_fwd_kernel`, which matches no name rule.

        It landed as `unknown`, so the LARGEST prefill row on Qwen3.5-2B (298.5 us, the
        attention operator itself) was not a donor -- and Phase 2.1's escalation gate then
        demanded a fusion candidate for the model's own attention. The parent op is
        authoritative: it is the registered op the kernel ran under.
        """
        stage, rule, source = mapping._stage_detail(
            "_fwd_kernel", "kernel", "vllm::unified_attention_with_output")
        self.assertEqual(stage, "attn")
        self.assertEqual(source, "parent_operator")
        self.assertEqual(rule, "attention.full.parent")

    def test_parent_op_fallback_still_prefers_linear_attention(self):
        stage, _rule, _src = mapping._stage_detail(
            "chunk_fwd_kernel", "kernel", "ChunkGatedDeltaRuleFunction")
        self.assertEqual(stage, "linear_attn")

    def test_kernel_name_rules_still_win_over_the_parent(self):
        # A kernel whose own name is conclusive must not be re-decided by its parent.
        stage, _rule, source = mapping._stage_detail(
            "kernel_paged_attention_2d", "kernel", "ChunkGatedDeltaRuleFunction")
        self.assertEqual(stage, "attn")
        self.assertEqual(source, "kernel_name")

    def test_linear_attention_pattern_does_not_require_an_attn_kernel(self):
        """A gated-delta layer emits linear_attn, never attn.

        Requiring `attn` of it made a CORRECTLY segmented hybrid layer fail the
        plausibility gate -- and, before boundaries were accurate, made an INCORRECTLY
        segmented one pass, because a 4-layer blob always contained a real attn kernel.
        """
        linear = {"pattern_id": "P_lin", "attention_type": "linear"}
        self.assertEqual(mapping._required_stages(linear), {"linear_attn", "gemm"})

    def test_full_attention_and_undeclared_patterns_keep_the_attn_requirement(self):
        self.assertEqual(mapping._required_stages({"pattern_id": "P_full",
                                                   "attention_type": "full"}),
                         {"attn", "gemm"})
        # Undeclared: attention_type is a REQUIRED signature field, so its absence is a
        # malformed doc. Keep the historical requirement rather than weakening the gate.
        self.assertEqual(mapping._required_stages({"pattern_id": "P_x"}),
                         {"attn", "gemm"})

    def test_moe_stages_are_added_on_top_of_the_attention_requirement(self):
        self.assertEqual(
            mapping._required_stages({"pattern_id": "P_lin_moe",
                                      "attention_type": "linear",
                                      "ffn_type": "moe"}),
            {"linear_attn", "gemm", "moe", "topk"})

    def test_unannotated_single_file_trace_names_the_missing_annotation(self):
        """vllm without the phase-annotation hook: NOTHING is phase-tagged.

        Pinned because the old wording for this case was `no_decode_trace_analysed`, which
        reads as "we did not look at a decode trace" and sends you to re-capture -- when the
        actual fault is that the trace carries no step spans at all, so prefill is equally
        untagged. The remedy is arming the hook, not another capture.
        """
        coverage = mapping._phase_coverage(
            instances=[{"phase": None}],
            tables=[{"phase": None, "rows": [{"shape": {"source": "unresolved"}}]}],
            trace_paths=["vllm-instance-rank-0.1234.pt.trace.json.gz"],
            adopted_siblings=[], table_phases=None, require_phases=None)
        self.assertEqual(coverage["trace_phase_tags"], [])
        self.assertFalse(coverage["phase_annotation_present"])
        self.assertEqual(coverage["decode_evidence"], "no_phase_annotation_in_trace")

    def test_annotated_single_file_trace_blames_the_window_not_the_capture(self):
        """Annotation worked, but this window held only prefill steps."""
        coverage = mapping._phase_coverage(
            instances=[{"phase": "extend"}],
            tables=[{"phase": "prefill", "rows": [
                {"shape": {"source": "kernel_exact"}}]}],
            trace_paths=["vllm-instance-rank-0.1234.pt.trace.json.gz"],
            adopted_siblings=[], table_phases=None, require_phases=None)
        self.assertTrue(coverage["phase_annotation_present"])
        self.assertEqual(coverage["decode_evidence"],
                         "mixed_trace_no_decode_steps_in_window")

    def test_split_file_capture_keeps_its_original_verdict(self):
        """sglang's per-phase filenames are conclusive; that arm must not shift."""
        coverage = mapping._phase_coverage(
            instances=[{"phase": "extend"}],
            tables=[{"phase": "prefill", "rows": [
                {"shape": {"source": "kernel_exact"}}]}],
            trace_paths=["a-TP-0-EXTEND.trace.json.gz"],
            adopted_siblings=[], table_phases=None, require_phases=None)
        self.assertEqual(coverage["decode_evidence"], "no_decode_trace_analysed")

    def test_sequence_and_shape_coverage_fail_independently(self):
        """A replay DECODE trace gives the sequence but no shapes."""
        coverage = mapping._phase_coverage(
            instances=[{"phase": "extend"}, {"phase": "decode"}],
            tables=[
                {"phase": "prefill", "rows": [
                    {"shape": {"source": "kernel_exact"}}]},
                {"phase": "decode", "rows": [
                    {"shape": {"source": "unresolved"}},
                    {"shape": {"source": "unresolved"}}]}],
            trace_paths=["a-TP-0-EXTEND.trace.json.gz",
                         "a-TP-0-DECODE.trace.json.gz"],
            adopted_siblings=[{"phase": "DECODE", "path": "a-TP-0-DECODE.trace.json.gz"}],
            table_phases=None, require_phases=None)
        self.assertTrue(coverage["decode_sequence_covered"])
        self.assertFalse(coverage["decode_shapes_covered"])
        self.assertFalse(coverage["decode_covered"])
        self.assertTrue(coverage["decode_requires_eager_probe"])
        self.assertEqual(coverage["decode_evidence"],
                         "sequence_only_shapes_unresolved")
        self.assertEqual(
            coverage["shape_resolution_by_phase"]["decode"]["resolved_fraction"],
            0.0)

    def test_require_phases_reports_the_missing_one(self):
        coverage = mapping._phase_coverage(
            instances=[], tables=[{"phase": "prefill", "rows": []}],
            trace_paths=["a-TP-0-EXTEND.trace.json.gz"],
            adopted_siblings=[], table_phases=None,
            require_phases=["prefill", "decode"])
        self.assertEqual(coverage["missing_required_phases"], ["decode"])

    def test_multi_trace_load_orders_by_first_timestamp(self):
        """DECODE follows EXTEND in wall clock; concatenation must preserve it."""
        with tempfile.TemporaryDirectory() as tmp:
            early = os.path.join(tmp, "early.trace.json")
            late = os.path.join(tmp, "late.trace.json")
            with open(early, "w") as fh:
                json.dump({"traceEvents": [{"name": "e", "ts": 10.0}]}, fh)
            with open(late, "w") as fh:
                json.dump({"traceEvents": [{"name": "l", "ts": 900.0}]}, fh)
            merged = mapping._load_events_multi([late, early])
            self.assertEqual([e["name"] for e in merged], ["e", "l"])


class TruncatedWindowSegmentationTest(unittest.TestCase):
    """A module-less window holds the layer bodies it holds -- no more."""

    PATTERNS = {
        "num_hidden_layers_main": 61,
        "patterns": [
            {"pattern_id": "P0", "ffn_type": "dense_mlp",
             "attention_type": "MLA", "layer_ids": [0, 1, 2]},
            {"pattern_id": "P1", "ffn_type": "moe_with_shared_expert",
             "attention_type": "MLA", "layer_ids": list(range(3, 61))},
        ],
    }

    def _runs(self, stages):
        runs, index = [], 0
        for stage in stages:
            runs.append({"stage": stage, "start": index, "end": index})
            index += 1
        return runs

    def _window(self, dense_bodies, moe_bodies):
        """A window of whole layer bodies, dense ones first."""
        stages = []
        for _ in range(dense_bodies):
            stages += ["norm", "attn", "gemm", "activation", "gemm"]
        for _ in range(moe_bodies):
            stages += ["norm", "attn", "gemm", "topk", "moe", "activation",
                       "moe", "gemm"]
        return self._runs(stages)

    def test_counts_bodies_present_not_layers_configured(self):
        runs = self._window(3, 18)
        patterns = mapping._pattern_index(self.PATTERNS)
        anchored = mapping._anchor_runs(runs, 61, patterns)
        self.assertIsNotNone(anchored)
        # 21 bodies are physically present; the config declares 61.
        self.assertEqual(anchored["observed_layer_bodies"], 21)
        self.assertEqual(anchored["segment_validity"], 1.0)
        self.assertEqual(anchored["layer_id_offset"], 0)

    def test_anchor_rejects_stage_firing_twice_per_layer(self):
        """A stage that fires twice per layer halves every body."""
        runs = self._window(3, 18)
        patterns = mapping._pattern_index(self.PATTERNS)
        anchored = mapping._anchor_runs(runs, 61, patterns)
        # "gemm" appears twice per body and would report ~42 bodies.
        self.assertNotEqual(anchored["anchor_stage"], "gemm")

    def test_offset_recovers_window_starting_mid_model(self):
        runs = self._window(0, 12)
        patterns = mapping._pattern_index(self.PATTERNS)
        anchored = mapping._anchor_runs(runs, 61, patterns)
        self.assertIsNotNone(anchored)
        # No dense bodies -> the window cannot start at layer 0.
        self.assertGreaterEqual(anchored["layer_id_offset"], 3)

    def test_plausibility_gate_rejects_degenerate_moe_representative(self):
        """The defect this gate exists for: a 2-kernel 'MoE layer'."""
        rows = [{"stage": stage} for stage in
                ("norm", "attn", "gemm", "topk", "moe", "activation")]
        tables = [{"pattern_id": "P1", "phase": "decode",
                   "representative_layer_id": 28,
                   "rows": [{"stage": "gemm"}, {"stage": "elementwise"}]}]
        gate = mapping._representative_plausibility(self.PATTERNS, tables, rows)
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["tables"][0]["missing_stages"],
                         ["attn", "moe", "topk"])

    def test_plausibility_gate_ignores_stages_absent_from_the_trace(self):
        """No expert kernels anywhere means the capture, not a bad cut."""
        rows = [{"stage": stage} for stage in ("norm", "attn", "gemm")]
        tables = [{"pattern_id": "P1", "phase": "decode",
                   "representative_layer_id": 28,
                   "rows": [{"stage": "attn"}, {"stage": "gemm"}]}]
        gate = mapping._representative_plausibility(self.PATTERNS, tables, rows)
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["tables"][0]["unobserved_in_trace"],
                         ["moe", "topk"])


if __name__ == "__main__":
    unittest.main()
