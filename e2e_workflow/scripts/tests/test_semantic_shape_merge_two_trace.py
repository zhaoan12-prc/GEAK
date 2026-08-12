import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_shape_merge as merge


class ShapeMergeTwoTraceTest(unittest.TestCase):
    def _write(self, root, name, value, jsonl=False):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            if jsonl:
                for item in value:
                    fh.write(json.dumps(item) + "\n")
            else:
                json.dump(value, fh)
        return path

    def _table(self):
        return {
            "schema_version": 2,
            "tables": [{
                "phase": "decode",
                "pattern_id": "P",
                "pattern_layer_ids": [1, 2, 3],
                "pattern_layer_count": 3,
                "representative_layer_id": 2,
                "selected_bucket": {
                    "phase": "decode", "batch_size": 4, "input_tokens": 4},
                "event_count": 2,
                "layer_total_us": 3.0,
                "rows": [
                    {
                        "pos": 0, "row_id": "event-1",
                        "raw_event_index": 1, "device_seq_index": 1,
                        "raw_name": "exact", "short_name": "exact",
                        "duration_us": 1.0,
                        "shape": {"source": "kernel_exact",
                                  "input_dims": [[4, 8]]},
                        "parent_operator": {"canonical_op": "aten::mm",
                                            "mapping_level": "external_id"},
                    },
                    {
                        "pos": 1, "row_id": "event-2",
                        "raw_event_index": 2, "device_seq_index": 2,
                        "raw_name": "native_gemm",
                        "short_name": "native_gemm", "duration_us": 2.0,
                        "shape": {"source": "unresolved", "input_dims": None},
                        "parent_operator": {"canonical_op": "unresolved",
                                            "mapping_level": "unresolved"},
                    },
                ],
            }],
        }

    def _two_trace_map(self, dims=None):
        return {
            "schema_version": 1,
            "entries": {
                "event-2": {
                    "row_id": "event-2",
                    "pattern_id": "P",
                    "phase": "decode",
                    "formal_pos": 1,
                    "mapping_pos": 1,
                    "raw_name": "native_gemm",
                    "match": {
                        "binding": "kernel_name_plus_ordered_position",
                        "kernel_name_match": "exact",
                        "position_delta": 0,
                        "representative_layer_match": "same_layer",
                        "confidence": "high",
                    },
                    "op": {
                        "canonical_op": "model.layers.2.self_attn.o_proj",
                        "op_instance_id": "ext-77",
                        "mapping_level": "external_id",
                        "mapping_cardinality": "1:1",
                        "device_launch_count": 1,
                        "recovered": True,
                    },
                    "shape": {
                        "input_dims": dims if dims is not None else [[4, 16]],
                        "input_types": ["c10::BFloat16"],
                        "source": "kernel_exact",
                        "recovered": dims is not False,
                    },
                },
            },
        }

    def _run(self, tmp, two_trace_map=None, records=None):
        plan = {"capture_targets": [{
            "row_id": "event-2",
            "candidate_wrapper": "model.layers.2.proj",
            "candidate_terminal_launcher": None,
            "mapping_cardinality": "1:N",
            "parent_operator": "unresolved",
        }]}
        map_path = ""
        if two_trace_map is not None:
            map_path = self._write(tmp, "two_trace.json", two_trace_map)
        return merge.merge(
            self._write(tmp, "table.json", self._table()),
            self._write(tmp, "plan.json", plan),
            self._write(tmp, "shape.jsonl", records or [], jsonl=True),
            os.path.join(tmp, "out"),
            map_path)

    def test_two_trace_dims_produce_kernel_scope_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp, self._two_trace_map())
            self.assertEqual(result["status"], "pass")
            with open(result["semantic_table_json"]) as fh:
                rows = json.load(fh)["tables"][0]["rows"]
            evidence = rows[1]["semantic_evidence"]
            self.assertEqual(evidence["level"], "P")
            self.assertEqual(evidence["probe_scope"], "kernel")
            self.assertEqual(evidence["evidence_origin"], "two_trace_mapping")
            self.assertEqual(
                evidence["binding"], "kernel_name_plus_ordered_position")
            self.assertEqual(rows[1]["shape"]["source"], "two_trace_kernel_dims")
            self.assertEqual(rows[1]["shape"]["input_dims"], [[4, 16]])

    def test_operator_attribution_is_transplanted(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp, self._two_trace_map())
            with open(result["semantic_table_json"]) as fh:
                rows = json.load(fh)["tables"][0]["rows"]
            parent = rows[1]["parent_operator"]
            self.assertEqual(
                parent["canonical_op"], "model.layers.2.self_attn.o_proj")
            self.assertEqual(parent["mapping_level"], "two_trace_external_id")
            self.assertEqual(parent["recovered_by"], "two_trace_mapping")

    def test_clean_trace_k_evidence_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            two_trace = self._two_trace_map()
            two_trace["entries"]["event-1"] = dict(
                two_trace["entries"]["event-2"], row_id="event-1")
            result = self._run(tmp, two_trace)
            with open(result["semantic_table_json"]) as fh:
                rows = json.load(fh)["tables"][0]["rows"]
            self.assertEqual(rows[0]["semantic_evidence"]["level"], "K")
            self.assertEqual(rows[0]["shape"]["source"], "kernel_exact")
            self.assertEqual(rows[0]["shape"]["input_dims"], [[4, 8]])
            self.assertEqual(
                rows[0]["parent_operator"]["canonical_op"], "aten::mm")

    def test_row_identity_and_durations_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp, self._two_trace_map())
            with open(result["shape_type_verification_json"]) as fh:
                verification = json.load(fh)
            self.assertTrue(verification["clean_trace_identity_unchanged"])
            self.assertTrue(verification["two_trace_mapping"]["enabled"])
            self.assertEqual(verification["two_trace_mapping"]["shape_rows"], 1)
            self.assertEqual(
                verification["two_trace_mapping"]["operator_rows"], 1)

    def test_tensor_list_argument_yields_one_tensor_per_element(self):
        # aten::cat takes a TensorList, so the trace nests one shape per element
        # in a single argument. Those are the shapes the row is about.
        with tempfile.TemporaryDirectory() as tmp:
            two_trace = self._two_trace_map(
                dims=[[[4, 16, 512], [4, 16, 64]], []])
            two_trace["entries"]["event-2"]["shape"]["input_types"] = [
                "TensorList", "Scalar"]
            result = self._run(tmp, two_trace)
            self.assertEqual(result["status"], "pass")
            with open(result["semantic_table_json"]) as fh:
                rows = json.load(fh)["tables"][0]["rows"]
            tensors = rows[1]["shape"]["logger_schema"]["tensors"]
            self.assertEqual(
                [tensor["arg_name"] for tensor in tensors],
                ["args[0][0]", "args[0][1]"])
            self.assertEqual(
                [tensor["shape"] for tensor in tensors],
                [[4, 16, 512], [4, 16, 64]])
            # The scalar argument carries no shape and is not invented.
            self.assertTrue(
                all(tensor["dtype"] == "TensorList" for tensor in tensors))

    def test_unrepresentable_argument_is_skipped_not_crashed(self):
        for dims in ([[4, [16]], [8, 8]], [[[[4, 8]]], [8, 8]]):
            with tempfile.TemporaryDirectory() as tmp:
                result = self._run(tmp, self._two_trace_map(dims=dims))
                self.assertEqual(result["status"], "pass")
                with open(result["semantic_table_json"]) as fh:
                    rows = json.load(fh)["tables"][0]["rows"]
                tensors = rows[1]["shape"]["logger_schema"]["tensors"]
                self.assertEqual(
                    [tensor["shape"] for tensor in tensors], [[8, 8]],
                    "unreadable argument %r should yield no tensor" % (
                        dims[0],))

    def test_without_map_the_row_stays_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp, None)
            with open(result["semantic_table_json"]) as fh:
                rows = json.load(fh)["tables"][0]["rows"]
            self.assertEqual(rows[1]["semantic_evidence"]["level"], "U")
            with open(result["shape_type_verification_json"]) as fh:
                self.assertFalse(
                    json.load(fh)["two_trace_mapping"]["enabled"])


if __name__ == "__main__":
    unittest.main()
