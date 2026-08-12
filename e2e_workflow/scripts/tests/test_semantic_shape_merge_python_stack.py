import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_shape_merge as merge


LAUNCH_SITE = "gemm_a8w8_blockscale.py(19): gemm_a8w8_blockscale"


class ShapeMergePythonStackTest(unittest.TestCase):
    """A Triton row reaches no cpu_op, so Semantics 1.1 hands the merge a
    python_stack parent. It must survive the shape-logger probe, which knows
    only the enclosing module, and still lose to a two-trace binding.
    """

    def _write(self, root, name, value, jsonl=False):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            if jsonl:
                for item in value:
                    fh.write(json.dumps(item) + "\n")
            else:
                json.dump(value, fh)
        return path

    def _table(self, mapping_level="python_stack"):
        return {
            "schema_version": 2,
            "tables": [{
                "phase": "prefill",
                "pattern_id": "P",
                "pattern_layer_ids": [1, 2, 3],
                "pattern_layer_count": 3,
                "representative_layer_id": 1,
                "selected_bucket": {
                    "phase": "prefill", "batch_size": 1, "input_tokens": 8},
                "event_count": 1,
                "layer_total_us": 2.0,
                "rows": [{
                    "pos": 0, "row_id": "event-1",
                    "raw_event_index": 1, "device_seq_index": 1,
                    "raw_name": "_gemm_a8w8_blockscale_kernel",
                    "short_name": "_gemm_a8w8_blockscale_kernel",
                    "duration_us": 2.0,
                    "shape": {"source": "unresolved", "input_dims": None},
                    "parent_operator": {
                        "canonical_op": (
                            LAUNCH_SITE if mapping_level == "python_stack"
                            else "unresolved"),
                        "mapping_level": mapping_level,
                        "python_launch_site": (
                            LAUNCH_SITE if mapping_level == "python_stack"
                            else None),
                    },
                }],
            }],
        }

    def _records(self):
        return [{
            "event": "wrapper",
            "phase": "prefill", "batch_size": 1, "input_tokens": 8,
            "layer_id": 1, "op_path": "model.layers.1",
            "op_instance_id": "wrapper-1",
            "tensors": [{"arg_name": "wrapper_input_0", "io": "input",
                         "shape": [8, 16], "dtype": "c10::BFloat16"}],
        }]

    def _run(self, tmp, table=None, records=None, two_trace=None):
        plan = {"capture_targets": [{
            "row_id": "event-1",
            "candidate_wrapper": "model.layers.1",
            "candidate_terminal_launcher": None,
            "mapping_cardinality": "1:N",
            "parent_operator": "unresolved",
        }]}
        map_path = ""
        if two_trace is not None:
            map_path = self._write(tmp, "two_trace.json", two_trace)
        return merge.merge(
            self._write(tmp, "table.json", table or self._table()),
            self._write(tmp, "plan.json", plan),
            self._write(tmp, "shape.jsonl",
                        self._records() if records is None else records,
                        jsonl=True),
            os.path.join(tmp, "out"),
            map_path)

    def _row(self, result):
        with open(result["semantic_table_json"]) as fh:
            return json.load(fh)["tables"][0]["rows"][0]

    def test_launch_site_survives_the_wrapper_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(self._run(tmp))
            evidence = row["semantic_evidence"]
            self.assertEqual(evidence["level"], "P")
            self.assertEqual(evidence["probe_scope"], "kernel")
            self.assertEqual(evidence["source"], "python_stack_launch_site")
            self.assertEqual(evidence["contained_by"], LAUNCH_SITE)
            # The op is kernel-scope, the shape is not; both are stated.
            self.assertEqual(evidence["shape_scope"], "wrapper")
            self.assertEqual(evidence["wrapper_op_path"], "model.layers.1")
            self.assertEqual(row["parent_operator"]["canonical_op"],
                             LAUNCH_SITE)

    def test_without_a_launch_site_the_wrapper_still_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = self._row(self._run(tmp, table=self._table("unresolved")))
            self.assertEqual(row["semantic_evidence"]["probe_scope"], "wrapper")
            self.assertEqual(
                row["parent_operator"]["canonical_op"], "model.layers.1")

    def test_launch_site_without_any_shape_probe_is_p_not_u(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run(tmp, records=[])
            row = self._row(result)
            evidence = row["semantic_evidence"]
            self.assertEqual(evidence["level"], "P")
            self.assertEqual(evidence["probe_scope"], "kernel")
            self.assertEqual(evidence["shape_scope"], "none")
            with open(result["semantic_table_md"]) as fh:
                self.assertIn("shape unavailable", fh.read())

    def test_two_trace_outranks_the_python_stack(self):
        # Two-trace carries Input Dims as well as an op, so it wins -- but the
        # published level must say which route produced it.
        with tempfile.TemporaryDirectory() as tmp:
            two_trace = {"schema_version": 1, "entries": {"event-1": {
                "row_id": "event-1", "pattern_id": "P", "phase": "prefill",
                "match": {"binding": "kernel_name_plus_ordered_position",
                          "kernel_name_match": "exact", "position_delta": 0,
                          "representative_layer_match": "same_layer",
                          "confidence": "high"},
                "op": {"canonical_op": "aten::mm", "op_instance_id": "ext-3",
                       "mapping_level": "external_id", "recovered": True},
                "shape": {"input_dims": [[8, 16]],
                          "input_types": ["c10::BFloat16"],
                          "source": "kernel_exact", "recovered": True},
            }}}
            row = self._row(self._run(tmp, two_trace=two_trace))
            self.assertEqual(row["parent_operator"]["canonical_op"], "aten::mm")
            self.assertEqual(row["parent_operator"]["mapping_level"],
                             "two_trace_external_id")

    def test_two_trace_does_not_restate_the_stack_from_further_away(self):
        # Two-trace binds to a different layer instance in a different run. If
        # its op also came from a python stack and it brings no dims, it adds
        # distance, not information -- this trace already said the same thing.
        with tempfile.TemporaryDirectory() as tmp:
            two_trace = {"schema_version": 1, "entries": {"event-1": {
                "row_id": "event-1", "pattern_id": "P", "phase": "prefill",
                "match": {"binding": "kernel_name_plus_ordered_position",
                          "kernel_name_match": "exact", "position_delta": 0,
                          "representative_layer_match":
                              "different_layer_same_pattern",
                          "confidence": "medium"},
                "op": {"canonical_op": LAUNCH_SITE, "op_instance_id": None,
                       "mapping_level": "python_stack", "recovered": True},
                "shape": {"input_dims": [], "input_types": [],
                          "source": "unresolved", "recovered": False},
            }}}
            row = self._row(self._run(tmp, two_trace=two_trace))
            self.assertEqual(row["parent_operator"]["mapping_level"],
                             "python_stack")
            self.assertEqual(row["semantic_evidence"]["evidence_origin"],
                             "python_stack")

    def test_two_trace_via_python_stack_is_labelled_as_such(self):
        # The graph-on decode case: this trace has no stack of its own (the
        # replay erased it), so the mapping trace's stack is the only route.
        with tempfile.TemporaryDirectory() as tmp:
            two_trace = {"schema_version": 1, "entries": {"event-1": {
                "row_id": "event-1", "pattern_id": "P", "phase": "prefill",
                "match": {"binding": "kernel_name_plus_ordered_position",
                          "kernel_name_match": "exact", "position_delta": 0,
                          "representative_layer_match": "same_layer",
                          "confidence": "high"},
                "op": {"canonical_op": LAUNCH_SITE, "op_instance_id": None,
                       "mapping_level": "python_stack", "recovered": True},
                "shape": {"input_dims": [], "input_types": [],
                          "source": "unresolved", "recovered": False},
            }}}
            row = self._row(self._run(
                tmp, table=self._table("unresolved"), two_trace=two_trace))
            self.assertEqual(row["parent_operator"]["mapping_level"],
                             "two_trace_python_stack")
            self.assertEqual(row["parent_operator"]["canonical_op"],
                             LAUNCH_SITE)


if __name__ == "__main__":
    unittest.main()
