import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_shape_merge as merge


# The capture plan reports the frame where the launcher is DEFINED.
PLAN_OPERATOR = (
    "aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py(19): "
    "gemm_a8w8_blockscale")
# The probe wraps the binding the caller actually invokes, which is an alias.
LAUNCHER_TARGET = (
    "sglang.srt.layers.quantization.fp8_utils:triton_gemm_a8w8_blockscale")


def _tensor(shape, dtype):
    return {"kind": "tensor", "shape": shape, "dtype": dtype,
            "device": "cuda:0", "stride": [shape[-1], 1], "contiguous": True}


class ShapeMergeLauncherProbeTest(unittest.TestCase):
    """A targeted-launcher probe carries the kernel's own operands.

    ``semantic_runtime_capture`` wraps the real launcher and logs that call's
    arguments, so for a blockscale GEMM it holds the true FP8 A/B and hence
    M/N/K. The enclosing module hook only ever sees the layer's bf16 hidden
    state. The merge must therefore prefer the launcher record, and must not
    require ``candidate_terminal_launcher`` (which semantic_source_mapping
    leaves unset when it reports ``not_found``) or a ``1:1`` cardinality.
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

    def _row(self, row_id, pos, index):
        return {
            "pos": pos, "row_id": row_id,
            "raw_event_index": index, "device_seq_index": index,
            "raw_name": "_gemm_a8w8_blockscale_kernel_GROUP_K_128",
            "short_name": "_gemm_a8w8_blockscale_kernel",
            "duration_us": 2.0,
            "stage": "gemm",
            "shape": {"source": "unresolved", "input_dims": None},
            "parent_operator": {
                "canonical_op": PLAN_OPERATOR,
                "mapping_level": "python_stack",
                "python_launch_site": PLAN_OPERATOR,
            },
        }

    def _table(self, rows):
        return {
            "schema_version": 2,
            "tables": [{
                "phase": "prefill",
                "pattern_id": "P",
                "pattern_layer_ids": [1],
                "pattern_layer_count": 1,
                "representative_layer_id": 1,
                "selected_bucket": {
                    "phase": "prefill", "batch_size": 1, "input_tokens": 8},
                "event_count": len(rows),
                "layer_total_us": 2.0 * len(rows),
                "rows": rows,
            }],
        }

    def _launcher_record(self, module, oid, n_out, passes=1):
        """One launcher probe record, as semantic_runtime_capture writes it."""
        return [{
            "schema": "geak.semantics_runtime.v2",
            "phase": "prefill", "batch_size": 1, "input_tokens": 8,
            "rank": 0, "layer_id": 1,
            "op_path": "%s::launcher:%s" % (module, LAUNCHER_TARGET),
            "op_instance_id": "%s-%d" % (oid, index),
            "evidence_level": "targeted_launcher_probe",
            "mapping_cardinality": "probe_required",
            "inputs": {"kind": "tuple", "items": [
                _tensor([8, 16], "float8_e4m3fnuz"),
                _tensor([n_out, 16], "float8_e4m3fnuz"),
            ]},
            "output": _tensor([8, n_out], "bfloat16"),
        } for index in range(passes)]

    def _module_record(self, module, oid):
        return {
            "schema": "geak.semantics_runtime.v2",
            "phase": "prefill", "batch_size": 1, "input_tokens": 8,
            "rank": 0, "layer_id": 1,
            "op_path": module, "op_instance_id": oid,
            "inputs": {"kind": "tuple", "items": [
                _tensor([8, 999], "bfloat16")]},
            "output": _tensor([8, 999], "bfloat16"),
        }

    def _run(self, tmp, table, records):
        plan = {"capture_targets": [{
            "row_id": row["row_id"],
            "candidate_op_path": None,
            "candidate_wrapper": None,
            # semantic_source_mapping reports not_found for these rows.
            "candidate_terminal_launcher": None,
            "mapping_cardinality": "unresolved",
            "parent_operator": PLAN_OPERATOR,
        } for row in table["tables"][0]["rows"]]}
        return merge.merge(
            self._write(tmp, "table.json", table),
            self._write(tmp, "plan.json", plan),
            self._write(tmp, "shape.jsonl", records, jsonl=True),
            os.path.join(tmp, "out"))

    def _rows(self, result):
        with open(result["semantic_table_json"]) as fh:
            return json.load(fh)["tables"][0]["rows"]

    def test_launcher_probe_gives_the_row_kernel_scope_operands(self):
        table = self._table([self._row("event-1", 0, 1)])
        records = ([self._module_record("model.layers.1.mlp.down_proj", "w-1")]
                   + self._launcher_record(
                       "model.layers.1.mlp.down_proj", "call-1", 32))
        with tempfile.TemporaryDirectory() as tmp:
            row = self._rows(self._run(tmp, table, records))[0]
            evidence = row["semantic_evidence"]
            self.assertEqual(evidence["level"], "P")
            self.assertEqual(evidence["probe_scope"], "kernel")
            self.assertEqual(
                evidence["source"], "shape_logger_terminal_launcher")
            # No shape_scope override means the dims are kernel scope too.
            self.assertNotEqual(evidence.get("shape_scope"), "wrapper")
            self.assertEqual(row["shape"]["source"], "runtime_probe_kernel")
            shapes = [t["shape"] for t in evidence["schema"]["tensors"]]
            # The launcher's FP8 operands, not the module's [8, 999] bf16.
            self.assertIn([8, 16], shapes)
            self.assertIn([32, 16], shapes)
            self.assertNotIn([8, 999], shapes)

    def test_repeated_passes_collapse_to_one_binding_per_module(self):
        """The replay captures several forwards; the table has one row each."""
        table = self._table([self._row("event-1", 0, 1)])
        records = self._launcher_record(
            "model.layers.1.mlp.down_proj", "call", 32, passes=3)
        with tempfile.TemporaryDirectory() as tmp:
            row = self._rows(self._run(tmp, table, records))[0]
            self.assertEqual(
                row["semantic_evidence"]["probe_scope"], "kernel")
            self.assertEqual(row["shape"]["source"], "runtime_probe_kernel")

    def test_two_modules_bind_in_captured_order(self):
        table = self._table(
            [self._row("event-1", 0, 1), self._row("event-2", 1, 2)])
        records = (
            self._launcher_record("model.layers.1.attn.q_proj", "c1", 32)
            + self._launcher_record("model.layers.1.mlp.down_proj", "c2", 64))
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(self._run(tmp, table, records))
            first, second = rows[0], rows[1]
            self.assertEqual(
                first["semantic_evidence"]["probe_scope"], "kernel")
            self.assertEqual(
                second["semantic_evidence"]["probe_scope"], "kernel")
            self.assertIn(
                [32, 16],
                [t["shape"] for t in first["semantic_evidence"]["schema"]["tensors"]])
            self.assertIn(
                [64, 16],
                [t["shape"] for t in second["semantic_evidence"]["schema"]["tensors"]])

    def test_ambiguous_counts_do_not_bind(self):
        """Two rows but one probe record: binding would be a guess."""
        table = self._table(
            [self._row("event-1", 0, 1), self._row("event-2", 1, 2)])
        records = ([self._module_record("model.layers.1.mlp.down_proj", "w-1")]
                   + self._launcher_record(
                       "model.layers.1.mlp.down_proj", "call-1", 32))
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(self._run(tmp, table, records))
            for row in rows:
                self.assertNotEqual(
                    row["semantic_evidence"].get("source"),
                    "shape_logger_terminal_launcher")

    def test_kernel_exact_rows_do_not_block_a_sibling_binding(self):
        """A Clean Trace row never consults probe groups, so it must not be
        counted when checking that rows and probe records line up 1:1."""
        resolved = self._row("event-0", 0, 0)
        resolved["shape"] = {
            "source": "kernel_exact", "input_dims": [[8, 16], [32, 16]]}
        table = self._table([resolved, self._row("event-1", 1, 1)])
        records = self._launcher_record(
            "model.layers.1.mlp.down_proj", "call-1", 32)
        with tempfile.TemporaryDirectory() as tmp:
            rows = self._rows(self._run(tmp, table, records))
            self.assertEqual(rows[0]["semantic_evidence"]["level"], "K")
            # probe_scope alone is not proof: a python_stack launch site
            # already reports a kernel-scope OP with a wrapper-scope shape.
            # The launcher source and the FP8 dims are what only binding gives.
            self.assertEqual(rows[1]["semantic_evidence"]["source"],
                             "shape_logger_terminal_launcher")
            self.assertEqual(rows[1]["shape"]["source"], "runtime_probe_kernel")
            self.assertIn(
                [32, 16],
                [t["shape"]
                 for t in rows[1]["semantic_evidence"]["schema"]["tensors"]])


if __name__ == "__main__":
    unittest.main()
