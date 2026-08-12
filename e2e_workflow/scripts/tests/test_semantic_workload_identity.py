import os
import sys
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_workload_identity as identity


def _setup(**overrides):
    setup = {
        "model": "/data/models/deepseek-ai/DeepSeek-R1-0528",
        "tensor_parallel_size": 8,
        "benchmark": "benchmarks/single_node/fixed_seq_len/dsr1_fp8_mi300x.sh",
        "benchmark_repository": "/data/wangxd/GEAK_OP/InferenceX",
        "workload": {
            "concurrency": 4,
            "input_length": 8192,
            "output_length": 1024,
            "random_range_ratio": 0.8,
        },
        "disable_cuda_graph": False,
    }
    setup.update(overrides)
    return setup


def _one(batch_size=4, input_tokens=0, pattern="P", phase="decode"):
    return {
        "pattern_id": pattern,
        "phase": phase,
        "selected_bucket": {
            "phase": phase,
            "batch_size": batch_size,
            "input_tokens": input_tokens,
        },
    }


def _table(batch_size=4, input_tokens=0, pattern="P", phase="decode"):
    return {"tables": [_one(batch_size, input_tokens, pattern, phase)]}


class WorkloadIdentityTest(unittest.TestCase):
    def test_identical_workload_passes_when_mapping_is_graph_off(self):
        result = identity.verify(
            _setup(), _setup(disable_cuda_graph=True),
            _table(), _table())
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["failure_reasons"], [])

    def test_workload_field_mismatch_raises(self):
        with self.assertRaises(identity.WorkloadIdentityError) as ctx:
            identity.verify(
                _setup(),
                _setup(disable_cuda_graph=True,
                       workload={"concurrency": 8, "input_length": 8192,
                                 "output_length": 1024,
                                 "random_range_ratio": 0.8}),
                _table(), _table())
        self.assertIn("concurrency", str(ctx.exception))

    def test_graph_on_mapping_run_is_rejected(self):
        with self.assertRaises(identity.WorkloadIdentityError) as ctx:
            identity.verify(
                _setup(), _setup(disable_cuda_graph=False),
                _table(), _table())
        self.assertIn("graph-off", str(ctx.exception))

    def test_observed_bucket_mismatch_skips_that_table(self):
        """Launched identically is not enough; the profiled step must match.

        But it is a verdict on one table, not on the comparison.
        """
        result = identity.verify(
            _setup(), _setup(disable_cuda_graph=True),
            _table(batch_size=4), _table(batch_size=1))
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["failure_reasons"], [])
        self.assertEqual(result["skipped_table_keys"], ["P|decode"])
        self.assertEqual(result["bound_tables"], [])
        self.assertEqual(
            result["skipped_tables"][0]["reason"], "bucket_differs")

    def test_a_table_the_mapping_replay_missed_does_not_sink_the_rest(self):
        # The mapping replay's window is routinely shorter than the formal run's.
        formal = {"tables": [
            _one(phase="prefill", input_tokens=8192),
            _one(phase="decode"),
        ]}
        mapping = {"tables": [_one(phase="prefill", input_tokens=8192)]}
        result = identity.verify(
            _setup(), _setup(disable_cuda_graph=True), formal, mapping)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["failure_reasons"], [])
        self.assertEqual(result["bound_tables"], ["P|prefill"])
        self.assertEqual(result["skipped_table_keys"], ["P|decode"])
        self.assertEqual(
            result["skipped_tables"][0]["reason"], "missing_in_mapping_trace")

    def test_no_comparable_table_is_a_no_op_not_an_error(self):
        result = identity.verify(
            _setup(), _setup(disable_cuda_graph=True),
            _table(), {"tables": []})
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["failure_reasons"], [])
        self.assertEqual(result["bound_tables"], [])

    def test_declared_mismatch_still_raises_even_when_buckets_line_up(self):
        # A per-table verdict cannot rescue two runs that are not the same
        # experiment: nothing from them may be bound.
        with self.assertRaises(identity.WorkloadIdentityError):
            identity.verify(
                _setup(),
                _setup(disable_cuda_graph=True, tensor_parallel_size=4),
                _table(), _table())

    def test_undeclared_workload_field_raises(self):
        broken = _setup(disable_cuda_graph=True)
        broken.pop("model")
        with self.assertRaises(identity.WorkloadIdentityError) as ctx:
            identity.verify(_setup(), broken, _table(), _table())
        self.assertIn("model", str(ctx.exception))

    def test_float_ratio_is_compared_canonically(self):
        left = _setup()
        right = _setup(disable_cuda_graph=True)
        right["workload"]["random_range_ratio"] = 0.8000000000000001
        result = identity.verify(left, right, _table(), _table())
        self.assertEqual(result["status"], "pass")

    def test_non_strict_reports_a_declared_mismatch_without_raising(self):
        result = identity.verify(
            _setup(), _setup(disable_cuda_graph=True, model="other"),
            _table(), _table(), strict=False)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
