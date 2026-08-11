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


def _table(batch_size=4, input_tokens=0, pattern="P", phase="decode"):
    return {"tables": [{
        "pattern_id": pattern,
        "phase": phase,
        "selected_bucket": {
            "phase": phase,
            "batch_size": batch_size,
            "input_tokens": input_tokens,
        },
    }]}


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

    def test_observed_bucket_mismatch_raises(self):
        """Launched identically is not enough; the profiled step must match."""
        with self.assertRaises(identity.WorkloadIdentityError) as ctx:
            identity.verify(
                _setup(), _setup(disable_cuda_graph=True),
                _table(batch_size=4), _table(batch_size=1))
        self.assertIn("bucket differs", str(ctx.exception))

    def test_missing_mapping_table_raises(self):
        with self.assertRaises(identity.WorkloadIdentityError):
            identity.verify(
                _setup(), _setup(disable_cuda_graph=True),
                _table(), {"tables": []})

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

    def test_non_strict_reports_without_raising(self):
        result = identity.verify(
            _setup(), _setup(disable_cuda_graph=True),
            _table(batch_size=4), _table(batch_size=2), strict=False)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
