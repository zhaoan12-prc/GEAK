import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import serving_shape_envelope as sse  # noqa: E402


class VllmEnvelopeTest(unittest.TestCase):
    def test_capture_size_and_chunk_budget_pin_both_phases(self):
        env = sse.derive("vllm", "--tensor-parallel-size 2 --max-cudagraph-capture-size 1024 "
                                 "--max-num-batched-tokens 16384")
        self.assertEqual(env["phases"]["graph"]["max_tokens"], 1024)
        self.assertEqual(env["phases"]["prefill"]["max_tokens"], 16384)
        self.assertEqual(sse.required_points(env, "decode"), [1, 1024])
        self.assertEqual(sse.required_points(env, "prefill"), [1, 16384])

    def test_compilation_config_capture_sizes(self):
        env = sse.derive("vllm", "--compilation-config '{\"cudagraph_capture_sizes\": [8, 16, 512]}'")
        self.assertEqual(sse.required_points(env, "decode"), [8, 512])

    def test_eager_or_unpinned_leaves_the_phase_out(self):
        for args in ("--enforce-eager --max-cudagraph-capture-size 512", "--tensor-parallel-size 2"):
            env = sse.derive("vllm", args)
            self.assertIsNone(sse.required_points(env, "decode"))
            self.assertTrue(env["notes"])


class SglangEnvelopeTest(unittest.TestCase):
    def test_dspark_verify_rows_are_batch_times_block_plus_one(self):
        env = sse.derive("sglang", "--speculative-algorithm DSPARK --speculative-dspark-block-size 5 "
                                   "--cuda-graph-max-bs-decode 64 --chunked-prefill-size 16384")
        self.assertEqual(sse.required_points(env, "verify"), [1, 384])
        self.assertEqual(sse.required_points(env, "extend"), [1, 16384])

    def test_decode_flag_wins_over_the_generic_one_and_eagle_uses_draft_tokens(self):
        env = sse.derive("sglang", "--cuda-graph-max-bs 160 --cuda-graph-max-bs-decode 32 "
                                   "--speculative-num-draft-tokens 4")
        self.assertEqual(sse.required_points(env, "decode"), [1, 128])

    def test_unknown_backend_or_phase_is_unpinned(self):
        self.assertIsNone(sse.required_points(sse.derive("atom", "--x 1"), "decode"))
        env = sse.derive("vllm", "--max-cudagraph-capture-size 64")
        self.assertIsNone(sse.required_points(env, "weird_phase"))


if __name__ == "__main__":
    unittest.main()
