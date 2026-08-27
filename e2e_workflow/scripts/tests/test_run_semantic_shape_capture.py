import os
import sys
import tempfile
import json
import unittest
from unittest import mock


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import run_semantic_shape_capture as capture


class RunSemanticShapeCaptureTest(unittest.TestCase):
    def test_required_phases_are_inferred_from_capture_plan(self):
        plan = {
            "target_buckets": [
                {"phase": "decode"},
                {"phase": "prefill"},
                {"phase": "decode"},
            ]
        }
        self.assertEqual(
            capture._required_phases(plan), ["decode", "prefill"])

    def test_injects_eager_flag_for_inline_server_arguments(self):
        text = "launch --disable-radix-cache $EVAL_CONTEXT_ARGS"
        self.assertIn(
            "--disable-radix-cache --disable-cuda-graph",
            capture._with_disable_cuda_graph(text))

    def test_injects_eager_flag_for_multiline_server_arguments(self):
        text = (
            "launch \\\n"
            "  --disable-radix-cache \\\n"
            "  --max-prefill-tokens 32768")
        result = capture._with_disable_cuda_graph(text)
        self.assertIn(
            "--disable-radix-cache --disable-cuda-graph \\", result)

    def test_existing_eager_flag_is_idempotent(self):
        text = "--disable-radix-cache --disable-cuda-graph"
        self.assertEqual(capture._with_disable_cuda_graph(text), text)



class OwnProcessGroupTeardownTest(unittest.TestCase):
    """Regression tests for B6 (docs/decode-coverage-bugs.md)."""

    def test_b6_teardown_group_kills_recorded_pgid_never_pattern_kills(self):
        with mock.patch.object(capture, "_docker") as docker:
            capture._stop_service("container", "/out/server.pgid", 8935)
        command = docker.call_args[0][1]
        self.assertNotIn("pkill", command)
        self.assertNotIn("pgrep", command)
        self.assertIn("/out/server.pgid", command)
        # The leading '-' is what makes this a process-group kill.
        self.assertIn('kill -TERM -"$pgid"', command)
        self.assertIn('kill -KILL -"$pgid"', command)
        # Liveness is polled on the port: the server leaves unreaped
        # children behind, so `kill -0` on the group never clears.
        self.assertIn("/dev/tcp/127.0.0.1/8935", command)

    def test_b6_teardown_is_a_noop_when_no_pgid_was_recorded(self):
        with mock.patch.object(capture, "_docker") as docker:
            capture._stop_service("container", "/out/server.pgid", 8935)
        command = docker.call_args[0][1]
        self.assertIn('[ -s "$pgid_file" ] || exit 0', command)

    def test_b6_busy_port_is_reported_not_killed(self):
        completed = mock.Mock(stdout=b"BUSY\n")
        with mock.patch.object(capture.subprocess, "run",
                               return_value=completed):
            with self.assertRaises(RuntimeError) as caught:
                capture._assert_port_free("container", 8935)
        self.assertIn("8935", str(caught.exception))
        self.assertIn("did not start", str(caught.exception))

    def test_b6_free_port_passes(self):
        completed = mock.Mock(stdout=b"FREE\n")
        with mock.patch.object(capture.subprocess, "run",
                               return_value=completed):
            capture._assert_port_free("container", 8935)


class DecodeProbeTest(unittest.TestCase):
    """Regression tests for B4/B5 (docs/decode-coverage-bugs.md)."""

    def test_b5_rank_filter_matches_profile_by_stage_names(self):
        """The old filter tested for '-TP-0.trace.json', which never matches."""
        with tempfile.TemporaryDirectory() as tmp:
            names = ["s-TP-0-EXTEND.trace.json.gz",
                     "s-TP-0-DECODE.trace.json.gz",
                     "s-TP-5-EXTEND.trace.json.gz",
                     "s-TP-5-DECODE.trace.json.gz"]
            for name in names:
                open(os.path.join(tmp, name), "w").close()
            by_phase = capture._traces_by_phase(tmp, 0)
            self.assertIn("EXTEND", by_phase)
            self.assertIn("DECODE", by_phase)
            for path in by_phase.values():
                self.assertIn("-TP-0-", os.path.basename(path))
            # deterministic: never a race with whichever rank wrote last
            self.assertIn("-TP-0-EXTEND", capture._latest_trace(tmp, 0))

    def test_b4_narrowed_phases_are_reported(self):
        notes = capture._warn_narrowed_phases(
            ["prefill"], {"target_buckets": [{"phase": "prefill"}]},
            "plan.target_buckets")
        self.assertTrue(notes)
        joined = " ".join(notes)
        self.assertIn("decode", joined)
        self.assertIn("eager probe", joined)

    def test_b4_full_phase_set_warns_about_nothing(self):
        self.assertEqual(
            capture._warn_narrowed_phases(["prefill", "decode"], {}, "cli"), [])

    def test_observed_phases_reads_the_shape_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "shape.jsonl")
            with open(log, "w") as fh:
                fh.write(json.dumps({"context": {"phase": "EXTEND"}}) + "\n")
                fh.write(json.dumps({"context": {"phase": "DECODE"}}) + "\n")
                fh.write("not json\n")
            # EXTEND is an alias of prefill; the audit must compare like for like
            self.assertEqual(capture._observed_phases(log),
                             {"prefill", "decode"})


if __name__ == "__main__":
    unittest.main()
