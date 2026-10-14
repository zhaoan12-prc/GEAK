"""The profile window must open on the real load, not on the client's warmup.

A window armed at background-client start captured the client's own warmup
requests: Kimi-K2.5 on vLLM 0.21 (CONC 16) recorded decode at bs=8 in both
the baseline Profile and the KernelFusion boundary donor, and the donor then
had no pass matching the bs=16 decode step it had to cut. bench_e2e.sh now
waits for the client to report its warmup done before arming the window.
"""
import os
import re
import subprocess
import tempfile
import time
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(SCRIPTS, "bench_e2e.sh")


def _function_source():
    with open(BENCH) as fh:
        src = fh.read()
    m = re.search(r"^_profile_wait_client_warmup\(\) \{\n.*?^\}\n", src, re.S | re.M)
    assert m, "_profile_wait_client_warmup not found in bench_e2e.sh"
    return m.group(0)


def _run(log, pid, wait_sec=10):
    script = "PROFILE_CLIENT_WARMUP_WAIT_SEC=%d\n%s\n_profile_wait_client_warmup %s %s\n" % (
        wait_sec, _function_source(), log, pid)
    start = time.time()
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return result, time.time() - start


class ClientWarmupWaitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.tmp.name, "bg.log")
        # A live process standing in for the background client.
        self.client = subprocess.Popen(["sleep", "60"])

    def tearDown(self):
        self.client.kill()
        self.client.wait()
        self.tmp.cleanup()

    def test_waits_until_inferencex_reports_warmup_done(self):
        with open(self.log, "w") as fh:
            fh.write("Warming up with 8 requests...\n")
        writer = subprocess.Popen(
            ["bash", "-c", "sleep 3; echo 'Warmup completed.' >> %s" % self.log])
        result, elapsed = _run(self.log, self.client.pid)
        writer.wait()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(elapsed, 3)
        self.assertIn("arming the profile window on the real load", result.stdout)

    def test_no_warmup_returns_immediately(self):
        # InferenceX with --num-warmups 0 goes straight to the traffic banner;
        # vllm bench prints "Starting main benchmark run...".
        for line in ("Traffic request rate: inf\n", "Starting main benchmark run...\n"):
            with open(self.log, "w") as fh:
                fh.write(line)
            result, elapsed = _run(self.log, self.client.pid)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertLess(elapsed, 2)
            self.assertEqual(result.stderr, "")

    def test_native_vllm_warmup_marker(self):
        with open(self.log, "w") as fh:
            fh.write("Warming up with 8 requests...\nWarmup run completed.\n")
        result, elapsed = _run(self.log, self.client.pid)
        self.assertEqual(result.returncode, 0)
        self.assertLess(elapsed, 2)

    def test_dead_client_does_not_hang(self):
        open(self.log, "w").close()
        self.client.kill()
        self.client.wait()
        result, elapsed = _run(self.log, self.client.pid)
        self.assertEqual(result.returncode, 0)
        self.assertLess(elapsed, 4)

    def test_timeout_warns_and_arms_anyway(self):
        with open(self.log, "w") as fh:
            fh.write("Warming up with 8 requests...\n")
        result, elapsed = _run(self.log, self.client.pid, wait_sec=2)
        self.assertEqual(result.returncode, 0)
        self.assertIn("arming the window anyway", result.stderr)


class CallSiteTest(unittest.TestCase):
    def test_background_load_is_logged_unbuffered_and_waited_on(self):
        with open(BENCH) as fh:
            src = fh.read()
        block = src[src.index('_BG_LOG="$OUT_DIR/profile_bg_load.log"'):]
        block = block[:block.index("adapter_profile_window ||")]
        self.assertIn("PYTHONUNBUFFERED=1", block)
        self.assertIn('>"$_BG_LOG" 2>&1 &', block)
        self.assertNotIn(">/dev/null 2>&1 &", block)
        self.assertLess(block.index('_profile_wait_client_warmup "$_BG_LOG" "$_bg_load"'),
                        block.index('sleep "$PROFILE_WARMUP_SEC"'))


if __name__ == "__main__":
    unittest.main()
