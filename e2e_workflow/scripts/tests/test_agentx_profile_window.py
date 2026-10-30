"""AgentX client: the profile window opens in steady state, not on the replay's cold-prefill ramp."""
import os
import subprocess
import unittest

CLIENT = os.path.join(os.path.dirname(__file__), "..", "adapters", "clients", "agentx.sh")
BENCH = os.path.join(os.path.dirname(__file__), "..", "bench_e2e.sh")


def sh(cmd, **env):
    e = {k: v for k, v in os.environ.items()
         if not k.startswith(("GEAK_AGENTX_", "AGENTX_", "MEASUREMENT_PURPOSE", "PROFILE_"))}
    e.update(env)
    return subprocess.run(["bash", "-c", f"source {CLIENT}; {cmd}"], capture_output=True, text=True,
                          env=e, timeout=30).stdout.strip()


class AgentxProfileWindowTest(unittest.TestCase):
    def test_duration_follows_the_purpose(self):
        self.assertEqual(sh("_agentx_duration"), "900")
        self.assertEqual(sh("_agentx_duration", MEASUREMENT_PURPOSE="parity"), "3600")
        self.assertEqual(sh("_agentx_duration", MEASUREMENT_PURPOSE="validation",
                            GEAK_AGENTX_DURATION_S="1800"), "1800")

    def test_window_is_placed_in_steady_state(self):
        self.assertEqual(sh("adapter_profile_warmup_s"), "675")                  # 75% of a 900 s leg
        self.assertEqual(sh("adapter_profile_warmup_s", MEASUREMENT_PURPOSE="parity"), "2700")
        self.assertEqual(sh("adapter_profile_warmup_s", GEAK_AGENTX_LOOP_DURATION_S="100"), "20")  # fits
        self.assertEqual(sh("adapter_profile_warmup_s", AGENTX_PROFILE_WARMUP_S="42"), "42")

    def test_bench_e2e_consults_the_hook_unless_the_caller_set_the_warmup(self):
        src = open(BENCH).read()
        hook = src.index('declare -F adapter_profile_warmup_s')
        self.assertLess(src.index('source "$CLIENT_ADAPTER"'), hook)   # the hook exists by then
        self.assertIn('[ -z "${PROFILE_WARMUP_SEC:-}" ] && declare -F adapter_profile_warmup_s', src)
        self.assertIn('elif [ "${GEAK_ISL_OSL_INACTIVE:-0}" = "1" ]; then', src)


if __name__ == "__main__":
    unittest.main()
