"""The apply-back A/B stop rule: stop early only when the answer is already clear."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fusion_ab_decide as ab  # noqa: E402


class FusionAbDecideTest(unittest.TestCase):
    def test_one_pair_is_never_enough(self):
        self.assertEqual(ab.decide([191.1], [202.7])["decision"], "continue")

    def test_clear_win_stops_after_two_pairs(self):
        # Measured on Qwen3.5-397B (router fusion e02): +6.0%, spread ~0.15%.
        out = ab.decide([191.154, 191.096], [202.704, 202.411])
        self.assertEqual(out["decision"], "accept")
        self.assertTrue(out["nonoverlap"])
        self.assertGreater(out["delta_pct"], 5.9)

    def test_small_real_win_needs_the_third_pair(self):
        ref, cand = [200.0, 200.4], [202.0, 202.3]      # +0.97%, spread 0.2% -> not 10x
        self.assertEqual(ab.decide(ref, cand)["decision"], "continue")
        self.assertEqual(ab.decide(ref + [200.2], cand + [202.1])["decision"], "accept")

    def test_overlap_at_max_pairs_rejects(self):
        out = ab.decide([200.0, 201.5, 200.8], [201.0, 202.0, 201.2])
        self.assertFalse(out["nonoverlap"])
        self.assertEqual(out["decision"], "reject")

    def test_inside_noise_band_rejects_at_max_pairs(self):
        out = ab.decide([200.0, 200.02, 200.01], [200.5, 200.52, 200.51])   # +0.25%
        self.assertTrue(out["nonoverlap"])
        self.assertEqual(out["decision"], "reject")

    def test_slower_candidate_rejects_immediately(self):
        self.assertEqual(ab.decide([200.0, 200.2], [199.0, 199.1])["decision"], "reject")

    def test_identical_legs_do_not_make_any_delta_clear(self):
        # Zero spread would make 10 x spread = 0; the floor keeps a 0.4% delta from "clearing".
        out = ab.decide([200.0, 200.0], [200.8, 200.8], noise_band_pct=0.5)
        self.assertEqual(out["decision"], "continue")

    def test_non_positive_leg_rejects(self):
        self.assertEqual(ab.decide([200.0, 0.0], [210.0, 211.0])["decision"], "reject")

    def test_cli_prints_the_decision_line(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ab.main(["--ref", "191.1", "191.0", "--cand", "202.7", "202.4"])
        self.assertIn("AB_DECISION=accept", buf.getvalue())


class SmallGainRouteTest(unittest.TestCase):
    """Expected gain below the noise band: a solo A/B cannot resolve it (B.11)."""

    # Qwen3.5-397B, CONC=4 / NUM_PROMPTS=40: one config spans 263-268 tok/s (+-1%).
    REF = [265.3, 263.9]

    def test_in_noise_non_regressing_small_gain_goes_to_the_stack(self):
        out = ab.decide(self.REF, [265.9, 264.6], noise_band_pct=1.0, expected_gain_pct=0.66)
        self.assertTrue(out["small_gain_route"])
        self.assertEqual(out["decision"], "stack_pending")

    def test_small_gain_regression_is_still_rejected(self):
        out = ab.decide(self.REF, [260.0, 259.5], noise_band_pct=1.0, expected_gain_pct=0.66)
        self.assertTrue(out["regression"])
        self.assertEqual(out["decision"], "reject")

    def test_slightly_slower_but_inside_the_band_is_not_a_regression(self):
        out = ab.decide(self.REF, [263.5, 264.0], noise_band_pct=1.0, expected_gain_pct=0.96)
        self.assertEqual(out["decision"], "stack_pending")

    def test_a_clear_win_is_accepted_even_when_small_was_expected(self):
        out = ab.decide([191.154, 191.096], [202.704, 202.411], noise_band_pct=1.0,
                        expected_gain_pct=0.5)
        self.assertEqual(out["decision"], "accept")

    def test_one_pair_is_still_not_enough(self):
        out = ab.decide([265.3], [265.9], noise_band_pct=1.0, expected_gain_pct=0.66)
        self.assertEqual(out["decision"], "continue")

    def test_an_expected_gain_above_the_band_uses_the_solo_rule(self):
        out = ab.decide(self.REF + [264.5], [265.9, 264.6, 265.0], noise_band_pct=1.0,
                        expected_gain_pct=6.0)
        self.assertFalse(out["small_gain_route"])
        self.assertEqual(out["decision"], "reject")

    def test_cli_prints_stack_pending(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ab.main(["--ref", "265.3", "263.9", "--cand", "265.9", "264.6",
                     "--noise-band-pct", "1.0", "--expected-gain-pct", "0.66"])
        self.assertIn("AB_DECISION=stack_pending", buf.getvalue())


class StackStepTest(unittest.TestCase):
    def test_clear_no_regression_keeps_early(self):
        out = ab.decide_stack_step([265.0, 265.1], [265.4, 265.3], noise_band_pct=1.0)
        self.assertEqual(out["decision"], "keep")
        self.assertEqual(out["keep_threshold_pct"], -0.5)

    def test_clear_regression_drops_early(self):
        out = ab.decide_stack_step([265.0, 265.1], [260.0, 260.2], noise_band_pct=1.0)
        self.assertEqual(out["decision"], "drop")

    def test_borderline_runs_to_max_pairs_then_uses_the_threshold(self):
        ref, cand = [265.0, 266.0], [264.0, 264.2]
        self.assertEqual(ab.decide_stack_step(ref, cand, 1.0)["decision"], "continue")
        out = ab.decide_stack_step(ref + [265.5], cand + [264.1], 1.0)
        self.assertEqual(out["decision"], "drop" if out["delta_pct"] < -0.5 else "keep")

    def test_cli_stack_step(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ab.main(["--ref", "265", "265.1", "--cand", "265.4", "265.3",
                     "--noise-band-pct", "1.0", "--stack-step"])
        self.assertIn("AB_DECISION=keep", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
