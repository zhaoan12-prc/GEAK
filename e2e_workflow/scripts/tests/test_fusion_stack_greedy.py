"""Greedy stacking state machine (GEAK_structure 二.10.4)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import fusion_stack_greedy as g  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(HERE), "fusion_stack_greedy.py")


def _cands():
    return [
        {"id": "e05", "source": "stack_pending", "expected_gain_pct": 0.4,
         "overlay_path": "/ov/e05", "row": {"exec_id": "e05", "throughput_delta_pct": 0.2}},
        {"id": "e02", "source": "stack_pending", "expected_gain_pct": 0.9,
         "overlay_path": "/ov/e02", "row": {"exec_id": "e02", "throughput_delta_pct": 0.3}},
        {"id": "p01", "source": "candidate_pool", "expected_gain_pct": 0.2, "unit_status": "pass"},
        {"id": "p02", "source": "candidate_pool", "expected_gain_pct": 5.0, "unit_status": "fail"},
        {"id": "p03", "source": "candidate_pool", "expected_gain_pct": None, "unit_status": "pass"},
    ]


class GreedyTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.state = os.path.join(self.dir, "stack", "greedy_state.json")

    def _run(self, *args, payload=None):
        argv = list(args)
        if payload is not None:
            path = os.path.join(self.dir, "in_%d.json" % len(os.listdir(self.dir)))
            with open(path, "w") as fh:
                json.dump(payload, fh)
            argv += ["--input", path]
        out = subprocess.run([sys.executable, SCRIPT] + argv + ["--state", self.state],
                             capture_output=True, text=True)
        return out.returncode, json.loads(out.stdout.strip().splitlines()[-1])

    def _init(self, noise=1.0, retries=1):
        return self._run("init", payload={
            "candidates": _cands(), "noise_band_pct": noise, "max_retries": retries,
            "entry": {"overlay": "", "flags": "--x", "env": "", "tput": 265},
            "base": {"overlay": "/ov/acc", "flags": "--x", "env": "", "tput": 270,
                     "members": ["e03"]}})

    def test_order_is_expected_gain_desc_and_pool_needs_unit_pass(self):
        code, out = self._init()
        self.assertEqual(code, 0)
        self.assertEqual(out["queue"], ["e02", "e05", "p01", "p03"])
        self.assertEqual(out["keep_threshold_pct"], -0.5)

    def test_init_is_idempotent(self):
        self._init()
        code, out = self._run("init", payload={"candidates": [], "noise_band_pct": 1.0})
        self.assertTrue(out["resumed"])
        self.assertEqual(out["queue"], ["e02", "e05", "p01", "p03"])

    def test_keep_drop_retry_then_confirm_and_land(self):
        self._init()
        _, nxt = self._run("next")
        self.assertEqual((nxt["action"], nxt["candidate"]["id"]), ("step", "e02"))
        self.assertEqual(nxt["stack"]["overlay"], "/ov/acc")
        # e02 keeps (small positive), stack moves.
        _, rec = self._run("record", payload={"id": "e02", "measured": True, "delta_pct": 0.3,
                                              "accuracy_pass": True, "stack_overlay": "/ov/s1",
                                              "cand_tok_s": 271})
        self.assertEqual(rec["decision"], "keep")
        # e05: unmeasured once -> retry, same step returned again.
        _, rec = self._run("record", payload={"id": "e05", "measured": False,
                                              "reason_code": "gpu_busy"})
        self.assertEqual(rec["decision"], "retry")
        _, nxt = self._run("next")
        self.assertEqual(nxt["candidate"]["id"], "e05")
        self.assertEqual(nxt["stack"]["overlay"], "/ov/s1")
        self.assertEqual(nxt["attempt"], 2)
        # e05 regresses beyond half the band -> drop (stack_harmful).
        _, rec = self._run("record", payload={"id": "e05", "measured": True, "delta_pct": -0.6,
                                              "accuracy_pass": True, "stack_overlay": "/ov/s2"})
        self.assertEqual(rec["decision"], "drop")
        # p01: within -band/2 -> keep; p03: accuracy fail -> drop.
        self._run("record", payload={"id": "p01", "measured": True, "delta_pct": -0.4,
                                     "accuracy_pass": True, "stack_overlay": "/ov/s3"})
        self._run("record", payload={"id": "p03", "measured": True, "delta_pct": 0.8,
                                     "accuracy_pass": False, "stack_overlay": "/ov/s4"})
        _, nxt = self._run("next")
        self.assertEqual(nxt["action"], "confirm")
        self.assertEqual(nxt["kept"], ["e02", "p01"])
        self.assertEqual(nxt["stack"]["overlay"], "/ov/s3")
        _, rec = self._run("record", "--confirm", payload={
            "measured": True, "ab_decision": "accept", "delta_pct": 2.4, "accuracy_pass": True,
            "final_overlay": "/ov/s3"})
        self.assertEqual(rec["decision"], "pass")
        _, nxt = self._run("next")
        self.assertEqual(nxt["action"], "done")
        _, summ = self._run("summary")
        self.assertTrue(summ["landed"])
        self.assertEqual(summ["kept"], ["e02", "p01"])
        self.assertEqual(summ["harmful"], ["e05", "p03"])
        self.assertEqual([r["exec_id"] for r in summ["accepted"]], ["e02"])
        self.assertEqual(summ["accepted"][0]["throughput_delta_pct"], 0.3)
        self.assertEqual(summ["accepted"][0]["single_delta_pct"], 0.3)
        harm = summ["rejected"][0]
        self.assertEqual((harm["exec_id"], harm["blocked_evidence"]), ("e05", "stack_harmful"))
        self.assertTrue(harm["stack_result"])
        pool = {p["id"]: p["status"] for p in summ["pool"]}
        self.assertEqual(pool, {"p01": "landed", "p03": "stack_harmful"})
        self.assertEqual(summ["final_stack"]["overlay"], "/ov/s3")

    def test_a_skipped_accuracy_step_counts_as_accuracy_pass(self):
        self._init()
        _, rec = self._run("record", payload={
            "id": "e02", "measured": True, "delta_pct": 0.3, "stack_overlay": "/ov/s1",
            "accuracy": {"mode": "skipped", "reason": "calibration_unusable"}})
        self.assertEqual(rec["decision"], "keep")
        # Without either field the step is still unmeasured.
        _, rec = self._run("record", payload={"id": "e05", "measured": True, "delta_pct": 0.3,
                                              "stack_overlay": "/ov/s2"})
        self.assertEqual(rec["decision"], "retry")

    def test_failed_confirmation_lands_nothing_new(self):
        self._init()
        self._run("record", payload={"id": "e02", "measured": True, "delta_pct": 0.1,
                                     "accuracy_pass": True, "stack_overlay": "/ov/s1"})
        for cid in ("e05", "p01", "p03"):
            self._run("record", payload={"id": cid, "measured": True, "delta_pct": -2,
                                         "accuracy_pass": True, "stack_overlay": "/x"})
        self._run("next")
        self._run("record", "--confirm", payload={"measured": True, "ab_decision": "reject",
                                                  "delta_pct": 0.3, "accuracy_pass": True})
        _, summ = self._run("summary")
        self.assertFalse(summ["landed"])
        self.assertEqual(summ["accepted"], [])
        self.assertEqual(summ["final_stack"]["overlay"], "/ov/acc")
        e02 = next(r for r in summ["rejected"] if r["exec_id"] == "e02")
        self.assertEqual(e02["blocked_evidence"], "stack_reject")

    def test_nothing_kept_skips_confirmation(self):
        self._init()
        for cid in ("e02", "e05", "p01", "p03"):
            self._run("record", payload={"id": cid, "measured": True, "delta_pct": -3,
                                         "accuracy_pass": True})
        _, nxt = self._run("next")
        self.assertEqual(nxt["action"], "done")
        _, summ = self._run("summary")
        self.assertEqual(summ["confirm"]["status"], "skipped")

    def test_budget_stop_skips_the_rest_and_still_confirms(self):
        self._init()
        self._run("record", payload={"id": "e02", "measured": True, "delta_pct": 0.5,
                                     "accuracy_pass": True, "stack_overlay": "/ov/s1"})
        _, nxt = self._run("next", "--budget-left-s", "3000", "--step-cost-s", "2700")
        self.assertEqual(nxt["action"], "confirm")
        _, summ = self._run("summary")
        self.assertEqual(summ["skipped_budget"], ["e05", "p01", "p03"])
        self.assertEqual(summ["stopped"], "budget")
        # skipped rows keep stack_pending without stack_result (never re-gated).
        self.assertFalse(any(r["exec_id"] == "e05" for r in summ["rejected"]))

    def test_unmeasured_after_retries_moves_on(self):
        self._init(retries=1)
        self._run("record", payload={"id": "e02", "measured": False})
        _, rec = self._run("record", payload={"id": "e02", "measured": False})
        self.assertEqual(rec["decision"], "unmeasured")
        _, nxt = self._run("next")
        self.assertEqual(nxt["candidate"]["id"], "e05")

    def test_record_for_the_wrong_step_is_refused(self):
        self._init()
        code, out = self._run("record", payload={"id": "e05", "measured": True, "delta_pct": 1,
                                                 "accuracy_pass": True, "stack_overlay": "/o"})
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])

    def test_keep_without_overlay_is_not_a_measurement(self):
        self._init()
        _, rec = self._run("record", payload={"id": "e02", "measured": True, "delta_pct": 1,
                                              "accuracy_pass": True})
        self.assertEqual(rec["decision"], "retry")

    def test_build_queue_prefers_stack_pending_on_duplicate(self):
        q = g.build_queue([{"id": "x", "source": "candidate_pool", "unit_status": "pass",
                            "expected_gain_pct": 1},
                           {"id": "x", "source": "stack_pending", "expected_gain_pct": 1}])
        self.assertEqual([i["source"] for i in q], ["stack_pending"])


if __name__ == "__main__":
    unittest.main()
