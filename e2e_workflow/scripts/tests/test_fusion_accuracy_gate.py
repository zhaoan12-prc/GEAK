"""Calibrated accuracy gate: an accuracy test that fails base-vs-base may not reject (B.10)."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fusion_accuracy_gate as fag  # noqa: E402


def _paired(n, both, lost, gained):
    """Per-question {idx: ok} for two legs: `both` right on both, `lost` right only on
    the first, `gained` right only on the second, the rest wrong on both."""
    a, b = {}, {}
    for i in range(n):
        if i < both:
            a[i], b[i] = True, True
        elif i < both + lost:
            a[i], b[i] = True, False
        elif i < both + lost + gained:
            a[i], b[i] = False, True
        else:
            a[i], b[i] = False, False
    return a, b


def _probe(token_lists, logprobs=None):
    return {"prompts": [{"id": i, "tokens": t,
                         "token_logprobs": (logprobs[i] if logprobs else None)}
                        for i, t in enumerate(token_lists)]}


LONG = [list("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH") for _ in range(8)]


class CalibrationTest(unittest.TestCase):
    def test_the_qwen35_control_is_not_usable(self):
        # Measured: two identical base boots, n=200 no-thinking: 0.555 vs 0.465, 48% flips.
        a, b = _paired(200, both=54, lost=57, gained=39)
        self.assertAlmostEqual(sum(a.values()) / 200.0, 0.555)
        self.assertAlmostEqual(sum(b.values()) / 200.0, 0.465)
        cal = fag.calibrate(a, b)
        self.assertFalse(cal["gsm8k"]["usable"])
        self.assertEqual(cal["gsm8k"]["flip_rate"], 0.48)
        self.assertEqual(cal["mode"], "unit_equivalence_only")
        self.assertFalse(cal["accuracy_can_reject"])
        self.assertIn("full 1319", cal["recommendation"])

    def test_a_stable_control_is_usable(self):
        a, b = _paired(200, both=170, lost=3, gained=3)
        cal = fag.calibrate(a, b)
        self.assertTrue(cal["gsm8k"]["usable"])
        self.assertEqual(cal["mode"], "gsm8k")

    def test_a_passing_but_flip_heavy_control_is_not_usable(self):
        # Same accuracy, but 30% of the questions change answer run to run.
        a, b = _paired(200, both=70, lost=30, gained=30)
        cal = fag.calibrate(a, b)
        self.assertEqual(cal["gsm8k"]["control_forward"]["verdict"], "pass")
        self.assertFalse(cal["gsm8k"]["usable"])

    def test_deterministic_probe_is_calibrated_too(self):
        cal = fag.calibrate(det_a=_probe(LONG), det_b=_probe(LONG))
        self.assertTrue(cal["deterministic"]["usable"])
        self.assertEqual(cal["mode"], "deterministic")
        noisy_b = [["x"] + t[1:] for t in LONG]
        cal = fag.calibrate(det_a=_probe(LONG), det_b=_probe(noisy_b))
        self.assertFalse(cal["deterministic"]["usable"])


class DecideTest(unittest.TestCase):
    def setUp(self):
        a, b = _paired(200, both=54, lost=57, gained=39)
        self.uncal = fag.calibrate(a, b)
        self.base = a
        # e03 candidate: 0.42, McNemar p=0.004 -- a "fail" by the raw gate.
        self.cand = {i: (i < 84) for i in range(200)}

    def test_an_uncalibrated_gsm8k_drop_cannot_reject_an_equivalent_fusion(self):
        # e02: fp8 ybit 0.9988 vs the vendor kernel.
        out = fag.decide(self.uncal, self.base, self.cand, ybit=0.9988)
        self.assertEqual(out["verdict"], "pass")
        self.assertEqual(out["test_used"], "unit_equivalence")
        self.assertIn("gsm8k_uncalibrated", out["informational"])

    def test_uncalibrated_and_not_equivalent_is_inconclusive_not_fail(self):
        out = fag.decide(self.uncal, self.base, self.cand, rel_err=0.3)
        self.assertEqual(out["verdict"], "inconclusive")
        self.assertIn("next_step", out)

    def test_a_calibrated_gsm8k_may_reject(self):
        a, b = _paired(200, both=170, lost=3, gained=3)
        cal = fag.calibrate(a, b)
        bad = {i: (i < 120) for i in range(200)}
        out = fag.decide(cal, a, bad, ybit=0.9988)
        self.assertEqual(out["test_used"], "gsm8k_calibrated")
        self.assertEqual(out["verdict"], "fail")

    def test_deterministic_probe_decides_when_gsm8k_is_uncalibrated(self):
        cal = fag.calibrate(det_a=_probe(LONG), det_b=_probe(LONG))
        cal["tol"] = 0.01
        late = [t[:40] + ["Z"] + t[41:] for t in LONG]
        out = fag.decide(cal, det_base=_probe(LONG), det_cand=_probe(late))
        self.assertEqual(out["verdict"], "pass")
        self.assertEqual(out["test_used"], "deterministic")
        broken = [["?"] + t[1:] for t in LONG]
        out = fag.decide(cal, det_base=_probe(LONG), det_cand=_probe(broken))
        self.assertEqual(out["verdict"], "fail")

    def test_logprob_error_fails_a_probe_that_matches_tokens(self):
        lp = [[-0.1] * len(t) for t in LONG]
        far = [[-2.0] * len(t) for t in LONG]
        cal = fag.calibrate(det_a=_probe(LONG, lp), det_b=_probe(LONG, lp))
        cmp_ = fag.det_compare(_probe(LONG, lp), _probe(LONG, far))
        verdict, _ = fag.det_verdict(cal["deterministic"]["control"], cmp_)
        self.assertEqual(verdict, "fail")

    def test_cli_round_trip_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            def dump(name, per_q):
                path = os.path.join(tmp, name)
                with open(path, "w") as fh:
                    json.dump({"results": [{"idx": i, "ok": ok} for i, ok in per_q.items()]}, fh)
                return path
            a, b = _paired(200, both=54, lost=57, gained=39)
            pa, pb, pc = dump("a.json", a), dump("b.json", b), dump("c.json", self.cand)
            cal = os.path.join(tmp, "cal.json")
            self.assertEqual(fag.main(["calibrate", "--gsm8k-a", pa, "--gsm8k-b", pb,
                                       "--out", cal]), 0)
            out = os.path.join(tmp, "dec.json")
            self.assertEqual(fag.main(["decide", "--calibration", cal, "--base-gsm8k", pa,
                                       "--cand-gsm8k", pc, "--unit-ybit", "0.9988",
                                       "--out", out]), 0)
            self.assertEqual(fag.main(["decide", "--calibration", cal, "--base-gsm8k", pa,
                                       "--cand-gsm8k", pc, "--out", out]), 3)

    def test_probe_is_sequential_greedy_with_logprobs(self):
        seen = []

        def fake_post(url, body, timeout):
            seen.append(body)
            return {"choices": [{"text": " 72", "logprobs": {
                "tokens": [" 7", "2"], "token_logprobs": [-0.01, -0.02]}}]}
        doc = fag.probe("http://h/v1", "m", ["p1", "p2"], post=fake_post)
        self.assertEqual([b["temperature"] for b in seen], [0.0, 0.0])
        self.assertTrue(all(b["logprobs"] == 1 for b in seen))
        self.assertEqual(doc["prompts"][0]["tokens"], [" 7", "2"])
        self.assertEqual(doc["concurrency"], 1)


UNUSABLE_CAL = {"schema": "fusion_accuracy_calibration/1", "mode": "unit_equivalence_only",
                "accuracy_can_reject": False, "tol": 0.01, "alpha": 0.05,
                "gsm8k": {"usable": False, "measured": True},
                "deterministic": {"usable": False, "measured": True}}


class SkipModeTest(unittest.TestCase):
    def test_skip_with_an_unusable_calibration_passes_without_any_e2e_test(self):
        out = fag.decide(UNUSABLE_CAL, accuracy_mode="skip")
        self.assertEqual(out["verdict"], "pass")
        self.assertEqual(out["test_used"], "skipped")
        self.assertEqual(out["accuracy"], {"mode": "skipped", "reason": "calibration_unusable"})

    def test_skip_fails_only_on_a_unit_parity_failure(self):
        out = fag.decide(UNUSABLE_CAL, accuracy_mode="skip", unit_parity="fail")
        self.assertEqual(out["verdict"], "fail")
        self.assertEqual(out["test_used"], "unit_parity")
        self.assertEqual(fag.decide(UNUSABLE_CAL, accuracy_mode="skip",
                                    unit_parity="pass")["verdict"], "pass")

    def test_skip_is_ignored_when_the_calibration_has_a_usable_test(self):
        cal = dict(UNUSABLE_CAL, mode="gsm8k", accuracy_can_reject=True,
                   gsm8k={"usable": True, "measured": True})
        a, b = _paired(200, 150, 30, 0)
        out = fag.decide(cal, a, b, accuracy_mode="skip")
        self.assertEqual(out["test_used"], "gsm8k_calibrated")
        self.assertEqual(out["verdict"], "fail")
        self.assertIn("ignored", out["informational"]["accuracy_mode"])

    def test_cli_skip_mode_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            cal = os.path.join(tmp, "cal.json")
            with open(cal, "w") as fh:
                json.dump(UNUSABLE_CAL, fh)
            out = os.path.join(tmp, "dec.json")
            code = fag.main(["decide", "--calibration", cal, "--accuracy-mode", "skip",
                             "--unit-parity", "pass", "--out", out])
            self.assertEqual(code, 0)
            self.assertEqual(json.load(open(out))["accuracy"]["mode"], "skipped")


KEY = {"model_path": "/models/Qwen3.5", "backend": "sglang", "tp": 8,
       "flags": "--tp 8 --cuda-graph-max-bs 4 --mem-fraction-static 0.85",
       "env": "SGLANG_USE_AITER=1 HIP_FORCE_DEV_KERNARG=1", "overlay": "/ov/a:/ov/b",
       "workload": {"isl": 8192, "osl": 1024, "conc": 4},
       "harness": {"gsm8k_concurrency": 4, "thinking": False}}
VERS = {"sglang": "0.5.3", "aiter": "0.1.5", "vllm": None, "torch": "2.8.0"}


class CacheKeyTest(unittest.TestCase):
    def test_option_order_and_env_order_do_not_change_the_key(self):
        other = dict(KEY, flags="--mem-fraction-static 0.85 --tp 8 --cuda-graph-max-bs 4",
                     env="HIP_FORCE_DEV_KERNARG=1 SGLANG_USE_AITER=1")
        self.assertEqual(fag.cache_key(KEY, VERS)[1], fag.cache_key(other, VERS)[1])

    def test_values_stay_with_their_option(self):
        k1 = dict(KEY, flags="--a 1 --b 2")
        k2 = dict(KEY, flags="--a 2 --b 1")
        self.assertNotEqual(fag.cache_key(k1, VERS)[1], fag.cache_key(k2, VERS)[1])

    def test_overlay_paths_are_exact(self):
        k2 = dict(KEY, overlay="/other/a:/other/b")   # same basenames, different paths
        self.assertNotEqual(fag.cache_key(KEY, VERS)[1], fag.cache_key(k2, VERS)[1])

    def test_model_tp_workload_and_versions_are_in_the_key(self):
        base = fag.cache_key(KEY, VERS)[1]
        for change in ({"model_path": "/models/other"}, {"tp": 4}, {"backend": "vllm"},
                       {"workload": {"isl": 1024, "osl": 1024, "conc": 4}}):
            self.assertNotEqual(base, fag.cache_key(dict(KEY, **change), VERS)[1], change)
        self.assertNotEqual(base, fag.cache_key(KEY, dict(VERS, sglang="0.5.4"))[1])

    def test_installed_versions_are_read_when_not_given(self):
        key, _ = fag.cache_key(KEY)
        self.assertEqual(set(key["versions"]), set(fag.CACHE_VERSION_PACKAGES))


class CacheStoreLookupTest(unittest.TestCase):
    def _write(self, tmp, name, doc):
        path = os.path.join(tmp, name)
        with open(path, "w") as fh:
            json.dump(doc, fh)
        return path

    def test_miss_then_store_then_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, ".fusion_accuracy_cache")
            self.assertEqual(fag.cache_lookup(cache, KEY, VERS)["hit"], False)
            cal = self._write(tmp, "calibration.json", UNUSABLE_CAL)
            st = fag.cache_store(cache, KEY, cal, VERS)
            self.assertTrue(st["stored"])
            hit = fag.cache_lookup(cache, KEY, VERS)
            self.assertTrue(hit["hit"])
            self.assertEqual(hit["mode"], "unit_equivalence_only")
            self.assertFalse(hit["gsm8k_usable"])
            self.assertFalse(hit["deterministic_usable"])
            self.assertFalse(hit["accuracy_can_reject"])
            self.assertTrue(hit["calibration_json"].startswith(cache))
            self.assertNotEqual(hit["calibration_json"], cal)   # the cache keeps its own copy
            self.assertEqual(json.load(open(hit["calibration_json"])), UNUSABLE_CAL)
            self.assertTrue(hit["stored_at"])
            self.assertEqual(hit["source_calibration_json"], os.path.abspath(cal))
            self.assertFalse(fag.cache_lookup(cache, dict(KEY, tp=4), VERS)["hit"])

    def test_a_calibration_that_measured_nothing_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "c")
            cal = self._write(tmp, "cal.json", {"mode": "unit_equivalence_only",
                                                "gsm8k": {"measured": False},
                                                "deterministic": {"measured": False}})
            self.assertFalse(fag.cache_store(cache, KEY, cal, VERS)["stored"])
            self.assertFalse(fag.cache_lookup(cache, KEY, VERS)["hit"])

    def test_an_edited_entry_key_is_a_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "c")
            cal = self._write(tmp, "cal.json", UNUSABLE_CAL)
            sha = fag.cache_store(cache, KEY, cal, VERS)["key_sha"]
            with open(os.path.join(cache, sha, "key.json"), "w") as fh:
                json.dump({"key": {"model_path": "x"}}, fh)
            self.assertFalse(fag.cache_lookup(cache, KEY, VERS)["hit"])

    def test_cli_lookup_and_store_print_one_json_line(self):
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            cache = os.path.join(tmp, "c")
            key = self._write(tmp, "key.json", KEY)
            cal = self._write(tmp, "cal.json", UNUSABLE_CAL)
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(fag.main(["cache-lookup", "--cache-dir", cache,
                                           "--key-json", key]), 0)
            self.assertEqual(json.loads(buf.getvalue().strip().splitlines()[-1])["hit"], False)
            buf = io.StringIO()
            with redirect_stdout(buf):
                fag.main(["cache-store", "--cache-dir", cache, "--key-json", key,
                          "--calibration-json", cal])
            self.assertTrue(json.loads(buf.getvalue().strip().splitlines()[-1])["stored"])
            buf = io.StringIO()
            with redirect_stdout(buf):
                fag.main(["cache-lookup", "--cache-dir", cache, "--key-json", key])
            out = json.loads(buf.getvalue().strip().splitlines()[-1])
            self.assertTrue(out["hit"])
            for field in ("mode", "gsm8k_usable", "deterministic_usable",
                          "accuracy_can_reject", "calibration_json"):
                self.assertIn(field, out)


if __name__ == "__main__":
    unittest.main()
