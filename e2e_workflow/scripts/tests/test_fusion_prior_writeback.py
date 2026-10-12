"""Measured-prior write-back at the end of KernelFusion (GEAK_structure 二.12)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import fusion_prior_writeback as wb  # noqa: E402

SCRIPT = os.path.join(os.path.dirname(HERE), "fusion_prior_writeback.py")

TOPK = {
    "execution_list": [
        {"exec_id": "e01", "kind": "serial", "seam_key": "serial:attn.qk_norm_rope",
         "kernel_names": ["rms_norm", "rope"], "title": "qk norm + rope",
         "seam_descriptor": {"kinds": ["serial"], "max_span": 4,
                             "members": [{"cls": "norm", "module": "self_attn.q_norm"},
                                         {"cls": "rope"}]}},
        {"exec_id": "e02", "kind": "epilogue_forward", "seam_key": "ef:ar_quant",
         "kernel_names": ["cross_device_reduce", "per_group_quant"],
         "alt_ladder": [{"rung": 0, "seam_key": "ef:ar_quant"},
                        {"rung": 1, "seam_key": "ef:quant_into_get_quant", "kind": "serial",
                         "kernel_names": ["per_group_quant"]}]},
        {"exec_id": "e03", "kind": "serial", "seam_key": "serial:act_quant",
         "kernel_names": ["silu_and_mul", "per_group_quant"],
         "matched_prior_id": "xb-act-quant", "seam_members": [{"cls": "activation"},
                                                              {"cls": "quant"}]},
        {"exec_id": "e04", "kind": "serial", "seam_key": "serial:gated_norm_quant"},
        {"exec_id": "e05", "kind": "sibling", "seam_key": "sibling:in_proj"},
        {"exec_id": "e06", "kind": "serial", "seam_key": "serial:untested"},
    ],
    "candidate_pool": [{"exec_id": "p01", "kind": "serial", "seam_key": "serial:pool_one"}],
}

APPLY = {
    "accepted_fusions": [
        {"exec_id": "e01", "fusion": "qk-norm-rope", "throughput_delta_pct": 1.4,
         "overlay_path": "/ov/e01"},
        {"exec_id": "e03", "stacked": True, "throughput_delta_pct": 0.3,
         "stack_result": {"delta_pct": 0.3, "evidence": "/stack/step1.json"}},
    ],
    "rejected": [
        {"exec_id": "e04", "measured": True, "blocked_evidence": "unit_fail",
         "reason": "0.8x", "implementation": "self_authored"},
        {"exec_id": "e05", "measured": True, "blocked_evidence": "stack_harmful",
         "stack_pending": True, "stack_harmful": True,
         "stack_result": {"delta_pct": -0.9, "evidence": "/stack/step2.json"}},
        {"exec_id": "e02", "rung": 1, "measured": True, "blocked_evidence": "e2e_no_gain",
         "throughput_delta_pct": 0.1, "reason": "in noise"},
    ],
    "rung_attempts": [
        {"exec_id": "e02", "rung": 0, "measured": True, "blocked_evidence": "e2e_regression",
         "throughput_delta_pct": -2.1, "reason": "-2.1%", "evidence": "/ab/e02_r0.json"},
    ],
    "deferred": [{"exec_id": "e06", "deferred_budget": True, "reason_code": "budget_reserve",
                  "reason": "stacking reserve", "overlay_path": "/ov/e06"}],
    "pool_results": [{"id": "p01", "exec_id": "p01", "status": "landed",
                      "stack_result": {"delta_pct": 0.2}}],
}


class WritebackTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.apply = os.path.join(self.dir, "apply.json")
        self.topk = os.path.join(self.dir, "topk.json")
        self.priors = os.path.join(self.dir, "measured_priors.json")
        for path, doc in ((self.apply, APPLY), (self.topk, TOPK)):
            with open(path, "w") as fh:
                json.dump(doc, fh)

    def _argv(self, *extra):
        return ["--apply", self.apply, "--topk", self.topk, "--priors", self.priors,
                "--model", "Qwen3.5-397B-A17B-FP8", "--gfx", "gfx942", "--framework", "sglang",
                "--protocol", "ab_interleaved:isl=8192,osl=1024,conc=4", "--date", "2026-10-11",
                "--noise-band-pct", "1.0"] + list(extra)

    def _plan(self):
        code, out = wb.run(wb.main.__globals__["argparse"].Namespace(
            apply=self.apply, topk=self.topk, priors=self.priors, sidecar="",
            model="Qwen3.5-397B-A17B-FP8", model_family="", gfx="gfx942", framework="sglang",
            protocol="p", source="s", date="2026-10-11", noise_band_pct="1.0", dry_run=True))
        self.assertEqual(code, 0)
        return out

    def test_classification(self):
        out = self._plan()
        planned = out["planned"]
        pol = {(p["seam_record"]["seam"]["seam_key"], p["polarity"]): p["measurement"]["gain_pct"]
               for p in planned["priors"]}
        self.assertEqual(pol, {
            ("serial:attn.qk_norm_rope", "positive"): 1.4,
            ("serial:act_quant", "positive"): 0.3,
            ("sibling:in_proj", "negative"): -0.9,
            ("ef:ar_quant", "negative"): -2.1,       # rung 0 measured regression
            ("serial:pool_one", "positive"): 0.2,
        })
        # unit_fail is recorded against the implementation, never as a negative seam prior.
        self.assertEqual([r["seam_key"] for r in planned["implementation_failures"]],
                         ["serial:gated_norm_quant"])
        self.assertEqual(planned["implementation_failures"][0]["implementation"], "self_authored")
        self.assertFalse(any(p["seam_record"]["seam"]["seam_key"] == "serial:gated_norm_quant"
                             for p in planned["priors"]))
        # an in-noise rung-1 reject is no seam evidence either way.
        self.assertFalse(any(p["seam_record"]["seam"]["seam_key"] == "ef:quant_into_get_quant"
                             for p in planned["priors"]))
        self.assertEqual([u["seam_key"] for u in planned["untested_candidates"]],
                         ["serial:untested"])
        m = planned["priors"][0]["measurement"]
        self.assertEqual(set(m), {"gain_pct", "date", "protocol", "source", "evidence"})
        self.assertEqual(planned["priors"][0]["seam_record"]["model_family"], "qwen35")

    def test_dry_run_writes_nothing(self):
        out = subprocess.run([sys.executable, SCRIPT] + self._argv("--dry-run"),
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        doc = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertTrue(doc["dry_run"])
        self.assertEqual(doc["counts"]["positive"], 3)
        self.assertEqual(doc["counts"]["negative"], 2)
        self.assertFalse(os.path.exists(self.priors))

    def test_fallback_appends_schema_v2_and_is_append_only(self):
        with open(self.priors, "w") as fh:
            json.dump({"schema_version": 2, "priors": [
                {"prior_id": "keep-me", "title": "t", "polarity": "positive",
                 "seam": {"seam_key": "serial:attn.qk_norm_rope"}, "model_family": "qwen35",
                 "gfx": "gfx942", "measurements": [{"gain_pct": 0.9, "date": "2026-09-01"}]}]}, fh)
        with mock.patch.object(wb, "_append_prior", wraps=wb._append_prior):
            with mock.patch.dict(sys.modules, {"fusion_priors": type(sys)("fusion_priors")}):
                for _ in range(2):
                    code, _out = wb.run(wb.main.__globals__["argparse"].Namespace(
                        apply=self.apply, topk=self.topk, priors=self.priors, sidecar="",
                        model="Qwen3.5-397B-A17B-FP8", model_family="", gfx="gfx942",
                        framework="sglang", protocol="p", source="s", date="2026-10-11",
                        noise_band_pct="1.0", dry_run=False))
                    self.assertEqual(code, 0)
        with open(self.priors) as fh:
            doc = json.load(fh)
        keep = doc["priors"][0]
        self.assertEqual(keep["prior_id"], "keep-me")
        # matched by seam_key: the new run APPENDS, the old measurement stays; re-running the
        # same write-back does not duplicate it.
        self.assertEqual([m["gain_pct"] for m in keep["measurements"]], [0.9, 1.4])
        neg = next(p for p in doc["priors"] if p["seam"]["seam_key"] == "sibling:in_proj")
        self.assertEqual(neg["polarity"], "negative")
        self.assertEqual(len(neg["measurements"]), 1)
        for prior in doc["priors"]:
            for key in ("prior_id", "title", "polarity", "seam", "measurements"):
                self.assertIn(key, prior)
        with open(os.path.join(self.dir, "measured_priors_sidecar.json")) as fh:
            side = json.load(fh)
        self.assertEqual(len(side["implementation_failures"]), 2)
        self.assertEqual(len(side["untested_candidates"]), 2)

    def test_wire_failure_is_the_next_rounds_highest_priority(self):
        doc = dict(APPLY, rejected=APPLY["rejected"] + [
            {"exec_id": "e06", "measured": True, "blocked_evidence": "wire_failure",
             "wire_cause": "not_captured_by_cuda_graph", "reason": "eager only"}], deferred=[])
        with open(self.apply, "w") as fh:
            json.dump(doc, fh)
        out = self._plan()
        un = out["planned"]["untested_candidates"]
        self.assertEqual(un[0]["seam_key"], "serial:untested")
        self.assertEqual(un[0]["priority"], "highest")
        self.assertEqual(out["counts"]["highest_priority_next_round"], 1)
        self.assertFalse(any(p["seam_record"]["seam"]["seam_key"] == "serial:untested"
                             for p in out["planned"]["priors"]))

    def test_a_known_seam_appends_to_its_matched_prior(self):
        out = self._plan()
        by_key = {p["seam_record"]["seam"]["seam_key"]: p["seam_record"] for p in out["planned"]["priors"]}
        self.assertEqual(by_key["serial:act_quant"]["prior_id"], "xb-act-quant")
        self.assertTrue(by_key["serial:act_quant"]["known_prior"])
        qk = by_key["serial:attn.qk_norm_rope"]
        self.assertTrue(qk["prior_id"].startswith("auto-qwen35-gfx942-"))
        self.assertEqual(qk["seam"]["members"][0], {"cls": "norm", "module": "self_attn.q_norm"})
        self.assertEqual(qk["seam"]["max_span"], 4)
        # rung 1 of e02 is its own window, not the head entry's seam
        self.assertIn("ef:ar_quant", by_key)

    def test_protocol_is_an_object(self):
        self.assertEqual(wb.parse_protocol("ab_interleaved:isl=8192,osl=1024,tp=8"),
                         {"name": "ab_interleaved", "isl": 8192, "osl": 1024, "tp": 8})
        self.assertEqual(wb.parse_protocol(""), {})

    def test_real_fusion_priors_api_round_trip(self):
        """Against the real fusion_priors.append_measurement (skipped if not there yet)."""
        import fusion_priors
        if not hasattr(fusion_priors, "append_measurement"):
            self.skipTest("fusion_priors.append_measurement not available")
        code, out = wb.run(wb.main.__globals__["argparse"].Namespace(
            apply=self.apply, topk=self.topk, priors=self.priors, sidecar="",
            model="Qwen3.5-397B-A17B-FP8", model_family="", gfx="gfx942", framework="sglang",
            protocol="ab_interleaved:isl=8192,osl=1024,conc=4", source="/eval",
            date="2026-10-11", noise_band_pct="1.0", dry_run=False))
        self.assertEqual(code, 0, out)
        self.assertEqual(out["api"], ["fusion_priors"])
        priors = fusion_priors.load_measured_priors(self.priors)
        self.assertTrue(priors or os.path.exists(self.priors))

    def test_uses_the_fusion_priors_api_when_present(self):
        calls = []
        fake = type(sys)("fusion_priors")
        fake.append_measurement = lambda path, rec, meas, pol: calls.append((path, rec["prior_id"], pol))
        fake.model_family = lambda text: "qwen35"
        with mock.patch.dict(sys.modules, {"fusion_priors": fake}):
            code, out = wb.run(wb.main.__globals__["argparse"].Namespace(
                apply=self.apply, topk=self.topk, priors=self.priors, sidecar="",
                model="Qwen3.5", model_family="", gfx="gfx942", framework="sglang",
                protocol="p", source="s", date="2026-10-11", noise_band_pct="1.0",
                dry_run=False))
        self.assertEqual(code, 0)
        self.assertEqual(out["api"], ["fusion_priors"])
        self.assertEqual(len(calls), 5)
        self.assertFalse(os.path.exists(self.priors))


if __name__ == "__main__":
    unittest.main()
