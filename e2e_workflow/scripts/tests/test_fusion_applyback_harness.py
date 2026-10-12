import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_applyback_harness as ab


class FusionApplyBackTest(unittest.TestCase):
    def _write(self, root, name, value):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            json.dump(value, fh)
        return path

    def _topk(self, **over):
        board = {
            "schema_version": 3,
            "candidate_total": 5, "candidates_on_board": 4, "truncated_count": 0,
            "phase_coverage": {"shape_resolution_by_phase": {
                "decode": {"rows": 10, "resolved": 9}}},
            "region_coverage": {"regions_total": 3, "covered": 3, "deferred": 0,
                                "uncovered": 0},
            "execution_list": [
                {"exec_id": "e01", "rank": 1, "phase": "decode", "tier": "B",
                 "action": "Decode 接入 AR+Norm+Quant 融合", "handle": "fused_ar_rmsnorm_quant",
                 "candidate_ids": ["dc_arnq"], "forward_us": 1188, "forward_pct": 5.49},
                {"exec_id": "e02", "rank": 2, "phase": "decode", "tier": "B",
                 "action": "Decode 接入 AR+Norm 融合", "handle": "fused_ar_rmsnorm",
                 "candidate_ids": ["dc_ar"], "forward_us": 900, "forward_pct": 4.1},
                {"exec_id": "e03", "rank": 3, "phase": "decode", "tier": "B",
                 "action": "Decode 接入 norm quant 融合", "handle": "add_rmsnorm_quant",
                 "candidate_ids": ["dc_nq"], "forward_us": 579, "forward_pct": 2.67},
                {"exec_id": "e04", "rank": 4, "phase": "prefill", "tier": "B",
                 "action": "Prefill 接入 kv layout 融合", "handle": "concat_and_cast",
                 "candidate_ids": ["pf_kv"], "forward_us": 477, "forward_pct": 0.07},
            ],
            "exclusive_groups": [
                {"group_id": "x01", "phase": "decode", "choose": 1,
                 "conflict_edges": [["e01", "e02"]],
                 "members": [{"exec_id": "e01"}, {"exec_id": "e02"}]}],
        }
        board.update(over)
        return board

    def _apply(self, **over):
        payload = {"accepted_fusions": [
            {"exec_id": "e01", "fusion": "fused_ar_rmsnorm_quant", "rung": "maximal",
             "overlay_path": "/ov/e01", "tpot_delta_pct": -3.2,
             "throughput_delta_pct": 2.9, "engaged": True}],
            "final_overlay": "/ov/stacked", "e2e_throughput_tok_s": 1234.5}
        payload.update(over)
        return payload

    def _run(self, tmp, topk=None, apply=None, **kw):
        tp = self._write(tmp, "topk.json", topk or self._topk())
        ap = self._write(tmp, "apply.json", apply if apply is not None else self._apply())
        return ab.run(tp, ap, os.path.join(tmp, "out.md"),
                      os.path.join(tmp, "out.json"), **kw)

    # ---- the core recall gate ---------------------------------------------
    def test_unmentioned_entries_fail_the_gate(self):
        """e03/e04 were never mentioned: not applied, not blocked, not deferred."""
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp)
            self.assertEqual(res["status"], "fail")
            self.assertEqual(res["coverage"]["unaccounted"], 2)
            self.assertEqual(sorted(res["coverage"]["unaccounted_exec_ids"]),
                             ["e03", "e04"])
            md = open(os.path.join(tmp, "out.md")).read()
            self.assertIn("无交代", md)
            self.assertIn("`e03`", md)

    def test_every_entry_accounted_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, apply=self._apply(
                rejected=[{"exec_id": "e03", "reason": "aiter kernel not built on gfx942"}],
                deferred=[{"exec_id": "e04", "reason": "prefill 收益 0.07%，下一轮再做"}]))
            self.assertEqual(res["status"], "pass")
            self.assertEqual(res["coverage"]["unaccounted"], 0)
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e01"], "applied")
            self.assertEqual(disp["e02"], "blocked_by_exclusion")
            self.assertEqual(disp["e03"], "blocked")
            self.assertEqual(disp["e04"], "deferred_with_reason")

    def test_skip_mode_reports_unguarded_accuracy_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            apply = self._apply(
                rejected=[{"exec_id": "e03", "measured": True, "blocked_evidence": "e2e_no_gain",
                           "reason": "flat"}],
                deferred=[{"exec_id": "e04", "reason": "prefill 收益 0.07%"}])
            apply["accepted_fusions"][0]["accuracy"] = {
                "mode": "skipped", "reason": "calibration_unusable",
                "calibration_json": "/eval/fusion/accuracy_calibration/calibration.json"}
            res = self._run(tmp, apply=apply, accuracy_mode="skip",
                            calibration_json="/eval/fusion/accuracy_calibration/calibration.json")
            self.assertEqual(res["status"], "pass")
            self.assertEqual(res["accuracy_gate"]["mode"], "skip")
            self.assertFalse(res["accuracy_gate"]["e2e_accuracy_gated"])
            self.assertEqual(res["accuracy_gate"]["statement"], "e2e 精度未把关")
            md = open(os.path.join(tmp, "out.md")).read()
            self.assertIn("e2e 精度未把关", md)
            self.assertIn("/eval/fusion/accuracy_calibration/calibration.json", md)

    def test_gate_mode_report_does_not_claim_unguarded_accuracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp)
            self.assertTrue(res["accuracy_gate"]["e2e_accuracy_gated"])
            self.assertNotIn("e2e 精度未把关", open(os.path.join(tmp, "out.md")).read())

    def test_on_demand_rung_records_are_reported_and_need_no_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            apply = self._apply(
                rejected=[{"exec_id": "e03", "measured": True, "rung": 1,
                           "candidate_ids": ["e03.r1"], "blocked_evidence": "e2e_no_gain",
                           "reason": "flat"}],
                deferred=[{"exec_id": "e04", "reason": "prefill 收益 0.07%"}],
                rung_attempts=[{"exec_id": "e03", "rung": 0, "blocked_evidence": "unit_fail",
                                "reason": "0.9x"}],
                unit_on_demand=[{"exec_id": "e03", "candidate_id": "e03.r1", "rung": 1,
                                 "status": "pass"}])
            res = self._run(tmp, apply=apply)
            self.assertEqual(res["status"], "pass", res["errors"])
            self.assertEqual(res["unit_on_demand"][0]["candidate_id"], "e03.r1")
            self.assertIn("`e03.r1` r1 → pass", open(os.path.join(tmp, "out.md")).read())

    def test_a_blocked_row_without_a_reason_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, apply=self._apply(
                rejected=[{"exec_id": "e03"}, {"exec_id": "e04", "reason": "ok"}]))
            self.assertEqual(res["status"], "fail")
            self.assertTrue(any("no reason" in e for e in res["errors"]),
                            res["errors"])

    # ---- exclusion is derived, and only from a REAL conflict ---------------
    def test_exclusion_only_blocks_a_genuine_conflict(self):
        """e02 conflicts with e01 and is auto-blocked; e03 does not conflict with
        anything and stays a hole even though it is in the same phase."""
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, apply=self._apply(
                deferred=[{"exec_id": "e04", "reason": "next round"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e02"], "blocked_by_exclusion")
            self.assertEqual(disp["e03"], "unaccounted")

    def test_a_non_clique_group_does_not_auto_block_a_compatible_row(self):
        """e01-e02 conflict, e01-e03 do not. Applying e01 must not silently
        excuse e03 just because they share a group."""
        board = self._topk()
        board["exclusive_groups"] = [{
            "group_id": "x01", "phase": "decode", "choose": "compatible_subset",
            "conflict_edges": [["e01", "e02"]],
            "members": [{"exec_id": "e01", "conflicts_with": ["e02"]},
                        {"exec_id": "e02", "conflicts_with": ["e01"]},
                        {"exec_id": "e03", "conflicts_with": []}]}]
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, topk=board, apply=self._apply(
                deferred=[{"exec_id": "e04", "reason": "next round"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e02"], "blocked_by_exclusion")
            self.assertEqual(disp["e03"], "unaccounted")
            self.assertEqual(res["status"], "fail")

    # ---- budget --------------------------------------------------------------
    def test_budget_excuses_only_the_tail_and_names_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, budget=2)
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e03"], "deferred_budget")
            self.assertEqual(disp["e04"], "deferred_budget")
            self.assertEqual(res["status"], "pass")
            self.assertEqual(res["budget"]["beyond_budget"], 2)
            md = open(os.path.join(tmp, "out.md")).read()
            self.assertIn("超预算未尝试", md)
            self.assertIn("`e03`", md)

    def test_budget_does_not_excuse_a_hole_inside_the_budget(self):
        """A rank-2 row skipped while rank-4 was applied is not a budget effect."""
        board = self._topk()
        board["exclusive_groups"] = []
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, topk=board, budget=3, apply=self._apply(
                accepted_fusions=[{"exec_id": "e04", "fusion": "concat_and_cast"}],
                deferred=[{"exec_id": "e01", "reason": "wire failure, next round"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e02"], "unaccounted")
            self.assertEqual(disp["e03"], "unaccounted")
            self.assertEqual(disp["e04"], "applied")
            self.assertEqual(res["status"], "fail")

    # ---- 单侧 handoff --------------------------------------------------------
    def test_a_unitside_failure_stands_as_a_disposition(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit = self._write(tmp, "unit.json", {"results": [
                {"candidate_id": "dc_nq", "unit_side_status": "fail",
                 "reason": "isolated_speedup 0.980 <= 1.000"},
                {"candidate_id": "pf_kv", "unit_side_status": "blocked",
                 "reason": "size-guard fallback"}]})
            res = self._run(tmp, unitside_path=unit)
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e03"], "blocked")
            self.assertEqual(disp["e04"], "blocked")
            self.assertEqual(res["results"][2]["source"], "unitside")
            self.assertEqual(res["status"], "pass")

    def test_a_unitside_pass_that_nobody_applied_is_still_a_hole(self):
        """Passing 单侧 and then vanishing is the exact DSR1 failure."""
        with tempfile.TemporaryDirectory() as tmp:
            unit = self._write(tmp, "unit.json", {"results": [
                {"candidate_id": "dc_nq", "unit_side_status": "pass",
                 "reason": "1.9x"},
                {"candidate_id": "pf_kv", "unit_side_status": "pass",
                 "reason": "1.2x"}]})
            res = self._run(tmp, unitside_path=unit)
            self.assertEqual(res["coverage"]["unaccounted"], 2)
            self.assertEqual(res["status"], "fail")

    def test_in_budget_unitside_pass_cannot_be_deferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit = self._write(tmp, "unit.json", {"results": [
                {"candidate_id": "dc_nq", "unit_side_status": "pass",
                 "reason": "1.9x"},
                {"candidate_id": "pf_kv", "unit_side_status": "equivalent_pass",
                 "reason": "representative passed"}]})
            res = self._run(
                tmp, unitside_path=unit, budget=4,
                apply=self._apply(deferred=[
                    {"exec_id": "e03", "reason": "next round"},
                    {"exec_id": "e04", "reason": "server launch is expensive"},
                ]))

            self.assertEqual(res["status"], "fail")
            self.assertTrue(any("e03 is an in-budget" in e for e in res["errors"]),
                            res["errors"])
            self.assertTrue(any("e04 is an in-budget" in e for e in res["errors"]),
                            res["errors"])
            required = {r["exec_id"]: r["applyback_required"]
                        for r in res["results"]}
            self.assertTrue(required["e03"])
            self.assertTrue(required["e04"])
            self.assertEqual(
                res["applyback_completion"]["incomplete_exec_ids"],
                ["e03", "e04"])
            self.assertFalse(res["applyback_completion"]["complete"])
            with open(os.path.join(tmp, "out.md")) as fh:
                md = fh.read()
            self.assertIn("预算内单侧通过项不得延期", md)

    def test_out_of_budget_unitside_pass_may_be_deferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit = self._write(tmp, "unit.json", {"results": [
                {"candidate_id": cid, "unit_side_status": "pass", "reason": "1.2x"}
                for cid in ("dc_arnq", "dc_ar", "dc_nq", "pf_kv")]})
            res = self._run(
                tmp, unitside_path=unit, budget=3,
                apply=self._apply(
                    rejected=[{"exec_id": "e03", "reason": "A/B inside noise"}],
                    deferred=[{"exec_id": "e04", "reason": "rank beyond apply-back budget"}]))

            self.assertEqual(res["status"], "pass")
            row = next(r for r in res["results"] if r["exec_id"] == "e04")
            self.assertFalse(row["applyback_required"])

    def test_unitside_blocked_rows_do_not_use_the_budget(self):
        """1001 run: e01/e04 failed 单侧 yet took two of six slots, so e07 (passed) was
        deferred_budget. The budget counts 单侧-eligible rows only."""
        with tempfile.TemporaryDirectory() as tmp:
            unit = self._write(tmp, "unit.json", {"results": [
                {"candidate_id": "dc_arnq", "unit_side_status": "fail",
                 "reason": "isolated_speedup 0.848"},
                {"candidate_id": "dc_ar", "unit_side_status": "pass", "reason": "1.4x"},
                {"candidate_id": "dc_nq", "unit_side_status": "pass", "reason": "1.3x"},
                {"candidate_id": "pf_kv", "unit_side_status": "pass", "reason": "1.2x"}]})
            res = self._run(
                tmp, unitside_path=unit, budget=2,
                apply=self._apply(accepted_fusions=[
                    {"exec_id": "e02", "fusion": "fused_ar_rmsnorm"},
                    {"exec_id": "e03", "fusion": "add_rmsnorm_quant"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp, {"e01": "blocked", "e02": "applied",
                                    "e03": "applied", "e04": "deferred_budget"})
            self.assertEqual(res["budget"]["cutoff_rank"], 3)
            self.assertEqual(res["status"], "pass")

    def test_a_passing_row_inside_the_shifted_budget_cannot_be_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            unit = self._write(tmp, "unit.json", {"results": [
                {"candidate_id": "dc_arnq", "unit_side_status": "fail", "reason": "0.85x"},
                {"candidate_id": "dc_ar", "unit_side_status": "pass", "reason": "1.4x"},
                {"candidate_id": "dc_nq", "unit_side_status": "pass", "reason": "1.3x"}]})
            res = self._run(
                tmp, unitside_path=unit, budget=2,
                apply=self._apply(accepted_fusions=[
                    {"exec_id": "e02", "fusion": "fused_ar_rmsnorm"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            # rank 3 is the 2nd eligible row: inside the budget, so silence is a hole.
            self.assertEqual(disp["e03"], "unaccounted")
            self.assertEqual(res["status"], "fail")

    def test_a_row_with_no_unitside_verdict_says_where_the_gap_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp)
            row = [r for r in res["results"] if r["exec_id"] == "e03"][0]
            self.assertIn("Phase 3.0", row["reason"])

    # ---- provenance ----------------------------------------------------------
    def test_an_apply_record_matching_no_row_is_an_error(self):
        """Crediting a stray record to whichever row it resembles would turn a hole
        into a false 'applied'; it is reported instead."""
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, apply=self._apply(
                rejected=[{"fusion": "some_kernel_not_on_the_board",
                           "reason": "n/a"},
                          {"exec_id": "e03", "reason": "not built"}],
                deferred=[{"exec_id": "e04", "reason": "next round"}]))
            self.assertEqual(res["status"], "fail")
            self.assertTrue(any("matches no execution_list entry" in e
                                for e in res["errors"]), res["errors"])

    def test_a_record_can_be_matched_by_candidate_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, apply=self._apply(
                rejected=[{"candidate_id": "dc_nq", "reason": "kernel not built"}],
                deferred=[{"fusion": "concat_and_cast", "reason": "next round"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e03"], "blocked")
            self.assertEqual(disp["e04"], "deferred_with_reason")
            self.assertEqual(res["status"], "pass")

    def test_a_board_without_an_execution_list_is_refused(self):
        board = self._topk()
        board.pop("execution_list")
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, topk=board)
            self.assertEqual(res["status"], "fail")
            self.assertTrue(any("no execution_list" in e for e in res["errors"]))

    def test_waiver_needs_a_reason_and_a_real_exec_id(self):
        with self.assertRaises(SystemExit):
            ab._parse_waivers(["e03="])
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, waivers={"e99": "typo"})
            self.assertTrue(any("unknown exec_id" in e for e in res["errors"]))

    def test_waiver_closes_a_hole_with_its_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, waivers={
                "e03": "aiter build lacks the gfx942 variant; round 2",
                "e04": "prefill 收益 0.07%"})
            self.assertEqual(res["status"], "pass")
            row = [r for r in res["results"] if r["exec_id"] == "e03"][0]
            self.assertEqual(row["source"], "waiver")
            self.assertIn("gfx942", row["reason"])

    def test_partial_coverage_can_be_allowed_but_stays_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, require_coverage=False)
            self.assertEqual(res["status"], "pass")
            self.assertEqual(res["coverage"]["unaccounted"], 2)
            self.assertIn("无交代", open(os.path.join(tmp, "out.md")).read())

    # ---- the report is the final fusion report ------------------------------
    def test_the_report_carries_the_whole_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, budget=4, waivers={
                "e03": "r2", "e04": "r2"})
            md = open(os.path.join(tmp, "out.md")).read()
            self.assertIn("可融合区间 3 个", md)
            self.assertIn("候选 5 条", md)
            self.assertIn("1234.5 tok/s", md)
            self.assertEqual(res["region_coverage"]["regions_total"], 3)


    # ---- "blocked" needs measured evidence (Qwen3.5-397B 20261010, B.9) ------
    def _qwen35_like(self):
        return self._apply(
            rejected=[
                {"exec_id": "e02", "measured": True, "blocked_evidence": "e2e_no_gain",
                 "reason": "A/B -0.01%, overlapping, inside the noise band"},
                {"exec_id": "e03", "measured": False, "engaged": True,
                 "reason": "the interleaved A/B was still on its first ref leg at return"},
                {"exec_id": "e04", "reason": "PHYSICAL BLOCK: all 8 serving GPUs 0-7 are held "
                                             "by the orphaned e03 A/B driver"}])

    def test_unmeasured_rows_are_never_accounted_even_with_partial_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, apply=self._qwen35_like(), require_coverage=False)
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e02"], "blocked")
            self.assertEqual(disp["e03"], "unmeasured")
            self.assertEqual(disp["e04"], "unmeasured")       # by its non-terminal wording
            self.assertEqual(res["coverage"]["unmeasured_exec_ids"], ["e03", "e04"])
            self.assertEqual(res["coverage"]["accounted"], 2)  # e01 applied + e02 blocked
            self.assertFalse(res["coverage"]["complete"])
            md = open(os.path.join(tmp, "out.md")).read()
            self.assertIn("未测", md)
            self.assertIn("`e03`", md)

    def test_the_final_report_fails_on_unmeasured_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, apply=self._qwen35_like())
            self.assertEqual(res["status"], "fail")

    def test_reason_code_and_retry_queue_are_unmeasured(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, require_coverage=False, apply=self._apply(
                rejected=[{"exec_id": "e02", "reason_code": "gpu_busy",
                           "reason": "lease held by apply_e01"}],
                retry=[{"exec_id": "e03", "reason_code": "not_finished",
                        "reason": "A/B still running when the call returned"}],
                unmeasured=[{"exec_id": "e04", "reason_code": "gpu_busy",
                             "reason": "retries exhausted"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual([disp[e] for e in ("e02", "e03", "e04")], ["unmeasured"] * 3)

    def test_a_measured_retry_supersedes_the_unmeasured_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, apply=self._apply(
                rejected=[{"exec_id": "e02", "measured": False, "reason": "gpu busy"},
                          {"exec_id": "e02", "measured": True,
                           "blocked_evidence": "accuracy_fail",
                           "reason": "calibrated gsm8k drop p=0.001"},
                          {"exec_id": "e03", "measured": True,
                           "blocked_evidence": "unit_fail", "reason": "0.8x"},
                          {"exec_id": "e04", "measured": True,
                           "blocked_evidence": "e2e_regression", "reason": "-3%"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e02"], "blocked")
            self.assertEqual(res["status"], "pass")

    def test_unknown_blocked_evidence_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, require_coverage=False, apply=self._apply(
                rejected=[{"exec_id": "e03", "measured": True,
                           "blocked_evidence": "felt_slow", "reason": "?"}]))
            self.assertTrue(any("blocked_evidence" in e for e in res["errors"]))

    def test_a_legacy_block_without_evidence_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, require_coverage=False, apply=self._apply(
                rejected=[{"exec_id": "e03", "reason": "kernel not built"}]))
            self.assertTrue(any("without declared measured evidence" in w
                                for w in res["warnings"]))

    def test_stack_pending_is_accounted_only_after_the_stack_measured_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            pending = {"exec_id": "e03", "measured": True, "stack_pending": True,
                       "overlay_path": "/ov/e03", "reason": "in-noise solo"}
            res = self._run(tmp, require_coverage=False, apply=self._apply(
                rejected=[pending], deferred=[{"exec_id": "e04", "reason": "r2"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e03"], "stack_pending")
            self.assertFalse(res["coverage"]["complete"])
            res = self._run(tmp, apply=self._apply(
                rejected=[dict(pending, stack_result={"decision": "reject", "delta_pct": 0.2})],
                deferred=[{"exec_id": "e04", "reason": "r2"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e03"], "blocked")
            self.assertEqual(res["status"], "pass")

    def test_the_best_unit_side_of_a_candidate_is_its_status(self):
        """An entry is benched on two sides (existing kernel + self-authored)."""
        with tempfile.TemporaryDirectory() as tmp:
            unit = self._write(tmp, "unit.json", {"results": [
                {"candidate_id": "dc_nq", "unit_side_status": "fail", "reason": "existing 0.9x"},
                {"candidate_id": "dc_nq", "unit_side_status": "pass", "reason": "self 1.4x"}]})
            res = self._run(tmp, unitside_path=unit, require_coverage=False)
            row = next(r for r in res["results"] if r["exec_id"] == "e03")
            self.assertEqual(row["unit_side_statuses"], ["pass"])
            self.assertEqual(row["disposition"], "unaccounted")

    def test_a_time_budget_deferral_is_accounted_even_for_a_unit_pass(self):
        """Apply-back stopped at the stacking reserve: deferred_budget, not a hole."""
        with tempfile.TemporaryDirectory() as tmp:
            unit = self._write(tmp, "unit.json", {"results": [
                {"candidate_id": "dc_nq", "unit_side_status": "pass", "reason": "1.4x"}]})
            res = self._run(tmp, unitside_path=unit, apply=self._apply(
                deferred=[{"exec_id": "e03", "deferred_budget": True,
                           "reason_code": "budget_reserve",
                           "reason": "KernelFusion budget reserved for greedy stacking"},
                          {"exec_id": "e04", "reason": "r2"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e03"], "deferred_budget")
            self.assertEqual(res["status"], "pass")

    def test_stack_harmful_is_measured_blocked_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, apply=self._apply(
                rejected=[{"exec_id": "e03", "measured": True, "stack_pending": True,
                           "blocked_evidence": "stack_harmful", "reason": "stack_harmful: -0.9%",
                           "stack_result": {"decision": "drop", "delta_pct": -0.9}}],
                deferred=[{"exec_id": "e04", "reason": "r2"}]))
            disp = {r["exec_id"]: r["disposition"] for r in res["results"]}
            self.assertEqual(disp["e03"], "blocked")
            self.assertEqual(res["status"], "pass")
            self.assertEqual(res["errors"], [])

    def test_below_the_landing_target_every_entry_names_its_stop_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, min_landed=2, apply=self._apply(
                rejected=[{"exec_id": "e03", "measured": True, "blocked_evidence": "wire_failure",
                           "wire_cause": "not_engaged", "reason": "never ENGAGED",
                           "evidence": "/ab/e03.json"}],
                deferred=[{"exec_id": "e04", "deferred_budget": True,
                           "reason_code": "budget_reserve", "reason": "reserve"}],
                wire_failures=[{"exec_id": "e03", "wire_cause": "not_engaged"}]))
            self.assertEqual(res["landed"], {"count": 1, "min": 2, "met": False})
            stages = {r["exec_id"]: r["stage"] for r in res["stop_stages"]}
            self.assertEqual(stages, {"e02": "exclusion", "e03": "wire", "e04": "budget"})
            md = open(os.path.join(tmp, "out.md")).read()
            self.assertIn("逐项止步环节", md)
            self.assertIn("下一轮最高优先级", md)
            self.assertIn("🔴 未达标", md)

    def test_landing_target_met_counts_pool_items_and_lists_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(tmp, min_landed=2, apply=self._apply(
                pool_accepted=[{"exec_id": "p01"}],
                rejected=[{"exec_id": "e03", "reason": "x", "measured": True,
                           "blocked_evidence": "unit_fail"}],
                deferred=[{"exec_id": "e04", "reason": "r2"}]))
            self.assertEqual(res["landed"]["count"], 2)
            self.assertTrue(res["landed"]["met"])
            self.assertEqual(res["stop_stages"], [])
            self.assertEqual(
                ab.stop_stage("blocked", {"blocked_evidence": "accuracy_fail"}), "accuracy")
            self.assertEqual(ab.stop_stage("blocked", {"blocked_evidence": "stack_harmful"}), "stack")
            self.assertEqual(ab.stop_stage("blocked", {}, "unitside"), "unit")


if __name__ == "__main__":
    unittest.main()
