#!/usr/bin/env python3
"""Apply-back coverage gate for the KernelFusion stage.

Deterministic validator of what the `fusion_integrator` role actually did with the
Phase 2.2 Top-K board. It runs no server and measures nothing of its own; it answers
one question the rest of the pipeline could not:

    every row the board put on the table -- did it end somewhere explicit?

WHY THIS EXISTS
---------------
Phase 2.1 gates the candidate SET against the table. Phase 3.0 gates the 单侧
verdicts against the candidate set. Both are recall gates now. Apply-back had
neither: `FUSION_APPLY_SCHEMA` requires `accepted_fusions` and nothing else, so a run
that integrated 3 fusions and quietly walked past the other 9 returned
`accepted_fusions: [3]` and read as a success. On DSR1 2026-08-26, 26 of 42
candidates reached the end of the pipeline with no disposition of any kind -- not
applied, not rejected, not deferred. Just absent.

So the Top-K `execution_list` is the denominator, and every `exec_id` must end in
exactly one of:

  * `applied`              -- integrated, gated, kept
  * `blocked`              -- tried or ruled out, WITH a reason (wire failure, accuracy
                              gate, no 单侧 win, kernel unavailable...)
  * `blocked_by_exclusion` -- a conflicting entry in its exclusive group was applied.
                              Derived, not asserted: only entries that genuinely
                              conflict with an APPLIED entry get this.
  * `deferred_with_reason` -- knowingly left for next round, WITH a reason. This is
                              INVALID for an in-budget tier-A/B row whose unit-side
                              status passed: that row must be applied or blocked by an
                              actual apply-back attempt.
  * `deferred_budget`      -- past `--budget` in board rank order, or a `deferred`
                              record with `deferred_budget: true` / reason_code
                              `budget_reserve` (apply-back stopped at the stacking
                              reserve of the KernelFusion time budget). Legitimate, but
                              recorded and named. A budget is a decision to stop, and
                              a decision to stop is a thing the reader must SEE.
  * `unaccounted`          -- nobody said anything → the gate FAILS.
  * `unmeasured`           -- somebody said something, but nothing was MEASURED: GPU
                              busy, A/B not finished, "not reached in this call".
                              NOT accounted (see below).
  * `stack_pending`        -- passed its unit gates, engaged, did not regress, but its
                              solo A/B sat in the noise band; it is waiting for the
                              stacked re-gate. Accounted only once that re-gate has
                              measured it (`stack_result`), otherwise NOT accounted.

`unaccounted` is the whole point. "Not mentioned" is not "skipped"; it is a hole in
the report, and it must be red.

BLOCKED NEEDS MEASURED EVIDENCE
-------------------------------
Qwen3.5-397B (20261010_test_e2e_qwen35): e04..e08 came back `rejected` with reasons
like "all 8 serving GPUs held by the orphaned e03 A/B driver; NO A/B numbers
reported" and `measured: false`. The gate counted them as `blocked`, i.e. accounted,
and 05_FUSION_APPLYBACK.md rendered "已交代 8 / 无交代 0 ✅ pass" over a board where
six of eight entries had never been measured. "GPU busy" is a fact about US, not about
the fusion. So a `rejected` record is `blocked` only with measured evidence -- a unit
failure, an e2e A/B regression / no gain, an accuracy failure, a measured engagement
failure, a wire/launch failure that actually ran -- and is `unmeasured` when it says
`measured: false`, carries a non-terminal `reason_code` (gpu_busy / not_finished /
not_reached), sits in the integrator's `retry` / `unmeasured` lists, or its reason is
one of the non-terminal phrasings. `--allow-partial-coverage` keeps an unmeasured row
from failing an INTERMEDIATE report (the workflow re-runs it from its retry queue),
but it never turns it into an accounted row: it stays listed, counted and red.

BUDGET TRUNCATION
-----------------
`--budget N` is not a licence to ignore the tail. It counts only rows whose 单侧
status is apply-back eligible (pass / equivalent_pass / subsumed_pass) -- a row the
unit-side gate already blocked uses no slot -- and excuses entries ranked after the
N-th such row (plain board rank N when no --unitside is given). If a rank-2 entry is unaccounted while rank-9 was applied, the
budget is not the explanation and the gate still fails -- the loop skipped something
inside its own budget.

Usage:
  python3 fusion_applyback_harness.py --topk fusion_topk.json \
      --apply apply_result.json [--unitside fusion_unitside.json] \
      --out-md FUSION_APPLYBACK.md --out-json fusion_applyback.json \
      [--budget 6] [--allow-partial-coverage] [--waive <exec_id>=<reason> ...] \
      [--accuracy-mode skip --calibration-json <calibration.json>]

--accuracy-mode skip records that the round ran no e2e accuracy test because its
base-vs-base calibration found no usable one; accuracy then rests on unit-side numerical
equivalence, rows without an accuracy measurement are not failed for it, and the report
states 「e2e 精度未把关」 with the calibration evidence path.
"""
import argparse
import json
import os
import sys


ACCOUNTED = {"applied", "blocked", "blocked_by_exclusion",
             "deferred_with_reason", "deferred_budget"}
EXPLICIT = {"applied", "blocked", "deferred_with_reason", "deferred_budget", "unmeasured",
            "stack_pending"}
APPLYBACK_ELIGIBLE = {"pass", "equivalent_pass", "subsumed_pass"}
# Statuses that mean "we did not get a measurement", never a verdict on the fusion.
NON_TERMINAL_CODES = {"gpu_busy", "not_finished", "not_reached", "retry", "deferred_retry",
                      "lease_busy", "timeout", "agent_failed"}
# Measured evidence a `blocked` record may stand on.
BLOCKED_EVIDENCE = {"unit_fail", "unit_parity_fail", "e2e_regression", "e2e_no_gain",
                    "ab_reject", "accuracy_fail", "not_engaged", "wire_failure",
                    "kernel_unavailable", "stack_reject", "stack_harmful"}
# A deferred record carrying one of these was stopped by the KernelFusion TIME budget (the
# apply-back loop keeps >= 30% of the budget for greedy stacking): a recorded decision to
# stop, accounted like a rank-budget cut, and written back as next-round candidate input.
BUDGET_DEFERRED_CODES = {"budget_reserve", "budget_exhausted", "deferred_budget"}
# Order a unit-side row is preferred in when one candidate carries several sides
# (existing kernel + self-authored): the best measured side is the candidate's status.
UNIT_STATUS_RANK = {"pass": 0, "equivalent_pass": 0, "subsumed_pass": 0, "fail": 1,
                    "equivalent_fail": 1, "blocked": 2, "equivalent_blocked": 2,
                    "needs_diagnosis": 3, "equivalent_needs_diagnosis": 3}
_UNMEASURED_PHRASES = (
    "not reached a verified terminal", "not reach a terminal", "no a/b or gsm8k numbers",
    "no a/b throughput", "gpu busy", "gpus busy", "gpus are held", "gpus 0-7 are held",
    "serving gpus (0-", "still on its first", "still running at return",
    "measurement not finished", "not measured")


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _norm(text):
    """Lowercase, collapse everything that is not alnum/underscore to a separator."""
    out = []
    for ch in str(text or "").lower():
        out.append(ch if (ch.isalnum() or ch == "_") else " ")
    return "_".join(t for t in "".join(out).split() if t)


def _keys(exec_id=None, candidate_ids=(), handle=None, action=None):
    """The identity set an execution row and an apply record are matched on.

    Deliberately NOT a fuzzy name match. An apply record has to name an exec_id, a
    candidate_id, or the exact handle; anything else is reported as unmatched rather
    than credited to whichever row it happens to resemble. Crediting the wrong row is
    worse than failing to credit: it turns a hole into a false 'applied'.
    """
    keys = set()
    if exec_id:
        keys.add("exec:%s" % _norm(exec_id))
    for cid in candidate_ids or ():
        if cid:
            keys.add("cand:%s" % _norm(cid))
    if handle:
        keys.add("handle:%s" % _norm(handle))
    if action:
        keys.add("handle:%s" % _norm(action))
    return keys


def _rec_keys(rec):
    cids = rec.get("candidate_ids")
    if cids is None:
        cids = [rec.get("candidate_id")] if rec.get("candidate_id") else []
    return _keys(rec.get("exec_id"),
                 cids,
                 rec.get("handle") or rec.get("fusion") or rec.get("name"),
                 rec.get("action"))


def _collect_records(apply_payload, errors):
    """Normalize the integrator's return into [(disposition, reason, record)].

    Three accepted shapes, so an existing role return still works:
      * `dispositions: [{exec_id, disposition, reason}]`  -- the explicit form
      * `accepted_fusions: [...]`                          -- → applied
      * `rejected: [...]` / `deferred: [...]`              -- → blocked / deferred
    """
    records = []

    def _reason(rec):
        for field in ("reason", "why", "notes", "note"):
            val = str(rec.get(field) or "").strip()
            if val:
                return val
        return ""

    def _rejected_disposition(rec):
        """blocked only with measured evidence; otherwise unmeasured / stack_pending."""
        if _is_unmeasured(rec):
            return "unmeasured"
        if rec.get("stack_pending") and not rec.get("stack_result"):
            return "stack_pending"
        evidence = rec.get("blocked_evidence")
        if evidence and str(evidence) not in BLOCKED_EVIDENCE:
            errors.append(
                "rejected record %r declares blocked_evidence %r; must be one of %s"
                % (rec.get("exec_id") or rec.get("fusion"), evidence,
                   sorted(BLOCKED_EVIDENCE)))
        return "blocked"

    for rec in apply_payload.get("dispositions") or []:
        disp = str(rec.get("disposition") or "").strip().lower()
        if disp in ("deferred", "defer"):
            disp = "deferred_with_reason"
        if disp == "blocked":
            disp = _rejected_disposition(rec)
        if disp not in EXPLICIT:
            errors.append(
                "dispositions[] entry %r has disposition %r; must be one of %s"
                % (rec.get("exec_id") or rec.get("fusion"), rec.get("disposition"),
                   sorted(EXPLICIT)))
            continue
        records.append((disp, _reason(rec), rec))
    for rec in apply_payload.get("accepted_fusions") or []:
        records.append(("applied", _reason(rec) or "integrated and kept", rec))
    for rec in apply_payload.get("rejected") or []:
        records.append((_rejected_disposition(rec), _reason(rec), rec))
    for rec in apply_payload.get("deferred") or []:
        if rec.get("deferred_budget") or str(rec.get("reason_code") or "").lower() \
                in BUDGET_DEFERRED_CODES:
            records.append(("deferred_budget", _reason(rec) or "KernelFusion time budget", rec))
            continue
        records.append(("unmeasured" if _is_unmeasured(rec) else "deferred_with_reason",
                        _reason(rec), rec))
    # The workflow's retry queue and the entries it exhausted: never a verdict.
    for field in ("retry", "unmeasured"):
        for rec in apply_payload.get(field) or []:
            records.append(("unmeasured", _reason(rec) or (
                "not measured (%s)" % (rec.get("reason_code") or field)), rec))

    # A blocked/deferred row without a reason is the failure mode this gate exists to
    # stop, just one level down: "we looked at it" with nothing behind it.
    for disp, reason, rec in records:
        if disp != "applied" and not reason:
            errors.append(
                "apply record %r is %s with no reason; a disposition without a reason "
                "is not a disposition" % (rec.get("exec_id") or rec.get("fusion")
                                          or rec.get("candidate_id"), disp))
    return records


def _is_unmeasured(rec):
    """True when a record reports that nothing was measured (never a verdict)."""
    if rec.get("measured") is False:
        return True
    code = str(rec.get("reason_code") or rec.get("status") or "").strip().lower()
    if code in NON_TERMINAL_CODES:
        return True
    if rec.get("measured") is True or rec.get("blocked_evidence"):
        return False
    reason = " ".join(str(rec.get(f) or "") for f in ("reason", "why", "notes")).lower()
    return any(p in reason for p in _UNMEASURED_PHRASES)


# Where an entry stopped (二.10.6d), read off its measured evidence.
_STAGE_BY_EVIDENCE = {
    "unit_fail": "unit", "unit_parity_fail": "unit",
    "wire_failure": "wire", "not_engaged": "wire", "kernel_unavailable": "wire",
    "e2e_regression": "e2e", "e2e_no_gain": "e2e", "ab_reject": "e2e",
    "accuracy_fail": "accuracy",
    "stack_harmful": "stack", "stack_reject": "stack",
}
_STAGE_BY_DISPOSITION = {
    "applied": "landed", "stack_pending": "stack", "unmeasured": "unmeasured",
    "deferred_budget": "budget", "deferred_with_reason": "deferred",
    "blocked_by_exclusion": "exclusion", "unaccounted": "none",
}


def stop_stage(disposition, rec, source=""):
    """unit / wire / e2e / accuracy / stack (or landed / budget / unmeasured / ...)."""
    if disposition == "blocked":
        if source == "unitside":
            return "unit"
        evidence = str((rec or {}).get("blocked_evidence") or "")
        if evidence in _STAGE_BY_EVIDENCE:
            return _STAGE_BY_EVIDENCE[evidence]
        if (rec or {}).get("stack_result"):
            return "stack"
        return "e2e" if (rec or {}).get("throughput_delta_pct") is not None else "unknown"
    return _STAGE_BY_DISPOSITION.get(disposition, disposition or "unknown")


def _stage_evidence(rec):
    rec = rec or {}
    for key in ("evidence", "evidence_path", "ab_json", "applyback_json", "overlay_path"):
        if rec.get(key):
            return str(rec[key])
    sr = rec.get("stack_result") or {}
    if sr.get("evidence"):
        return str(sr["evidence"])
    if rec.get("unit_side_statuses"):
        return "unit-side: " + ", ".join(str(s) for s in rec["unit_side_statuses"])
    if rec.get("throughput_delta_pct") is not None:
        return "A/B delta %s%%" % rec["throughput_delta_pct"]
    return ""


def _unitside_index(unitside_payload):
    """candidate_id -> (unit_side_status, reason) of its BEST measured side."""
    index = {}
    for row in (unitside_payload or {}).get("results") or []:
        cid = row.get("candidate_id")
        if not cid:
            continue
        cur = (str(row.get("unit_side_status") or ""), str(row.get("reason") or ""))
        prev = index.get(cid)
        if prev is None or UNIT_STATUS_RANK.get(cur[0], 9) < UNIT_STATUS_RANK.get(prev[0], 9):
            index[cid] = cur
    return index


def _conflict_map(topk):
    """exec_id -> set(exec_ids it conflicts with), read off the exclusive groups.

    Uses the pairwise `conflict_edges` when present (the Top-K harness emits them for
    a `compatible_subset` group). A `choose: 1` group is a clique, so every pair in it
    conflicts. Falling back to "everyone in the group conflicts" for a non-clique group
    would auto-block entries that could legally have landed together.
    """
    conflicts = {}

    def _link(a, b):
        conflicts.setdefault(a, set()).add(b)
        conflicts.setdefault(b, set()).add(a)

    for group in topk.get("exclusive_groups") or []:
        ids = [m.get("exec_id") for m in group.get("members") or [] if m.get("exec_id")]
        edges = group.get("conflict_edges")
        if edges:
            for edge in edges:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    _link(edge[0], edge[1])
        elif group.get("choose") == 1 or group.get("choose") == "1":
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    _link(a, b)
        else:
            for member in group.get("members") or []:
                for other in member.get("conflicts_with") or []:
                    if member.get("exec_id"):
                        _link(member["exec_id"], other)
    return conflicts


def _budget_cutoff_rank(execution_list, unit_index, budget):
    """The last board rank the apply-back budget reaches, or None for "every row".

    The budget counts only rows whose 单侧 result makes them apply-back eligible: a row
    the unit-side gate already blocked costs no serving A/B, so it must not use up a
    slot that a later passing row needs. Without unit-side evidence there is nothing to
    filter on and the budget is the plain board rank, as before."""
    if budget is None:
        return None
    if not unit_index:
        return int(budget)
    if int(budget) <= 0:
        return 0
    eligible = 0
    for entry in execution_list:
        statuses = {unit_index[cid][0] for cid in entry.get("candidate_ids") or []
                    if cid in unit_index}
        if statuses & APPLYBACK_ELIGIBLE:
            eligible += 1
            if eligible == int(budget) and entry.get("rank") is not None:
                return int(entry["rank"])
    return None


def accuracy_gate_record(accuracy_mode=None, calibration_json=None):
    mode = str(accuracy_mode or "gate").lower()
    rec = {"mode": mode, "e2e_accuracy_gated": mode != "skip",
           "calibration_json": os.path.abspath(calibration_json) if calibration_json else ""}
    if mode == "skip":
        rec["statement"] = "e2e 精度未把关"
        rec["reason"] = ("calibration_unusable: no e2e accuracy test (gsm8k / deterministic "
                         "probe) passed the base-vs-base control; accuracy rests on unit-side "
                         "numerical equivalence")
    return rec


def validate(topk_path, apply_path, unitside_path=None, budget=None,
             require_coverage=True, waivers=None, min_landed=None,
             accuracy_mode=None, calibration_json=None):
    topk = _load(topk_path)
    apply_payload = _load(apply_path)
    unitside = _load(unitside_path) if unitside_path else None
    waivers = dict(waivers or {})

    errors = []
    warnings = []

    execution_list = topk.get("execution_list") or []
    if not execution_list:
        errors.append(
            "Top-K json carries no execution_list — this gate has no denominator to "
            "check against. Regenerate FUSION_TOPK with fusion_topk_harness.py "
            "(schema_version >= 3).")

    records = _collect_records(apply_payload, errors)
    unit_index = _unitside_index(unitside)
    conflicts = _conflict_map(topk)
    cutoff_rank = _budget_cutoff_rank(execution_list, unit_index, budget)

    # ---- pass 1: attach explicit records to rows -----------------------------
    rows = []
    matched_records = set()
    for entry in execution_list:
        exec_id = entry.get("exec_id")
        keys = _keys(exec_id, entry.get("candidate_ids"),
                     entry.get("handle"), entry.get("action"))
        hits = [(i, d, r, rec) for i, (d, r, rec) in enumerate(records)
                if _rec_keys(rec) & keys]
        for i, _d, _r, _rec in hits:
            matched_records.add(i)
        rows.append({"entry": entry, "keys": keys, "hits": hits})

    unmatched = [i for i in range(len(records)) if i not in matched_records]
    for i in unmatched:
        disp, _reason, rec = records[i]
        errors.append(
            "apply record %r (%s) matches no execution_list entry — it names no "
            "exec_id / candidate_id / handle on the board, so it cannot be credited "
            "to any row"
            % (rec.get("exec_id") or rec.get("fusion") or rec.get("candidate_id"),
               disp))

    applied_ids = set()
    for row in rows:
        if any(d == "applied" for _i, d, _r, _rec in row["hits"]):
            applied_ids.add(row["entry"].get("exec_id"))

    # ---- pass 2: derive the disposition of every row -------------------------
    results = []
    for row in rows:
        entry = row["entry"]
        exec_id = entry.get("exec_id")
        rank = entry.get("rank")
        hits = row["hits"]
        unit_states = [unit_index.get(cid) for cid in entry.get("candidate_ids") or []]
        unit_states = [u for u in unit_states if u]
        unit_statuses = sorted({u[0] for u in unit_states})
        within_budget = (
            cutoff_rank is None or rank is None or int(rank) <= cutoff_rank
        )
        applyback_required = bool(
            str(entry.get("tier") or "").upper() in ("A", "B") and
            within_budget and
            any(status in APPLYBACK_ELIGIBLE for status in unit_statuses)
        )
        disposition, reason, source = None, "", ""
        chosen = None

        if hits:
            # An explicit record wins. If the loop said several things about one row,
            # 'applied' is the strongest claim and the rest are its history.
            order = {"applied": 0, "blocked": 1, "stack_pending": 2,
                     "deferred_with_reason": 3, "deferred_budget": 3, "unmeasured": 4}
            _i, disposition, reason, rec = sorted(
                hits, key=lambda h: order.get(h[1], 9))[0]
            chosen = rec
            source = "integrator"
            if len({d for _i, d, _r, _rec in hits}) > 1:
                warnings.append(
                    "%s carries several apply records (%s); taking '%s'"
                    % (exec_id, sorted({d for _i, d, _r, _rec in hits}), disposition))
        elif exec_id in waivers:
            disposition, reason, source = ("deferred_with_reason",
                                           waivers[exec_id], "waiver")
        else:
            # No claim was made. Before calling it a hole, check the two places a
            # disposition legitimately comes from without the integrator saying so.
            if unit_states and all(u[0] in ("fail", "blocked") for u in unit_states):
                disposition = "blocked"
                reason = "单侧 gate: " + "; ".join(
                    "%s (%s)" % (u[0], u[1]) for u in unit_states[:2])
                source = "unitside"
            elif unit_states and all(
                    u[0] in ("budget_skipped", "not_validated", "fail", "blocked")
                    for u in unit_states):
                # NEVER MEASURED is not a verdict. It must not inherit `blocked`
                # (which reads as "we looked and it lost") and it must not be
                # laundered into the tidy `deferred_budget` row either -- the
                # unit-side budget ran out, so nobody knows what this fusion does.
                disposition = "unaccounted"
                reason = ("单侧 never measured it (%s) — no evidence either way; "
                          "raise fusion_unitside_budget or hang it off a passing "
                          "ladder top" % "; ".join(
                              "%s: %s" % (u[0], u[1]) for u in unit_states[:2]))
                source = "unitside_gap"
            elif conflicts.get(exec_id, set()) & applied_ids:
                disposition = "blocked_by_exclusion"
                reason = ("conflicts with applied %s"
                          % sorted(conflicts[exec_id] & applied_ids))
                source = "exclusion"
            elif not within_budget:
                disposition = "deferred_budget"
                reason = ("rank %s beyond --budget %s (counted over 单侧-passing rows, "
                          "cut at rank %s) — not attempted this round"
                          % (rank, budget, cutoff_rank))
                source = "budget"
            else:
                disposition = "unaccounted"
                unit_note = ""
                if entry.get("candidate_ids") and not unit_states:
                    unit_note = (" (its candidates have no 单侧 verdict either — the "
                                 "gap starts at Phase 3.0)")
                reason = ("no disposition anywhere: the integrator did not report it, "
                          "no exclusion covers it, and it is inside the budget"
                          + unit_note)
                source = "none"

        if disposition == "blocked" and source == "integrator":
            rec = sorted(hits, key=lambda h: order.get(h[1], 9))[0][3]
            if rec.get("measured") is not True and not rec.get("blocked_evidence"):
                warnings.append(
                    "%s is blocked without declared measured evidence (set measured:true "
                    "and blocked_evidence ∈ %s); a block must stand on a measurement"
                    % (exec_id, sorted(BLOCKED_EVIDENCE)))

        if disposition == "deferred_with_reason" and applyback_required:
            errors.append(
                "%s is an in-budget tier-%s row with passing unit-side evidence (%s), "
                "so deferred_with_reason is not a valid terminal state. Run live "
                "apply-back and report applied, or blocked with the attempted "
                "engagement/A/B/accuracy evidence."
                % (exec_id, entry.get("tier"), ", ".join(unit_statuses)))

        results.append({
            "exec_id": exec_id,
            "rank": rank,
            "phase": entry.get("phase"),
            "tier": entry.get("tier"),
            "action": entry.get("action"),
            "handle": entry.get("handle"),
            "candidate_ids": entry.get("candidate_ids") or [],
            "unit_side_statuses": unit_statuses,
            "applyback_required": applyback_required,
            "forward_us": entry.get("forward_us"),
            "forward_pct": entry.get("forward_pct"),
            "workload_weight": entry.get("workload_weight"),
            "workload_savings_us": entry.get("workload_savings_us"),
            "workload_forward_pct": entry.get("workload_forward_pct"),
            "disposition": disposition,
            "reason": reason,
            "source": source,
            "stop_stage": stop_stage(disposition, chosen, source),
            "stage_evidence": _stage_evidence(chosen),
        })

    unaccounted = [r["exec_id"] for r in results if r["disposition"] == "unaccounted"]
    unmeasured = [r["exec_id"] for r in results if r["disposition"] == "unmeasured"]
    stack_waiting = [r["exec_id"] for r in results if r["disposition"] == "stack_pending"]
    waived_unknown = sorted(
        set(waivers) - {e.get("exec_id") for e in execution_list})
    for exec_id in waived_unknown:
        errors.append("--waive references unknown exec_id '%s'" % exec_id)

    counts = {}
    for r in results:
        counts[r["disposition"]] = counts.get(r["disposition"], 0) + 1

    required_applyback = [r for r in results if r["applyback_required"]]
    incomplete_applyback = [
        r for r in required_applyback
        if r["disposition"] not in ("applied", "blocked", "blocked_by_exclusion")
    ]

    truncated = [r["exec_id"] for r in results if r["disposition"] == "deferred_budget"]
    budget_record = {
        "declared": budget,
        "execution_list_size": len(execution_list),
        "beyond_budget": len(truncated),
        "beyond_budget_exec_ids": truncated,
        "cutoff_rank": cutoff_rank,
    }
    if cutoff_rank is not None and len(execution_list) > cutoff_rank:
        # Count the rows the budget ACTUALLY excused, not len-cutoff: a row past
        # the budget that the integrator deferred on its own merits was attempted
        # as a decision, and folding it into the budget number overstates the cut.
        warnings.append(
            "--budget %s truncates a %d-row board at rank %s: %d row(s) rank beyond "
            "it, %d of which were left entirely unattempted"
            % (budget, len(execution_list), cutoff_rank,
               len(execution_list) - cutoff_rank, len(truncated)))

    # Landing target (二.10.6): landed = applied execution-list rows (solo accepts and members of
    # a landed greedy stack) + candidate-pool items the landed stack carried.
    landed_count = (sum(1 for r in results if r["disposition"] == "applied")
                    + len(apply_payload.get("pool_accepted") or []))
    landed = {"count": landed_count, "min": min_landed,
              "met": None if min_landed is None else landed_count >= int(min_landed)}
    if landed["met"] is False:
        warnings.append(
            "only %d fusion(s) landed, below the landing target %s; every entry's stop stage "
            "(unit / wire / e2e / accuracy / stack) is listed in the report"
            % (landed_count, min_landed))

    accuracy_gate = accuracy_gate_record(accuracy_mode, calibration_json)
    if not accuracy_gate["e2e_accuracy_gated"]:
        warnings.append(
            "e2e 精度未把关: the round's accuracy calibration found no usable test%s; "
            "accuracy rests on unit-side numerical equivalence"
            % (" (%s)" % accuracy_gate["calibration_json"]
               if accuracy_gate["calibration_json"] else ""))
        for r in results:
            if r["disposition"] == "blocked" and str(
                    (r.get("stop_stage") or "")) == "accuracy":
                warnings.append(
                    "%s is blocked on accuracy although no e2e accuracy test ran this round"
                    % r["exec_id"])

    coverage_fail = bool(require_coverage and (unaccounted or unmeasured or stack_waiting))
    accounted = sum(1 for r in results if r["disposition"] in ACCOUNTED)
    result = {
        "schema_version": 1,
        "phase": "applyback_gate",
        "status": "fail" if (errors or coverage_fail) else "pass",
        "topk_json": os.path.abspath(topk_path),
        "apply_json": os.path.abspath(apply_path),
        "unitside_json": os.path.abspath(unitside_path) if unitside_path else "",
        "errors": errors,
        "warnings": warnings,
        "coverage": {
            "required": require_coverage,
            "execution_list_size": len(execution_list),
            "accounted": accounted,
            "unaccounted": len(unaccounted),
            "unaccounted_exec_ids": unaccounted,
            # Said something, measured nothing. Never accounted, with or without
            # --allow-partial-coverage.
            "unmeasured": len(unmeasured),
            "unmeasured_exec_ids": unmeasured,
            "stack_pending": len(stack_waiting),
            "stack_pending_exec_ids": stack_waiting,
            "complete": not (unaccounted or unmeasured or stack_waiting),
        },
        "coverage_ok": not coverage_fail,
        "applyback_completion": {
            "required": len(required_applyback),
            "completed": len(required_applyback) - len(incomplete_applyback),
            "incomplete": len(incomplete_applyback),
            "incomplete_exec_ids": [r["exec_id"] for r in incomplete_applyback],
            "complete": not incomplete_applyback,
        },
        "budget": budget_record,
        "landed": landed,
        "stop_stages": ([{"exec_id": r["exec_id"], "stage": r["stop_stage"],
                          "evidence": r["stage_evidence"], "reason": r["reason"]}
                         for r in results if r["disposition"] != "applied"]
                        if landed["met"] is False else []),
        "wire_failures": apply_payload.get("wire_failures") or [],
        "accuracy_gate": accuracy_gate,
        # Alternative-group rungs unit-tested on demand during apply-back (outside the
        # Top-K coverage denominator; recorded for the report only).
        "unit_on_demand": apply_payload.get("unit_on_demand") or [],
        "counts": counts,
        "results": results,
        # The board's own coverage travels one more hop, so the FINAL report carries
        # the whole chain: which phases were analysed → which regions were covered →
        # which candidates made the board → which of those actually landed.
        "phase_coverage": topk.get("phase_coverage"),
        "region_coverage": topk.get("region_coverage"),
        "family_coverage": topk.get("family_coverage"),
        "candidate_total": topk.get("candidate_total"),
        "candidates_on_board": topk.get("candidates_on_board"),
        "truncated_count": topk.get("truncated_count"),
        "e2e": {
            "throughput_tok_s": apply_payload.get("e2e_throughput_tok_s"),
            "final_overlay": apply_payload.get("final_overlay"),
            "notes": apply_payload.get("notes"),
        },
        "accepted_fusions": apply_payload.get("accepted_fusions") or [],
    }
    return result


def _esc(value):
    return str(value if value is not None else "").replace("|", "\\|")


_DISP_LABEL = {
    "applied": "✅ 已落地",
    "blocked": "⛔ 被挡",
    "blocked_by_exclusion": "⛔ 互斥被挡",
    "deferred_with_reason": "⏸ 延后",
    "deferred_budget": "⏸ 超预算",
    "unaccounted": "🔴 无交代",
    "unmeasured": "⏳ 未测（不算交代）",
    "stack_pending": "⏸ 待叠加（未复测）",
}


def render_markdown(result):
    lines = ["# KernelFusion Apply-back 结果", ""]
    lines.append(
        "融合的**最终结果报告**：Phase 2.2 执行清单上的每一条，在这里都必须有一个明确去向。"
        "「没提到」不是「跳过」，是报告的漏洞——它会被标成 🔴 无交代并让本 gate fail。")
    lines.append("")

    acc_gate = result.get("accuracy_gate") or {}
    if acc_gate.get("mode") == "skip":
        lines.append("> ⚠ **e2e 精度未把关**：本轮基线对基线校准中，gsm8k 与确定性检查都未通过"
                     "（基线本身不可信），apply-back 与叠加步骤均未跑 e2e 精度测试，精度只靠单测"
                     "数值等价把关。校准证据：`%s`"
                     % _esc(acc_gate.get("calibration_json") or "(未提供)"))
        lines.append("")
    cov = result.get("coverage") or {}
    lines.append("## 交代覆盖率（先看这里）")
    lines.append("")
    lines.append("执行清单 **%d** 条：已交代 **%d** / 无交代 **%d** / 未测 **%d** / "
                 "待叠加 **%d**。"
                 % (cov.get("execution_list_size", 0), cov.get("accounted", 0),
                    cov.get("unaccounted", 0), cov.get("unmeasured", 0),
                    cov.get("stack_pending", 0)))
    completion = result.get("applyback_completion") or {}
    if completion.get("required"):
        lines.append("")
        lines.append("预算内且单侧通过、必须完成 Applyback **%d** 条：已完成 **%d** / "
                     "**未完成 %d**。"
                     % (completion.get("required", 0),
                        completion.get("completed", 0),
                        completion.get("incomplete", 0)))
    landed = result.get("landed") or {}
    if landed.get("min") is not None:
        lines.append("")
        lines.append("落地数 **%d** / 目标 N_min **%s**：%s。"
                     % (landed.get("count", 0), landed.get("min"),
                        "✅ 达标" if landed.get("met") else "🔴 未达标"))
    if result.get("stop_stages"):
        lines.append("")
        lines.append("### 未达落地目标：逐项止步环节")
        lines.append("")
        lines.append("| exec_id | 止步环节 | 证据 | 原因 |")
        lines.append("|---|---|---|---|")
        for row in result["stop_stages"]:
            lines.append("| `%s` | %s | %s | %s |" % (
                _esc(row["exec_id"]), _esc(row["stage"]), _esc(row["evidence"])[:160],
                _esc(row["reason"])[:200]))
        if result.get("wire_failures"):
            lines.append("")
            lines.append("接入失败（wire_failure，已排查并重测一次）的项为下一轮最高优先级：" + ", ".join(
                "`%s`" % _esc(w.get("exec_id")) for w in result["wire_failures"]) + "。")
    on_demand = result.get("unit_on_demand") or []
    if on_demand:
        lines.append("")
        lines.append("按需单测的备选级（前一级 unit_fail 或被实测否决后才测）：" + "、".join(
            "`%s` r%s → %s" % (_esc(u.get("candidate_id")), _esc(u.get("rung")),
                              _esc(u.get("status")))
            for u in on_demand) + "。")
    c = result.get("counts") or {}
    lines.append("")
    lines.append("去向分布：" + " / ".join(
        "%s %d" % (_DISP_LABEL.get(k, k), v) for k, v in sorted(c.items())) + "。")
    bud = result.get("budget") or {}
    if bud.get("declared") is not None:
        lines.append("")
        lines.append("预算 `--budget %s`：清单 %d 条，超预算未尝试 **%d** 条%s。"
                     % (bud.get("declared"), bud.get("execution_list_size", 0),
                        bud.get("beyond_budget", 0),
                        ("（%s）" % "、".join("`%s`" % e for e in
                                             bud.get("beyond_budget_exec_ids") or []))
                        if bud.get("beyond_budget") else ""))
    pc = result.get("phase_coverage") or {}
    rc = result.get("region_coverage") or {}
    if pc or rc:
        lines.append("")
        chain = []
        if pc.get("shape_resolution_by_phase"):
            chain.append("语义阶段：" + "、".join(
                "%s %d/%d" % (ph, v.get("resolved", 0), v.get("rows", 0))
                for ph, v in sorted(pc["shape_resolution_by_phase"].items())))
        if rc:
            chain.append("可融合区间 %d 个：已覆盖 %d / 已延后 %d / 未覆盖 %d"
                         % (rc.get("regions_total", rc.get("regions", 0)),
                            rc.get("covered", 0), rc.get("deferred", 0),
                            rc.get("uncovered", 0)))
        if result.get("candidate_total"):
            chain.append("候选 %d 条 → 进板 %s 条 → 清单 %d 条"
                         % (result.get("candidate_total"),
                            result.get("candidates_on_board"),
                            cov.get("execution_list_size", 0)))
        lines.append("链路：" + "；".join(chain) + "。")
    lines.append("")
    if cov.get("unaccounted"):
        lines.append(
            "> 🔴 **%d 条无交代**：%s。这些融合既没落地、也没被挡、也没延后——"
            "报告里根本没提。补一条明确处置（已落地/被挡+原因/延后+原因）或 "
            "`--waive <exec_id>=<理由>`。"
            % (cov.get("unaccounted"),
               "、".join("`%s`" % e for e in cov.get("unaccounted_exec_ids") or [])))
        lines.append("")
    if cov.get("unmeasured"):
        lines.append(
            "> 🔴 **%d 条未测**：%s。它们有一条记录，但没有任何实测数据（GPU 被占 / "
            "测量没跑完 / 本次调用没跑到）。「被挡」只允许用于有实测数据支撑的拒绝"
            "（单测不过、e2e 退化、精度不过），这些条目必须重试到有测量为止，"
            "不能算作已交代。"
            % (cov.get("unmeasured"),
               "、".join("`%s`" % e for e in cov.get("unmeasured_exec_ids") or [])))
        lines.append("")
    if cov.get("stack_pending"):
        lines.append(
            "> ⏸ **%d 条待叠加，尚未叠加复测**：%s。单测和接入都过、没有退化、只是单项收益"
            "落在噪声带里；叠加复测跑完之前它们不算已交代。"
            % (cov.get("stack_pending"),
               "、".join("`%s`" % e for e in cov.get("stack_pending_exec_ids") or [])))
        lines.append("")
    if completion.get("incomplete"):
        lines.append(
            "> 🔴 **预算内单侧通过项不得延期**：%s。它们必须完成 live Applyback，"
            "最终落到 applied，或在实际接入/engagement/A/B/accuracy 尝试后 blocked。"
            % "、".join("`%s`" % e for e in
                         completion.get("incomplete_exec_ids") or []))
        lines.append("")

    e2e = result.get("e2e") or {}
    if e2e.get("throughput_tok_s"):
        lines.append("## 端到端结果")
        lines.append("")
        lines.append("融合后吞吐 **%s tok/s**；stacked overlay：`%s`。"
                     % (e2e.get("throughput_tok_s"), e2e.get("final_overlay") or "-"))
        if e2e.get("notes"):
            lines.append("")
            lines.append(str(e2e["notes"]))
        lines.append("")

    lines.append("## 逐条去向")
    lines.append("")
    lines.append("| # | exec | 阶段 | 难度 | 融合 | 对应候选 | 预期 forward 收益 | 去向 | 依据 | 说明 |")
    lines.append("|---:|---|:--:|:--:|---|---|---:|:--:|:--:|---|")
    for r in result["results"]:
        if (r.get("workload_savings_us") is not None and
                r.get("workload_forward_pct") is not None):
            gain = "%.0f µs（%.2f%% workload；单次 %.0f µs / phase %.2f%%）" % (
                r["workload_savings_us"], r["workload_forward_pct"],
                r.get("forward_us") or 0.0, r.get("forward_pct") or 0.0)
        else:
            gain = ("%d µs（%.2f%%）" % (r["forward_us"], r["forward_pct"])
                    if r.get("forward_us") is not None
                    and r.get("forward_pct") is not None else "-")
        lines.append("| %s | `%s` | %s | %s | %s | %s | %s | **%s** | %s | %s |" % (
            _esc(r.get("rank")), _esc(r.get("exec_id")), _esc(r.get("phase")),
            _esc(r.get("tier")), _esc(r.get("action")),
            _esc("、".join("`%s`" % c for c in r.get("candidate_ids") or [])),
            gain, _DISP_LABEL.get(r["disposition"], r["disposition"]),
            _esc(r.get("source")), _esc(r.get("reason"))))
    lines.append("")

    accepted = result.get("accepted_fusions") or []
    if accepted:
        lines.append("## 已落地融合明细")
        lines.append("")
        lines.append("| 融合 | rung | overlay | TPOT Δ% | 吞吐 Δ% | gsm8k base→cand | engaged |")
        lines.append("|---|---|---|---:|---:|---|:--:|")
        for f in accepted:
            lines.append("| %s | %s | `%s` | %s | %s | %s → %s | %s |" % (
                _esc(f.get("fusion") or f.get("exec_id") or f.get("candidate_id")),
                _esc(f.get("rung")), _esc(f.get("overlay_path")),
                _esc(f.get("tpot_delta_pct")), _esc(f.get("throughput_delta_pct")),
                _esc(f.get("gsm8k_base")), _esc(f.get("gsm8k_cand")),
                _esc(f.get("engaged"))))
        lines.append("")

    if result.get("warnings"):
        lines.append("## 提示")
        lines.append("")
        for w in result["warnings"]:
            lines.append("- %s" % w)
        lines.append("")
    if result.get("errors"):
        lines.append("## 错误（必须修，不得当作通过）")
        lines.append("")
        for e in result["errors"]:
            lines.append("- %s" % e)
        lines.append("")
    lines.append(
        "说明：本 gate 不跑 server、不产生自己的性能数字；它只校验 apply-back 对 "
        "Phase 2.2 执行清单的**交代完整性**。`applied` 的收益数字来自 fusion_integrator "
        "的实测 A/B，`blocked_by_exclusion` 由互斥冲突图自动推导（只有与**已落地**条目"
        "真正冲突的才会被自动挡），`deferred_budget` 由 `--budget` 推导（只数单侧通过的条目，单侧被挡的不占预算）——"
        "预算只能解释排名在预算之外的条目，预算之内漏掉的仍然是 🔴 无交代。")
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_waivers(pairs):
    out = {}
    for item in pairs or []:
        key, sep, reason = str(item).partition("=")
        key, reason = key.strip(), reason.strip()
        if not sep or not key or not reason:
            raise SystemExit(
                "--waive must be <exec_id>=<reason> with a non-empty reason; got %r"
                % item)
        out[key] = reason
    return out


def run(topk_path, apply_path, out_md, out_json, unitside_path=None, budget=None,
        require_coverage=True, waivers=None, min_landed=None, accuracy_mode=None,
        calibration_json=None):
    result = validate(topk_path, apply_path, unitside_path, budget,
                      require_coverage=require_coverage, waivers=waivers,
                      min_landed=min_landed, accuracy_mode=accuracy_mode,
                      calibration_json=calibration_json)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w") as fh:
        fh.write(render_markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", required=True,
                        help="fusion_topk.json (schema_version >= 3, with execution_list)")
    parser.add_argument("--apply", required=True,
                        help="the fusion_integrator return json")
    parser.add_argument("--unitside", default=None,
                        help="fusion_unitside.json — lets a 单侧 fail/blocked stand as "
                             "a disposition without the integrator repeating it")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--budget", type=int, default=None,
                        help="FUSION_BUDGET actually applied, counted over 单侧-eligible "
                             "rows; excuses rows ranked after the N-th one "
                             "(recorded and named, never silent)")
    parser.add_argument("--allow-partial-coverage", action="store_true",
                        help="intermediate reports only: unaccounted / unmeasured / "
                             "stack_pending rows do not fail the status, but they are never "
                             "counted as accounted and stay listed")
    parser.add_argument("--waive", action="append", metavar="EXEC_ID=REASON", default=[])
    parser.add_argument("--min-landed", type=int, default=None,
                        help="landing target N_min; below it the report lists every entry's "
                             "stop stage (unit / wire / e2e / accuracy / stack) with evidence")
    parser.add_argument("--accuracy-mode", choices=("gate", "skip"), default="gate",
                        help="skip: the round ran no e2e accuracy test (calibration unusable); "
                             "the report states it with --calibration-json as evidence")
    parser.add_argument("--calibration-json", default="",
                        help="the round's accuracy calibration.json (evidence path)")
    args = parser.parse_args()
    result = run(args.topk, args.apply, args.out_md, args.out_json,
                 unitside_path=args.unitside, budget=args.budget,
                 require_coverage=not args.allow_partial_coverage,
                 waivers=_parse_waivers(args.waive), min_landed=args.min_landed,
                 accuracy_mode=args.accuracy_mode, calibration_json=args.calibration_json)
    cov = result.get("coverage") or {}
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "errors": len(result["errors"]),
                      "coverage": {k: cov.get(k) for k in
                                   ("execution_list_size", "accounted",
                                    "unaccounted", "unmeasured", "unmeasured_exec_ids",
                                    "stack_pending", "complete")},
                      "landed": result.get("landed"),
                      "accuracy_gate": result.get("accuracy_gate")},
                     indent=2, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
