#!/usr/bin/env python3
"""Greedy stacking for KernelFusion (GEAK_structure 二.10.4) as a deterministic state machine.

WHY GREEDY, NOT "STACK EVERYTHING THEN DROP"
-------------------------------------------
A fusion whose expected end-to-end gain is below the measured noise band cannot be
resolved by a solo A/B; it is routed to stacking (`stack_pending`). Stacking every pending
fusion at once and then dropping members on failure confounds them: one harmful member
hides several useful ones, and the drop order is a guess. Greedy stacking measures each
candidate's MARGINAL effect on the current stack instead:

    S0 = the accepted stack after solo apply-back
    for c in candidates ordered by expected e2e gain (descending):
        A/B (S + c) vs S, plus the calibrated accuracy decision
        keep c   iff delta >= -noise_band/2 and accuracy passes   (S <- S + c)
        drop c   otherwise, and record it as stack_harmful
    confirm: one A/B of the final S vs the KernelFusion entry stack; land S only on pass

The keep bar is "does not regress beyond half the noise band", not "shows a gain": each
member is too small to show alone; the confirmation A/B is where the stack has to clear
the band as a whole.

Candidates are `stack_pending` rows from solo apply-back plus `candidate_pool` items whose
unit-side test passed. The state lives in one JSON file so the workflow (which has no
shell) can drive it one subcommand at a time through a script-runner agent, and so a
retried or resumed call never re-measures a decided step.

Subcommands (each prints ONE JSON object on its last stdout line):

  init     --state S --input init.json [--reset]
           init.json: {candidates: [...], noise_band_pct, entry: {overlay, flags, env, tput},
                       base: {overlay, flags, env, tput, members: [...]}, max_retries}
           An existing state is returned unchanged unless --reset (idempotent re-runs).
  next     --state S [--budget-left-s X --step-cost-s Y]
           -> {action: step, candidate, stack, entry, step, attempt}
            | {action: confirm, stack, entry, kept}
            | {action: done}
           With a budget: when what is left cannot pay for one more step plus the
           confirmation, the remaining candidates are marked skipped_budget (they keep
           their overlay and unit results for the next round) and the machine moves to
           the confirmation; when it cannot pay even for the confirmation, the
           confirmation is skipped and nothing new lands.
  record   --state S --input result.json [--confirm]
           step result:    {id, measured, delta_pct, accuracy_pass, stack_overlay,
                            stack_flags, stack_env, cand_tok_s, ref_tok_s, reason_code,
                            reason, evidence}
           confirm result: {measured, ab_decision, delta_pct, nonoverlap, accuracy_pass,
                            final_overlay, cand_tok_s, reason, evidence}
           Not measured (GPU busy / not finished) is never a verdict: the same step is
           returned by `next` again, up to max_retries, then the candidate is unmeasured.
  summary  --state S
           -> rows the workflow merges into its apply-back aggregate:
              accepted (landed members from the execution list), rejected (stack_harmful /
              stack_reject rows, each with stack_result), pool (candidate_pool outcomes),
              kept / harmful / unmeasured / skipped_budget id lists, confirm, landed,
              final_stack.
"""
import argparse
import json
import os
import sys
import tempfile

SCHEMA_VERSION = 1
UNIT_PASS = {"pass", "equivalent_pass", "subsumed_pass"}
SOURCE_ORDER = {"stack_pending": 0, "candidate_pool": 1}


def _num(value):
    try:
        if value is None or value == "" or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _save(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".greedy.", dir=os.path.dirname(os.path.abspath(path)))
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _cand_id(c):
    return str(c.get("id") or c.get("exec_id") or c.get("candidate_id") or "").strip()


def _expected(c):
    for key in ("expected_gain_pct", "expected_e2e_gain_pct"):
        v = _num(c.get(key))
        if v is not None:
            return v
    row = c.get("row") or {}
    for key in ("expected_gain_pct", "expected_e2e_gain_pct"):
        v = _num(row.get(key))
        if v is not None:
            return v
    return None


def build_queue(candidates):
    """Deterministic candidate order: expected gain desc (unknown last), stack_pending
    before candidate_pool on ties, then id. Pool items need a passing unit test; an id
    that appears twice keeps its stack_pending form (it has measured serving evidence)."""
    by_id = {}
    for raw in candidates or []:
        if not isinstance(raw, dict):
            continue
        cid = _cand_id(raw)
        if not cid:
            continue
        source = str(raw.get("source") or "stack_pending")
        if source == "candidate_pool":
            status = str(raw.get("unit_status") or "")
            if status not in UNIT_PASS:
                continue
        item = {
            "id": cid,
            "exec_id": str(raw.get("exec_id") or cid),
            "source": source,
            "expected_gain_pct": _expected(raw),
            "overlay_path": raw.get("overlay_path") or (raw.get("row") or {}).get("overlay_path") or "",
            "unit_status": raw.get("unit_status") or "",
            "row": raw.get("row") or {},
        }
        prev = by_id.get(cid)
        if prev is None or SOURCE_ORDER.get(source, 9) < SOURCE_ORDER.get(prev["source"], 9):
            by_id[cid] = item

    def key(item):
        gain = item["expected_gain_pct"]
        return (0 if gain is not None else 1, -(gain or 0.0),
                SOURCE_ORDER.get(item["source"], 9), item["id"])

    return sorted(by_id.values(), key=key)


def _stack_view(stack):
    return {k: stack.get(k) for k in ("overlay", "flags", "env", "tput", "members")}


def cmd_init(a):
    if os.path.exists(a.state) and not a.reset:
        state = _load(a.state)
        return 0, {"ok": True, "action": "init", "resumed": True,
                   "queue": [q["id"] for q in state["queue"]], "state": a.state}
    spec = _load(a.input)
    noise = _num(spec.get("noise_band_pct"))
    if noise is None or noise < 0:
        return 1, {"ok": False, "reason": "noise_band_pct is required (the Setup measurement)"}
    base = dict(spec.get("base") or {})
    base["members"] = [str(m) for m in base.get("members") or []]
    entry = dict(spec.get("entry") or {})
    queue = build_queue(spec.get("candidates"))
    state = {
        "schema_version": SCHEMA_VERSION,
        "noise_band_pct": noise,
        "keep_threshold_pct": -noise / 2.0,
        "max_retries": int(spec.get("max_retries") if spec.get("max_retries") is not None else 2),
        "entry": entry,
        "base": base,
        "stack": dict(base),
        "queue": queue,
        "cursor": 0,
        "attempts": {},
        "results": {},
        "confirm": None,
        "confirm_attempts": 0,
        "stopped": None,
        "log": [],
    }
    _save(a.state, state)
    return 0, {"ok": True, "action": "init", "resumed": False, "state": a.state,
               "queue": [q["id"] for q in queue], "noise_band_pct": noise,
               "keep_threshold_pct": state["keep_threshold_pct"]}


def _kept(state):
    return [q["id"] for q in state["queue"]
            if (state["results"].get(q["id"]) or {}).get("decision") == "keep"]


def _confirm_or_done(state):
    confirm = state.get("confirm")
    if confirm is None:
        if not _kept(state):
            state["confirm"] = {"status": "skipped",
                                "reason": "no stacked candidate was kept; nothing new to land"}
            return {"action": "done"}
        state["confirm"] = {"status": "pending"}
    if state["confirm"].get("status") == "pending":
        return {"action": "confirm", "stack": _stack_view(state["stack"]),
                "entry": state["entry"], "kept": _kept(state),
                "attempt": state.get("confirm_attempts", 0) + 1}
    return {"action": "done"}


def cmd_next(a):
    state = _load(a.state)
    left = _num(a.budget_left_s)
    cost = _num(a.step_cost_s)
    remaining = state["queue"][state["cursor"]:]
    if remaining and state.get("stopped") is None and left is not None and cost is not None \
            and left < 2.0 * cost:
        for item in remaining:
            state["results"][item["id"]] = {
                "decision": "skipped_budget",
                "reason": ("KernelFusion budget left %.0fs < one stacking step plus the "
                           "confirmation (2 x %.0fs); kept for the next round with its "
                           "overlay and unit results" % (left, cost))}
        state["cursor"] = len(state["queue"])
        state["stopped"] = "budget"
        state["log"].append({"event": "stopped_budget", "left_s": left, "step_cost_s": cost})
    if state["cursor"] < len(state["queue"]):
        item = state["queue"][state["cursor"]]
        out = {"action": "step", "candidate": item, "stack": _stack_view(state["stack"]),
               "entry": state["entry"], "step": state["cursor"] + 1,
               "attempt": state["attempts"].get(item["id"], 0) + 1,
               "keep_threshold_pct": state["keep_threshold_pct"],
               "noise_band_pct": state["noise_band_pct"]}
        _save(a.state, state)
        return 0, dict(ok=True, **out)
    if (state.get("confirm") is None and _kept(state) and left is not None and cost is not None
            and left < cost):
        state["confirm"] = {"status": "skipped_budget",
                            "reason": ("KernelFusion budget left %.0fs < one confirmation A/B "
                                       "(%.0fs); the kept stack does not land" % (left, cost))}
    out = _confirm_or_done(state)
    _save(a.state, state)
    return 0, dict(ok=True, **out)


def _accuracy_pass(res):
    """accuracy_pass of one measurement. A result whose e2e accuracy test was skipped for the
    round (accuracy.mode == 'skipped': the calibration found no usable test) passes on
    unit-side numerical equivalence."""
    acc = res.get("accuracy_pass")
    if acc is None and str((res.get("accuracy") or {}).get("mode") or "").lower() == "skipped":
        return True
    return acc


def _record_step(state, res):
    if state["cursor"] >= len(state["queue"]):
        return 1, {"ok": False, "reason": "no stacking step is in flight"}
    item = state["queue"][state["cursor"]]
    rid = str(res.get("id") or res.get("exec_id") or "")
    if rid and rid != item["id"]:
        return 1, {"ok": False, "reason": "result for %r but the step in flight is %r"
                   % (rid, item["id"])}
    delta = _num(res.get("delta_pct"))
    acc = _accuracy_pass(res)
    measured = res.get("measured") is not False and delta is not None and acc in (True, False)
    if measured and acc is True and delta >= state["keep_threshold_pct"] \
            and not res.get("stack_overlay"):
        # A keep with no stacked overlay cannot be the base of the next step.
        measured = False
        res = dict(res, reason_code="not_finished",
                   reason="kept but no stack_overlay returned: " + str(res.get("reason") or ""))
    if not measured:
        n = state["attempts"].get(item["id"], 0) + 1
        state["attempts"][item["id"]] = n
        if n > state["max_retries"]:
            state["results"][item["id"]] = {
                "decision": "unmeasured", "attempts": n,
                "reason_code": res.get("reason_code") or "not_finished",
                "reason": "still unmeasured after %d attempt(s): %s" % (n, res.get("reason") or "")}
            state["cursor"] += 1
            return 0, {"ok": True, "decision": "unmeasured", "id": item["id"]}
        return 0, {"ok": True, "decision": "retry", "id": item["id"], "attempts": n,
                   "reason_code": res.get("reason_code") or "not_finished"}
    keep = acc is True and delta >= state["keep_threshold_pct"]
    result = {"decision": "keep" if keep else "drop", "delta_pct": round(delta, 4),
              "accuracy_pass": acc, "step": state["cursor"] + 1,
              "stack_before": _stack_view(state["stack"]),
              "ref_tok_s": res.get("ref_tok_s"), "cand_tok_s": res.get("cand_tok_s"),
              "evidence": res.get("evidence") or "", "attempts": state["attempts"].get(item["id"], 0) + 1}
    if keep:
        result["reason"] = ("marginal delta %.3f%% >= keep threshold %.3f%% and accuracy passed"
                            % (delta, state["keep_threshold_pct"]))
        stack = state["stack"]
        stack["members"] = list(stack.get("members") or []) + [item["id"]]
        stack["overlay"] = res.get("stack_overlay")
        if res.get("stack_flags") is not None:
            stack["flags"] = res.get("stack_flags")
        if res.get("stack_env") is not None:
            stack["env"] = res.get("stack_env")
        cand = res.get("cand_tok_s")
        cand = _num(cand if not isinstance(cand, list) else (sorted(cand)[len(cand) // 2] if cand else None))
        if cand is not None:
            stack["tput"] = cand
    else:
        why = []
        if acc is not True:
            why.append("accuracy failed")
        if delta < state["keep_threshold_pct"]:
            why.append("marginal delta %.3f%% < keep threshold %.3f%%"
                       % (delta, state["keep_threshold_pct"]))
        result["reason"] = "stack_harmful: " + "; ".join(why)
        result["stack_harmful"] = True
    state["results"][item["id"]] = result
    state["cursor"] += 1
    return 0, {"ok": True, "decision": result["decision"], "id": item["id"],
               "delta_pct": result["delta_pct"], "stack": _stack_view(state["stack"])}


def _record_confirm(state, res):
    confirm = state.get("confirm") or {}
    if confirm.get("status") != "pending":
        return 1, {"ok": False, "reason": "no confirmation A/B is in flight"}
    delta = _num(res.get("delta_pct"))
    acc = _accuracy_pass(res)
    measured = res.get("measured") is not False and delta is not None and acc in (True, False)
    if not measured:
        state["confirm_attempts"] = state.get("confirm_attempts", 0) + 1
        if state["confirm_attempts"] > state["max_retries"]:
            state["confirm"] = {"status": "unmeasured",
                                "reason": "confirmation never measured: %s" % (res.get("reason") or "")}
            return 0, {"ok": True, "decision": "unmeasured"}
        return 0, {"ok": True, "decision": "retry", "attempts": state["confirm_attempts"]}
    decision = str(res.get("ab_decision") or "").lower()
    if decision:
        perf = decision == "accept"
    else:
        perf = delta > state["noise_band_pct"] and res.get("nonoverlap") is True
    passed = perf and acc is True
    state["confirm"] = {
        "status": "pass" if passed else "fail", "delta_pct": round(delta, 4),
        "ab_decision": decision or None, "nonoverlap": res.get("nonoverlap"),
        "accuracy_pass": acc, "final_overlay": res.get("final_overlay") or state["stack"].get("overlay"),
        "cand_tok_s": res.get("cand_tok_s"), "evidence": res.get("evidence") or "",
        "reason": res.get("reason") or (
            "final stack vs KernelFusion entry: delta %.3f%%, %s, accuracy %s"
            % (delta, "A/B accept" if perf else "A/B did not clear the noise band",
               "pass" if acc else "fail"))}
    return 0, {"ok": True, "decision": state["confirm"]["status"], "delta_pct": round(delta, 4)}


def cmd_record(a):
    state = _load(a.state)
    res = _load(a.input)
    code, out = (_record_confirm if a.confirm else _record_step)(state, res)
    if out.get("ok"):
        state["log"].append({"event": "confirm" if a.confirm else "step", "result": out})
        _save(a.state, state)
    return code, out


def summarise(state):
    confirm = state.get("confirm") or {"status": "not_run"}
    landed = confirm.get("status") == "pass"
    accepted, rejected, pool = [], [], []
    kept, harmful, unmeasured, skipped = [], [], [], []
    for item in state["queue"]:
        res = state["results"].get(item["id"]) or {"decision": "not_reached"}
        decision = res.get("decision")
        row = dict(item.get("row") or {})
        row.setdefault("exec_id", item["exec_id"])
        stack_result = {k: res.get(k) for k in ("decision", "delta_pct", "accuracy_pass", "step",
                                                "reason", "evidence", "ref_tok_s", "cand_tok_s")
                        if res.get(k) is not None}
        stack_result["confirm"] = {k: confirm.get(k) for k in ("status", "delta_pct", "reason")
                                   if confirm.get(k) is not None}
        if decision == "keep":
            kept.append(item["id"])
        elif decision == "drop":
            harmful.append(item["id"])
        elif decision == "skipped_budget":
            skipped.append(item["id"])
        else:
            unmeasured.append(item["id"])
        if item["source"] == "candidate_pool":
            pool.append({"id": item["id"], "exec_id": item["exec_id"],
                         "expected_gain_pct": item["expected_gain_pct"],
                         "status": ("landed" if decision == "keep" and landed else
                                    ("stack_harmful" if decision == "drop" else
                                     ("kept_not_landed" if decision == "keep" else decision))),
                         "stack_result": stack_result, "row": row})
            continue
        if decision == "keep" and landed:
            row.pop("blocked_evidence", None)
            accepted.append(dict(row, measured=True, stacked=True, stack_pending=False,
                                 stacked_with=[m for m in state["stack"].get("members") or []
                                               if m != item["id"]],
                                 single_delta_pct=row.get("throughput_delta_pct"),
                                 throughput_delta_pct=res.get("delta_pct"),
                                 stack_result=stack_result,
                                 reason="kept by greedy stacking; the final stack passed the "
                                        "confirmation A/B"))
        elif decision == "keep":
            rejected.append(dict(row, measured=True, stack_pending=True,
                                 blocked_evidence="stack_reject", stack_result=stack_result,
                                 reason="kept by greedy stacking but the final stack did not "
                                        "land (confirmation %s)" % confirm.get("status")))
        elif decision == "drop":
            rejected.append(dict(row, measured=True, stack_pending=True, stack_harmful=True,
                                 blocked_evidence="stack_harmful", stack_result=stack_result,
                                 reason=res.get("reason") or "stack_harmful"))
        # unmeasured / skipped_budget rows stay stack_pending without stack_result: the
        # apply-back gate reports them as never re-gated, which is what they are.
    return {
        "ok": True, "action": "summary",
        "noise_band_pct": state["noise_band_pct"],
        "keep_threshold_pct": state["keep_threshold_pct"],
        "order": [q["id"] for q in state["queue"]],
        "kept": kept, "harmful": harmful, "unmeasured": unmeasured,
        "skipped_budget": skipped, "stopped": state.get("stopped"),
        "confirm": confirm, "landed": landed,
        "final_stack": _stack_view(state["stack"]) if landed else _stack_view(state["base"]),
        "accepted": accepted, "rejected": rejected, "pool": pool,
    }


def cmd_summary(a):
    return 0, summarise(_load(a.state))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--state", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--reset", action="store_true")
    p = sub.add_parser("next")
    p.add_argument("--state", required=True)
    p.add_argument("--budget-left-s", default=None)
    p.add_argument("--step-cost-s", default=None)
    p = sub.add_parser("record")
    p.add_argument("--state", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--confirm", action="store_true")
    p = sub.add_parser("summary")
    p.add_argument("--state", required=True)
    a = ap.parse_args(argv)
    handler = {"init": cmd_init, "next": cmd_next, "record": cmd_record,
               "summary": cmd_summary}[a.cmd]
    try:
        code, out = handler(a)
    except (OSError, ValueError, KeyError) as exc:
        code, out = 1, {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}
    print(json.dumps(out, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
