#!/usr/bin/env python3
"""Phase 3.0: the 单侧 (isolated) gate for kernel-fusion candidates.

Deterministic validator of the per-candidate microbench VERDICTS produced by the
fusion_unit_validator role. It does NOT run kernels and claims no perf of its own —
it VALIDATES that each verdict is trustworthy (well-formed + provenance-consistent
with the Phase 2.1 candidate it claims to test) and derives the gate:

  unit_side_status ∈ {pass, fail, blocked}

  * pass    — parity==pass AND isolated_speedup > 1+margin AND (for a collective) the
              fused path actually engaged. This fusion is eligible for apply-back.
  * fail    — the microbench ran but the fusion is not a win (parity failed, or no
              speedup). Do NOT apply back.
  * blocked — a collective whose fused path did NOT engage at this shape (size-guard
              fallback to split). Not a fail — it matches the Top-K non-actionable
              verdict; there is simply nothing to apply at this shape.

A verdict that is malformed, references an unknown candidate, tests a DIFFERENT shape
than the candidate captured, or names a fused_fn that is not the candidate's existing
API is an ERROR (untrustworthy → the harness FAILS so it is re-run) -- never silently a
pass. This is the anti-cheat that keeps 单侧 honest, mirroring the provenance checks in
fusion_candidate_harness.py.

COVERAGE (recall) gate
----------------------
The checks above are all *precision*: they ask whether a SUBMITTED verdict is
trustworthy. They cannot see a candidate for which no verdict was ever submitted,
because the validator loops over verdicts, not over candidates. A run that
microbenched 16 of 42 candidates therefore rendered as
"总计 16 条 verdict：pass 16 / fail 0 ... status=pass" -- a 38%-coverage run that is
indistinguishable from a complete one. Observed on DSR1 2026-08-26: the highest
roofline decode candidate (kv_write_cluster) and all 20 prefill candidates silently
never reached the microbench, and nothing anywhere turned red.

So the denominator is now explicit. Every candidate that CITES an existing fused API
(i.e. is 单侧-testable at all) must end up in exactly one of:

  * a real verdict (pass / fail / blocked), or
  * an explicit `--waive <candidate_id>=<reason>` (status `waived`), or
  * `not_validated` → the harness FAILS.

tier-C candidates (`implementation_class: new_helper_kernel`, i.e. no existing kernel
to bench) are reported as `deferred_author` and are legitimately out of scope -- they
are counted and shown, never silently dropped. Pass `--allow-partial-coverage` to
report the gap without failing (the gap is still printed loudly either way).

Usage:
  python3 fusion_unitside_harness.py --candidates fusion_candidates.json \
      --verdicts <dir-of-*.json | combined.json> \
      --out-md FUSION_UNITSIDE.md --out-json fusion_unitside.json \
      [--min-speedup 1.0] [--allow-partial-coverage] \
      [--waive <candidate_id>=<reason> ...]
"""
import argparse
import glob
import json
import os
import sys


REQUIRED_VERDICT_FIELDS = (
    "candidate_id", "parity", "isolated_speedup", "tested_shape", "fused_fn")


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _load_verdicts(path):
    """A directory of *.json (one verdict each) OR a combined json ({verdicts:[...]}
    or a bare list)."""
    if os.path.isdir(path):
        out = []
        for fp in sorted(glob.glob(os.path.join(path, "*.json"))):
            out.append(_load(fp))
        return out
    data = _load(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("verdicts") or [data]
    return []


def _norm(text):
    """Normalize an API/fn name for a robust contains-match: lowercase, keep only
    alnum + underscore runs (drops the ' (--flag)' suffixes, '::', 'aiter ' prefix)."""
    text = str(text or "").lower()
    keep = []
    for ch in text:
        keep.append(ch if (ch.isalnum() or ch == "_") else " ")
    return [tok for tok in "".join(keep).split() if tok]


def _fn_matches_api(fused_fn, existing_apis):
    """The verdict's fused_fn must be one of the candidate's declared existing APIs
    (the microbench cannot claim a win for a kernel the candidate never cited)."""
    fn_toks = set(_norm(fused_fn))
    if not fn_toks:
        return False
    for api in existing_apis or []:
        api_toks = set(_norm(api.get("name")))
        if not api_toks:
            continue
        # the fused_fn's distinctive tokens (drop the generic 'aiter') are a subset of
        # the API name's tokens, or vice-versa — either direction is a match.
        core = fn_toks - {"aiter"}
        acore = api_toks - {"aiter"}
        if core and (core <= api_toks or acore <= fn_toks):
            return True
    return False


def _candidate_shapes(candidate):
    """The set of captured member input-dim rows (as tuples) the candidate legitimately
    operates on. A verdict must have tested ONE of these."""
    shapes = set()
    for member in candidate.get("members", []) or []:
        dims = ((member.get("shape") or {}).get("input_dims")) or []
        for row in dims:
            if isinstance(row, list) and row:
                shapes.add(tuple(int(x) for x in row))
    return shapes


def _tested_shape_tuple(tested_shape):
    """Accept [tokens, hidden] or [[..],[..]]; return the primary 2-D row as a tuple,
    or None if unparseable."""
    if not isinstance(tested_shape, list) or not tested_shape:
        return None
    first = tested_shape[0]
    row = first if isinstance(first, list) else tested_shape
    try:
        return tuple(int(x) for x in row)
    except (TypeError, ValueError):
        return None


# tier-C: the candidate has no existing fused kernel, so there is nothing to
# microbench against the split reference. Legitimately out of 单侧 scope -- but it is
# REPORTED (deferred_author), never dropped from the denominator silently.
DEFERRED_CLASSES = ("new_helper_kernel",)

ROW_FIELDS = ("candidate_id", "family", "phase", "tier_hint", "unit_side_status",
              "reason", "parity", "isolated_speedup", "ref_ms", "cand_ms",
              "engaged", "tested_shape", "fused_fn", "tp", "tol")


def _in_scope(candidate):
    """True iff this candidate is 单侧-testable: it cites an existing fused API.

    tier-C (author a new kernel) has nothing to bench -> out of scope (deferred),
    not missing. Everything else MUST produce a verdict or an explicit waiver."""
    if str(candidate.get("implementation_class") or "") in DEFERRED_CLASSES:
        return False
    return bool(candidate.get("existing_apis"))


def _row(cid, candidate, status, reason, **extra):
    """One result row with a uniform schema, so coverage rows (which have no
    measurement) render alongside verdict rows without KeyErrors."""
    row = {f: None for f in ROW_FIELDS}
    row.update({
        "candidate_id": cid,
        "family": candidate.get("family") if candidate else None,
        "phase": candidate.get("phase") if candidate else None,
        "tier_hint": candidate.get("implementation_class") if candidate else None,
        "unit_side_status": status,
        "reason": reason,
    })
    row.update(extra)
    return row


def validate(candidates_path, verdicts_path, min_speedup=1.0,
             require_coverage=True, waivers=None):
    payload = _load(candidates_path)
    candidate_list = payload.get("candidates", [])
    candidates = {c["candidate_id"]: c for c in candidate_list}
    verdicts = _load_verdicts(verdicts_path)
    waivers = dict(waivers or {})

    errors = []
    results = []
    # Every candidate_id a verdict was SUBMITTED for -- including ones whose
    # verdict turned out untrustworthy. An untrustworthy verdict is a loud error
    # already; it must not ALSO be reported as a silent coverage gap.
    submitted = set()
    for index, verdict in enumerate(verdicts):
        vp = "verdict[%d]" % index
        cid = verdict.get("candidate_id")
        if cid:
            submitted.add(cid)
        # 1. well-formed
        missing = [f for f in REQUIRED_VERDICT_FIELDS if verdict.get(f) is None]
        if missing:
            errors.append("%s missing fields %s" % (vp, missing))
            continue
        vp = "verdict[%s]" % cid
        # 2. known candidate (provenance)
        candidate = candidates.get(cid)
        if candidate is None:
            errors.append("%s references unknown candidate_id" % vp)
            continue
        family = str(candidate.get("family") or "")
        # 3. fused_fn is one of the candidate's declared APIs (anti-cheat)
        if not _fn_matches_api(verdict.get("fused_fn"),
                               candidate.get("existing_apis")):
            errors.append(
                "%s fused_fn '%s' is not one of the candidate's existing_apis %s"
                % (vp, verdict.get("fused_fn"),
                   [a.get("name") for a in candidate.get("existing_apis", [])]))
            continue
        # 4. shape provenance — the microbench must have tested the shape the candidate
        #    actually ran on (a pass on a different shape is meaningless). Two cases:
        #    (a) members carry exact captured input_dims (e.g. prefill kernel_exact):
        #        tested_shape MUST be one of them (strict).
        #    (b) members carry no dims (e.g. decode runtime_probe_wrapper): fall back to
        #        the selected_bucket token count — tested_shape's leading (token) dim
        #        MUST equal batch_size (decode) / input_tokens (prefill). This still
        #        stops a pass faked on a different token count.
        tested = _tested_shape_tuple(verdict.get("tested_shape"))
        cand_shapes = _candidate_shapes(candidate)
        if tested is None:
            errors.append("%s tested_shape %s is unparseable"
                          % (vp, verdict.get("tested_shape")))
            continue
        if cand_shapes:
            if tested not in cand_shapes:
                errors.append(
                    "%s tested_shape %s is not among the candidate's captured member "
                    "shapes %s (cannot trust a 单侧 pass on a different shape)"
                    % (vp, verdict.get("tested_shape"), sorted(cand_shapes)))
                continue
        else:
            bucket = candidate.get("selected_bucket") or {}
            expect_tok = (bucket.get("batch_size")
                          if candidate.get("phase") == "decode"
                          else bucket.get("input_tokens"))
            if expect_tok and int(tested[0]) != int(expect_tok):
                errors.append(
                    "%s tested_shape leading dim %d != selected_bucket token count %d "
                    "(cannot trust a 单侧 pass on a different token count)"
                    % (vp, tested[0], int(expect_tok)))
                continue
        # ---- verdict is TRUSTWORTHY; derive the gate from its content ------------
        try:
            speedup = float(verdict.get("isolated_speedup"))
        except (TypeError, ValueError):
            errors.append("%s isolated_speedup is not a number" % vp)
            continue
        parity = str(verdict.get("parity")).lower()
        is_collective = family.startswith("collective")
        engaged = verdict.get("engaged")
        status, reason = "pass", ""
        if is_collective and engaged is False:
            status = "blocked"
            reason = ("fused collective path did not engage at this shape "
                      "(size-guard fallback to split) — nothing to apply here")
        elif parity != "pass":
            status = "fail"
            reason = "parity != pass (fused output diverges from the split reference)"
        elif speedup <= min_speedup:
            status = "fail"
            reason = ("isolated_speedup %.3f <= %.3f — not a win, do not apply back"
                      % (speedup, min_speedup))
        else:
            reason = ("parity pass, isolated_speedup %.3fx%s"
                      % (speedup, ", engaged" if is_collective else ""))
        results.append({
            "candidate_id": cid,
            "family": family,
            "phase": candidate.get("phase"),
            "tier_hint": candidate.get("implementation_class"),
            "unit_side_status": status,
            "reason": reason,
            "parity": parity,
            "isolated_speedup": round(speedup, 4),
            "ref_ms": verdict.get("ref_ms"),
            "cand_ms": verdict.get("cand_ms"),
            "engaged": engaged,
            "tested_shape": verdict.get("tested_shape"),
            "fused_fn": verdict.get("fused_fn"),
            "tp": verdict.get("tp"),
            "tol": verdict.get("tol"),
        })

    # ---- COVERAGE (recall): close the loop over CANDIDATES, not verdicts ------
    # Up to here the loop was verdict-driven, so a candidate nobody benched was
    # invisible. Walk the candidate list and give every in-scope candidate an
    # explicit disposition.
    coverage_rows = []
    unvalidated = []
    waived_unknown = sorted(set(waivers) - set(candidates))
    for candidate in candidate_list:
        cid = candidate.get("candidate_id")
        if not cid or cid in submitted:
            continue
        if not _in_scope(candidate):
            coverage_rows.append(_row(
                cid, candidate, "deferred_author",
                "no existing fused kernel (tier-C, 需自写) — out of 单侧 scope, "
                "counted but not benched"))
            continue
        if cid in waivers:
            coverage_rows.append(_row(
                cid, candidate, "waived",
                "explicitly waived: %s" % waivers[cid]))
            continue
        coverage_rows.append(_row(
            cid, candidate, "not_validated",
            "NO verdict submitted — this candidate never reached the microbench"))
        unvalidated.append(cid)
    results.extend(coverage_rows)
    for cid in waived_unknown:
        errors.append("--waive references unknown candidate_id '%s'" % cid)

    in_scope = [c for c in candidate_list if _in_scope(c)]
    by_phase = {}
    for candidate in in_scope:
        ph = str(candidate.get("phase") or "?")
        slot = by_phase.setdefault(ph, {"in_scope": 0, "validated": 0,
                                        "waived": 0, "not_validated": 0})
        slot["in_scope"] += 1
        cid = candidate.get("candidate_id")
        if cid in submitted:
            slot["validated"] += 1
        elif cid in waivers:
            slot["waived"] += 1
        else:
            slot["not_validated"] += 1
    coverage = {
        "required": require_coverage,
        "in_scope": len(in_scope),
        "validated": sum(1 for c in in_scope if c.get("candidate_id") in submitted),
        "waived": sum(1 for c in in_scope
                      if c.get("candidate_id") not in submitted
                      and c.get("candidate_id") in waivers),
        "not_validated": len(unvalidated),
        "not_validated_ids": unvalidated,
        "deferred_author": sum(1 for c in candidate_list if not _in_scope(c)),
        "candidates_total": len(candidate_list),
        "by_phase": by_phase,
        "complete": not unvalidated,
    }

    counts = {"pass": 0, "fail": 0, "blocked": 0,
              "not_validated": 0, "waived": 0, "deferred_author": 0}
    for r in results:
        counts[r["unit_side_status"]] = counts.get(r["unit_side_status"], 0) + 1

    coverage_fail = bool(require_coverage and unvalidated)
    result = {
        "schema_version": 2,
        "phase": "unitside_gate",
        "status": "fail" if (errors or coverage_fail) else "pass",
        "min_speedup": min_speedup,
        "candidates_json": os.path.abspath(candidates_path),
        "errors": errors,
        "coverage": coverage,
        "coverage_ok": not coverage_fail,
        "counts": counts,
        "verdict_count": len(verdicts),
        "results": results,
    }
    return result


def _esc(value):
    return str(value if value is not None else "").replace("|", "\\|")


def render_markdown(result):
    lines = ["# Kernel Fusion 单侧 Gate (Phase 3.0)", ""]
    lines.append(
        "隔离验证每个 fusion 的 **fused kernel vs split 参考**：正确性(parity) + "
        "isolated speedup。**`pass` 才可进 apply-back**；`blocked` = 该 shape 下 "
        "fused 路径未生效(size-guard 回退)，非失败；`fail` = 无收益/不正确。")
    lines.append("")
    c = result["counts"]
    cov = result.get("coverage") or {}
    # Coverage FIRST: the denominator is the headline. A report that leads with
    # "16 verdict, 16 pass" reads as complete at 38% coverage -- that is exactly the
    # failure this section exists to make impossible.
    if cov:
        lines.append(
            "**覆盖率**：在范围内候选 **%d** / 已验证 **%d** / 未验证 **%d** / 豁免 %d"
            "；tier-C 待自写 %d（不计入范围）；候选总数 %d。"
            % (cov.get("in_scope", 0), cov.get("validated", 0),
               cov.get("not_validated", 0), cov.get("waived", 0),
               cov.get("deferred_author", 0), cov.get("candidates_total", 0)))
        by_phase = cov.get("by_phase") or {}
        if by_phase:
            lines.append("")
            lines.append("分阶段覆盖：" + "；".join(
                "%s **%d/%d**" % (ph, v.get("validated", 0) + v.get("waived", 0),
                                  v.get("in_scope", 0))
                for ph, v in sorted(by_phase.items())) + "。")
        lines.append("")
        if cov.get("not_validated"):
            lines.append(
                "> 🔴 **覆盖率不完整**：%d 条在范围内的候选从未提交 verdict。"
                "单侧结论对它们**没有任何结论**——不是 pass，也不是 fail，是没测。"
                "补测或用 `--waive <id>=<理由>` 显式豁免。"
                % cov.get("not_validated", 0))
            lines.append("")
    lines.append("总计 %d 条 verdict：pass %d / fail %d / blocked %d；harness status=**%s**（错误 %d）。"
                 % (result["verdict_count"], c.get("pass", 0), c.get("fail", 0),
                    c.get("blocked", 0), result["status"], len(result["errors"])))
    lines.append("")
    lines.append("| 候选 | 阶段 | family | 单侧结论 | parity | isolated speedup | engaged | tested shape | fused fn | 说明 |")
    lines.append("|---|:--:|---|:--:|:--:|---:|:--:|---|---|---|")
    for r in result["results"]:
        sp = ("%.3fx" % r["isolated_speedup"]) if r["isolated_speedup"] is not None else "-"
        lines.append("| `%s` | %s | %s | **%s** | %s | %s | %s | %s | `%s` | %s |" % (
            _esc(r["candidate_id"]), _esc(r["phase"]), _esc(r["family"]),
            r["unit_side_status"], _esc(r["parity"]), sp, _esc(r["engaged"]),
            _esc(r["tested_shape"]), _esc(r["fused_fn"]), _esc(r["reason"])))
    lines.append("")
    gaps = [r for r in result["results"]
            if r.get("unit_side_status") == "not_validated"]
    if gaps:
        lines.append("## 覆盖率缺口：未验证候选（必须补 verdict 或显式豁免）")
        lines.append("")
        lines.append("| 候选 | 阶段 | family | tier | 说明 |")
        lines.append("|---|:--:|---|---|---|")
        for r in gaps:
            lines.append("| `%s` | %s | %s | %s | %s |" % (
                _esc(r["candidate_id"]), _esc(r["phase"]), _esc(r["family"]),
                _esc(r["tier_hint"]), _esc(r["reason"])))
        lines.append("")
    if result["errors"]:
        lines.append("## 不可信 verdict（harness 错误，需按报错重跑，不得当作 pass）")
        lines.append("")
        for e in result["errors"]:
            lines.append("- %s" % e)
        lines.append("")
    lines.append(
        "说明：本 gate 只校验 microbench verdict 的**可信度**（字段完整 + 候选存在 + "
        "tested_shape 属于候选抓到的 member shape + fused_fn 属于候选 existing_apis + "
        "collective 是否真生效）并据此判 pass/fail/blocked；它不跑 kernel、不产生自己的 "
        "perf 数字。isolated speedup / parity 由 fusion_unit_validator 的隔离 microbench 实测。"
        "**并且**校验覆盖率：每条在范围内的候选都必须有 verdict 或显式豁免，"
        "否则 `not_validated` 且 harness fail——「没测」不得渲染成「测过了」。")
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_waivers(pairs):
    """--waive dc_kv=needs paged-KV state -> {"dc_kv": "needs paged-KV state"}.
    A waiver without a reason is rejected: "skipped" is not a reason."""
    out = {}
    for item in pairs or []:
        cid, sep, reason = str(item).partition("=")
        cid, reason = cid.strip(), reason.strip()
        if not sep or not cid or not reason:
            raise SystemExit(
                "--waive must be <candidate_id>=<reason> with a non-empty reason; "
                "got %r" % item)
        out[cid] = reason
    return out


def run(candidates_path, verdicts_path, out_md, out_json, min_speedup=1.0,
        require_coverage=True, waivers=None):
    result = validate(candidates_path, verdicts_path, min_speedup,
                      require_coverage=require_coverage, waivers=waivers)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with open(out_md, "w") as fh:
        fh.write(render_markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--verdicts", required=True,
                        help="dir of per-candidate *.json verdicts OR a combined json")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--min-speedup", type=float, default=1.0,
                        help="isolated_speedup must exceed this for a pass (default 1.0)")
    parser.add_argument("--allow-partial-coverage", action="store_true",
                        help="report the coverage gap without failing the gate "
                             "(the gap is printed loudly either way)")
    parser.add_argument("--waive", action="append", metavar="ID=REASON", default=[],
                        help="explicitly waive one candidate from the coverage "
                             "requirement, with a reason (repeatable)")
    args = parser.parse_args()
    result = run(args.candidates, args.verdicts, args.out_md, args.out_json,
                 args.min_speedup,
                 require_coverage=not args.allow_partial_coverage,
                 waivers=_parse_waivers(args.waive))
    cov = result.get("coverage") or {}
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "errors": len(result["errors"]),
                      "coverage": {k: cov.get(k) for k in
                                   ("in_scope", "validated", "not_validated",
                                    "waived", "deferred_author", "complete")}},
                     indent=2))
    if result["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
