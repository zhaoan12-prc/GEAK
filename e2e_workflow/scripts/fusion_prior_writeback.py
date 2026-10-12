#!/usr/bin/env python3
"""Write KernelFusion results back to the measured-prior library (GEAK_structure 二.12).

At the end of KernelFusion every execution item's measured outcome is appended to
knowledge/fusion/measured_priors.json so the next round ranks with it:

  accepted (solo) / kept in a landed greedy stack   -> POSITIVE  (measured e2e gain)
  measured e2e regression / stack_harmful            -> NEGATIVE  (measured degradation)
  unit_fail (and other implementation-level failures: accuracy_fail, wire_failure,
  kernel_unavailable, not_engaged, unit_parity_fail)
                                                     -> recorded against THAT implementation
                                                        in the sidecar; the seam is NOT marked
                                                        negative, because another implementation
                                                        of the same seam may still win
  wire_failure (not engaged / not captured by the CUDA graph / overlay not loaded, still
  failing after one diagnosed re-measure)            -> implementation record AND an untested
                                                        candidate with priority "highest"
  never measured because the budget ran out (deferred_budget, stack_pending never
  re-gated, skipped_budget)                          -> sidecar `untested_candidates`, with the
                                                        overlay and unit results, as next-round
                                                        candidate input
  in-noise / unmeasured / stack_reject               -> nothing (no evidence either way)

The library is APPEND-ONLY: a seam measured several times keeps every measurement (the
ranking side takes the median of the latest three). Appends go through
fusion_priors.append_measurement(path, prior_id_or_seam_record, measurement, polarity)
when that API exists; otherwise this script appends the same record itself in schema v2:

  {"schema_version": 2, "priors": [{"prior_id", "title", "model_family", "model", "gfx",
    "framework", "polarity", "seam", "measurements": [{"gain_pct", "date", "protocol",
    "source", "evidence"}]}]}

`--dry-run` prints what would be appended and writes nothing.

Usage:
  python3 fusion_prior_writeback.py --apply apply_result.json [--topk fusion_topk.json] \
      --model MODEL --gfx gfx942 --framework sglang --protocol ab_interleaved:isl=..,osl=.. \
      [--noise-band-pct 1.0] [--source EVAL_DIR] [--priors PATH] [--sidecar PATH] [--dry-run]
"""
import argparse
import datetime
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEFAULT_PRIORS = os.path.abspath(os.path.join(HERE, "..", "knowledge", "fusion",
                                              "measured_priors.json"))

# Measured evidence that is about one implementation, not about the seam.
IMPLEMENTATION_EVIDENCE = {"unit_fail", "unit_parity_fail", "accuracy_fail", "wire_failure",
                           "kernel_unavailable", "not_engaged"}
REGRESSION_EVIDENCE = {"e2e_regression"}
SOFT_REJECT_EVIDENCE = {"ab_reject", "e2e_no_gain"}


def _num(value):
    try:
        if value is None or value == "" or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def parse_protocol(text):
    """`name:k=v,k=v` (or a JSON object) -> the measurement's protocol object."""
    text = str(text or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            doc = json.loads(text)
            return doc if isinstance(doc, dict) else {"name": text}
        except ValueError:
            return {"name": text}
    name, _, rest = text.partition(":")
    out = {"name": name}
    for part in rest.split(","):
        key, eq, value = part.partition("=")
        if not eq or not key.strip():
            continue
        value = value.strip()
        try:
            out[key.strip()] = int(value)
        except ValueError:
            try:
                out[key.strip()] = float(value)
            except ValueError:
                out[key.strip()] = value
    return out


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")[:80]


def _model_family(model):
    try:
        import fusion_priors
        fam = fusion_priors.model_family(model)
        if fam:
            return fam
    except (ImportError, AttributeError):
        pass
    match = re.search(r"([a-z]+)[-_ ]?(\d+(?:[._]\d+)?)?", str(model or "").lower())
    if not match:
        return ""
    return match.group(1) + re.sub(r"[^0-9]", "", match.group(2) or "")


def _index_board(topk):
    """exec_id -> entry, and (exec_id, rung) -> rung item, over execution_list + candidate_pool."""
    entries, rungs = {}, {}
    for field in ("execution_list", "candidate_pool"):
        for entry in (topk or {}).get(field) or []:
            if not isinstance(entry, dict):
                continue
            eid = str(entry.get("exec_id") or entry.get("candidate_id") or "")
            if not eid:
                continue
            entries.setdefault(eid, entry)
            for item in entry.get("alt_ladder") or []:
                if isinstance(item, dict) and item.get("rung") is not None:
                    rungs[(eid, int(item["rung"]))] = item
    return entries, rungs


def seam_record(row, entries, rungs, ctx):
    """The seam a measurement belongs to: the board entry, or the alt-ladder rung it tested.

    A board entry names the measured prior it IS (`matched_prior_id`: the prior that forced
    it, or an exact measured-prior seam match), so a re-measured known seam is appended to
    that prior's history; otherwise a new prior is created from the entry's
    `seam_descriptor` / `seam_members` (the screen's matching vocabulary)."""
    eid = str(row.get("exec_id") or row.get("id") or row.get("candidate_id") or "")
    src = dict(entries.get(eid) or {})
    rung = row.get("rung")
    if rung is not None:
        try:
            item = rungs.get((eid, int(rung)))
        except (TypeError, ValueError):
            item = None
        if item and int(rung) != 0:
            # A deeper rung is a different window: none of the head entry's seam identity applies.
            for key in ("seam_descriptor", "seam_members", "matched_prior_id", "seam_key",
                        "forced_by_prior", "prior_polarity"):
                src.pop(key, None)
            src.update(item)
    for key in ("seam_key", "kind", "kernel_names", "op_tags", "title", "matched_prior_id",
                "seam_descriptor", "seam_members"):
        if src.get(key) in (None, "", [], {}) and row.get(key) not in (None, "", [], {}):
            src[key] = row.get(key)
    kernel_names = [str(k) for k in src.get("kernel_names") or []]
    seam_key = str(src.get("seam_key") or "")
    if not seam_key:
        seam_key = "%s:%s" % (src.get("kind") or "chain", "+".join(kernel_names) or eid)
    descriptor = src.get("seam_descriptor") if isinstance(src.get("seam_descriptor"), dict) else {}
    members = []
    for m in descriptor.get("members") or src.get("seam_members") or []:
        if isinstance(m, dict) and m.get("cls"):
            members.append({k: m[k] for k in ("cls", "module", "module_any", "name", "op", "role")
                            if m.get(k)})
    seam = {"seam_key": seam_key,
            "kinds": list(descriptor.get("kinds") or ([src["kind"]] if src.get("kind") else [])),
            "kernel_names": kernel_names, "op_tags": list(src.get("op_tags") or [])}
    if descriptor.get("max_span"):
        seam["max_span"] = descriptor["max_span"]
    if members:
        seam["members"] = members
    matched = [k.get("name") for k in src.get("matched_existing_kernels") or []
               if isinstance(k, dict) and k.get("name")]
    prior_id = str(src.get("matched_prior_id") or "")
    record = {
        "prior_id": prior_id or "auto-%s-%s-%s" % (ctx["model_family"] or "model",
                                                   _slug(ctx["gfx"]), _slug(seam_key)),
        "title": str(src.get("title") or src.get("fusion") or row.get("fusion") or seam_key),
        "model_family": ctx["model_family"], "model": ctx["model"], "gfx": ctx["gfx"],
        "framework": ctx["framework"], "seam": seam, "exec_id": eid,
        "matched_existing_kernels": matched, "known_prior": bool(prior_id),
    }
    if rung is not None:
        record["rung"] = rung
    return record


def _gain(row, *keys):
    for key in keys:
        if "." in key:
            head, tail = key.split(".", 1)
            v = _num((row.get(head) or {}).get(tail))
        else:
            v = _num(row.get(key))
        if v is not None:
            return v
    return None


def _evidence(row):
    for key in ("evidence", "evidence_path", "ab_json", "applyback_json", "overlay_path"):
        if row.get(key):
            return str(row[key])
    sr = row.get("stack_result") or {}
    if sr.get("evidence"):
        return str(sr["evidence"])
    return str(row.get("reason") or "")[:300]


def _implementation(row, seam):
    impl = row.get("implementation") or row.get("side") or row.get("fused_fn")
    if impl:
        return str(impl)
    if seam.get("matched_existing_kernels"):
        return "existing:%s" % seam["matched_existing_kernels"][0]
    return "self_authored"


def plan(apply_doc, topk, ctx, noise_band):
    """Classify every row of the final apply-back aggregate into write-back records."""
    entries, rungs = _index_board(topk)
    priors, impl, untested, skipped = [], [], [], []

    def measurement(gain, row):
        return {"gain_pct": round(gain, 4), "date": ctx["date"], "protocol": ctx["protocol"],
                "source": ctx["source"], "evidence": _evidence(row)}

    def add_prior(row, polarity, gain, why):
        seam = seam_record(row, entries, rungs, ctx)
        priors.append({"polarity": polarity, "seam_record": seam,
                       "measurement": measurement(gain, row), "why": why})

    def add_impl(row, evidence):
        seam = seam_record(row, entries, rungs, ctx)
        impl.append({"seam_key": seam["seam"]["seam_key"], "prior_id": seam["prior_id"],
                     "exec_id": seam["exec_id"], "rung": row.get("rung"),
                     "implementation": _implementation(row, seam), "evidence_kind": evidence,
                     "reason": str(row.get("reason") or "")[:500],
                     "unit_side_statuses": row.get("unit_side_statuses") or [],
                     "model_family": ctx["model_family"], "model": ctx["model"],
                     "gfx": ctx["gfx"], "framework": ctx["framework"],
                     "date": ctx["date"], "source": ctx["source"], "evidence": _evidence(row)})

    def add_untested(row, why, priority="normal"):
        seam = seam_record(row, entries, rungs, ctx)
        untested.append({"seam_key": seam["seam"]["seam_key"], "prior_id": seam["prior_id"],
                         "exec_id": seam["exec_id"], "why": why, "priority": priority,
                         "expected_gain_pct": _gain(row, "expected_gain_pct",
                                                    "expected_e2e_gain_pct"),
                         "overlay_path": row.get("overlay_path") or "",
                         "unit_side": row.get("unit_side") or row.get("unit_side_statuses") or [],
                         "model_family": ctx["model_family"], "model": ctx["model"],
                         "gfx": ctx["gfx"], "framework": ctx["framework"],
                         "date": ctx["date"], "source": ctx["source"]})

    def rejected_row(row):
        ev = str(row.get("blocked_evidence") or "")
        if row.get("measured") is False:
            skipped.append({"exec_id": row.get("exec_id"), "why": "not measured"})
            return
        if row.get("stack_harmful") or ev == "stack_harmful":
            gain = _gain(row, "stack_result.delta_pct", "throughput_delta_pct")
            if gain is None:
                skipped.append({"exec_id": row.get("exec_id"), "why": "stack_harmful without a delta"})
                return
            add_prior(row, "negative", gain, "stack_harmful")
            return
        if row.get("stack_pending") and not row.get("stack_result"):
            add_untested(row, "stack_pending, never re-gated in the stack")
            return
        if ev in IMPLEMENTATION_EVIDENCE:
            add_impl(row, ev)
            if ev in ("wire_failure", "not_engaged"):
                # A wiring failure says nothing about the fusion's gain: it is the next round's
                # highest-priority candidate (二.10.6d).
                add_untested(row, "wire_failure (%s): diagnosed and re-measured once, still not "
                                  "wired" % (row.get("wire_cause") or ev), priority="highest")
            return
        gain = _gain(row, "throughput_delta_pct", "fusion_only_delta_pct")
        if ev in REGRESSION_EVIDENCE or (ev in SOFT_REJECT_EVIDENCE and gain is not None
                                         and gain < -abs(noise_band)):
            if gain is None:
                skipped.append({"exec_id": row.get("exec_id"), "why": "regression without a delta"})
                return
            add_prior(row, "negative", gain, ev or "measured regression")
            return
        skipped.append({"exec_id": row.get("exec_id"),
                        "why": "no seam-level evidence (%s)" % (ev or "rejected")})

    for row in apply_doc.get("accepted_fusions") or []:
        gain = _gain(row, "throughput_delta_pct", "fusion_only_delta_pct",
                     "stack_result.delta_pct")
        if gain is None:
            skipped.append({"exec_id": row.get("exec_id"), "why": "accepted without a measured delta"})
            continue
        add_prior(row, "positive", gain, "kept in the landed stack" if row.get("stacked")
                  else "accepted")
    for row in apply_doc.get("rejected") or []:
        rejected_row(row)
    for row in apply_doc.get("rung_attempts") or []:
        rejected_row(row)
    for row in apply_doc.get("deferred") or []:
        if row.get("deferred_budget") or str(row.get("reason_code") or "") == "budget_reserve":
            add_untested(row, "apply-back stopped at the stacking reserve")
    for row in apply_doc.get("pool_results") or []:
        status = str(row.get("status") or "")
        base = dict(row.get("row") or {}, **{k: v for k, v in row.items() if k != "row"})
        base.setdefault("exec_id", row.get("exec_id") or row.get("id"))
        if status == "landed":
            gain = _gain(row, "stack_result.delta_pct")
            if gain is not None:
                add_prior(base, "positive", gain, "candidate_pool item kept in the landed stack")
        elif status == "stack_harmful":
            gain = _gain(row, "stack_result.delta_pct")
            if gain is not None:
                add_prior(base, "negative", gain, "candidate_pool item stack_harmful")
        elif status in ("skipped_budget", "not_reached", "unmeasured", "kept_not_landed"):
            add_untested(base, "candidate_pool item %s" % status)
    untested.sort(key=lambda u: 0 if u.get("priority") == "highest" else 1)
    return {"priors": priors, "implementation_failures": impl,
            "untested_candidates": untested, "skipped": skipped}


class _Locked:
    def __init__(self, path):
        self.path = path + ".lock"
        self.fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self.fh = open(self.path, "a")
        try:
            import fcntl
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        return self

    def __exit__(self, *exc):
        try:
            self.fh.close()
        except OSError:
            pass


def _atomic_write(path, doc):
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".wb.", dir=d)
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, path)


def fallback_append(path, seam, measurement, polarity):
    """Append one measurement in schema v2, never rewriting an existing measurement."""
    with _Locked(path):
        doc = _load(path, None)
        if not isinstance(doc, dict):
            doc = {"schema_version": 2, "priors": []}
        doc.setdefault("priors", [])
        target = None
        for prior in doc["priors"]:
            if prior.get("prior_id") == seam["prior_id"]:
                target = prior
                break
        if target is None:
            key = seam["seam"]["seam_key"]
            for prior in doc["priors"]:
                if ((prior.get("seam") or {}).get("seam_key") == key
                        and prior.get("model_family", seam["model_family"]) == seam["model_family"]
                        and prior.get("gfx", seam["gfx"]) == seam["gfx"]):
                    target = prior
                    break
        if target is None:
            target = {k: seam[k] for k in ("prior_id", "title", "model_family", "model", "gfx",
                                           "framework", "seam")}
            target["polarity"] = polarity
            target["measurements"] = []
            doc["priors"].append(target)
        # Same rule as fusion_priors.append_measurement: an identical measurement is not
        # appended twice (a re-run write-back is idempotent); the prior's polarity is that of
        # its most recent disposition, the full history stays in `measurements`.
        key = (measurement["date"], float(measurement["gain_pct"]), measurement.get("source") or "",
               measurement.get("evidence"))
        meas = target.setdefault("measurements", [])
        if not any((m.get("date"), float(m.get("gain_pct", 0.0)), m.get("source") or "",
                    m.get("evidence")) == key for m in meas):
            meas.append(dict(measurement))
        target["polarity"] = polarity
        _atomic_write(path, doc)
    return target["prior_id"]


def _append_prior(path, rec):
    seam, meas, pol = rec["seam_record"], rec["measurement"], rec["polarity"]
    try:
        import fusion_priors
        api = getattr(fusion_priors, "append_measurement")
    except (ImportError, AttributeError):
        api = None
    if api is not None:
        api(path, seam, meas, pol)
        return "fusion_priors"
    fallback_append(path, seam, meas, pol)
    return "fallback"


def _append_sidecar(path, field, records):
    if not records:
        return
    with _Locked(path):
        doc = _load(path, None)
        if not isinstance(doc, dict):
            doc = {"schema_version": 1, "artifact": "fusion_measured_priors_sidecar",
                   "note": ("Implementation-level failures (a unit_fail is about one "
                            "implementation, never a negative seam prior) and candidates the "
                            "budget left untested (next-round candidate input). Append-only."),
                   "implementation_failures": [], "untested_candidates": []}
        doc.setdefault(field, []).extend(records)
        _atomic_write(path, doc)


def run(args):
    apply_doc = _load(args.apply, None)
    if not isinstance(apply_doc, dict):
        return 1, {"ok": False, "reason": "cannot read --apply %s" % args.apply}
    topk = _load(args.topk, {}) if args.topk else {}
    ctx = {"model": args.model or "", "model_family": args.model_family or _model_family(args.model),
           "gfx": args.gfx or "", "framework": args.framework or "",
           "protocol": parse_protocol(args.protocol), "source": args.source or os.path.dirname(
               os.path.abspath(args.apply)),
           "date": args.date or datetime.date.today().isoformat()}
    noise = _num(args.noise_band_pct)
    noise = noise if noise is not None else _num(apply_doc.get("noise_band_pct")) or 0.5
    planned = plan(apply_doc, topk, ctx, noise)
    sidecar = args.sidecar or os.path.join(os.path.dirname(os.path.abspath(args.priors)),
                                           "measured_priors_sidecar.json")
    out = {"ok": True, "dry_run": bool(args.dry_run), "priors_path": args.priors,
           "sidecar_path": sidecar, "context": ctx,
           "counts": {"positive": sum(1 for p in planned["priors"] if p["polarity"] == "positive"),
                      "negative": sum(1 for p in planned["priors"] if p["polarity"] == "negative"),
                      "implementation_failures": len(planned["implementation_failures"]),
                      "untested_candidates": len(planned["untested_candidates"]),
                      "highest_priority_next_round": sum(
                          1 for u in planned["untested_candidates"] if u.get("priority") == "highest"),
                      "skipped": len(planned["skipped"])},
           "planned": planned}
    if args.dry_run:
        return 0, out
    apis = []
    for rec in planned["priors"]:
        apis.append(_append_prior(args.priors, rec))
    _append_sidecar(sidecar, "implementation_failures", planned["implementation_failures"])
    _append_sidecar(sidecar, "untested_candidates", planned["untested_candidates"])
    out["api"] = sorted(set(apis)) or ["none"]
    return 0, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", required=True, help="final apply-back aggregate (apply_result.json)")
    ap.add_argument("--topk", default="", help="fusion_topk.json (seam descriptors)")
    ap.add_argument("--priors", default=DEFAULT_PRIORS)
    ap.add_argument("--sidecar", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--model-family", default="")
    ap.add_argument("--gfx", default="")
    ap.add_argument("--framework", default="")
    ap.add_argument("--protocol", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--noise-band-pct", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="", help="also write the JSON report here")
    a = ap.parse_args(argv)
    try:
        code, out = run(a)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        code, out = 1, {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}
    text = json.dumps(out, sort_keys=True)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
