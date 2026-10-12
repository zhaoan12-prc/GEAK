#!/usr/bin/env python3
"""Calibrated accuracy gate for KernelFusion apply-back.

WHY THIS EXISTS
---------------
Qwen3.5-397B-A17B-FP8 (20261010_test_e2e_qwen35): the apply-back accuracy gate was
gsm8k n=200, thinking off, served at CONC=4. Two boots of the IDENTICAL base config
scored 0.555 and 0.465 (McNemar p=0.041, 48% of questions flipped). A test that fails
the base against itself cannot reject anything: an authored MoE quant kernel that was
statistically identical to the vendor kernel (same MSE, bias ~5e-7, fp8 ybit 0.9988)
"dropped" to 0.36 on it, and a GDN out-norm fusion was failed at p=0.004 -- by the
same harness whose control had just failed.

So the accuracy test is CALIBRATED before it is allowed to reject:

  calibrate   run the accuracy test twice on the identical base config (base-vs-base).
              gsm8k is `usable` only if the control passes scripts/accuracy_gate.py in
              BOTH directions and its question-flip rate is <= --max-flip-rate.
              The deterministic probe (fixed prompts, greedy, sequential requests,
              token + logprob comparison) is `usable` when the two base runs agree on a
              long prefix (median first divergence >= --min-prefix or >= half exact).

  decide      in this order:
              1. gsm8k usable  -> accuracy_gate.py's verdict on base vs cand (a
                 calibrated test may reject);
              2. otherwise the gsm8k result is INFORMATIONAL and can never fail the
                 fusion; the deterministic probe decides when it is usable;
              3. otherwise, a fusion that is numerically equivalent at the unit level
                 (fp8 ybit >= --equiv-ybit, or relative error within the quantization
                 noise --quant-noise-rel) passes: an uncalibrated test may not reject it;
              4. otherwise `inconclusive`, with the next step named (larger n with
                 thinking on, or the deterministic probe) -- never a silent pass/fail.

  probe       the deterministic probe itself: fixed prompts against /v1/completions,
              temperature 0, one request at a time (no batching nondeterminism),
              recording output tokens and their logprobs.

  decide --accuracy-mode skip
              the round's calibration found NO usable e2e accuracy test (neither gsm8k
              nor the deterministic probe passes base-vs-base), so the workflow runs no
              e2e accuracy test at all for the rest of the round. Accuracy is then gated
              only by unit-level numerical equivalence (the unit-side parity verdict):
              pass unless --unit-parity fail. A calibration that DOES have a usable test
              ignores the skip request and decides normally.

  cache-lookup / cache-store
              calibration results are cached per (model, serving config, server image).
              The key is the sha256 of a canonical JSON of the model path, backend, TP,
              serving flags (grouped per option, sorted), env (sorted), the exact base
              overlay path list, the workload, the accuracy harness settings and the
              installed sglang / aiter / vllm / torch versions. A hit returns the cached
              calibration (a copy kept in the cache dir); a miss returns {"hit": false}.

Exit codes for `decide`: 0 pass, 1 fail, 3 inconclusive (mirrors accuracy_gate.py).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accuracy_gate  # noqa: E402

DEFAULT_PROMPTS = [
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:",
    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many "
    "bolts in total does it take?\nAnswer:",
    "Write a Python function that returns the n-th Fibonacci number iteratively.\n",
    "Explain in three sentences why the sky appears blue during the day.\n",
    "Translate to French: The weather is nice today and we are going to the park.\n",
    "List the first ten prime numbers, separated by commas:",
    "The capital of Australia is",
    "Summarize the plot of Romeo and Juliet in two sentences.\n",
    "Question: If a train travels 60 miles per hour for 2.5 hours, how far does it travel?\n"
    "Answer:",
    "def quicksort(arr):\n",
    "Give a short definition of entropy in thermodynamics.\n",
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
    "babysitting. How much did she earn?\nAnswer:",
    "Write a haiku about autumn leaves.\n",
    "What is the derivative of x^3 * sin(x)? Show the steps.\n",
    "Name three differences between TCP and UDP.\n",
    "Continue the sequence: 2, 6, 12, 20, 30,",
]


# --------------------------------------------------------------------------- probe
def _post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _split_text(text):
    return re.findall(r"\s+|\w+|[^\w\s]", text or "")


def probe(base_url, model, prompts, max_tokens=128, timeout=600, post=_post):
    out = []
    url = base_url.rstrip("/") + "/completions"
    for idx, prompt in enumerate(prompts):
        body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0.0, "top_p": 1.0, "seed": 0, "logprobs": 1}
        doc = post(url, body, timeout)
        choice = (doc.get("choices") or [{}])[0]
        lp = choice.get("logprobs") or {}
        tokens = lp.get("tokens")
        logprobs = lp.get("token_logprobs")
        text = choice.get("text") or ""
        out.append({"id": idx, "prompt": prompt, "text": text,
                    "tokens": tokens if tokens else _split_text(text),
                    "token_source": "server" if tokens else "text_split",
                    "token_logprobs": logprobs if tokens else None})
    return {"schema": "fusion_det_probe/1", "max_tokens": max_tokens,
            "temperature": 0.0, "concurrency": 1, "prompts": out}


# --------------------------------------------------------------------------- compare
def _first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def det_compare(a_doc, b_doc, early_tokens=8):
    a = {p["id"]: p for p in a_doc.get("prompts", [])}
    b = {p["id"]: p for p in b_doc.get("prompts", [])}
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("the two probes share no prompt ids")
    first, exact, early, maes = [], 0, 0, []
    for i in ids:
        ta, tb = a[i].get("tokens") or [], b[i].get("tokens") or []
        fd = _first_div(ta, tb)
        first.append(fd)
        if ta == tb:
            exact += 1
        elif fd < early_tokens:
            early += 1
        la, lb = a[i].get("token_logprobs"), b[i].get("token_logprobs")
        if la and lb:
            k = min(fd, len(la), len(lb))
            diffs = [abs(float(x) - float(y)) for x, y in zip(la[:k], lb[:k])
                     if x is not None and y is not None]
            if diffs:
                maes.append(sum(diffs) / len(diffs))
    return {"n": len(ids), "exact_rate": round(exact / float(len(ids)), 4),
            "early_div_rate": round(early / float(len(ids)), 4),
            "early_tokens": early_tokens,
            "median_first_div": statistics.median(first),
            "min_first_div": min(first),
            "logprob_mae": round(statistics.mean(maes), 6) if maes else None}


def det_verdict(control, cand, min_prefix=32, exact_tol=0.25, logprob_fail=0.5):
    """cand = det_compare(base, cand); control = det_compare(base_a, base_b)."""
    c_lp, d_lp = control.get("logprob_mae"), cand.get("logprob_mae")
    if cand["early_div_rate"] >= 0.5 and control["early_div_rate"] <= 0.1:
        return "fail", ("candidate diverges from the base within %d tokens on %.0f%% of "
                        "the fixed prompts while the base-vs-base control does on %.0f%%"
                        % (cand["early_tokens"], cand["early_div_rate"] * 100,
                           control["early_div_rate"] * 100))
    if (d_lp is not None and c_lp is not None and d_lp > logprob_fail
            and c_lp < logprob_fail / 5.0):
        return "fail", ("chosen-token logprob error %.4f nats on the shared prefix vs "
                        "control %.4f" % (d_lp, c_lp))
    if (cand["exact_rate"] >= control["exact_rate"] - exact_tol
            or cand["median_first_div"] >= min(min_prefix, control["median_first_div"])):
        return "pass", ("deterministic probe: exact %.2f (control %.2f), median first "
                        "divergence %s tokens (control %s)"
                        % (cand["exact_rate"], control["exact_rate"],
                           cand["median_first_div"], control["median_first_div"]))
    return "inconclusive", ("deterministic probe between control and failure: exact %.2f "
                            "vs control %.2f, median first divergence %s vs %s"
                            % (cand["exact_rate"], control["exact_rate"],
                               cand["median_first_div"], control["median_first_div"]))


# --------------------------------------------------------------------------- calibrate
def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def calibrate(gsm8k_a=None, gsm8k_b=None, det_a=None, det_b=None, tol=0.01,
              alpha=0.05, max_flip_rate=0.2, min_prefix=32):
    out = {"schema": "fusion_accuracy_calibration/1", "tol": tol, "alpha": alpha,
           "max_flip_rate": max_flip_rate, "min_prefix": min_prefix,
           "gsm8k": {"usable": False, "measured": False},
           "deterministic": {"usable": False, "measured": False}}
    if gsm8k_a is not None and gsm8k_b is not None:
        fwd = accuracy_gate.gate(gsm8k_a, gsm8k_b, tol, alpha)
        rev = accuracy_gate.gate(gsm8k_b, gsm8k_a, tol, alpha)
        n = fwd["n"]
        flips = fwd["base_right_cand_wrong"] + fwd["cand_right_base_wrong"]
        flip_rate = flips / float(n) if n else 1.0
        usable = (fwd["verdict"] == "pass" and rev["verdict"] == "pass"
                  and flip_rate <= max_flip_rate)
        reasons = []
        if fwd["verdict"] != "pass" or rev["verdict"] != "pass":
            reasons.append("base-vs-base control does not pass the gate (%s / %s)"
                           % (fwd["verdict"], rev["verdict"]))
        if flip_rate > max_flip_rate:
            reasons.append("%.0f%% of questions flip between two identical base runs "
                           "(> %.0f%%)" % (flip_rate * 100, max_flip_rate * 100))
        out["gsm8k"] = {"usable": usable, "measured": True, "n": n,
                        "base_a_accuracy": fwd["base_accuracy"],
                        "base_b_accuracy": fwd["cand_accuracy"],
                        "flip_rate": round(flip_rate, 4),
                        "control_forward": fwd, "control_reverse": rev,
                        "reason": "; ".join(reasons) if reasons else
                        "base-vs-base control passes both ways"}
    if det_a is not None and det_b is not None:
        ctl = det_compare(det_a, det_b)
        usable = ctl["median_first_div"] >= min_prefix or ctl["exact_rate"] >= 0.5
        out["deterministic"] = {
            "usable": usable, "measured": True, "control": ctl,
            "reason": ("base-vs-base probe agrees (median first divergence %s, exact %.2f)"
                       % (ctl["median_first_div"], ctl["exact_rate"])) if usable else
                      ("the probe is not deterministic on this server (median first "
                       "divergence %s < %d, exact %.2f)"
                       % (ctl["median_first_div"], min_prefix, ctl["exact_rate"]))}
    if out["gsm8k"]["usable"]:
        out["mode"] = "gsm8k"
    elif out["deterministic"]["usable"]:
        out["mode"] = "deterministic"
    else:
        out["mode"] = "unit_equivalence_only"
    out["accuracy_can_reject"] = out["mode"] != "unit_equivalence_only"
    if out["mode"] != "gsm8k" and out["gsm8k"]["measured"]:
        out["recommendation"] = (
            "gsm8k control failed: do not reject on it. Use the deterministic probe; to "
            "make gsm8k usable re-calibrate with a larger --limit (full 1319) and "
            "thinking on")
    return out


# --------------------------------------------------------------------------- decide
def unit_equivalent(ybit=None, rel_err=None, equiv_ybit=0.99, quant_noise_rel=0.0625):
    if ybit is not None and float(ybit) >= equiv_ybit:
        return True, "unit fp8 ybit %.4f >= %.2f" % (float(ybit), equiv_ybit)
    if rel_err is not None and float(rel_err) <= quant_noise_rel:
        return True, "unit relative error %.4g <= quantization noise %.4g" % (
            float(rel_err), quant_noise_rel)
    return False, ""


def calibration_usable(calibration):
    """True when the calibration has at least one e2e accuracy test that may reject."""
    cal = calibration or {}
    return bool((cal.get("gsm8k") or {}).get("usable")
                or (cal.get("deterministic") or {}).get("usable")
                or cal.get("gsm8k_usable") or cal.get("deterministic_usable"))


def decide_skipped(calibration, unit_parity=None, ybit=None, rel_err=None,
                   equiv_ybit=0.99, quant_noise_rel=0.0625):
    """Accuracy decision when the round runs no e2e accuracy test (mode skip).

    The only accuracy evidence is the unit-side numerical equivalence; a unit parity
    failure fails, anything else passes."""
    out = {"schema": "fusion_accuracy_decision/1",
           "calibration_mode": (calibration or {}).get("mode"),
           "accuracy": {"mode": "skipped", "reason": "calibration_unusable"},
           "informational": {}}
    eq, why = unit_equivalent(ybit, rel_err, equiv_ybit, quant_noise_rel)
    if ybit is not None or rel_err is not None:
        out["informational"]["unit_equivalence"] = {"equivalent": eq, "detail": why}
    if str(unit_parity or "").lower() == "fail":
        out.update(verdict="fail", test_used="unit_parity",
                   reason="e2e accuracy not gated this round (calibration unusable) and the "
                          "unit-side parity check failed")
        return out
    out.update(verdict="pass", test_used="skipped",
               reason="e2e accuracy not gated this round: no accuracy test passed its "
                      "base-vs-base calibration; accuracy rests on unit-side numerical "
                      "equivalence (parity %s)" % (unit_parity or "not reported"))
    return out


def decide(calibration, base_gsm8k=None, cand_gsm8k=None, det_base=None, det_cand=None,
           ybit=None, rel_err=None, equiv_ybit=0.99, quant_noise_rel=0.0625,
           min_prefix=32, accuracy_mode="gate", unit_parity=None):
    skip_note = ""
    if str(accuracy_mode or "gate").lower() == "skip":
        if not calibration_usable(calibration):
            return decide_skipped(calibration, unit_parity, ybit, rel_err, equiv_ybit,
                                  quant_noise_rel)
        skip_note = ("--accuracy-mode skip ignored: the calibration has a usable accuracy "
                     "test, so it decides")
    tol = float(calibration.get("tol", 0.01))
    alpha = float(calibration.get("alpha", 0.05))
    out = {"schema": "fusion_accuracy_decision/1", "calibration_mode": calibration.get("mode"),
           "verdict": None, "test_used": None, "reason": "", "informational": {}}
    if skip_note:
        out["informational"]["accuracy_mode"] = skip_note
    g_usable = bool((calibration.get("gsm8k") or {}).get("usable"))
    d_cal = calibration.get("deterministic") or {}
    gsm = None
    if base_gsm8k is not None and cand_gsm8k is not None:
        gsm = accuracy_gate.gate(base_gsm8k, cand_gsm8k, tol, alpha)
    if gsm is not None and g_usable:
        out.update(verdict=gsm["verdict"], test_used="gsm8k_calibrated",
                   reason=gsm["reason"], gsm8k=gsm)
        if gsm["verdict"] != "inconclusive":
            return out
    elif gsm is not None:
        out["informational"]["gsm8k_uncalibrated"] = dict(
            gsm, note="the base-vs-base control failed this test; it cannot reject")

    if det_base is not None and det_cand is not None and d_cal.get("usable"):
        cmp_ = det_compare(det_base, det_cand)
        verdict, reason = det_verdict(d_cal["control"], cmp_, min_prefix=min_prefix)
        out["deterministic"] = cmp_
        out.update(verdict=verdict, test_used="deterministic", reason=reason)
        if verdict != "inconclusive":
            return out

    eq, why = unit_equivalent(ybit, rel_err, equiv_ybit, quant_noise_rel)
    if eq:
        out.update(verdict="pass", test_used="unit_equivalence",
                   reason=("numerically equivalent at the unit level (%s); no calibrated "
                           "accuracy test contradicts it, so it is not rejected" % why))
        return out
    out.update(verdict="inconclusive", test_used=out["test_used"] or "none",
               reason=(out["reason"] or "no calibrated accuracy test and no unit-level "
                       "numerical equivalence"),
               next_step=("run the deterministic probe on both legs, or re-calibrate "
                          "gsm8k with the full 1319 questions and thinking on, then "
                          "decide again; never accept or reject on this verdict"))
    return out


# --------------------------------------------------------------------------- cache
CACHE_VERSION_PACKAGES = ("sglang", "aiter", "vllm", "torch")
_NUMERIC = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def _split_tokens(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v) != ""]
    try:
        return shlex.split(str(value))
    except ValueError:
        return str(value).split()


def canonical_flags(value):
    """Serving flags as sorted option groups: each option keeps its own values, so
    reordering options does not change the key but moving a value between options does."""
    groups, cur = [], None
    for tok in _split_tokens(value):
        if tok.startswith("-") and not _NUMERIC.match(tok):
            if cur is not None:
                groups.append(cur)
            cur = tok
        else:
            cur = tok if cur is None else "%s %s" % (cur, tok)
    if cur is not None:
        groups.append(cur)
    return sorted(groups)


def canonical_env(value):
    if isinstance(value, dict):
        return sorted("%s=%s" % (k, v) for k, v in value.items())
    return sorted(_split_tokens(value))


def canonical_overlay(value):
    """The exact base overlay path list (PYTHONPATH order kept, paths not normalized)."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v)]
    return [p for p in re.split(r"[:,]", str(value)) if p]


def package_versions(packages=CACHE_VERSION_PACKAGES):
    out = {}
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - py<3.8
        metadata = None
    for name in packages:
        ver = None
        if metadata is not None:
            try:
                ver = metadata.version(name)
            except Exception:
                ver = None
        out[name] = ver
    return out


def cache_key(key_doc, versions=None):
    """(canonical key dict, sha256 hex) for one calibration context."""
    doc = dict(key_doc or {})
    key = {
        "schema": "fusion_accuracy_cache_key/1",
        "model_path": str(doc.get("model_path") or ""),
        "backend": str(doc.get("backend") or ""),
        "tp": doc.get("tp"),
        "flags": canonical_flags(doc.get("flags")),
        "env": canonical_env(doc.get("env")),
        "overlay": canonical_overlay(doc.get("overlay")),
        "workload": doc.get("workload"),
        "harness": doc.get("harness"),
        "versions": versions if versions is not None else (
            doc.get("versions") if isinstance(doc.get("versions"), dict)
            else package_versions()),
    }
    blob = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return key, hashlib.sha256(blob.encode()).hexdigest()


def _cal_summary(cal):
    return {"mode": cal.get("mode"),
            "gsm8k_usable": bool((cal.get("gsm8k") or {}).get("usable")),
            "deterministic_usable": bool((cal.get("deterministic") or {}).get("usable")),
            "accuracy_can_reject": bool(cal.get("accuracy_can_reject",
                                                cal.get("mode") not in (None, "unit_equivalence_only")))}


def cache_lookup(cache_dir, key_doc, versions=None):
    key, sha = cache_key(key_doc, versions)
    entry = os.path.join(cache_dir, sha)
    cal_path = os.path.join(entry, "calibration.json")
    try:
        stored_key = _load_json(os.path.join(entry, "key.json"))
        cal = _load_json(cal_path)
    except (OSError, ValueError):
        return {"ok": True, "hit": False, "key_sha": sha}
    if stored_key.get("key") != key:
        return {"ok": True, "hit": False, "key_sha": sha,
                "reason": "cache entry key does not match (hash collision or edited entry)"}
    meta = {}
    try:
        meta = _load_json(os.path.join(entry, "meta.json"))
    except (OSError, ValueError):
        pass
    out = {"ok": True, "hit": True, "key_sha": sha,
           "calibration_json": os.path.abspath(cal_path),
           "stored_at": meta.get("stored_at"),
           "source_calibration_json": meta.get("source_calibration_json")}
    out.update(_cal_summary(cal))
    return out


def cache_store(cache_dir, key_doc, calibration_json, versions=None, now=None):
    key, sha = cache_key(key_doc, versions)
    try:
        cal = _load_json(calibration_json)
    except (OSError, ValueError) as exc:
        return {"ok": False, "stored": False, "key_sha": sha,
                "reason": "cannot read calibration %s: %s" % (calibration_json, exc)}
    if not cal.get("mode") or not ((cal.get("gsm8k") or {}).get("measured")
                                   or (cal.get("deterministic") or {}).get("measured")):
        return {"ok": True, "stored": False, "key_sha": sha,
                "reason": "calibration measured no accuracy test; not cached"}
    entry = os.path.join(cache_dir, sha)
    os.makedirs(entry, exist_ok=True)
    dst = os.path.join(entry, "calibration.json")
    tmp = dst + ".tmp"
    shutil.copyfile(calibration_json, tmp)
    os.replace(tmp, dst)
    stamp = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(entry, "key.json"), "w") as fh:
        json.dump({"key": key, "key_sha": sha}, fh, indent=1, sort_keys=True)
    with open(os.path.join(entry, "meta.json"), "w") as fh:
        json.dump({"stored_at": stamp,
                   "source_calibration_json": os.path.abspath(calibration_json)}, fh, indent=1)
    out = {"ok": True, "stored": True, "key_sha": sha, "calibration_json": os.path.abspath(dst),
           "stored_at": stamp}
    out.update(_cal_summary(cal))
    return out


# --------------------------------------------------------------------------- CLI
def _gsm(path):
    return accuracy_gate._load(path) if path else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:30000/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--prompts-file", default="", help="JSON list of prompt strings")
    p.add_argument("--out", required=True)

    p = sub.add_parser("calibrate")
    p.add_argument("--gsm8k-a", default="")
    p.add_argument("--gsm8k-b", default="")
    p.add_argument("--det-a", default="")
    p.add_argument("--det-b", default="")
    p.add_argument("--tol", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-flip-rate", type=float, default=0.2)
    p.add_argument("--min-prefix", type=int, default=32)
    p.add_argument("--out", required=True)

    p = sub.add_parser("decide")
    p.add_argument("--calibration", required=True)
    p.add_argument("--base-gsm8k", default="")
    p.add_argument("--cand-gsm8k", default="")
    p.add_argument("--det-base", default="")
    p.add_argument("--det-cand", default="")
    p.add_argument("--unit-ybit", type=float, default=None)
    p.add_argument("--unit-rel-err", type=float, default=None)
    p.add_argument("--equiv-ybit", type=float, default=0.99)
    p.add_argument("--quant-noise-rel", type=float, default=0.0625)
    p.add_argument("--accuracy-mode", choices=("gate", "skip"), default="gate",
                   help="skip: the round's calibration found no usable e2e accuracy test; "
                        "decide on unit-side numerical equivalence only")
    p.add_argument("--unit-parity", choices=("pass", "fail"), default=None,
                   help="unit-side parity verdict (used by --accuracy-mode skip)")
    p.add_argument("--out", required=True)

    p = sub.add_parser("cache-lookup")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--key-json", required=True)

    p = sub.add_parser("cache-store")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--key-json", required=True)
    p.add_argument("--calibration-json", required=True)
    a = ap.parse_args(argv)

    if a.cmd in ("cache-lookup", "cache-store"):
        try:
            key_doc = _load_json(a.key_json)
        except (OSError, ValueError) as exc:
            print(json.dumps({"ok": False, "hit": False, "stored": False,
                              "reason": "cannot read key json %s: %s" % (a.key_json, exc)}))
            return 1
        if a.cmd == "cache-lookup":
            result = cache_lookup(a.cache_dir, key_doc)
        else:
            result = cache_store(a.cache_dir, key_doc, a.calibration_json)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("ok") else 1

    if a.cmd == "probe":
        prompts = _load_json(a.prompts_file) if a.prompts_file else DEFAULT_PROMPTS
        result = probe(a.base_url, a.model, prompts, a.max_tokens)
        code = 0
    elif a.cmd == "calibrate":
        result = calibrate(_gsm(a.gsm8k_a), _gsm(a.gsm8k_b),
                           _load_json(a.det_a) if a.det_a else None,
                           _load_json(a.det_b) if a.det_b else None,
                           a.tol, a.alpha, a.max_flip_rate, a.min_prefix)
        code = 0
    else:
        result = decide(_load_json(a.calibration), _gsm(a.base_gsm8k), _gsm(a.cand_gsm8k),
                        _load_json(a.det_base) if a.det_base else None,
                        _load_json(a.det_cand) if a.det_cand else None,
                        a.unit_ybit, a.unit_rel_err, a.equiv_ybit, a.quant_noise_rel,
                        accuracy_mode=a.accuracy_mode, unit_parity=a.unit_parity)
        code = {"pass": 0, "fail": 1}.get(result["verdict"], 3)
    text = json.dumps(result, indent=2)
    with open(a.out, "w") as fh:
        fh.write(text + "\n")
    if a.cmd != "probe":
        print(text)
        print("ACCURACY_VERDICT=%s" % result.get("verdict", result.get("mode")))
    else:
        print(json.dumps({"out": a.out, "prompts": len(result["prompts"])}))
    return code


if __name__ == "__main__":
    sys.exit(main())
