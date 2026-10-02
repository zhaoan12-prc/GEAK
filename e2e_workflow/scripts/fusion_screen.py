#!/usr/bin/env python3
"""Screen a semantic kernel table into a fusion Top-K.

Trace capture stays upstream. This script is the policy after that table exists:

  serial / parallel chains, cut at fusion barriers
  -> match each chain against the installed fusion-kernel catalog
  -> keep chains that clear the time, speedup, and optional gates
  -> initial list, every passing chain, ordered by estimated time saved
  -> one pass = Top-K by Score = time saved / difficulty
  -> two passes, merged into the final Top-K

Chains. A serial chain is up to 4 adjacent compute kernels (5 when an existing
fusion kernel covers it, so the chain can grow to the kernel's boundary, e.g.
q_norm+k_norm+RoPE -> +KV write). Pure data movement (copy / contiguous) is
timed as part of the chain but never classified, counted, or matched.
Attention, collectives and GEMMs are barriers: never inside a chain, and at an
end only when a catalog kernel covers the whole chain (quant->GEMM prologue).

Existing kernel. Chain members are tagged with fusion_catalog's op vocabulary
(norm / add_residual / quant / rope / kv_cache / activation / ...). A catalog
kernel whose op_tags cover the chain's tags at a compatible quant dtype is an
existing kernel; the chain then uses difficulty 1.0 and records the match.
A trace kernel that already is a catalog fused kernel is reported as
"already fused in the baseline".

Time gate. The chain must be >= 3% of the end-to-end step, or >= 1% when an
existing kernel covers it. When the workload is known, the same seam in
prefill and decode (same op sequence, same occurrence in the layer) is gated
on its combined request-level share, and the final list carries it as one
execution entry.

Time saved is the decode saving for a decode chain and the chain's own saving
otherwise. Difficulty is 1.0 with an existing kernel, otherwise the hardest
compute member:

  elementwise / quant / cast / RoPE / act_and_mul   1.3
  row reduction (norm)                              1.5
  layout / KV-cache write                           1.8
  topk / sort / gather                              2.2
  collective, large GEMM, attention                 5

Merge: a chain in both passes ranks ahead of a chain in one; ties break on the
sum of the two ranks, with a missing pass counted as rank K+1. Two chains that
share a kernel keep only the one with the better merged rank; every shorter
sub-chain is its own candidate, so the unshared remainder can still take a
slot. Every chain in the final list is optimized.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fusion_catalog as catalog_lib  # noqa: E402

ADJACENT_GAP_US = 50.0
LAUNCH_US = 5.0
CHAIN_E2E_MIN = 0.03
EXISTING_CHAIN_E2E_MIN = 0.01
LARGE_TIME_SHARE = 0.05
LARGE_COMPUTE_SHARE = 0.10
SMALL_SPEEDUP = 1.25
LARGE_SPEEDUP = 1.15
LAUNCH_FRACTION_MIN = 0.10
MEMORY_FRACTION_MIN = 0.25
TINY_US = 20.0
TINY_FRACTION_MIN = 0.50
SERIAL_MAX = 4
EXISTING_SERIAL_MAX = SERIAL_MAX + 1
FINAL_K = 8
SCREEN_PASSES = 2
EXISTING_DIFFICULTY = 1.0
NEAR_MISS_MIN = 0.005
NEAR_MISS_LIMIT = 12

# Datasheet peaks used only when the box has not been measured.
PEAK_TFLOPS = {
    "gfx942": {"bf16": 1307.0, "fp8": 2615.0},
    "gfx950": {"bf16": 2500.0, "fp8": 5000.0},
}

_ATTENTION_TOKENS = ("attention", "fmha", "flash_attn", "paged_attn", "mla_decode",
                     "mla_prefill", "attn_fwd")
_COLLECTIVE_TOKENS = ("allreduce", "all_reduce", "rccl", "nccl")
_GEMM_TOKENS = ("gemm", "addmm", "hipblas", "cublas", "wvsplit", "cijk_")

# (difficulty, name fragments). First match wins, so the hard classes come first.
_TIERS = (
    (5.0, _COLLECTIVE_TOKENS),
    (5.0, _ATTENTION_TOKENS),
    (5.0, _GEMM_TOKENS),
    (2.2, ("topk", "top_k", "sort", "gather", "scatter")),
    (1.8, ("kv_cache", "kvcache", "cache_write", "reshape", "transpose",
           "permute", "layout")),
    (1.5, ("norm", "softmax")),
    (1.3, ("quant", "cast", "silu", "gelu", "elementwise", "ew_", "rotary",
           "rope", "act_and_mul", "act_mul")),
)
HARDEST_DIFFICULTY = 5.0
UNKNOWN_DIFFICULTY = 2.2

# Copies that a fused kernel absorbs. Timed, never classified.
_DATA_MOVE_TOKENS = ("direct_copy", "copy_kernel", "_to_copy", "contiguous")

# Trace spellings the catalog vocabulary does not carry.
_TRACE_TAG_EXTRA = (
    ("activation", ("act_and_mul", "silu_and_mul", "gelu_and_mul")),
)
# Region dtype is the quant format; activation dtypes (bf16/fp16) would wrongly
# exclude fp8 kernels, whose inputs are bf16 too.
_QUANT_DTYPES = {"fp8", "fp8_blockscale", "fp4"}
# Tags a GEMM earns by position, not by name.
_POSITION_TAGS = {"gemm_prologue", "gemm_epilogue"}
# Catalog helpers that are predicates, not kernels.
_NOT_KERNEL_PREFIX = ("is_", "has_", "check_")


def _name(row):
    return str(row.get("raw_name") or row.get("short_name") or "")


def _duration(row):
    return float(row.get("duration_us") or 0.0)


def _layers(row):
    try:
        return max(int(row.get("pattern_layer_count") or 1), 1)
    except (TypeError, ValueError):
        return 1


def load_rows(table):
    """Flatten pattern/phase tables. One representative layer is scaled later."""
    tables = table.get("tables") if isinstance(table, dict) else None
    rows = []
    if tables:
        for spec in tables:
            nlayers = spec.get("pattern_layer_count") or 1
            for row in spec.get("rows") or []:
                item = dict(row)
                item["phase"] = item.get("phase") or spec.get("phase") or "unresolved"
                item["pattern_id"] = item.get("pattern_id") or spec.get("pattern_id")
                item["pattern_layer_count"] = nlayers
                if not item.get("layer_total_us"):
                    item["layer_total_us"] = spec.get("layer_total_us")
                rows.append(item)
    elif isinstance(table, dict) and table.get("rows"):
        rows = [dict(row) for row in table["rows"]]
    for index, row in enumerate(rows):
        row.setdefault("row_id", "row-%d" % index)
        row.setdefault("phase", "unresolved")
        row["duration_us"] = _duration(row)
    return rows


def phase_e2e_us(rows):
    """Full-step kernel time. A representative layer is multiplied by its layer count."""
    seen = {}
    for row in rows:
        key = (row.get("phase"), row.get("pattern_id"))
        total = row.get("layer_total_us")
        if total is None:
            continue
        seen[key] = float(total) * _layers(row)
    if seen:
        out = {}
        for (phase, _), total in seen.items():
            out[phase] = out.get(phase, 0.0) + total
        return out
    out = {}
    for row in rows:
        phase = row.get("phase")
        out[phase] = out.get(phase, 0.0) + _duration(row) * _layers(row)
    return out


def phase_weights(rows, workload):
    """Forwards of each phase one request pays, or {} when the workload is unknown.

    A prefill step that carried T tokens costs ISL/T of itself per request; a
    decode step over B requests costs 1/B per request, times OSL-1 forwards.
    Without step metadata this is the legacy 1 prefill + OSL-1 decode forwards.
    """
    workload = workload or {}
    isl, osl = workload.get("isl"), workload.get("osl")
    if not isl or not osl:
        return {}
    weights = {"prefill": 1.0, "decode": float(max(int(osl) - 1, 0))}
    for row in rows:
        phase = row.get("phase")
        if phase == "prefill" and row.get("step_input_tokens"):
            weights["prefill"] = float(isl) / float(row["step_input_tokens"])
        elif phase == "decode" and row.get("step_batch_size"):
            weights["decode"] = float(max(int(osl) - 1, 0)) / float(row["step_batch_size"])
    return weights


def _dtype(row):
    types = " ".join(str(item).lower() for item in ((row.get("shape") or {}).get("input_types") or []))
    blob = types + " " + _name(row).lower()
    if "fp8" in blob or "float8" in blob:
        return "fp8"
    return "bf16"


def _is_gemm(row):
    return _op_tier(_name(row)) == HARDEST_DIFFICULTY and any(
        token in _name(row).lower() for token in _GEMM_TOKENS)


def gemm_flops(row):
    """GEMM FLOPs are 2*M*N*K. Anything else is unknown and stays a small op."""
    if not _is_gemm(row):
        return None
    dims = (row.get("shape") or {}).get("input_dims") or []
    for dim in dims:
        if isinstance(dim, (list, tuple)) and len(dim) >= 3:
            tail = dim[-3:]
            if all(isinstance(value, (int, float)) and value > 1 for value in tail):
                return 2.0 * tail[0] * tail[1] * tail[2]
    flat = [value for value in dims if isinstance(value, (int, float)) and value > 1]
    if len(flat) >= 3:
        return 2.0 * flat[-3] * flat[-2] * flat[-1]
    return None


def _op_tier(name):
    lowered = name.lower()
    for difficulty, tokens in _TIERS:
        if any(token in lowered for token in tokens):
            return difficulty
    return UNKNOWN_DIFFICULTY


def _is_unknown(name):
    lowered = name.lower()
    return not any(any(token in lowered for token in tokens) for _, tokens in _TIERS)


def _is_data_move(row):
    lowered = _name(row).lower()
    return any(token in lowered for token in _DATA_MOVE_TOKENS)


def _barrier(row):
    lowered = _name(row).lower()
    if any(token in lowered for token in _COLLECTIVE_TOKENS):
        return "allreduce"
    if any(token in lowered for token in _ATTENTION_TOKENS):
        return "attention"
    if any(token in lowered for token in _GEMM_TOKENS):
        return "gemm"
    return None


def _tags(name):
    lowered = name.lower()
    tags = set(catalog_lib._op_tags(name))
    for tag, tokens in _TRACE_TAG_EXTRA:
        if any(token in lowered for token in tokens):
            tags.add(tag)
    return tags - _POSITION_TAGS


def _quant_dtypes(name):
    return set(catalog_lib._dtype_tags(name)) & _QUANT_DTYPES


def classify_kernel(row, e2e_us, gfx):
    duration = _duration(row)
    scaled = duration * _layers(row)
    time_share = scaled / e2e_us if e2e_us else 0.0
    flops = gemm_flops(row)
    compute_share = 0.0
    if flops and duration > 0:
        tflops = flops / (duration * 1e-6) / 1e12
        peak = PEAK_TFLOPS.get(gfx, PEAK_TFLOPS["gfx942"])[_dtype(row)]
        compute_share = tflops / peak if peak else 0.0
    large = time_share >= LARGE_TIME_SHARE and compute_share >= LARGE_COMPUTE_SHARE
    return {
        "time_share": time_share,
        "compute_share": compute_share,
        "large": large,
        "flops": flops,
    }


def _adjacent(left, right):
    if left.get("stream") != right.get("stream"):
        return False
    if (left.get("layer_id") is not None and right.get("layer_id") is not None
            and left.get("layer_id") != right.get("layer_id")):
        return False
    if left.get("timestamp") is None or right.get("timestamp") is None:
        return True
    end = float(left["timestamp"]) + _duration(left)
    return 0 <= float(right["timestamp"]) - end <= ADJACENT_GAP_US


def _serial_chains(rows):
    """Every contiguous window of an adjacent run with 2..EXISTING_SERIAL_MAX compute kernels."""
    groups = {}
    for row in rows:
        groups.setdefault((row.get("phase"), row.get("pattern_id"), row.get("stream")), []).append(row)
    chains = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: (
            row.get("device_seq_index", row.get("pos", 0)),
            row.get("timestamp") or 0))
        runs = []
        for row in ordered:
            if runs and _adjacent(runs[-1][-1], row):
                runs[-1].append(row)
            else:
                runs.append([row])
        for run in runs:
            for start in range(len(run)):
                compute = 0
                for end in range(start, len(run)):
                    if not _is_data_move(run[end]):
                        compute += 1
                    if compute > EXISTING_SERIAL_MAX:
                        break
                    if compute >= 2:
                        chains.append(("serial", run[start:end + 1]))
    return chains


def _overlaps(left, right):
    if left.get("timestamp") is None or right.get("timestamp") is None:
        return False
    left_end = float(left["timestamp"]) + _duration(left)
    right_end = float(right["timestamp"]) + _duration(right)
    return float(left["timestamp"]) < right_end and float(right["timestamp"]) < left_end


def _parallel_chains(rows, classes):
    small = [row for row in rows if not classes[row["row_id"]]["large"] and not _barrier(row)]
    by_phase = {}
    for row in small:
        by_phase.setdefault(row.get("phase"), []).append(row)
    chains = []
    for group in by_phase.values():
        ordered = sorted(group, key=lambda row: float(row.get("timestamp") or 0))
        used = set()
        for index, row in enumerate(ordered):
            if row["row_id"] in used or row.get("timestamp") is None:
                continue
            window = [row]
            streams = {row.get("stream")}
            for other in ordered[index + 1:]:
                if other["row_id"] in used:
                    continue
                if other.get("stream") in streams:
                    continue
                if all(_overlaps(member, other) for member in window):
                    window.append(other)
                    streams.add(other.get("stream"))
            if len(window) >= 2 and len(streams) >= 2:
                chains.append(("parallel", window))
                used.update(member["row_id"] for member in window)
    return chains


def estimate(members):
    durations = [_duration(row) for row in members]
    chain = sum(durations)
    launches = [min(LAUNCH_US, duration) for duration in durations]
    launch_saved = sum(launches) - min(launches)
    longest = max(range(len(durations)), key=lambda index: durations[index])
    memory_saved = 0.0
    for index, duration in enumerate(durations):
        if index == longest:
            continue
        memory_saved += 0.5 * max(0.0, duration - launches[index])
    fused = max(max(durations), chain - launch_saved - memory_saved)
    speedup = chain / fused if fused else 1.0
    return {
        "chain_us": chain,
        "launch_us": sum(launches),
        "launch_fraction": sum(launches) / chain if chain else 0.0,
        "memory_saved_us": memory_saved,
        "memory_fraction": memory_saved / chain if chain else 0.0,
        "tiny_us": sum(duration for duration in durations if duration < TINY_US),
        "fused_us": fused,
        "speedup": speedup,
        "saved_us": chain - fused,
    }


def _optional_hit(est):
    tiny_fraction = est["tiny_us"] / est["chain_us"] if est["chain_us"] else 0.0
    hits = []
    if est["launch_fraction"] >= LAUNCH_FRACTION_MIN:
        hits.append("launch")
    if est["memory_fraction"] >= MEMORY_FRACTION_MIN:
        hits.append("memory")
    if tiny_fraction >= TINY_FRACTION_MIN:
        hits.append("tiny")
    return hits


def _catalog_kernels(catalog):
    """Catalog entries with op_tags/dtype_tags; name-only entries are tagged here."""
    if not catalog:
        return []
    kernels = catalog.get("kernels") if isinstance(catalog, dict) else catalog
    out = []
    for kernel in kernels or []:
        entry = {"name": kernel} if isinstance(kernel, str) else dict(kernel)
        name = str(entry.get("name") or "")
        if not name:
            continue
        if not entry.get("op_tags"):
            entry["op_tags"] = catalog_lib._op_tags(name)
        if entry.get("dtype_tags") is None:
            entry["dtype_tags"] = sorted(catalog_lib._dtype_tags(name))
        entry.setdefault("sources", [])
        out.append(entry)
    return out


def _chain_tags(compute):
    """Op tags a covering kernel must carry, with GEMM prologue/epilogue by position."""
    tags = set()
    for index, member in enumerate(compute):
        tags |= _tags(_name(member))
        kind = _barrier(member)
        if kind:
            tags.add(kind)
            if kind == "gemm" and index > 0:
                tags.add("gemm_prologue")
            if kind == "gemm" and index < len(compute) - 1:
                tags.add("gemm_epilogue")
    return tags


def existing_matches(compute, kernels):
    """Catalog kernels covering the chain, aiter first, tightest cover first."""
    tags = _chain_tags(compute)
    if len(tags) < 2 or not kernels:
        return [], sorted(tags)
    dtypes = set()
    for member in compute:
        dtypes |= _quant_dtypes(_name(member))
    hits = [k for k in catalog_lib.covers(kernels, sorted(tags), sorted(dtypes))
            if not k["name"].startswith(_NOT_KERNEL_PREFIX)]
    wants_mrope = any("mrope" in _name(member).lower() for member in compute)
    hits.sort(key=lambda k: ("aiter" not in (k.get("sources") or []),
                             ("mrope" in k["name"]) != wants_mrope,
                             len(set(k["op_tags"]) - tags), k["name"]))
    return [{"name": k["name"], "op_tags": sorted(k["op_tags"]),
             "extra_tags": sorted(set(k["op_tags"]) - catalog_lib._expand_implied(tags)),
             "modules": (k.get("modules") or [])[:1]} for k in hits[:5]], sorted(tags)


def baseline_fused(rows, kernels, e2e_us):
    """Trace kernels that already are a catalog fused kernel."""
    found = {}
    for row in rows:
        if _barrier(row) or _is_data_move(row):
            continue
        name = _name(row)
        tags = _tags(name)
        if len(tags) < 2:
            continue
        lowered = name.lower()
        for kernel in kernels:
            kname = kernel["name"].lower()
            if len(kname) < 8 or kname not in lowered:
                continue
            if not tags.issubset(catalog_lib._expand_implied(kernel["op_tags"])):
                continue
            slot = found.setdefault(kernel["name"], {
                "catalog_kernel": kernel["name"], "op_tags": sorted(tags),
                "trace_kernel": name, "phases": {}, "row_ids": []})
            slot["row_ids"].append(row["row_id"])
            phase = row.get("phase")
            share = _duration(row) * _layers(row) / (e2e_us.get(phase) or 1.0)
            slot["phases"][phase] = slot["phases"].get(phase, 0.0) + share
            break
    return sorted(found.values(), key=lambda item: -sum(item["phases"].values()))


def _chain_record(kind, members, classes, e2e_us, kernels):
    phase = members[0].get("phase")
    compute = [member for member in members if not _is_data_move(member)]
    est = estimate(members)
    nlayers = _layers(members[0])
    scaled = est["chain_us"] * nlayers
    denominator = e2e_us.get(phase) or 0.0
    e2e_share = scaled / denominator if denominator else 0.0
    large = any(classes[member["row_id"]]["large"] for member in members)
    need = LARGE_SPEEDUP if large else SMALL_SPEEDUP
    hits = _optional_hit(est)
    matches, tags = existing_matches(compute, kernels)
    existing = bool(matches)
    invalid = ""
    barriers = [index for index, member in enumerate(compute) if _barrier(member)]
    if any(0 < index < len(compute) - 1 for index in barriers):
        invalid = "barrier_inside"
    elif barriers and not existing:
        invalid = "barrier_without_existing_kernel"
    elif len(compute) > SERIAL_MAX and not existing:
        invalid = "too_long_without_existing_kernel"
    if existing:
        difficulty = EXISTING_DIFFICULTY
    else:
        difficulty = max(_op_tier(_name(member)) for member in compute)
    step_saved = scaled - est["fused_us"] * nlayers
    decode_saved = step_saved if phase == "decode" else 0.0
    score = step_saved / difficulty if difficulty else 0.0
    return {
        "kind": kind,
        "phase": phase,
        "phases": [phase],
        "pattern_id": members[0].get("pattern_id"),
        "row_ids": [member["row_id"] for member in members],
        "kernel_names": [_name(member) for member in members],
        "data_move_row_ids": [member["row_id"] for member in members if _is_data_move(member)],
        "op_tags": tags,
        "e2e_share": e2e_share,
        "gate_share": e2e_share,
        "gate_min": EXISTING_CHAIN_E2E_MIN if existing else CHAIN_E2E_MIN,
        "has_large_op": large,
        "required_speedup": need,
        "optional": hits,
        "invalid": invalid,
        "passed": False,
        "difficulty": difficulty,
        "has_existing_kernel": existing,
        "matched_existing_kernels": matches,
        "unknown_kernels": [_name(member) for member in compute if _is_unknown(_name(member))],
        "decode_saved_us": decode_saved,
        "step_saved_us": step_saved,
        "score": score,
        "score_basis": "decode" if phase == "decode" else "chain",
        "_scaled_us": scaled,
        **est,
    }


def _seam_signature(record, rows_by_id):
    parts = []
    for row_id in record["row_ids"]:
        row = rows_by_id[row_id]
        if _is_data_move(row):
            continue
        name = _name(row)
        tags = _tags(name)
        parts.append(tuple(sorted(tags)) if tags else name.split("<")[0].split("(")[0])
    return (record["kind"], record["pattern_id"], tuple(parts))


def _pair_seams(records, rows_by_id, e2e_us, weights):
    """Pair the k-th prefill window with the k-th decode window of the same op sequence."""
    if not weights:
        return
    by_key = {}
    for record in records:
        if record["invalid"] or record["kind"] != "serial":
            continue
        key = _seam_signature(record, rows_by_id)
        by_key.setdefault(key, {}).setdefault(record["phase"], []).append(record)
    denom = sum((e2e_us.get(phase) or 0.0) * weight for phase, weight in weights.items())
    if not denom:
        return
    order = {row_id: index for index, row_id in enumerate(rows_by_id)}
    for phases in by_key.values():
        if "prefill" not in phases or "decode" not in phases:
            continue
        pre = sorted(phases["prefill"], key=lambda item: order[item["row_ids"][0]])
        dec = sorted(phases["decode"], key=lambda item: order[item["row_ids"][0]])
        for left, right in zip(pre, dec):
            share = (left["_scaled_us"] * weights["prefill"]
                     + right["_scaled_us"] * weights["decode"]) / denom
            for item, other in ((left, right), (right, left)):
                item["seam_share"] = share
                item["gate_share"] = share
                item["phases"] = ["prefill", "decode"]
                item["seam_partner_key"] = tuple(other["row_ids"])
                item["partner_row_ids"] = list(other["row_ids"])


def screen_candidates(rows, gfx="gfx942", catalog=None, workload=None):
    """Every chain that passed the gates, ordered by estimated time saved."""
    e2e_us = phase_e2e_us(rows)
    classes = {}
    for row in rows:
        classes[row["row_id"]] = classify_kernel(row, e2e_us.get(row.get("phase")) or 0.0, gfx)
    kernels = _catalog_kernels(catalog)
    rows_by_id = {row["row_id"]: row for row in rows}
    found = []
    for kind, members in _serial_chains(rows) + _parallel_chains(rows, classes):
        found.append(_chain_record(kind, members, classes, e2e_us, kernels))
    weights = phase_weights(rows, workload)
    _pair_seams(found, rows_by_id, e2e_us, weights)
    for item in found:
        item["passed"] = (not item["invalid"]
                          and item["gate_share"] >= item["gate_min"]
                          and item["speedup"] > item["required_speedup"]
                          and bool(item["optional"]))
    passed = [item for item in found if item["passed"]]
    initial = sorted(passed, key=lambda item: (-item["step_saved_us"], item["row_ids"]))
    for index, item in enumerate(initial, 1):
        item["candidate_id"] = "c%02d" % index
    by_key = {tuple(item["row_ids"]): item for item in initial}
    for item in initial:
        partner = by_key.get(item.get("seam_partner_key") or ())
        if partner:
            item["partner_candidate_id"] = partner["candidate_id"]
    rejected = {}
    for item in found:
        if item["passed"]:
            continue
        reason = item["invalid"] or ("below_time_gate" if item["gate_share"] < item["gate_min"]
                                     else "below_speedup_or_optional_gate")
        rejected[reason] = rejected.get(reason, 0) + 1
    near = [item for item in found if not item["passed"] and not item["invalid"]
            and NEAR_MISS_MIN <= item["gate_share"] < item["gate_min"]]
    near.sort(key=lambda item: -item["gate_share"])
    fused = baseline_fused(rows, kernels, e2e_us)
    for item in fused:
        # A baseline fused kernel that still sits in a passing chain leaves work
        # behind it (e.g. add_rmsnorm_quant followed by a separate group quant).
        rows_in = set(item["row_ids"])
        item["still_in_candidates"] = sorted({
            cand["candidate_id"] for cand in initial if rows_in & set(cand["row_ids"])})
    return {
        "e2e_us": e2e_us,
        "phase_weights": weights,
        "kernel_class": classes,
        "considered": len(found),
        "rejected": rejected,
        "near_misses": near[:NEAR_MISS_LIMIT],
        "baseline_fused": fused,
        "initial_topk": initial,
    }


def score_pass(candidates, top_k=FINAL_K):
    """One pass: Top-K by Score, a shared kernel keeps the higher score."""
    return _resolve_conflicts(candidates)[:top_k]


def merge_passes(passes, top_k=FINAL_K):
    """In both passes beats in one; then the smaller rank sum wins.

    A seam partner (the same seam in the other phase) rides on its kept
    partner's entry instead of taking a slot of its own.
    """
    missing = top_k + 1
    by_key = {}
    for pass_index, ranked in enumerate(passes):
        for rank, item in enumerate(ranked, 1):
            key = tuple(item["row_ids"])
            slot = by_key.setdefault(key, {"item": item, "ranks": [missing] * len(passes)})
            slot["ranks"][pass_index] = rank
    merged = []
    for slot in by_key.values():
        item = dict(slot["item"])
        item["pass_ranks"] = slot["ranks"]
        item["pass_hits"] = sum(1 for rank in slot["ranks"] if rank != missing)
        item["rank_sum"] = sum(slot["ranks"])
        merged.append(item)
    merged.sort(key=lambda item: (-item["pass_hits"], item["rank_sum"],
                                  -item["score"], item["row_ids"]))
    kept = []
    kept_by_key = {}
    taken = set()
    for item in merged:
        ids = set(item["row_ids"])
        if ids & taken:
            continue
        partner = kept_by_key.get(item.get("seam_partner_key") or ())
        if partner is not None:
            partner["partner_row_ids"] = list(item["row_ids"])
            partner["partner_candidate_id"] = item.get("candidate_id")
            partner["phases"] = sorted({*partner.get("phases", []), item["phase"]})
            taken |= ids
            continue
        if len(kept) == top_k:
            continue
        kept.append(item)
        kept_by_key[tuple(item["row_ids"])] = item
        taken |= ids
    for item in kept:
        if item.get("partner_row_ids") and tuple(item["partner_row_ids"]) not in by_key:
            # Partner lost every pass: this entry covers its own phase only.
            item["phases"] = [item["phase"]]
            item.pop("partner_row_ids", None)
    return kept


def screen(rows, gfx="gfx942", catalog=None, top_k=FINAL_K, passes=SCREEN_PASSES,
           extra_passes=None, workload=None):
    found = screen_candidates(rows, gfx=gfx, catalog=catalog, workload=workload)
    ranked = [score_pass(found["initial_topk"], top_k) for _ in range(max(1, passes))]
    ranked.extend(extra_passes or [])
    final = merge_passes(ranked, top_k)
    for index, item in enumerate(final, 1):
        item["exec_id"] = "e%02d" % index
        item["candidate_ids"] = [item["candidate_id"]] + (
            [item["partner_candidate_id"]] if item.get("partner_row_ids")
            and item.get("partner_candidate_id") else [])
        item["estimated_speedup"] = item["speedup"]
        item["compare_existing"] = item["has_existing_kernel"]
        item["self_author"] = True
    found["passes"] = ranked
    found["execution_list"] = final
    return found


def _resolve_conflicts(ranked_by_benefit):
    """Higher score wins when two chains contain the same kernel."""
    ordered = sorted(ranked_by_benefit, key=lambda item: (-item["score"], -item["step_saved_us"], item["row_ids"]))
    kept = []
    taken = set()
    for item in ordered:
        ids = set(item["row_ids"])
        if ids & taken:
            continue
        kept.append(item)
        taken |= ids
    return kept


def _short(name):
    return name.split("<")[0].split("(")[0][:60]


def _markdown(result):
    lines = ["# Fusion Top-K", "",
             "Initial list is every chain that passed the gates, ordered by estimated time saved.",
             "Each pass is Score = time saved / difficulty. %d passes merged into K=%d; every row is optimized."
             % (len(result.get("passes") or []), len(result["execution_list"])), ""]
    lines.append("## Final")
    lines.append("")
    lines.append("| exec | kind | phase | score | difficulty | gate share | pass ranks | speedup | existing kernel | kernels |")
    lines.append("|---|---|---|---:|---:|---:|---|---:|---|---|")
    for item in result["execution_list"]:
        existing = ", ".join(match["name"] for match in item.get("matched_existing_kernels") or []) or "no"
        lines.append("| %s | %s | %s | %.3f | %.1f | %.2f%% | %s | %.3f | %s | %s |" % (
            item["exec_id"], item["kind"], "+".join(item.get("phases") or [item["phase"]]),
            item["score"], item["difficulty"], 100.0 * item.get("gate_share", item["e2e_share"]),
            "/".join(str(rank) for rank in item.get("pass_ranks", [])),
            item["speedup"], existing, ", ".join(item["kernel_names"])))
    lines.append("")
    if result.get("baseline_fused"):
        lines.append("## Already fused in the baseline")
        lines.append("")
        lines.append("| catalog kernel | op tags | share by phase | trace kernel | still in candidates |")
        lines.append("|---|---|---|---|---|")
        for item in result["baseline_fused"]:
            shares = ", ".join("%s %.2f%%" % (phase, 100.0 * share)
                               for phase, share in sorted(item["phases"].items()))
            lines.append("| %s | %s | %s | %s | %s |" % (
                item["catalog_kernel"], "/".join(item["op_tags"]), shares, _short(item["trace_kernel"]),
                ", ".join(item.get("still_in_candidates") or []) or "no (nothing left to fuse)"))
        lines.append("")
    if result.get("near_misses"):
        lines.append("## Below the time gate")
        lines.append("")
        lines.append("| phase | gate share | needed | existing kernel | kernels |")
        lines.append("|---|---:|---:|---|---|")
        for item in result["near_misses"]:
            existing = ", ".join(match["name"] for match in item.get("matched_existing_kernels") or []) or "no"
            lines.append("| %s | %.2f%% | %.0f%% | %s | %s |" % (
                "+".join(item.get("phases") or [item["phase"]]), 100.0 * item["gate_share"],
                100.0 * item["gate_min"], existing,
                ", ".join(_short(name) for name in item["kernel_names"])))
        lines.append("")
    unknown = sorted({name for item in result["initial_topk"] for name in item.get("unknown_kernels") or []})
    if unknown:
        lines.append("## Unclassified kernels (difficulty defaulted to %.1f)" % UNKNOWN_DIFFICULTY)
        lines.append("")
        lines.extend("- `%s`" % _short(name) for name in unknown)
        lines.append("")
    if result.get("rejected"):
        lines.append("Rejected windows: " + ", ".join(
            "%s %d" % (reason, count) for reason, count in sorted(result["rejected"].items())))
        lines.append("")
    return "\n".join(lines) + "\n"


def _public(item):
    keep = (
        "candidate_id", "exec_id", "candidate_ids", "kind", "phase", "phases", "pattern_id",
        "row_ids", "partner_row_ids", "partner_candidate_id", "data_move_row_ids",
        "kernel_names", "op_tags", "e2e_share", "seam_share", "gate_share", "gate_min",
        "has_large_op", "required_speedup", "optional", "difficulty", "has_existing_kernel",
        "matched_existing_kernels", "unknown_kernels", "decode_saved_us",
        "step_saved_us", "score", "score_basis", "chain_us", "launch_fraction",
        "memory_fraction", "speedup", "estimated_speedup", "saved_us",
        "pass_ranks", "pass_hits", "rank_sum", "compare_existing", "self_author",
    )
    return {key: item[key] for key in keep if key in item}


def _int_or_none(value):
    try:
        return int(value) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-table", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--catalog", default="")
    parser.add_argument("--gfx", default="gfx942")
    parser.add_argument("--top-k", type=int, default=FINAL_K)
    parser.add_argument("--passes", type=int, default=SCREEN_PASSES)
    parser.add_argument("--isl", default="")
    parser.add_argument("--osl", default="")
    parser.add_argument("--conc", default="")
    parser.add_argument("--md", default="")
    args = parser.parse_args(argv)
    with open(args.semantic_table) as handle:
        table = json.load(handle)
    catalog = None
    if args.catalog and os.path.isfile(args.catalog):
        with open(args.catalog) as handle:
            catalog = json.load(handle)
    workload = {"isl": _int_or_none(args.isl), "osl": _int_or_none(args.osl),
                "conc": _int_or_none(args.conc)}
    result = screen(load_rows(table), gfx=args.gfx, catalog=catalog, top_k=args.top_k,
                    passes=args.passes, workload=workload)
    os.makedirs(args.out_dir, exist_ok=True)
    candidates_path = os.path.join(args.out_dir, "fusion_candidates.json")
    topk_path = os.path.join(args.out_dir, "fusion_topk.json")
    md_path = args.md or os.path.join(os.path.dirname(args.out_dir.rstrip("/")), "03_FUSION_TOPK.md")
    candidates = {
        "schema_version": 2,
        "phase": "screen",
        "considered": result["considered"],
        "passed": len(result["initial_topk"]),
        "workload": workload,
        "phase_weights": result["phase_weights"],
        "rejected": result["rejected"],
        "baseline_fused": result["baseline_fused"],
        "near_misses": [_public(item) for item in result["near_misses"]],
        "candidates": [_public(item) for item in result["initial_topk"]],
    }
    topk = {
        "schema_version": 2,
        "phase": "final_topk",
        "top_k": args.top_k,
        "passes": [[item["candidate_id"] for item in ranked] for ranked in result["passes"]],
        "execution_list": [_public(item) for item in result["execution_list"]],
    }
    with open(candidates_path, "w") as handle:
        json.dump(candidates, handle, indent=2)
    with open(topk_path, "w") as handle:
        json.dump(topk, handle, indent=2)
    parent = os.path.dirname(md_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(md_path, "w") as handle:
        handle.write(_markdown(result))
    summary = {
        "status": "pass" if topk["execution_list"] else "no_candidate",
        "fusion_candidates_json": os.path.abspath(candidates_path),
        "fusion_topk_json": os.path.abspath(topk_path),
        "fusion_topk_md": os.path.abspath(md_path),
        "execution_list": topk["execution_list"],
        "candidate_count": candidates["passed"],
    }
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
