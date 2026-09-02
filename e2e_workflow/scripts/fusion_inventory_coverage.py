#!/usr/bin/env python3
"""Phase 2.1 companion gate: close the denominator over the PROVIDER INVENTORY.

Every other gate in this pipeline is computed over things somebody already proposed.
`fusion_candidate_harness.py` checks that each candidate is well-founded;
`fusion_unitside_harness.py` checks that every candidate got a verdict. Neither can
see a fusion kernel that exists in the installed providers, matches work the trace
actually does, and was simply never turned into a candidate — that kernel is absent
from every denominator, so a run that enumerated 12 of the relevant APIs renders
identically to one that enumerated all of them.

Observed on DSR1 2026-08-31 (KB blind test): `fused_flatten_fp8_group_quant` sat in
`available_fusion_kernels.json` the whole time, fires on EVERY layer, and is worth
+1.6%/+2.1% e2e. It appears in no candidate, no board row, and no report — because
nothing ever asked. This gate asks.

WHY THIS IS THE WORKLOAD-GENERAL FIX
------------------------------------
A knowledge base of past wins only helps on workloads you have already run. What
transfers to a NEW workload is not the answers — it is the requirement to enumerate
from the provider inventory ∩ the trace, ranked by cost, and to say out loud why each
high-cost entry was not pursued. The KB then becomes an accelerator, not a
prerequisite.

WHAT COUNTS AS "RELEVANT"
-------------------------
A kernel is in scope when it is a FUSION (it does more than one thing) and every kind
of work it does is work this trace performs:

  fusion-ness  — op_tags >= 2, OR the name carries a fusion connective
                 (`fused_`, `_and_`, `_cat_`, `_absorb`). The tag vocabulary
                 under-tags real fusions: `fused_flatten_fp8_group_quant` is tagged
                 `['quant']` alone, so a tags-only test would have missed the very
                 kernel this gate exists to catch. Tags and name tokens are UNIONED.
  relevance    — with `--table`: some window of CONSECUTIVE trace rows on one stream
                 contains every kind of work the kernel does, across >=2 distinct
                 stages. Without `--table` this degrades to a flat "does the trace do
                 this kind of work anywhere" test, which admits ~everything and cannot
                 rank. Pass the table.

RANK = WHAT IT REMOVES x LAYER COVERAGE
---------------------------------------
NOT the cost of the window. A fusion does not delete the dominant compute it stands
next to — the GEMM still runs. Charging each kernel the window total handed every
candidate the GEMM's cost and saturated the entire board at one number (DSR1: 332 rows
tied). Rank is therefore the covered rows MINUS the single most expensive covered row —
the helper traffic the fusion actually deletes — times `pattern_layer_count`. A cheap
op that repeats over 61 layers outranks an expensive one that happens once, which is
"prefer the seam with the widest layer coverage" computed rather than remembered.

A BUDGET CUT IS A REAL CUT
--------------------------
This gate does not guarantee the small fish is in your top-N. On DSR1 the missed
`fused_flatten_fp8_group_quant` ranks #135 at 2818us against a #1 of 79543us — it is
genuinely small, and it is genuinely worth +1.6% e2e. What the gate changes is that the
cut is now a NUMBER YOU CHOSE with the tail printed underneath it, instead of an
invisible accident. Set `--budget` to what you can actually answer, and read the tail.

DISPOSITION IS MANDATORY, NOT THE PURSUIT
-----------------------------------------
This gate never says "you must fuse this". It says you must ANSWER. Each in-scope
kernel must end in one of:

  * enumerated   — some candidate cites it in `existing_apis`
  * --dispose NAME=REASON — a stated reason not to pursue it
  * un_dispositioned  -> the gate FAILS

Usage:
  fusion_inventory_coverage.py --inventory available_fusion_kernels.json \\
      --candidates fusion_candidates.json --out-md 02b_INVENTORY_COVERAGE.md \\
      --out-json fusion_inventory_coverage.json [--budget 25] \\
      [--dispose NAME=REASON ...] [--allow-undispositioned]

`--budget N` requires dispositions only for the top-N by rank, and still PRINTS the
tail so the cut is visible. It bounds the work without hiding the denominator.
"""

import argparse
import json
import os
import re
import sys


FUSION_NAME_MARKERS = ("fused_", "_and_", "_cat_", "_absorb", "fused", "_fused")

# Stage names as they appear in `stage_inventory[].stage`, mapped to the op_tag
# vocabulary of available_fusion_kernels.json. Kept explicit rather than inferred so a
# vocabulary drift shows up as an unmapped stage in the report instead of silently
# shrinking the in-scope set.
STAGE_TO_TAGS = {
    "quant": {"quant", "cast"},
    "gemm": {"gemm", "gemm_prologue", "gemm_epilogue"},
    "elementwise": {"activation", "add_residual", "cast", "layout"},
    "norm": {"norm", "add_residual"},
    "communication": {"allreduce"},
    "kv_cache": {"kv_cache", "rope"},
    "moe": {"moe", "topk"},
    "memory": {"layout", "cast"},
    "attn": {"rope", "kv_cache"},
    "activation": {"activation"},
    "topk": {"topk", "moe"},
}

# Name tokens used only when a kernel declares no op_tags at all.
TAG_TOKENS = {
    "norm": ("rmsnorm", "layernorm", "_norm"),
    "quant": ("quant", "fp8", "fp4", "mxfp"),
    "rope": ("rope", "mrope"),
    "kv_cache": ("cache", "kv"),
    "allreduce": ("allreduce", "all_reduce", "ar_"),
    "moe": ("moe", "expert"),
    "topk": ("topk",),
    "activation": ("silu", "gelu", "act_"),
    "gemm": ("gemm", "mm", "matmul", "bmm"),
    "layout": ("flatten", "shuffle", "transpose", "cat", "concat", "pad"),
    "add_residual": ("residual", "add_"),
}


# Wrapper/alias affixes that produce several inventory entries for ONE kernel:
# `fused_allreduce_rmsnorm_quant`, `..._`, `..._fake`, `fake_...`, and the
# `tensor_model_parallel_...` dispatcher in front of it are four names for one thing.
# Left un-collapsed they generated 368 "open" rows on DSR1 -- and a gate demanding 368
# answers is a gate that gets `--allow-undispositioned` on the first run, which is the
# same as not having it.
ALIAS_PREFIXES = ("fake_", "tensor_model_parallel_", "attention_tensor_model_parallel_",
                  "torch_", "aiter_", "triton_", "ck_", "asm_")
ALIAS_SUFFIXES = ("_fake", "_impl", "_wrapper", "_ASM", "_asm", "_CK", "_ck",
                  "_triton", "_torch", "_")


def _canonical(name):
    """Collapse wrapper/alias spellings onto one key. Case-insensitive, applied
    repeatedly because affixes stack (`fake_..._fake`)."""
    key = str(name).split(".")[-1]
    changed = True
    while changed:
        changed = False
        low = key.lower()
        for pre in ALIAS_PREFIXES:
            if low.startswith(pre) and len(key) > len(pre):
                key, changed = key[len(pre):], True
                break
        if changed:
            continue
        for suf in ALIAS_SUFFIXES:
            if key.endswith(suf) and len(key) > len(suf):
                key, changed = key[:-len(suf)], True
                break
    return key.lower()


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _is_fusion(kernel):
    """Two independent witnesses; either suffices.

    Tags alone are not enough (they under-tag), and the name alone is not enough
    (`fused_moe` is one op in this vocabulary). Requiring BOTH would reproduce the
    original miss, so this is deliberately an OR."""
    tags = kernel.get("op_tags") or []
    if len(tags) >= 2:
        return "op_tags>=2"
    name = str(kernel.get("name") or "").lower()
    for marker in FUSION_NAME_MARKERS:
        if marker in name:
            return "name marker %r" % marker
    return None


def _name_tags(name):
    name = str(name).lower()
    out = set()
    for tag, tokens in TAG_TOKENS.items():
        if any(tok in name for tok in tokens):
            out.add(tag)
    return out


def _seam_windows(table_payload, max_span=4):
    """Every adjacent op window the trace actually contains, with its cost.

    Relevance-by-tag-universe ("this kernel does quant, the trace does quant") is far
    too loose: on DSR1 it admitted 332 kernels, ranked them by whole-stage cost so that
    every kernel of a given stage tied, and buried the one that mattered at #238. A
    fusion is only relevant if the trace has the ADJACENT ops it would replace -- so
    the join is against windows of consecutive rows, not against a flat tag set, and
    the rank is the cost of the best window it could actually serve.

    Cost is `sum(duration_us) * pattern_layer_count`: an op pair that repeats over 61
    layers outranks a more expensive pair that happens once. That is "layer coverage x
    cost" computed rather than remembered."""
    windows = []
    for table in table_payload.get("tables", []) or []:
        try:
            layers = int(table.get("pattern_layer_count") or 1)
        except (TypeError, ValueError):
            layers = 1
        rows = sorted(
            [r for r in (table.get("rows") or []) if r.get("stage")],
            key=lambda r: (str(r.get("stream")), r.get("pos") or 0))
        for start in range(len(rows)):
            stream = rows[start].get("stream")
            tags, cost, stages, members = set(), 0.0, [], []
            for end in range(start, min(start + max_span, len(rows))):
                row = rows[end]
                if row.get("stream") != stream:
                    break
                stage = str(row.get("stage"))
                tags |= STAGE_TO_TAGS.get(stage, set())
                stages.append(stage)
                try:
                    dur = float(row.get("duration_us") or 0.0)
                except (TypeError, ValueError):
                    dur = 0.0
                cost += dur
                members.append((stage, dur))
                if end > start:
                    windows.append({
                        "phase": table.get("phase"),
                        "pattern_id": table.get("pattern_id"),
                        "stages": list(stages),
                        "members": list(members),
                        "tags": set(tags),
                        "layers": layers,
                        "cost_us": cost * layers,
                    })
    return windows


def _trace_profile(candidates_payload):
    """Cost and layer coverage per stage, from the candidates' own member rows.

    `duration_us` is per-occurrence; `pattern_layer_count` says how many layers the
    pattern repeats over. Their product is the quantity that should order candidate
    generation -- a cheap op on 61 layers outranks an expensive one on 1."""
    stage_cost = {}
    stage_layers = {}
    for cand in candidates_payload.get("candidates", []) or []:
        layers = cand.get("pattern_layer_count") or 1
        try:
            layers = int(layers)
        except (TypeError, ValueError):
            layers = 1
        for member in cand.get("members", []) or []:
            stage = str(member.get("stage") or "unknown")
            try:
                dur = float(member.get("duration_us") or 0.0)
            except (TypeError, ValueError):
                dur = 0.0
            stage_cost[stage] = stage_cost.get(stage, 0.0) + dur * layers
            stage_layers[stage] = max(stage_layers.get(stage, 0), layers)
    # Stages seen structurally but carrying no member cost still count as PRESENT --
    # relevance is about what the trace does, not about what happened to be expensive.
    for entry in candidates_payload.get("stage_inventory", []) or []:
        stage = str(entry.get("stage") or "unknown")
        stage_cost.setdefault(stage, 0.0)
        stage_layers.setdefault(stage, 1)
    return stage_cost, stage_layers


def _cited_apis(candidates_payload):
    cited = {}
    for cand in candidates_payload.get("candidates", []) or []:
        for api in cand.get("existing_apis") or []:
            name = api.get("name")
            if name:
                cited.setdefault(str(name).split(".")[-1], []).append(
                    cand.get("candidate_id"))
    return cited


def _parse_kv(items):
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit("--dispose expects NAME=REASON, got %r" % item)
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not value:
            raise SystemExit("--dispose %r has an empty reason" % key)
        out[key] = value
    return out


def audit(inventory_path, candidates_path, dispositions=None, budget=None,
          require_disposition=True, table_path=None):
    inv = _load(inventory_path)
    cands = _load(candidates_path)
    seam_windows = _seam_windows(_load(table_path)) if table_path else []
    dispositions = dict(dispositions or {})
    kernels = inv.get("kernels") or []
    stage_cost, stage_layers = _trace_profile(cands)
    trace_tags = set()
    unmapped_stages = []
    for stage in stage_cost:
        mapped = STAGE_TO_TAGS.get(stage)
        if mapped is None:
            unmapped_stages.append(stage)
            continue
        trace_tags |= mapped
    cited = _cited_apis(cands)
    candidate_stages = {
        str(member.get("stage"))
        for cand in (cands.get("candidates") or [])
        for member in (cand.get("members") or [])
        if member.get("stage")
    }
    canon_cited = {}
    for name, cids in cited.items():
        canon_cited.setdefault(_canonical(name), []).extend(cids)

    rows = []
    for kernel in kernels:
        name = str(kernel.get("name") or "")
        short = name.split(".")[-1]
        why_fusion = _is_fusion(kernel)
        if not why_fusion:
            continue
        # The declared op_tags routinely under-describe a fusion: the kernel this gate
        # exists to catch, `fused_flatten_fp8_group_quant`, is tagged `['quant']` alone
        # even though its name says it also flattens. Union the two sources rather than
        # choosing one -- choosing tags reproduces the original miss.
        declared = set(kernel.get("op_tags") or [])
        tags = declared | _name_tags(short)
        tag_source = "op_tags+name" if declared else "name tokens"
        if not tags:
            continue
        best = None
        if seam_windows:
            # A kernel is relevant iff some real adjacent window covers everything it
            # does AND it collapses at least two distinct kinds of work in that window.
            for window in seam_windows:
                if not tags.issubset(window["tags"]):
                    continue
                hit = [(stage, dur) for stage, dur in window["members"]
                       if tags & STAGE_TO_TAGS.get(stage, set())]
                if len({stage for stage, _ in hit}) < 2:
                    continue
                # A fusion does not delete the dominant compute in the window -- the
                # GEMM still has to happen. What it deletes is the surrounding helper
                # traffic. Ranking by the WINDOW total therefore hands every kernel the
                # cost of the biggest op it merely stands next to, and on DSR1 that
                # saturated the whole board at one number. Charge each kernel only the
                # rows it actually removes: everything it covers except the single most
                # expensive covered row.
                removable = sum(d for _, d in hit) - max(d for _, d in hit)
                value = removable * window["layers"]
                if best is None or value > best[0]:
                    best = (value, window, hit)
            if best is None:
                continue
            rank, win, hit = best
            touched = sorted({stage for stage, _ in hit})
            layers = win["layers"]
        else:
            if not tags.issubset(trace_tags):
                continue
            rank = 0.0
            touched = []
            layers = 0
            for stage, cost in stage_cost.items():
                if tags & STAGE_TO_TAGS.get(stage, set()):
                    rank += cost
                    touched.append(stage)
                    layers = max(layers, stage_layers.get(stage, 0))
        canon = _canonical(short)
        cite_hits = cited.get(short) or cited.get(canon) or canon_cited.get(canon)
        if cite_hits:
            status, reason = "enumerated", "cited by %s" % ", ".join(
                str(c) for c in cite_hits[:3])
        elif short in dispositions or canon in dispositions:
            status, reason = "dispositioned", dispositions.get(
                short, dispositions.get(canon))
        else:
            status, reason = "un_dispositioned", (
                "in the installed providers, matches trace stages %s, never proposed "
                "and never explained" % "/".join(sorted(touched)))
        # The stage this kernel is PRIMARILY about: the most expensive one it touches.
        # Used to stratify the budget so a loud family (collectives) cannot crowd every
        # other stage out of the answerable window -- which is how a cheap
        # every-layer quant fusion ends up at rank #276 and never gets looked at.
        primary = max(touched, key=lambda s: stage_cost.get(s, 0.0)) if touched \
            else "unknown"
        rows.append({
            "canonical": canon,
            "primary_stage": primary,
            "name": short,
            "modules": kernel.get("modules") or [],
            "op_tags": sorted(kernel.get("op_tags") or []),
            "tags_used": sorted(tags),
            "tag_source": tag_source,
            "fusion_witness": why_fusion,
            "stages_touched": sorted(touched),
            "layer_span": layers,
            "rank_us": round(rank, 3),
            "status": status,
            "reason": reason,
        })
    # ---- collapse aliases: one FAMILY per canonical key -------------------------
    families = {}
    for row in rows:
        fam = families.get(row["canonical"])
        if fam is None:
            fam = dict(row)
            fam["aliases"] = []
            families[row["canonical"]] = fam
        else:
            fam["aliases"].append(row["name"])
            fam["rank_us"] = max(fam["rank_us"], row["rank_us"])
            fam["layer_span"] = max(fam["layer_span"], row["layer_span"])
            fam["stages_touched"] = sorted(
                set(fam["stages_touched"]) | set(row["stages_touched"]))
            # An answer on ANY spelling answers the family; enumerated beats
            # dispositioned beats open, so the family takes the strongest answer.
            order = {"enumerated": 2, "dispositioned": 1, "un_dispositioned": 0}
            if order[row["status"]] > order[fam["status"]]:
                fam["status"], fam["reason"] = row["status"], row["reason"]
    rows = sorted(families.values(), key=lambda r: (-r["rank_us"], r["name"]))

    # ---- stratified budget: top-N per PRIMARY STAGE, not top-N overall ----------
    # A flat top-N is dominated by whichever stage is most expensive, so every kernel
    # of every other stage falls outside the answerable window regardless of merit.
    # Budgeting per stage guarantees each kind of work the trace does gets its best
    # un-enumerated candidates looked at.
    seen_per_stage = {}
    for index, row in enumerate(rows):
        row["rank"] = index + 1
        stage = row["primary_stage"]
        seen = seen_per_stage.get(stage, 0)
        row["stage_rank"] = seen + 1
        seen_per_stage[stage] = seen + 1
        row["forced_overlap"] = bool(
            row["status"] == "un_dispositioned" and
            set(row.get("stages_touched") or []) & candidate_stages
        )
        row["in_budget"] = (
            budget is None or row["stage_rank"] <= budget or
            row["forced_overlap"]
        )

    open_rows = [r for r in rows
                 if r["status"] == "un_dispositioned" and r["in_budget"]]
    deferred_tail = [r for r in rows
                     if r["status"] == "un_dispositioned" and not r["in_budget"]]
    errors = []
    for row in open_rows:
        errors.append(
            "inventory-coverage gap #%d: `%s` (%s) — %s. Propose a candidate citing "
            "it, or --dispose %s='<why not>'."
            % (row["rank"], row["name"], "/".join(row["modules"][:1]) or "?",
               row["reason"], row["name"]))
    if unmapped_stages:
        errors.append(
            "stage(s) %s are present in the trace but absent from STAGE_TO_TAGS — "
            "inventory relevance was computed WITHOUT them, so this audit is "
            "incomplete. Extend the map." % sorted(set(unmapped_stages)))

    counts = {"in_scope": len(rows)}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "schema_version": 1,
        "phase": "inventory_coverage",
        "seam_join": bool(seam_windows),
        "seam_windows": len(seam_windows),
        "status": "fail" if (errors and require_disposition) else "pass",
        "inventory_json": os.path.abspath(inventory_path),
        "candidates_json": os.path.abspath(candidates_path),
        "kernels_scanned": len(kernels),
        "providers_scanned": inv.get("providers_scanned"),
        "trace_tags": sorted(trace_tags),
        "candidate_stages": sorted(candidate_stages),
        "unmapped_stages": sorted(set(unmapped_stages)),
        "budget": budget,
        "counts": counts,
        "errors": errors,
        "open_count": len(open_rows),
        "deferred_tail_count": len(deferred_tail),
        "rows": rows,
    }


def _esc(value):
    return str(value if value is not None else "").replace("|", "\\|")


def render_markdown(result):
    lines = ["# Fusion 候选枚举覆盖率（provider inventory ∩ trace）", ""]
    lines.append(
        "本表的分母**不是**已经提出的候选，而是**装在容器里的 provider 里、并且做的事情"
        "这条 trace 确实在做**的每一个 fusion kernel。别的闸门都只能审已经被提出来的东西；"
        "一个存在于 provider、匹配 trace、但从没被提成候选的核，在所有别的表里都不存在。")
    lines.append("")
    lines.append(
        "扫描 %s 个 kernel（providers: %s）→ 在范围内 **%d** 个：已枚举 **%d** / "
        "已给出不做的理由 %d / **未答复 %d**。"
        % (result["kernels_scanned"], ", ".join(result.get("providers_scanned") or []),
           result["counts"].get("in_scope", 0),
           result["counts"].get("enumerated", 0),
           result["counts"].get("dispositioned", 0),
           result["counts"].get("un_dispositioned", 0)))
    lines.append("")
    if result["open_count"]:
        lines.append(
            "> 🔴 **%d 个未答复**。这个闸门从不要求你去融合它；它要求你**回答**。"
            "排序 = **该核能删掉的那部分**（覆盖到的行减去其中最贵的一行——那行是它"
            "删不掉的主计算）× 层覆盖。预算切口是你自己选的一个数字，尾巴在下面列着；"
            "以前那个切口是隐形的。" % result["open_count"])
        lines.append("")
    if result.get("deferred_tail_count"):
        lines.append("预算外（仍列出，切口可见）：%d 个。"
                     % result["deferred_tail_count"])
        lines.append("")
    lines.append("| # | kernel | 排序依据 us | 层跨度 | 触及 stage | 状态 | 说明 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in result["rows"]:
        mark = "" if row["in_budget"] else " *(预算外)*"
        lines.append("| %d | `%s`%s | %.1f | %d | %s | **%s** | %s |" % (
            row["rank"], _esc(row["name"]), mark, row["rank_us"], row["layer_span"],
            _esc("/".join(row["stages_touched"])), row["status"],
            _esc(row["reason"])))
    lines.append("")
    if result["errors"]:
        lines.append("## 未答复项（本闸门因此 fail）")
        for err in result["errors"]:
            lines.append("- %s" % err)
        lines.append("")
    return "\n".join(lines) + "\n"


def run(inventory_path, candidates_path, out_md, out_json, dispositions=None,
        budget=None, require_disposition=True, table_path=None):
    result = audit(inventory_path, candidates_path, dispositions, budget,
                   require_disposition, table_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    with open(out_md, "w") as fh:
        fh.write(render_markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True,
                        help="available_fusion_kernels.json from Phase 2.0")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--table", default=None,
                        help="pattern_layer_kernel_table.json. STRONGLY recommended: "
                             "without it relevance degrades to a flat tag-universe "
                             "match, which cannot rank and admits ~everything")
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--budget", type=int, default=None,
                        help="require a disposition only for the top-N by rank; the "
                             "tail is still printed")
    parser.add_argument("--dispose", action="append", default=[],
                        metavar="NAME=REASON",
                        help="state why an in-scope kernel is not being pursued "
                             "(repeatable)")
    parser.add_argument("--allow-undispositioned", action="store_true",
                        help="report the gaps without failing the gate")
    args = parser.parse_args()
    result = run(args.inventory, args.candidates, args.out_md, args.out_json,
                 _parse_kv(args.dispose), args.budget,
                 require_disposition=not args.allow_undispositioned,
                 table_path=args.table)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "open": result["open_count"],
                      "top_open": [r["name"] for r in result["rows"]
                                   if r["status"] == "un_dispositioned"][:10]},
                     indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
