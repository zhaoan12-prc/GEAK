#!/usr/bin/env python3
"""Reuse an earlier run's KernelFusion semantic result instead of re-capturing it.

KernelFusion normally captures a fusion trace, builds the semantic layer table and
completes its shapes before screening -- the expensive, server-bound part of the phase.
When a previous run of the same model / stack / workload already produced that table,
the workflow may start from it (args.fusion_reuse_semantics_dir). Everything downstream
(screen -> unit-side -> apply-back -> greedy stacking -> write-back) still runs normally,
and every artifact of the new run must live in the NEW eval dir, so this script copies
the inputs in and records where they came from:

  <eval-dir>/fusion_capture/reused/
      pattern_layer_kernel_table.json     the semantic table the screen reads
      STRUCTURAL_LAYER_PATTERNS.json      structural patterns next to the table (if any)
      profile_trace_manifest.json         the trace manifest (if any)
      available_fusion_kernels.json       the kernel catalog (if any)
      01_SEMANTIC.md                      the human report (if any)
      reused_from.json                    source path + sha256 of every copied file
  <eval-dir>/01_SEMANTIC.md               the report, at the eval-dir root as usual

Source resolution (explicit overrides win):
  table     --semantic-table, else <src>/profile/round_fusion_capture/semantics_1_2/
            pattern_layer_kernel_table.json, else .../semantics/pattern_layer_kernel_table.json
  manifest  --trace-manifest, else <src>/fusion_capture/profile_trace_manifest.json
  catalog   <src>/fusion_capture/fusion/available_fusion_kernels.json
  report    <src>/01_SEMANTIC.md

The decode-coverage check still applies: the printed JSON carries
semantic_phase_coverage.coverage() of the copied table, and the workflow marks the round
phase-blind when a required phase is missing (there is no re-capture in reuse mode).

Prints ONE JSON object on its last stdout line; exit 1 when no table can be found.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import semantic_phase_coverage  # noqa: E402

TABLE_CANDIDATES = (
    ("profile", "round_fusion_capture", "semantics_1_2", "pattern_layer_kernel_table.json"),
    ("profile", "round_fusion_capture", "semantics", "pattern_layer_kernel_table.json"),
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_sources(src_dir, table=None, manifest=None):
    src_dir = os.path.abspath(src_dir) if src_dir else ""
    out = {}
    if table:
        out["table"] = os.path.abspath(table)
    elif src_dir:
        for parts in TABLE_CANDIDATES:
            path = os.path.join(src_dir, *parts)
            if os.path.isfile(path):
                out["table"] = path
                break
    if manifest:
        out["manifest"] = os.path.abspath(manifest)
    elif src_dir:
        out["manifest"] = os.path.join(src_dir, "fusion_capture", "profile_trace_manifest.json")
    if out.get("table"):
        out["structural_patterns"] = os.path.join(os.path.dirname(out["table"]),
                                                  "STRUCTURAL_LAYER_PATTERNS.json")
    if src_dir:
        out["catalog"] = os.path.join(src_dir, "fusion_capture", "fusion",
                                      "available_fusion_kernels.json")
        out["semantic_md"] = os.path.join(src_dir, "01_SEMANTIC.md")
    return {k: v for k, v in out.items() if v and os.path.isfile(v)}


DEST_NAMES = {
    "table": "pattern_layer_kernel_table.json",
    "structural_patterns": "STRUCTURAL_LAYER_PATTERNS.json",
    "manifest": "profile_trace_manifest.json",
    "catalog": "available_fusion_kernels.json",
    "semantic_md": "01_SEMANTIC.md",
}


def reuse(src_dir, eval_dir, table=None, manifest=None):
    sources = resolve_sources(src_dir, table, manifest)
    if "table" not in sources:
        return 1, {"ok": False, "reason": "no semantic table found (looked for %s under %s%s)"
                   % (" / ".join("/".join(p) for p in TABLE_CANDIDATES), src_dir,
                      "; override %s missing" % table if table else "")}
    eval_dir = os.path.abspath(eval_dir)
    dest_dir = os.path.join(eval_dir, "fusion_capture", "reused")
    os.makedirs(dest_dir, exist_ok=True)
    copied = {}
    for key, src in sorted(sources.items()):
        dest = os.path.join(dest_dir, DEST_NAMES[key])
        shutil.copyfile(src, dest)
        copied[key] = {"source": src, "dest": dest, "sha256": _sha256(dest),
                       "bytes": os.path.getsize(dest)}
    root_md = ""
    if "semantic_md" in copied:
        root_md = os.path.join(eval_dir, "01_SEMANTIC.md")
        shutil.copyfile(copied["semantic_md"]["dest"], root_md)
    with open(copied["table"]["dest"]) as fh:
        cov = semantic_phase_coverage.coverage(json.load(fh))
    record = {
        "schema_version": 1, "artifact": "fusion_semantics_reuse",
        "reused_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_eval_dir": os.path.abspath(src_dir) if src_dir else "",
        "files": copied, "coverage": cov,
    }
    reused_from = os.path.join(dest_dir, "reused_from.json")
    with open(reused_from, "w") as fh:
        json.dump(record, fh, indent=1, sort_keys=True)
    out = {
        "ok": True,
        "semantic_table_json": copied["table"]["dest"],
        "structural_patterns_json": copied.get("structural_patterns", {}).get("dest", ""),
        "trace_manifest_json": copied.get("manifest", {}).get("dest", ""),
        "catalog_json": copied.get("catalog", {}).get("dest", ""),
        "semantic_md": root_md,
        "reused_from_json": reused_from,
        "source_eval_dir": record["source_eval_dir"],
        "coverage": cov,
        "decode_covered": bool(cov.get("decode_covered")),
        "missing_phases": cov.get("missing_phases") or [],
    }
    return 0, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src-dir", default="", help="earlier run's eval dir")
    ap.add_argument("--eval-dir", required=True, help="this run's eval dir")
    ap.add_argument("--semantic-table", default="", help="explicit table override")
    ap.add_argument("--trace-manifest", default="", help="explicit manifest override")
    a = ap.parse_args(argv)
    try:
        code, out = reuse(a.src_dir, a.eval_dir, a.semantic_table or None, a.trace_manifest or None)
    except (OSError, ValueError) as exc:
        code, out = 1, {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc)}
    print(json.dumps(out, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
