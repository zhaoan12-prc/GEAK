#!/usr/bin/env python3
"""Deterministic harness for the two Semantic Mapping agent phases.

The agent owns the model-specific reasoning that produces structural Patterns and
selects runtime source evidence.  Once those inputs exist, this harness owns the
mechanical pipeline so the agent does not spend turns re-discovering commands or
manually stitching intermediate JSON files together.
"""
import argparse
import json
import os

import semantic_kernel_mapping
import semantic_shape_merge
import run_semantic_shape_capture
import validate_structural_patterns


def _write_result(path, result):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)


def build_table(patterns, config, runtime_sources, trace, out_dir,
                table_phases=None, result_json=""):
    """Validate Agent-defined Patterns, then build the Clean Trace table."""
    os.makedirs(out_dir, exist_ok=True)
    validated_patterns = os.path.join(out_dir, "STRUCTURAL_LAYER_PATTERNS.json")
    validation = validate_structural_patterns.validate(
        patterns, config, runtime_sources, validated_patterns)
    mapping = semantic_kernel_mapping.build(
        trace, validated_patterns, out_dir, table_phases)
    result = dict(mapping)
    result.update({
        "phase": "build_table",
        "structural_patterns_json": validated_patterns,
        "structural_pattern_validation": validation.get("validation", {}),
    })
    _write_result(result_json, result)
    return result


def complete_table(table, capture_plan, shape_log, out_dir, result_json=""):
    """Merge one metadata-only replay without changing Clean Trace identity."""
    result = semantic_shape_merge.merge(
        table, capture_plan, shape_log, out_dir)
    result = dict(result)
    result["phase"] = "complete_table"
    _write_result(result_json, result)
    return result


def capture_complete(table, capture_plan, setup, out_dir,
                     disable_cuda_graph=False, phases=None,
                     forwards_per_bucket=1, result_json=""):
    """Run exactly one metadata replay and merge it into the Clean Trace table."""
    capture_dir = os.path.join(out_dir, "capture")
    merge_dir = os.path.join(out_dir, "merged")
    capture = run_semantic_shape_capture.capture(
        setup, capture_plan, capture_dir, disable_cuda_graph,
        phases, forwards_per_bucket)
    merged = complete_table(
        table, capture_plan, capture["shape_log"], merge_dir)
    result = dict(merged)
    result.update({
        "phase": "capture_complete",
        "capture": capture,
    })
    _write_result(result_json, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build-table")
    build.add_argument("--patterns", required=True)
    build.add_argument("--config", required=True)
    build.add_argument("--runtime-source", action="append", default=[])
    build.add_argument("--trace", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--table-phases", default="all")
    build.add_argument("--result-json", default="")

    complete = sub.add_parser("complete-table")
    complete.add_argument("--table", required=True)
    complete.add_argument("--capture-plan", required=True)
    complete.add_argument("--shape-log", required=True)
    complete.add_argument("--out-dir", required=True)
    complete.add_argument("--result-json", default="")

    capture = sub.add_parser("capture-complete")
    capture.add_argument("--table", required=True)
    capture.add_argument("--capture-plan", required=True)
    capture.add_argument("--setup", required=True)
    capture.add_argument("--out-dir", required=True)
    capture.add_argument("--disable-cuda-graph", action="store_true")
    capture.add_argument("--phase", action="append", default=[])
    capture.add_argument("--forwards-per-bucket", type=int, default=1)
    capture.add_argument("--result-json", default="")

    args = parser.parse_args()
    if args.command == "build-table":
        requested = set(
            value.strip() for value in args.table_phases.split(",")
            if value.strip())
        phases = None if "all" in requested else requested or None
        result = build_table(
            args.patterns, args.config, args.runtime_source, args.trace,
            args.out_dir, phases, args.result_json)
    elif args.command == "complete-table":
        result = complete_table(
            args.table, args.capture_plan, args.shape_log, args.out_dir,
            args.result_json)
    elif args.command == "capture-complete":
        result = capture_complete(
            args.table, args.capture_plan, args.setup, args.out_dir,
            args.disable_cuda_graph, args.phase,
            args.forwards_per_bucket, args.result_json)
    else:
        parser.error("a command is required")
    print(json.dumps(result))
    return 0 if result.get("status") not in ("fail", "failed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
