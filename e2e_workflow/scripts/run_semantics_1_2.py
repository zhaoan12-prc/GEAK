#!/usr/bin/env python3
"""Run the complete Semantics Mapping 1.2 pipeline using GEAK scripts only."""
import argparse
import hashlib
import json
import os
import shutil
import sys

import semantic_kernel_mapping
import semantic_evidence_ledger
import semantic_runtime_marker_mapping
import semantic_shape_merge
import semantic_source_mapping
import validate_structural_patterns
import run_semantic_shape_capture

# Both model phases are required by default. See the require_phases block in
# run() for why this is not an env-var opt-in any more.
DEFAULT_REQUIRE_PHASES = ("prefill", "decode")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_graph_capture_phases(capture_results):
    """Phases whose clean rows were fully checked against graph-capture data."""
    verified = set()
    for capture in capture_results:
        if capture.get("shape_capture_execution") != "graph_capture":
            continue
        mapping = capture.get("runtime_marker_mapping") or {}
        audit = mapping.get("clean_table_sequence_audit") or {}
        passed_groups = {}
        for group in audit.get("groups", []):
            phase = str(group.get("phase") or "").lower()
            passed_groups.setdefault(phase, []).append(
                group.get("status") == "pass")
        for phase, stats in (mapping.get(
                "shape_mapping_by_phase") or {}).items():
            phase = str(phase or "").lower()
            eligible = int(stats.get("shape_eligible_target_count", 0) or 0)
            matched = int(
                stats.get("shape_eligible_matched_target_count", 0) or 0)
            if (eligible > 0 and matched == eligible
                    and passed_groups.get(phase)
                    and all(passed_groups[phase])):
                verified.add(phase)
    return verified


def run(config_path, trace_path, shape_log_path, out_dir,
        config_key="", runtime_sources=None, capture_setup_path="",
        capture_result_path="", capture_result_paths=None,
        structural_patterns_path="", require_phases=None):
    os.makedirs(out_dir, exist_ok=True)
    runtime_sources = list(runtime_sources or [])
    if not structural_patterns_path:
        raise ValueError(
            "structural_patterns_path is required; Layer Patterns must be "
            "defined by the semantics_mapper Agent from config and runtime "
            "source before Semantics 1.2")
    structural_patterns_input = os.path.abspath(
        structural_patterns_path)
    structural_patterns_input_sha256 = _sha256(
        structural_patterns_input)
    callable_kernel_map = []
    source_wrapper_map = []
    for source_path in runtime_sources:
        if not source_path.endswith(".json"):
            continue
        try:
            with open(source_path) as fh:
                source_config = json.load(fh)
            callable_kernel_map.extend(
                source_config.get("callable_kernel_map", []))
            source_wrapper_map.extend(
                source_config.get("source_wrapper_map", []))
        except (OSError, ValueError):
            pass
    patterns_path = os.path.join(out_dir, "STRUCTURAL_LAYER_PATTERNS.json")
    structural_validation = validate_structural_patterns.validate(
        structural_patterns_input, config_path, runtime_sources,
        patterns_path)

    # trace_path may be a single trace or a list; the mapper auto-adopts the
    # EXTEND/DECODE phase sibling so a run cannot silently cover one phase.
    #
    # Both phases are required BY DEFAULT. This used to read an env var that
    # defaulted to EMPTY, i.e. "require no phase at all" — so unless a caller
    # happened to export GEAK_SEMANTICS_REQUIRE_PHASES, a prefill-only capture
    # sailed through and every downstream fusion decision was made on half the
    # model's execution. A decode-blind semantic table is not a lighter table,
    # it is a wrong one: decode is where the small-tensor fusions actually pay.
    #
    # GEAK_SEMANTICS_REQUIRE_PHASES may still NARROW the requirement (e.g. a
    # deliberate prefill-only probe run sets it to "prefill"), but that is now
    # an explicit, logged act rather than the silent default.
    if require_phases is None:
        configured = os.environ.get("GEAK_SEMANTICS_REQUIRE_PHASES")
        if configured is None:
            require_phases = list(DEFAULT_REQUIRE_PHASES)
        else:
            require_phases = [
                value.strip() for value in configured.split(",")
                if value.strip()]
            narrowed = [
                phase for phase in DEFAULT_REQUIRE_PHASES
                if phase not in require_phases]
            if narrowed:
                print(
                    "[semantics] GEAK_SEMANTICS_REQUIRE_PHASES=%r NARROWS the "
                    "default two-phase requirement; dropping %s. Fusion "
                    "candidates for the dropped phase(s) will not be "
                    "trustworthy." % (configured, narrowed),
                    file=sys.stderr)
    semantic = semantic_kernel_mapping.build(
        trace_path, patterns_path, out_dir, require_phases=require_phases)

    # --- Initial layer-boundary evidence -------------------------------------
    # semantic_kernel_mapping resolves per-layer boundaries from python_function
    # `nn.Module: ...DecoderLayer_<id>` spans, captured only when the torch
    # profiler ran with with_stack/with_modules. Without them the boundary step
    # degrades to sequence segmentation.  Do not accept that evidence by itself;
    # graph-capture completion below must independently verify every degraded
    # phase against the clean Kernel sequence before the run can pass.
    with open(semantic["layer_instance_audit_json"]) as fh:
        module_scope_count = int(
            json.load(fh).get("module_scope_count", 0) or 0)
    boundary_evidence = (
        "module_span" if module_scope_count > 0
        else "degraded_no_module_span")

    phase_1_1_json = os.path.join(
        out_dir, "pattern_layer_kernel_table_1_1.json")
    phase_1_1_md = os.path.join(
        out_dir, "ORDERED_UNIQUE_LAYER_TABLES_1_1.md")
    shutil.copyfile(semantic["semantic_table_json"], phase_1_1_json)
    shutil.copyfile(semantic["semantic_table_md"], phase_1_1_md)
    semantic["semantic_table_json"] = phase_1_1_json
    semantic["semantic_table_md"] = phase_1_1_md
    source_plan_path = os.path.join(
        out_dir, "SHAPE_CAPTURE_PLAN_SOURCE_MAPPED.json")
    semantic_source_mapping.map_plan(
        semantic["shape_capture_plan_json"], runtime_sources,
        source_plan_path)

    capture_results = []
    if capture_setup_path:
        capture_results.append(run_semantic_shape_capture.capture(
            capture_setup_path, source_plan_path,
            os.path.join(out_dir, "capture")))
    result_paths = list(capture_result_paths or [])
    if capture_result_path:
        if isinstance(capture_result_path, (list, tuple)):
            result_paths.extend(capture_result_path)
        else:
            result_paths.append(capture_result_path)
    for path in result_paths:
        with open(path) as fh:
            capture_results.append(json.load(fh))
    if not shape_log_path and not capture_results:
        raise ValueError(
            "shape_log_path, capture_setup_path, or capture_result_path "
            "is required")

    probe_tables = []
    probe_runs = []
    if shape_log_path:
        direct_dir = os.path.join(out_dir, "probe_runs", "direct")
        direct = semantic_shape_merge.merge(
            phase_1_1_json, source_plan_path, shape_log_path, direct_dir)
        probe_tables.append(direct["semantic_table_json"])
        probe_runs.append({
            "kind": "direct_shape_log",
            "shape_log": os.path.abspath(shape_log_path),
            "shape_merge": direct,
        })
    for index, capture_result in enumerate(capture_results):
        run_dir = os.path.join(out_dir, "probe_runs", "run_%02d" % index)
        os.makedirs(run_dir, exist_ok=True)
        merge_plan_path = os.path.join(
            run_dir, "SHAPE_CAPTURE_PLAN_RUNTIME_MAPPED.json")
        marker_mapping = semantic_runtime_marker_mapping.map_plan(
            source_plan_path, capture_result["capture_trace"],
            merge_plan_path, capture_result.get("shape_log", ""),
            capture_result.get(
                "callable_kernel_map", callable_kernel_map),
            capture_result.get(
                "source_wrapper_map", source_wrapper_map),
            clean_table_path=phase_1_1_json,
            required_phases=capture_result.get("capture_phases"))
        capture_result["runtime_marker_mapping"] = marker_mapping
        merged_probe = semantic_shape_merge.merge(
            phase_1_1_json, merge_plan_path,
            capture_result["shape_log"], run_dir)
        probe_tables.append(merged_probe["semantic_table_json"])
        probe_runs.append({
            "kind": "runtime_capture",
            "capture": capture_result,
            "mapped_plan": merge_plan_path,
            "shape_merge": merged_probe,
        })

    merged_dir = os.path.join(out_dir, "semantics_1_2")
    merged = semantic_evidence_ledger.merge(
        phase_1_1_json, probe_tables, merged_dir)
    published_json = os.path.join(
        out_dir, "pattern_layer_kernel_table.json")
    published_md = os.path.join(
        out_dir, "ORDERED_UNIQUE_LAYER_TABLES.md")
    shutil.copyfile(merged["semantic_table_json"], published_json)
    shutil.copyfile(merged["semantic_table_md"], published_md)
    capture_phase_coverage_complete = all(
        capture.get("runtime_marker_mapping", {}).get(
            "phase_coverage_complete", False)
        for capture in capture_results)
    graph_capture_verified_phases = _verified_graph_capture_phases(
        capture_results)
    boundary_rebuild = {
        "applied": bool(graph_capture_verified_phases),
        "status": (
            "graph_capture_sequence_verified"
            if graph_capture_verified_phases else "unavailable"),
        "reason": (
            "degraded clean-trace layer boundaries are accepted only for "
            "phases whose graph-capture shape-eligible rows all mapped and "
            "whose pattern Kernel sequences matched the clean table"),
        "verified_phases": sorted(graph_capture_verified_phases),
        "traces": [
            os.path.abspath(capture["capture_trace"])
            for capture in capture_results
            if capture.get("shape_capture_execution") == "graph_capture"
            and capture.get("capture_trace")],
    }
    # --- Per-phase boundary evidence -----------------------------------------
    # `boundary_evidence` above is an AGGREGATE over the whole run: prefill's 61
    # module spans set it to "module_span" even when every decode layer fell
    # back to anchor_repeat_segmentation.  That fail-open is what let a rotated,
    # identity-less decode table reach the fusion analyst marked healthy.  Grade
    # each phase separately and require BOTH.
    phase_boundary_evidence = {}
    try:
        with open(phase_1_1_json) as fh:
            for table in json.load(fh).get("tables", []):
                phase = table.get("phase")
                if not phase:
                    continue
                levels = {row.get("layer_evidence")
                          for row in table.get("rows", [])}
                degraded = {level for level in levels
                            if not str(level).startswith(
                                ("module_span", "python_module_span"))}
                phase_boundary_evidence[phase] = (
                    "degraded:" + ",".join(sorted(str(x) for x in degraded))
                    if degraded else "module_span")
    except Exception as exc:  # pragma: no cover - diagnostics only
        phase_boundary_evidence = {"error": str(exc)}
    degraded_phases = sorted(
        phase for phase, level in phase_boundary_evidence.items()
        if level != "module_span")
    blocking_degraded_phases = [
        phase for phase in degraded_phases
        if phase not in graph_capture_verified_phases]

    status = "pass" if (
        semantic["status"] != "fail"
        and merged["status"] == "pass"
        and capture_phase_coverage_complete
        and not blocking_degraded_phases
    ) else "fail"
    result = {
        "schema_version": 1,
        "pipeline": "geak_semantics_1_2",
        "evidence_policy": {
            "levels": ["K", "P", "U"],
            "K": "clean trace Input Dims via External id",
            "P": "runtime shape_logger probe (kernel or wrapper scope)",
            "U": "unavailable after probes with mandatory reason_code",
            "priority": ["K", "P(kernel)", "P(wrapper)", "U"],
            "additive_across_probe_runs": True,
        },
        "status": status,
        "capture_phase_coverage_complete": (
            capture_phase_coverage_complete),
        "boundary_evidence": boundary_evidence,
        "phase_boundary_evidence": phase_boundary_evidence,
        "degraded_boundary_phases": degraded_phases,
        "blocking_degraded_boundary_phases": blocking_degraded_phases,
        "boundary_rebuild": boundary_rebuild,
        "module_scope_count": module_scope_count,
        "inputs": {
            "config": {
                "path": os.path.abspath(config_path),
                "sha256": _sha256(config_path),
            },
            "trace": {
                "path": os.path.abspath(trace_path),
                "sha256": _sha256(trace_path),
            },
            "shape_log": {
                "path": os.path.abspath(shape_log_path),
                "sha256": _sha256(shape_log_path),
            } if shape_log_path else None,
            "runtime_sources": [
                os.path.abspath(path) for path in runtime_sources],
            "agent_structural_patterns": {
                "path": structural_patterns_input,
                "sha256": structural_patterns_input_sha256,
            },
        },
        "structural_patterns_json": patterns_path,
        "structural_pattern_validation": (
            structural_validation.get("validation", {})),
        "semantic_mapping": semantic,
        "shape_merge": merged,
        "probe_runs": probe_runs,
        "runtime_captures": capture_results,
        "published_semantic_table_json": published_json,
        "published_semantic_table_md": published_md,
    }
    result_path = os.path.join(out_dir, "SEMANTICS_1_2_RUN.json")
    result["result_json"] = result_path
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-key", default="")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--shape-log", default="")
    parser.add_argument("--capture-setup", default="")
    parser.add_argument("--capture-result", action="append", default=[])
    parser.add_argument("--runtime-source", action="append", default=[])
    parser.add_argument("--structural-patterns", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    result = run(
        args.config, args.trace, args.shape_log, args.out_dir,
        args.config_key, args.runtime_source,
        args.capture_setup, capture_result_paths=args.capture_result,
        structural_patterns_path=args.structural_patterns)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
