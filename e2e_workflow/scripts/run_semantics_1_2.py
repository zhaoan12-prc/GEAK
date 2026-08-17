#!/usr/bin/env python3
"""Run the complete Semantics Mapping 1.2 pipeline using GEAK scripts only."""
import argparse
import hashlib
import json
import os
import shutil

import semantic_kernel_mapping
import semantic_evidence_ledger
import semantic_phase_tables
import semantic_runtime_marker_mapping
import semantic_shape_merge
import semantic_source_mapping
import semantic_two_trace_mapping
import semantic_workload_identity
import validate_structural_patterns
import run_semantic_shape_capture


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _two_trace_mapping(out_dir, patterns_path, phase_1_1_json,
                       capture_results, mapping_traces, capture_setup_path,
                       formal_workload_path, mapping_setup_path=""):
    """Recover graph-erased op attribution from a workload-identical graph-off run.

    The mapping traces default to the rank-0 stage windows of the shape-capture
    replay, which already runs with ``disable_cuda_graph`` -- so the graph-off
    trace this needs is normally produced for free by Semantics 1.2 itself.

    The identity gate needs the mapping run's *declared* workload as well as its
    trace.  When the capture runs here that is ``capture_setup_path``, but a
    re-analysis of traces captured earlier reuses them via ``--capture-result``
    and so has no setup to run; ``mapping_setup_path`` supplies the descriptor
    in that case, and takes precedence when both are given.
    """
    trace_paths = list(mapping_traces or [])
    mapping_setup = {}
    setup_path = mapping_setup_path or capture_setup_path
    if setup_path:
        with open(setup_path) as fh:
            mapping_setup = json.load(fh)
    if not trace_paths:
        for capture in capture_results:
            if not capture.get("disable_cuda_graph"):
                continue
            stages = capture.get("rank0_stage_traces") or {}
            trace_paths.extend(stages[key] for key in sorted(stages))
            mapping_setup.setdefault(
                "disable_cuda_graph", capture.get("disable_cuda_graph"))
    trace_paths = [path for path in trace_paths if path]
    if not trace_paths:
        return "", None

    if not formal_workload_path:
        raise ValueError(
            "two-trace op mapping requires --formal-workload: the Clean Trace "
            "run's declared workload must be supplied so it can be proven "
            "identical to the mapping run")
    with open(formal_workload_path) as fh:
        formal_workload = json.load(fh)

    two_trace_dir = os.path.join(out_dir, "two_trace")
    os.makedirs(two_trace_dir, exist_ok=True)
    with open(phase_1_1_json) as fh:
        formal_table_doc = json.load(fh)
    document, mapping_table_doc = semantic_two_trace_mapping.build_from_traces(
        formal_table_doc, trace_paths, patterns_path, two_trace_dir)
    # A declared-workload mismatch raises: the two runs are not the same
    # experiment, so nothing from them may be bound. A per-table bucket
    # difference only disqualifies that table -- the mapping replay's window is
    # routinely shorter than the formal run's, and one uncovered (pattern,
    # phase) is no reason to discard the ones that did line up.
    identity = semantic_workload_identity.verify(
        formal_workload, mapping_setup, formal_table_doc, mapping_table_doc,
        os.path.join(two_trace_dir, "WORKLOAD_IDENTITY.json"), strict=True)
    document["workload_identity"] = identity
    semantic_two_trace_mapping.drop_tables(
        document, identity.get("skipped_table_keys") or [])
    map_path = os.path.join(two_trace_dir, "TWO_TRACE_OP_MAP.json")
    with open(map_path, "w") as fh:
        json.dump(document, fh, indent=2)
    document["result_json"] = map_path
    return map_path, document


def _two_trace_coverage(two_trace_document):
    """Per-table mapping coverage, and the tables that recovered ~nothing.

    The identity gate compares each table's `selected_bucket` -- phase, batch
    size, input tokens -- and nothing else. A mapping table holding a single
    device row has a *matching* bucket, so identity reports `pass` while that
    table in fact binds no ops at all and every row silently keeps its weaker
    fallback (the enclosing module marker, `model.layers.N...`).

    That happens when the graph-off replay's window is device-truncated: in
    eager mode the CPU runs ahead of the GPU, so at `stop_profile` the tail
    layers have python/module spans but no kernels. If the mapping run's
    representative for a pattern lands in that tail, its table is empty.

    Coverage is therefore reported separately from identity. It stays
    non-gating -- unbound rows keep valid, weaker evidence -- but a table at
    0.0 is the signal to lengthen the mapping window or to pick a
    representative with complete device coverage.
    """
    tables = (two_trace_document or {}).get("tables") or []
    coverage = []
    for table in tables:
        formal = int(table.get("formal_rows", 0) or 0)
        mapping = int(table.get("mapping_rows", 0) or 0)
        matched = int(table.get("matched_rows", 0) or 0)
        coverage.append({
            "table": table.get("table"),
            "formal_rows": formal,
            "mapping_rows": mapping,
            "matched_rows": matched,
            "matched_fraction": round(matched / formal, 4) if formal else 0.0,
        })
    starved = sorted(
        item["table"] for item in coverage
        if item["formal_rows"] and not item["matched_rows"])
    return coverage, starved


def run(config_path, trace_path, shape_log_path, out_dir,
        config_key="", runtime_sources=None, capture_setup_path="",
        capture_result_path="", capture_result_paths=None,
        structural_patterns_path="", mapping_traces=None,
        formal_workload_path="", mapping_setup_path="",
        layer_boundary_map=""):
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

    # A CUDA-graph-replayed stage (typically decode) emits no DecoderLayer span
    # of its own. ``semantic_decode_boundary_transfer`` recovers those
    # boundaries from a workload-identical graph-off run; pass its
    # DECODE_BOUNDARY_TRANSFER.json here so the gate below can see
    # ``transferred_boundary_rows`` instead of always reading zero. Boundaries
    # only -- device order and duration stay this trace's own.
    semantic = semantic_kernel_mapping.build(
        trace_path, patterns_path, out_dir,
        boundary_map_path=layer_boundary_map)

    # --- Layer-boundary evidence gate (module spans) -------------------------
    # semantic_kernel_mapping resolves per-layer boundaries from python_function
    # `nn.Module: ...DecoderLayer_<id>` spans, captured only when the torch
    # profiler ran with with_stack/with_modules. Without them the boundary step
    # degrades to forced_best_alignment and mis-places layer start/end. Fail
    # loudly instead of silently emitting unreliable boundaries; allow an
    # explicit opt-in to proceed degraded.
    with open(semantic["layer_instance_audit_json"]) as fh:
        layer_audit = json.load(fh)
    module_scope_count = int(layer_audit.get("module_scope_count", 0) or 0)
    # A CUDA-graph-replayed stage emits no module span of its own, but
    # semantic_decode_boundary_transfer can carry the boundary over from a
    # workload-identical graph-off run. That boundary is still module-span
    # derived and passes its own completeness/monotonicity checks, so it
    # satisfies this gate; only timing stays this trace's own.
    transferred_boundary_rows = int(
        layer_audit.get("transferred_boundary_rows", 0) or 0)
    allow_no_module_spans = os.environ.get(
        "GEAK_SEMANTICS_ALLOW_NO_MODULE_SPANS", "0") in ("1", "true", "True")
    boundary_evidence = (
        "module_span" if module_scope_count > 0
        else "transferred_module_span" if transferred_boundary_rows > 0
        else "degraded_no_module_span")
    if (module_scope_count == 0 and transferred_boundary_rows == 0
            and not allow_no_module_spans):
        # Non-gating sidecar: do NOT crash or re-capture. Return an explicit
        # failed status so the caller (e2e_workflow) skips Semantics 1.2 +
        # semantic/fusion and falls back to the native optimization flow.
        notes = (
            "clean trace has no DecoderLayer module spans "
            "(module_scope_count=0): the torch profiler was captured without "
            "with_stack/with_modules, so per-layer kernel boundaries would "
            "degrade to forced_best_alignment and are untrustworthy. Skipping "
            "Semantics 1.2 + semantic/fusion; native flow should proceed. "
            "Re-capture the trace with SGLANG_PROFILE_WITH_STACK=1 (keep "
            "record_shapes on), or set GEAK_SEMANTICS_ALLOW_NO_MODULE_SPANS=1 "
            "to force a degraded run.")
        result = {
            "schema_version": 1,
            "pipeline": "geak_semantics_1_2",
            "status": "failed",
            "boundary_evidence": boundary_evidence,
            "module_scope_count": module_scope_count,
            "notes": notes,
            "inputs": {
                "config": {
                    "path": os.path.abspath(config_path),
                    "sha256": _sha256(config_path),
                },
                "trace": {
                    "path": os.path.abspath(trace_path),
                    "sha256": _sha256(trace_path),
                },
                "agent_structural_patterns": {
                    "path": structural_patterns_input,
                    "sha256": structural_patterns_input_sha256,
                },
            },
            "structural_patterns_json": patterns_path,
            "semantic_mapping": semantic,
            "published_semantic_table_json": semantic["semantic_table_json"],
            "published_semantic_table_md": semantic["semantic_table_md"],
        }
        result_path = os.path.join(out_dir, "SEMANTICS_1_2_RUN.json")
        result["result_json"] = result_path
        with open(result_path, "w") as fh:
            json.dump(result, fh, indent=2)
        return result

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

    two_trace_map_path, two_trace_document = _two_trace_mapping(
        out_dir, patterns_path, phase_1_1_json, capture_results,
        mapping_traces, capture_setup_path, formal_workload_path,
        mapping_setup_path)

    probe_tables = []
    probe_runs = []
    if shape_log_path:
        direct_dir = os.path.join(out_dir, "probe_runs", "direct")
        direct = semantic_shape_merge.merge(
            phase_1_1_json, source_plan_path, shape_log_path, direct_dir,
            two_trace_map_path)
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
                "source_wrapper_map", source_wrapper_map))
        capture_result["runtime_marker_mapping"] = marker_mapping
        merged_probe = semantic_shape_merge.merge(
            phase_1_1_json, merge_plan_path,
            capture_result["shape_log"], run_dir, two_trace_map_path)
        probe_tables.append(merged_probe["semantic_table_json"])
        probe_runs.append({
            "kind": "runtime_capture",
            "capture": capture_result,
            "mapped_plan": merge_plan_path,
            "shape_merge": merged_probe,
        })

    two_trace_coverage, two_trace_unmapped = _two_trace_coverage(
        two_trace_document)

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
    # `transferred_module_span` is accepted alongside `module_span` for the same
    # reason the boundary gate above accepts it: a boundary carried over from a
    # workload-identical graph-off run is still module-span derived and passed
    # its own completeness/monotonicity checks. Requiring the literal
    # "module_span" here would make every CUDA-graph-replayed stage (i.e. every
    # decode Clean Trace) unconditionally fail, contradicting that gate.
    status = "pass" if (
        semantic["status"] != "fail"
        and merged["status"] == "pass"
        and capture_phase_coverage_complete
        and boundary_evidence in ("module_span", "transferred_module_span")
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
        # Additive: a table the mapping replay could not cover is skipped, not
        # transplanted, so this never gates `status`. It is surfaced because a
        # long run of "partial" is the signal to lengthen the mapping window.
        "two_trace_identity_status": (
            (two_trace_document or {}).get(
                "workload_identity", {}).get("status", "not_run")),
        "two_trace_skipped_tables": (
            (two_trace_document or {}).get(
                "workload_identity", {}).get("skipped_table_keys", [])),
        # Identity `pass` does not mean the mapping run actually covered every
        # table; see _two_trace_coverage. A table listed here recovered no op
        # attribution at all, so its rows fall back to the enclosing module
        # marker rather than a real operator.
        "two_trace_table_coverage": two_trace_coverage,
        "two_trace_unmapped_tables": two_trace_unmapped,
        "boundary_evidence": boundary_evidence,
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
        "two_trace_mapping": two_trace_document,
        "two_trace_map_json": two_trace_map_path,
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


def run_stages(config_path, traces, out_dir, capture_setups=None,
               layer_boundary_maps=None, **kwargs):
    """Run one Semantics 1.2 pass per stage, publish one combined table.

    The official capture writes prefill and decode to separate rank-0 files
    and only the prefill file carries DecoderLayer module spans, so the two
    stages cannot share a run: they need different traces, different capture
    setups (the probe must aim at each stage's own representative layers, which
    differ) and, for decode, a transferred layer boundary.

    They should still produce ONE deliverable. Each stage keeps its complete
    audit set under ``stages/<phase>/`` and ``out_dir`` holds the single
    combined prefill+decode table.
    """
    traces = list(traces)
    capture_setups = list(capture_setups or [])
    layer_boundary_maps = list(layer_boundary_maps or [])
    capture_setups += [""] * (len(traces) - len(capture_setups))
    layer_boundary_maps += [""] * (len(traces) - len(layer_boundary_maps))

    # A capture belongs to exactly one stage: its probe aimed at that stage's
    # representative layers. Handing every stage the whole list would bind one
    # stage's shapes onto another's rows, so pair them positionally and refuse
    # anything ambiguous rather than guess.
    capture_results = list(kwargs.pop("capture_result_paths", None) or [])
    if capture_results and len(capture_results) != len(traces):
        raise ValueError(
            "multi-stage run got %d --capture-result for %d --trace: pass one "
            "per stage in the same order, or analyse the stages one at a time"
            % (len(capture_results), len(traces)))
    per_stage_results = (
        [[path] for path in capture_results] if capture_results
        else [[] for _ in traces])

    stage_results = []
    stage_paths = []
    used_labels = set()
    for index, trace_path in enumerate(traces):
        label = semantic_kernel_mapping._stage_label(trace_path, None, index)
        if label in used_labels:
            raise ValueError(
                "two stage traces resolve to the same label %r; each stage "
                "needs its own output directory" % label)
        used_labels.add(label)
        stage_result = run(
            config_path, trace_path, kwargs.get("shape_log_path", ""),
            os.path.join(out_dir, "stages", label),
            capture_setup_path=capture_setups[index],
            layer_boundary_map=layer_boundary_maps[index],
            capture_result_paths=per_stage_results[index],
            **{key: value for key, value in kwargs.items()
               if key != "shape_log_path"})
        stage_result["stage"] = label
        stage_results.append(stage_result)
        stage_paths.append(
            (label, stage_result["published_semantic_table_json"]))

    combined = semantic_phase_tables.combine(
        stage_paths,
        os.path.join(out_dir, "pattern_layer_kernel_table.json"),
        os.path.join(out_dir, "ORDERED_UNIQUE_LAYER_TABLES.md"),
        kind="1_2")
    result = {
        "schema_version": 1,
        "pipeline": "geak_semantics_1_2",
        "status": (
            "pass" if all(item["status"] == "pass" for item in stage_results)
            else "fail"),
        "stages": stage_results,
        # Per stage, because they genuinely differ: a prefill Clean Trace has
        # its own module spans, a decode one only has a transferred boundary.
        "boundary_evidence": {
            item["stage"]: item["boundary_evidence"]
            for item in stage_results},
        "table_phases": combined["table_phases"],
        "published_semantic_table_json": combined["semantic_table_json"],
        "published_semantic_table_md": combined["semantic_table_md"],
        "row_count": combined["row_count"],
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
    parser.add_argument(
        "--trace", required=True, action="append",
        help=("rank-0 Clean Trace stage file. Repeat once per stage "
              "(prefill and decode are separate files) to publish both "
              "phases as one combined table."))
    parser.add_argument("--shape-log", default="")
    parser.add_argument(
        "--capture-setup", action="append",
        help=("shape-capture setup. Repeat to pair one with each --trace, in "
              "the same order: each stage's probe must aim at that stage's "
              "own representative layers, which differ between phases."))
    parser.add_argument("--capture-result", action="append", default=[])
    parser.add_argument("--runtime-source", action="append", default=[])
    parser.add_argument("--structural-patterns", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--mapping-trace", action="append", default=[],
        help=("Graph-off trace of a workload-identical run, used to recover "
              "op attribution the graph-on Clean Trace cannot carry. Repeat "
              "per stage window. Defaults to the shape-capture replay's own "
              "rank-0 stage traces when it ran with disable_cuda_graph."))
    parser.add_argument(
        "--mapping-setup", default="",
        help=("JSON describing the mapping run's declared workload, for when "
              "its capture is reused via --capture-result instead of being "
              "run here. Same schema as --capture-setup, which supplies this "
              "when the capture does run."))
    parser.add_argument(
        "--formal-workload", default="",
        help=("JSON describing the Clean Trace run's workload. Required "
              "whenever two-trace mapping is active: it is checked field by "
              "field against the mapping run and the run fails on mismatch."))
    parser.add_argument(
        "--layer-boundary-map", action="append",
        help=("DECODE_BOUNDARY_TRANSFER.json from "
              "semantic_decode_boundary_transfer, supplying layer boundaries "
              "for a CUDA-graph-replayed stage that emits no module span of "
              "its own. Required to run a decode Clean Trace through 1.2; "
              "boundaries only, timing stays this trace's own. Repeat to "
              "pair one with each --trace, in the same order; pass an empty "
              "string for a stage that needs none."))
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()

    def _pad(values):
        values = list(values or [])
        return values + [""] * (len(args.trace) - len(values))

    capture_setups = _pad(args.capture_setup)
    boundary_maps = _pad(args.layer_boundary_map)
    shared = dict(
        config_key=args.config_key,
        runtime_sources=args.runtime_source,
        capture_result_paths=args.capture_result,
        structural_patterns_path=args.structural_patterns,
        mapping_traces=args.mapping_trace,
        formal_workload_path=args.formal_workload,
        mapping_setup_path=args.mapping_setup,
    )
    if len(args.trace) == 1:
        result = run(
            args.config, args.trace[0], args.shape_log, args.out_dir,
            capture_setup_path=capture_setups[0],
            layer_boundary_map=boundary_maps[0], **shared)
    else:
        result = run_stages(
            args.config, args.trace, args.out_dir,
            capture_setups=capture_setups,
            layer_boundary_maps=boundary_maps,
            shape_log_path=args.shape_log, **shared)
    if args.result_json:
        with open(args.result_json, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
