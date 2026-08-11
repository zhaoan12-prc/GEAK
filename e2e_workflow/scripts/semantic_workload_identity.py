#!/usr/bin/env python3
"""Prove the mapping trace and the formal Clean Trace ran the SAME workload.

Two-trace op mapping transplants attribution from a graph-off ``mapping`` trace
onto the graph-on ``formal`` Clean Trace table.  That is only sound when both
runs executed the same work: same model, same parallelism, same request shape,
and -- critically -- the same profiled step bucket.  If the two runs differ in
anything other than the graph/fusion switch, positional alignment silently binds
unrelated kernels and every downstream shape becomes wrong.

So this module is a gate, not a report.  It checks two independent levels:

1. **Declared workload** -- the setup both runs were launched with.  Every field
   in ``WORKLOAD_FIELDS`` must be identical.  Only ``ALLOWED_DIFFERENCES`` may
   differ, and the mapping run must be the graph-off side.
2. **Observed bucket** -- what the profiler actually captured, read back out of
   the two semantic tables (``selected_bucket``: phase, batch size, input
   tokens).  This is the stronger check: it catches a run that was *launched*
   identically but landed on a different step.

``verify()`` raises :class:`WorkloadIdentityError` on any mismatch.  Callers must
not downgrade that to a warning.
"""
import argparse
import json
import os


WORKLOAD_FIELDS = (
    "model",
    "tensor_parallel_size",
    "benchmark",
    "benchmark_repository",
    "concurrency",
    "input_length",
    "output_length",
    "random_range_ratio",
)

# Fields the two runs are EXPECTED to differ on. Everything else must match.
ALLOWED_DIFFERENCES = ("disable_cuda_graph",)


class WorkloadIdentityError(RuntimeError):
    """The mapping and formal runs did not execute the same workload."""


def _canonical(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        # Guard against 0.8 vs 0.80000000001 from different JSON writers.
        return round(float(value), 6)
    if value is None:
        return None
    return str(value).strip()


def descriptor(setup, role):
    """Build a comparable workload descriptor from a capture/run setup dict."""
    workload = setup.get("workload") or {}
    merged = dict(setup)
    for key, value in workload.items():
        merged.setdefault(key, value)
    fields = {}
    for name in WORKLOAD_FIELDS:
        fields[name] = _canonical(merged.get(name))
    return {
        "role": role,
        "fields": fields,
        "disable_cuda_graph": bool(merged.get("disable_cuda_graph", False)),
    }


def _table_buckets(table_doc):
    buckets = {}
    for table in table_doc.get("tables", []):
        key = "%s|%s" % (table.get("pattern_id"), table.get("phase"))
        bucket = table.get("selected_bucket") or {}
        buckets[key] = {
            "phase": _canonical(bucket.get("phase")),
            "batch_size": _canonical(bucket.get("batch_size")),
            "input_tokens": _canonical(bucket.get("input_tokens")),
        }
    return buckets


def compare_declared(formal, mapping):
    mismatches = []
    for name in WORKLOAD_FIELDS:
        left = formal["fields"].get(name)
        right = mapping["fields"].get(name)
        if left != right:
            mismatches.append({
                "field": name,
                "formal": left,
                "mapping": right,
            })
    missing = [
        name for name in WORKLOAD_FIELDS
        if formal["fields"].get(name) is None
    ]
    return {
        "mismatches": mismatches,
        "undeclared_fields": missing,
        "allowed_differences": list(ALLOWED_DIFFERENCES),
        "formal_disable_cuda_graph": formal["disable_cuda_graph"],
        "mapping_disable_cuda_graph": mapping["disable_cuda_graph"],
    }


def compare_buckets(formal_table_doc, mapping_table_doc):
    formal_buckets = _table_buckets(formal_table_doc)
    mapping_buckets = _table_buckets(mapping_table_doc)
    mismatches = []
    compared = []
    for key, formal_bucket in sorted(formal_buckets.items()):
        mapping_bucket = mapping_buckets.get(key)
        if mapping_bucket is None:
            mismatches.append({
                "table": key,
                "reason": "missing_in_mapping_trace",
                "formal": formal_bucket,
                "mapping": None,
            })
            continue
        compared.append(key)
        if formal_bucket != mapping_bucket:
            mismatches.append({
                "table": key,
                "reason": "bucket_differs",
                "formal": formal_bucket,
                "mapping": mapping_bucket,
            })
    return {
        "compared_tables": compared,
        "mismatches": mismatches,
        "formal_buckets": formal_buckets,
        "mapping_buckets": mapping_buckets,
    }


def verify(formal_setup, mapping_setup, formal_table_doc,
           mapping_table_doc, out_path="", strict=True):
    formal = descriptor(formal_setup, "formal")
    mapping = descriptor(mapping_setup, "mapping")
    declared = compare_declared(formal, mapping)
    buckets = compare_buckets(formal_table_doc, mapping_table_doc)

    reasons = []
    if declared["mismatches"]:
        reasons.append(
            "declared workload differs on: %s" % ", ".join(
                item["field"] for item in declared["mismatches"]))
    if declared["undeclared_fields"]:
        reasons.append(
            "workload fields not declared on both sides: %s"
            % ", ".join(declared["undeclared_fields"]))
    if not mapping["disable_cuda_graph"]:
        reasons.append(
            "mapping run must be the graph-off side "
            "(disable_cuda_graph=true); op attribution cannot be recovered "
            "from a graph-on mapping trace")
    if buckets["mismatches"]:
        reasons.append(
            "profiled bucket differs for: %s" % ", ".join(
                item["table"] for item in buckets["mismatches"]))
    if not buckets["compared_tables"]:
        reasons.append("no table was comparable between the two traces")

    result = {
        "schema_version": 1,
        "status": "pass" if not reasons else "fail",
        "formal": formal,
        "mapping": mapping,
        "declared_workload": declared,
        "observed_bucket": buckets,
        "failure_reasons": reasons,
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
    if strict and reasons:
        raise WorkloadIdentityError(
            "mapping and formal runs are not workload-identical: %s"
            % "; ".join(reasons))
    return result


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-setup", required=True)
    parser.add_argument("--mapping-setup", required=True)
    parser.add_argument("--formal-table", required=True)
    parser.add_argument("--mapping-table", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    result = verify(
        _load(args.formal_setup), _load(args.mapping_setup),
        _load(args.formal_table), _load(args.mapping_table),
        args.out, strict=not args.no_strict)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
