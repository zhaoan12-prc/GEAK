#!/usr/bin/env python3
"""Two-trace op mapping: recover attribution the graph-on trace cannot carry.

Why
---
When CUDA/HIP graphs are enabled, a whole decode step replays as a single
``hipGraphLaunch``.  The device kernels are still timed correctly, but they lose
their ``External id`` link to a ``cpu_op`` parent, because the Python that
launched them ran at graph *capture* time and does not run again on replay.  So
in a graph-on Clean Trace, decode kernels have ``parent_operator.mapping_level =
unresolved`` and no ``Input Dims`` -- op attribution AND shape are lost together,
for the same reason.

A graph-off run of the SAME workload does keep those links.  This module runs
the ordinary semantic kernel mapping over that graph-off ``mapping`` trace and
transplants ``parent_operator`` and ``shape`` onto the graph-on ``formal`` table
rows.  Only attribution moves; timing, order, row ids and durations always stay
the formal trace's, so the published table is unchanged as a performance record.

How rows are bound
------------------
NOT by kernel name alone.  A layer launches the same kernel name many times
(every projection is the same GEMM), so name-only matching is ambiguous exactly
where it matters -- that ambiguity is what produced ``non_unique_native_kernel_name``
unavailable rows.

Binding is **kernel name + ordered position inside the layer instance**:

* a pair may only match when the kernel names match (exact, or normalized in a
  second pass) -- a name mismatch is never bridged;
* among equal names, the k-th occurrence in the formal layer binds to the k-th
  occurrence in the mapping layer, because the alignment is a longest common
  subsequence over the two ordered name streams, which is order-preserving by
  construction;
* anything not on that common subsequence stays unmatched.  No fuzzy
  prefix/substring fallback: an unmatched row keeps its original (weaker)
  evidence rather than acquiring a plausible-but-unproven parent.

Choosing between equally long alignments
----------------------------------------
Order preservation alone is not enough.  An LCS maximises the *number* of
matched pairs and is indifferent to which occurrence of a repeated name each
pair uses, so when the mapping layer instance carries extra leading kernels
there are several maximum-length alignments and a plain greedy backtrack takes
the earliest one.  20260812v3's Qwen decode table is what that looks like: the
mapping instance began with four kernels the formal window does not have (a MoE
quant, ``ck::kernel_moe_gemm``, a DtoD ``Memcpy`` and the closing all-reduce),
its first kernel shares a name with the formal row 0, and the greedy backtrack
bound formal row 0 -- the attention input quant -- to the MoE quant.  Every
other row shifted by 4, so the table's ``position_delta`` read ``{0: 1, 4: 12,
5: 10}``: a single row out of step with all of its neighbours, holding MoE
shapes on an attention row.

So among alignments of equal length this module prefers the one whose pairs
share a consistent offset.  :func:`_lcs_pairs` takes a ``preferred_delta`` and,
at a name match, defers the pair when doing so moves ``j - i`` toward that
offset *and* costs no length -- the DP table is what proves the deferral free,
so an alignment can never be shortened to make it tidier.  :func:`align_rows`
runs the plain pass first, takes the dominant offset of the result, and
re-runs aimed at it until the alignment stops changing.

What a transplanted shape claims
-------------------------------
The dims move with the scope they had in the mapping trace.  A bound row is
``kernel`` scope only when the mapping row was ``kernel_exact`` -- its ``cpu_op``
parent launched exactly one device kernel, so the dims are that kernel's own
operands.  A 1:N parent also carries dims, but every kernel it launched carries
the same list, so those stay ``wrapper`` scope.  See :func:`_shape_scope`.

The two traces must be workload-identical; see :mod:`semantic_workload_identity`,
which callers are expected to run first and which skips any table it cannot
prove identical.
"""
import argparse
import collections
import json
import os
import re

import semantic_kernel_mapping


CONFIDENCE_ORDER = {"high": 3, "medium": 2, "low": 1, "none": 0}

# align_rows re-aims the backtrack at the offset the previous pass settled on.
# Each pass can only change the alignment by moving pairs onto that offset, so
# this converges almost immediately; the bound just makes termination obvious.
MAX_REALIGN_PASSES = 4


def _normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _delta_mode(pairs):
    """The offset most of an alignment's pairs agree on, or None if empty.

    Ties break on the offset closest to zero so the choice never depends on
    dict ordering.
    """
    if not pairs:
        return None
    counts = collections.Counter(j - i for i, j in pairs)
    return min(counts.items(),
               key=lambda item: (-item[1], abs(item[0]), item[0]))[0]


def _lcs_pairs(left_keys, right_keys, preferred_delta=None):
    """Longest common subsequence over two key streams -> ordered index pairs.

    Order preservation is what makes repeated kernel names resolvable: the k-th
    occurrence on the left can only pair with the k-th occurrence on the right.

    ``preferred_delta`` picks between alignments that are all of maximum length.
    At a name match the backtrack may defer the pair when that moves ``j - i``
    toward the preferred offset, but only when the DP table proves the deferral
    free: advancing a side is allowed exactly when the longest subsequence
    reachable from there is the same as from here.  Length is therefore
    invariant -- this reorders which occurrences pair, never how many.
    """
    n, m = len(left_keys), len(right_keys)
    if not n or not m:
        return []
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row = table[i]
        nxt = table[i + 1]
        for j in range(m - 1, -1, -1):
            if left_keys[i] == right_keys[j]:
                row[j] = nxt[j + 1] + 1
            else:
                row[j] = nxt[j] if nxt[j] >= row[j + 1] else row[j + 1]
    pairs = []
    i = j = 0
    while i < n and j < m:
        if left_keys[i] == right_keys[j]:
            if preferred_delta is not None:
                delta = j - i
                # Skipping this occurrence of the name is only permitted while
                # the reachable subsequence stays exactly as long.
                if delta < preferred_delta and table[i][j + 1] == table[i][j]:
                    j += 1
                    continue
                if delta > preferred_delta and table[i + 1][j] == table[i][j]:
                    i += 1
                    continue
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _gap_pairs(formal_rows, mapping_rows, anchors, preferred_delta=None):
    """Second pass: normalized-name LCS inside the gaps between exact anchors.

    Confined to each gap so the global order established by the exact pass can
    never be violated.  A gap's indices are contiguous ranges, so a pair's
    offset in table coordinates is the enclosing anchor's offset plus the pair's
    offset inside the slice -- which is what converts the caller's preferred
    offset into the one this slice should aim for.
    """
    extra = []
    bounds = [(-1, -1)] + anchors + [(len(formal_rows), len(mapping_rows))]
    for (prev_i, prev_j), (next_i, next_j) in zip(bounds, bounds[1:]):
        left_slice = list(range(prev_i + 1, next_i))
        right_slice = list(range(prev_j + 1, next_j))
        if not left_slice or not right_slice:
            continue
        left_keys = [
            _normalize_name(formal_rows[i].get("raw_name")) for i in left_slice]
        right_keys = [
            _normalize_name(mapping_rows[j].get("raw_name"))
            for j in right_slice]
        local_preferred = None
        if preferred_delta is not None:
            local_preferred = preferred_delta - (
                right_slice[0] - left_slice[0])
        for a, b in _lcs_pairs(left_keys, right_keys, local_preferred):
            if left_keys[a]:
                extra.append((left_slice[a], right_slice[b]))
    return extra


def align_rows(formal_rows, mapping_rows):
    """Bind formal rows to mapping rows by name + ordered position.

    The first pass is the plain backtrack.  Its dominant offset then aims a
    re-run, which can only move pairs onto that offset and never shorten the
    alignment, so repeating it until the result stops changing settles on the
    alignment whose pairs agree with each other.  Without this a mapping layer
    instance that starts with extra kernels binds its first repeated name to the
    wrong occurrence and leaves one row out of step with the whole table.
    """
    formal_keys = [str(row.get("raw_name") or "") for row in formal_rows]
    mapping_keys = [str(row.get("raw_name") or "") for row in mapping_rows]
    anchors = _lcs_pairs(formal_keys, mapping_keys)
    preferred_delta = _delta_mode(anchors)
    for _ in range(MAX_REALIGN_PASSES):
        if preferred_delta is None:
            break
        retry = _lcs_pairs(formal_keys, mapping_keys, preferred_delta)
        # The DP guard makes a shorter retry impossible; refuse it anyway rather
        # than trade matched rows for a tidier offset.
        if retry == anchors or len(retry) < len(anchors):
            break
        anchors = retry
        next_delta = _delta_mode(anchors)
        if next_delta == preferred_delta:
            break
        preferred_delta = next_delta
    matches = {i: (j, "exact") for i, j in anchors}
    for i, j in _gap_pairs(formal_rows, mapping_rows, anchors, preferred_delta):
        matches.setdefault(i, (j, "normalized"))
    return matches


def _confidence(name_match, delta, representative_match):
    if name_match == "exact" and delta == 0:
        level = "high"
    elif name_match == "exact" and abs(delta) <= 2:
        level = "medium"
    elif name_match == "exact":
        level = "low"
    elif abs(delta) <= 2:
        level = "low"
    else:
        return "low"
    if representative_match != "same_layer" and level == "high":
        # Same Pattern guarantees structural identity, but a different layer
        # instance is one step further from the measured row.
        level = "medium"
    return level


def _usable_shape(shape):
    if not isinstance(shape, dict):
        return False
    dims = shape.get("input_dims")
    return bool(dims)


def _shape_scope(shape):
    """Whether the bound row's dims are that kernel's operands or its wrapper's.

    ``semantic_kernel_mapping`` marks a row ``kernel_exact`` only when its
    ``cpu_op`` parent launched exactly one device kernel.  A 1:N parent still
    supplies dims -- every kernel it launched carries the same ``Input Dims``
    list -- but those describe the wrapper call, not the individual kernel.
    Transplanting them as kernel scope would assert that N different kernels
    have identical operands, which is the claim ``one_to_one_launch`` exists to
    prevent on the formal side.  Carry the distinction instead of erasing it.
    """
    return ("kernel" if (shape or {}).get("source") == "kernel_exact"
            else "wrapper")


def _table_key(table):
    return "%s|%s" % (table.get("pattern_id"), table.get("phase"))


def build(formal_table_doc, mapping_table_doc):
    """Return the per-row two-trace op map plus a summary."""
    mapping_tables = {
        _table_key(table): table
        for table in mapping_table_doc.get("tables", [])
    }
    entries = {}
    table_reports = []
    for formal_table in formal_table_doc.get("tables", []):
        key = _table_key(formal_table)
        mapping_table = mapping_tables.get(key)
        formal_rows = formal_table.get("rows", [])
        if mapping_table is None:
            table_reports.append({
                "table": key,
                "status": "no_mapping_table",
                "formal_rows": len(formal_rows),
                "matched_rows": 0,
            })
            continue
        mapping_rows = mapping_table.get("rows", [])
        formal_layer = formal_table.get("representative_layer_id")
        mapping_layer = mapping_table.get("representative_layer_id")
        representative_match = (
            "same_layer" if formal_layer == mapping_layer
            else "different_layer_same_pattern")
        matches = align_rows(formal_rows, mapping_rows)

        matched = op_recovered = shape_recovered = 0
        kernel_scope_shapes = 0
        for index, formal_row in enumerate(formal_rows):
            if index not in matches:
                continue
            mapping_index, name_match = matches[index]
            mapping_row = mapping_rows[mapping_index]
            delta = mapping_index - index
            matched += 1
            parent = mapping_row.get("parent_operator") or {}
            shape = mapping_row.get("shape") or {}
            has_op = parent.get("mapping_level") not in (None, "unresolved")
            has_shape = _usable_shape(shape)
            if has_op:
                op_recovered += 1
            if has_shape:
                shape_recovered += 1
                if _shape_scope(shape) == "kernel":
                    kernel_scope_shapes += 1
            entries[formal_row["row_id"]] = {
                "row_id": formal_row["row_id"],
                "pattern_id": formal_table.get("pattern_id"),
                "phase": formal_table.get("phase"),
                "formal_pos": formal_row.get("pos", index),
                "mapping_pos": mapping_row.get("pos", mapping_index),
                "raw_name": formal_row.get("raw_name"),
                "match": {
                    "binding": "kernel_name_plus_ordered_position",
                    "kernel_name_match": name_match,
                    "position_delta": delta,
                    "representative_layer_match": representative_match,
                    "formal_representative_layer_id": formal_layer,
                    "mapping_representative_layer_id": mapping_layer,
                    "confidence": _confidence(
                        name_match, delta, representative_match),
                },
                "op": {
                    "canonical_op": parent.get("canonical_op", "unresolved"),
                    "op_instance_id": parent.get("op_instance_id"),
                    "mapping_level": parent.get("mapping_level", "unresolved"),
                    "mapping_cardinality": parent.get(
                        "mapping_cardinality", "unresolved"),
                    "device_launch_count": parent.get("device_launch_count"),
                    "recovered": bool(has_op),
                },
                "shape": {
                    "input_dims": shape.get("input_dims"),
                    "input_types": shape.get("input_types"),
                    "source": shape.get("source", "unresolved"),
                    "scope": _shape_scope(shape),
                    "recovered": bool(has_shape),
                },
            }
        table_reports.append({
            "table": key,
            "status": "mapped",
            "formal_rows": len(formal_rows),
            "mapping_rows": len(mapping_rows),
            "matched_rows": matched,
            "op_recovered_rows": op_recovered,
            "shape_recovered_rows": shape_recovered,
            "kernel_scope_shape_rows": kernel_scope_shapes,
            "wrapper_scope_shape_rows": shape_recovered - kernel_scope_shapes,
            "representative_layer_match": representative_match,
        })

    total_rows = sum(item["formal_rows"] for item in table_reports)
    return {
        "schema_version": 1,
        "producer": "semantic_two_trace_mapping.py",
        "binding": "kernel_name_plus_ordered_position",
        "binding_notes": (
            "kernel names must match (exact, then normalized within gaps); "
            "among equal names the k-th occurrence binds to the k-th "
            "occurrence via an order-preserving longest common subsequence; "
            "unmatched rows keep their original evidence"),
        "row_count": total_rows,
        "matched_row_count": sum(
            item["matched_rows"] for item in table_reports),
        "op_recovered_row_count": sum(
            item.get("op_recovered_rows", 0) for item in table_reports),
        "shape_recovered_row_count": sum(
            item.get("shape_recovered_rows", 0) for item in table_reports),
        "kernel_scope_shape_row_count": sum(
            item.get("kernel_scope_shape_rows", 0) for item in table_reports),
        "wrapper_scope_shape_row_count": sum(
            item.get("wrapper_scope_shape_rows", 0) for item in table_reports),
        "tables": table_reports,
        "entries": entries,
    }


def drop_tables(document, table_keys):
    """Remove the bindings for tables the identity gate would not vouch for.

    Called after :func:`semantic_workload_identity.verify` reports which
    ``(pattern, phase)`` windows the mapping replay did not cover the same way.
    Their rows keep whatever evidence they already had, so dropping is always
    safe; leaving them would let a positional alignment across two different
    step buckets reach the published table.
    """
    keys = set(table_keys or ())
    if not keys:
        return document
    dropped = {
        row_id: entry for row_id, entry in
        (document.get("entries") or {}).items()
        if "%s|%s" % (entry.get("pattern_id"), entry.get("phase")) in keys}
    for row_id in dropped:
        del document["entries"][row_id]
    for report in document.get("tables", []):
        if report.get("table") in keys:
            report["status"] = "skipped_not_workload_identical"
            for name in ("matched_rows", "op_recovered_rows",
                         "shape_recovered_rows", "kernel_scope_shape_rows",
                         "wrapper_scope_shape_rows"):
                if name in report:
                    report[name] = 0
    document["skipped_tables"] = sorted(keys)
    document["dropped_entry_count"] = len(dropped)
    for name, field in (
            ("matched_row_count", "matched_rows"),
            ("op_recovered_row_count", "op_recovered_rows"),
            ("shape_recovered_row_count", "shape_recovered_rows"),
            ("kernel_scope_shape_row_count", "kernel_scope_shape_rows"),
            ("wrapper_scope_shape_row_count", "wrapper_scope_shape_rows")):
        if name in document:
            document[name] = sum(
                report.get(field, 0) for report in document.get("tables", []))
    return document


def build_from_traces(formal_table_doc, mapping_trace_paths, patterns_path,
                      out_dir):
    """Run the ordinary semantic mapping over the graph-off trace(s), then bind.

    ``profile_by_stage`` emits one rank-0 window per phase, so several mapping
    traces may be supplied; each contributes the tables for the phase it
    captured.  A later trace never overwrites a table an earlier one already
    produced for the same (pattern, phase).
    """
    if isinstance(mapping_trace_paths, str):
        mapping_trace_paths = [mapping_trace_paths]
    combined = {"tables": []}
    seen = set()
    mapping_results = []
    for index, trace_path in enumerate(mapping_trace_paths):
        mapping_dir = os.path.join(
            out_dir, "mapping_trace_semantics", "trace_%02d" % index)
        os.makedirs(mapping_dir, exist_ok=True)
        mapping_result = semantic_kernel_mapping.build(
            trace_path, patterns_path, mapping_dir)
        mapping_result["mapping_trace"] = os.path.abspath(trace_path)
        mapping_results.append(mapping_result)
        with open(mapping_result["semantic_table_json"]) as fh:
            doc = json.load(fh)
        for table in doc.get("tables", []):
            key = _table_key(table)
            if key in seen:
                continue
            seen.add(key)
            combined["tables"].append(table)
    document = build(formal_table_doc, combined)
    document["mapping_traces"] = [
        os.path.abspath(path) for path in mapping_trace_paths]
    document["mapping_semantics"] = mapping_results
    return document, combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-table", required=True)
    parser.add_argument("--mapping-trace", action="append", required=True)
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--result-json", default="")
    args = parser.parse_args()
    with open(args.formal_table) as fh:
        formal_table_doc = json.load(fh)
    os.makedirs(args.out_dir, exist_ok=True)
    document, _ = build_from_traces(
        formal_table_doc, args.mapping_trace, args.patterns, args.out_dir)
    out_path = args.result_json or os.path.join(
        args.out_dir, "TWO_TRACE_OP_MAP.json")
    with open(out_path, "w") as fh:
        json.dump(document, fh, indent=2)
    print(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
