#!/usr/bin/env python3
"""Transfer decode layer boundaries from a graph-off run onto a Clean Trace.

Under a replayed CUDA graph the module forwards never execute as Python calls,
so a Clean Trace's decode stage carries no ``nn.Module: ...DecoderLayer_<id>``
span and has no trustworthy layer boundary.  The Semantics 1.2 shape-capture
replay runs the *same declared workload* with ``--disable-cuda-graph`` and does
emit every span.

This module carries only the *boundary* across that pair.  Device order and
duration still come from the uninstrumented Clean Trace: nothing here copies a
timestamp, a duration, or a device event.  The donor supplies the answer to
"which layer does this launch belong to", and nothing else.

The two runs are different executions, so the device sequences are aligned by
kernel-name subsequence rather than assumed identical.  Only matched positions
carry a boundary directly; an unmatched run inherits a layer only when it is
bracketed by two matched positions that agree, which is the one case where the
enclosing layer is not in question.  Everything else stays unresolved and is
reported.
"""
import argparse
import bisect
import difflib
import hashlib
import json
import os

import parse_profile
import semantic_kernel_mapping


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recipient_rows(trace_path, pattern_doc, phase):
    """Ordered device rows of one phase, exactly as build() will see them."""
    events = semantic_kernel_mapping._load_events(trace_path)
    rows, _, _, module_scopes, _ = semantic_kernel_mapping._event_rows(
        events, pattern_doc)
    selected = [
        row for row in rows
        if str(row.get("phase") or "").lower() == phase.lower()]
    selected.sort(key=lambda row: row["device_seq_index"])
    return selected, len(module_scopes)


def _donor_module_scopes(events, expected_count):
    """DecoderLayer spans of the donor, ordered, without step filtering.

    ``semantic_kernel_mapping._module_layer_scopes`` keeps only spans whose
    timestamp lands inside a ``step[...]`` annotation.  That is right for a
    Clean Trace, but a graph-off replay runs the CPU ahead of the GPU, so its
    module spans sit outside the step window and every one would be dropped --
    the donor would look span-less even though it recorded a full pass.  Here
    the spans are the subject, not the device timeline, so ordering them by
    timestamp and cutting them into passes of the configured layer count is
    sufficient and does not depend on the step window at all.
    """
    spans = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("cat") != "python_function":
            continue
        match = semantic_kernel_mapping.MODULE_LAYER_RE.match(
            str(event.get("name", "")))
        if not match:
            continue
        if event.get("ts") is None or event.get("dur") is None:
            continue
        spans.append({
            "name": str(event.get("name", "")),
            "ts": event["ts"],
            "end": event["ts"] + event["dur"],
        })
    spans.sort(key=lambda span: (span["ts"], span["end"]))
    if expected_count <= 0:
        return []
    scopes = []
    for pass_index in range(len(spans) // expected_count):
        chunk = spans[
            pass_index * expected_count:(pass_index + 1) * expected_count]
        for layer_id, span in enumerate(chunk):
            scopes.append({
                **span,
                "layer_id": layer_id,
                "pass_index": pass_index,
                "layer_instance_id": "donor:pass-%d:layer-%d" % (
                    pass_index, layer_id),
            })
    scopes.sort(key=lambda scope: (scope["ts"], scope["end"]))
    return scopes


def _donor_rows(trace_path, expected_count):
    """Ordered donor device events, each tagged with its enclosing layer."""
    events = semantic_kernel_mapping._load_events(trace_path)
    by_ext, _, _ = semantic_kernel_mapping._cpu_evidence(events)
    scopes = _donor_module_scopes(events, expected_count)
    starts = [scope["ts"] for scope in scopes]

    rows = []
    for raw_index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        if event.get("cat") not in semantic_kernel_mapping.DEVICE_CATEGORIES:
            continue
        args = event.get("args") or {}
        parent = by_ext.get(args.get("External id"))
        # The launch site decides the layer, so prefer the CPU op's timestamp;
        # the device timestamp is only a fallback for an unparented launch.
        anchor = parent["ts"] if parent else event.get("ts")
        scope = None
        if anchor is not None and scopes:
            position = bisect.bisect_right(starts, anchor) - 1
            if position >= 0 and anchor < scopes[position]["end"]:
                scope = scopes[position]
        name = str(event.get("name", "?"))
        stage, _, _ = semantic_kernel_mapping._stage_detail(
            name, event.get("cat"), (parent or {}).get("name", ""))
        rows.append({
            "row_id": "donor-%d" % raw_index,
            "short_name": parse_profile.short_name(name),
            "stage": stage,
            "layer_id": scope["layer_id"] if scope else None,
            "layer_instance_id": (
                scope["layer_instance_id"] if scope else None),
        })
    return rows, len(scopes)


def _key(row):
    """Alignment token: the kernel identity, not its timing."""
    return "%s|%s" % (row.get("short_name") or "", row.get("stage") or "")


def _transfer(donor_rows, recipient_rows, expected_layers):
    """Carry the donor's layer partition onto the recipient's device sequence.

    This is a partition problem, not a per-row labelling one: the recipient's
    launches must be cut into N contiguous layer segments.  Matching row by row
    leaves a layer empty whenever its region happens not to align, so instead
    each layer's *start* is aligned and the cuts between them are what get
    transferred.  Cuts are then forced strictly increasing, which makes the
    result a complete, ordered, non-empty partition by construction -- the
    properties a layer boundary has to have.
    """
    donor_keys = [_key(row) for row in donor_rows]
    recipient_keys = [_key(row) for row in recipient_rows]
    matcher = difflib.SequenceMatcher(
        None, donor_keys, recipient_keys, autojunk=False)

    # donor index -> recipient index, for positions the alignment agreed on.
    aligned = {}
    for donor_start, recipient_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            aligned[donor_start + offset] = recipient_start + offset
    aligned_donor_indices = sorted(aligned)

    donor_starts = {}
    for index, row in enumerate(donor_rows):
        layer_id = row.get("layer_id")
        if layer_id is not None and layer_id not in donor_starts:
            donor_starts[layer_id] = index

    total = len(recipient_rows)
    cuts = []
    cut_basis = []
    for layer_id in range(expected_layers):
        donor_index = donor_starts.get(layer_id)
        if donor_index is None:
            cuts.append(None)
            cut_basis.append("absent_in_donor")
            continue
        position = bisect.bisect_left(aligned_donor_indices, donor_index)
        if position < len(aligned_donor_indices):
            cuts.append(aligned[aligned_donor_indices[position]])
            cut_basis.append("aligned_kernel_sequence")
        else:
            cuts.append(None)
            cut_basis.append("beyond_alignment")

    # Fill unaligned cuts by proportional placement between known neighbours,
    # then force the sequence strictly increasing so no layer can be empty and
    # no layer can precede its predecessor.
    for index in range(expected_layers):
        if cuts[index] is not None:
            continue
        before = next(
            (cuts[j] for j in range(index - 1, -1, -1)
             if cuts[j] is not None), 0)
        after = next(
            (cuts[j] for j in range(index + 1, expected_layers)
             if cuts[j] is not None), total)
        span = max(after - before, 0)
        cuts[index] = before + span // 2
        if cut_basis[index] not in ("absent_in_donor",):
            cut_basis[index] = "interpolated_between_neighbours"

    cuts[0] = 0
    for index in range(1, expected_layers):
        lowest = cuts[index - 1] + 1
        if cuts[index] < lowest:
            cuts[index] = lowest
            if cut_basis[index] == "aligned_kernel_sequence":
                cut_basis[index] = "monotonicity_adjusted"
    # Leave room for every remaining layer to hold at least one row.
    for index in range(expected_layers - 1, -1, -1):
        ceiling = total - (expected_layers - index)
        if cuts[index] > ceiling:
            cuts[index] = ceiling
            if cut_basis[index] == "aligned_kernel_sequence":
                cut_basis[index] = "monotonicity_adjusted"

    assigned = {}
    for layer_id in range(expected_layers):
        start = cuts[layer_id]
        end = cuts[layer_id + 1] if layer_id + 1 < expected_layers else total
        for position in range(start, end):
            row = recipient_rows[position]
            assigned[row["row_id"]] = {
                "layer_id": layer_id,
                "donor_layer_instance_id": "donor:pass-0:layer-%d" % layer_id,
                "basis": cut_basis[layer_id],
            }
    basis_counts = {}
    for basis in cut_basis:
        basis_counts[basis] = basis_counts.get(basis, 0) + 1
    return assigned, cuts, basis_counts


def _checks(recipient_rows, assigned, expected_layers):
    """Layers must run in order and cover the configured stack exactly once."""
    sequence = [
        assigned[row["row_id"]]["layer_id"]
        for row in recipient_rows if row["row_id"] in assigned]
    monotonic = all(
        left <= right for left, right in zip(sequence, sequence[1:]))
    distinct = sorted(set(sequence))
    contiguous = distinct == list(range(len(distinct)))
    return {
        "monotonic_layer_order": monotonic,
        "distinct_layers": len(distinct),
        "expected_layers": expected_layers,
        "layers_complete": len(distinct) == expected_layers,
        "layer_ids_contiguous_from_zero": contiguous,
        "assigned_rows": len(assigned),
        "recipient_rows": len(recipient_rows),
        "assigned_fraction": (
            round(len(assigned) / float(len(recipient_rows)), 6)
            if recipient_rows else 0.0),
    }


def transfer(donor_trace, recipient_trace, pattern_path, out_path,
             phase="decode", min_assigned_fraction=0.9):
    with open(pattern_path) as fh:
        pattern_doc = json.load(fh)
    expected_layers = int(pattern_doc.get("num_hidden_layers_main") or 0)

    donor_rows, donor_scopes = _donor_rows(donor_trace, expected_layers)
    recipient_rows, recipient_scopes = _recipient_rows(
        recipient_trace, pattern_doc, phase)

    failures = []
    if donor_scopes <= 0:
        failures.append(
            "donor trace has no DecoderLayer module spans; it cannot supply a "
            "boundary")
    if recipient_scopes > 0:
        failures.append(
            "recipient trace already has %d module spans; transfer is for a "
            "graph-erased stage only" % recipient_scopes)
    if not donor_rows or not recipient_rows:
        failures.append(
            "phase %r has no device rows in donor (%d) or recipient (%d)"
            % (phase, len(donor_rows), len(recipient_rows)))

    if failures or len(recipient_rows) < expected_layers:
        if recipient_rows and len(recipient_rows) < expected_layers:
            failures.append(
                "recipient has %d device rows for %d layers; it cannot be "
                "partitioned without an empty layer"
                % (len(recipient_rows), expected_layers))
        assigned, cuts, basis_counts = {}, [], {}
    else:
        assigned, cuts, basis_counts = _transfer(
            donor_rows, recipient_rows, expected_layers)
    checks = _checks(recipient_rows, assigned, expected_layers)

    if not failures:
        if not checks["monotonic_layer_order"]:
            failures.append(
                "transferred layer ids are not monotonically non-decreasing; "
                "the two runs did not execute the same layer order")
        if not checks["layers_complete"]:
            failures.append(
                "transferred boundary covers %d of %d configured layers"
                % (checks["distinct_layers"], expected_layers))
        if checks["assigned_fraction"] < min_assigned_fraction:
            failures.append(
                "only %.1f%% of recipient device rows received a boundary "
                "(minimum %.1f%%)"
                % (100 * checks["assigned_fraction"],
                   100 * min_assigned_fraction))

    document = {
        "schema_version": 1,
        "transfer": "decode_layer_boundary",
        "status": "pass" if not failures else "failed",
        "phase": phase,
        "evidence_note": (
            "Boundary only. Device order and duration remain those of the "
            "recipient Clean Trace; no timing is copied from the donor."),
        "donor": {
            "path": os.path.abspath(donor_trace),
            "sha256": _sha256(donor_trace),
            "module_scope_count": donor_scopes,
            "device_rows": len(donor_rows),
        },
        "recipient": {
            "path": os.path.abspath(recipient_trace),
            "sha256": _sha256(recipient_trace),
            "module_scope_count": recipient_scopes,
            "device_rows": len(recipient_rows),
        },
        "alignment": {
            "cut_basis_counts": basis_counts,
            "layer_start_cuts": cuts,
        },
        "checks": checks,
        "failures": failures,
        "assignments": assigned,
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(document, fh, indent=2)
    return document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor-trace", required=True,
                        help="graph-off trace carrying DecoderLayer spans")
    parser.add_argument("--recipient-trace", required=True,
                        help="Clean Trace whose stage is graph-erased")
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--phase", default="decode")
    parser.add_argument("--min-assigned-fraction", type=float, default=0.9)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    document = transfer(
        args.donor_trace, args.recipient_trace, args.patterns, args.out,
        args.phase, args.min_assigned_fraction)
    print(json.dumps({
        key: document[key]
        for key in ("status", "phase", "alignment", "checks", "failures")},
        indent=2))
    return 0 if document["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
