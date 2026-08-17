#!/usr/bin/env python3
"""Combine per-stage Semantics tables into one final prefill+decode table.

``profile_by_stage=True`` -- which the official benchmark hard-codes -- writes
prefill and decode to *separate* rank-0 trace files. Each stage therefore has
to be analysed by its own Semantics run: only the prefill file carries
DecoderLayer module spans, and only the decode file needs a transferred layer
boundary. That is a property of the capture, not a choice.

What was a choice, and a bad one, is publishing those runs as two sibling
output trees. The deliverable is one Pattern/Phase/Layer/Kernel table, so the
per-stage runs are an implementation detail and belong under one output
directory with one published table.

This module does the last step: concatenate the per-stage table documents into
one, prefill tables first and decode tables after, which is the phase-1
presentation contract. It is deliberately a *concatenation*: no row is
recomputed, reordered inside its table, re-timed, or re-attributed. Device
order and duration stay each stage's own, as they must -- the two stages were
profiled in different windows and their microsecond clocks are not comparable.
"""
import json
import os

import semantic_kernel_mapping
import semantic_shape_merge


PHASE_ORDER = ("prefill", "decode")


def _phase_rank(table):
    phase = str(table.get("phase") or "")
    return (PHASE_ORDER.index(phase) if phase in PHASE_ORDER
            else len(PHASE_ORDER), phase)


def combine_documents(stage_documents, kind="1_2", quality_status=""):
    """Merge stage table documents into one, prefill first then decode.

    ``stage_documents`` is a list of ``(stage_label, document)`` pairs. A
    document is the ``pattern_layer_kernel_table.json`` of one stage's run.
    """
    tables = []
    sources = []
    seen = {}
    duplicate_row_ids = []
    for label, document in stage_documents:
        for table in document.get("tables", []):
            key = (table.get("pattern_id"), table.get("phase"))
            if key in seen:
                raise ValueError(
                    "two stages published the same (pattern, phase) table "
                    "%s: they are not different stages" % (key,))
            seen[key] = label
            tables.append(table)
        sources.append({
            "stage": label,
            "trace_path": document.get("trace_path"),
            "trace_sha256": document.get("trace_sha256"),
            "table_phases": document.get("table_phases"),
            "tables": [
                "%s|%s" % (table.get("pattern_id"), table.get("phase"))
                for table in document.get("tables", [])],
        })
    tables.sort(key=_phase_rank)

    # row_id is an index into one trace's event list, so two stages can mint
    # the same one. Nothing downstream consumes this combined document -- it
    # is the deliverable -- but say so rather than let a reader assume the
    # ids are global.
    by_row = {}
    for table in tables:
        for row in table.get("rows", []):
            key = row.get("row_id")
            by_row.setdefault(key, []).append(table.get("phase"))
    duplicate_row_ids = sorted(
        key for key, phases in by_row.items() if len(phases) > 1)

    first = stage_documents[0][1] if stage_documents else {}
    combined = {
        "schema_version": first.get("schema_version", 2),
        "patterns_path": first.get("patterns_path"),
        "table_phases": [
            phase for phase in PHASE_ORDER
            if any(table.get("phase") == phase for table in tables)],
        "stage_sources": sources,
        "combination": {
            "producer": "semantic_phase_tables.combine_documents",
            "method": "verbatim_table_concatenation",
            "order": "prefill tables first, then decode tables",
            "rows_recomputed": False,
            "row_id_scope": "per stage trace, not unique across stages",
            "duplicate_row_ids_across_stages": len(duplicate_row_ids),
            "note": (
                "profile_by_stage writes prefill and decode to separate rank0 "
                "files, so each stage is analysed by its own run. Every table "
                "here is that run's untouched output; only the table lists "
                "were concatenated. Durations are per stage window and are "
                "not comparable across phases."),
        },
        "tables": tables,
    }
    if kind == "1_1":
        markdown = semantic_kernel_mapping._markdown(
            tables, {"status": quality_status or "see per-stage quality"})
    else:
        markdown = semantic_shape_merge._markdown(combined)
    return combined, markdown


def combine(stage_paths, out_json, out_md, kind="1_2", quality_status=""):
    """Combine stage table JSON files into ``out_json``/``out_md``."""
    documents = []
    for label, path in stage_paths:
        with open(path) as fh:
            documents.append((label, json.load(fh)))
    combined, markdown = combine_documents(documents, kind, quality_status)
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(combined, fh, indent=2)
    with open(out_md, "w") as fh:
        fh.write(markdown)
    return {
        "semantic_table_json": out_json,
        "semantic_table_md": out_md,
        "table_phases": combined["table_phases"],
        "table_count": len(combined["tables"]),
        "row_count": sum(
            len(table.get("rows", [])) for table in combined["tables"]),
        "stage_sources": combined["stage_sources"],
    }
