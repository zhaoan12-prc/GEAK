#!/usr/bin/env python3
"""Which phases a semantic layer table actually carries.

KernelFusion screens whatever the table holds. On a decode-dominated workload a
table without decode rows is not "partial", it is blind to almost all of the
request time: Qwen3.5-397B-A17B-FP8 (ISL 8192 / OSL 1024 / CONC 4, decode weight
255.75 vs prefill 0.5) lost its decode table when the shape-capture replay timed
out on the serving-GPU lock, and the screen built a one-row board from prefill
look-alikes. The workflow reads this before screening and re-runs the replay once.

Prints ONE JSON object:
  {"ok": true, "table_phases": [...], "required_phases": [...],
   "missing_phases": [...], "decode_covered": bool}
"""
import argparse
import json
import sys


def coverage(table):
    phases = set()
    for spec in table.get("tables") or []:
        if spec.get("rows"):
            phases.add(str(spec.get("phase") or "unresolved"))
    for row in table.get("rows") or []:
        phases.add(str(row.get("phase") or "unresolved"))
    meta = table.get("phase_coverage") or {}
    required = set(meta.get("required_phases") or [])
    return {
        "ok": True,
        "table_phases": sorted(phases),
        "required_phases": sorted(required),
        "missing_phases": sorted(required - phases),
        "decode_covered": "decode" in phases,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("table")
    a = ap.parse_args(argv)
    try:
        with open(a.table) as fh:
            table = json.load(fh)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "reason": "unreadable table: %s" % exc}))
        return 1
    print(json.dumps(coverage(table), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
