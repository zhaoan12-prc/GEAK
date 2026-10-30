"""kf_workflow.js control flow under scripted agents (needs node: PATH or GEAK_NODE_BIN)."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
E2E_DIR = HERE.parents[1]
NODE = os.environ.get("GEAK_NODE_BIN") or shutil.which("node")
pytestmark = pytest.mark.skipif(not NODE, reason="node not available (set GEAK_NODE_BIN)")
KF = "/tmp/kf_mock/kf"


def run(scenario):
    p = subprocess.run([NODE, str(HERE / "kf_workflow_mockrun.mjs"), scenario, str(E2E_DIR)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def labels(out):
    return [c["label"] for c in out["calls"]]


def rejected(r):
    return [(x["id"], x["stage"]) for x in r["rejected"]]


def test_accept_path_reuses_geak_roles_and_confirms_the_stack():
    out = run("accept")
    r = out["result"]
    # c1 is recommended at difficulty 3, so its smaller exclusive alternative c1s is authored too;
    # c3 is never touched (not recommended, no seam).
    assert labels(out) == ["kf:director:setup", "kf:profiler", "kf:analyst",
                           "kf:extract c1", "kf:extract c1s", "kf:extract c2",
                           "kf:integrate c1", "kf:director:validate"]
    for c in out["calls"]:
        if c["label"] != "kf:analyst":   # every other agent is a stock GEAK role file that exists
            m = re.search(r"Read (\S+/roles/(\w+)\.md)", c["prompt"])
            assert m and Path(m.group(1)).is_file(), c["label"]
    analyst = out["calls"][2]["prompt"]
    assert "Read /prompts/p.txt" in analyst and "kf_trace_digest.py" in analyst and "fallback (id" in analyst
    assert f"TRACE_DIR: {KF}/profile" in analyst
    assert all(w["scriptPath"].endswith("kernel_workflow/kernel_lane.js") and w["mode"] == "author"
               for w in out["workflows"]) and len(out["workflows"]) == 3
    integ = out["calls"][6]["prompt"]
    assert '"winner_kind":"authored"' in integ and '"code_patch":"diff c1"' in integ
    validate = out["calls"][7]["prompt"]
    assert f"FINAL_OVERLAY: {KF}/overlay/cand_accepted" in validate and "MEASUREMENT_MODE: isolated_server" in validate
    assert "REPLICAS: 2" in validate
    assert r["status"] == "accepted" and r["final_overlay"] == f"{KF}/overlay/cand_accepted"
    assert r["throughput_speedup"] == pytest.approx(1.025)          # the director's combined number
    assert [a["id"] for a in r["accepted"]] == ["c1"]
    assert rejected(r) == [("c2", "author")]


def test_fallback_is_applied_when_the_primary_has_no_isolated_win():
    out = run("fallback")
    assert "kf:integrate c1" not in labels(out) and "kf:integrate c1s" in labels(out)
    r = out["result"]
    assert r["status"] == "accepted" and [(a["id"], a["slot"]) for a in r["accepted"]] == [("c1s", "c1")]


def test_fallback_is_applied_when_the_primary_fails_the_ab():
    out = run("reject")
    assert labels(out)[-3:] == ["kf:integrate c1", "kf:integrate c1s", "kf:director:validate"]
    r = out["result"]
    assert [a["id"] for a in r["accepted"]] == ["c1s"] and ("c1", "apply") in rejected(r)


def test_incomplete_ab_is_finished_before_deciding():
    out = run("retry")
    assert labels(out)[-3:] == ["kf:integrate c1", "kf:integrate c1 (finish 1)", "kf:director:validate"]
    assert out["result"]["status"] == "accepted"


def test_stack_is_carried_but_needs_the_combined_confirmation():
    r = run("stack")["result"]
    assert [a["gate"] for a in r["accepted"]] == ["stack"]
    assert r["validation"]["validation_status"] == "validated_no_win"
    assert r["status"] == "no_gain" and r["final_overlay"] == "" and r["throughput_speedup"] == 1


def test_extraction_without_frozen_baseline_is_not_authored():
    out = run("unfrozen")
    assert sorted(w["kernel_path"] for w in out["workflows"]) == [f"{KF}/kernels/c1s_task", f"{KF}/kernels/c2_task"]
    r = out["result"]
    assert ("c1", "extract") in rejected(r) and [a["id"] for a in r["accepted"]] == ["c1s"]


def test_analysis_only_with_given_traces_needs_no_server():
    out = run("analysis_only")
    assert labels(out) == ["kf:analyst"]
    assert "TRACE_DIR: /traces/x" in out["calls"][0]["prompt"]
    assert out["result"]["status"] == "analysis_only" and out["workflows"] == []


def test_setup_failure_returns_cleanly():
    r = run("setup_fail")["result"]
    assert r["status"] == "error" and r["final_overlay"] == ""
