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


def run(scenario):
    p = subprocess.run([NODE, str(HERE / "kf_workflow_mockrun.mjs"), scenario, str(E2E_DIR)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def labels(out):
    return [c["label"] for c in out["calls"]]


def test_accept_path_reuses_geak_roles_and_banks_the_overlay():
    out = run("accept")
    r = out["result"]
    assert labels(out) == ["kf:director:setup", "kf:profiler", "kf:analyst",
                           "kf:extract c1", "kf:extract c2", "kf:integrate c1"]
    # every non-analyst agent is a stock GEAK role file, and each one exists
    for c in out["calls"]:
        m = re.search(r"Read (\S+/roles/(\w+)\.md)", c["prompt"])
        if c["label"] != "kf:analyst":
            assert m and Path(m.group(1)).is_file(), c["label"]
    analyst = out["calls"][2]["prompt"]
    assert "Read /prompts/p.txt" in analyst and "kf_trace_digest.py" in analyst
    assert "TRACE_DIR: /tmp/kf_mock/kf/profile" in analyst
    # author = the stock kernel lane in author mode, one per recommended candidate (c3 not recommended)
    assert [w["scriptPath"].endswith("kernel_workflow/kernel_lane.js") for w in out["workflows"]] == [True, True]
    assert {w["mode"] for w in out["workflows"]} == {"author"}
    assert r["status"] == "accepted"
    assert r["final_overlay"] == "/tmp/kf_mock/kf/overlay/cand_c1"
    assert r["throughput_speedup"] == pytest.approx(1.03)
    assert [a["id"] for a in r["accepted"]] == ["c1"]
    assert [(x["id"], x["stage"]) for x in r["rejected"]] == [("c2", "author")]
    integ = out["calls"][-1]["prompt"]
    assert '"winner_kind":"authored"' in integ and '"code_patch":"diff --git a b"' in integ
    assert "Read " + str(E2E_DIR) + "/roles/e2e_integrator.md" in integ


def test_incomplete_ab_is_finished_before_deciding():
    out = run("retry")
    assert labels(out)[-2:] == ["kf:integrate c1", "kf:integrate c1 (finish 1)"]
    assert out["result"]["status"] == "accepted"
    assert out["result"]["throughput_speedup"] == pytest.approx(1.02)


def test_stack_and_reject_do_not_bank():
    for scen in ("stack", "reject"):
        r = run(scen)["result"]
        assert r["status"] == "no_gain" and r["accepted"] == [] and r["final_overlay"] == ""
        assert r["throughput_speedup"] == 1


def test_extraction_without_frozen_baseline_is_not_authored():
    out = run("unfrozen")
    assert [w["kernel_path"] for w in out["workflows"]] == ["/tmp/kf_mock/kf/kernels/c2_task"]
    r = out["result"]
    assert r["status"] == "no_gain" and ("c1", "extract") in [(x["id"], x["stage"]) for x in r["rejected"]]


def test_analysis_only_with_given_traces_needs_no_server():
    out = run("analysis_only")
    assert labels(out) == ["kf:analyst"]
    assert "TRACE_DIR: /traces/x" in out["calls"][0]["prompt"]
    assert out["result"]["status"] == "analysis_only" and out["workflows"] == []


def test_setup_failure_returns_cleanly():
    r = run("setup_fail")["result"]
    assert r["status"] == "error" and r["final_overlay"] == ""
