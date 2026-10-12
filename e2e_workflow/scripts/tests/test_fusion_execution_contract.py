"""KernelFusion execution (unit-side gate, lease, retry queue, stacking, final gate).

Root causes B.8-B.11 of the Qwen3.5-397B run (20261010_test_e2e_qwen35):
  B.8  Phase 3.0 never ran (04_FUSION_UNITSIDE.md "未生成");
  B.9  e03 backgrounded its A/B and returned; e04..e08 saw the GPUs held and returned
       "blocked" -- and "blocked" was terminal, never retried;
  B.10 the accuracy gate was never calibrated (base-vs-base failed it);
  B.11 the stacked re-gate needed >= 2 stack_pending entries and never ran.

The behaviour tests below cut the REAL KernelFusion execution block and the lease helpers
out of e2e_workflow.js and run them under node with scripted agents (skipped when node is
absent; set NODE=/path/to/node).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW = os.path.dirname(os.path.dirname(HERE))
NODE = os.environ.get("NODE") or shutil.which("node")


def _source():
    with open(os.path.join(WORKFLOW, "e2e_workflow.js")) as fh:
        return fh.read()


def _role(name):
    with open(os.path.join(WORKFLOW, "roles", name)) as fh:
        return fh.read()


def _between(src, start, end):
    i = src.index(start)
    return src[i:src.index(end, i)]


class SourceContractTest(unittest.TestCase):
    def test_unitside_runs_before_apply_back_and_is_skippable_only_explicitly(self):
        src = _source()
        unit = src.index("roleAgent('fusion_unit_validator', 'validate_one'")
        agg = src.index("roleAgent('fusion_unit_validator', 'aggregate'")
        apply_one = src.index("roleAgent('fusion_integrator', 'apply_one'")
        self.assertLess(unit, agg)
        self.assertLess(agg, apply_one)
        self.assertIn("A.fusion_unitside != null ? A.fusion_unitside : 'true'", src)

    def test_every_serving_call_runs_under_the_lease(self):
        src = _source()
        for needle in ("roleAgent('fusion_integrator', 'apply_one'",
                       "roleAgent('fusion_integrator', isConfirm ? 'apply_stack_confirm' : 'apply_stack_step'",
                       "roleAgent('fusion_integrator', 'calibrate_accuracy'"):
            i = src.index(needle)
            window = src[i - 400:i]
            self.assertIn("withFusionServingLease(", window, needle)

    def test_lease_rules_forbid_backgrounding_and_explain_polling(self):
        src = _source()
        rules = _between(src, "function fusionLeaseRules(", "\nasync function fusionLeaseOp(")
        for needle in ("nohup", "setsid", "trailing &", "--max-s 540", "alive_count 0",
                       "reason_code gpu_busy|not_finished|not_reached"):
            self.assertIn(needle, rules)

    def test_lease_runner_cannot_kill_anything_itself(self):
        src = _source()
        # Every lease op goes through the one-command script runner, which carries the rule.
        op = _between(src, "async function fusionLeaseOp(", "\n}\n")
        self.assertIn("fusionScriptOp(", op)
        runner = _between(src, "function fusionScriptOp(", "\n}\n")
        self.assertIn("do NOT signal, kill", runner)
        self.assertNotRegex(op + runner, r"pkill|killall|xargs kill")

    def test_stacking_is_greedy_not_stack_all_then_drop(self):
        src = _source()
        self.assertIn("if (stackPendingRows.length || poolStackCands.length) {", src)
        self.assertIn("scripts/fusion_stack_greedy.py", src)
        self.assertNotIn("roleAgent('fusion_integrator', 'apply_stack',", src)
        self.assertNotIn("STACK_MODE", src)

    def test_landing_target_is_an_arg_defaulting_to_two(self):
        src = _source()
        self.assertIn("A.fusion_min_landed != null ? A.fusion_min_landed : 2", src)
        self.assertIn("--min-landed ${FUSION_MIN_LANDED}", src)
        self.assertIn("landed: { count: landedN, min: FUSION_MIN_LANDED, met: landedN >= FUSION_MIN_LANDED }", src)

    def test_screen_gets_the_fusion_budget_and_item_minutes(self):
        src = _source()
        screen = _between(src, "roleAgent('kernel_fusion_analyst', 'screen'", "label: 'fusion-screen'")
        for needle in ("--fusion-budget-s FUSION_BUDGET_S", "--item-minutes ITEM_MINUTES",
                       "FUSION_BUDGET_S: Math.round(FUSION_BUDGET_MS / 1000)",
                       "ITEM_MINUTES: FUSION_ITEM_MINUTES"):
            self.assertIn(needle, screen)
        self.assertIn("FUSION_STACK_RESERVE_FRAC = Math.max(0.3,", src)

    def test_final_gate_runs_without_partial_coverage(self):
        src = _source()
        gate = _between(src, "'You are the KernelFusion apply-back gate runner.", "schema: FUSION_APPLYBACK_GATE_SCHEMA")
        self.assertIn("fusion_applyback_harness.py", gate)
        self.assertNotIn("--allow-partial-coverage", gate)

    def test_schema_has_a_non_terminal_retry_list(self):
        src = _source()
        schema = _between(src, "const FUSION_APPLY_SCHEMA", "['accepted_fusions']);")
        self.assertIn("retry: arrObj", schema)
        self.assertIn("unmeasured: arrObj", schema)

    def test_process_safety_text_is_unchanged(self):
        src = _source()
        self.assertIn("NEVER run global or pattern-matched process cleanup", src)
        self.assertIn("function withProcessSafety(prompt)", src)

    def test_fast_mode_still_skips_fusion(self):
        self.assertIn("if (!FAST_MODE && FUSION_DISCOVERY_ON && FUSION_BACKENDS.has(BACKEND))",
                      _source())


class RoleContractTest(unittest.TestCase):
    def test_integrator_forbids_backgrounding_and_returns_retry(self):
        role = _role("fusion_integrator.md")
        for needle in ("Never background an A/B and return", "nohup", "setsid",
                       "GPU_LEASE_SCRIPT", "--max-s 540", "reason_code: gpu_busy",
                       "PHASE=calibrate_accuracy", "fusion_accuracy_gate",
                       "PHASE=apply_stack_step", "PHASE=apply_stack_confirm", "--stack-step",
                       "KEEP_THRESHOLD_PCT", "TARGET_RUNG", "RUNG_ATTEMPTS", "deferred_budget",
                       "WIRE_RETRY", "wire_failure", "FALLBACK_MODE", "MIN_LANDED",
                       "`unmeasured`", "blocked_evidence"):
            self.assertIn(needle, role)
        self.assertNotIn("There is no `FUSION_UNITSIDE_JSON` gate", role)
        self.assertNotIn("let the call fail", role)

    def test_integrator_skips_e2e_accuracy_only_when_told(self):
        role = " ".join(_role("fusion_integrator.md").split())
        for needle in ("If ACCURACY_MODE=skip", "do NOT run gsm8k or the deterministic probe",
                       "UNIT_SIDE parity", "mode: 'skipped', reason: 'calibration_unusable'",
                       "decide on the A/B alone", "--accuracy-mode skip"):
            self.assertIn(needle, role)

    def test_unit_validator_knows_deeper_rungs_are_tested_on_demand(self):
        role = " ".join(_role("fusion_unit_validator.md").split())
        self.assertIn("ON DEMAND during apply-back", role)
        self.assertIn("UNIT_JOB_KIND=alt, one single call", role)

    def test_director_stops_a_reused_server_before_returning(self):
        role = _role("director.md")
        self.assertIn("REUSE_SERVER=1", role)
        self.assertIn("reap-recorded", role)
        self.assertIn("never by name or pattern", role)

    def test_unit_validator_covers_screen_format_and_both_sides(self):
        role = _role("fusion_unit_validator.md")
        for needle in ("SEMANTIC_TABLE_JSON", "self_authored", "kernel_source",
                       "unit_status_by_candidate", "--semantic-table", "GPU_LEASE_ID",
                       "UNIT_JOB_KIND", "<exec_id>.r<rung>", "candidate_pool"):
            self.assertIn(needle, role)


@unittest.skipUnless(NODE, "node not available")
class ExecutionBehaviourTest(unittest.TestCase):
    """Run the real execution block with scripted agents."""

    def _run(self, scenario, entries, unitside_on=True, calibration_on=True, retries=2,
             accepted_at_entry=0, pool=None, budget_s=28800, item_min=45, noise=1.0,
             min_landed=2, unit_all_rungs=False, accuracy_cache=True, return_root=False,
             root=None):
        src = _source()
        helpers = _between(src, "const FUSION_LEASE_SCRIPT =", "\n// Needs the whole serving TP group")
        helpers += _between(src, "// Needs the whole serving TP group", "\n}\n") + "\n}\n"
        block = _between(src, "  // ---- Phase 3.0: 单侧 gate (unit-side), MANDATORY",
                         "  // KernelFusion is an optional optimization track.")
        if root is None:
            root = tempfile.mkdtemp(prefix="kf_exec_")
            self.addCleanup(shutil.rmtree, root, True)
        # EVAL_DIR sits inside the experiment root, which holds .fusion_accuracy_cache.
        evaldir = tempfile.mkdtemp(prefix="eval_", dir=root)
        script = r"""
const SCEN = %(scen)s;
const calls = []; const logs = [];
const log = (m) => logs.push(String(m));
const WORKFLOW_DIR = %(wfdir)s; let EVAL_DIR = %(evaldir)s; const EXEC_PREFIX = '';
const PY = %(py)s;
const cp = require('child_process'); const fs = require('fs'); const pathm = require('path');
const ISL = 8192, OSL = 1024, CONC = 4; const BACKEND = 'sglang'; const FUSION_GFX = 'gfx942';
const FUSION_BUDGET_MS = %(budget_ms)d; const FUSION_STACK_RESERVE_FRAC = 0.3;
const FUSION_ITEM_MINUTES = %(item_min)d; const FUSION_POOL_MAX = 8;
const FUSION_PRIOR_WRITEBACK_ON = true;
const FUSION_MIN_LANDED = %(min_landed)d; const FUSION_FALLBACK_CHECKPOINT_FRAC = 0.5;
const fusionItemDurationsMin = [];
// Scripted KernelFusion clock: every serving call advances it by SCEN.call_ms (default 0).
const fusionClock = {timed: true, t: 0, elapsed() { return this.t; }, charge() {}};
let fusionCandidatePool = %(pool)s; let fusionBoardK = null; let fusionBoardNoiseBand = null;
const STACK_FS_WRITES = [];
function runScript(prompt) {
  // Mirrors the script-runner agent: optional "write JSON to path", then the one command.
  const w = prompt.match(/to (\S+) \(python3 json\.dump/);
  if (w) {
    const doc = prompt.match(/```json\n([\s\S]*)\n```/)[1];
    fs.mkdirSync(pathm.dirname(w[1]), {recursive: true});
    fs.writeFileSync(w[1], doc);
    STACK_FS_WRITES.push(w[1]);
  }
  let cmd = prompt.match(/\n\n    (.+)\n\n/)[1].replace(/^python3 /, PY + ' ');
  if (/fusion_prior_writeback\.py/.test(cmd)) cmd += ' --dry-run';
  let out;
  try { out = cp.execSync(cmd, {encoding: 'utf8'}); }
  catch (e) { out = String(e.stdout || ''); }
  const lines = out.trim().split('\n');
  return JSON.parse(lines[lines.length - 1]);
}
const SERVING_GPU = '0,1,2,3'; const SERVING_TP = 4; const GPU_LIST = ['0','1','2','3'];
const MODEL_PATH = '/m'; const MODEL_NAME = 'm'; const WORKLOAD = {};
const FUSION_RUNTIME_INPUTS = {}; const ACCURACY_INPUTS = SCEN.accuracy_inputs || {};
const FUSION_LEASE_WAIT_S = 540; const FUSION_LEASE_WAIT_ROUNDS = 2;
const FUSION_UNITSIDE_ON = %(unit_on)s; const FUSION_ACCURACY_CALIBRATION_ON = %(cal_on)s;
const FUSION_UNIT_ALL_RUNGS = %(all_rungs)s; const FUSION_ACCURACY_CACHE_ON = %(acc_cache)s;
const FUSION_APPLY_RETRIES = %(retries)d; const FUSION_AB_MAX_PAIRS = 3;
const FUSION_GSM8K_THINKING = false; const FUSION_APPLY_TIMEOUT_MS = 0;
const NOISE_BAND = %(noise)s; const BASELINE_TPUT = 265;
const FUSION_UNIT_SCHEMA = {}, FUSION_UNIT_AGG_SCHEMA = {}, FUSION_LEASE_SCHEMA = {},
  FUSION_ACCURACY_CAL_SCHEMA = {}, FUSION_APPLY_SCHEMA = {}, FUSION_APPLYBACK_GATE_SCHEMA = {},
  FUSION_STACK_SCHEMA = {}, FUSION_STACK_STEP_SCHEMA = {}, FUSION_WRITEBACK_SCHEMA = {},
  FUSION_ACCURACY_CACHE_SCHEMA = {};
let FUSION_INPUTS = {FUSION_TOPK_JSON: '/eval/topk.json', FUSION_CANDIDATES_JSON: '/eval/c.json',
  FUSION_VALIDATION_JSON: '', FUSION_UNITSIDE_JSON: ''};
let fusionExecutionList = %(entries)s;
let semantics = {semantic_table_json: '/eval/table.json'};
let curOverlay = '', curFlags = '--x', curEnv = '', curTput = 265;
const acceptedFusions = [];
const fusionAcceptedAtEntry = %(acc_at_entry)d;
const fusionEntryTput = 265;
let fusionAccuracyRef = null, fusionAccuracyCalibration = null, fusionUnitStatus = null;
const fusionRecoveryState = {entryOverlay: '', entryFlags: '--x', entryEnv: '', overlay: '',
  flags: '--x', env: '', throughput: 265, acceptedFusionCount: 0, inputs: {}, applyResult: null};
let fusionFailureStage = '';
function fusionStackKey() { return JSON.stringify({f: acceptedFusions.map(r => r.exec_id).sort(), o: curOverlay}); }
function fusionGsm8kConcurrency() { return 4; }
function makeSem(ids) {
  const free = ids.slice(); const waiters = [];
  const pump = () => { while (waiters.length && waiters[0].n <= free.length) {
    const w = waiters.shift(); w.resolve(free.splice(0, w.n)); } };
  return { acquire(n = 1) { if (n <= free.length) return Promise.resolve(free.splice(0, n));
      return new Promise((resolve) => { waiters.push({ n, resolve }); }); },
    release(got) { free.push(...got); pump(); },
    async with(n, fn) { const g = await this.acquire(n); try { return await fn(g); } finally { this.release(g); } } };
}
async function parallel(thunks) { return Promise.all(thunks.map((t) => t())); }
function roleAgent(role, phase, intro, inputs) { return {role, phase, intro, inputs}; }
const queues = JSON.parse(JSON.stringify(SCEN));
function nextOf(key, dflt) { const q = queues[key]; return (q && q.length) ? q.shift() : dflt; }
async function safeAgent(prompt, opts) {
  const label = opts.label;
  calls.push({label, inputs: prompt && prompt.inputs ? prompt.inputs : null,
              intro: prompt && prompt.intro ? prompt.intro : (typeof prompt === 'string' ? prompt : '')});
  if (label.startsWith('fusion-stack:') || label === 'fusion-prior-writeback' ||
      label.startsWith('fusion-accuracy-cache:')) {
    const res = runScript(prompt);
    calls[calls.length - 1].result = res;
    return res;
  }
  if (label.startsWith('fusion_integrator:apply_stack_step:')) {
    fusionClock.t += (SCEN.call_ms || 0);
    const id = label.split(':')[2];
    const r = nextOf(label, {measured: true, delta_pct: 0.2, accuracy_pass: true,
                             stack_overlay: '/ov/stack_' + id});
    return r === 'null' ? null : r;
  }
  if (label === 'fusion_integrator:apply_stack_confirm') {
    fusionClock.t += (SCEN.call_ms || 0);
    const r = nextOf(label, {measured: true, ab_decision: 'accept', delta_pct: 2.5,
                             accuracy_pass: true, cand_tok_s: [272, 273]});
    return r === 'null' ? null : r;
  }
  if (label.startsWith('fusion-lease:')) {
    const op = label.split(':')[1];
    return nextOf(label, nextOf('lease:' + op, {ok: true, alive_count: 0, busy_holders: [], signalled: []}));
  }
  if (label === 'fusion_unit_validator:aggregate') {
    return {status: 'pass', fusion_unitside_json: '/eval/fusion/fusion_unitside.json',
            unit_status_by_candidate: SCEN.unit_status || {}};
  }
  if (label.startsWith('fusion_unit_validator:')) {
    const r = nextOf(label, {candidate_id: label.split(':')[2], verdict_path: '/v'});
    return r === 'null' ? null : r;
  }
  if (label === 'fusion_integrator:calibrate_accuracy') {
    const r = nextOf(label, {calibration_json: '/eval/fusion/accuracy_calibration/calibration.json',
            mode: 'deterministic', gsm8k_usable: false, deterministic_usable: true,
            base_reference: {exact_match: 0.555, n: 200, path: '/a.json'}});
    if (r === 'null') return null;
    if (r && r.calibration_json) r.calibration_json = r.calibration_json.replace('$EVAL', EVAL_DIR);
    if (r && r.write_calibration) {
      // Scripted calibration that leaves a real calibration.json behind (for cache-store).
      fs.mkdirSync(pathm.dirname(r.calibration_json), {recursive: true});
      fs.writeFileSync(r.calibration_json, JSON.stringify(r.write_calibration));
    }
    return r;
  }
  if (label.startsWith('fusion_integrator:apply_one:')) {
    fusionClock.t += (SCEN.call_ms || 0);
    const prior = prompt.inputs.PRIOR_APPLY_RESULT;
    const r = nextOf(label, null);
    if (r === null || r === 'null') return null;
    return Object.assign({}, prior, {
      accepted_fusions: (prior.accepted_fusions || []).concat(r.accepted || []),
      rejected: (prior.rejected || []).concat(r.rejected || []),
      deferred: (prior.deferred || []).concat(r.deferred || []),
      retry: r.retry || [], final_overlay: r.final_overlay || prior.final_overlay,
      e2e_throughput_tok_s: r.tput || prior.e2e_throughput_tok_s});
  }
  if (label === 'fusion-applyback-gate') return {status: 'fail', coverage: {}};
  return null;
}
%(helpers)s
(async () => {
  try {
%(block)s
    }
    console.log(JSON.stringify({fapply, labels: calls.map(c => c.label),
      calls, logs, curOverlay, curTput, acceptedFusions, fusionApplyUnprocessed,
      unitside: FUSION_INPUTS.FUSION_UNITSIDE_JSON, fusionUnitStatus,
      fusionBudgetDeferred, fusionPriorWriteback, fusionAccuracyMode, fusionAccuracyCalibration}));
  } catch (e) { console.log(JSON.stringify({error: String(e && e.stack || e)})); }
})();
""" % {"scen": json.dumps(scenario), "entries": json.dumps(entries),
       "unit_on": "true" if unitside_on else "false",
       "cal_on": "true" if calibration_on else "false", "retries": retries,
       "acc_at_entry": accepted_at_entry, "helpers": helpers, "block": block,
       "wfdir": json.dumps(WORKFLOW), "evaldir": json.dumps(evaldir),
       "py": json.dumps(sys.executable), "budget_ms": budget_s * 1000, "item_min": item_min,
       "pool": json.dumps(pool or []), "noise": json.dumps(noise), "min_landed": min_landed,
       "all_rungs": "true" if unit_all_rungs else "false",
       "acc_cache": "true" if accuracy_cache else "false"}
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(script)
            path = fh.name
        try:
            out = subprocess.run([NODE, path], capture_output=True, text=True, timeout=60)
        finally:
            os.unlink(path)
        self.assertEqual(out.returncode, 0, out.stderr)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertNotIn("error", res, res.get("error"))
        if return_root:
            return res, root
        return res

    ENTRIES = [
        {"exec_id": "e02", "candidate_ids": ["c02"], "op_tags": ["quant"], "e2e_share": 0.05,
         "estimated_speedup": 1.5},
        {"exec_id": "e03", "candidate_ids": ["c05"], "op_tags": ["norm", "quant"],
         "e2e_share": 0.05, "estimated_speedup": 1.5},
        {"exec_id": "e08", "candidate_ids": ["c40"], "op_tags": ["allreduce", "quant"],
         "kernel_names": ["cross_device_reduce_1stage"], "e2e_share": 0.01,
         "estimated_speedup": 1.3},
    ]

    def test_qwen35_replay_unmeasured_entries_are_retried_and_stacked(self):
        scen = {
            "unit_status": {"c02": "pass", "c05": "pass", "c40": "pass"},
            "fusion_integrator:apply_one:e02": [
                {"rejected": [{"exec_id": "e02", "measured": True,
                               "blocked_evidence": "e2e_no_gain", "reason": "A/B -0.01%"}]}],
            # e03: the A/B was still running at return -> retry, then measured accept.
            "fusion_integrator:apply_one:e03": [
                {"retry": [{"exec_id": "e03", "reason_code": "not_finished",
                            "reason": "A/B on its first ref leg"}]},
                {"accepted": [{"exec_id": "e03", "fusion": "gdn-outnorm-quant",
                               "overlay_path": "/ov/e03"}], "final_overlay": "/ov/combined",
                 "tput": 270}],
            # e08: lease busy once, then small-gain stack_pending.
            "fusion-lease:acquire:apply_e08_try1": [
                {"ok": False, "busy_holders": [], "serving_lock_held": True,
                 "reason": "gpu_busy: serving lock held outside the registry"},
                {"ok": False, "busy_holders": [], "serving_lock_held": True,
                 "reason": "gpu_busy"}],
            "fusion_integrator:apply_one:e08": [
                {"rejected": [{"exec_id": "e08", "measured": True, "stack_pending": True,
                               "overlay_path": "/ov/e08", "reason": "in noise"}]}],
        }
        res = self._run(scen, self.ENTRIES)
        labels = res["labels"]
        # Phase 3.0 ran for every candidate, aggregate before any apply-back.
        self.assertIn("fusion_unit_validator:e02:c02", labels)
        self.assertIn("fusion_unit_validator:e08:c40", labels)
        self.assertLess(labels.index("fusion_unit_validator:aggregate"),
                        labels.index("fusion_integrator:apply_one:e02"))
        self.assertEqual(res["unitside"], "/eval/fusion/fusion_unitside.json")
        # Calibration ran under its own lease before the first apply_one.
        self.assertLess(labels.index("fusion_integrator:calibrate_accuracy"),
                        labels.index("fusion_integrator:apply_one:e02"))
        # Every apply_one was preceded by an acquire and followed by a wait (settle).
        for i, lab in enumerate(labels):
            if lab.startswith("fusion_integrator:apply_one:"):
                self.assertTrue(labels[i - 1].startswith("fusion-lease:acquire:"), labels[i - 1])
                self.assertTrue(labels[i + 1].startswith("fusion-lease:wait:"), labels[i + 1])
        self.assertEqual(labels.count("fusion_integrator:apply_one:e03"), 2)
        self.assertEqual(labels.count("fusion_integrator:apply_one:e08"), 1)
        fapply = res["fapply"]
        self.assertEqual(sorted(r["exec_id"] for r in fapply["accepted_fusions"]), ["e03", "e08"])
        self.assertEqual([r["exec_id"] for r in fapply["rejected"]], ["e02"])
        self.assertEqual(fapply["retry"], [])
        self.assertEqual(fapply["unmeasured"], [])
        # The single stack_pending entry went through greedy stacking on top of the accepted
        # e03 (one marginal A/B vs the current stack), then one confirmation vs the entry stack.
        step = next(c for c in res["calls"]
                    if c["label"] == "fusion_integrator:apply_stack_step:e08")
        self.assertEqual(step["inputs"]["CURRENT_STACK"]["members"], ["e03"])
        self.assertEqual(step["inputs"]["CURRENT_STACK"]["overlay"], "/ov/combined")
        self.assertEqual(step["inputs"]["KEEP_THRESHOLD_PCT"], -0.5)
        self.assertTrue(labels[labels.index("fusion_integrator:apply_stack_step:e08") - 1]
                        .startswith("fusion-lease:acquire:stack_e08"))
        self.assertIn("fusion_integrator:apply_stack_confirm", labels)
        self.assertEqual(res["curOverlay"], "/ov/stack_e08")
        e08 = next(r for r in fapply["accepted_fusions"] if r["exec_id"] == "e08")
        self.assertTrue(e08["stacked"])
        self.assertEqual(fapply["stack_greedy"]["kept"], ["e08"])
        self.assertTrue(fapply["stack_greedy"]["landed"])
        self.assertIn("fusion-applyback-gate", labels)
        # The A/B prompt carries the lease and forbids backgrounding.
        e03 = next(c for c in res["calls"] if c["label"] == "fusion_integrator:apply_one:e03")
        self.assertEqual(e03["inputs"]["GPU_LEASE_ID"], "apply_e03_try1")
        self.assertIn("nohup", e03["intro"])

    def test_small_gain_entries_are_told_to_route_to_the_stack(self):
        res = self._run({"unit_status": {}}, self.ENTRIES[2:])
        call = next(c for c in res["calls"] if c["label"] == "fusion_integrator:apply_one:e08")
        self.assertTrue(call["inputs"]["SMALL_GAIN_ROUTE"])
        self.assertLess(call["inputs"]["EXPECTED_GAIN_PCT"], 1.0)
        self.assertIn("--expected-gain-pct", call["intro"])

    def test_exhausted_retries_are_unmeasured_never_blocked(self):
        scen = {"fusion_integrator:apply_one:e02": ["null", "null", "null"]}
        res = self._run(scen, self.ENTRIES[:1], retries=2)
        self.assertEqual(res["labels"].count("fusion_integrator:apply_one:e02"), 3)
        fapply = res["fapply"]
        self.assertEqual(fapply["rejected"], [])
        self.assertEqual([r["exec_id"] for r in fapply["unmeasured"]], ["e02"])
        self.assertEqual(res["fusionApplyUnprocessed"], ["e02"])

    def test_an_unmeasured_rejected_row_is_moved_to_the_retry_queue(self):
        scen = {"fusion_integrator:apply_one:e02": [
            {"rejected": [{"exec_id": "e02", "measured": False,
                           "reason": "GPUs held by an orphan"}]},
            {"rejected": [{"exec_id": "e02", "measured": True, "blocked_evidence": "unit_fail",
                           "reason": "0.8x"}]}]}
        res = self._run(scen, self.ENTRIES[:1])
        self.assertEqual(res["labels"].count("fusion_integrator:apply_one:e02"), 2)
        self.assertEqual([r["measured"] for r in res["fapply"]["rejected"]], [True])

    def test_a_measured_unit_loss_skips_the_serving_ab(self):
        res = self._run({"unit_status": {"c02": "fail"}}, self.ENTRIES[:1])
        self.assertNotIn("fusion_integrator:apply_one:e02", res["labels"])
        row = res["fapply"]["rejected"][0]
        self.assertEqual(row["blocked_evidence"], "unit_fail")
        self.assertTrue(row["measured"])

    def test_collective_unit_tests_take_the_serving_lease_single_ones_do_not(self):
        res = self._run({"unit_status": {}}, self.ENTRIES)
        labels = res["labels"]
        self.assertIn("fusion-lease:acquire:unit_c40", labels)
        self.assertNotIn("fusion-lease:acquire:unit_c02", labels)
        single = next(c for c in res["calls"] if c["label"] == "fusion_unit_validator:e02:c02")
        self.assertEqual(len(single["inputs"]["GPU_IDS"].split(",")), 1)
        coll = next(c for c in res["calls"] if c["label"] == "fusion_unit_validator:e08:c40")
        self.assertEqual(coll["inputs"]["GPU_IDS"], "0,1,2,3")
        # the single-GPU wave is settled (barrier) before the collective acquire
        self.assertLess(labels.index("fusion-lease:wait:*"),
                        labels.index("fusion-lease:acquire:unit_c40"))

    def test_unitside_can_be_skipped_only_explicitly(self):
        res = self._run({}, self.ENTRIES[:1], unitside_on=False)
        self.assertFalse(any(l.startswith("fusion_unit_validator") for l in res["labels"]))
        self.assertTrue(any("SKIPPED by explicit args.fusion_unitside=false" in l for l in res["logs"]))

    def test_a_stale_workflow_holder_is_reaped_before_the_next_call(self):
        scen = {
            "fusion-lease:acquire:apply_e02_try1": [
                {"ok": False, "busy_holders": [{"lease_id": "apply_e01_try1", "alive_count": 1}],
                 "reason": "gpu_busy: apply_e01_try1"},
                {"ok": True}],
            "fusion-lease:wait:apply_e01_try1": [{"ok": True, "alive_count": 1},
                                                 {"ok": True, "alive_count": 1}],
            "fusion_integrator:apply_one:e02": [
                {"rejected": [{"exec_id": "e02", "measured": True,
                               "blocked_evidence": "e2e_regression", "reason": "-2%"}]}]}
        res = self._run(scen, self.ENTRIES[:1], unitside_on=False, calibration_on=False)
        labels = res["labels"]
        self.assertIn("fusion-lease:reap:apply_e01_try1", labels)
        self.assertLess(labels.index("fusion-lease:reap:apply_e01_try1"),
                        labels.index("fusion_integrator:apply_one:e02"))

    # ---- GEAK_structure 二.10: ladder, ordering, greedy stacking, budget, write-back ----
    def test_apply_back_runs_in_expected_gain_order(self):
        entries = [
            {"exec_id": "e01", "candidate_ids": ["c1"], "expected_e2e_gain_pct": 1.2},
            {"exec_id": "e02", "candidate_ids": ["c2"], "expected_e2e_gain_pct": 3.0},
            {"exec_id": "e03", "candidate_ids": ["c3"]},
            {"exec_id": "e04", "candidate_ids": ["c4"], "expected_e2e_gain_pct": 2.0},
        ]
        scen = {"fusion_integrator:apply_one:%s" % e: [
            {"rejected": [{"exec_id": e, "measured": True, "blocked_evidence": "e2e_no_gain",
                           "reason": "flat"}]}] for e in ("e01", "e02", "e03", "e04")}
        res = self._run(scen, entries, unitside_on=False, calibration_on=False)
        order = [l.split(":")[2] for l in res["labels"]
                 if l.startswith("fusion_integrator:apply_one:")]
        self.assertEqual(order, ["e02", "e04", "e01", "e03"])

    LADDER = [{"exec_id": "e05", "alt_group": "g1", "candidate_ids": ["c50"],
               "expected_e2e_gain_pct": 2.0,
               "alt_ladder": [
                   {"rung": 0, "kind": "epilogue_forward", "seam_key": "ef:ar_quant",
                    "expected_e2e_gain_pct": 2.0},
                   {"rung": 1, "kind": "serial", "seam_key": "s:quant_get_quant",
                    "kernel_names": ["per_group_quant"], "row_ids": ["r9"],
                    "expected_e2e_gain_pct": 1.5},
                   {"rung": 2, "kind": "seam_rewrite", "seam_key": "sr:moe_in",
                    "expected_e2e_gain_pct": 1.1}]}]

    def test_alt_group_unit_tests_every_rung_up_front_when_opted_in_and_walks_the_ladder(self):
        scen = {
            "unit_status": {"c50": "fail", "e05.r1": "pass", "e05.r2": "pass"},
            "fusion_integrator:apply_one:e05:r1": [
                {"rejected": [{"exec_id": "e05", "measured": True,
                               "blocked_evidence": "e2e_regression",
                               "throughput_delta_pct": -1.8, "reason": "-1.8%"}]}],
            "fusion_integrator:apply_one:e05:r2": [
                {"accepted": [{"exec_id": "e05", "fusion": "moe-in-rewrite",
                               "throughput_delta_pct": 1.3, "overlay_path": "/ov/e05r2"}],
                 "final_overlay": "/ov/e05r2", "tput": 268}],
        }
        res = self._run(scen, self.LADDER, unit_all_rungs=True)
        labels = res["labels"]
        # fusion_unit_all_rungs=true: every rung was unit-tested up front, as its own candidate.
        for cid in ("c50", "e05.r1", "e05.r2"):
            self.assertIn("fusion_unit_validator:e05:%s" % cid, labels)
        rung1 = next(c for c in res["calls"] if c["label"] == "fusion_unit_validator:e05:e05.r1")
        self.assertEqual(rung1["inputs"]["UNIT_JOB_KIND"], "alt")
        self.assertEqual(rung1["inputs"]["TARGET_EXECUTION"]["seam_key"], "s:quant_get_quant")
        # 10.2: rung 0 (unit_fail) costs no server; rung 1 is measured and rejected; rung 2 lands.
        self.assertNotIn("fusion_integrator:apply_one:e05", labels)
        self.assertEqual(labels.count("fusion_integrator:apply_one:e05:r1"), 1)
        self.assertEqual(labels.count("fusion_integrator:apply_one:e05:r2"), 1)
        call = next(c for c in res["calls"] if c["label"] == "fusion_integrator:apply_one:e05:r2")
        self.assertEqual(call["inputs"]["TARGET_RUNG"], 2)
        self.assertEqual([a["rung"] for a in call["inputs"]["RUNG_ATTEMPTS"]], [0, 1])
        fapply = res["fapply"]
        self.assertEqual([(r["exec_id"], r["rung"]) for r in fapply["accepted_fusions"]],
                         [("e05", 2)])
        self.assertEqual(fapply["rejected"], [])
        self.assertEqual([(a["rung"], a["blocked_evidence"]) for a in fapply["rung_attempts"]],
                         [(0, "unit_fail"), (1, "e2e_regression")])
        # The write-back sees the rung-1 regression as a negative for THAT seam.
        wb = next(c for c in res["calls"] if c["label"] == "fusion-prior-writeback")["result"]
        neg = [p for p in wb["planned"]["priors"] if p["polarity"] == "negative"]
        self.assertEqual([p["measurement"]["gain_pct"] for p in neg], [-1.8])

    # ---- 二.10.1: deeper alternative rungs are unit-tested on demand ----
    def _unit_labels(self, labels):
        return [l for l in labels if l.startswith("fusion_unit_validator:") and not l.endswith(":aggregate")]

    def test_only_rung_zero_is_unit_tested_up_front(self):
        scen = {"unit_status": {"c50": "pass"},
                "fusion_integrator:apply_one:e05": [
                    {"accepted": [{"exec_id": "e05", "fusion": "ef", "overlay_path": "/ov/e05"}],
                     "final_overlay": "/ov/e05", "tput": 270}]}
        res = self._run(scen, self.LADDER)
        labels = res["labels"]
        # Rung 0 lands: rungs 1 and 2 are never unit-tested.
        self.assertEqual(self._unit_labels(labels), ["fusion_unit_validator:e05:c50"])
        self.assertTrue(any("2 deeper alternative rung(s) unit-tested on demand" in l for l in res["logs"]))
        self.assertEqual(res["fapply"]["unit_on_demand"], [])
        self.assertEqual([r["exec_id"] for r in res["fapply"]["accepted_fusions"]], ["e05"])

    def test_rungs_are_unit_tested_lazily_after_a_unit_fail(self):
        scen = {"unit_status": {"c50": "fail"},
                "fusion_unit_validator:e05:e05.r1": [{"candidate_id": "e05.r1", "parity": "pass",
                                                       "isolated_speedup": 0.8, "verdict_path": "/v/r1"}],
                "fusion_unit_validator:e05:e05.r2": [{"candidate_id": "e05.r2", "parity": "pass",
                                                       "isolated_speedup": 1.4}],
                "fusion_integrator:apply_one:e05:r2": [
                    {"accepted": [{"exec_id": "e05", "fusion": "moe-in", "overlay_path": "/ov/r2"}],
                     "final_overlay": "/ov/r2", "tput": 268}]}
        res = self._run(scen, self.LADDER)
        labels = res["labels"]
        agg = labels.index("fusion_unit_validator:aggregate")
        r1 = labels.index("fusion_unit_validator:e05:e05.r1")
        r2 = labels.index("fusion_unit_validator:e05:e05.r2")
        # Not part of the Phase 3.0 wave; r1 tested only once rung 0 was unit_fail, r2 only
        # after r1's on-demand unit test measured a loss; neither rung 0 nor r1 got a server.
        self.assertLess(agg, r1)
        self.assertLess(r1, r2)
        self.assertLess(r2, labels.index("fusion_integrator:apply_one:e05:r2"))
        self.assertNotIn("fusion_integrator:apply_one:e05", labels)
        self.assertNotIn("fusion_integrator:apply_one:e05:r1", labels)
        call = next(c for c in res["calls"] if c["label"] == "fusion_unit_validator:e05:e05.r1")
        self.assertEqual(call["inputs"]["UNIT_JOB_KIND"], "alt")
        self.assertEqual(call["inputs"]["GPU_IDS"], "0")
        self.assertEqual(call["inputs"]["GPU_LEASE_ID"], "unit_e05.r1")
        self.assertIn("on demand during apply-back", call["intro"])
        # Single-card on-demand test: own registry settled right after, no serving acquire.
        self.assertEqual(labels[r1 + 1], "fusion-lease:wait:unit_e05.r1")
        self.assertNotIn("fusion-lease:acquire:unit_e05.r1", labels)
        self.assertEqual(res["fusionUnitStatus"]["e05.r1"], "fail")
        self.assertEqual(res["fusionUnitStatus"]["e05.r2"], "pass")
        self.assertEqual([(u["candidate_id"], u["status"]) for u in res["fapply"]["unit_on_demand"]],
                         [("e05.r1", "fail"), ("e05.r2", "pass")])
        self.assertEqual([(a["rung"], a["blocked_evidence"]) for a in res["fapply"]["rung_attempts"]],
                         [(0, "unit_fail"), (1, "unit_fail")])
        self.assertEqual([(r["exec_id"], r["rung"]) for r in res["fapply"]["accepted_fusions"]],
                         [("e05", 2)])

    def test_rung_one_is_unit_tested_only_after_a_measured_rejection(self):
        scen = {"unit_status": {"c50": "pass"},
                "fusion_integrator:apply_one:e05": [
                    {"rejected": [{"exec_id": "e05", "measured": True,
                                   "blocked_evidence": "e2e_regression", "reason": "-1%"}]}],
                "fusion_unit_validator:e05:e05.r1": [{"candidate_id": "e05.r1", "parity": "pass",
                                                       "isolated_speedup": 1.2}],
                "fusion_integrator:apply_one:e05:r1": [
                    {"accepted": [{"exec_id": "e05", "fusion": "q", "overlay_path": "/ov/r1"}],
                     "final_overlay": "/ov/r1", "tput": 268}]}
        res = self._run(scen, self.LADDER)
        labels = res["labels"]
        self.assertLess(labels.index("fusion_integrator:apply_one:e05"),
                        labels.index("fusion_unit_validator:e05:e05.r1"))
        self.assertLess(labels.index("fusion_unit_validator:e05:e05.r1"),
                        labels.index("fusion_integrator:apply_one:e05:r1"))
        self.assertNotIn("fusion_unit_validator:e05:e05.r2", labels)
        call = next(c for c in res["calls"] if c["label"] == "fusion_integrator:apply_one:e05:r1")
        self.assertEqual(call["inputs"]["UNIT_SIDE"]["statuses"], ["pass"])
        self.assertTrue(call["inputs"]["UNIT_SIDE"]["eligible"])

    def test_a_lazy_rung_without_a_verdict_is_not_validated_and_still_applied(self):
        scen = {"unit_status": {"c50": "fail"},
                "fusion_unit_validator:e05:e05.r1": ["null"],
                "fusion_integrator:apply_one:e05:r1": [
                    {"accepted": [{"exec_id": "e05", "fusion": "q", "overlay_path": "/ov/r1"}],
                     "final_overlay": "/ov/r1", "tput": 268}]}
        res = self._run(scen, self.LADDER)
        self.assertEqual(res["fusionUnitStatus"]["e05.r1"], "not_validated")
        self.assertEqual(res["labels"].count("fusion_unit_validator:e05:e05.r1"), 1)
        self.assertIn("fusion_integrator:apply_one:e05:r1", res["labels"])

    def test_a_collective_lazy_rung_takes_the_serving_lease(self):
        ladder = json.loads(json.dumps(self.LADDER))
        ladder[0]["op_tags"] = ["allreduce", "quant"]
        scen = {"unit_status": {"c50": "fail"},
                "fusion_unit_validator:e05:e05.r1": [{"candidate_id": "e05.r1", "parity": "pass",
                                                       "isolated_speedup": 1.2}],
                "fusion_integrator:apply_one:e05:r1": [
                    {"accepted": [{"exec_id": "e05", "fusion": "q"}], "tput": 268}]}
        res = self._run(scen, ladder)
        labels = res["labels"]
        i = labels.index("fusion_unit_validator:e05:e05.r1")
        self.assertEqual(labels[i - 1], "fusion-lease:acquire:unit_e05.r1")
        call = next(c for c in res["calls"] if c["label"] == "fusion_unit_validator:e05:e05.r1")
        self.assertEqual(call["inputs"]["GPU_IDS"], "0,1,2,3")

    # ---- 二.10.3: unusable calibration -> no e2e accuracy test; calibration cache ----
    UNUSABLE_CAL = {"calibration_json": "$EVAL/fusion/accuracy_calibration/calibration.json",
                    "mode": "unit_equivalence_only", "gsm8k_usable": False,
                    "deterministic_usable": False, "accuracy_can_reject": False,
                    "write_calibration": {"schema": "fusion_accuracy_calibration/1",
                                          "mode": "unit_equivalence_only",
                                          "accuracy_can_reject": False,
                                          "gsm8k": {"usable": False, "measured": True},
                                          "deterministic": {"usable": False, "measured": True}}}

    def test_unusable_calibration_skips_e2e_accuracy_for_every_call(self):
        scen = {"accuracy_inputs": {"ACCURACY_GATE": "gsm8k",
                                    "GSM8K_EVAL_SCRIPT": "/wf/scripts/gsm8k_eval.py"},
                "fusion_integrator:calibrate_accuracy": [self.UNUSABLE_CAL],
                "fusion_integrator:apply_one:e08": [
                    {"rejected": [{"exec_id": "e08", "measured": True, "stack_pending": True,
                                   "overlay_path": "/ov/e08", "reason": "in noise"}]}],
                # A stacking step that ran no accuracy test and returned no accuracy_pass.
                "fusion_integrator:apply_stack_step:e08": [
                    {"measured": True, "delta_pct": 0.3, "stack_overlay": "/ov/stack_e08"}]}
        res = self._run(scen, self.ENTRIES[2:])
        self.assertEqual(res["fusionAccuracyMode"], "skip")
        self.assertTrue(any("ACCURACY_MODE=skip" in l for l in res["logs"]))
        calls = {c["label"]: c for c in res["calls"]}
        for label in ("fusion_integrator:apply_one:e08", "fusion_integrator:apply_stack_step:e08",
                      "fusion_integrator:apply_stack_confirm"):
            self.assertEqual(calls[label]["inputs"]["ACCURACY_MODE"], "skip", label)
            self.assertEqual(calls[label]["inputs"]["GSM8K_EVAL_SCRIPT"], "", label)
            self.assertIn("ACCURACY_MODE=skip", calls[label]["intro"], label)
            self.assertTrue(calls[label]["inputs"]["ACCURACY_CALIBRATION_JSON"].endswith("calibration.json"))
        # The step without accuracy_pass was kept on unit equivalence.
        self.assertEqual(res["fapply"]["stack_greedy"]["kept"], ["e08"])
        gate = calls["fusion-applyback-gate"]["intro"]
        self.assertIn("--accuracy-mode skip --calibration-json ", gate)

    def test_usable_calibration_keeps_the_accuracy_gate(self):
        scen = {"accuracy_inputs": {"ACCURACY_GATE": "gsm8k",
                                    "GSM8K_EVAL_SCRIPT": "/wf/scripts/gsm8k_eval.py"},
                "fusion_integrator:apply_one:e02": [
                    {"accepted": [{"exec_id": "e02", "fusion": "q"}], "tput": 270}]}
        res = self._run(scen, self.ENTRIES[:1])
        self.assertEqual(res["fusionAccuracyMode"], "gate")
        call = next(c for c in res["calls"] if c["label"] == "fusion_integrator:apply_one:e02")
        self.assertEqual(call["inputs"]["ACCURACY_MODE"], "gate")
        self.assertEqual(call["inputs"]["GSM8K_EVAL_SCRIPT"], "/wf/scripts/gsm8k_eval.py")
        self.assertNotIn("ACCURACY_MODE=skip", call["intro"])
        gate = next(c for c in res["calls"] if c["label"] == "fusion-applyback-gate")["intro"]
        self.assertNotIn("--accuracy-mode", gate)

    def test_calibration_that_did_not_return_keeps_the_gate(self):
        res = self._run({"fusion_integrator:calibrate_accuracy": ["null"]}, self.ENTRIES[:1])
        self.assertEqual(res["fusionAccuracyMode"], "gate")
        self.assertNotIn("fusion-accuracy-cache:store", res["labels"])

    def test_calibration_cache_miss_stores_and_hit_skips_the_calibration_agent(self):
        scen = {"fusion_integrator:calibrate_accuracy": [self.UNUSABLE_CAL],
                "fusion_integrator:apply_one:e02": [
                    {"accepted": [{"exec_id": "e02", "fusion": "q"}], "tput": 270}]}
        first, root = self._run(scen, self.ENTRIES[:1], return_root=True)
        labels = first["labels"]
        self.assertLess(labels.index("fusion-accuracy-cache:lookup"),
                        labels.index("fusion_integrator:calibrate_accuracy"))
        self.assertLess(labels.index("fusion_integrator:calibrate_accuracy"),
                        labels.index("fusion-accuracy-cache:store"))
        lookup = next(c for c in first["calls"] if c["label"] == "fusion-accuracy-cache:lookup")
        self.assertFalse(lookup["result"]["hit"])
        store = next(c for c in first["calls"] if c["label"] == "fusion-accuracy-cache:store")
        self.assertTrue(store["result"]["stored"], store["result"])
        cache_dir = os.path.join(root, ".fusion_accuracy_cache")
        self.assertEqual(len(os.listdir(cache_dir)), 1)

        # Next round, same model / serving config / image: the cached calibration is reused.
        second = self._run({"fusion_integrator:apply_one:e02": [
            {"accepted": [{"exec_id": "e02", "fusion": "q"}], "tput": 270}]},
            self.ENTRIES[:1], root=root)
        self.assertNotIn("fusion_integrator:calibrate_accuracy", second["labels"])
        self.assertNotIn("fusion-acquire:accuracy_calibration", second["labels"])
        self.assertNotIn("fusion-accuracy-cache:store", second["labels"])
        cal = second["fusionAccuracyCalibration"]
        self.assertTrue(cal["cached"])
        self.assertTrue(cal["calibration_json"].startswith(cache_dir))
        self.assertEqual(second["fusionAccuracyMode"], "skip")
        self.assertTrue(any("cache HIT" in l for l in second["logs"]))

    def test_calibration_cache_can_be_turned_off(self):
        res = self._run({}, self.ENTRIES[:1], accuracy_cache=False)
        self.assertFalse(any(l.startswith("fusion-accuracy-cache:") for l in res["labels"]))
        self.assertIn("fusion_integrator:calibrate_accuracy", res["labels"])

    def test_an_all_rungs_unit_fail_blocks_the_group_without_a_server(self):
        scen = {"unit_status": {"c50": "fail", "e05.r1": "fail", "e05.r2": "equivalent_fail"}}
        res = self._run(scen, self.LADDER)
        self.assertFalse(any(l.startswith("fusion_integrator:apply_one:") for l in res["labels"]))
        row = res["fapply"]["rejected"][0]
        self.assertEqual(row["blocked_evidence"], "unit_fail")
        self.assertEqual(len(row["rung_attempts"]), 2)
        self.assertIn("every rung of alternative group g1", row["reason"])

    def test_a_stack_pending_rung_ends_the_group(self):
        scen = {"unit_status": {"c50": "pass"},
                "fusion_integrator:apply_one:e05": [
                    {"rejected": [{"exec_id": "e05", "measured": True, "stack_pending": True,
                                   "overlay_path": "/ov/e05", "reason": "in noise"}]}]}
        res = self._run(scen, self.LADDER)
        self.assertNotIn("fusion_integrator:apply_one:e05:r1", res["labels"])
        self.assertIn("fusion_integrator:apply_stack_step:e05", res["labels"])

    def test_greedy_stacking_keeps_drops_and_lands_after_confirmation(self):
        entries = [
            {"exec_id": "e10", "candidate_ids": ["c10"], "expected_e2e_gain_pct": 0.4},
            {"exec_id": "e11", "candidate_ids": ["c11"], "expected_e2e_gain_pct": 0.8},
        ]
        pool = [{"exec_id": "p01", "candidate_ids": ["cp1"], "expected_e2e_gain_pct": 0.2,
                 "kind": "serial", "seam_key": "s:pool"},
                {"exec_id": "p02", "candidate_ids": ["cp2"], "expected_e2e_gain_pct": 0.9}]
        pend = lambda e: [{"rejected": [{"exec_id": e, "measured": True, "stack_pending": True,
                                         "overlay_path": "/ov/" + e, "reason": "small gain"}]}]
        scen = {
            "unit_status": {"c10": "pass", "c11": "pass", "cp1": "pass", "cp2": "fail"},
            "fusion_integrator:apply_one:e10": pend("e10"),
            "fusion_integrator:apply_one:e11": pend("e11"),
            "fusion_integrator:apply_stack_step:e11": [
                {"measured": True, "delta_pct": -0.8, "accuracy_pass": True,
                 "stack_overlay": "/ov/stack_e11"}],
        }
        res = self._run(scen, entries, pool=pool)
        labels = res["labels"]
        # pool items were unit-tested; only the passing one is a stacking candidate.
        self.assertIn("fusion_unit_validator:p01:cp1", labels)
        steps = [l.split(":")[2] for l in labels if l.startswith("fusion_integrator:apply_stack_step:")]
        self.assertEqual(steps, ["e11", "e10", "p01"])
        # e11 regressed beyond -band/2 -> dropped; the next step stacks on the unchanged base.
        e10 = next(c for c in res["calls"] if c["label"] == "fusion_integrator:apply_stack_step:e10")
        self.assertEqual(e10["inputs"]["CURRENT_STACK"]["members"], [])
        p01 = next(c for c in res["calls"] if c["label"] == "fusion_integrator:apply_stack_step:p01")
        self.assertEqual(p01["inputs"]["CURRENT_STACK"]["members"], ["e10"])
        self.assertEqual(p01["inputs"]["CURRENT_STACK"]["overlay"], "/ov/stack_e10")
        conf = next(c for c in res["calls"] if c["label"] == "fusion_integrator:apply_stack_confirm")
        self.assertEqual(conf["inputs"]["FINAL_STACK"]["members"], ["e10", "p01"])
        self.assertEqual(conf["inputs"]["FUSION_ENTRY_FLAGS"], "--x")
        fapply = res["fapply"]
        self.assertEqual([r["exec_id"] for r in fapply["accepted_fusions"]], ["e10"])
        harm = next(r for r in fapply["rejected"] if r["exec_id"] == "e11")
        self.assertEqual(harm["blocked_evidence"], "stack_harmful")
        self.assertTrue(harm["stack_result"])
        self.assertEqual([p["exec_id"] for p in fapply["pool_accepted"]], ["p01"])
        self.assertEqual(sorted(r["exec_id"] for r in res["acceptedFusions"]), ["e10", "p01"])
        self.assertEqual(res["curOverlay"], "/ov/stack_p01")
        self.assertEqual(res["curTput"], 272.5)
        wb = next(c for c in res["calls"] if c["label"] == "fusion-prior-writeback")["result"]
        self.assertEqual(wb["counts"]["positive"], 2)     # e10 + p01 kept in the landed stack
        self.assertEqual(wb["counts"]["negative"], 1)     # e11 stack_harmful

    def test_a_failed_confirmation_lands_nothing(self):
        entries = [{"exec_id": "e10", "candidate_ids": ["c10"], "expected_e2e_gain_pct": 0.4}]
        scen = {"fusion_integrator:apply_one:e10": [
                    {"rejected": [{"exec_id": "e10", "measured": True, "stack_pending": True,
                                   "overlay_path": "/ov/e10", "reason": "small"}]}],
                "fusion_integrator:apply_stack_confirm": [
                    {"measured": True, "ab_decision": "reject", "delta_pct": 0.3,
                     "accuracy_pass": True}]}
        res = self._run(scen, entries)
        self.assertEqual(res["fapply"]["accepted_fusions"], [])
        self.assertEqual(res["curOverlay"], "")
        row = res["fapply"]["rejected"][0]
        self.assertEqual(row["blocked_evidence"], "stack_reject")

    def test_budget_reserve_stops_solo_apply_back_and_defers_with_unit_results(self):
        entries = [{"exec_id": "e%02d" % i, "candidate_ids": ["c%d" % i],
                    "expected_e2e_gain_pct": 5.0 - i} for i in range(1, 5)]
        scen = {"unit_status": {"c%d" % i: "pass" for i in range(1, 5)},
                "call_ms": 45 * 60000}
        for i in range(1, 5):
            scen["fusion_integrator:apply_one:e%02d" % i] = [
                {"rejected": [{"exec_id": "e%02d" % i, "measured": True,
                               "blocked_evidence": "e2e_no_gain", "reason": "flat"}]}]
        # 2h budget, 30% reserve = 36 min, 45 min per item: one solo item fits.
        res = self._run(scen, entries, budget_s=7200, item_min=45)
        applied = [l for l in res["labels"] if l.startswith("fusion_integrator:apply_one:")]
        self.assertEqual(applied, ["fusion_integrator:apply_one:e01"])
        self.assertEqual(res["fusionBudgetDeferred"], ["e02", "e03", "e04"])
        deferred = {r["exec_id"]: r for r in res["fapply"]["deferred"]}
        self.assertTrue(deferred["e02"]["deferred_budget"])
        self.assertEqual(deferred["e02"]["reason_code"], "budget_reserve")
        self.assertEqual(deferred["e02"]["unit_side"]["statuses"], ["pass"])
        wb = next(c for c in res["calls"] if c["label"] == "fusion-prior-writeback")["result"]
        self.assertEqual(wb["counts"]["untested_candidates"], 3)

    def test_writeback_runs_once_after_the_final_gate(self):
        res = self._run({"fusion_integrator:apply_one:e02": [
            {"accepted": [{"exec_id": "e02", "fusion": "q", "throughput_delta_pct": 2.0}],
             "final_overlay": "/ov/e02", "tput": 270}]}, self.ENTRIES[:1])
        labels = res["labels"]
        self.assertEqual(labels.count("fusion-prior-writeback"), 1)
        self.assertLess(labels.index("fusion-applyback-gate"), labels.index("fusion-prior-writeback"))
        call = next(c for c in res["calls"] if c["label"] == "fusion-prior-writeback")
        for needle in ("fusion_prior_writeback.py --apply ", "/fusion/apply_result_final.json",
                       "--topk /eval/topk.json", "--gfx gfx942", "--framework sglang",
                       "--noise-band-pct 1"):
            self.assertIn(needle, call["intro"])
        self.assertEqual(call["result"]["counts"]["positive"], 1)
        self.assertTrue(res["fusionPriorWriteback"]["ok"])

    # ---- GEAK_structure 二.10.6: landing guarantee ----
    def test_first_item_is_the_highest_confidence_then_confidence_weighted_gain(self):
        entries = [
            {"exec_id": "e01", "candidate_ids": ["c1"], "expected_e2e_gain_pct": 5.0},
            {"exec_id": "e02", "candidate_ids": ["c2"], "expected_e2e_gain_pct": 1.5,
             "prior_polarity": "positive"},
            {"exec_id": "e03", "candidate_ids": ["c3"], "expected_e2e_gain_pct": 3.0,
             "matched_existing_kernels": [{"name": "k"}]},
            {"exec_id": "e04", "candidate_ids": ["c4"], "expected_e2e_gain_pct": 20.0,
             "negative_prior": True},
        ]
        scen = {"fusion_integrator:apply_one:%s" % e: [
            {"rejected": [{"exec_id": e, "measured": True, "blocked_evidence": "e2e_no_gain",
                           "reason": "flat"}]}] for e in ("e01", "e02", "e03", "e04")}
        res = self._run(scen, entries, unitside_on=False, calibration_on=False)
        order = [l.split(":")[2] for l in res["labels"]
                 if l.startswith("fusion_integrator:apply_one:")]
        # e02 (positive prior, conf 1.0) first; then 5.0x0.5=2.5, 3.0x0.7=2.1, 20x0.1=2.0
        self.assertEqual(order, ["e02", "e01", "e03", "e04"])

    def _fallback_entries(self):
        mk = lambda e, c, g, **kw: dict({"exec_id": e, "candidate_ids": [c],
                                          "expected_e2e_gain_pct": g}, **kw)
        return [mk("e10", "c10", 2.0, prior_polarity="positive"), mk("e11", "c11", 6.0),
                mk("e12", "c12", 5.0), mk("e15", "c15", 3.5), mk("e16", "c16", 3.2),
                mk("e13", "c13", 1.6), mk("e14", "c14", 0.5)]

    def test_fallback_checkpoint_runs_known_good_entries_with_doubled_pairs(self):
        flat = lambda e: [{"rejected": [{"exec_id": e, "measured": True,
                                         "blocked_evidence": "e2e_no_gain", "reason": "flat"}]}]
        scen = {"unit_status": {"c13": "pass", "c14": "pass"}, "call_ms": 60 * 60000}
        for e in ("e10", "e11", "e12", "e15", "e16"):
            scen["fusion_integrator:apply_one:" + e] = flat(e)
        scen["fusion_integrator:apply_one:e13"] = [
            {"accepted": [{"exec_id": "e13", "throughput_delta_pct": 1.7}],
             "final_overlay": "/ov/e13", "tput": 270}]
        # 10h budget: apply-back budget 7h, checkpoint at 3.5h; 1h per call.
        res = self._run(scen, self._fallback_entries(), budget_s=36000, item_min=60)
        labels = res["labels"]
        order = [l.split(":")[2] for l in labels if l.startswith("fusion_integrator:apply_one:")]
        # e16 (exploratory) is paused at the checkpoint; e13 (unit pass) jumps ahead; e14 (unit
        # pass, gain below the band) is routed to stacking onto the landed set; with the
        # fallback candidates exhausted, exploration resumes with e16.
        self.assertEqual(order, ["e10", "e11", "e12", "e15", "e13", "e16"])
        calls = {c["label"]: c for c in res["calls"]}
        self.assertEqual(calls["fusion_integrator:apply_one:e13"]["inputs"]["AB_MAX_PAIRS"], 6)
        self.assertTrue(calls["fusion_integrator:apply_one:e13"]["inputs"]["FALLBACK_MODE"])
        self.assertIn("FALLBACK MODE", calls["fusion_integrator:apply_one:e13"]["intro"])
        self.assertEqual(calls["fusion_integrator:apply_one:e16"]["inputs"]["AB_MAX_PAIRS"], 3)
        self.assertFalse(calls["fusion_integrator:apply_one:e16"]["inputs"]["FALLBACK_MODE"])
        self.assertNotIn("fusion_integrator:apply_one:e14", labels)
        step = calls["fusion_integrator:apply_stack_step:e14"]
        self.assertEqual(step["inputs"]["CURRENT_STACK"]["members"], ["e13"])
        fapply = res["fapply"]
        self.assertTrue(fapply["fallback"]["entered"])
        self.assertEqual([f["route"] for f in fapply["fallback"]["entries"]],
                         ["solo_2x_pairs", "stack"])
        self.assertEqual(sorted(r["exec_id"] for r in fapply["accepted_fusions"]), ["e13", "e14"])
        self.assertTrue(any("FALLBACK MODE" in l for l in res["logs"]))

    def test_no_fallback_once_the_landing_target_is_met(self):
        scen = {"unit_status": {"c13": "pass", "c14": "pass"}, "call_ms": 60 * 60000}
        for e in ("e10", "e11", "e12", "e15", "e16", "e13"):
            scen["fusion_integrator:apply_one:" + e] = [
                {"accepted": [{"exec_id": e, "throughput_delta_pct": 1.5}], "tput": 270}]
        res = self._run(scen, self._fallback_entries(), budget_s=36000, item_min=60)
        self.assertFalse(res["fapply"]["fallback"]["entered"])
        order = [l.split(":")[2] for l in res["labels"]
                 if l.startswith("fusion_integrator:apply_one:")]
        self.assertEqual(order[:5], ["e10", "e11", "e12", "e15", "e16"])

    def test_wire_failure_is_diagnosed_and_remeasured_once(self):
        entries = [{"exec_id": "e20", "candidate_ids": ["c20"], "expected_e2e_gain_pct": 2.0}]
        scen = {"fusion_integrator:apply_one:e20": [
                    {"rejected": [{"exec_id": "e20", "measured": True,
                                   "blocked_evidence": "not_engaged",
                                   "reason": "fused path not captured by the CUDA graph"}]}],
                "fusion_integrator:apply_one:e20:wire": [
                    {"accepted": [{"exec_id": "e20", "throughput_delta_pct": 2.1}],
                     "final_overlay": "/ov/e20", "tput": 271}]}
        res = self._run(scen, entries, unitside_on=False, calibration_on=False)
        labels = res["labels"]
        self.assertEqual([l for l in labels if l.startswith("fusion_integrator:apply_one:")],
                         ["fusion_integrator:apply_one:e20", "fusion_integrator:apply_one:e20:wire"])
        retry = next(c for c in res["calls"] if c["label"] == "fusion_integrator:apply_one:e20:wire")
        self.assertEqual(retry["inputs"]["WIRE_RETRY"]["blocked_evidence"], "wire_failure")
        self.assertEqual(retry["inputs"]["WIRE_RETRY"]["wire_cause"], "not_engaged")
        self.assertIn("WIRE RETRY", retry["intro"])
        fapply = res["fapply"]
        self.assertEqual([r["exec_id"] for r in fapply["accepted_fusions"]], ["e20"])
        self.assertEqual(fapply["rejected"], [])
        self.assertEqual(len(fapply["wire_failures"]), 1)

    def test_a_second_wire_failure_stands_as_wire_failure(self):
        entries = [{"exec_id": "e20", "candidate_ids": ["c20"], "expected_e2e_gain_pct": 2.0}]
        wire = lambda: [{"rejected": [{"exec_id": "e20", "measured": True,
                                       "blocked_evidence": "not_engaged", "reason": "not engaged"}]}]
        scen = {"fusion_integrator:apply_one:e20": wire(),
                "fusion_integrator:apply_one:e20:wire": wire()}
        res = self._run(scen, entries, unitside_on=False, calibration_on=False)
        self.assertEqual(sum(1 for l in res["labels"] if l.startswith("fusion_integrator:apply_one:")), 2)
        row = res["fapply"]["rejected"][0]
        self.assertEqual(row["blocked_evidence"], "wire_failure")
        self.assertEqual(row["wire_cause"], "not_engaged")
        gate = next(c for c in res["calls"] if c["label"] == "fusion-applyback-gate")
        self.assertIn("--min-landed 2", gate["intro"])
        wb = next(c for c in res["calls"] if c["label"] == "fusion-prior-writeback")["result"]
        self.assertEqual(wb["counts"]["highest_priority_next_round"], 1)

    def test_expected_gain_helper(self):
        src = _source()
        fn = _between(src, "function fusionExpectedGainPct(entry)", "\n}\n") + "\n}\n"
        script = fn + ("console.log(JSON.stringify([fusionExpectedGainPct({e2e_share: 0.01, "
                       "estimated_speedup: 1.25}), fusionExpectedGainPct({expected_e2e_gain_pct: 0.66}), "
                       "fusionExpectedGainPct({}), fusionExpectedGainPct({expected_e2e_gain_pct: null, "
                       "e2e_share: 0.01, estimated_speedup: 1.25})]));")
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(script)
            path = fh.name
        try:
            out = subprocess.run([NODE, path], capture_output=True, text=True, timeout=30)
        finally:
            os.unlink(path)
        self.assertEqual(json.loads(out.stdout), [0.2, 0.66, None, 0.2])


@unittest.skipUnless(NODE, "node not available")
class SemanticReuseTest(unittest.TestCase):
    """args.fusion_reuse_semantics_dir: screen an earlier run's semantic table, no re-capture."""

    def _src_dir(self, phases=("prefill", "decode")):
        src = tempfile.mkdtemp(prefix="kf_src_")
        self.addCleanup(shutil.rmtree, src, True)
        sem = os.path.join(src, "profile", "round_fusion_capture", "semantics_1_2")
        os.makedirs(sem)
        os.makedirs(os.path.join(src, "fusion_capture", "fusion"))
        table = {"phase_coverage": {"required_phases": ["decode", "prefill"]},
                 "tables": [{"phase": ph, "rows": [{"row_id": "%s1" % ph, "phase": ph}]}
                            for ph in phases]}
        with open(os.path.join(sem, "pattern_layer_kernel_table.json"), "w") as fh:
            json.dump(table, fh)
        with open(os.path.join(sem, "STRUCTURAL_LAYER_PATTERNS.json"), "w") as fh:
            json.dump({"patterns": []}, fh)
        with open(os.path.join(src, "fusion_capture", "profile_trace_manifest.json"), "w") as fh:
            json.dump({"schema_version": 1}, fh)
        with open(os.path.join(src, "fusion_capture", "fusion",
                               "available_fusion_kernels.json"), "w") as fh:
            json.dump({"kernels": []}, fh)
        with open(os.path.join(src, "01_SEMANTIC.md"), "w") as fh:
            fh.write("# semantic\n")
        return src

    def _run(self, src, osl=1024):
        source = _source()
        helpers = _between(source, "const FUSION_LEASE_SCRIPT =", "\n// Needs the whole serving TP group")
        block = _between(source, "  // Scope all run-local discovery artifacts together.",
                         "  // ---- Phase 3.0")
        evaldir = tempfile.mkdtemp(prefix="kf_reuse_")
        self.addCleanup(shutil.rmtree, evaldir, True)
        script = r"""
const calls = []; const logs = []; const log = (m) => logs.push(String(m));
const cp = require('child_process');
const A = {}; const WORKFLOW_DIR = %(wf)s; const EVAL_DIR = %(ev)s; const EXEC_PREFIX = '';
const PY = %(py)s;
const FUSION_REUSE_SEMANTICS_DIR = %(src)s; const FUSION_REUSE_TABLE_JSON = '';
const FUSION_REUSE_MANIFEST_JSON = ''; const FUSION_REUSE_ON = true;
const SEMANTICS_MAPPING_ON = true, FUSION_SHAPE_CAPTURE_ON = true, VLLM_PHASE_ANNOTATE = false;
const SEMANTICS_SHAPE_CAPTURE_SETUP = ''; const TRACELENS_INPUTS = {}; const FUSION_RUNTIME_INPUTS = {};
const BACKEND = 'sglang'; const GPU_LIST = ['0']; const ISL = 8192, OSL = %(osl)d, CONC = 4;
const MODEL_PATH = '/m', MODEL_NAME = 'm', WORKLOAD = {}; const FUSION_TOP_K = 8;
const FUSION_SCREEN_PASSES = 2; const FUSION_BUDGET_MS = 28800000; const FUSION_ITEM_MINUTES = 45;
const FUSION_GFX = 'gfx942'; const FUSION_LEASE_WAIT_S = 540; const FUSION_LEASE_WAIT_ROUNDS = 2;
const SERVING_GPU = '0'; const SERVING_TP = 1;
const CAPTURE_SCHEMA = {}, SEMANTICS_SCHEMA = {}, FUSION_RANK_SCHEMA = {}, FUSION_LEASE_SCHEMA = {},
  SEMANTIC_COVERAGE_SCHEMA = {}, FUSION_REUSE_SCHEMA = {};
let FUSION_INPUTS = {FUSION_TOPK_JSON: '', FUSION_CANDIDATES_JSON: ''};
let profile = null, semantics = null, fusionCapture = null, curOverlay = '', curFlags = '', curEnv = '';
let fusionExecutionList = [], fusionCandidatePool = [], fusionBoardK = null, fusionBoardNoiseBand = null,
  fusionBoardPhaseBlind = [];
let fusionDecodeBlind = false, fusionSemanticsReuse = null, fusionFailureStage = '';
function roleAgent(role, phase, intro, inputs) { return {role, phase, intro, inputs}; }
async function safeAgent(prompt, opts) {
  calls.push({label: opts.label, inputs: prompt && prompt.inputs ? prompt.inputs : null});
  if (opts.label === 'fusion-reuse-semantics') {
    const cmd = prompt.match(/\n\n    (.+)\n\n/)[1].replace(/^python3 /, PY + ' ');
    let out; try { out = cp.execSync(cmd, {encoding: 'utf8'}); } catch (e) { out = String(e.stdout || ''); }
    const lines = out.trim().split('\n'); return JSON.parse(lines[lines.length - 1]);
  }
  if (opts.label.startsWith('fusion-lease:')) return {ok: true, alive_count: 0, signalled: []};
  if (opts.label === 'fusion-screen') return {status: 'pass', fusion_topk_json: EVAL_DIR + '/topk.json',
    execution_list: [{exec_id: 'e01'}], candidate_pool: [{exec_id: 'p01'}], k_effective: 9, noise_band_pct: 1.0,
    phase_blind: []};
  return null;
}
%(helpers)s
(async () => {
  try {
%(block)s
    console.log(JSON.stringify({labels: calls.map(c => c.label), calls, logs, semantics,
      fusionSemanticsReuse, fusionDecodeBlind, fusionCandidatePool, fusionBoardK, FUSION_INPUTS}));
  } catch (e) { console.log(JSON.stringify({error: String(e && e.stack || e)})); }
})();
""" % {"wf": json.dumps(WORKFLOW), "ev": json.dumps(evaldir), "py": json.dumps(sys.executable),
       "src": json.dumps(src), "osl": osl, "helpers": helpers, "block": block}
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(script)
            path = fh.name
        try:
            out = subprocess.run([NODE, path], capture_output=True, text=True, timeout=60)
        finally:
            os.unlink(path)
        self.assertEqual(out.returncode, 0, out.stderr)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertNotIn("error", res, res.get("error"))
        return res, evaldir

    def test_reuse_skips_capture_and_semantics_and_screens_the_copied_table(self):
        res, evaldir = self._run(self._src_dir())
        labels = res["labels"]
        self.assertIn("fusion-reuse-semantics", labels)
        self.assertFalse(any(l.startswith("fusion-trace-collector") for l in labels), labels)
        self.assertFalse(any(l.startswith("semantics-mapper") for l in labels), labels)
        screen = next(c for c in res["calls"] if c["label"] == "fusion-screen")
        copied = os.path.join(evaldir, "fusion_capture", "reused", "pattern_layer_kernel_table.json")
        self.assertEqual(screen["inputs"]["SEMANTIC_TABLE_JSON"], copied)
        self.assertTrue(os.path.isfile(copied))
        self.assertTrue(os.path.isfile(os.path.join(evaldir, "01_SEMANTIC.md")))
        with open(os.path.join(evaldir, "fusion_capture", "reused", "reused_from.json")) as fh:
            prov = json.load(fh)
        self.assertEqual(len(prov["files"]["table"]["sha256"]), 64)
        self.assertEqual(res["semantics"]["status"], "pass")
        self.assertFalse(res["fusionDecodeBlind"])
        self.assertEqual(res["fusionCandidatePool"], [{"exec_id": "p01"}])
        self.assertEqual(res["fusionBoardK"], 9)

    def test_a_reused_table_without_decode_is_phase_blind_and_not_recaptured(self):
        res, _ = self._run(self._src_dir(phases=("prefill",)))
        self.assertTrue(res["fusionDecodeBlind"])
        self.assertEqual(res["fusionSemanticsReuse"]["phase_blind"], ["decode"])
        self.assertEqual(res["semantics"]["status"], "partial")
        self.assertFalse(any(l.startswith("semantics-mapper") for l in res["labels"]))
        self.assertIn("fusion-screen", res["labels"])


class LeftoverServerContractTest(unittest.TestCase):
    """A server this run left up must not starve the semantics shape replay."""

    def test_bench_records_server_identity(self):
        with open(os.path.join(WORKFLOW, "scripts", "bench_e2e.sh")) as fh:
            src = fh.read()
        rec = src.index('server_record_identity "$SERVER_PID"')
        ident = src.index("server_identity.json", rec)
        self.assertLess(ident - rec, 1200)

    def test_kernelfusion_reaps_recorded_servers_before_capture_and_replay(self):
        src = _source()
        body = src[src.index("phase('KernelFusion');"):]
        entry = body.index("fusionReapRecordedServers('kernelfusion-entry')")
        capture = body.index("roleAgent('fusion_trace_collector', 'capture'")
        replay_reap = body.index("fusionReapRecordedServers('before-shape-replay')")
        first_replay = body.index("let completed = await completeTable(1)")
        self.assertLess(entry, capture)
        self.assertLess(replay_reap, first_replay)

    def test_decode_less_table_triggers_one_retry_and_is_flagged(self):
        src = _source()
        self.assertIn("completeTable(2)", src)
        self.assertIn("fusionSemanticCoverage(", src)
        self.assertIn("decode_blind: fusionDecodeBlind", src)


if __name__ == "__main__":
    unittest.main()
