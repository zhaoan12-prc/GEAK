"""KernelFusion apply-back speed-ups: gsm8k gate harness/cache, pair-by-pair A/B, and
parallel unit-side waves.

The scheduling code is executed for real under node (skipped when node is absent), so
the budget order, ladder release and GPU pinning are checked as behaviour, not as text.
"""
import json
import os
import re
import shutil
import subprocess
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


def _run_node(script):
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        out = subprocess.run([NODE, path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    if out.returncode != 0:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout.strip().splitlines()[-1])


class AccuracyGateContractTest(unittest.TestCase):
    def test_apply_one_gets_the_gate_harness_and_cache(self):
        src = _source()
        block = src[src.index("roleAgent('fusion_integrator', 'apply_one'"):]
        block = block[:block.index("schema: FUSION_APPLY_SCHEMA")]
        for key in ("GSM8K_CONCURRENCY: fusionGsm8kConcurrency(curFlags)",
                    "GSM8K_THINKING: FUSION_GSM8K_THINKING",
                    "ACCURACY_STACK_KEY: accuracyKey",
                    "fusionAccuracyRef.stack_key === accuracyKey",
                    "AB_DECIDE_SCRIPT:", "AB_MAX_PAIRS: FUSION_AB_MAX_PAIRS"):
            self.assertIn(key, block)

    def test_thinking_is_off_by_default(self):
        self.assertIn("A.fusion_gsm8k_thinking != null ? A.fusion_gsm8k_thinking : 'false'", _source())

    def test_cache_follows_the_stack(self):
        src = _source()
        # Base measured for this stack is kept; an accepted fusion re-keys it to the new stack.
        self.assertIn("measuredRef.stack_key === accuracyKey", src)
        self.assertIn("{ ...acc, stack_key: fusionStackKey() } : null", src)
        key_fn = src[src.index("function fusionStackKey()"):]
        key_fn = key_fn[:key_fn.index("\n}\n")]
        # The combined overlay is rewritten in place, so the key must include the fusion ids.
        self.assertIn("acceptedFusions", key_fn)

    def test_schema_carries_the_cache_fields(self):
        src = _source()
        schema = src[src.index("const FUSION_APPLY_SCHEMA"):]
        schema = schema[:schema.index("['accepted_fusions']);")]
        self.assertIn("accuracy_reference", schema)
        self.assertIn("accepted_accuracy", schema)

    def test_role_orders_ab_before_accuracy_and_pins_concurrency(self):
        role = _role("fusion_integrator.md")
        self.assertLess(role.index("throughput A/B FIRST"), role.index("Accuracy verification"))
        self.assertIn('--concurrency "$GSM8K_CONCURRENCY"', role)
        self.assertIn("AB_DECISION=continue", role)
        self.assertIn("ACCURACY_REFERENCE", role)
        self.assertNotIn("≥4 reps/leg", role)


@unittest.skipUnless(NODE, "node not available")
class Gsm8kConcurrencyBehaviourTest(unittest.TestCase):
    def _eval(self, flags, conc=4):
        src = _source()
        fn = src[src.index("function fusionGsm8kConcurrency(flags)"):]
        fn = fn[:fn.index("\n}\n") + 3]
        return _run_node(f"const CONC = {conc};\n{fn}\nconsole.log(JSON.stringify("
                         f"fusionGsm8kConcurrency({json.dumps(flags)})));")

    def test_uses_cuda_graph_max_bs(self):
        self.assertEqual(self._eval("--attention-backend aiter --cuda-graph-max-bs 4 --tp 8", conc=64), 4)

    def test_last_occurrence_and_equals_form(self):
        self.assertEqual(self._eval("--cuda-graph-max-bs 8 --x --cuda-graph-max-bs=16"), 16)

    def test_falls_back_to_workload_concurrency(self):
        self.assertEqual(self._eval("--attention-backend aiter", conc=32), 32)


@unittest.skipUnless(NODE, "node not available")
class ApplyBudgetBehaviourTest(unittest.TestCase):
    """The apply-back budget counts only entries with a 单侧-eligible candidate."""

    # 1001 run: e01/e04 failed 单侧, e07 passed but was cut by the old first-6 slice.
    LIST = [{"exec_id": "e%02d" % i, "candidate_ids": ["c%02d" % i]} for i in range(1, 11)]
    ELIGIBLE = ["c02", "c03", "c05", "c06", "c07"]

    def _entries(self, eligible, budget):
        src = _source()
        fn = src[src.index("function fusionApplyBudgetEntries("):]
        fn = fn[:fn.index("\n}\n") + 3]
        out = _run_node(f"{fn}\nconsole.log(JSON.stringify(fusionApplyBudgetEntries("
                        f"{json.dumps(self.LIST)}, {json.dumps(eligible)}, {budget})"
                        f".map((e) => e.exec_id)));")
        return out

    def test_unitside_failures_do_not_take_a_slot(self):
        got = self._entries(self.ELIGIBLE, 6)
        self.assertIn("e07", got)
        # Only five entries passed, so a budget of six never cuts.
        self.assertEqual(got, [e["exec_id"] for e in self.LIST])
        self.assertEqual(self._entries(self.ELIGIBLE, 5),
                         ["e01", "e02", "e03", "e04", "e05", "e06", "e07"])

    def test_cut_after_the_nth_eligible_entry(self):
        self.assertEqual(self._entries(self.ELIGIBLE, 2), ["e01", "e02", "e03"])

    def test_without_the_eligible_list_it_is_the_plain_slice(self):
        self.assertEqual(self._entries(None, 6), ["e01", "e02", "e03", "e04", "e05", "e06"])

    def test_zero_budget_calls_nothing(self):
        self.assertEqual(self._entries(self.ELIGIBLE, 0), [])

    def test_loop_and_aggregate_are_wired(self):
        src = _source()
        self.assertIn("fusionApplyBudgetEntries(\n      fusionExecutionList, fusionUnitEligibleIds, FUSION_BUDGET)", src)
        self.assertIn("fusionUnitEligibleIds = (unitAgg && Array.isArray(unitAgg.applyback_eligible_ids))", src)
        schema = src[src.index("const FUSION_UNIT_AGG_SCHEMA"):]
        self.assertIn("applyback_eligible_ids", schema[:schema.index("});")])


@unittest.skipUnless(NODE, "node not available")
class CombinedApplyBackBehaviourTest(unittest.TestCase):
    """Runs the real apply-back block (combined + serial fallback) with stub agents."""

    ENTRIES = [{"exec_id": "e%02d" % i, "candidate_ids": ["c%02d" % i]} for i in range(1, 6)]
    ELIGIBLE = ["c02", "c03", "c05"]

    def _run(self, mode="combined", decision="accept", gpus="0,1"):
        src = _source()
        start = src.index("    let applyState = {")
        end = src.index("        commitStep(step, accuracyKey, applyEntry.exec_id);\n      }\n    }\n")
        end += len("        commitStep(step, accuracyKey, applyEntry.exec_id);\n      }\n    }\n")
        block = src[start:end]
        harness = f"""
const calls = [];
const log = () => {{}};
const EVAL_DIR = '/e', MODEL_PATH = '/m', SERVING_TP = 8, SERVING_GPU = '0-7', WORKLOAD = {{}};
const WORKFLOW_DIR = '/w', BASELINE_TPUT = 190, NOISE_BAND = 0.5, FUSION_APPLY_TIMEOUT_MS = 0;
const FUSION_RUNTIME_INPUTS = {{}}, ACCURACY_INPUTS = {{}}, FUSION_AUTHOR_SCHEMA = {{}}, FUSION_APPLY_SCHEMA = {{}};
const FUSION_APPLYBACK_MODE = {json.dumps(mode)}, FUSION_BUDGET = 10, FUSION_UNIT_PARALLEL = 8;
const FUSION_COMBINED_AB_REPEATS = 3, FUSION_GSM8K_THINKING = false, FUSION_AB_MAX_PAIRS = 3;
const GPU_LIST = {json.dumps(gpus)}.split(',');
const FUSION_INPUTS = {{ FUSION_TOPK_JSON: '/t', FUSION_UNITSIDE_JSON: '/u' }};
const fusionApplyEntries = {json.dumps(self.ENTRIES)};
const fusionUnitEligibleIds = {json.dumps(self.ELIGIBLE)};
const fusionGsm8kConcurrency = () => 4;
let curOverlay = '', curFlags = '', curEnv = '', curTput = 190;
let fapply = null, fusionApplyFailedExec = '', fusionApplyUnprocessed = [];
let fusionFailureStage = '', fusionAccuracyRef = null;
const acceptedFusions = [], fusionRecoveryState = {{}};
const fusionStackKey = () => JSON.stringify(acceptedFusions.map((r) => r.exec_id));
const roleAgent = (role, phase, intro, inputs) => ({{ phase, inputs }});
const safeAgent = async (call) => {{
  calls.push({{ phase: call.phase, exec: call.inputs.TARGET_EXEC_ID || '', gpu: call.inputs.GPU_ID || '',
                authored: call.inputs.AUTHORED_OVERLAY ? call.inputs.AUTHORED_OVERLAY.overlay_dir : '' }});
  if (call.phase === 'author_one') {{
    const id = call.inputs.TARGET_EXEC_ID;
    return id === 'e03' ? {{ exec_id: id, status: 'blocked', reason: 'no seam' }}
                        : {{ exec_id: id, status: 'authored', overlay_dir: '/o/' + id, banner_tag: id }};
  }}
  if (call.phase === 'combined_ab') {{
    if ({json.dumps(decision)} !== 'accept') return {{ combined_decision: 'reject', accepted_fusions: [] }};
    return {{ combined_decision: 'accept', final_overlay: '/o/combined', e2e_throughput_tok_s: 247,
             accepted_fusions: [{{ exec_id: 'e02' }}, {{ exec_id: 'e05' }}],
             rejected: [{{ exec_id: 'e03', reason: 'no seam' }}] }};
  }}
  return {{ accepted_fusions: [{{ exec_id: call.inputs.TARGET_EXEC_ID }}],
           final_overlay: '/o/serial', e2e_throughput_tok_s: curTput + 1 }};
}};
(async () => {{
{block}
  console.log(JSON.stringify({{ calls, curTput, curOverlay,
    accepted: acceptedFusions.map((r) => r.exec_id) }}));
}})();
"""
        return _run_node(harness)

    def test_accepted_stack_needs_no_per_entry_ab(self):
        out = self._run()
        phases = [c["phase"] for c in out["calls"]]
        self.assertEqual(phases.count("author_one"), 3)        # only unit-side passes
        self.assertEqual(phases.count("combined_ab"), 1)
        self.assertNotIn("apply_one", phases)
        self.assertEqual(out["accepted"], ["e02", "e05"])
        self.assertEqual(out["curTput"], 247)
        self.assertEqual(out["curOverlay"], "/o/combined")

    def test_authoring_is_pinned_one_gpu_each(self):
        out = self._run(gpus="0,1")
        authored = [c for c in out["calls"] if c["phase"] == "author_one"]
        self.assertEqual([c["gpu"] for c in authored], ["0", "1", "0"])   # waves of 2

    def test_rejected_stack_falls_back_to_serial_and_reuses_overlays(self):
        out = self._run(decision="reject")
        serial = [c for c in out["calls"] if c["phase"] == "apply_one"]
        self.assertEqual([c["exec"] for c in serial], [e["exec_id"] for e in self.ENTRIES])
        reuse = {c["exec"]: c["authored"] for c in serial}
        self.assertEqual(reuse["e02"], "/o/e02")
        self.assertEqual(reuse["e03"], "")                     # blocked at authoring

    def test_serial_mode_never_authors_separately(self):
        phases = [c["phase"] for c in self._run(mode="serial")["calls"]]
        self.assertNotIn("author_one", phases)
        self.assertNotIn("combined_ab", phases)
        self.assertEqual(phases.count("apply_one"), len(self.ENTRIES))


class CombinedModeDefaultsTest(unittest.TestCase):
    def test_combined_is_default_and_widens_the_budget(self):
        src = _source()
        self.assertIn("String(A.fusion_applyback_mode || 'combined').trim() === 'serial'", src)
        self.assertIn("(FUSION_APPLYBACK_MODE === 'combined' ? 10 : 6)", src)

    def test_combined_ab_gets_the_script_and_gsm8k_harness(self):
        src = _source()
        block = src[src.index("roleAgent('fusion_integrator', 'combined_ab'"):]
        block = block[:block.index("schema: FUSION_APPLY_SCHEMA")]
        for key in ("COMBINED_AB_SCRIPT:", "AB_REPEATS: FUSION_COMBINED_AB_REPEATS",
                    "GSM8K_CONCURRENCY: fusionGsm8kConcurrency(curFlags)", "AUTHORED: authored"):
            self.assertIn(key, block)

    def test_role_documents_both_phases(self):
        role = _role("fusion_integrator.md")
        self.assertIn("## PHASE=author_one", role)
        self.assertIn("## PHASE=combined_ab", role)
        self.assertIn('bash "$COMBINED_AB_SCRIPT"', role)

    def test_run_e2e_forwards_the_mode(self):
        with open(os.path.join(os.path.dirname(WORKFLOW), "interface", "run_e2e.py")) as fh:
            self.assertIn('"fusion_applyback_mode"', fh.read())
