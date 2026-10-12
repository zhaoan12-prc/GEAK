"""KernelFusion apply-back speed-ups: gsm8k gate harness/cache and pair-by-pair A/B.

Behaviour tests execute workflow code for real under node (skipped when node is absent).
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
                    "AB_DECIDE_SCRIPT:", "AB_MAX_PAIRS: abPairs"):
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
