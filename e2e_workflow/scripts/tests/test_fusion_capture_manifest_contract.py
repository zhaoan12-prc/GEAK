#!/usr/bin/env python3
"""Regression checks for the deterministic KernelFusion manifest contract."""

import os
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
WORKFLOW = os.path.dirname(SCRIPTS)


class FusionCaptureManifestContractTest(unittest.TestCase):
    def _workflow_source(self):
        with open(os.path.join(WORKFLOW, "e2e_workflow.js")) as fh:
            return fh.read()

    def test_capture_generates_and_validates_manifest(self):
        with open(os.path.join(SCRIPTS, "bench_e2e.sh")) as fh:
            source = fh.read()
        self.assertIn('--trace-dir "$PROFILE_DIR"', source)
        self.assertIn('--out "$_TRACE_MANIFEST"', source)
        self.assertIn('doc.get("status") != "pass"', source)
        self.assertIn('not doc.get("analysis_rank_trace")', source)

    def test_workflow_falls_back_to_deterministic_manifest(self):
        with open(os.path.join(WORKFLOW, "e2e_workflow.js")) as fh:
            source = fh.read()
        self.assertIn(
            'const fusionCaptureDir = `${EVAL_DIR}/${fusionRound}`;',
            source,
        )
        self.assertIn(
            'const expectedFusionManifest = `${fusionCaptureDir}/profile_trace_manifest.json`;',
            source,
        )
        self.assertIn("CAPTURE_DIR: fusionCaptureDir", source)
        self.assertIn("TRACE_MANIFEST_JSON: expectedFusionManifest", source)
        self.assertNotIn("CAPTURE_REPEATS:", source)
        self.assertNotIn("CAPTURE_NUM_PROMPTS:", source)
        self.assertIn('TRACE_MANIFEST_JSON: fusionTraceManifest', source)
        self.assertNotIn(
            "notes: 'fusion capture produced no raw trace manifest'",
            source,
        )

    def test_capture_role_uses_deterministic_output_and_workload_sizing(self):
        with open(os.path.join(WORKFLOW, "roles", "fusion_trace_collector.md")) as fh:
            source = fh.read()
        self.assertIn("`OUT_DIR=CAPTURE_DIR`", source)
        self.assertIn("Do not override", source)
        self.assertIn("workload-derived `NUM_PROMPTS`, `REPEATS`", source)
        self.assertNotIn("CAPTURE_NUM_PROMPTS", source)
        self.assertNotIn("CAPTURE_REPEATS", source)
        self.assertIn("publish exactly `TRACE_MANIFEST_JSON`", source)
        self.assertIn("Do not return before the foreground capture command", source)

    def test_explicit_fusion_discovery_is_not_silently_skipped(self):
        with open(os.path.join(WORKFLOW, "e2e_workflow.js")) as fh:
            source = fh.read()
        self.assertIn("const FUSION_REQUIRED", source)
        self.assertIn(
            "KernelFusion was explicitly required but produced no apply-back-ready Top-K",
            source,
        )

    def test_external_fusion_artifacts_cannot_bypass_run_local_discovery(self):
        source = self._workflow_source()
        self.assertNotIn("suppliedFusionPriorComplete", source)
        self.assertNotIn("fusionInputsComplete", source)
        self.assertNotIn("const FU =", source)
        self.assertNotIn("(A.fusion && typeof A.fusion === 'object')", source)
        self.assertNotIn("complete fusion prior/state supplied", source)
        self.assertIn("if (!FAST_MODE && FUSION_DISCOVERY_ON)", source)
        self.assertIn("FUSION_TOPK_JSON: ''", source)
        self.assertIn("FUSION_CANDIDATES_JSON: ''", source)
        self.assertIn("FUSION_UNITSIDE_JSON: ''", source)

    def test_applyback_cannot_time_out_into_a_stale_profile(self):
        source = self._workflow_source()
        self.assertIn("const FUSION_APPLY_TIMEOUT_MS", source)
        self.assertIn(
            "timeoutMs: FUSION_APPLY_TIMEOUT_MS",
            source,
        )

    def test_completed_shape_table_is_the_source_for_candidates_and_topk(self):
        """The eager-merged table, not the pre-probe table, must flow forward."""
        source = self._workflow_source()
        complete = source.index("if (completed) semantics = completed;")
        generate = source.index(
            "label: 'fusion-analyst:generate'", complete)
        rank = source.index("label: 'fusion-analyst:rank'", generate)
        self.assertLess(complete, generate)
        self.assertLess(generate, rank)
        generate_block = source[complete:generate]
        rank_block = source[generate:rank]
        self.assertIn(
            "SEMANTIC_TABLE_JSON: semantics.semantic_table_json",
            generate_block,
        )
        self.assertIn(
            "SEMANTIC_TABLE_JSON: semantics.semantic_table_json",
            rank_block,
        )
        self.assertIn(
            "FUSION_CANDIDATES_JSON: discover.fusion_candidates_json",
            rank_block,
        )
        self.assertIn(
            "FUSION_VALIDATION_JSON: discover.validation_json",
            rank_block,
        )

    def test_partial_semantics_cannot_enter_fusion_discovery(self):
        source = self._workflow_source()
        self.assertIn(
            "if (semantics && semantics.status === 'pass' && semantics.semantic_table_json)",
            source,
        )

    def test_topk_execution_list_drives_exact_unitside_candidates(self):
        """Every unit-side call must be named by Top-K exec/candidate ids."""
        source = self._workflow_source()
        start = source.index("const execById = new Map();")
        aggregate = source.index(
            "label: 'fusion-unit:aggregate'", start)
        block = source[start:aggregate]
        self.assertIn("ranked.execution_list || []", block)
        self.assertIn("unit_representative_candidate_id", block)
        self.assertIn("unit_equivalent_candidate_ids", block)
        self.assertIn("equivalentCovered", block)
        self.assertIn("EXEC_ID: item.exec_id", block)
        self.assertIn("CANDIDATE_ID: item.candidate_id", block)
        self.assertIn(
            "FUSION_CANDIDATES_JSON: discover.fusion_candidates_json",
            block,
        )
        self.assertIn("FUSION_TOPK_JSON: ranked.fusion_topk_json", block)

    def test_unitside_aggregate_is_the_only_applyback_gate(self):
        """Apply-back must consume the aggregate produced for this Top-K."""
        source = self._workflow_source()
        aggregate = source.index("label: 'fusion-unit:aggregate'")
        preserve = source.index(
            "FUSION_INPUTS.FUSION_UNITSIDE_JSON = aggregate.fusion_unitside_json;",
            aggregate,
        )
        applyback = source.index("label: 'fusion_integrator:apply_back'", preserve)
        self.assertLess(aggregate, preserve)
        self.assertLess(preserve, applyback)
        apply_block = source[preserve:applyback]
        self.assertIn(
            "FUSION_INPUTS.FUSION_TOPK_JSON && FUSION_INPUTS.FUSION_UNITSIDE_JSON",
            apply_block,
        )
        self.assertIn(
            "FUSION_UNITSIDE_JSON: FUSION_INPUTS.FUSION_UNITSIDE_JSON",
            apply_block,
        )

    def test_applyback_failure_preserves_prefusion_state_and_continues_profile(self):
        source = self._workflow_source()
        fallback = source.index(
            "KernelFusion apply-back failed or returned no terminal result")
        profile = source.index("phase('Profile');", fallback)
        block = source[fallback:profile]
        self.assertLess(fallback, profile)
        self.assertIn("pre-Fusion overlay/flags/env/throughput", block)
        self.assertIn("applied: [], blocked: [], deferred: []", block)
        self.assertIn("else if (!fapply) fusionStatus = 'applyback_failed';", block)
        self.assertNotIn(
            "KernelFusion apply-back did not reach a terminal state", source)


if __name__ == "__main__":
    unittest.main()
