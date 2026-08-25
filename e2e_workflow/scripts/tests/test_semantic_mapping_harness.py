import json
import os
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_mapping_harness as harness


class SemanticMappingHarnessTest(unittest.TestCase):
    def test_build_table_owns_validation_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = os.path.join(tmp, "result.json")
            with mock.patch.object(
                    harness.validate_structural_patterns, "validate",
                    return_value={"validation": {"definition_preserved": True}}) as validate, \
                    mock.patch.object(
                        harness.semantic_kernel_mapping, "build",
                        return_value={
                            "status": "pass",
                            "semantic_table_json": os.path.join(tmp, "table.json"),
                        }) as build:
                result = harness.build_table(
                    "draft.json", "config.json", ["model.py"], "trace.json",
                    tmp, {"decode"}, result_path)
            validate.assert_called_once_with(
                "draft.json", "config.json", ["model.py"],
                os.path.join(tmp, "STRUCTURAL_LAYER_PATTERNS.json"))
            build.assert_called_once_with(
                "trace.json", os.path.join(tmp, "STRUCTURAL_LAYER_PATTERNS.json"),
                tmp, {"decode"})
            self.assertEqual(result["phase"], "build_table")
            self.assertTrue(result["structural_pattern_validation"]
                            ["definition_preserved"])
            with open(result_path) as fh:
                self.assertEqual(json.load(fh)["status"], "pass")

    def test_complete_table_delegates_to_shape_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                    harness.semantic_shape_merge, "merge",
                    return_value={"status": "partial"}) as merge:
                result = harness.complete_table(
                    "table.json", "plan.json", "shape.jsonl", tmp)
            merge.assert_called_once_with(
                "table.json", "plan.json", "shape.jsonl", tmp)
            self.assertEqual(result["phase"], "complete_table")
            self.assertEqual(result["status"], "partial")

    def test_capture_complete_runs_one_replay_then_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            shape_log = os.path.join(tmp, "shape.jsonl")
            with mock.patch.object(
                    harness.run_semantic_shape_capture, "capture",
                    return_value={"status": "pass", "shape_log": shape_log}) as capture, \
                    mock.patch.object(
                        harness, "complete_table",
                        return_value={"status": "pass"}) as complete:
                result = harness.capture_complete(
                    "table.json", "plan.json", "setup.json", tmp,
                    phases=["decode"])
            capture.assert_called_once_with(
                "setup.json", "plan.json", os.path.join(tmp, "capture"),
                False, ["decode"], 1)
            complete.assert_called_once_with(
                "table.json", "plan.json", shape_log,
                os.path.join(tmp, "merged"))
            self.assertEqual(result["phase"], "capture_complete")
            self.assertEqual(result["capture"]["shape_log"], shape_log)


if __name__ == "__main__":
    unittest.main()
