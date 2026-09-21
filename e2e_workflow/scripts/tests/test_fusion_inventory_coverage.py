import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_inventory_coverage as coverage


class FusionInventoryCoverageTest(unittest.TestCase):
    def _write(self, root, name, payload):
        path = os.path.join(root, name)
        with open(path, "w") as fh:
            json.dump(payload, fh)
        return path

    def test_stage_overlap_cannot_be_hidden_by_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory = {"providers_scanned": ["test"], "kernels": [
                {"name": "fused_norm_quant_a", "op_tags": ["norm", "quant"]},
                {"name": "fused_norm_quant_b", "op_tags": ["norm", "quant"]},
            ]}
            candidates = {"stage_inventory": [], "candidates": [{
                "candidate_id": "existing", "existing_apis": [{
                    "name": "fused_norm_quant_a"}],
                "members": [
                    {"stage": "norm", "duration_us": 2.0},
                    {"stage": "quant", "duration_us": 1.0},
                ],
            }]}
            table = {"tables": [{
                "phase": "decode", "pattern_id": "P0",
                "pattern_layer_count": 2,
                "rows": [
                    {"row_id": "n", "stream": 0, "pos": 0,
                     "stage": "norm", "duration_us": 2.0},
                    {"row_id": "q", "stream": 0, "pos": 1,
                     "stage": "quant", "duration_us": 1.0},
                ],
            }]}
            result = coverage.audit(
                self._write(tmp, "inventory.json", inventory),
                self._write(tmp, "candidates.json", candidates),
                budget=0, require_disposition=True,
                table_path=self._write(tmp, "table.json", table))
            open_row = next(
                row for row in result["rows"]
                if row["name"] == "fused_norm_quant_b")
            self.assertTrue(open_row["forced_overlap"])
            self.assertTrue(open_row["in_budget"])
            self.assertEqual(result["status"], "fail")

    def test_unmapped_stage_warns_without_blocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory = {"providers_scanned": ["test"], "kernels": []}
            candidates = {
                "stage_inventory": [{"stage": "linear_attn"}, {"stage": "rope"}],
                "candidates": [],
            }
            result = coverage.audit(
                self._write(tmp, "inventory.json", inventory),
                self._write(tmp, "candidates.json", candidates),
                require_disposition=True)

            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["open_count"], 0)
            self.assertEqual(result["errors"], [])
            self.assertEqual(result["unmapped_stages"], ["linear_attn", "rope"])
            self.assertTrue(result["warnings"])
            self.assertIn("审计范围提示（不阻塞）", coverage.render_markdown(result))

    def test_real_gap_still_blocks_with_unmapped_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory = {"providers_scanned": ["test"], "kernels": [
                {"name": "fused_norm_quant", "op_tags": ["norm", "quant"]},
            ]}
            candidates = {
                "stage_inventory": [
                    {"stage": "norm"},
                    {"stage": "quant"},
                    {"stage": "linear_attn"},
                ],
                "candidates": [],
            }
            result = coverage.audit(
                self._write(tmp, "inventory.json", inventory),
                self._write(tmp, "candidates.json", candidates),
                require_disposition=True)

            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["open_count"], 1)
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(result["unmapped_stages"], ["linear_attn"])
            self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
