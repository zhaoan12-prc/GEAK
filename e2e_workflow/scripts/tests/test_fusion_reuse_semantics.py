"""Reusing an earlier run's semantic result (args.fusion_reuse_semantics_dir)."""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fusion_reuse_semantics as rs  # noqa: E402


class ReuseTest(unittest.TestCase):
    def setUp(self):
        self.src = tempfile.mkdtemp()
        self.dst = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.src)
        self.addCleanup(shutil.rmtree, self.dst)

    def _table(self, sub, phases):
        d = os.path.join(self.src, "profile", "round_fusion_capture", sub)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "pattern_layer_kernel_table.json")
        with open(path, "w") as fh:
            json.dump({"phase_coverage": {"required_phases": ["decode", "prefill"]},
                       "tables": [{"phase": p, "rows": [{"row_id": p}]} for p in phases]}, fh)
        return path

    def test_prefers_semantics_1_2_and_records_provenance(self):
        self._table("semantics", ["prefill"])
        self._table("semantics_1_2", ["prefill", "decode"])
        code, out = rs.reuse(self.src, self.dst)
        self.assertEqual(code, 0)
        self.assertTrue(out["decode_covered"])
        self.assertTrue(out["semantic_table_json"].startswith(self.dst))
        with open(out["reused_from_json"]) as fh:
            prov = json.load(fh)
        self.assertIn("semantics_1_2", prov["files"]["table"]["source"])
        self.assertEqual(out["trace_manifest_json"], "")   # none in the source: not invented

    def test_falls_back_to_semantics_and_reports_missing_decode(self):
        self._table("semantics", ["prefill"])
        code, out = rs.reuse(self.src, self.dst)
        self.assertEqual(code, 0)
        self.assertFalse(out["decode_covered"])
        self.assertEqual(out["missing_phases"], ["decode"])

    def test_explicit_table_override_and_missing_table(self):
        code, out = rs.reuse(self.src, self.dst)
        self.assertEqual(code, 1)
        self.assertFalse(out["ok"])
        other = os.path.join(self.dst, "t.json")
        with open(other, "w") as fh:
            json.dump({"rows": [{"phase": "decode"}]}, fh)
        code, out = rs.reuse("", self.dst, table=other)
        self.assertEqual(code, 0)
        self.assertTrue(out["decode_covered"])


if __name__ == "__main__":
    unittest.main()
