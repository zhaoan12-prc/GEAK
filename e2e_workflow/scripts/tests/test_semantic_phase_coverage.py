"""semantic_phase_coverage: a decode-less table must be visible as decode-less."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import semantic_phase_coverage as cov  # noqa: E402


def _table(phases, required=("decode", "prefill")):
    return {"tables": [{"phase": p, "pattern_id": "P0", "rows": [{"row_id": p + "-0"}]}
                       for p in phases],
            "phase_coverage": {"required_phases": list(required)}}


class PhaseCoverageTest(unittest.TestCase):
    def test_prefill_only_table_reports_decode_missing(self):
        out = cov.coverage(_table(["prefill"]))
        self.assertFalse(out["decode_covered"])
        self.assertEqual(out["missing_phases"], ["decode"])

    def test_full_table_is_covered(self):
        out = cov.coverage(_table(["prefill", "decode"]))
        self.assertTrue(out["decode_covered"])
        self.assertEqual(out["missing_phases"], [])

    def test_an_empty_phase_table_does_not_count(self):
        table = _table(["prefill"])
        table["tables"].append({"phase": "decode", "pattern_id": "P0", "rows": []})
        self.assertFalse(cov.coverage(table)["decode_covered"])

    def test_cli_prints_one_json_object(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(_table(["prefill"]), fh)
            path = fh.name
        try:
            out = subprocess.run([sys.executable, os.path.join(HERE, "semantic_phase_coverage.py"), path],
                                 capture_output=True, text=True, timeout=30,
                                 env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        finally:
            os.unlink(path)
        self.assertEqual(out.returncode, 0)
        self.assertFalse(json.loads(out.stdout)["decode_covered"])


if __name__ == "__main__":
    unittest.main()
