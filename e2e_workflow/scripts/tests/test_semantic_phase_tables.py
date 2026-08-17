import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_kernel_mapping as mapping
import semantic_phase_tables as phase_tables


def _document(phase, trace, rows_by_pattern):
    return {
        "schema_version": 2,
        "trace_path": trace,
        "trace_sha256": "sha-%s" % phase,
        "patterns_path": "/patterns.json",
        "table_phases": [phase],
        "tables": [
            {
                "phase": phase,
                "pattern_id": pattern_id,
                "pattern_display_name": pattern_id,
                "pattern_layer_ids": [0, 1],
                "pattern_layer_count": 2,
                "representative_layer_id": 1,
                "selected_bucket": {"phase": phase, "batch_size": 1},
                "event_count": len(rows),
                "layer_total_us": 1.0,
                "rows": rows,
            }
            for pattern_id, rows in rows_by_pattern.items()],
    }


def _rows(count, start=0):
    return [{"row_id": "event-%d" % (start + index), "pos": index,
             "stage": "gemm", "short_name": "k%d" % index,
             "duration_us": 1.0, "layer_total_pct": 1.0,
             "shape": {}, "semantic_evidence": {"level": "K"}}
            for index in range(count)]


class PhaseTableCombineTest(unittest.TestCase):
    def test_prefill_tables_come_before_decode_tables(self):
        documents = [
            # deliberately supplied decode-first
            ("decode", _document("decode", "/d.gz", {"P0": _rows(2)})),
            ("prefill", _document("prefill", "/p.gz", {"P0": _rows(3)})),
        ]
        combined, markdown = phase_tables.combine_documents(documents)
        self.assertEqual(
            [table["phase"] for table in combined["tables"]],
            ["prefill", "decode"])
        self.assertEqual(combined["table_phases"], ["prefill", "decode"])
        self.assertLess(
            markdown.index("## PREFILL"), markdown.index("## DECODE"))

    def test_rows_are_copied_verbatim(self):
        prefill = _document("prefill", "/p.gz", {"P0": _rows(3)})
        decode = _document("decode", "/d.gz", {"P0": _rows(2)})
        combined, _ = phase_tables.combine_documents(
            [("prefill", prefill), ("decode", decode)])
        self.assertEqual(combined["tables"][0]["rows"], prefill["tables"][0]["rows"])
        self.assertEqual(combined["tables"][1]["rows"], decode["tables"][0]["rows"])
        self.assertFalse(combined["combination"]["rows_recomputed"])

    def test_each_stage_trace_is_recorded(self):
        combined, _ = phase_tables.combine_documents([
            ("prefill", _document("prefill", "/p.gz", {"P0": _rows(1)})),
            ("decode", _document("decode", "/d.gz", {"P0": _rows(1)})),
        ])
        self.assertEqual(
            [source["trace_path"] for source in combined["stage_sources"]],
            ["/p.gz", "/d.gz"])
        # a single top-level trace_path would be a lie for a combined document
        self.assertNotIn("trace_path", combined)

    def test_colliding_row_ids_across_stages_are_counted_not_hidden(self):
        combined, _ = phase_tables.combine_documents([
            ("prefill", _document("prefill", "/p.gz", {"P0": _rows(2)})),
            ("decode", _document("decode", "/d.gz", {"P0": _rows(2)})),
        ])
        self.assertEqual(
            combined["combination"]["duplicate_row_ids_across_stages"], 2)

    def test_two_stages_publishing_the_same_table_is_rejected(self):
        with self.assertRaises(ValueError):
            phase_tables.combine_documents([
                ("a", _document("decode", "/a.gz", {"P0": _rows(1)})),
                ("b", _document("decode", "/b.gz", {"P0": _rows(1)})),
            ])

    def test_combine_writes_both_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for phase, trace in (("prefill", "/p.gz"), ("decode", "/d.gz")):
                path = os.path.join(tmp, "%s.json" % phase)
                with open(path, "w") as fh:
                    json.dump(_document(phase, trace, {"P0": _rows(2)}), fh)
                paths.append((phase, path))
            out_json = os.path.join(tmp, "out", "table.json")
            out_md = os.path.join(tmp, "out", "table.md")
            result = phase_tables.combine(paths, out_json, out_md)
            self.assertTrue(os.path.exists(out_json))
            self.assertTrue(os.path.exists(out_md))
            self.assertEqual(result["row_count"], 4)
            self.assertEqual(result["table_phases"], ["prefill", "decode"])


class StageLabelTest(unittest.TestCase):
    def test_stage_files_map_to_their_phase(self):
        self.assertEqual(
            mapping._stage_label("run-TP-0-EXTEND.trace.json.gz", None, 0),
            "prefill")
        self.assertEqual(
            mapping._stage_label("run-TP-0-DECODE.trace.json.gz", None, 1),
            "decode")

    def test_prefill_and_decode_never_share_a_label(self):
        """They become directory names; a collision overwrites an audit set."""
        labels = {
            mapping._stage_label("x-TP-0-EXTEND.trace.json.gz", None, 0),
            mapping._stage_label("x-TP-0-DECODE.trace.json.gz", None, 1),
        }
        self.assertEqual(len(labels), 2)

    def test_unrecognised_name_falls_back_to_requested_phases(self):
        self.assertEqual(
            mapping._stage_label("custom.json", {"decode"}, 0), "decode")
        self.assertEqual(
            mapping._stage_label("custom.json", None, 3), "stage-03")


if __name__ == "__main__":
    unittest.main()
