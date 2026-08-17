import json
import os
import sys
import tempfile
import unittest


SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import semantic_decode_boundary_transfer as transfer


class DecodeBoundaryTransferTest(unittest.TestCase):
    """A graph-replayed stage has no module span of its own.

    The boundary is carried over from a workload-identical graph-off run. Only
    the layer assignment moves: no timestamp, duration or device event is
    copied, so the recipient's timing stays its own.
    """

    LAYERS = 4

    def _donor(self, per_layer=3, extra=0):
        rows = []
        for layer_id in range(self.LAYERS):
            for index in range(per_layer):
                rows.append({
                    "row_id": "donor-%d-%d" % (layer_id, index),
                    "short_name": "k%d" % index,
                    "stage": "gemm",
                    "layer_id": layer_id,
                    "layer_instance_id": "donor:pass-0:layer-%d" % layer_id,
                })
            for index in range(extra):
                # Eager mode launches work a replay does not.
                rows.append({
                    "row_id": "donor-extra-%d-%d" % (layer_id, index),
                    "short_name": "eager_only",
                    "stage": "memory",
                    "layer_id": layer_id,
                    "layer_instance_id": "donor:pass-0:layer-%d" % layer_id,
                })
        return rows

    def _recipient(self, per_layer=3):
        rows = []
        position = 0
        for layer_id in range(self.LAYERS):
            for index in range(per_layer):
                rows.append({
                    "row_id": "event-%d" % position,
                    "short_name": "k%d" % index,
                    "stage": "gemm",
                    "device_seq_index": position,
                })
                position += 1
        return rows

    def _run(self, donor, recipient):
        assigned, cuts, basis = transfer._transfer(
            donor, recipient, self.LAYERS)
        return assigned, cuts, basis

    def test_identical_sequences_transfer_exactly(self):
        donor, recipient = self._donor(), self._recipient()
        assigned, cuts, _ = self._run(donor, recipient)
        self.assertEqual(len(assigned), len(recipient))
        self.assertEqual(cuts, [0, 3, 6, 9])
        for row in recipient:
            expected = int(row["row_id"].split("-")[1]) // 3
            self.assertEqual(assigned[row["row_id"]]["layer_id"], expected)

    def test_extra_donor_launches_do_not_shift_the_partition(self):
        """The graph-off run does more work; the cuts must still land right."""
        donor, recipient = self._donor(extra=2), self._recipient()
        assigned, cuts, _ = self._run(donor, recipient)
        self.assertEqual(cuts, [0, 3, 6, 9])
        self.assertEqual(len(assigned), len(recipient))

    def test_every_layer_is_present_and_non_empty(self):
        """A per-row matcher can leave a layer empty; a partition cannot."""
        donor = self._donor()
        # Make one layer's kernels unrecognisable to the matcher.
        for row in donor:
            if row["layer_id"] == 2:
                row["short_name"] = "unaligned_kernel"
        assigned, cuts, _ = self._run(donor, self._recipient())
        layers = sorted(set(v["layer_id"] for v in assigned.values()))
        self.assertEqual(layers, list(range(self.LAYERS)))
        self.assertEqual(sorted(cuts), cuts)
        self.assertEqual(len(set(cuts)), len(cuts))

    def test_cuts_are_monotonic_and_leave_room_for_every_layer(self):
        donor = self._donor()
        recipient = self._recipient(per_layer=1)
        assigned, cuts, _ = self._run(donor, recipient)
        self.assertEqual(sorted(cuts), cuts)
        self.assertEqual(len(set(cuts)), self.LAYERS)
        self.assertEqual(
            sorted(set(v["layer_id"] for v in assigned.values())),
            list(range(self.LAYERS)))

    def test_checks_reject_an_incomplete_partition(self):
        recipient = self._recipient()
        partial = {
            recipient[0]["row_id"]: {"layer_id": 0},
            recipient[1]["row_id"]: {"layer_id": 1},
        }
        checks = transfer._checks(recipient, partial, self.LAYERS)
        self.assertFalse(checks["layers_complete"])
        self.assertLess(checks["assigned_fraction"], 1.0)

    def test_checks_reject_non_monotonic_layers(self):
        recipient = self._recipient()
        scrambled = {
            recipient[0]["row_id"]: {"layer_id": 3},
            recipient[1]["row_id"]: {"layer_id": 0},
        }
        checks = transfer._checks(recipient, scrambled, self.LAYERS)
        self.assertFalse(checks["monotonic_layer_order"])

    def test_a_failed_map_is_refused_by_the_consumer(self):
        """semantic_kernel_mapping must not adopt a map that failed."""
        import semantic_kernel_mapping as mapping
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "map.json")
            with open(path, "w") as fh:
                json.dump({"status": "failed",
                           "failures": ["layers incomplete"],
                           "assignments": {}}, fh)
            with self.assertRaises(ValueError):
                mapping._apply_boundary_map([], path, {})


if __name__ == "__main__":
    unittest.main()
