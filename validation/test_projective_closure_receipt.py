from copy import deepcopy
from fractions import Fraction
import hashlib
import random
import unittest

from projective_closure_receipt import (
    build_snapshot,
    make_score_receipt,
    verify_score_receipt,
)


def _manifest(label="manifest"):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reference_score(entries):
    selected = [entry for entry in entries if entry["selected"]]
    r = len(selected)
    if r == 0:
        return Fraction(0)
    return sum(
        (
            max(
                Fraction(0),
                (Fraction(1, r) if entry["selected"] else Fraction(0))
                - Fraction(entry["b"]),
            )
            for entry in entries
        ),
        Fraction(0),
    )


class ProjectiveClosureReceiptTests(unittest.TestCase):
    def _round_trip(self, entries, sequence=1, previous=None):
        config = {
            "alpha": "1/20",
            "weight_digest": _manifest("weights"),
            "evidence_version": "evidence-v1",
            "tree_schema": 1,
        }
        snapshot = build_snapshot(
            entries,
            manifest_root=_manifest(),
            sequence=sequence,
            previous_snapshot_hash=previous,
            configuration=config,
        )
        receipt = make_score_receipt(
            snapshot,
            alpha=Fraction(1, 20),
            weight_digest=_manifest("weights"),
            evidence_version="evidence-v1",
        )
        score = verify_score_receipt(
            receipt,
            expected_snapshot_hash=snapshot.snapshot_hash,
            expected_manifest_root=_manifest(),
            expected_weight_digest=_manifest("weights"),
            expected_evidence_version="evidence-v1",
            expected_alpha=Fraction(1, 20),
            expected_selected_digest=receipt["selected_digest"],
        )
        self.assertEqual(score, _reference_score(entries))
        return snapshot, receipt

    def test_randomized_exact_receipts_match_linear_oracle(self):
        rng = random.Random(975901)
        for case in range(500):
            n = rng.randint(1, 40)
            entries = [
                {
                    "hypothesis_id": f"h-{case}-{i}",
                    "b": Fraction(rng.randint(-30, 30), rng.randint(1, 31)),
                    "selected": rng.random() < 0.45,
                }
                for i in range(n)
            ]
            if not any(entry["selected"] for entry in entries):
                entries[-1]["selected"] = True
            self._round_trip(entries)

    def test_duplicate_keys_and_strict_threshold_ties(self):
        entries = [
            {"hypothesis_id": "a", "b": Fraction(-1, 2), "selected": True},
            {"hypothesis_id": "b", "b": Fraction(0), "selected": True},
            {"hypothesis_id": "c", "b": Fraction(1, 3), "selected": True},
            {"hypothesis_id": "d", "b": Fraction(1, 3), "selected": False},
            {"hypothesis_id": "e", "b": Fraction(2), "selected": False},
        ]
        self._round_trip(entries)

    def test_rejects_nonportable_float_snapshot_configuration(self):
        with self.assertRaises(TypeError):
            build_snapshot(
                [{"hypothesis_id": "h", "b": Fraction(1, 4), "selected": True}],
                manifest_root=_manifest(),
                sequence=1,
                configuration={"alpha": 0.05},
            )

    def test_rejects_duplicate_ids_and_float_evidence(self):
        with self.assertRaises(ValueError):
            build_snapshot(
                [
                    {"hypothesis_id": "same", "b": 1, "selected": True},
                    {"hypothesis_id": "same", "b": 2, "selected": False},
                ],
                manifest_root=_manifest(),
                sequence=1,
            )
        with self.assertRaises(TypeError):
            build_snapshot(
                [{"hypothesis_id": "h", "b": 0.1, "selected": True}],
                manifest_root=_manifest(),
                sequence=1,
            )

    def test_tampered_threshold_score_and_selection_digest_fail(self):
        entries = [
            {"hypothesis_id": f"h{i}", "b": Fraction(i - 2, 5), "selected": i < 3}
            for i in range(8)
        ]
        snapshot, receipt = self._round_trip(entries)

        for mutation in (
            lambda x: x["prefix_lt_threshold"].update({"threshold": ["999", "1"]}),
            lambda x: x.update({"closure_score": ["999", "1"]}),
            lambda x: x.update({"selected_digest": _manifest("tampered")}),
            lambda x: x.update({"rejection_count": x["rejection_count"] + 1}),
        ):
            changed = deepcopy(receipt)
            mutation(changed)
            with self.assertRaises(ValueError):
                verify_score_receipt(
                    changed,
                    expected_snapshot_hash=snapshot.snapshot_hash,
                    expected_manifest_root=_manifest(),
                )

    def test_tampered_path_node_or_sibling_aggregate_fails(self):
        entries = [
            {"hypothesis_id": f"h{i}", "b": Fraction(i - 4, 7), "selected": i % 2 == 0}
            for i in range(12)
        ]
        snapshot, receipt = self._round_trip(entries)

        altered_node = deepcopy(receipt)
        altered_node["prefix_lt_threshold"]["path"][0]["selected"] = not (
            altered_node["prefix_lt_threshold"]["path"][0]["selected"]
        )
        with self.assertRaises(ValueError):
            verify_score_receipt(
                altered_node,
                expected_snapshot_hash=snapshot.snapshot_hash,
                expected_manifest_root=_manifest(),
            )

        altered_aggregate = deepcopy(receipt)
        ref = altered_aggregate["prefix_lt_threshold"]["path"][0]["left"]
        ref["aggregate"]["selected_count"] = 999
        with self.assertRaises(ValueError):
            verify_score_receipt(
                altered_aggregate,
                expected_snapshot_hash=snapshot.snapshot_hash,
                expected_manifest_root=_manifest(),
            )

    def test_wrong_snapshot_manifest_or_config_pin_fails(self):
        entries = [
            {"hypothesis_id": f"h{i}", "b": Fraction(i - 2, 9), "selected": i < 3}
            for i in range(6)
        ]
        snapshot, receipt = self._round_trip(entries)
        with self.assertRaises(ValueError):
            verify_score_receipt(
                receipt,
                expected_snapshot_hash=_manifest("other snapshot"),
                expected_manifest_root=_manifest(),
            )
        with self.assertRaises(ValueError):
            verify_score_receipt(
                receipt,
                expected_snapshot_hash=snapshot.snapshot_hash,
                expected_manifest_root=_manifest("other manifest"),
            )
        with self.assertRaises(ValueError):
            verify_score_receipt(
                receipt,
                expected_snapshot_hash=snapshot.snapshot_hash,
                expected_manifest_root=_manifest(),
                expected_weight_digest=_manifest("other weights"),
            )

    def test_receipt_parameters_cannot_be_rewritten_outside_snapshot(self):
        entries = [
            {"hypothesis_id": f"h{i}", "b": Fraction(i - 1, 5), "selected": i < 2}
            for i in range(5)
        ]
        snapshot, receipt = self._round_trip(entries)
        changed = deepcopy(receipt)
        changed["weight_digest"] = _manifest("different weights")
        with self.assertRaises(ValueError):
            verify_score_receipt(
                changed,
                expected_snapshot_hash=snapshot.snapshot_hash,
                expected_manifest_root=_manifest(),
            )

    def test_snapshot_chain_metadata_is_committed(self):
        entries = [
            {"hypothesis_id": f"h{i}", "b": Fraction(i - 1, 4), "selected": i < 2}
            for i in range(4)
        ]
        first, _ = self._round_trip(entries, sequence=1)
        second, _ = self._round_trip(
            entries, sequence=2, previous=first.snapshot_hash
        )
        self.assertNotEqual(first.snapshot_hash, second.snapshot_hash)
        self.assertEqual(second.previous_snapshot_hash, first.snapshot_hash)


if __name__ == "__main__":
    unittest.main()
