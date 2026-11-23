"""Known-answer and adversarial tests for contrastive-pair lineage validation."""
from __future__ import annotations
import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import semi_pair_manifest_reference as validator  # noqa: E402
from semi_pair_manifest_reference import PairManifestError  # noqa: E402

FIXTURE = HERE.parents[1] / "fixtures" / "contrastive-pair-manifest-known-answer-v1.json"


class PairManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_known_answer(self):
        result = validator.validate_pair_manifest(self.manifest)
        expected = json.loads((HERE.parents[1] / "fixtures" / "contrastive-pair-manifest-known-answer-v1.expected.json").read_text())
        self.assertEqual(result, expected)

    def test_rejects_cross_split_pair(self):
        m = copy.deepcopy(self.manifest)
        m["pairs"][0]["right_sample_id"] = "V01"
        with self.assertRaisesRegex(PairManifestError, "cross-split"):
            validator.validate_pair_manifest(m)

    def test_rejects_pair_scope_mismatch(self):
        m = copy.deepcopy(self.manifest)
        m["pairs"][0]["pair_scope"] = "test"
        with self.assertRaisesRegex(PairManifestError, "pair_scope mismatch"):
            validator.validate_pair_manifest(m)

    def test_rejects_self_pair(self):
        m = copy.deepcopy(self.manifest)
        m["pairs"][0]["right_sample_id"] = m["pairs"][0]["left_sample_id"]
        with self.assertRaisesRegex(PairManifestError, "self-pair"):
            validator.validate_pair_manifest(m)

    def test_rejects_duplicate_pair_id(self):
        m = copy.deepcopy(self.manifest)
        m["pairs"][1]["pair_id"] = m["pairs"][0]["pair_id"]
        with self.assertRaisesRegex(PairManifestError, "duplicate pair identity"):
            validator.validate_pair_manifest(m)

    def test_rejects_duplicate_unordered_edge(self):
        m = copy.deepcopy(self.manifest)
        duplicate = copy.deepcopy(m["pairs"][0])
        duplicate["pair_id"] = "P4"
        duplicate["left_sample_id"], duplicate["right_sample_id"] = duplicate["right_sample_id"], duplicate["left_sample_id"]
        m["pairs"].append(duplicate)
        with self.assertRaisesRegex(PairManifestError, "duplicate unordered"):
            validator.validate_pair_manifest(m)

    def test_rejects_orphan_endpoint(self):
        m = copy.deepcopy(self.manifest)
        m["pairs"][0]["left_sample_id"] = "MISSING"
        with self.assertRaisesRegex(PairManifestError, "orphan"):
            validator.validate_pair_manifest(m)

    def test_rejects_same_lot_without_matching_group(self):
        m = copy.deepcopy(self.manifest)
        m["sample_assignments"][1]["group_id"] = "LOT-X"
        with self.assertRaisesRegex(PairManifestError, "matching group identity"):
            validator.validate_pair_manifest(m)

    def test_rejects_time_pair_outside_embargo(self):
        m = copy.deepcopy(self.manifest)
        m["sample_assignments"][-1]["observed_at"] = "2008-07-19T18:00:00"
        with self.assertRaisesRegex(PairManifestError, "exceeds embargo"):
            validator.validate_pair_manifest(m)

    def test_rejects_missing_time_evidence(self):
        m = copy.deepcopy(self.manifest)
        m["sample_assignments"][-1].pop("observed_at")
        with self.assertRaisesRegex(PairManifestError, "lacks timestamps"):
            validator.validate_pair_manifest(m)

    def test_rejects_proximity_reason_without_proximity_basis(self):
        m = copy.deepcopy(self.manifest)
        m["pairs"][1]["proximity_reason"] = "same_lot"
        with self.assertRaisesRegex(PairManifestError, "without production_proximity"):
            validator.validate_pair_manifest(m)


if __name__ == "__main__":
    unittest.main()
