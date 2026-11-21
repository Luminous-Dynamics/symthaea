"""Known-answer and adversarial tests for SEMI-BENCH split manifests."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import semi_split_manifest_reference as split_validator  # noqa: E402
from semi_split_manifest_reference import SplitContractError  # noqa: E402

FIXTURE = HERE.parents[1] / "fixtures" / "split-manifest-known-answer-v1.json"


class SplitManifestReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_group_disjoint_known_answer_and_stable_digest(self) -> None:
        result = split_validator.validate_split_manifest(self.manifest)
        expected_path = HERE.parents[1] / "fixtures" / "split-manifest-known-answer-v1.expected.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(result, expected)
        reordered = copy.deepcopy(self.manifest)
        reordered["assignments"].reverse()
        self.assertEqual(split_validator.validate_split_manifest(reordered), result)

    def test_rejects_duplicate_sample_identity(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["assignments"][1]["sample_id"] = manifest["assignments"][0]["sample_id"]
        with self.assertRaisesRegex(SplitContractError, "duplicate"):
            split_validator.validate_split_manifest(manifest)

    def test_rejects_group_leakage(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["assignments"][-1]["group_id"] = manifest["assignments"][0]["group_id"]
        with self.assertRaisesRegex(SplitContractError, "group leakage"):
            split_validator.validate_split_manifest(manifest)

    def test_rejects_missing_required_split(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        for row in manifest["assignments"]:
            if row["split"] == "test":
                row["split"] = "train"
        with self.assertRaisesRegex(SplitContractError, "empty required split"):
            split_validator.validate_split_manifest(manifest)

    def test_forward_time_requires_strict_chronology(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["split_scheme"] = "forward_time"
        for index, row in enumerate(manifest["assignments"]):
            split_date = {"train": "2026-01-01", "validation": "2026-01-03", "test": "2026-01-05"}[row["split"]]
            row.pop("group_id")
            row["observed_at"] = f"{split_date}T0{index % 9}:00:00+00:00"
        result = split_validator.validate_split_manifest(manifest)
        self.assertTrue(result["guarantees"]["strict_forward_time"])
        self.assertEqual(result["claim_limits"]["temporal_independence"], "established-for-this-manifest")

    def test_rejects_overlapping_forward_time_windows(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["split_scheme"] = "forward_time"
        for row in manifest["assignments"]:
            row.pop("group_id")
            row["observed_at"] = {
                "train": "2026-01-01T00:00:00+00:00",
                "validation": "2026-01-02T00:00:00+00:00",
                "test": "2026-01-03T00:00:00+00:00",
            }[row["split"]]
        manifest["assignments"][0]["observed_at"] = "2026-01-04T00:00:00+00:00"
        with self.assertRaisesRegex(SplitContractError, "strictly chronological"):
            split_validator.validate_split_manifest(manifest)

    def test_rejects_timezone_free_timestamps(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["split_scheme"] = "forward_time"
        for row in manifest["assignments"]:
            row.pop("group_id")
            row["observed_at"] = "2026-01-01T12:00:00"
        with self.assertRaisesRegex(SplitContractError, "timezone offset"):
            split_validator.validate_split_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
