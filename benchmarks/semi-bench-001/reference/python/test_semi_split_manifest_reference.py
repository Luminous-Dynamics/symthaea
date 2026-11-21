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
        manifest["time_basis"] = "offset_aware"
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
        manifest["time_basis"] = "offset_aware"
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
        manifest["time_basis"] = "offset_aware"
        for row in manifest["assignments"]:
            row.pop("group_id")
            row["observed_at"] = "2026-01-01T12:00:00"
        with self.assertRaisesRegex(SplitContractError, "timezone offset"):
            split_validator.validate_split_manifest(manifest)


    def test_accepts_explicit_source_local_naive_clock_domain(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["split_scheme"] = "forward_time"
        manifest["time_basis"] = "source_local_naive"
        manifest["clock_domain_id"] = "single-secom-source-clock"
        for row in manifest["assignments"]:
            row.pop("group_id")
            row["observed_at"] = {
                "train": "2008-07-19T11:55:00",
                "validation": "2008-09-20T06:08:00",
                "test": "2008-10-02T20:54:00",
            }[row["split"]]
        result = split_validator.validate_split_manifest(manifest)
        self.assertTrue(result["guarantees"]["strict_forward_time"])
        self.assertEqual(result["time_basis"], "source_local_naive")
        self.assertEqual(result["clock_domain_id"], "single-secom-source-clock")
        self.assertEqual(
            result["claim_limits"]["temporal_independence"],
            "established-within-declared-source-clock-domain-only",
        )

    def test_rejects_naive_clock_without_declared_clock_domain(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["split_scheme"] = "forward_time"
        manifest["time_basis"] = "source_local_naive"
        for row in manifest["assignments"]:
            row.pop("group_id")
            row["observed_at"] = {
                "train": "2008-07-19T11:55:00",
                "validation": "2008-09-20T06:08:00",
                "test": "2008-10-02T20:54:00",
            }[row["split"]]
        with self.assertRaisesRegex(SplitContractError, "clock_domain_id"):
            split_validator.validate_split_manifest(manifest)

    def test_rejects_offset_aware_values_in_naive_clock_domain(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["split_scheme"] = "forward_time"
        manifest["time_basis"] = "source_local_naive"
        manifest["clock_domain_id"] = "single-secom-source-clock"
        for row in manifest["assignments"]:
            row.pop("group_id")
            row["observed_at"] = "2008-07-19T11:55:00+02:00"
        with self.assertRaisesRegex(SplitContractError, "must be timezone-free"):
            split_validator.validate_split_manifest(manifest)


    def test_frozen_secom_source_clock_manifest_has_expected_support(self) -> None:
        repo_root = HERE.parents[4]
        manifest_path = (
            repo_root / "docs" / "research" / "semi-bench-001" / "splits"
            / "uci-secom-mirror-0d27427a" / "chronological-60-20-20-v0.1.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = split_validator.validate_split_manifest(manifest)
        self.assertEqual(manifest["dataset_snapshot_sha256"],
                         "26c59ab70a7339d82ab4e4006a5230d1a2ffff48c2df3fc423a458f9ef152281")
        self.assertEqual(manifest["time_basis"], "source_local_naive")
        self.assertEqual(report["clock_domain_id"], "uci-secom-mirror-single-source-clock")
        self.assertEqual(report["split_counts"]["train"], {
            "samples": 940, "positive": 76, "negative": 864
        })
        self.assertEqual(report["split_counts"]["validation"], {
            "samples": 313, "positive": 11, "negative": 302
        })
        self.assertEqual(report["split_counts"]["test"], {
            "samples": 314, "positive": 17, "negative": 297
        })
        self.assertEqual(
            report["claim_limits"]["temporal_independence"],
            "established-within-declared-source-clock-domain-only",
        )


if __name__ == "__main__":
    unittest.main()
