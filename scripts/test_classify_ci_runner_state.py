from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.classify_ci_runner_state import (
    COMPLETED_SKIPPED,
    COMPLETED_SUCCESS,
    SUMMARY_STALE,
    TIMEOUT_INCONSISTENT,
    FixtureError,
    classify_fixture,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "ci_runner_state"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class RunnerStateClassifierTests(unittest.TestCase):
    def test_stale_summary_is_detected_from_terminal_steps(self) -> None:
        result = classify_fixture(load_fixture("stale_summary.json"))
        self.assertEqual(result["jobs"][0]["classification"], SUMMARY_STALE)
        self.assertEqual(result["runner_plane_anomaly_count"], 1)
        self.assertFalse(result["required_checks_terminal_success"])

    def test_timeout_inconsistency_is_distinct_from_running(self) -> None:
        result = classify_fixture(load_fixture("timeout_inconsistent.json"))
        self.assertEqual(result["jobs"][0]["classification"], TIMEOUT_INCONSISTENT)
        self.assertFalse(result["required_checks_terminal_success"])

    def test_run_and_runner_provenance_is_preserved(self) -> None:
        result = classify_fixture(load_fixture("stale_summary.json"))
        self.assertEqual(result["run"]["workflow_id"], 243210908)
        self.assertEqual(result["run"]["workflow_name"], "CI")
        self.assertEqual(result["run"]["run_attempt"], 1)
        self.assertEqual(result["jobs"][0]["runner_class"], "ubuntu-latest")
        self.assertEqual(result["jobs"][0]["timeout_minutes"], 75)
        self.assertEqual(
            result["jobs"][0]["started_at"], "2026-09-15T20:17:36+00:00"
        )

    def test_all_required_terminal_success_is_narrow_runner_statement(self) -> None:
        fixture = load_fixture("stale_summary.json")
        fixture["jobs"] = [
            {
                "id": 1,
                "name": "focused exact-head qualification",
                "required": True,
                "runner_class": "ubuntu-slim",
                "status": "completed",
                "conclusion": "success",
                "steps": [
                    {
                        "name": "Complete job",
                        "status": "completed",
                        "conclusion": "success",
                    }
                ],
            },
            {
                "id": 2,
                "name": "optional diagnostic",
                "required": False,
                "runner_class": "ubuntu-slim",
                "status": "completed",
                "conclusion": "failure",
                "steps": [
                    {
                        "name": "diagnostic",
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ],
            },
        ]
        result = classify_fixture(fixture)
        self.assertTrue(result["required_checks_terminal_success"])
        self.assertEqual(result["jobs"][0]["classification"], COMPLETED_SUCCESS)
        self.assertIn("runner evidence only", result["claim_boundary"])

    def test_skipped_required_job_never_counts_as_success(self) -> None:
        fixture = load_fixture("stale_summary.json")
        fixture["jobs"] = [
            {
                "id": 3,
                "name": "required but skipped",
                "required": True,
                "runner_class": "ubuntu-slim",
                "status": "completed",
                "conclusion": "skipped",
                "steps": [],
            }
        ]
        result = classify_fixture(fixture)
        self.assertEqual(result["jobs"][0]["classification"], COMPLETED_SKIPPED)
        self.assertFalse(result["required_checks_terminal_success"])

    def test_fixture_requires_explicit_required_flag(self) -> None:
        fixture = load_fixture("stale_summary.json")
        fixture["jobs"][0].pop("required")
        with self.assertRaises(FixtureError):
            classify_fixture(fixture)

    def test_fixture_requires_runner_class(self) -> None:
        fixture = load_fixture("stale_summary.json")
        fixture["jobs"][0].pop("runner_class")
        with self.assertRaises(FixtureError):
            classify_fixture(fixture)

    def test_naive_timestamp_is_rejected(self) -> None:
        fixture = load_fixture("stale_summary.json")
        fixture["observed_at"] = "2026-09-16T20:25:00"
        with self.assertRaises(FixtureError):
            classify_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
