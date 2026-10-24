"""Adversarial tests for the exact-head CI report generator."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from write_socioeconomic_run_bundle_ci_report import (
    ReportFailure,
    build_attestation_predicate,
    build_report,
)

SOURCE_PATHS = (
    "scripts/validate_socioeconomic_run_bundle.py",
    "scripts/test_validate_socioeconomic_run_bundle.py",
    "scripts/write_socioeconomic_run_bundle_ci_report.py",
    "scripts/test_write_socioeconomic_run_bundle_ci_report.py",
    "docs/research/finance/SYM-FIN-SYS-016_run_bundle_manifest_v1.schema.json",
    "docs/research/finance/SYM-FIN-SYS-016_validator_execution_record_v1.schema.json",
    "docs/research/finance/SYM-FIN-SYS-016_validator_attestation_predicate_v1.schema.json",
    "scripts/socioeconomic_run_receipt_requirements.txt",
    ".github/workflows/socioeconomic-run-bundle.yml",
)
TEST_LOG_PATH = "evidence/test-output.txt"


class ExecutionReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for relative in SOURCE_PATHS:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative == "scripts/test_validate_socioeconomic_run_bundle.py":
                source = "".join(f"def test_fake_{index}():\n    assert True\n" for index in range(14))
            elif relative == "scripts/test_write_socioeconomic_run_bundle_ci_report.py":
                source = "".join(f"def test_fake_{index}():\n    assert True\n" for index in range(11))
            else:
                source = "def test_example():\n    assert True\n"
            path.write_text(source, encoding="utf-8")
        log = self.root / TEST_LOG_PATH
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("Ran 14 tests in 0.001s\n\nOK\nRan 11 tests in 0.001s\n\nOK\n", encoding="utf-8")

        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=self.root, check=True)
        self.revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.env = {
            "GITHUB_SHA": self.revision,
            "CIVOS_SOURCE_REVISION": self.revision,
            "GITHUB_REPOSITORY": "example/repo",
            "GITHUB_WORKFLOW": "run-bundle-tests",
            "GITHUB_WORKFLOW_REF": "example/repo/.github/workflows/run-bundle-tests.yml@refs/heads/main",
            "GITHUB_WORKFLOW_SHA": self.revision,
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_JOB": "exact-head-tests",
            "RUNNER_OS": "Linux",
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_report_binds_exact_checkout_and_input_bytes(self) -> None:
        report = build_report(self.root, self.env, TEST_LOG_PATH)
        self.assertEqual(report["source_revision"], self.revision)
        self.assertEqual(report["event_sha"], self.revision)
        self.assertEqual(report["workflow_sha"], self.revision)
        self.assertEqual(report["workflow_ref"], self.env["GITHUB_WORKFLOW_REF"])
        self.assertEqual(report["result"], "passed")
        self.assertEqual(report["claim_ceiling"], "synthetic_deterministic_only")
        self.assertEqual(report["scope"], "run_bundle_validator_and_execution_record_unit_tests")
        self.assertEqual(report["authored_test_function_count"], 25)
        self.assertEqual(report["executed_test_count"], 25)
        self.assertEqual(report["executed_test_counts_by_suite"], [14, 11])
        self.assertEqual(len(report["artifacts"]), len(SOURCE_PATHS) + 1)
        self.assertTrue(all(item["digest"].startswith("sha256:") for item in report["artifacts"]))

    def test_attestation_predicate_binds_exact_report_bytes(self) -> None:
        report = build_report(self.root, self.env, TEST_LOG_PATH)
        report_bytes = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
        predicate = build_attestation_predicate(report, report_bytes)
        self.assertEqual(
            predicate["execution_record_digest"],
            "sha256:" + hashlib.sha256(report_bytes).hexdigest(),
        )
        self.assertEqual(predicate["source_revision"], self.revision)
        self.assertEqual(predicate["result"], "passed")
        self.assertFalse(predicate["authority_boundary"]["can_settle_real_assets"])

    def test_attestation_predicate_is_a_distinct_schema_with_no_authority(self) -> None:
        report = build_report(self.root, self.env, TEST_LOG_PATH)
        report_bytes = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
        predicate = build_attestation_predicate(report, report_bytes)
        self.assertEqual(
            predicate["schema"],
            "civos.socioeconomic_interoperability.validator_attestation_predicate.v1",
        )
        self.assertNotEqual(predicate["schema"], report["schema"])
        self.assertFalse(predicate["authority_boundary"]["proves_simulation_truth"])

    def test_missing_run_metadata_is_rejected(self) -> None:
        broken = dict(self.env)
        del broken["GITHUB_RUN_ID"]
        with self.assertRaisesRegex(ReportFailure, "GITHUB_RUN_ID"):
            build_report(self.root, broken, TEST_LOG_PATH)

    def test_wrong_checkout_sha_is_rejected(self) -> None:
        broken = dict(self.env)
        broken["CIVOS_SOURCE_REVISION"] = "0" * 40
        with self.assertRaisesRegex(ReportFailure, "does not match CIVOS_SOURCE_REVISION"):
            build_report(self.root, broken, TEST_LOG_PATH)

    def test_test_inventory_shrinkage_is_rejected(self) -> None:
        source = self.root / "scripts" / "test_validate_socioeconomic_run_bundle.py"
        source.write_text("def test_only_one():\n    assert True\n", encoding="utf-8")
        with self.assertRaisesRegex(ReportFailure, "versioned test inventory changed"):
            build_report(self.root, self.env, TEST_LOG_PATH)

    def test_test_log_must_report_each_suite_with_exact_count(self) -> None:
        (self.root / TEST_LOG_PATH).write_text(
            "Ran 99 tests in 0.001s\n\nOK\nRan 1 test in 0.001s\n\nOK\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ReportFailure, "test log summaries do not match"):
            build_report(self.root, self.env, TEST_LOG_PATH)

    def test_symlinked_source_artifact_is_rejected(self) -> None:
        target = self.root / "scripts" / "validate_socioeconomic_run_bundle.py"
        target.unlink()
        target.symlink_to(self.root / "scripts" / "test_validate_socioeconomic_run_bundle.py")
        with self.assertRaisesRegex(ReportFailure, "symlink components are forbidden"):
            build_report(self.root, self.env, TEST_LOG_PATH)

    def test_empty_test_log_is_rejected(self) -> None:
        (self.root / TEST_LOG_PATH).write_bytes(b"")
        with self.assertRaisesRegex(ReportFailure, "test log is empty"):
            build_report(self.root, self.env, TEST_LOG_PATH)

    def test_unsafe_log_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ReportFailure, "unsafe evidence path"):
            build_report(self.root, self.env, "../test-output.txt")

    def test_authority_boundary_is_always_non_authoritative(self) -> None:
        report = build_report(self.root, self.env, TEST_LOG_PATH)
        self.assertFalse(report["authority_boundary"]["proves_simulation_truth"])
        self.assertFalse(report["authority_boundary"]["can_authorize_real_actions"])
        self.assertFalse(report["authority_boundary"]["can_settle_real_assets"])


if __name__ == "__main__":
    unittest.main()
