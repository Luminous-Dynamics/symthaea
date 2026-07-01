#!/usr/bin/env python3
"""Regression tests for coding backend baseline checks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_coding_backend_baseline.py"


def base_report() -> dict[str, object]:
    return {
        "benchmark": "coding_backends_smoke",
        "feature_geodesic": True,
        "task_count": 31,
        "pass_rate": 1.0,
        "quality_pass_rate": 1.0,
        "mean_attempts_per_task": 1.0,
        "certificates_sheaf_incoherent": 0,
        "certificates_with_sheaf": 31,
        "broca_eval_gate_passed": True,
        "broca_selection_score": 1.0,
        "repair_success_rate": 0.0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "repair_prior_uses": 0,
        "repair_prior_label_count": 0,
        "repair_memory_hits": 0,
        "repair_memory_successes": 0,
        "repair_memory_success_rate": 0.0,
        "geodesic_rejection_shadow_hits": 0,
        "geodesic_rejection_shadow_true_positives": 0,
        "geodesic_rejection_shadow_false_positives": 0,
        "hard_geodesic_rejections": 0,
        "category_pass_rates": {
            "linear": {"task_count": 1, "accepted_count": 1, "pass_rate": 1.0}
        },
    }


def base_baseline() -> dict[str, object]:
    return {
        "benchmark": "coding_backends_smoke",
        "require_feature_geodesic": True,
        "min_task_count": 31,
        "min_pass_rate": 0.95,
        "min_quality_pass_rate": 0.95,
        "max_mean_attempts_per_task": 1.2,
        "max_certificates_sheaf_incoherent": 0,
        "min_certificates_with_sheaf": 31,
        "require_broca_eval_gate_passed": True,
        "min_broca_selection_score": 0.95,
        "min_repair_success_rate": 0.0,
        "max_repair_attempts": 0,
        "max_geodesic_rejection_shadow_false_positives": 0,
        "max_hard_geodesic_rejections": 0,
        "min_category_pass_rates": {"linear": 0.9},
    }


class CodingBackendBaselineCheckerTests(unittest.TestCase):
    def run_checker(
        self, report: dict[str, object], baseline: dict[str, object]
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            baseline_path = root / "baseline.json"
            report_path.write_text(json.dumps(report))
            baseline_path.write_text(json.dumps(baseline))
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--baseline",
                    str(baseline_path),
                    "--report",
                    str(report_path),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_accepts_report_matching_geodesic_fast_fail_contract(self) -> None:
        result = self.run_checker(base_report(), base_baseline())

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("hard_geodesic_rejections=0", result.stdout)

    def test_rejects_geodesic_shadow_false_positive_regression(self) -> None:
        report = base_report()
        report["geodesic_rejection_shadow_false_positives"] = 1

        result = self.run_checker(report, base_baseline())

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("geodesic_rejection_shadow_false_positives", result.stderr)

    def test_rejects_unexpected_hard_geodesic_rejections(self) -> None:
        report = base_report()
        report["hard_geodesic_rejections"] = 1

        result = self.run_checker(report, base_baseline())

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hard_geodesic_rejections", result.stderr)


if __name__ == "__main__":
    unittest.main()
