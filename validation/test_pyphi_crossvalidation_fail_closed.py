# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for fail-closed Phi cross-validation qualification."""

import tempfile
import unittest
from pathlib import Path

from validation.pyphi_crossvalidation import (
    PyPhiValidator,
    TopologySpec,
    ValidationReport,
    ValidationResult,
)


class FailClosedQualificationTests(unittest.TestCase):
    def setUp(self):
        self.validator = PyPhiValidator()

    def test_missing_rust_adapter_produces_no_numeric_measurement(self):
        topology = TopologySpec("ring_3", 3, [(0, 1), (1, 2), (2, 0)])
        with self.assertRaisesRegex(RuntimeError, "NOT QUALIFIED"):
            self.validator.compute_hdc_phi(topology)
        self.assertFalse(hasattr(self.validator, "_estimate_hdc_phi"))

    def test_all_failed_cases_cannot_qualify(self):
        report = ValidationReport(results=[
            ValidationResult("ring_3", 3, 0.0, None, error_message="adapter unavailable"),
            ValidationResult("line_3", 3, 0.0, None, error_message="adapter unavailable"),
        ], n_failed=2)
        self.validator._compute_report_metrics(report)
        self.assertEqual(report.qualification_status, "NOT_QUALIFIED")
        self.assertEqual(report.n_validated, 0)

    def test_constant_score_does_not_yield_correlation_qualification(self):
        report = ValidationReport(results=[
            ValidationResult(f"case_{i}", 3, 0.5, float(i))
            for i in range(5)
        ])
        self.validator._compute_report_metrics(report)
        self.assertEqual(report.qualification_status, "NOT_QUALIFIED")
        self.assertEqual(report.correlation, 0.0)

    def test_report_never_claims_pass_from_correlation_threshold(self):
        report = ValidationReport(correlation=0.99, n_validated=5)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            self.validator.generate_report(report, output)
            rendered = output.read_text()
        self.assertIn("NOT_QUALIFIED", rendered)
        self.assertNotIn("VALIDATION PASSED", rendered)
        self.assertNotIn("**PASS**", rendered)


if __name__ == "__main__":
    unittest.main()
