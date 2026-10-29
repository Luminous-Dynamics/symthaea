# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the fail-closed PyPhi comparison scaffold."""

import subprocess
import unittest
from unittest.mock import patch

import numpy as np

try:
    from validation import pyphi_crossvalidation as subject
except ImportError:
    import pyphi_crossvalidation as subject


class ComputeHdcPhiTests(unittest.TestCase):
    def setUp(self):
        self.validator = subject.PyPhiValidator()
        self.topology = subject.TopologySpec(
            name="line_3",
            n_nodes=3,
            edges=[(0, 1), (1, 2)],
        )

    @patch.object(subject.subprocess, "run")
    def test_valid_json_returns_measured_values(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout='{"mean":0.25,"std":0.02}', stderr=""
        )
        mean, std, elapsed = self.validator.compute_hdc_phi(self.topology, n_samples=2)
        self.assertEqual(mean, 0.25)
        self.assertEqual(std, 0.02)
        self.assertGreaterEqual(elapsed, 0.0)
        run.assert_called_once()

    @patch.object(subject.subprocess, "run")
    def test_nonzero_exit_fails_instead_of_substituting_estimate(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["cargo"], returncode=17, stdout="", stderr="adapter failure"
        )
        with self.assertRaisesRegex(RuntimeError, "exited 17"):
            self.validator.compute_hdc_phi(self.topology)
        self.assertFalse(hasattr(self.validator, "_estimate_hdc_phi"))

    @patch.object(subject.subprocess, "run")
    def test_timeout_fails_closed(self, run):
        run.side_effect = subprocess.TimeoutExpired(cmd=["cargo"], timeout=60)
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            self.validator.compute_hdc_phi(self.topology)

    @patch.object(subject.subprocess, "run")
    def test_missing_executable_fails_closed(self, run):
        run.side_effect = FileNotFoundError("cargo not found")
        with self.assertRaisesRegex(RuntimeError, "executable not found"):
            self.validator.compute_hdc_phi(self.topology)

    @patch.object(subject.subprocess, "run")
    def test_malformed_json_fails_closed(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout="not-json", stderr=""
        )
        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            self.validator.compute_hdc_phi(self.topology)

    @patch.object(subject.subprocess, "run")
    def test_non_object_json_fails_closed(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout="[0.2,0.1]", stderr=""
        )
        with self.assertRaisesRegex(RuntimeError, "must be an object"):
            self.validator.compute_hdc_phi(self.topology)

    @patch.object(subject.subprocess, "run")
    def test_missing_required_field_fails_closed(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout='{"mean":0.25}', stderr=""
        )
        with self.assertRaisesRegex(RuntimeError, "field 'std'"):
            self.validator.compute_hdc_phi(self.topology)

    @patch.object(subject.subprocess, "run")
    def test_non_finite_value_fails_closed(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout='{"mean":NaN,"std":0.02}', stderr=""
        )
        with self.assertRaisesRegex(RuntimeError, "must be finite"):
            self.validator.compute_hdc_phi(self.topology)

    @patch.object(subject.subprocess, "run")
    def test_boolean_is_not_accepted_as_numeric(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout='{"mean":true,"std":0.02}', stderr=""
        )
        with self.assertRaisesRegex(RuntimeError, "must be numeric"):
            self.validator.compute_hdc_phi(self.topology)

    @patch.object(subject.subprocess, "run")
    def test_negative_standard_deviation_fails_closed(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["cargo"], returncode=0, stdout='{"mean":0.25,"std":-0.02}', stderr=""
        )
        with self.assertRaisesRegex(RuntimeError, "non-negative"):
            self.validator.compute_hdc_phi(self.topology)


class QualificationGateTests(unittest.TestCase):
    def setUp(self):
        self.validator = subject.PyPhiValidator()
        self.topology = subject.TopologySpec(
            name="line_3", n_nodes=3, edges=[(0, 1), (1, 2)]
        )

    def test_hdc_failure_leaves_measurement_missing(self):
        with patch.object(
            self.validator, "compute_hdc_phi", side_effect=RuntimeError("broken adapter")
        ):
            result = self.validator.validate_topology(self.topology)
        self.assertIsNone(result.hdc_phi)
        self.assertIsNone(result.pyphi_phi)
        self.assertEqual(result.status, "failed")
        self.assertIn("broken adapter", result.error_message)

    def test_missing_pyphi_is_inconclusive_not_zero(self):
        with patch.object(subject, "PYPHI_AVAILABLE", False), patch.object(
            self.validator, "compute_hdc_phi", return_value=(0.25, 0.01, 0.001)
        ):
            result = self.validator.validate_topology(self.topology)
        self.assertEqual(result.hdc_phi, 0.25)
        self.assertIsNone(result.pyphi_phi)
        self.assertEqual(result.status, "inconclusive")

    def test_constant_series_suppresses_undefined_correlation(self):
        report = subject.ValidationReport(results=[
            subject.ValidationResult("a", 2, 0.5, 1.0),
            subject.ValidationResult("b", 2, 0.5, 2.0),
            subject.ValidationResult("c", 2, 0.5, 3.0),
        ])
        self.validator._compute_report_metrics(report)
        self.assertIsNone(report.correlation)
        self.assertEqual(report.qualification_status, "NOT_QUALIFIED")
        self.assertTrue(any("constant" in reason for reason in report.qualification_reasons))

    def test_high_correlation_does_not_qualify_mismatched_models(self):
        report = subject.ValidationReport(results=[
            subject.ValidationResult("a", 2, 1.0, 2.0),
            subject.ValidationResult("b", 2, 2.0, 4.0),
            subject.ValidationResult("c", 2, 3.0, 6.0),
        ])
        self.validator._compute_report_metrics(report)
        self.assertAlmostEqual(report.correlation, 1.0)
        self.assertEqual(report.qualification_status, "NOT_QUALIFIED")
        self.assertTrue(report.qualification_reasons)


if __name__ == "__main__":
    unittest.main()
