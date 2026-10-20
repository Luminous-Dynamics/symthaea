"""Regression tests for the fail-closed PyPhi cross-validation scaffold."""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.pyphi_crossvalidation import (  # noqa: E402
    PyPhiValidator,
    TopologySpec,
    ValidationReport,
    ValidationResult,
)


class FailClosedPyPhiValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = PyPhiValidator()
        self.topology = TopologySpec(
            name="line_2",
            n_nodes=2,
            edges=[(0, 1)],
            description="two-node smoke fixture",
        )

    def test_missing_rust_adapter_raises_instead_of_synthesizing_a_score(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            self.validator.compute_hdc_phi(self.topology)
        self.assertFalse(hasattr(self.validator, "_estimate_hdc_phi"))

    def test_topology_result_retains_missing_values_and_failure_reason(self) -> None:
        result = self.validator.validate_topology(self.topology)
        self.assertIsNone(result.hdc_phi)
        self.assertIsNone(result.pyphi_phi)
        self.assertIsNotNone(result.error_message)
        self.assertIn("blocked", result.error_message or "")

    def test_all_failed_run_is_not_qualified(self) -> None:
        report = self.validator.run_validation(max_nodes=2)
        self.assertEqual(report.n_validated, 0)
        self.assertEqual(report.n_failed, 1)
        self.assertIsNone(report.correlation)
        self.assertIsNone(report.mean_absolute_error)
        self.assertIsNone(report.max_absolute_error)
        self.assertIsNone(report.agreement_within_10_percent)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "nested" / "report.md"
            self.validator.generate_report(report, destination)
            text = destination.read_text(encoding="utf-8")
        self.assertIn("Status: NOT QUALIFIED", text)
        self.assertIn("N/A", text)
        self.assertNotIn("Status: PASS", text)
        self.assertNotIn("0.0000", text)

    def test_constant_score_pairs_cannot_create_a_correlation_pass(self) -> None:
        report = ValidationReport(
            results=[
                ValidationResult(
                    topology_name=f"constant_{index}",
                    n_nodes=2,
                    hdc_phi=0.25,
                    pyphi_phi=0.25,
                )
                for index in range(3)
            ]
        )
        self.validator._compute_report_metrics(report)
        self.assertIsNone(report.correlation)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "report.md"
            self.validator.generate_report(report, destination)
            text = destination.read_text(encoding="utf-8")
        self.assertIn("Status: NOT QUALIFIED", text)

    def test_pyphi_version_and_iit_formalism_are_explicit(self) -> None:
        source = inspect.getsource(PyPhiValidator.compute_pyphi_phi)
        self.assertIn('installed_version != "2.0.0rc4"', source)
        self.assertIn('formalism="IIT_3_0"', source)


if __name__ == "__main__":
    unittest.main()
