"""Regression tests for the fail-closed PyPhi cross-validation scaffold."""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validation.pyphi_crossvalidation import (  # noqa: E402
    PyPhiValidator,
    TopologySpec,
    main,
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
        self.assertIsNone(report.spearman_correlation)
        self.assertIsNone(report.root_mean_square_error)
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
            ],
            n_validated=3,
            n_failed=0,
        )
        self.validator._compute_report_metrics(report)
        self.assertIsNone(report.correlation)
        # Guard against the old split-brain behavior where the CLI could exit 0
        # even though the report correctly said NOT QUALIFIED.
        self.assertFalse(self.validator.is_qualified(report))

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "report.md"
            self.validator.generate_report(report, destination)
            text = destination.read_text(encoding="utf-8")
        self.assertIn("Status: NOT QUALIFIED", text)


    def test_reference_model_digest_is_stable_and_covers_graph_dynamics(self) -> None:
        adjacency_a, tpm_a, digest_a = self.validator.build_reference_model(self.topology)
        adjacency_b, tpm_b, digest_b = self.validator.build_reference_model(self.topology)
        self.assertEqual(digest_a, digest_b)
        self.assertEqual(adjacency_a.tolist(), adjacency_b.tolist())
        self.assertEqual(tpm_a.tolist(), tpm_b.tolist())
        self.assertEqual(tpm_a.shape, (4, 2))
        self.assertTrue((tpm_a >= 0.0).all())
        self.assertTrue((tpm_a <= 1.0).all())

        no_edges = TopologySpec(
            name="isolated_pair",
            n_nodes=2,
            edges=[],
            description="two self-memory nodes",
        )
        adjacency_isolated, _, digest_isolated = self.validator.build_reference_model(no_edges)
        self.assertNotEqual(digest_a, digest_isolated)
        self.assertEqual(adjacency_isolated[0, 0], 1)
        self.assertEqual(adjacency_isolated[1, 1], 1)

    def test_reference_model_rejects_invalid_edges_and_excessive_size(self) -> None:
        invalid_edge = TopologySpec(
            name="invalid",
            n_nodes=2,
            edges=[(0, 2)],
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            self.validator.build_reference_model(invalid_edge)

        too_large = TopologySpec(
            name="too_large",
            n_nodes=9,
            edges=[],
        )
        with self.assertRaisesRegex(ValueError, "capped at 8"):
            self.validator.build_reference_model(too_large)


    def test_failed_case_prevents_pass_even_when_remaining_pairs_correlate(self) -> None:
        results = [
            ValidationResult(
                topology_name=f"valid_{index}",
                n_nodes=2,
                hdc_phi=value,
                pyphi_phi=value,
            )
            for index, value in enumerate((0.1, 0.2, 0.3))
        ]
        results.append(
            ValidationResult(
                topology_name="failed_case",
                n_nodes=2,
                hdc_phi=None,
                pyphi_phi=None,
                error_message="reference unavailable",
            )
        )
        report = ValidationReport(results=results, n_validated=3, n_failed=1)
        self.validator._compute_report_metrics(report)
        self.assertAlmostEqual(report.correlation or 0.0, 1.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "report.md"
            self.validator.generate_report(report, destination)
            text = destination.read_text(encoding="utf-8")
        self.assertIn("Status: NOT QUALIFIED", text)
        self.assertIn("FAILED / NOT RUN", text)

    def test_nonfinite_measurements_do_not_count_as_valid_pairs(self) -> None:
        report = ValidationReport(
            results=[
                ValidationResult("finite_1", 2, 0.1, 0.1),
                ValidationResult("finite_2", 2, 0.2, 0.2),
                ValidationResult("infinite", 2, float("inf"), 0.3),
            ]
        )
        self.validator._compute_report_metrics(report)
        self.assertIsNone(report.correlation)

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "report.md"
            self.validator.generate_report(report, destination)
            text = destination.read_text(encoding="utf-8")
        self.assertIn("Status: NOT QUALIFIED", text)
        self.assertIn("N/A", text)

    def test_perfect_correlated_pairs_can_qualify_only_when_all_gates_hold(self) -> None:
        values = (0.1, 0.2, 0.3)
        report = ValidationReport(
            results=[
                ValidationResult(
                    topology_name=f"matched_{index}",
                    n_nodes=2,
                    hdc_phi=value,
                    pyphi_phi=value,
                    reference_model_digest=f"model-{index}",
                    symthaea_model_digest=f"model-{index}",
                    reference_prior_digest=f"prior-{index}",
                    symthaea_prior_digest=f"prior-{index}",
                )
                for index, value in enumerate(values)
            ],
            correlation=1.0,
            spearman_correlation=1.0,
            mean_absolute_error=0.0,
            root_mean_square_error=0.0,
            max_absolute_error=0.0,
            agreement_within_10_percent=100.0,
            pyphi_version="2.0.0",
            shared_model_verified=True,
            n_validated=3,
            n_failed=0,
        )
        report.shared_model_verified = False
        self.assertFalse(self.validator.is_qualified(report))
        report.shared_model_verified = True
        self.assertTrue(self.validator.is_qualified(report))

        # Matching transition rules are insufficient if state priors differ.
        report.results[0].symthaea_prior_digest = "different-prior"
        self.assertFalse(self.validator.is_qualified(report))
        report.results[0].symthaea_prior_digest = "prior-0"
        self.assertTrue(self.validator.is_qualified(report))

        # A release candidate must never become publication-qualified.
        report.pyphi_version = "2.0.0rc4"
        self.assertFalse(self.validator.is_qualified(report))
        report.pyphi_version = "2.0.0"

        # A mismatched per-case model digest also blocks qualification.
        report.results[0].symthaea_model_digest = "different-model"
        self.assertFalse(self.validator.is_qualified(report))

    def test_cli_exits_nonzero_when_no_shared_adapter_can_produce_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "report.md"
            captured = StringIO()
            with patch(
                "sys.argv",
                ["pyphi_crossvalidation.py", "--max-nodes", "2", "--output", str(destination)],
            ), redirect_stdout(captured):
                exit_code = main()

        self.assertEqual(exit_code, 2)
        self.assertIn("Status: NOT QUALIFIED", captured.getvalue())
        self.assertTrue(destination.exists())
        self.assertIn("Status: NOT QUALIFIED", destination.read_text(encoding="utf-8"))

    def test_pyphi_version_and_iit_formalism_are_explicit(self) -> None:
        source = inspect.getsource(PyPhiValidator.compute_pyphi_phi)
        self.assertIn('installed_version != "2.0.0rc4"', source)
        self.assertIn("pyphi.Substrate(", source)
        self.assertIn('formalism="IIT_3_0"', source)


if __name__ == "__main__":
    unittest.main()
