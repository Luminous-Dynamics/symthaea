#!/usr/bin/env python3
"""Fail-closed policy tests for the NumPy eigen reference comparator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qc_eigen_reference_compare as qc  # noqa: E402


def sample_case() -> dict:
    root_half = 2.0**-0.5
    return {
        "case_id": "symmetric_2x2",
        "dimension": 2,
        "matrix_row_major": [1.0, 0.5, 0.5, 1.0],
        "native_result": {
            "status": "converged",
            "converged": True,
            "stopping_reason": "converged",
            "iterations": 1,
            "max_off_diagonal": 0.0,
            "max_relative_eigenpair_residual": 0.0,
            "max_orthogonality_residual": 0.0,
            "eigenvalues": [1.5, 0.5],
            "eigenvectors_row_major": [
                root_half,
                -root_half,
                root_half,
                root_half,
            ],
        },
    }


def sample_report() -> dict:
    return {
        "schema_version": 1,
        "producer": "symthaea-quantum-chemistry",
        "producer_version": "test",
        "source_revision": "0123456789abcdef0123456789abcdef01234567",
        "source_tree_sha": "89abcdef0123456789abcdef0123456789abcdef",
        "worktree_clean": True,
        "cases": [
            {**sample_case(), "case_id": case_id}
            for case_id in qc.EXPECTED_CASE_IDS
        ],
    }


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], returncode, stdout, "")


class QcEigenReferenceCompareTests(unittest.TestCase):
    def test_matching_spectrum_and_residuals_pass(self):
        record = qc.compare_case(sample_case())
        self.assertEqual(record["comparison_status"], "passed")
        self.assertLessEqual(
            record["max_relative_eigenvalue_error"],
            qc.MAX_RELATIVE_EIGENVALUE_ERROR,
        )
        self.assertLessEqual(
            record["max_relative_reference_eigenpair_residual"],
            qc.MAX_RELATIVE_EIGENPAIR_RESIDUAL,
        )

    def test_native_nonconvergence_cannot_be_upgraded_by_numpy(self):
        case = sample_case()
        case["native_result"]["status"] = "non_converged"
        case["native_result"]["converged"] = False
        record = qc.compare_case(case)
        self.assertEqual(record["comparison_status"], "not_comparable")
        self.assertIn("cannot upgrade it", record["failure_reason"])

    def test_wrong_spectrum_fails_even_if_native_report_claims_zero_residual(self):
        case = sample_case()
        case["native_result"]["eigenvalues"] = [1.6, 0.4]
        record = qc.compare_case(case)
        self.assertEqual(record["comparison_status"], "failed_numerical_contract")
        self.assertGreater(record["max_relative_eigenvalue_error"], 1e-10)
        self.assertGreater(
            record["max_relative_reference_eigenpair_residual"],
            qc.MAX_RELATIVE_EIGENPAIR_RESIDUAL,
        )

    def test_malformed_matrix_is_retained_as_not_comparable(self):
        case = sample_case()
        case["matrix_row_major"] = [1.0, 0.5]
        record = qc.compare_case(case)
        self.assertEqual(record["comparison_status"], "not_comparable")
        self.assertIn("length does not match", record["failure_reason"])
        json.dumps(record, allow_nan=False)

    def test_nonfinite_native_values_produce_json_safe_failure_record(self):
        case = sample_case()
        case["native_result"]["eigenvalues"][0] = float("nan")
        record = qc.compare_case(case)
        self.assertEqual(record["comparison_status"], "not_comparable")
        json.dumps(record, allow_nan=False)

    def test_schema_requires_integer_version_not_boolean_or_float(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            report = sample_report()
            path.write_text(json.dumps(report), encoding="utf-8")
            loaded, digest = qc.load_input(path)
            self.assertEqual(loaded["schema_version"], 1)
            self.assertEqual(len(digest), 64)

            for version in (True, 1.0, 2):
                report["schema_version"] = version
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unsupported eigen fixture schema"):
                    qc.load_input(path)

    def test_load_input_requires_complete_unique_frozen_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"

            report = sample_report()
            report["cases"] = report["cases"][:-1]
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corpus census mismatch"):
                qc.load_input(path)

            report = sample_report()
            report["cases"][1]["case_id"] = report["cases"][0]["case_id"]
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate fixture case IDs"):
                qc.load_input(path)

            report = sample_report()
            report["cases"][0]["case_id"] = "unexpected_case"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corpus census mismatch"):
                qc.load_input(path)

            report = sample_report()
            report["cases"].reverse()
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "case order differs"):
                qc.load_input(path)

    def test_source_binding_requires_exact_current_head_and_clean_tree(self):
        report = sample_report()
        commit = report["source_revision"]
        tree = report["source_tree_sha"]
        with patch.object(
            qc.subprocess,
            "run",
            side_effect=[
                completed(commit + "\\n"),
                completed(tree + "\\n"),
                completed(commit + "\\n"),
                completed(tree + "\\n"),
                completed(""),
            ],
        ) as run:
            qc.verify_source_binding(commit, tree)
        self.assertEqual(run.call_count, 5)

    def test_source_binding_rejects_other_head_and_dirty_tree(self):
        report = sample_report()
        commit = report["source_revision"]
        tree = report["source_tree_sha"]
        other_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        with patch.object(
            qc.subprocess,
            "run",
            side_effect=[
                completed(commit + "\\n"),
                completed(tree + "\\n"),
                completed(other_commit + "\\n"),
                completed(tree + "\\n"),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "does not equal current Git HEAD"):
                qc.verify_source_binding(commit, tree)

        with patch.object(
            qc.subprocess,
            "run",
            side_effect=[
                completed(commit + "\\n"),
                completed(tree + "\\n"),
                completed(commit + "\\n"),
                completed(tree + "\\n"),
                completed(" M src/lib.rs\\n"),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "worktree is dirty"):
                qc.verify_source_binding(commit, tree)


if __name__ == "__main__":
    unittest.main(verbosity=2)
