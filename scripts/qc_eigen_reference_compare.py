#!/usr/bin/env python3
"""Compare checked native Jacobi results against NumPy's independent symmetric solver.

This is a small-matrix numerical cross-check, not a quantum-chemistry accuracy
claim. The input report must be source-bound to the current clean Git checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

INPUT_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1
MAX_RELATIVE_EIGENVALUE_ERROR = 1e-10
MAX_RELATIVE_EIGENPAIR_RESIDUAL = 1e-10
MAX_ORTHOGONALITY_RESIDUAL = 1e-10

# Frozen corpus census: a shorter report must never become a green comparison
# merely because a difficult or failing matrix was silently omitted.
EXPECTED_CASE_IDS = (
    "scalar_3_25",
    "diagonal_3_1",
    "degenerate_identity_scaled",
    "dense_symmetric_3x3",
    "negative_off_diagonal",
    "near_linear_dependence",
    "small_scale_symmetric",
    "subnormal_scale_symmetric",
)


def full_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def verify_source_binding(source_revision: str, source_tree_sha: str) -> None:
    """Require input provenance to match the current clean checkout exactly."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", f"{source_revision}^{{commit}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        tree = subprocess.run(
            ["git", "rev-parse", "--verify", f"{source_revision}^{{tree}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            check=False,
            capture_output=True,
            text=True,
        )
        head_tree = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{tree}"],
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"could not invoke Git for source verification: {exc}") from exc

    for label, result in [
        ("source commit", commit),
        ("source tree", tree),
        ("current HEAD", head),
        ("current HEAD tree", head_tree),
        ("worktree status", status),
    ]:
        if result.returncode != 0:
            raise ValueError(f"Git failed to resolve {label}: {result.stderr.strip()}")

    if commit.stdout.strip().lower() != source_revision.lower():
        raise ValueError("Git resolved source revision to a different commit")
    if tree.stdout.strip().lower() != source_tree_sha.lower():
        raise ValueError("declared source tree does not belong to declared source revision")
    if head.stdout.strip().lower() != source_revision.lower():
        raise ValueError("input report source revision does not equal current Git HEAD")
    if head_tree.stdout.strip().lower() != source_tree_sha.lower():
        raise ValueError("input report source tree does not equal current HEAD tree")
    if status.stdout.strip():
        raise ValueError("current Git worktree is dirty; rerun the fixture on a clean checkout")


def load_input(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"fixture input is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("fixture report root must be a JSON object")
    schema_version = report.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != INPUT_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported eigen fixture schema: {schema_version!r}")
    if report.get("producer") != "symthaea-quantum-chemistry":
        raise ValueError("unexpected native fixture producer")
    if report.get("worktree_clean") is not True:
        raise ValueError("native report must explicitly bind worktree_clean=true")
    if not full_sha(report.get("source_revision")) or not full_sha(report.get("source_tree_sha")):
        raise ValueError("native report must contain full Git commit and tree SHAs")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("native report must contain a non-empty cases array")
    case_ids = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"cases[{index}] must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"cases[{index}].case_id must be a non-empty string")
        case_ids.append(case_id)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("native report contains duplicate fixture case IDs")
    expected_ids = set(EXPECTED_CASE_IDS)
    actual_ids = set(case_ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"fixture corpus census mismatch: missing={missing!r}; unexpected={unexpected!r}"
        )
    if tuple(case_ids) != EXPECTED_CASE_IDS:
        raise ValueError("fixture corpus case order differs from the frozen reference order")
    return report, digest


def compare_case(case: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": case.get("case_id", "missing-case-id") if isinstance(case, dict) else "invalid-case",
        "native_status": None,
        "reference_status": "not_run",
        "comparison_status": "not_comparable",
        "max_relative_eigenvalue_error": None,
        "max_relative_reference_eigenpair_residual": None,
        "max_reference_orthogonality_residual": None,
        "native_reported_relative_eigenpair_residual": None,
        "native_reported_orthogonality_residual": None,
        "max_off_diagonal": None,
        "iterations": None,
        "failure_reason": None,
    }
    try:
        if not isinstance(case, dict):
            raise ValueError("case must be an object")
        case_id = case.get("case_id")
        dimension = case.get("dimension")
        matrix = case.get("matrix_row_major")
        native = case.get("native_result")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(matrix, list) or len(matrix) != dimension * dimension:
            raise ValueError("matrix_row_major length does not match dimension")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in matrix
        ):
            raise ValueError("matrix contains non-finite or non-numeric entries")
        if not isinstance(native, dict):
            raise ValueError("native_result must be an object")
        result["native_status"] = native.get("status")
        result["native_reported_relative_eigenpair_residual"] = native.get(
            "max_relative_eigenpair_residual"
        )
        result["native_reported_orthogonality_residual"] = native.get(
            "max_orthogonality_residual"
        )
        result["max_off_diagonal"] = native.get("max_off_diagonal")
        result["iterations"] = native.get("iterations")
        if (
            native.get("status") != "converged"
            or native.get("converged") is not True
            or native.get("stopping_reason") != "converged"
        ):
            raise ValueError(
                "native checked eigensolver did not declare convergence; reference cannot upgrade it"
            )
        iterations = native.get("iterations")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
            raise ValueError("native rotation count must be a nonnegative integer")

        matrix_np = np.asarray(matrix, dtype=np.float64).reshape((dimension, dimension))
        scale = float(np.max(np.abs(matrix_np)))
        scale = scale if scale > 0.0 else 1.0
        if scale == 0.0:
            scale = 1.0
        normalized_asymmetry = float(np.max(np.abs(matrix_np - matrix_np.T))) / scale
        if not math.isfinite(normalized_asymmetry) or normalized_asymmetry > 1e-12:
            raise ValueError(f"matrix is not symmetric within contract: {normalized_asymmetry:.12g}")

        native_values = native.get("eigenvalues")
        native_vectors_flat = native.get("eigenvectors_row_major")
        if not isinstance(native_values, list) or len(native_values) != dimension:
            raise ValueError("native eigenvalue vector has wrong length")
        if not isinstance(native_vectors_flat, list) or len(native_vectors_flat) != dimension * dimension:
            raise ValueError("native eigenvector matrix has wrong shape")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in native_values + native_vectors_flat
        ):
            raise ValueError("native eigenvalues/eigenvectors contain non-finite values")

        native_residual = native.get("max_relative_eigenpair_residual")
        native_orthogonality = native.get("max_orthogonality_residual")
        native_off_diagonal = native.get("max_off_diagonal")
        for label, value in [
            ("native relative eigenpair residual", native_residual),
            ("native orthogonality residual", native_orthogonality),
            ("native maximum off-diagonal", native_off_diagonal),
        ]:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{label} is missing, negative, or non-finite")
        if native_residual > MAX_RELATIVE_EIGENPAIR_RESIDUAL:
            raise ValueError("native reported eigenpair residual exceeds the contract")
        if native_orthogonality > MAX_ORTHOGONALITY_RESIDUAL:
            raise ValueError("native reported eigenvector orthogonality exceeds the contract")
        if native_off_diagonal / scale > 1e-14:
            raise ValueError("native reported off-diagonal residual exceeds the contract")

        native_values_np = np.sort(np.asarray(native_values, dtype=np.float64))
        native_vectors_np = np.asarray(native_vectors_flat, dtype=np.float64).reshape(
            (dimension, dimension)
        )
        reference_values, _ = np.linalg.eigh(matrix_np)
        relative_eigenvalue_error = float(
            np.max(np.abs(native_values_np - reference_values)) / scale
        )
        reference_residual = float(
            np.max(np.abs(matrix_np @ native_vectors_np - native_vectors_np @ np.diag(native_values)))
            / scale
        )
        gram = native_vectors_np.T @ native_vectors_np
        orthogonality_residual = float(np.max(np.abs(gram - np.eye(dimension))))
        if not all(
            math.isfinite(value)
            for value in (
                relative_eigenvalue_error,
                reference_residual,
                orthogonality_residual,
            )
        ):
            raise ValueError("NumPy comparison produced non-finite derived diagnostics")

        result["reference_status"] = "computed"
        result["max_relative_eigenvalue_error"] = relative_eigenvalue_error
        result["max_relative_reference_eigenpair_residual"] = reference_residual
        result["max_reference_orthogonality_residual"] = orthogonality_residual

        violations = []
        if relative_eigenvalue_error > MAX_RELATIVE_EIGENVALUE_ERROR:
            violations.append(
                f"eigenvalue error {relative_eigenvalue_error:.12g} exceeds "
                f"{MAX_RELATIVE_EIGENVALUE_ERROR:g}"
            )
        if reference_residual > MAX_RELATIVE_EIGENPAIR_RESIDUAL:
            violations.append(
                f"independent eigenpair residual {reference_residual:.12g} exceeds "
                f"{MAX_RELATIVE_EIGENPAIR_RESIDUAL:g}"
            )
        if orthogonality_residual > MAX_ORTHOGONALITY_RESIDUAL:
            violations.append(
                f"eigenvector orthogonality residual {orthogonality_residual:.12g} exceeds "
                f"{MAX_ORTHOGONALITY_RESIDUAL:g}"
            )
        if native.get("max_relative_eigenpair_residual") is None or native.get(
            "max_orthogonality_residual"
        ) is None:
            violations.append("native solver omitted its residual diagnostics")
        if violations:
            result["comparison_status"] = "failed_numerical_contract"
            result["failure_reason"] = "; ".join(violations)
        else:
            result["comparison_status"] = "passed"
        return result
    except Exception as exc:
        result["reference_status"] = "failed"
        result["comparison_status"] = "not_comparable"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="source-bound native eigen fixture JSON")
    parser.add_argument("--output", type=Path, help="comparison JSON output path; defaults to stdout")
    args = parser.parse_args(argv)
    if args.output is not None and args.input.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the native input report")

    try:
        data, input_digest = load_input(args.input)
        verify_source_binding(data["source_revision"], data["source_tree_sha"])
    except (OSError, ValueError) as exc:
        print(f"qc_eigen_reference_compare: {exc}", file=sys.stderr)
        return 2

    results = [compare_case(case) for case in data["cases"]]
    passed = sum(record["comparison_status"] == "passed" for record in results)
    failed = sum(record["comparison_status"].startswith("failed") for record in results)
    not_comparable = len(results) - passed - failed
    report = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "producer": "symthaea-numpy-eigen-reference-comparator",
        "numpy_version": np.__version__,
        "native_input_sha256": input_digest,
        "native_source_revision": data["source_revision"],
        "native_source_tree_sha": data["source_tree_sha"],
        "source_tree_binding_verified_against_local_git": True,
        "comparison_contract": {
            "max_relative_eigenvalue_error": MAX_RELATIVE_EIGENVALUE_ERROR,
            "max_relative_eigenpair_residual": MAX_RELATIVE_EIGENPAIR_RESIDUAL,
            "max_eigenvector_orthogonality_residual": MAX_ORTHOGONALITY_RESIDUAL,
            "reference_solver": "numpy.linalg.eigh",
            "note": "Matrix-level numerical cross-check only; not a chemistry accuracy claim.",
        },
        "summary": {
            "cases": len(results),
            "passed": passed,
            "failed": failed,
            "not_comparable": not_comparable,
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        temporary = args.output.with_name(f"{args.output.name}.tmp-{os.getpid()}")
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(rendered, encoding="utf-8")
            os.replace(temporary, args.output)
        except OSError as exc:
            print(f"qc_eigen_reference_compare: could not publish report: {exc}", file=sys.stderr)
            return 2
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    print(
        f"NumPy eigen comparison: {passed}/{len(results)} passed; "
        f"{failed} failed; {not_comparable} not comparable.",
        file=sys.stderr,
    )
    return 0 if passed == len(results) and failed == 0 and not_comparable == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
