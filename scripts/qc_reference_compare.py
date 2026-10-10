#!/usr/bin/env python3
"""Compare native Symthaea RHF fixtures against independent PySCF results.

Run the native fixture producer from a clean worktree in a Rust-enabled shell first:
  cargo run -p symthaea-quantum-chemistry --example qc_native_reference_fixtures > /tmp/qc-native.json

Then run this script inside the repository's PySCF shell:
  nix develop .#qc-verify --command python scripts/qc_reference_compare.py \
    --input /tmp/qc-native.json --output /tmp/qc-comparison.json

Every input case appears in the output, including native failures, PySCF
failures, and mismatches. The comparator never loosens thresholds to make a
known failure pass. Cross-backend agreement is a software-validation signal,
not a claim of chemical accuracy or experimental validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from numbers import Real
from pathlib import Path
from typing import Any


SUPPORTED_SCHEMA_VERSION = 2
COMPARISON_REPORT_SCHEMA_VERSION = 2
DEFAULT_COMPARISON_TOLERANCE_HARTREE = 1e-6
MAX_COMPARISON_TOLERANCE_HARTREE = 1e-6
MAX_OVERLAP_MATRIX_RESIDUAL = 1e-8
BASIS_ALIASES = {"STO-3G": "sto-3g", "6-31G": "6-31g"}


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def verify_source_binding(source_revision: str, source_tree_sha: str) -> None:
    """Verify that the declared commit exists locally and owns the declared tree.

    This checks Git-object consistency only. It is not a signature or an
    attestation that a supplied JSON report was actually produced by that code.
    """
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{source_revision}^{{commit}}"],
            check=False,
            capture_output=True,
            text=True,
        )
        tree_result = subprocess.run(
            ["git", "rev-parse", "--verify", f"{source_revision}^{{tree}}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"could not invoke git to verify source binding: {exc}") from exc
    if commit_result.returncode != 0:
        raise ValueError(
            "source_revision is not available as a Git commit in this checkout: "
            + commit_result.stderr.strip()
        )
    if tree_result.returncode != 0:
        raise ValueError(
            "could not resolve the Git tree for source_revision: "
            + tree_result.stderr.strip()
        )
    if commit_result.stdout.strip().lower() != source_revision.lower():
        raise ValueError("Git resolved source_revision to a different commit")
    if tree_result.stdout.strip().lower() != source_tree_sha.lower():
        raise ValueError("source_tree_sha does not match the tree of source_revision")

    # Exact-head evidence must be generated and compared from the same checkout.
    try:
        head_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            check=False,
            capture_output=True,
            text=True,
        )
        head_tree_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{tree}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"could not verify the current Git HEAD: {exc}") from exc
    if head_result.returncode != 0 or head_tree_result.returncode != 0:
        raise ValueError("could not resolve the current Git HEAD/tree")
    if head_result.stdout.strip().lower() != source_revision.lower():
        raise ValueError("native report source_revision does not equal the current Git HEAD")
    if head_tree_result.stdout.strip().lower() != source_tree_sha.lower():
        raise ValueError("native report source_tree_sha does not equal the current HEAD tree")
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"could not verify the current Git worktree: {exc}") from exc
    if status_result.returncode != 0:
        raise ValueError("could not inspect the current Git worktree status")
    if status_result.stdout.strip():
        raise ValueError(
            "current Git worktree is dirty; rerun the native fixture and comparison from a clean checkout"
        )


def load_input(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"input is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("input report root must be a JSON object")
    schema_version = data.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SUPPORTED_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported input schema {schema_version!r}; "
            f"supported schema is {SUPPORTED_SCHEMA_VERSION}"
        )
    def is_full_sha(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 40
            and all(character in "0123456789abcdefABCDEF" for character in value)
        )

    if not is_full_sha(data.get("source_revision")):
        raise ValueError("input report must bind source_revision to a full 40-character Git commit SHA")
    if not is_full_sha(data.get("source_tree_sha")):
        raise ValueError("input report must bind source_tree_sha to a full 40-character Git tree SHA")
    if data.get("worktree_clean") is not True:
        raise ValueError("input report must explicitly attest worktree_clean=true")
    if data.get("producer") != "symthaea-quantum-chemistry":
        raise ValueError("input report producer must be symthaea-quantum-chemistry")
    if data.get("coordinate_unit") != "bohr":
        raise ValueError("input report coordinate_unit must be explicitly 'bohr'")
    if data.get("method") != "RHF":
        raise ValueError("input report method must be RHF for this comparator lane")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("input report must contain a non-empty cases array")
    return data, digest


def compare_case(case: Any, pyscf: Any, tolerance_hartree: float) -> dict[str, Any]:
    # Always create a result record for every input case; never filter failed cases.
    base: dict[str, Any] = {
        "case_id": case.get("case_id", "missing-case-id") if isinstance(case, dict) else "invalid-case-record",
        "native_status": None,
        "native_basis_functions": None,
        "native_overlap_matrix_max_abs_delta": None,
        "reference_overlap_matrix_max_abs_entry": None,
        "overlap_matrix_status": "not_run",
        "native_total_energy_hartree": None,
        "native_electronic_energy_hartree": None,
        "native_nuclear_repulsion_hartree": None,
        "reference_status": "not_run",
        "reference_total_energy_hartree": None,
        "reference_electronic_energy_hartree": None,
        "reference_nuclear_repulsion_hartree": None,
        "reference_iterations": None,
        "reference_basis_functions": None,
        "reference_solver": "PySCF",
        "reference_solver_version": getattr(pyscf, "__version__", "unknown"),
        "comparison_status": "not_comparable",
        "delta_native_minus_pyscf_hartree": None,
        "delta_native_minus_pyscf_kcal_mol": None,
        "delta_electronic_energy_hartree": None,
        "delta_nuclear_repulsion_hartree": None,
        "native_energy_decomposition_residual_hartree": None,
        "reference_energy_decomposition_residual_hartree": None,
        "comparison_tolerance_hartree": tolerance_hartree,
        "failure_reason": None,
    }
    if not isinstance(case, dict):
        base["reference_status"] = "invalid_input"
        base["failure_reason"] = "case record must be an object"
        return base

    native = case.get("native_result")
    native = native if isinstance(native, dict) else {}
    base["native_status"] = native.get("status")
    native_basis_count = native.get("basis_functions")
    if isinstance(native_basis_count, int) and not isinstance(native_basis_count, bool):
        base["native_basis_functions"] = native_basis_count
    base["native_total_energy_hartree"] = finite_float(native.get("total_energy_hartree"))
    base["native_electronic_energy_hartree"] = finite_float(
        native.get("electronic_energy_hartree")
    )
    base["native_nuclear_repulsion_hartree"] = finite_float(
        native.get("nuclear_repulsion_hartree")
    )

    try:
        if case.get("coordinate_unit") != "bohr":
            raise ValueError("coordinate_unit must be explicitly 'bohr'")
        method = case.get("method")
        if method != "RHF":
            raise ValueError(f"unsupported comparison method {method!r}; this lane supports RHF only")
        basis_label = case.get("basis")
        if basis_label not in BASIS_ALIASES:
            raise ValueError(f"unsupported basis label {basis_label!r}")
        molecule = case.get("molecule")
        if not isinstance(molecule, dict):
            raise ValueError("molecule must be an object")
        atoms = molecule.get("atoms")
        if not isinstance(atoms, list) or not atoms:
            raise ValueError("molecule atoms must be a non-empty array")
        charge = molecule.get("charge")
        multiplicity = molecule.get("multiplicity")
        if isinstance(charge, bool) or not isinstance(charge, int):
            raise ValueError("molecule charge must be an integer")
        if isinstance(multiplicity, bool) or not isinstance(multiplicity, int) or multiplicity < 1:
            raise ValueError("molecule multiplicity must be a positive integer")

        atom_spec = []
        declared_atomic_numbers: list[int] = []
        for index, atom in enumerate(atoms):
            if not isinstance(atom, dict):
                raise ValueError(f"atom {index} must be an object")
            symbol = atom.get("symbol")
            atomic_number = atom.get("atomic_number")
            coordinates = atom.get("position_bohr")
            if not isinstance(symbol, str) or not symbol:
                raise ValueError(f"atom {index} has no element symbol")
            if (
                isinstance(atomic_number, bool)
                or not isinstance(atomic_number, int)
                or not 1 <= atomic_number <= 118
            ):
                raise ValueError(f"atom {index} atomic_number must be an integer in 1..=118")
            if not isinstance(coordinates, list) or len(coordinates) != 3:
                raise ValueError(f"atom {index} position_bohr must have three coordinates")
            xyz = [finite_float(coord) for coord in coordinates]
            if any(coord is None for coord in xyz):
                raise ValueError(f"atom {index} contains a non-finite coordinate")
            atom_spec.append((symbol, tuple(xyz)))
            declared_atomic_numbers.append(atomic_number)

        # This comparator lane is RHF, not an open-shell or spin-polarized check.
        if multiplicity != 1:
            raise ValueError("RHF reference comparison requires multiplicity 1")
        electron_count = sum(declared_atomic_numbers) - charge
        if electron_count <= 0 or electron_count % 2 != 0:
            raise ValueError(
                f"RHF singlet geometry requires a positive even electron count; got {electron_count}"
            )

        # Spin follows PySCF's convention N_alpha - N_beta = 2S = multiplicity - 1.
        from pyscf import gto, scf  # imported lazily so input errors are still reported cleanly

        mol = gto.M(
            atom=atom_spec,
            unit="Bohr",
            charge=charge,
            spin=multiplicity - 1,
            basis=BASIS_ALIASES[basis_label],
            cart=True,
            symmetry=False,
            verbose=0,
        )
        if [int(mol.atom_charge(i)) for i in range(mol.natm)] != declared_atomic_numbers:
            raise ValueError("atom symbols do not match the declared atomic numbers")

        # Equal basis labels and dimensions are not enough: compare the AO
        # overlap matrices of the actual represented basis functions.
        ref_overlap = mol.intor("int1e_ovlp")
        native_n = int(mol.nao_nr())
        expected_shape = (native_n, native_n)
        if getattr(ref_overlap, "shape", None) != expected_shape:
            raise ValueError("PySCF overlap matrix has unexpected dimensions")
        ref_overlap_rows = ref_overlap.tolist()
        if (
            not isinstance(ref_overlap_rows, list)
            or len(ref_overlap_rows) != native_n
            or any(not isinstance(row, list) or len(row) != native_n for row in ref_overlap_rows)
        ):
            raise ValueError("PySCF overlap matrix has malformed rows")
        ref_overlap_values = [
            finite_float(value)
            for row in ref_overlap_rows
            for value in row
        ]
        if any(value is None for value in ref_overlap_values):
            raise ValueError("PySCF AO overlap matrix contains non-finite entries")
        base["reference_overlap_matrix_max_abs_entry"] = max(
            (abs(value) for value in ref_overlap_values),
            default=0.0,
        )

        native_overlap = native.get("overlap_matrix_row_major")
        if not isinstance(native_overlap, list) or len(native_overlap) != native_n * native_n:
            base["overlap_matrix_status"] = "missing_native_overlap_matrix"
            base["failure_reason"] = "native report has no correctly sized overlap matrix"
        else:
            native_overlap_values = [finite_float(value) for value in native_overlap]
            if any(value is None for value in native_overlap_values):
                base["overlap_matrix_status"] = "invalid_native_overlap_matrix"
                base["failure_reason"] = "native overlap matrix contains non-finite entries"
            else:
                max_delta = max(
                    abs(
                        native_overlap_values[i * native_n + j]
                        - ref_overlap_values[i * native_n + j]
                    )
                    for i in range(native_n)
                    for j in range(native_n)
                )
                if not math.isfinite(max_delta):
                    base["overlap_matrix_status"] = "failed_overlap_mismatch"
                    base["failure_reason"] = "overlap matrix difference became non-finite"
                else:
                    base["native_overlap_matrix_max_abs_delta"] = max_delta
                    base["overlap_matrix_status"] = (
                        "passed"
                        if max_delta <= MAX_OVERLAP_MATRIX_RESIDUAL
                        else "failed_overlap_mismatch"
                    )
                    if base["overlap_matrix_status"] == "failed_overlap_mismatch":
                        base["failure_reason"] = (
                            f"native/PySCF overlap matrices differ by {max_delta:.12g}, "
                            f"exceeding the {MAX_OVERLAP_MATRIX_RESIDUAL:g} residual limit"
                        )

        reference = scf.RHF(mol)
        reference.max_cycle = 200
        reference.conv_tol = 1e-10
        reference.conv_tol_grad = 1e-8
        energy = finite_float(reference.kernel())
        base["reference_total_energy_hartree"] = energy
        base["reference_iterations"] = int(cycles) if (cycles := finite_float(getattr(reference, "cycles", None))) is not None else None
        base["reference_basis_functions"] = int(mol.nao_nr())
        electronic_energy, _ = reference.energy_elec()
        base["reference_electronic_energy_hartree"] = finite_float(electronic_energy)
        base["reference_nuclear_repulsion_hartree"] = finite_float(mol.energy_nuc())
        if energy is None:
            base["reference_status"] = "non_finite_result"
            base["failure_reason"] = "PySCF returned a non-finite total energy"
            return base
        if base["reference_electronic_energy_hartree"] is None or base["reference_nuclear_repulsion_hartree"] is None:
            base["reference_status"] = "non_finite_components"
            base["failure_reason"] = "PySCF returned a non-finite energy component"
            return base
        if not bool(reference.converged):
            base["reference_status"] = "not_converged"
            base["failure_reason"] = "PySCF did not meet its SCF convergence criteria"
            return base
        base["reference_status"] = "converged"

        native_energy = base["native_total_energy_hartree"]
        if native.get("status") != "converged":
            base["comparison_status"] = "failed_native"
            base["failure_reason"] = (
                "native solver status is "
                + str(native.get("status", "missing"))
                + "; a reference energy alone cannot upgrade it"
            )
            return base
        if native_energy is None:
            base["comparison_status"] = "failed_native"
            base["failure_reason"] = "native energy is missing or non-finite"
            return base

        native_basis_count = base["native_basis_functions"]
        reference_basis_count = base["reference_basis_functions"]
        if native_basis_count is None or native_basis_count <= 0:
            base["comparison_status"] = "failed_native"
            base["failure_reason"] = "native basis-function count is missing or invalid"
            return base
        if native_basis_count != reference_basis_count:
            base["comparison_status"] = "failed_basis_mismatch"
            base["failure_reason"] = (
                f"native basis has {native_basis_count} functions but PySCF has "
                f"{reference_basis_count}; energy comparison is not apples-to-apples"
            )
            return base

        if base["overlap_matrix_status"] != "passed":
            base["comparison_status"] = (
                "failed_overlap_mismatch"
                if base["overlap_matrix_status"] == "failed_overlap_mismatch"
                else "failed_native"
            )
            base["failure_reason"] = (
                "native/PySCF overlap-matrix comparison did not pass: "
                + str(base["overlap_matrix_status"])
                + f" (required max |ΔS| ≤ {MAX_OVERLAP_MATRIX_RESIDUAL:g})"
            )
            return base

        native_electronic = base["native_electronic_energy_hartree"]
        native_nuclear = base["native_nuclear_repulsion_hartree"]
        reference_electronic = base["reference_electronic_energy_hartree"]
        reference_nuclear = base["reference_nuclear_repulsion_hartree"]
        if any(value is None for value in (native_electronic, native_nuclear)):
            base["comparison_status"] = "failed_native"
            base["failure_reason"] = "native energy component is missing or non-finite"
            return base

        delta = native_energy - energy
        delta_electronic = native_electronic - reference_electronic
        delta_nuclear = native_nuclear - reference_nuclear
        native_decomposition_residual = native_energy - (native_electronic + native_nuclear)
        reference_decomposition_residual = energy - (
            reference_electronic + reference_nuclear
        )
        derived_values = (
            delta,
            delta * 627.509474063,
            delta_electronic,
            delta_nuclear,
            native_decomposition_residual,
            reference_decomposition_residual,
        )
        if not all(math.isfinite(value) for value in derived_values):
            base["comparison_status"] = "failed_non_finite_comparison"
            base["failure_reason"] = (
                "derived energy deltas or energy-decomposition residuals are non-finite"
            )
            return base

        base["delta_native_minus_pyscf_hartree"] = delta
        base["delta_native_minus_pyscf_kcal_mol"] = delta * 627.509474063
        base["delta_electronic_energy_hartree"] = delta_electronic
        base["delta_nuclear_repulsion_hartree"] = delta_nuclear
        base["native_energy_decomposition_residual_hartree"] = native_decomposition_residual
        base["reference_energy_decomposition_residual_hartree"] = reference_decomposition_residual
        discrepancies = {
            "total energy": delta,
            "electronic energy": delta_electronic,
            "nuclear repulsion": delta_nuclear,
            "native energy decomposition residual": native_decomposition_residual,
            "PySCF energy decomposition residual": reference_decomposition_residual,
        }
        failed = {
            name: abs(value)
            for name, value in discrepancies.items()
            if abs(value) > tolerance_hartree
        }
        if not failed:
            base["comparison_status"] = "passed"
        else:
            base["comparison_status"] = "failed_discrepancy"
            details = ", ".join(f"{name}={difference:.12g} Ha" for name, difference in failed.items())
            base["failure_reason"] = (
                f"cross-backend component discrepancy exceeds explicit "
                f"{tolerance_hartree:.12g} Ha threshold: {details}"
            )
        return base
    except Exception as exc:  # preserve the record; a backend/input failure is evidence
        base["reference_status"] = "failed"
        base["comparison_status"] = "not_comparable"
        base["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="native JSON fixture report")
    parser.add_argument("--output", type=Path, help="write JSON report here instead of stdout")
    parser.add_argument(
        "--tolerance-hartree",
        type=float,
        default=DEFAULT_COMPARISON_TOLERANCE_HARTREE,
        help=(
            "cross-backend numeric comparison tolerance; may be stricter than 1e-6 Ha, "
            "but never looser (not a chemistry accuracy tolerance)"
        ),
    )
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.tolerance_hartree)
        or args.tolerance_hartree <= 0
        or args.tolerance_hartree > MAX_COMPARISON_TOLERANCE_HARTREE
    ):
        parser.error(
            "--tolerance-hartree must be finite, strictly positive, and no greater than "
            f"{MAX_COMPARISON_TOLERANCE_HARTREE:g} Ha; loosening the acceptance bound "
            "requires a reviewed source change"
        )
    if args.output is not None and args.output.resolve() == args.input.resolve():
        parser.error("--output must not overwrite the native --input report")

    try:
        data, input_digest = load_input(args.input)
        verify_source_binding(data["source_revision"], data["source_tree_sha"])
    except (OSError, ValueError) as exc:
        print(f"qc_reference_compare: {exc}", file=sys.stderr)
        return 2

    # Import up front for a clear environment error, but keep per-case failures
    # in the report when a specific case cannot be evaluated.
    try:
        import pyscf
    except Exception as exc:
        print(f"qc_reference_compare: PySCF import failed: {exc}", file=sys.stderr)
        return 2

    results = [compare_case(case, pyscf, args.tolerance_hartree) for case in data["cases"]]
    passed = sum(record["comparison_status"] == "passed" for record in results)
    failed = sum(record["comparison_status"].startswith("failed") for record in results)
    not_comparable = len(results) - passed - failed
    report = {
        "schema_version": COMPARISON_REPORT_SCHEMA_VERSION,
        "producer": "symthaea-pyscf-reference-comparator",
        "reference_solver": "PySCF",
        "reference_solver_version": getattr(pyscf, "__version__", "unknown"),
        "input_report_sha256": input_digest,
        "native_source_revision": data.get("source_revision"),
        "native_source_tree_sha": data.get("source_tree_sha"),
        "native_worktree_clean": data.get("worktree_clean"),
        "source_tree_binding_verified_against_local_git": True,
        "native_producer_version": data.get("producer_version"),
        "comparison_tolerance_hartree": args.tolerance_hartree,
        "maximum_permitted_comparison_tolerance_hartree": MAX_COMPARISON_TOLERANCE_HARTREE,
        "maximum_permitted_overlap_matrix_residual": MAX_OVERLAP_MATRIX_RESIDUAL,
        "reference_settings": {
            "method": "RHF",
            "max_cycle": 200,
            "conv_tol_hartree": 1e-10,
            "conv_tol_grad": 1e-8,
            "geometry_unit": "Bohr",
            "cartesian_basis": True,
            "symmetry": False,
        },
        "comparison_policy": (
            "Compare identical serialized geometry (Bohr), atomic numbers, charge, multiplicity, RHF "
            "method, and basis. Native and PySCF AO overlap matrices must agree within the separate "
            "1e-8 maximum elementwise residual before energies are compared. Total, electronic, and "
            "nuclear-repulsion energy differences and each backend's total=electronic+nuclear "
            "decomposition residual must meet the explicit energy threshold. "
            "A pass means numerical cross-backend agreement only, not chemical accuracy. Cases with "
            "missing/non-converged/non-finite native results cannot pass."
        ),
        "summary": {
            "cases": len(results),
            "passed": passed,
            "failed": failed,
            "not_comparable": not_comparable,
        },
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        temporary_output = args.output.with_name(
            f"{args.output.name}.tmp-{os.getpid()}"
        )
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary_output.write_text(rendered, encoding="utf-8")
            os.replace(temporary_output, args.output)
        except OSError as exc:
            print(f"qc_reference_compare: could not write report: {exc}", file=sys.stderr)
            return 2
        finally:
            try:
                temporary_output.unlink(missing_ok=True)
            except OSError:
                pass
    else:
        sys.stdout.write(rendered)

    print(
        f"PySCF comparison: {passed}/{len(results)} passed; "
        f"{failed} failed; {not_comparable} not comparable. "
        f"Report: {args.output if args.output else 'stdout'}",
        file=sys.stderr,
    )
    return 0 if failed == 0 and not_comparable == 0 and passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
