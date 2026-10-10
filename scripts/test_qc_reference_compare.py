#!/usr/bin/env python3
"""Fast contract tests for qc_reference_compare.py using a fake PySCF backend.

These tests validate record retention and comparator policy only; the independent
backend's numerical output must still be tested in the real PySCF shell.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qc_reference_compare as qc  # noqa: E402


def sample_case() -> dict:
    return {
        "case_id": "H2/STO-3G/RHF",
        "coordinate_unit": "bohr",
        "method": "RHF",
        "basis": "STO-3G",
        "molecule": {
            "charge": 0,
            "multiplicity": 1,
            "atoms": [
                {"atomic_number": 1, "symbol": "H", "position_bohr": [0.0, 0.0, 0.0]},
                {"atomic_number": 1, "symbol": "H", "position_bohr": [0.0, 0.0, 1.4]},
            ],
        },
        "native_result": {
            "status": "converged",
            "basis_functions": 2,
            "overlap_matrix_row_major": [1.0, 0.2, 0.2, 1.0],
            "total_energy_hartree": -1.1175,
            "electronic_energy_hartree": -1.8317857142857143,
            "nuclear_repulsion_hartree": 0.7142857142857143,
        },
    }


def fake_pyscf(energy: float = -1.1175, converged: bool = True, error: Exception | None = None):
    module = types.ModuleType("pyscf")
    module.__version__ = "test-double-0"

    class FakeOverlap:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows), len(rows))

        def tolist(self):
            return self.rows

    class FakeMolecule:
        def __init__(self, kwargs):
            self.kwargs = kwargs
            self.atoms = kwargs["atom"]
            self.natm = len(self.atoms)

        def atom_charge(self, index):
            symbols = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8}
            return symbols[self.atoms[index][0]]

        def energy_nuc(self):
            return 1.0 / 1.4

        def nao_nr(self):
            return 2

        def intor(self, name):
            if name != "int1e_ovlp":
                raise ValueError(f"unexpected fake integral {name}")
            return FakeOverlap([[1.0, 0.2], [0.2, 1.0]])

    def build_molecule(**kwargs):
        return FakeMolecule(kwargs)

    class FakeRHF:
        max_cycle = 0
        conv_tol = 0.0
        conv_tol_grad = 0.0

        def __init__(self, molecule):
            self.molecule = molecule
            self.converged = converged
            self.cycles = 8

        def kernel(self):
            if error is not None:
                raise error
            return energy

        def energy_elec(self):
            return energy - self.molecule.energy_nuc(), 0.0

    module.gto = types.SimpleNamespace(M=build_molecule)
    module.scf = types.SimpleNamespace(RHF=FakeRHF)
    return module


class QcReferenceCompareTests(unittest.TestCase):
    def compare_with(self, case: dict, reference: types.ModuleType, tolerance: float = 1e-6):
        with patch.dict(sys.modules, {"pyscf": reference}):
            return qc.compare_case(case, reference, tolerance)

    def test_equal_finite_energies_pass(self):
        record = self.compare_with(sample_case(), fake_pyscf())
        self.assertEqual(record["comparison_status"], "passed")
        self.assertEqual(record["reference_status"], "converged")
        self.assertAlmostEqual(record["delta_native_minus_pyscf_hartree"], 0.0)
        self.assertEqual(record["overlap_matrix_status"], "passed")
        self.assertAlmostEqual(record["native_overlap_matrix_max_abs_delta"], 0.0)

    def test_overlap_matrix_mismatch_blocks_energy_comparison(self):
        case = sample_case()
        case["native_result"]["overlap_matrix_row_major"][1] = 0.2001
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["overlap_matrix_status"], "failed_overlap_mismatch")
        self.assertEqual(record["comparison_status"], "failed_overlap_mismatch")
        self.assertIsNone(record["delta_native_minus_pyscf_hartree"])
        self.assertIn("overlap-matrix comparison", record["failure_reason"])

    def test_missing_native_overlap_matrix_cannot_pass(self):
        case = sample_case()
        case["native_result"].pop("overlap_matrix_row_major")
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["overlap_matrix_status"], "missing_native_overlap_matrix")
        self.assertEqual(record["comparison_status"], "failed_native")

    def test_nonfinite_native_overlap_matrix_cannot_pass(self):
        case = sample_case()
        case["native_result"]["overlap_matrix_row_major"][1] = float("nan")
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["overlap_matrix_status"], "invalid_native_overlap_matrix")
        self.assertEqual(record["comparison_status"], "failed_native")

    def test_discrepancy_fails_without_tolerance_relaxation(self):
        record = self.compare_with(sample_case(), fake_pyscf(energy=-1.0))
        self.assertEqual(record["comparison_status"], "failed_discrepancy")
        self.assertGreater(abs(record["delta_native_minus_pyscf_hartree"]), 1e-6)
        self.assertIn("exceeds", record["failure_reason"])

    def test_basis_function_count_mismatch_cannot_pass(self):
        case = sample_case()
        case["native_result"]["basis_functions"] = 3
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["comparison_status"], "failed_basis_mismatch")
        self.assertIn("not apples-to-apples", record["failure_reason"])

    def test_component_mismatch_fails_even_when_total_matches(self):
        case = sample_case()
        case["native_result"]["nuclear_repulsion_hartree"] += 1e-3
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["comparison_status"], "failed_discrepancy")
        self.assertAlmostEqual(record["delta_native_minus_pyscf_hartree"], 0.0)
        self.assertGreater(abs(record["delta_nuclear_repulsion_hartree"]), 1e-6)

    def test_derived_energy_overflow_is_reported_without_non_json_numbers(self):
        record = self.compare_with(sample_case(), fake_pyscf(energy=1e308))
        self.assertEqual(record["comparison_status"], "failed_non_finite_comparison")
        self.assertIsNone(record["delta_native_minus_pyscf_hartree"])
        json.dumps(record, allow_nan=False)

    def test_native_nonconvergence_cannot_pass(self):
        case = sample_case()
        case["native_result"]["status"] = "non_converged"
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["comparison_status"], "failed_native")
        self.assertIsNone(record["delta_native_minus_pyscf_hartree"])

    def test_nonfinite_native_energy_cannot_pass(self):
        case = sample_case()
        case["native_result"]["total_energy_hartree"] = float("nan")
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["comparison_status"], "failed_native")
        self.assertIsNone(record["native_total_energy_hartree"])

    def test_failed_reference_is_retained(self):
        record = self.compare_with(
            sample_case(),
            fake_pyscf(error=RuntimeError("simulated backend error")),
        )
        self.assertEqual(record["reference_status"], "failed")
        self.assertEqual(record["comparison_status"], "not_comparable")
        self.assertIn("simulated backend error", record["failure_reason"])

    def test_invalid_unit_is_not_silently_converted(self):
        case = sample_case()
        case["coordinate_unit"] = "angstrom"
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["reference_status"], "failed")
        self.assertEqual(record["comparison_status"], "not_comparable")
        self.assertIn("explicitly 'bohr'", record["failure_reason"])

    def test_symbol_must_match_declared_atomic_number(self):
        case = sample_case()
        # Keep a valid even electron count so validation reaches the element map.
        for atom in case["molecule"]["atoms"]:
            atom["atomic_number"] = 6
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["reference_status"], "failed")
        self.assertIn("do not match", record["failure_reason"])

    def test_rhf_rejects_open_shell_multiplicity(self):
        case = sample_case()
        case["molecule"]["multiplicity"] = 2
        record = self.compare_with(case, fake_pyscf())
        self.assertEqual(record["reference_status"], "failed")
        self.assertIn("requires multiplicity 1", record["failure_reason"])

    def test_nonconverged_reference_is_not_comparable(self):
        record = self.compare_with(sample_case(), fake_pyscf(converged=False))
        self.assertEqual(record["reference_status"], "not_converged")
        self.assertEqual(record["comparison_status"], "not_comparable")

    def test_nonfinite_and_bool_values_are_not_numbers(self):
        self.assertIsNone(qc.finite_float(float("inf")))
        self.assertIsNone(qc.finite_float(float("nan")))
        self.assertIsNone(qc.finite_float(10**10000))
        self.assertIsNone(qc.finite_float(True))
        self.assertEqual(qc.finite_float(1), 1.0)

    def test_cli_cannot_loosen_the_comparison_tolerance(self):
        with self.assertRaises(SystemExit) as raised:
            qc.main(["--input", "/unused/native.json", "--tolerance-hartree", "0.001"])
        self.assertEqual(raised.exception.code, 2)

    def test_cli_refuses_to_overwrite_native_input_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.json"
            path.write_text("preserve native evidence", encoding="utf-8")
            with self.assertRaises(SystemExit) as raised:
                qc.main(["--input", str(path), "--output", str(path)])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve native evidence")

    def test_source_binding_requires_current_head_and_clean_worktree(self):
        source_revision = "0123456789abcdef0123456789abcdef01234567"
        source_tree = "89abcdef0123456789abcdef0123456789abcdef"
        with patch.object(
            qc.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0, source_revision + "\n", ""),
                subprocess.CompletedProcess([], 0, source_tree + "\n", ""),
                subprocess.CompletedProcess([], 0, source_revision + "\n", ""),
                subprocess.CompletedProcess([], 0, source_tree + "\n", ""),
                subprocess.CompletedProcess([], 0, "", ""),
            ],
        ) as run:
            qc.verify_source_binding(source_revision, source_tree)
        self.assertEqual(run.call_count, 5)
        self.assertIn(source_revision, run.call_args_list[0].args[0][3])
        self.assertEqual(run.call_args_list[-1].args[0][1:3], ["status", "--porcelain"])

    def test_source_binding_rejects_a_report_for_another_head(self):
        source_revision = "0123456789abcdef0123456789abcdef01234567"
        source_tree = "89abcdef0123456789abcdef0123456789abcdef"
        other_commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        with patch.object(
            qc.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0, source_revision + "\n", ""),
                subprocess.CompletedProcess([], 0, source_tree + "\n", ""),
                subprocess.CompletedProcess([], 0, other_commit + "\n", ""),
                subprocess.CompletedProcess([], 0, source_tree + "\n", ""),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "does not equal the current Git HEAD"):
                qc.verify_source_binding(source_revision, source_tree)

    def test_source_binding_rejects_dirty_current_worktree(self):
        source_revision = "0123456789abcdef0123456789abcdef01234567"
        source_tree = "89abcdef0123456789abcdef0123456789abcdef"
        with patch.object(
            qc.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0, source_revision + "\n", ""),
                subprocess.CompletedProcess([], 0, source_tree + "\n", ""),
                subprocess.CompletedProcess([], 0, source_revision + "\n", ""),
                subprocess.CompletedProcess([], 0, source_tree + "\n", ""),
                subprocess.CompletedProcess([], 0, " M src/lib.rs\n", ""),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "worktree is dirty"):
                qc.verify_source_binding(source_revision, source_tree)

    def test_source_binding_rejects_a_tree_mismatch(self):
        source_revision = "0123456789abcdef0123456789abcdef01234567"
        source_tree = "89abcdef0123456789abcdef0123456789abcdef"
        with patch.object(
            qc.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess([], 0, source_revision + "\n", ""),
                subprocess.CompletedProcess([], 0, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", ""),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "does not match the tree"):
                qc.verify_source_binding(source_revision, source_tree)

    def test_input_digest_binds_exact_bytes_and_schema_is_checked(self):
        valid = {
            "schema_version": 2,
            "producer": "symthaea-quantum-chemistry",
            "coordinate_unit": "bohr",
            "method": "RHF",
            "source_revision": "0123456789abcdef0123456789abcdef01234567",
            "source_tree_sha": "89abcdef0123456789abcdef0123456789abcdef",
            "worktree_clean": True,
            "cases": [sample_case()],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            _, digest = qc.load_input(path)
            self.assertEqual(len(digest), 64)
            path.write_text(json.dumps({"schema_version": 99, "cases": [sample_case()]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported input schema"):
                qc.load_input(path)

            path.write_text(
                json.dumps({
                    "schema_version": 2,
                    "source_revision": "main",
                    "cases": [sample_case()],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "full 40-character Git commit SHA"):
                qc.load_input(path)

            path.write_text(
                json.dumps({
                    "schema_version": True,
                    "source_revision": "0123456789abcdef0123456789abcdef01234567",
                    "source_tree_sha": "89abcdef0123456789abcdef0123456789abcdef",
                    "worktree_clean": True,
                    "cases": [sample_case()],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported input schema"):
                qc.load_input(path)

            path.write_text(
                json.dumps({
                    "schema_version": 2,
                    "source_revision": "0123456789abcdef0123456789abcdef01234567",
                    "source_tree_sha": "89abcdef0123456789abcdef0123456789abcdef",
                    "worktree_clean": False,
                    "cases": [sample_case()],
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "worktree_clean=true"):
                qc.load_input(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
