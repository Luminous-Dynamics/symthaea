#!/usr/bin/env python3
"""Compare Symthaea closed-shell LDA/VWN5 grid sweeps against PySCF/Libxc.

This deliberately reconstructs the same unpruned radial/angular rule and the
original Becke partition used by Symthaea, including its per-element radial
scale. It compares the same geometry, charge, multiplicity, STO-3G basis, and
LDA_X + LDA_C_VWN functional at each grid preset.

The energy threshold is an implementation-agreement gate for matched grids,
not a chemical-accuracy claim. Electron-count integration errors are reported
separately because a coarse grid can be SCF-converged while still integrating
the density imperfectly.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pyscf
    import scipy
    from pyscf import dft, gto
    from pyscf.dft import gen_grid, libxc
except ImportError as exc:
    print(f"PySCF, NumPy, and SciPy are required: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc


GRID_SETTINGS = {
    "coarse": (20, 6),
    "medium": (35, 26),
    "fine": (50, 50),
    "high_accuracy": (75, 302),
}
GRID_ORDER = ("coarse", "medium", "fine", "high_accuracy")
ELEMENT_SYMBOLS = {1: "H", 2: "He", 8: "O"}
HARTREE_TO_KCAL_MOL = 627.5094740631


def symthaea_becke_radial(n: int, charge: int, *args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce Symthaea's Becke-mapped second-kind Gauss-Chebyshev radial rule."""
    if n < 1:
        raise ValueError(f"radial point count must be positive, got {n}")
    if charge == 1:
        radius = 0.661
    elif charge == 2:
        radius = 0.567
    elif charge in (3, 4):
        radius = 2.268
    elif 5 <= charge <= 10:
        radius = 1.323
    else:
        radius = 1.5

    i = np.arange(1, n + 1, dtype=float)
    xi = np.cos(i * math.pi / (n + 1.0))
    radial = radius * (1.0 + xi) / (1.0 - xi)
    dr_dxi = 2.0 * radius / ((1.0 - xi) ** 2)
    radial_weights = (math.pi / (n + 1.0)) * np.sqrt(1.0 - xi * xi) * dr_dxi
    return radial, radial_weights


def require_finite(name: str, value: float) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} is not finite: {converted}")
    return converted


def parse_native(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"native evidence JSON is missing: {path}")
    native = json.loads(path.read_text(encoding="utf-8"))
    if native.get("schema") != "symthaea.qc-dft-molecular-grid-evidence.v1":
        raise ValueError(f"unexpected native evidence schema: {native.get('schema')!r}")
    for field in ("source_commit", "source_tree"):
        value = native.get(field)
        if not isinstance(value, str) or len(value) != 40:
            raise ValueError(f"native evidence has no source-bound {field}")
    if len(native.get("records", [])) != 12:
        raise ValueError(f"expected 12 native records, got {len(native.get('records', []))}")
    return native


def build_pyscf_reference(record: dict[str, Any]) -> dict[str, Any]:
    molecule_name = str(record["molecule"])
    grid_name = str(record["grid_quality"])
    if grid_name not in GRID_SETTINGS:
        raise ValueError(f"unknown grid quality {grid_name!r}")
    if record.get("basis") != "STO-3G":
        raise ValueError(f"{molecule_name}: expected native basis STO-3G, got {record.get('basis')!r}")
    if record.get("functional") != "Slater exchange + VWN5 correlation (LDA)":
        raise ValueError(
            f"{molecule_name}: unexpected native functional metadata {record.get('functional')!r}"
        )
    radial_count, angular_count = GRID_SETTINGS[grid_name]
    expected_points_per_atom = radial_count * angular_count
    if int(record.get("expected_grid_points_per_atom", -1)) != expected_points_per_atom:
        raise ValueError(
            f"{molecule_name}/{grid_name}: reported per-atom grid size "
            f"{record.get('expected_grid_points_per_atom')!r} != expected {expected_points_per_atom}"
        )
    geometry = record["geometry"]
    if geometry.get("coordinates_unit") != "bohr":
        raise ValueError(
            f"{molecule_name}: expected Bohr coordinates, got {geometry.get('coordinates_unit')!r}"
        )
    geometry_electrons = int(geometry["electron_count"])
    metadata_electrons = record.get("expected_electron_count")
    if metadata_electrons is not None and int(metadata_electrons) != geometry_electrons:
        raise ValueError(
            f"{molecule_name}: native record electron count {metadata_electrons} "
            f"disagrees with geometry count {geometry_electrons}"
        )
    atoms = []
    for atom in geometry["atoms"]:
        atomic_number = int(atom["atomic_number"])
        if atomic_number not in ELEMENT_SYMBOLS:
            raise ValueError(f"no benchmark symbol mapping for atomic number {atomic_number}")
        coordinates = tuple(require_finite("coordinate", v) for v in atom["position"])
        atoms.append((ELEMENT_SYMBOLS[atomic_number], coordinates))

    charge = int(geometry["charge"])
    multiplicity = int(geometry["multiplicity"])
    expected_electrons = geometry_electrons
    if multiplicity != 1:
        raise ValueError(
            f"{molecule_name}: this comparison lane supports closed-shell singlets only, "
            f"got multiplicity {multiplicity}"
        )
    if not atoms:
        raise ValueError(f"{molecule_name}: geometry contains no atoms")
    molecule = gto.Mole()
    molecule.atom = atoms
    molecule.unit = "Bohr"
    molecule.basis = "sto-3g"
    molecule.charge = charge
    molecule.spin = multiplicity - 1
    molecule.verbose = 0
    molecule.build()

    if molecule.nelectron != expected_electrons:
        raise ValueError(
            f"{molecule_name}: PySCF electron count {molecule.nelectron} "
            f"!= native geometry count {expected_electrons}"
        )
    parsed_x_id, parsed_c_id = libxc.parse_xc_name("lda,vwn")
    xc_codes = getattr(libxc, "XC_CODES", {})
    if parsed_x_id != xc_codes.get("LDA_X") or parsed_c_id != xc_codes.get("LDA_C_VWN"):
        raise RuntimeError(
            f"PySCF did not resolve lda,vwn to LDA_X + LDA_C_VWN: "
            f"{parsed_x_id}, {parsed_c_id}"
        )

    mean_field = dft.RKS(molecule)
    mean_field.xc = "lda,vwn"
    mean_field.conv_tol = 1e-10
    mean_field.conv_tol_grad = 1e-9
    mean_field.max_cycle = 200
    mean_field.grids.atom_grid = (radial_count, angular_count)
    mean_field.grids.prune = None
    mean_field.grids.radi_method = symthaea_becke_radial
    mean_field.grids.becke_scheme = gen_grid.original_becke
    mean_field.grids.radii_adjust = None
    total_energy = float(mean_field.kernel())

    density_matrix = mean_field.make_rdm1()
    ao_grid = mean_field._numint.eval_ao(molecule, mean_field.grids.coords)
    rho_grid = mean_field._numint.eval_rho(
        molecule, ao_grid, density_matrix, xctype="LDA"
    )
    integrated_electron_count = float(np.dot(rho_grid, mean_field.grids.weights))
    expected_total_energy = require_finite(f"{molecule_name} PySCF total energy", total_energy)
    integrated_electron_count = require_finite(
        f"{molecule_name} PySCF grid electron count", integrated_electron_count
    )

    return {
        "molecule": molecule_name,
        "geometry_id": record["geometry_id"],
        "grid_quality": grid_name,
        "radial_points": radial_count,
        "angular_points": angular_count,
        "basis": "STO-3G",
        "functional": "LDA_X + LDA_C_VWN",
        "libxc_functional_ids": {"LDA_X": int(parsed_x_id), "LDA_C_VWN": int(parsed_c_id)},
        "charge": charge,
        "multiplicity": multiplicity,
        "expected_electron_count": expected_electrons,
        "total_energy_hartree": expected_total_energy,
        "nuclear_repulsion_hartree": float(mean_field.energy_nuc()),
        "scf_converged": bool(mean_field.converged),
        "scf_iterations": (
            int(mean_field.cycles) if getattr(mean_field, "cycles", None) is not None else None
        ),
        "n_grid_points": int(mean_field.grids.weights.size),
        "integrated_electron_count": integrated_electron_count,
        "integrated_electron_count_abs_error": abs(integrated_electron_count - expected_electrons),
    }


def compare_one(native_record: dict[str, Any], reference: dict[str, Any], energy_tolerance: float) -> dict[str, Any]:
    native_energy_raw = native_record.get("total_energy_hartree")
    native_energy = None if native_energy_raw is None else require_finite("native total energy", native_energy_raw)
    reference_energy = require_finite("PySCF total energy", reference["total_energy_hartree"])
    energy_error = None if native_energy is None else abs(native_energy - reference_energy)
    native_count_raw = native_record.get("integrated_electron_count")
    native_count = None if native_count_raw is None else require_finite("native integrated electron count", native_count_raw)
    expected_count = int(
        native_record.get(
            "expected_electron_count",
            native_record.get("geometry", {}).get("electron_count"),
        )
    )
    reference_count = float(reference["integrated_electron_count"])
    native_count_error_raw = native_record.get("integrated_electron_count_abs_error")
    native_count_error = (
        None if native_count_error_raw is None else require_finite("native count error", native_count_error_raw)
    )
    reference_count_error = float(reference["integrated_electron_count_abs_error"])
    status = native_record.get("status")
    native_converged = status == "scf_converged"
    reference_converged = bool(reference["scf_converged"])
    receipt_present = bool(native_record.get("solver_receipt_present"))
    energy_within_tolerance = energy_error is not None and energy_error <= energy_tolerance
    passed = (
        native_converged
        and reference_converged
        and receipt_present
        and energy_within_tolerance
        and native_count is not None
        and native_count_error is not None
    )
    return {
        "molecule": native_record["molecule"],
        "geometry_id": native_record["geometry_id"],
        "grid_quality": native_record["grid_quality"],
        "native_scf_status": status,
        "native_solver_receipt_present": receipt_present,
        "pyscf_scf_converged": reference_converged,
        "native_total_energy_hartree": native_energy,
        "pyscf_total_energy_hartree": reference_energy,
        "energy_abs_error_hartree": energy_error,
        "energy_abs_error_kcal_mol": None if energy_error is None else energy_error * HARTREE_TO_KCAL_MOL,
        "energy_tolerance_hartree": energy_tolerance,
        "energy_within_tolerance": energy_within_tolerance,
        "expected_electron_count": expected_count,
        "native_integrated_electron_count": native_count,
        "pyscf_integrated_electron_count": reference_count,
        "native_electron_count_abs_error": native_count_error,
        "pyscf_electron_count_abs_error": reference_count_error,
        "integrated_electron_count_difference_abs": (
            None if native_count is None else abs(native_count - reference_count)
        ),
        "native_grid_points": native_record.get("n_grid_points"),
        "pyscf_grid_points": reference.get("n_grid_points"),
        "grid_point_count_matches": (
            native_record.get("n_grid_points") == reference.get("n_grid_points")
        ),
        "grid_point_count_note": (
            "Count equality is diagnostic only: Symthaea omits weights with "
            "absolute value <= 1e-20, while a reference generator may retain zero-weight points."
        ),
        "pass": passed,
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _absolute_grid_deltas(energies: dict[str, float | None]) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    previous_energy: float | None = None
    for quality in GRID_ORDER:
        energy = energies.get(quality)
        deltas[quality] = (
            None if energy is None or previous_energy is None
            else abs(energy - previous_energy)
        )
        previous_energy = energy
    return deltas


def build_grid_convergence(native_records: list[dict[str, Any]],
                           reference_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report within-engine grid deltas separately from cross-engine energy errors."""
    molecule_order = []
    for record in native_records:
        name = record.get("molecule")
        if name not in molecule_order:
            molecule_order.append(name)

    native_by_key = {
        (record.get("molecule"), record.get("grid_quality")): record
        for record in native_records
    }
    reference_by_key = {
        (record.get("molecule"), record.get("grid_quality")): record
        for record in reference_records
    }
    curves = []
    for molecule_name in molecule_order:
        native_energy = {
            quality: _finite_or_none(
                native_by_key.get((molecule_name, quality), {}).get("total_energy_hartree")
            )
            for quality in GRID_ORDER
        }
        reference_energy = {
            quality: _finite_or_none(
                reference_by_key.get((molecule_name, quality), {}).get("total_energy_hartree")
            )
            for quality in GRID_ORDER
        }
        native_count_error = {
            quality: _finite_or_none(
                native_by_key.get((molecule_name, quality), {}).get(
                    "integrated_electron_count_abs_error"
                )
            )
            for quality in GRID_ORDER
        }
        reference_count_error = {
            quality: _finite_or_none(
                reference_by_key.get((molecule_name, quality), {}).get(
                    "integrated_electron_count_abs_error"
                )
            )
            for quality in GRID_ORDER
        }
        native_status = {
            quality: native_by_key.get((molecule_name, quality), {}).get("status")
            for quality in GRID_ORDER
        }
        reference_converged = {
            quality: reference_by_key.get((molecule_name, quality), {}).get("scf_converged")
            for quality in GRID_ORDER
        }

        native_high = native_energy.get("high_accuracy")
        reference_high = reference_energy.get("high_accuracy")
        curves.append({
            "molecule": molecule_name,
            "grid_order": list(GRID_ORDER),
            "native_scf_status_by_grid": native_status,
            "pyscf_scf_converged_by_grid": reference_converged,
            "native_energy_hartree_by_grid": native_energy,
            "pyscf_energy_hartree_by_grid": reference_energy,
            "native_delta_to_previous_grid_hartree": _absolute_grid_deltas(native_energy),
            "pyscf_delta_to_previous_grid_hartree": _absolute_grid_deltas(reference_energy),
            "native_delta_to_high_accuracy_hartree": {
                quality: (
                    None if native_energy[quality] is None or native_high is None
                    else abs(native_energy[quality] - native_high)
                )
                for quality in GRID_ORDER
            },
            "pyscf_delta_to_high_accuracy_hartree": {
                quality: (
                    None if reference_energy[quality] is None or reference_high is None
                    else abs(reference_energy[quality] - reference_high)
                )
                for quality in GRID_ORDER
            },
            "native_integrated_electron_count_abs_error_by_grid": native_count_error,
            "pyscf_integrated_electron_count_abs_error_by_grid": reference_count_error,
            "interpretation":
                "Within-engine energy deltas isolate grid sensitivity only for cases whose SCF status is converged; the high-accuracy grid is a reference setting, not exact truth.",
        })
    return curves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--energy-tolerance-hartree", type=float, default=2e-4)
    args = parser.parse_args()
    if not math.isfinite(args.energy_tolerance_hartree) or args.energy_tolerance_hartree <= 0:
        raise ValueError("--energy-tolerance-hartree must be finite and positive")

    native = parse_native(args.native_report)
    references = []
    comparisons = []
    failure_reasons = []
    for native_record in native["records"]:
        try:
            reference = build_pyscf_reference(native_record)
            comparison = compare_one(native_record, reference, args.energy_tolerance_hartree)
            references.append(reference)
            comparisons.append(comparison)
            if not comparison["pass"]:
                failure_reasons.append(
                    f"{comparison['molecule']}/{comparison['grid_quality']}: "
                    f"native={comparison['native_scf_status']}, "
                    f"PySCF converged={comparison['pyscf_scf_converged']}, "
                    f"energy error={comparison['energy_abs_error_hartree']}"
                )
        except Exception as exc:
            failure_reasons.append(
                f"{native_record.get('molecule')}/{native_record.get('grid_quality')}: "
                f"{type(exc).__name__}: {exc}"
            )
            references.append({
                "molecule": native_record.get("molecule"),
                "grid_quality": native_record.get("grid_quality"),
                "error": f"{type(exc).__name__}: {exc}",
            })
            comparisons.append({
                "molecule": native_record.get("molecule"),
                "grid_quality": native_record.get("grid_quality"),
                "pass": False,
                "error": f"{type(exc).__name__}: {exc}",
            })

    passed_count = sum(bool(record.get("pass")) for record in comparisons)
    result = {
        "schema": "symthaea.qc-dft-pyscf-grid-comparison.v1",
        "native_source_commit": native["source_commit"],
        "native_source_tree": native["source_tree"],
        "pyscf_version": getattr(pyscf, "__version__", "unknown"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "basis": "STO-3G",
        "functional": "LDA_X + LDA_C_VWN (PySCF alias lda,vwn)",
        "grid_matching": {
            "radial_rule": "Symthaea Becke map + second-kind Gauss-Chebyshev nodes and weights",
            "angular_rule": "Lebedev–Laikov node count selected by the Symthaea preset; no pruning",
            "partition": "original Becke three-pass polynomial; no radius adjustment",
            "radial_scales": "exact hard-coded per-element scales used by Symthaea",
            "intent": "matched quadrature isolates implementation differences from grid-choice differences",
        },
        "energy_tolerance_hartree": args.energy_tolerance_hartree,
        "scope": "matched-grid implementation comparison; not a chemical-accuracy claim",
        "summary": {
            "cases": len(comparisons),
            "passed": passed_count,
            "failed": len(comparisons) - passed_count,
            "pass": bool(comparisons) and passed_count == len(comparisons),
            "failure_reasons": failure_reasons,
        },
        "pyscf_reference_records": references,
        "comparisons": comparisons,
        "grid_convergence": build_grid_convergence(native["records"], references),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["summary"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
