#!/usr/bin/env python3
"""Fail-closed validator for the preregistered Matter Si-001 execution recipe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

SCHEMA = "symthaea.matter.reference-si-001-execution-recipe/v1"
QE_VERSION = "7.5"
AIIDA_VERSION = "2.9.2"
PLUGIN_VERSION = "5.0.0"
SSSP_FAMILY = "SSSP/1.3/PBE/precision"
ENERGY_TOL = 1e-3
FORCE_TOL = 1e-2
STRESS_TOL_GPA = 0.1
CELL_CHANGE_TOL = 1e-4
IMAGINARY_TOL_THz = 0.1
BOHR_ANGSTROM = 0.529177210903
RY_EV = 13.605693122994


class ProtocolError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot read protocol: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError("protocol root must be an object")
    return value


def _eq(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ProtocolError(f"{name}: expected {expected!r}, got {actual!r}")


def _finite(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{name} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"{name} must be numeric") from error
    if not math.isfinite(numeric):
        raise ProtocolError(f"{name} must be finite")
    if positive and not numeric > 0.0:
        raise ProtocolError(f"{name} must be positive")
    if nonnegative and numeric < 0.0:
        raise ProtocolError(f"{name} must be nonnegative")
    return numeric


def _det3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def _validate_structure(protocol: dict[str, Any]) -> None:
    target = protocol["target"]
    _eq(target["atomic_numbers"], [14], "target atomic numbers")
    _eq(target["prototype"], "diamond", "target prototype")
    structure = target["initial_structure"]
    _eq(structure["qe_ibrav"], 2, "initial qe_ibrav")
    a = _finite(structure["conventional_lattice_parameter_angstrom"], "initial lattice parameter", positive=True)
    _eq(a, 5.6, "preregistered starting lattice parameter")
    cell = structure["cell_vectors_angstrom"]
    if not isinstance(cell, list) or len(cell) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in cell):
        raise ProtocolError("initial cell must be 3x3")
    cell_f = [[_finite(x, "cell component") for x in row] for row in cell]
    if abs(abs(_det3(cell_f)) - a**3 / 4.0) > 1e-10:
        raise ProtocolError("initial cell volume is not the preregistered fcc primitive volume")
    expected_cell = [[-a / 2, 0.0, a / 2], [0.0, a / 2, a / 2], [-a / 2, a / 2, 0.0]]
    if cell_f != expected_cell:
        raise ProtocolError("initial cell vectors drifted from the QE-compatible fcc primitive convention")
    sites = structure["sites_fractional"]
    expected_sites = [
        {"atomic_number": 14, "fractional": [0.0, 0.0, 0.0]},
        {"atomic_number": 14, "fractional": [0.25, 0.25, 0.25]},
    ]
    _eq(sites, expected_sites, "diamond basis")


def _validate_environment(protocol: dict[str, Any]) -> None:
    software = protocol["software"]
    _eq(software["quantum_espresso_version"], QE_VERSION, "QE version")
    _eq(software["aiida_core_version"], AIIDA_VERSION, "AiiDA version")
    _eq(software["aiida_quantumespresso_version"], PLUGIN_VERSION, "QE plugin version")
    _eq(software["python_environment"], "exact committed uv.lock required", "Python environment policy")
    _eq(software["calculation_caching"], "disabled", "calculation caching policy")
    pseudo = protocol["pseudopotential"]
    _eq(pseudo["family"], SSSP_FAMILY, "SSSP family")
    _eq(pseudo["element"], "Si", "pseudopotential element")


def _validate_convergence(protocol: dict[str, Any]) -> None:
    common = protocol["electronic_common"]
    _eq(common["input_dft"], "PBE", "DFT functional")
    _eq(common["occupations"], "fixed", "occupations")
    _eq(common["scf_must_converge"], True, "SCF must converge")
    if _finite(common["conv_thr_ry"], "SCF threshold", positive=True) > 1e-10:
        raise ProtocolError("SCF threshold became looser than preregistered 1e-10 Ry")
    _eq(common["kpoint_offset_fractional"], [0.5, 0.5, 0.5], "common k-point offset")

    cutoff = protocol["cutoff_convergence"]
    _eq(cutoff["fixed_kpoint_mesh"], [6, 6, 6], "cutoff bridge k mesh")
    wf = cutoff["ecutwfc_multipliers_of_sssp_recommendation"]
    rho = cutoff["ecutrho_multipliers_of_sssp_recommendation"]
    _eq(wf, [0.75, 1.0, 1.25, 1.5], "cutoff schedule")
    _eq(rho, wf, "charge-density cutoff schedule")
    _eq(cutoff["run_all_samples_even_if_early_plateau_is_observed"], True, "cutoff full-sweep rule")
    _eq(cutoff["acceptance"]["energy_tolerance_ev_per_atom"], ENERGY_TOL, "cutoff energy tolerance")
    _eq(cutoff["acceptance"]["required_final_consecutive_deltas_within_tolerance"], 2, "cutoff plateau count")

    kpoint = protocol["kpoint_convergence"]
    meshes = kpoint["meshes"]
    _eq(meshes, [[6, 6, 6], [8, 8, 8], [10, 10, 10], [12, 12, 12]], "k-point schedule")
    _eq(kpoint["offset_fractional"], [0.5, 0.5, 0.5], "k-point sweep offset")
    _eq(kpoint["ecutwfc_multiplier_of_sssp_recommendation"], wf[-1], "cutoff/k-point bridge ecutwfc")
    _eq(kpoint["ecutrho_multiplier_of_sssp_recommendation"], rho[-1], "cutoff/k-point bridge ecutrho")
    _eq(meshes[0], cutoff["fixed_kpoint_mesh"], "cutoff/k-point bridge mesh")
    _eq(kpoint["run_all_samples_even_if_early_plateau_is_observed"], True, "k-point full-sweep rule")
    _eq(kpoint["acceptance"]["energy_tolerance_ev_per_atom"], ENERGY_TOL, "k-point energy tolerance")
    _eq(kpoint["acceptance"]["required_final_consecutive_deltas_within_tolerance"], 2, "k-point plateau count")

    production = protocol["production_resolution"]
    _eq(production["ecutwfc_multiplier_of_sssp_recommendation"], wf[-1], "production ecutwfc")
    _eq(production["ecutrho_multiplier_of_sssp_recommendation"], rho[-1], "production ecutrho")
    _eq(production["kpoint_mesh"], meshes[-1], "production k mesh")
    _eq(production["kpoint_offset_fractional"], [0.5, 0.5, 0.5], "production k offset")


def _validate_relaxation(protocol: dict[str, Any]) -> None:
    relax = protocol["relaxation"]
    _eq(relax["calculation"], "vc-relax", "relaxation type")
    _eq(relax["qe_ibrav"], 2, "relaxation ibrav")
    _eq(relax["ion_dynamics"], "bfgs", "ion dynamics")
    _eq(relax["cell_dynamics"], "bfgs", "cell dynamics")
    _eq(relax["cell_dofree"], "ibrav", "cell_dofree")
    _eq(relax["target_pressure_kbar"], 0.0, "target pressure")
    pressure_threshold = _finite(relax["press_conv_thr_kbar"], "pressure threshold", positive=True)
    if pressure_threshold > STRESS_TOL_GPA * 10.0:  # 1 GPa = 10 kbar
        raise ProtocolError("QE pressure threshold is looser than the Matter stress criterion")
    force_threshold = _finite(relax["forc_conv_thr_ry_per_bohr"], "QE force threshold", positive=True)
    force_threshold_ev_ang = force_threshold * RY_EV / BOHR_ANGSTROM
    if force_threshold_ev_ang > FORCE_TOL:
        raise ProtocolError("QE force threshold is looser than the Matter force criterion")
    _finite(relax["etot_conv_thr_ry"], "QE relaxation energy threshold", positive=True)
    if int(relax["nstep"]) != 100:
        raise ProtocolError("relaxation nstep drifted")
    acceptance = relax["acceptance"]
    _eq(acceptance["max_force_ev_per_angstrom"], FORCE_TOL, "Matter force criterion")
    _eq(acceptance["max_abs_stress_gpa"], STRESS_TOL_GPA, "Matter stress criterion")
    _eq(acceptance["max_relative_cell_change"], CELL_CHANGE_TOL, "Matter cell-change criterion")
    _eq(protocol["production_scf"]["must_reuse_exact_final_relaxation_structure_node"], True, "exact structure reuse")
    _eq(protocol["production_scf"]["must_be_phonon_parent"], True, "phonon-parent rule")


def _validate_phonons(protocol: dict[str, Any]) -> None:
    phonons = protocol["phonons"]
    ph = phonons["ph"]
    _eq(ph["ldisp"], True, "ph ldisp")
    _eq(ph["q_grid"], [4, 4, 4], "DFPT q grid")
    _eq(ph["q_shift_fractional"], [0.0, 0.0, 0.0], "DFPT q shift")
    _eq(ph["lshift_q"], False, "q2r-compatible unshifted q grid")
    if _finite(ph["tr2_ph"], "ph SCF tolerance", positive=True) > 1e-14:
        raise ProtocolError("ph tr2_ph became looser than preregistered 1e-14")
    _eq(ph["epsil"], False, "Si-001 dielectric-response policy")
    _eq(phonons["q2r"]["zasr"], "crystal", "q2r ASR")
    matdyn = phonons["matdyn"]
    _eq(matdyn["asr"], "crystal", "matdyn ASR")
    _eq(matdyn["stability_sampling_mesh"], [16, 16, 16], "matdyn stability mesh")
    _eq(matdyn["stability_sampling_shift_fractional"], [0.0, 0.0, 0.0], "matdyn stability shift")
    if math.prod(matdyn["stability_sampling_mesh"]) <= math.prod(ph["q_grid"]):
        raise ProtocolError("final interpolation grid must be denser than the DFPT grid")
    _eq(phonons["acceptance"]["imaginary_frequency_tolerance_thz"], IMAGINARY_TOL_THz, "imaginary-frequency tolerance")


def _validate_authority(protocol: dict[str, Any]) -> None:
    _eq(protocol["schema_version"], SCHEMA, "schema version")
    _eq(protocol["recipe_id"], "matter-reference-si-001", "recipe ID")
    _eq(protocol["benchmark_id"], "matter-reference-si-v1", "benchmark ID")
    _eq(protocol["authority"], "PreregisteredExecutionRecipeOnly", "authority")
    _eq(protocol["thermodynamic_phase_eligibility"], "WithheldByProtocol", "phase eligibility")
    if protocol.get("expected_results", "missing") is not None:
        raise ProtocolError("expected_results must remain null before execution")
    phase = protocol["phase_stability"]
    _eq(phase["coverage"], "withheld", "phase coverage")
    _eq(phase["convex_hull_eligible"], False, "hull eligibility")
    policy = protocol["execution_policy"]
    for key in (
        "no_parameter_changes_after_first_scientific_execution",
        "no_adaptive_extension_of_convergence_series",
        "all_declared_samples_are_required",
        "failed_criterion_means_benchmark_failure_not_retuning",
        "expected_lattice_constant_not_encoded",
        "expected_total_energy_not_encoded",
        "expected_phonon_frequencies_not_encoded",
        "reference_comparison_occurs_only_after_capsule_is_frozen",
    ):
        _eq(policy[key], True, key)


def validate(protocol: dict[str, Any]) -> None:
    required = {
        "schema_version", "recipe_id", "benchmark_id", "target_label", "authority",
        "thermodynamic_phase_eligibility", "expected_results", "target", "software",
        "pseudopotential", "electronic_common", "cutoff_convergence", "kpoint_convergence",
        "production_resolution", "relaxation", "production_scf", "phonons",
        "phase_stability", "execution_policy", "limitations",
    }
    if set(protocol) != required:
        raise ProtocolError(f"top-level key set drifted: {sorted(set(protocol) ^ required)}")
    _validate_authority(protocol)
    _validate_structure(protocol)
    _validate_environment(protocol)
    _validate_convergence(protocol)
    _validate_relaxation(protocol)
    _validate_phonons(protocol)


def _self_test(protocol_path: Path) -> None:
    original = _load(protocol_path)
    validate(original)
    mutant = json.loads(json.dumps(original))
    mutant["kpoint_convergence"]["meshes"].append([14, 14, 14])
    try:
        validate(mutant)
    except ProtocolError:
        pass
    else:
        raise AssertionError("adaptive/undeclared k-point extension must fail")
    mutant = json.loads(json.dumps(original))
    mutant["expected_results"] = {"lattice_constant_angstrom": 5.43}
    try:
        validate(mutant)
    except ProtocolError:
        pass
    else:
        raise AssertionError("pre-filled expected result must fail")
    mutant = json.loads(json.dumps(original))
    mutant["phonons"]["matdyn"]["stability_sampling_mesh"] = [4, 4, 4]
    try:
        validate(mutant)
    except ProtocolError:
        pass
    else:
        raise AssertionError("coarse-only phonon stability sampling must fail")
    print("Si-001 frozen execution recipe self-test: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.self_test:
            _self_test(args.protocol)
        else:
            validate(_load(args.protocol))
            print("Si-001 frozen execution recipe: VALID")
        return 0
    except ProtocolError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
