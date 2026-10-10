#!/usr/bin/env python3
"""Fail-closed contract validator for an orbital-ring fault-scenario manifest.

Validates input completeness and basic types only. It does not run orbital,
structural, electromagnetic, or fault-response physics.
"""
import argparse
import json
import math
import sys
from pathlib import Path

SCHEMA = "luminous-dynamics.orbital-ring-failure-scenario/v1"
SCENARIO_ID = "F-SKYHOOK-LOAD-STEP"
REQUIRED_FIELDS = (
    "inertial_frame",
    "support_frame_transform",
    "ring_radius_m",
    "ring_plane_orientation",
    "initial_ring_position_m",
    "initial_ring_velocity_m_s",
    "initial_equilibrium_evidence_ref",
    "skyhook_support_locations_rad",
    "support_load_before_n",
    "support_load_after_n",
    "load_change_profile",
    "load_step_time_s",
    "rider_spacing_m",
    "rider_force_law",
    "rider_stiffness_n_per_m",
    "rider_damping_n_s_per_m",
    "rider_max_coupling_force_n",
    "actuation_measurement_delay_s",
    "force_model_reference",
    "energy_boundary",
    "momentum_boundary",
    "acceptance_limits",
    "independent_reference_case",
)
UNKNOWN_STRINGS = {"", "unknown", "tbd", "todo", "unspecified", "not specified", "?"}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in UNKNOWN_STRINGS:
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


def vector_valid(value, size=3, unit_vector=False):
    if not isinstance(value, list) or len(value) != size or not all(finite_number(x) for x in value):
        return False
    if unit_vector:
        norm = math.sqrt(sum(x * x for x in value))
        return abs(norm - 1.0) <= 1e-6
    return True


def typed_errors(name, value):
    errors = []
    text_fields = {
        "inertial_frame", "initial_equilibrium_evidence_ref", "rider_force_law",
        "force_model_reference", "independent_reference_case",
    }
    positive_scalars = {
        "ring_radius_m", "rider_spacing_m", "rider_stiffness_n_per_m",
        "rider_max_coupling_force_n",
    }
    nonnegative_scalars = {
        "load_step_time_s", "rider_damping_n_s_per_m", "actuation_measurement_delay_s",
    }
    vec3_fields = {
        "initial_ring_position_m", "initial_ring_velocity_m_s",
        "support_load_before_n", "support_load_after_n",
    }
    if name in text_fields:
        if not isinstance(value, str) or is_missing(value):
            errors.append(f"{name} must be a specified non-empty string/reference")
    elif name in positive_scalars:
        if not finite_number(value) or value <= 0:
            errors.append(f"{name} must be finite and > 0")
    elif name in nonnegative_scalars:
        if not finite_number(value) or value < 0:
            errors.append(f"{name} must be finite and >= 0")
    elif name in vec3_fields:
        if not vector_valid(value, 3):
            errors.append(f"{name} must be a finite 3-vector")
    elif name == "ring_plane_orientation":
        if not isinstance(value, dict) or not vector_valid(value.get("normal_unit_vector_in_inertial_frame"), 3, True):
            errors.append("ring_plane_orientation.normal_unit_vector_in_inertial_frame must be a unit 3-vector")
        if not isinstance(value, dict) or not isinstance(value.get("plane_reference"), str) or is_missing(value.get("plane_reference")):
            errors.append("ring_plane_orientation.plane_reference must identify the orientation convention")
    elif name == "skyhook_support_locations_rad":
        if not isinstance(value, list) or not value or not all(finite_number(x) for x in value):
            errors.append("skyhook_support_locations_rad must be a non-empty list of finite angles")
    elif name == "support_frame_transform":
        required = {
            "origin_position_m": lambda x: vector_valid(x, 3),
            "origin_velocity_m_s": lambda x: vector_valid(x, 3),
            "orientation_quaternion_xyzw": lambda x: vector_valid(x, 4) and abs(math.sqrt(sum(y*y for y in x)) - 1.0) <= 1e-6,
            "angular_velocity_rad_s": lambda x: vector_valid(x, 3),
            "angular_acceleration_rad_s2": lambda x: vector_valid(x, 3),
        }
        if not isinstance(value, dict):
            errors.append("support_frame_transform must be an object")
        else:
            for key, validator in required.items():
                if key not in value or not validator(value[key]):
                    errors.append(f"support_frame_transform.{key} is missing or invalid")
    elif name == "load_change_profile":
        if not isinstance(value, dict) or value.get("kind") not in {"step", "ramp"}:
            errors.append("load_change_profile.kind must be step or ramp")
        else:
            duration = value.get("transition_duration_s")
            if not finite_number(duration) or duration < 0:
                errors.append("load_change_profile.transition_duration_s must be finite and >= 0")
            elif value.get("kind") == "step" and duration != 0:
                errors.append("step profile must have zero transition duration")
            elif value.get("kind") == "ramp" and duration <= 0:
                errors.append("ramp profile must have positive transition duration")
            ref = value.get("profile_reference")
            if not isinstance(ref, str) or is_missing(ref):
                errors.append("load_change_profile.profile_reference must identify the command profile")
    elif name == "energy_boundary":
        required = {
            "reference_frame": lambda x: isinstance(x, str) and not is_missing(x),
            "included_bodies": lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(y, str) and not is_missing(y) for y in x),
            "external_work_terms": lambda x: isinstance(x, list),
            "boundary_flux_terms": lambda x: isinstance(x, list),
            "scope_complete": lambda x: x is True,
        }
        if not isinstance(value, dict):
            errors.append("energy_boundary must be an object")
        else:
            for key, validator in required.items():
                if key not in value or not validator(value[key]):
                    errors.append(f"energy_boundary.{key} is missing or invalid")
    elif name == "momentum_boundary":
        required = {
            "reference_frame": lambda x: isinstance(x, str) and not is_missing(x),
            "included_bodies": lambda x: isinstance(x, list) and len(x) > 0 and all(isinstance(y, str) and not is_missing(y) for y in x),
            "external_force_terms": lambda x: isinstance(x, list),
            "mass_momentum_flux_terms": lambda x: isinstance(x, list),
            "scope_complete": lambda x: x is True,
        }
        if not isinstance(value, dict):
            errors.append("momentum_boundary must be an object")
        else:
            for key, validator in required.items():
                if key not in value or not validator(value[key]):
                    errors.append(f"momentum_boundary.{key} is missing or invalid")
    elif name == "acceptance_limits":
        limits = {
            "max_ring_offset_m": lambda x: finite_number(x) and x > 0,
            "max_support_miss_distance_m": lambda x: finite_number(x) and x > 0,
            "max_rider_force_n": lambda x: finite_number(x) and x > 0,
            "max_energy_residual_j": lambda x: finite_number(x) and x >= 0,
            "max_momentum_residual_kg_m_s": lambda x: finite_number(x) and x >= 0,
        }
        if not isinstance(value, dict):
            errors.append("acceptance_limits must be an object")
        else:
            for key, validator in limits.items():
                if key not in value or not validator(value[key]):
                    errors.append(f"acceptance_limits.{key} is missing or invalid")
    return errors


def assess(manifest):
    inputs = manifest.get("required_inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    blockers = [name for name in REQUIRED_FIELDS if name not in inputs or is_missing(inputs.get(name))]
    type_errors = []
    for name in REQUIRED_FIELDS:
        if name in inputs and not is_missing(inputs[name]):
            type_errors.extend(typed_errors(name, inputs[name]))
    structural_errors = []
    if manifest.get("schema") != SCHEMA:
        structural_errors.append("schema mismatch")
    if manifest.get("scenario_id") != SCENARIO_ID:
        structural_errors.append("scenario_id mismatch")
    if manifest.get("architecture_mode") != "continuous_skyhook_ors":
        structural_errors.append("architecture_mode must be continuous_skyhook_ors")
    if tuple(manifest.get("required_fields", ())) != REQUIRED_FIELDS:
        structural_errors.append("required_fields must match the validator's frozen contract exactly")
    if manifest.get("analysis_state") != "NotRun":
        structural_errors.append("analysis_state must remain NotRun in this contract-only gate")
    if manifest.get("result_artifacts") != []:
        structural_errors.append("result_artifacts must be empty before analysis is run")
    expected_state = "Blocked" if blockers or type_errors else "ReadyForIndependentReview"
    if manifest.get("declared_state") != expected_state:
        structural_errors.append("declared_state does not match required-input completeness and type checks")
    return {
        "schema": "luminous-dynamics.orbital-ring-manifest-validation/v1",
        "validation_state": "passed_contract_checks" if not structural_errors else "failed_contract_checks",
        "scenario_id": manifest.get("scenario_id"),
        "expected_gate_state": expected_state,
        "declared_gate_state": manifest.get("declared_state"),
        "analysis_state": manifest.get("analysis_state"),
        "required_input_count": len(REQUIRED_FIELDS),
        "missing_or_unknown_fields": blockers,
        "type_errors": type_errors,
        "blocking_field_count": len(set(blockers)),
        "structural_errors": structural_errors,
        "physics_claim": "none; no simulation or equilibrium calculation is performed",
    }


def complete_typed_fixture():
    return {
        "inertial_frame": "GCRF",
        "support_frame_transform": {
            "origin_position_m": [0.0, 0.0, 0.0],
            "origin_velocity_m_s": [0.0, 0.0, 0.0],
            "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
            "angular_acceleration_rad_s2": [0.0, 0.0, 0.0],
        },
        "ring_radius_m": 7.0e6,
        "ring_plane_orientation": {"normal_unit_vector_in_inertial_frame": [0.0, 0.0, 1.0], "plane_reference": "declared inertial axes"},
        "initial_ring_position_m": [7.0e6, 0.0, 0.0],
        "initial_ring_velocity_m_s": [0.0, 7500.0, 0.0],
        "initial_equilibrium_evidence_ref": "fixture://independent-equilibrium-placeholder",
        "skyhook_support_locations_rad": [0.0, 1.0, 2.0],
        "support_load_before_n": [0.0, 1.0e6, 0.0],
        "support_load_after_n": [0.0, 5.0e5, 0.0],
        "load_change_profile": {"kind": "step", "transition_duration_s": 0.0, "profile_reference": "fixture://load-step"},
        "load_step_time_s": 10.0,
        "rider_spacing_m": 1.0e4,
        "rider_force_law": "fixture://specified-linear-rider-law",
        "rider_stiffness_n_per_m": 1.0e6,
        "rider_damping_n_s_per_m": 1.0e4,
        "rider_max_coupling_force_n": 1.0e8,
        "actuation_measurement_delay_s": 0.1,
        "force_model_reference": "fixture://independent-force-model",
        "energy_boundary": {"reference_frame": "GCRF", "included_bodies": ["ring", "support", "riders"], "external_work_terms": [], "boundary_flux_terms": [], "scope_complete": True},
        "momentum_boundary": {"reference_frame": "GCRF", "included_bodies": ["ring", "support", "riders"], "external_force_terms": [], "mass_momentum_flux_terms": [], "scope_complete": True},
        "acceptance_limits": {"max_ring_offset_m": 100.0, "max_support_miss_distance_m": 1000.0, "max_rider_force_n": 1.0e8, "max_energy_residual_j": 1.0e6, "max_momentum_residual_kg_m_s": 1.0e6},
        "independent_reference_case": "fixture://independent-reference-placeholder",
    }


def self_test():
    checks = []

    def record(name, ok):
        checks.append({"id": name, "status": "passed" if ok else "failed"})

    template = {
        "schema": SCHEMA, "scenario_id": SCENARIO_ID,
        "architecture_mode": "continuous_skyhook_ors",
        "declared_state": "Blocked", "analysis_state": "NotRun",
        "result_artifacts": [], "required_fields": list(REQUIRED_FIELDS),
        "required_inputs": {name: None for name in REQUIRED_FIELDS},
    }
    result = assess(template)
    record("contract_declares_23_required_fields", len(REQUIRED_FIELDS) == 23)
    record("empty_template_is_contract_valid_but_blocked", result["validation_state"] == "passed_contract_checks" and result["expected_gate_state"] == "Blocked")
    record("empty_template_blocks_all_required_fields", result["blocking_field_count"] == len(REQUIRED_FIELDS))
    record("analysis_not_run_and_no_physics_claim", result["analysis_state"] == "NotRun" and result["physics_claim"].startswith("none"))
    for field in REQUIRED_FIELDS:
        candidate = json.loads(json.dumps(template))
        candidate["required_inputs"][field] = 1.0
        candidate["declared_state"] = "Blocked"
        assessed = assess(candidate)
        record(f"missing_{field}_keeps_gate_blocked", assessed["expected_gate_state"] == "Blocked")
    candidate = json.loads(json.dumps(template))
    candidate["declared_state"] = "ReadyForIndependentReview"
    result = assess(candidate)
    record("premature_state_upgrade_rejected", result["validation_state"] == "failed_contract_checks" and result["expected_gate_state"] == "Blocked")
    candidate = json.loads(json.dumps(template))
    candidate["required_inputs"]["rider_force_law"] = "TBD"
    result = assess(candidate)
    record("tbd_force_law_still_blocks", result["expected_gate_state"] == "Blocked")
    candidate = json.loads(json.dumps(template))
    candidate["required_inputs"] = complete_typed_fixture()
    candidate["declared_state"] = "ReadyForIndependentReview"
    result = assess(candidate)
    record("complete_typed_fixture_is_review_ready_not_passed", result["expected_gate_state"] == "ReadyForIndependentReview" and result["validation_state"] == "passed_contract_checks" and result["analysis_state"] == "NotRun")
    candidate = json.loads(json.dumps(template))
    candidate["required_inputs"] = complete_typed_fixture()
    candidate["required_inputs"]["ring_radius_m"] = "7000000"
    candidate["declared_state"] = "ReadyForIndependentReview"
    result = assess(candidate)
    record("wrong_numeric_type_rejected", result["validation_state"] == "failed_contract_checks" and bool(result["type_errors"]))
    candidate = json.loads(json.dumps(template))
    candidate["required_inputs"] = complete_typed_fixture()
    candidate["required_inputs"]["ring_plane_orientation"]["normal_unit_vector_in_inertial_frame"] = [0.0, 0.0, 2.0]
    candidate["declared_state"] = "Blocked"
    result = assess(candidate)
    record("non_unit_ring_plane_normal_rejected", result["expected_gate_state"] == "Blocked" and bool(result["type_errors"]))
    candidate = json.loads(json.dumps(template))
    candidate["required_inputs"] = complete_typed_fixture()
    candidate["required_inputs"]["load_change_profile"] = {"kind": "step", "transition_duration_s": 1.0, "profile_reference": "fixture://bad-step"}
    candidate["declared_state"] = "Blocked"
    result = assess(candidate)
    record("step_duration_inconsistency_rejected", result["expected_gate_state"] == "Blocked" and bool(result["type_errors"]))
    report = {
        "schema": "luminous-dynamics.orbital-ring-manifest-self-test/v1",
        "checks_passed": sum(c["status"] == "passed" for c in checks),
        "checks_failed": sum(c["status"] != "passed" for c in checks),
        "checks": checks,
        "interpretation": "Contract validation only; review-ready does not mean physically validated or safe.",
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["checks_failed"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", help="manifest JSON path")
    parser.add_argument("--self-test", action="store_true", help="run built-in fail-closed contract tests")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.manifest:
        parser.error("manifest path required unless --self-test is used")
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"validation_state": "failed_to_read_manifest", "error": str(exc)}, sort_keys=True))
        return 2
    result = assess(manifest)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["validation_state"] == "passed_contract_checks" else 1


if __name__ == "__main__":
    raise SystemExit(main())
