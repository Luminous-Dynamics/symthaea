#!/usr/bin/env python3
"""Fail-closed contract validator for an orbital-ring fault-scenario manifest.

This validates scenario completeness and claim state only. It does not run
orbital, structural, electromagnetic, or fault-response physics.
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


def is_missing(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in UNKNOWN_STRINGS:
        return True
    if isinstance(value, (list, dict)) and not value:
        return True
    return False


def assess(manifest):
    inputs = manifest.get("required_inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    blockers = [name for name in REQUIRED_FIELDS if name not in inputs or is_missing(inputs.get(name))]
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
    if manifest.get("declared_state") != ("Blocked" if blockers else "ReadyForIndependentReview"):
        structural_errors.append("declared_state does not match required-input completeness")
    # These values may legitimately be zero in some physical models; the validator
    # therefore treats only null/empty/unknown as missing, never silently edits inputs.
    expected_state = "Blocked" if blockers else "ReadyForIndependentReview"
    return {
        "schema": "luminous-dynamics.orbital-ring-manifest-validation/v1",
        "validation_state": "passed_contract_checks" if not structural_errors else "failed_contract_checks",
        "scenario_id": manifest.get("scenario_id"),
        "expected_gate_state": expected_state,
        "declared_gate_state": manifest.get("declared_state"),
        "analysis_state": manifest.get("analysis_state"),
        "required_input_count": len(REQUIRED_FIELDS),
        "missing_or_unknown_fields": blockers,
        "blocking_field_count": len(blockers),
        "structural_errors": structural_errors,
        "physics_claim": "none; no simulation or equilibrium calculation is performed",
    }


def self_test():
    checks = []

    def record(name, ok):
        checks.append({"id": name, "status": "passed" if ok else "failed"})

    template = {
        "schema": SCHEMA,
        "scenario_id": SCENARIO_ID,
        "architecture_mode": "continuous_skyhook_ors",
        "declared_state": "Blocked",
        "analysis_state": "NotRun",
        "result_artifacts": [],
        "required_fields": list(REQUIRED_FIELDS),
        "required_inputs": {name: None for name in REQUIRED_FIELDS},
    }
    result = assess(template)
    record("all_required_fields_known_to_contract", len(REQUIRED_FIELDS) == 23)
    record("empty_template_is_valid_but_blocked", result["validation_state"] == "passed_contract_checks" and result["expected_gate_state"] == "Blocked")
    record("empty_template_blocks_all_required_fields", result["blocking_field_count"] == len(REQUIRED_FIELDS))
    record("analysis_never_inferred_from_manifest_presence", result["analysis_state"] == "NotRun" and result["physics_claim"].startswith("none"))
    for field in REQUIRED_FIELDS:
        candidate = json.loads(json.dumps(template))
        candidate["required_inputs"][field] = 1.0
        candidate["declared_state"] = "Blocked"
        assessed = assess(candidate)
        record(f"single_field_{field}_cannot_unblock_others", assessed["expected_gate_state"] == "Blocked")
    candidate = json.loads(json.dumps(template))
    candidate["declared_state"] = "ReadyForIndependentReview"
    result = assess(candidate)
    record("premature_state_upgrade_rejected", result["validation_state"] == "failed_contract_checks" and result["expected_gate_state"] == "Blocked")
    candidate = json.loads(json.dumps(template))
    candidate["required_inputs"]["rider_force_law"] = "TBD"
    result = assess(candidate)
    record("tbd_force_law_still_blocks", result["expected_gate_state"] == "Blocked")
    complete = json.loads(json.dumps(template))
    complete["required_inputs"] = {name: "specified-fixture" for name in REQUIRED_FIELDS}
    complete["declared_state"] = "ReadyForIndependentReview"
    result = assess(complete)
    record("complete_manifest_is_review_ready_not_passed", result["expected_gate_state"] == "ReadyForIndependentReview" and result["validation_state"] == "passed_contract_checks" and result["analysis_state"] == "NotRun")
    report = {
        "schema": "luminous-dynamics.orbital-ring-manifest-self-test/v1",
        "checks_passed": sum(c["status"] == "passed" for c in checks),
        "checks_failed": sum(c["status"] != "passed" for c in checks),
        "checks": checks,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["checks_failed"] == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", help="manifest JSON path; defaults to the versioned repository template")
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
