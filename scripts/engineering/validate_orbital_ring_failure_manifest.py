#!/usr/bin/env python3
"""Fail-closed completeness/provenance gate; this script never runs physics."""
import argparse
import json
import math
from pathlib import Path

SCHEMA = "luminous-dynamics.orbital-ring-failure-scenario/v2"
SCENARIO_ID = "F-SKYHOOK-LOAD-STEP"
FIELDS = (
    "inertial_frame", "support_frame_transform", "ring_radius_m", "ring_plane_orientation",
    "initial_ring_position_m", "initial_ring_velocity_m_s", "initial_equilibrium_evidence_ref",
    "skyhook_support_locations_rad", "support_load_before_n", "support_load_after_n", "load_change_profile",
    "load_step_time_s", "rider_spacing_m", "rider_force_law", "rider_stiffness_n_per_m",
    "rider_damping_n_s_per_m", "rider_max_coupling_force_n", "actuation_measurement_delay_s",
    "force_model_reference", "energy_boundary", "momentum_boundary", "acceptance_limits", "independent_reference_case",
)
TOP_LEVEL_KEYS = {
    "schema", "scenario_id", "scenario_version", "architecture_mode", "initial_path_constraint",
    "title", "purpose", "source_basis", "declared_state", "analysis_state", "evidence_state",
    "result_artifacts", "required_fields", "required_inputs", "required_outputs_when_unblocked",
    "blocking_reason", "not_claimed", "input_provenance",
}
UNKNOWN = {"", "unknown", "tbd", "todo", "unspecified", "not specified", "?"}
PROVENANCE_CLASSES = {"primary_source", "derived", "engineering_assumption", "measurement", "independent_model", "test_fixture"}
PROVENANCE_STATES = {"not_reviewed", "independently_reviewed", "rejected"}


def reject_duplicate_keys(pairs):
    """Reject duplicate JSON object keys instead of silently keeping the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def reject_nonstandard_constant(value):
    """Reject NaN/Infinity tokens, which are not valid JSON numbers."""
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def parse_manifest_json(text):
    return json.loads(text, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonstandard_constant)


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def missing(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in UNKNOWN) or (isinstance(v, (dict, list)) and not v)


def vec(v, n=3, unit=False):
    ok = isinstance(v, list) and len(v) == n and all(finite(x) for x in v)
    return ok and (not unit or abs(math.sqrt(sum(x*x for x in v)) - 1.0) <= 1e-6)


def field_errors(k, v):
    e = []
    text_fields = {"inertial_frame", "initial_equilibrium_evidence_ref", "rider_force_law", "force_model_reference", "independent_reference_case"}
    positive = {"ring_radius_m", "rider_spacing_m", "rider_stiffness_n_per_m", "rider_max_coupling_force_n"}
    nonnegative = {"load_step_time_s", "rider_damping_n_s_per_m", "actuation_measurement_delay_s"}
    if k in text_fields and (not isinstance(v, str) or missing(v)):
        e.append(f"{k}: expected specified text/reference")
    elif k in positive and (not finite(v) or v <= 0):
        e.append(f"{k}: expected finite number > 0")
    elif k in nonnegative and (not finite(v) or v < 0):
        e.append(f"{k}: expected finite number >= 0")
    elif k in {"initial_ring_position_m", "initial_ring_velocity_m_s", "support_load_before_n", "support_load_after_n"} and not vec(v):
        e.append(f"{k}: expected finite 3-vector")
    elif k == "skyhook_support_locations_rad" and (not isinstance(v, list) or not v or not all(finite(x) for x in v)):
        e.append(f"{k}: expected non-empty list of finite angles")
    elif k == "ring_plane_orientation":
        if not isinstance(v, dict) or not vec(v.get("normal_unit_vector_in_inertial_frame"), unit=True):
            e.append(f"{k}: normal must be a unit 3-vector")
        if not isinstance(v, dict) or not isinstance(v.get("plane_reference"), str) or missing(v.get("plane_reference")):
            e.append(f"{k}: plane_reference missing")
    elif k == "support_frame_transform":
        rules = {
            "origin_position_m": lambda x: vec(x), "origin_velocity_m_s": lambda x: vec(x),
            "orientation_quaternion_xyzw": lambda x: vec(x, 4) and abs(math.sqrt(sum(y*y for y in x))-1.0) <= 1e-6,
            "angular_velocity_rad_s": lambda x: vec(x), "angular_acceleration_rad_s2": lambda x: vec(x),
        }
        if not isinstance(v, dict):
            e.append(f"{k}: expected object")
        else:
            e.extend(f"{k}.{name}: missing/invalid" for name, rule in rules.items() if name not in v or not rule(v[name]))
            if k == "support_frame_transform":
                for name in ("from_frame", "to_frame", "epoch", "quaternion_convention"):
                    if not isinstance(v.get(name), str) or missing(v.get(name)):
                        e.append(f"{k}.{name}: missing/invalid")
    elif k == "load_change_profile":
        if not isinstance(v, dict) or v.get("kind") not in {"step", "ramp"}:
            e.append(f"{k}: kind must be step/ramp")
        else:
            dur = v.get("transition_duration_s")
            if not finite(dur) or dur < 0 or (v["kind"] == "step" and dur != 0) or (v["kind"] == "ramp" and dur <= 0):
                e.append(f"{k}: transition duration inconsistent with kind")
            if not isinstance(v.get("profile_reference"), str) or missing(v.get("profile_reference")):
                e.append(f"{k}: profile_reference missing")
    elif k in {"energy_boundary", "momentum_boundary"}:
        rules = {
            "reference_frame": lambda x: isinstance(x, str) and not missing(x),
            "included_bodies": lambda x: isinstance(x, list) and bool(x) and all(isinstance(y, str) and not missing(y) for y in x),
            "scope_complete": lambda x: x is True,
            "term_inventory_ref": lambda x: isinstance(x, str) and not missing(x),
        }
        if k == "energy_boundary":
            rules.update({"external_work_terms": lambda x: isinstance(x, list), "boundary_flux_terms": lambda x: isinstance(x, list)})
        else:
            rules.update({"external_force_terms": lambda x: isinstance(x, list), "mass_momentum_flux_terms": lambda x: isinstance(x, list)})
        if not isinstance(v, dict):
            e.append(f"{k}: expected object")
        else:
            e.extend(f"{k}.{name}: missing/invalid" for name, rule in rules.items() if name not in v or not rule(v[name]))
    elif k == "acceptance_limits":
        rules = {"max_ring_offset_m": lambda x: finite(x) and x > 0, "max_support_miss_distance_m": lambda x: finite(x) and x > 0,
                 "max_rider_force_n": lambda x: finite(x) and x > 0, "max_energy_residual_j": lambda x: finite(x) and x >= 0,
                 "max_momentum_residual_kg_m_s": lambda x: finite(x) and x >= 0,
                 "max_initial_radial_velocity_m_s": lambda x: finite(x) and x >= 0}
        if not isinstance(v, dict):
            e.append(f"{k}: expected object")
        else:
            e.extend(f"{k}.{name}: missing/invalid" for name, rule in rules.items() if name not in v or not rule(v[name]))
    return e


def provenance_errors(k, p):
    if not isinstance(p, dict):
        return [f"{k}: provenance record missing"]
    e = []
    if p.get("classification") not in PROVENANCE_CLASSES:
        e.append(f"{k}: invalid provenance classification")
    elif p.get("classification") == "test_fixture":
        e.append(f"{k}: test_fixture provenance cannot be used for a scenario candidate")
    refs = p.get("evidence_refs")
    if not isinstance(refs, list) or not refs or not all(isinstance(r, str) and r.strip() and not missing(r) for r in refs):
        e.append(f"{k}: evidence_refs must contain at least one explicit reference")
    if not isinstance(p.get("rationale"), str) or missing(p.get("rationale")):
        e.append(f"{k}: provenance rationale missing")
    if p.get("review_state") not in PROVENANCE_STATES or p.get("review_state") == "rejected":
        e.append(f"{k}: review_state must be not_reviewed or independently_reviewed")
    return e


def assess(m):
    if not isinstance(m, dict):
        return {
            "schema": "luminous-dynamics.orbital-ring-manifest-validation/v1",
            "validation_state": "failed_contract_checks",
            "scenario_id": None, "scenario_schema": None, "scenario_version": None,
            "expected_gate_state": "Blocked", "declared_gate_state": None, "analysis_state": None,
            "required_input_count": len(FIELDS), "missing_or_unknown_fields": list(FIELDS),
            "type_errors": ["manifest must be a JSON object"], "blocking_fields": list(FIELDS),
            "blocking_field_count": len(FIELDS), "structural_errors": ["manifest must be a JSON object"],
            "physics_claim": "none; no simulation or equilibrium calculation is performed"
        }
    raw_inputs, raw_provenance = m.get("required_inputs"), m.get("input_provenance")
    inputs = raw_inputs if isinstance(raw_inputs, dict) else {}
    provenance = raw_provenance if isinstance(raw_provenance, dict) else {}
    blocked = [k for k in FIELDS if k not in inputs or missing(inputs.get(k))]
    errors = [err for k in FIELDS if k in inputs and not missing(inputs[k]) for err in field_errors(k, inputs[k])]
    for k in FIELDS:
        if k in inputs and not missing(inputs[k]):
            errors.extend(provenance_errors(k, provenance.get(k)))

    # Cross-field consistency guards reject malformed states; they are not a dynamics solver.
    pos, radius = inputs.get("initial_ring_position_m"), inputs.get("ring_radius_m")
    orientation = inputs.get("ring_plane_orientation")
    normal = orientation.get("normal_unit_vector_in_inertial_frame") if isinstance(orientation, dict) else None
    velocity = inputs.get("initial_ring_velocity_m_s")
    if finite(radius) and vec(pos) and abs(math.sqrt(sum(x*x for x in pos)) - radius) > max(1.0, radius * 1e-6):
        errors.append("initial_ring_position_m: norm must agree with ring_radius_m within 1e-6 relative or 1 m")
    if vec(pos) and vec(normal, unit=True) and abs(sum(pos[i]*normal[i] for i in range(3))) > max(1.0, math.sqrt(sum(x*x for x in pos))*1e-6):
        errors.append("initial_ring_position_m: initial position must lie in ring_plane_orientation")
    if vec(velocity) and vec(normal, unit=True) and abs(sum(velocity[i]*normal[i] for i in range(3))) > max(1e-3, math.sqrt(sum(x*x for x in velocity))*1e-6):
        errors.append("initial_ring_velocity_m_s: initial velocity must lie in ring plane")
    before, after = inputs.get("support_load_before_n"), inputs.get("support_load_after_n")
    if vec(before) and vec(after) and math.sqrt(sum(x*x for x in after)) >= math.sqrt(sum(x*x for x in before)):
        errors.append("support_load_after_n: this scenario requires a strict reduction in load magnitude")
    supports = inputs.get("skyhook_support_locations_rad")
    if isinstance(supports, list) and supports and all(finite(x) for x in supports):
        if len(supports) < 2 or any(x < 0 or x >= 2*math.pi for x in supports):
            errors.append("skyhook_support_locations_rad: require at least two angles in [0, 2π)")
        elif any(min(abs(supports[i]-supports[j]), 2*math.pi-abs(supports[i]-supports[j])) < 1e-9
                 for i in range(len(supports)) for j in range(i+1, len(supports))):
            errors.append("skyhook_support_locations_rad: support angles must be unique modulo 2π")
    inertial = inputs.get("inertial_frame")
    transform = inputs.get("support_frame_transform")
    if isinstance(transform, dict) and inertial is not None and transform.get("to_frame") != inertial:
        errors.append("support_frame_transform: to_frame must match inertial_frame")
    for k in ("energy_boundary", "momentum_boundary"):
        b = inputs.get(k)
        if isinstance(b, dict) and inertial is not None and b.get("reference_frame") != inertial:
            errors.append(f"{k}: reference_frame must match inertial_frame")
    limits, max_coupling = inputs.get("acceptance_limits"), inputs.get("rider_max_coupling_force_n")
    if isinstance(limits, dict) and finite(max_coupling) and finite(limits.get("max_rider_force_n")) and limits["max_rider_force_n"] > max_coupling:
        errors.append("acceptance_limits: max_rider_force_n cannot exceed rider_max_coupling_force_n")
    if m.get("initial_path_constraint") == "circular_reference_path" and vec(pos) and vec(velocity):
        r_norm = math.sqrt(sum(x*x for x in pos))
        radial_limit = limits.get("max_initial_radial_velocity_m_s") if isinstance(limits, dict) else None
        if r_norm <= 0 or not finite(radial_limit) or radial_limit < 0:
            errors.append("acceptance_limits.max_initial_radial_velocity_m_s: required finite nonnegative radial-speed tolerance")
        else:
            radial_speed = abs(sum(pos[i] * velocity[i] for i in range(3))) / r_norm
            if radial_speed > radial_limit:
                errors.append("initial_ring_velocity_m_s: radial component exceeds max_initial_radial_velocity_m_s")

    structural = []
    unknown_top_level = sorted(set(m) - TOP_LEVEL_KEYS)
    if unknown_top_level: structural.append(f"manifest contains unknown top-level keys: {unknown_top_level}")
    if not isinstance(raw_inputs, dict): structural.append("required_inputs must be an object")
    if not isinstance(raw_provenance, dict): structural.append("input_provenance must be an object")
    unknown_inputs = sorted(set(inputs) - set(FIELDS))
    unknown_provenance = sorted(set(provenance) - set(FIELDS))
    if unknown_inputs: structural.append(f"required_inputs contains unknown keys: {unknown_inputs}")
    if unknown_provenance: structural.append(f"input_provenance contains unknown keys: {unknown_provenance}")
    if m.get("schema") != SCHEMA: structural.append("schema mismatch")
    if m.get("scenario_id") != SCENARIO_ID: structural.append("scenario_id mismatch")
    if m.get("architecture_mode") != "continuous_skyhook_ors": structural.append("architecture mismatch")
    if m.get("scenario_version") != 2: structural.append("scenario_version must be 2 for scenario schema v2")
    if m.get("initial_path_constraint") != "circular_reference_path": structural.append("initial_path_constraint must be circular_reference_path")
    declared_fields = m.get("required_fields")
    if not isinstance(declared_fields, list) or declared_fields != list(FIELDS):
        structural.append("required_fields must be the exact ordered list for schema v2")
    if m.get("analysis_state") != "NotRun": structural.append("analysis_state must remain NotRun")
    if m.get("result_artifacts") != []: structural.append("result_artifacts must be empty")
    if not isinstance(m.get("input_provenance"), dict): structural.append("input_provenance must be an object")

    blocking = set(blocked)
    for err in errors:
        root = err.split(":", 1)[0].split(".", 1)[0]
        if root in FIELDS:
            blocking.add(root)
    # Structural errors are fatal to readiness too. Keep the reported expected gate
    # Blocked whenever either field-level blockers or schema/contract errors exist.
    expected = "Blocked" if blocking or structural else "ReadyForIndependentReview"
    if m.get("declared_state") != expected:
        structural.append("declared_state inconsistent with completeness, provenance, and consistency checks")
        expected = "Blocked"
    return {
        "schema": "luminous-dynamics.orbital-ring-manifest-validation/v1",
        "validation_state": "passed_contract_checks" if not structural and not errors else "failed_contract_checks",
        "scenario_id": m.get("scenario_id"), "scenario_schema": m.get("schema"),
        "scenario_version": m.get("scenario_version"), "expected_gate_state": expected,
        "declared_gate_state": m.get("declared_state"), "analysis_state": m.get("analysis_state"),
        "required_input_count": len(FIELDS), "missing_or_unknown_fields": blocked,
        "type_errors": errors, "blocking_fields": sorted(blocking), "blocking_field_count": len(blocking),
        "structural_errors": structural, "physics_claim": "none; no simulation or equilibrium calculation is performed"
    }


def complete_fixture():
    return {
        "inertial_frame":"GCRF",
        "support_frame_transform":{"from_frame":"support-fixed","to_frame":"GCRF","epoch":"fixture://epoch","quaternion_convention":"Hamilton active rotation, scalar-last xyzw","origin_position_m":[0,0,0],"origin_velocity_m_s":[0,0,0],"orientation_quaternion_xyzw":[0,0,0,1],"angular_velocity_rad_s":[0,0,0],"angular_acceleration_rad_s2":[0,0,0]},
        "ring_radius_m":7e6,"ring_plane_orientation":{"normal_unit_vector_in_inertial_frame":[0,0,1],"plane_reference":"declared inertial axes"},
        "initial_ring_position_m":[7e6,0,0],"initial_ring_velocity_m_s":[0,7500,0],"initial_equilibrium_evidence_ref":"fixture://equilibrium",
        "skyhook_support_locations_rad":[0,1,2],"support_load_before_n":[0,1e6,0],"support_load_after_n":[0,5e5,0],
        "load_change_profile":{"kind":"step","transition_duration_s":0,"profile_reference":"fixture://load-step"},"load_step_time_s":10,
        "rider_spacing_m":1e4,"rider_force_law":"fixture://rider-law","rider_stiffness_n_per_m":1e6,"rider_damping_n_s_per_m":1e4,
        "rider_max_coupling_force_n":1e8,"actuation_measurement_delay_s":0.1,"force_model_reference":"fixture://force-model",
        "energy_boundary":{"reference_frame":"GCRF","included_bodies":["ring","support","riders"],"external_work_terms":[],"boundary_flux_terms":[],"term_inventory_ref":"fixture://energy-term-inventory","scope_complete":True},
        "momentum_boundary":{"reference_frame":"GCRF","included_bodies":["ring","support","riders"],"external_force_terms":[],"mass_momentum_flux_terms":[],"term_inventory_ref":"fixture://momentum-term-inventory","scope_complete":True},
        "acceptance_limits":{"max_ring_offset_m":100,"max_support_miss_distance_m":1000,"max_rider_force_n":1e8,"max_energy_residual_j":1e6,"max_momentum_residual_kg_m_s":1e6,"max_initial_radial_velocity_m_s":0.01},
        "independent_reference_case":"fixture://reference"
    }


def fixture_provenance():
    return {k: {"classification":"engineering_assumption","evidence_refs":[f"fixture://{k}"],
                "rationale":"Synthetic validator state-machine test only; not an engineering value or accepted physical assumption.",
                "review_state":"not_reviewed"} for k in FIELDS}


def self_test():
    tests = []
    def check(name, condition): tests.append({"id": name, "status": "passed" if condition else "failed"})
    m = {"schema":SCHEMA,"scenario_id":SCENARIO_ID,"scenario_version":2,"architecture_mode":"continuous_skyhook_ors",
         "initial_path_constraint":"circular_reference_path",
         "declared_state":"Blocked","analysis_state":"NotRun","result_artifacts":[],
         "required_fields":list(FIELDS),"required_inputs":{k:None for k in FIELDS},
         "input_provenance":{k:None for k in FIELDS}}
    r = assess(m)
    check("23_required_fields", len(FIELDS)==23)
    check("empty_template_contract_valid_but_blocked", r["validation_state"]=="passed_contract_checks" and r["expected_gate_state"]=="Blocked")
    check("empty_template_blocks_all_fields", r["blocking_field_count"]==23)
    check("not_run_is_not_a_physics_claim", r["analysis_state"]=="NotRun" and r["physics_claim"].startswith("none"))
    for k in FIELDS:
        c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
        c["required_inputs"][k]=None; c["declared_state"]="Blocked"; r=assess(c)
        check(f"missing_{k}_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1 and not r["structural_errors"])
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("complete_fixture_review_ready_not_physics_passed", r["expected_gate_state"]=="ReadyForIndependentReview" and r["validation_state"]=="passed_contract_checks" and r["analysis_state"]=="NotRun")
    for k in FIELDS:
        c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
        c["input_provenance"][k]=None; c["declared_state"]="Blocked"; r=assess(c)
        check(f"missing_provenance_{k}_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["required_inputs"]["ring_radius_m"]="7000000"; c["declared_state"]="Blocked"; r=assess(c)
    check("wrong_numeric_type_blocks", r["expected_gate_state"]=="Blocked" and bool(r["type_errors"]) and r["blocking_field_count"]==1)
    check("wrong_numeric_type_fails_contract_validation", r["validation_state"]=="failed_contract_checks")
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["input_provenance"]["ring_radius_m"]["classification"]="test_fixture"; c["declared_state"]="Blocked"; r=assess(c)
    check("invalid_populated_provenance_fails_contract_validation", r["validation_state"]=="failed_contract_checks" and bool(r["type_errors"]))
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["telemetry"]="unexpected"; c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("unknown_top_level_key_fails_structural_gate", r["validation_state"]=="failed_contract_checks" and any("unknown top-level keys" in e for e in r["structural_errors"]) and r["expected_gate_state"]=="Blocked")
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["required_inputs"]["rider_force_law"]="TBD"; c["declared_state"]="Blocked"; r=assess(c)
    check("tbd_force_law_blocks", r["expected_gate_state"]=="Blocked")
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["required_inputs"]["ring_plane_orientation"]["normal_unit_vector_in_inertial_frame"]=[0,0,2]; c["declared_state"]="Blocked"; r=assess(c)
    check("non_unit_plane_normal_blocks", r["expected_gate_state"]=="Blocked" and bool(r["type_errors"]) and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["required_inputs"]["load_change_profile"]["transition_duration_s"]=1; c["declared_state"]="Blocked"; r=assess(c)
    check("step_with_nonzero_duration_blocks", r["expected_gate_state"]=="Blocked" and bool(r["type_errors"]) and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("premature_gate_upgrade_rejected", r["validation_state"]=="failed_contract_checks" and r["expected_gate_state"]=="Blocked")
    scenarios = [
        ("position_radius_mismatch_blocks", "initial_ring_position_m", [6e6,0,0]),
        ("load_increase_in_reduction_scenario_blocks", "support_load_after_n", [0,2e6,0]),
        ("duplicate_support_locations_block", "skyhook_support_locations_rad", [0,0,2]),
    ]
    for name, key, value in scenarios:
        c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
        c["required_inputs"][key]=value; c["declared_state"]="Blocked"; r=assess(c)
        check(name, r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["required_inputs"]["energy_boundary"]["reference_frame"]="ITRF"; c["declared_state"]="Blocked"; r=assess(c)
    check("energy_boundary_frame_mismatch_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["required_inputs"]["acceptance_limits"]["max_rider_force_n"]=2e8; c["declared_state"]="Blocked"; r=assess(c)
    check("acceptance_limit_above_coupler_capacity_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    del c["required_inputs"]["support_frame_transform"]["from_frame"]; c["declared_state"]="Blocked"; r=assess(c)
    check("missing_transform_direction_metadata_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    del c["required_inputs"]["energy_boundary"]["term_inventory_ref"]; c["declared_state"]="Blocked"; r=assess(c)
    check("missing_energy_term_inventory_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    del c["required_inputs"]["momentum_boundary"]["term_inventory_ref"]; c["declared_state"]="Blocked"; r=assess(c)
    check("missing_momentum_term_inventory_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["required_inputs"]["support_frame_transform"]["to_frame"]="ITRF"; c["declared_state"]="Blocked"; r=assess(c)
    check("transform_target_frame_mismatch_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["input_provenance"]["ring_radius_m"]["classification"]="test_fixture"; c["declared_state"]="Blocked"; r=assess(c)
    check("test_fixture_provenance_blocks_candidate", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["input_provenance"]["initial_equilibrium_evidence_ref"]["review_state"]="rejected"; c["declared_state"]="Blocked"; r=assess(c)
    check("rejected_evidence_provenance_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["input_provenance"]["force_model_reference"]["evidence_refs"]=[]; c["declared_state"]="Blocked"; r=assess(c)
    check("empty_provenance_references_block", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("circular_reference_tangent_state_review_ready", r["expected_gate_state"]=="ReadyForIndependentReview" and not any("radial component" in e for e in r["type_errors"]))
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    c["required_inputs"]["initial_ring_velocity_m_s"]=[0.02,7500,0]; c["declared_state"]="Blocked"; r=assess(c)
    check("circular_reference_radial_speed_over_limit_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1 and any("radial component" in e for e in r["type_errors"]))
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance()
    del c["required_inputs"]["acceptance_limits"]["max_initial_radial_velocity_m_s"]; c["declared_state"]="Blocked"; r=assess(c)
    check("circular_reference_missing_radial_tolerance_blocks", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["initial_path_constraint"]="unspecified"; c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("missing_path_constraint_fails_structural_gate", r["validation_state"]=="failed_contract_checks" and bool(r["structural_errors"]))
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["scenario_version"]=1; c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("scenario_version_mismatch_fails_structural_gate", r["validation_state"]=="failed_contract_checks" and r["expected_gate_state"]=="Blocked" and "scenario_version must be 2 for scenario schema v2" in r["structural_errors"])
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["declared_state"]="Blocked"; r=assess(c)
    check("declared_gate_mismatch_keeps_expected_gate_blocked", r["validation_state"]=="failed_contract_checks" and r["expected_gate_state"]=="Blocked" and any("declared_state inconsistent" in e for e in r["structural_errors"]))
    r=assess([])
    check("non_object_manifest_fails_closed_without_exception", r["validation_state"]=="failed_contract_checks" and "manifest must be a JSON object" in r["structural_errors"])
    c=json.loads(json.dumps(m)); c["required_inputs"]=[]; r=assess(c)
    check("non_object_required_inputs_fails_structural_gate", r["validation_state"]=="failed_contract_checks" and "required_inputs must be an object" in r["structural_errors"])
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["ring_radiuss_m"]=7e6; c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("unexpected_input_key_fails_structural_gate", r["validation_state"]=="failed_contract_checks" and r["expected_gate_state"]=="Blocked" and any("required_inputs contains unknown keys" in e for e in r["structural_errors"]))
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["input_provenance"]["ring_radiuss_m"]=fixture_provenance()["ring_radius_m"]; c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("unexpected_provenance_key_fails_structural_gate", r["validation_state"]=="failed_contract_checks" and any("input_provenance contains unknown keys" in e for e in r["structural_errors"]))
    c=json.loads(json.dumps(m)); c["required_fields"]=None; c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("non_list_required_fields_fails_closed_without_exception", r["validation_state"]=="failed_contract_checks" and any("required_fields must be the exact ordered list" in e for e in r["structural_errors"]))
    c=json.loads(json.dumps(m)); c["required_fields"]={"value":list(FIELDS)}; c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("object_required_fields_fails_closed_without_exception", r["validation_state"]=="failed_contract_checks" and any("required_fields must be the exact ordered list" in e for e in r["structural_errors"]))
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_fields"]=["ring_radius_m"]; c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("required_fields_subset_fails_closed_and_blocks_promotion", r["validation_state"]=="failed_contract_checks" and r["expected_gate_state"]=="Blocked" and any("required_fields must be the exact ordered list" in e for e in r["structural_errors"]))
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["input_provenance"]["rider_force_law_extra"]=fixture_provenance()["rider_force_law"]; c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("unexpected_provenance_key_keeps_expected_gate_blocked", r["validation_state"]=="failed_contract_checks" and r["expected_gate_state"]=="Blocked" and any("input_provenance contains unknown keys" in e for e in r["structural_errors"]))
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["skyhook_support_locations_rad"]=[1e-10,2*math.pi-1e-10,2.0]; c["declared_state"]="Blocked"; r=assess(c)
    check("support_angles_duplicate_across_zero_wrap_block", r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1 and any("unique modulo 2π" in e for e in r["type_errors"]))
    try:
        parse_manifest_json('{"schema":"first","schema":"second"}')
        duplicate_key_rejected = False
    except ValueError as exc:
        duplicate_key_rejected = "duplicate JSON object key: schema" in str(exc)
    check("duplicate_json_object_keys_rejected", duplicate_key_rejected)
    try:
        parse_manifest_json('{"required_inputs":{"ring_radius_m":NaN}}')
        nonstandard_number_rejected = False
    except ValueError as exc:
        nonstandard_number_rejected = "non-standard JSON numeric constant: NaN" in str(exc)
    check("nonstandard_nan_token_rejected", nonstandard_number_rejected)
    report={"schema":"luminous-dynamics.orbital-ring-manifest-self-test/v1",
            "checks_passed":sum(t["status"]=="passed" for t in tests),"checks_failed":sum(t["status"]!="passed" for t in tests),
            "checks":tests,"interpretation":"Contract and provenance checks only; review-ready does not mean physically validated or safe."}
    print(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
    return 0 if report["checks_failed"]==0 else 1


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("manifest",nargs="?"); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: return self_test()
    if not a.manifest: p.error("manifest path required unless --self-test is used")
    try: m=parse_manifest_json(Path(a.manifest).read_text(encoding="utf-8"))
    except (OSError,ValueError) as exc:
        print(json.dumps({"validation_state":"failed_to_read_manifest","error":str(exc)},sort_keys=True)); return 2
    r=assess(m); print(json.dumps(r,indent=2,sort_keys=True,allow_nan=False))
    return 0 if r["validation_state"]=="passed_contract_checks" else 1


if __name__ == "__main__": raise SystemExit(main())
