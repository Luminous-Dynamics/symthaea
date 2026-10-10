#!/usr/bin/env python3
"""Validate required inputs for an orbital-ring fault scenario; never run physics."""
import argparse
import json
import math
from pathlib import Path

SCHEMA = "luminous-dynamics.orbital-ring-failure-scenario/v1"
SCENARIO_ID = "F-SKYHOOK-LOAD-STEP"
FIELDS = (
    "inertial_frame", "support_frame_transform", "ring_radius_m", "ring_plane_orientation",
    "initial_ring_position_m", "initial_ring_velocity_m_s", "initial_equilibrium_evidence_ref",
    "skyhook_support_locations_rad", "support_load_before_n", "support_load_after_n", "load_change_profile",
    "load_step_time_s", "rider_spacing_m", "rider_force_law", "rider_stiffness_n_per_m",
    "rider_damping_n_s_per_m", "rider_max_coupling_force_n", "actuation_measurement_delay_s",
    "force_model_reference", "energy_boundary", "momentum_boundary", "acceptance_limits", "independent_reference_case",
)
UNKNOWN = {"", "unknown", "tbd", "todo", "unspecified", "not specified", "?"}
PROVENANCE_CLASSES = {"primary_source", "derived", "engineering_assumption", "measurement", "independent_model", "test_fixture"}
PROVENANCE_STATES = {"not_reviewed", "independently_reviewed", "rejected"}


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def missing(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in UNKNOWN) or (isinstance(v, (dict, list)) and not v)


def vec(v, n=3, unit=False):
    ok = isinstance(v, list) and len(v) == n and all(finite(x) for x in v)
    return ok and (not unit or abs(math.sqrt(sum(x*x for x in v)) - 1.0) <= 1e-6)


def type_errors(name, v):
    errors = []
    text_fields = {"inertial_frame", "initial_equilibrium_evidence_ref", "rider_force_law", "force_model_reference", "independent_reference_case"}
    positive = {"ring_radius_m", "rider_spacing_m", "rider_stiffness_n_per_m", "rider_max_coupling_force_n"}
    nonnegative = {"load_step_time_s", "rider_damping_n_s_per_m", "actuation_measurement_delay_s"}
    vector3 = {"initial_ring_position_m", "initial_ring_velocity_m_s", "support_load_before_n", "support_load_after_n"}
    if name in text_fields and (not isinstance(v, str) or missing(v)):
        errors.append(f"{name}: expected specified text/reference")
    elif name in positive and (not finite(v) or v <= 0):
        errors.append(f"{name}: expected finite number > 0")
    elif name in nonnegative and (not finite(v) or v < 0):
        errors.append(f"{name}: expected finite number >= 0")
    elif name in vector3 and not vec(v):
        errors.append(f"{name}: expected finite 3-vector")
    elif name == "skyhook_support_locations_rad" and (not isinstance(v, list) or not v or not all(finite(x) for x in v)):
        errors.append(f"{name}: expected non-empty list of finite angles")
    elif name == "ring_plane_orientation":
        if not isinstance(v, dict) or not vec(v.get("normal_unit_vector_in_inertial_frame"), unit=True):
            errors.append("ring_plane_orientation: normal must be a unit 3-vector")
        if not isinstance(v, dict) or not isinstance(v.get("plane_reference"), str) or missing(v.get("plane_reference")):
            errors.append("ring_plane_orientation: plane_reference missing")
    elif name == "support_frame_transform":
        rules = {
            "origin_position_m": lambda x: vec(x), "origin_velocity_m_s": lambda x: vec(x),
            "orientation_quaternion_xyzw": lambda x: vec(x, 4) and abs(math.sqrt(sum(y*y for y in x))-1.0) <= 1e-6,
            "angular_velocity_rad_s": lambda x: vec(x), "angular_acceleration_rad_s2": lambda x: vec(x),
        }
        if not isinstance(v, dict): errors.append("support_frame_transform: expected object")
        else:
            errors.extend(f"support_frame_transform.{k}: missing/invalid" for k, rule in rules.items() if k not in v or not rule(v[k]))
    elif name == "load_change_profile":
        if not isinstance(v, dict) or v.get("kind") not in {"step", "ramp"}:
            errors.append("load_change_profile: kind must be step/ramp")
        else:
            dur = v.get("transition_duration_s")
            if not finite(dur) or dur < 0 or (v["kind"] == "step" and dur != 0) or (v["kind"] == "ramp" and dur <= 0):
                errors.append("load_change_profile: transition duration inconsistent with kind")
            if not isinstance(v.get("profile_reference"), str) or missing(v.get("profile_reference")):
                errors.append("load_change_profile: profile_reference missing")
    elif name in {"energy_boundary", "momentum_boundary"}:
        rules = {
            "reference_frame": lambda x: isinstance(x, str) and not missing(x),
            "included_bodies": lambda x: isinstance(x, list) and bool(x) and all(isinstance(y, str) and not missing(y) for y in x),
            "scope_complete": lambda x: x is True,
        }
        if name == "energy_boundary": rules.update({"external_work_terms": lambda x: isinstance(x, list), "boundary_flux_terms": lambda x: isinstance(x, list)})
        else: rules.update({"external_force_terms": lambda x: isinstance(x, list), "mass_momentum_flux_terms": lambda x: isinstance(x, list)})
        if not isinstance(v, dict): errors.append(f"{name}: expected object")
        else: errors.extend(f"{name}.{k}: missing/invalid" for k, rule in rules.items() if k not in v or not rule(v[k]))
    elif name == "acceptance_limits":
        rules = {"max_ring_offset_m": lambda x: finite(x) and x > 0, "max_support_miss_distance_m": lambda x: finite(x) and x > 0,
                 "max_rider_force_n": lambda x: finite(x) and x > 0, "max_energy_residual_j": lambda x: finite(x) and x >= 0,
                 "max_momentum_residual_kg_m_s": lambda x: finite(x) and x >= 0}
        if not isinstance(v, dict): errors.append("acceptance_limits: expected object")
        else: errors.extend(f"acceptance_limits.{k}: missing/invalid" for k, rule in rules.items() if k not in v or not rule(v[k]))
    return errors


def provenance_errors(name, record):
    errors = []
    if not isinstance(record, dict):
        return [f"{name}: provenance record missing"]
    if record.get("classification") not in PROVENANCE_CLASSES:
        errors.append(f"{name}: invalid provenance classification")
    refs = record.get("evidence_refs")
    if not isinstance(refs, list) or not refs or not all(isinstance(r, str) and r.strip() and not missing(r) for r in refs):
        errors.append(f"{name}: evidence_refs must contain at least one explicit reference")
    if not isinstance(record.get("rationale"), str) or missing(record.get("rationale")):
        errors.append(f"{name}: provenance rationale missing")
    if record.get("review_state") not in PROVENANCE_STATES or record.get("review_state") == "rejected":
        errors.append(f"{name}: review_state must be not_reviewed or independently_reviewed")
    return errors


def assess(m):
    inputs = m.get("required_inputs") if isinstance(m.get("required_inputs"), dict) else {}
    provenance = m.get("input_provenance") if isinstance(m.get("input_provenance"), dict) else {}
    blocked = [k for k in FIELDS if k not in inputs or missing(inputs.get(k))]
    errors = [e for k in FIELDS if k in inputs and not missing(inputs[k]) for e in type_errors(k, inputs[k])]
    for field in FIELDS:
        if field in inputs and not missing(inputs[field]):
            errors.extend(provenance_errors(field, provenance.get(field)))
    # Cross-field consistency guards are contract checks, not dynamics.
    pos, radius = inputs.get("initial_ring_position_m"), inputs.get("ring_radius_m")
    orientation = inputs.get("ring_plane_orientation")
    normal = orientation.get("normal_unit_vector_in_inertial_frame") if isinstance(orientation, dict) else None
    velocity = inputs.get("initial_ring_velocity_m_s")
    if finite(radius) and vec(pos) and abs(math.sqrt(sum(x*x for x in pos)) - radius) > max(1.0, radius * 1e-6):
        errors.append("initial_ring_position_m: norm must agree with ring_radius_m within 1e-6 relative or 1 m")
    if vec(pos) and vec(normal, unit=True) and abs(sum(pos[i] * normal[i] for i in range(3))) > max(1.0, math.sqrt(sum(x*x for x in pos)) * 1e-6):
        errors.append("initial_ring_position_m: initial position must lie in ring_plane_orientation")
    if vec(velocity) and vec(normal, unit=True) and abs(sum(velocity[i] * normal[i] for i in range(3))) > max(1e-3, math.sqrt(sum(x*x for x in velocity)) * 1e-6):
        errors.append("initial_ring_velocity_m_s: initial velocity must lie in ring plane")
    before, after = inputs.get("support_load_before_n"), inputs.get("support_load_after_n")
    if vec(before) and vec(after) and math.sqrt(sum(x*x for x in after)) >= math.sqrt(sum(x*x for x in before)):
        errors.append("support_load_after_n: this scenario requires a strict reduction in load magnitude")
    supports = inputs.get("skyhook_support_locations_rad")
    if isinstance(supports, list) and supports and all(finite(x) for x in supports):
        tau = 2.0 * math.pi
        if len(supports) < 2 or any(x < 0 or x >= tau for x in supports):
            errors.append("skyhook_support_locations_rad: require at least two angles in [0, 2π)")
        elif any(abs(supports[i] - supports[j]) < 1e-9 for i in range(len(supports)) for j in range(i+1, len(supports))):
            errors.append("skyhook_support_locations_rad: support angles must be unique")
    inertial = inputs.get("inertial_frame")
    for field in ("energy_boundary", "momentum_boundary"):
        boundary = inputs.get(field)
        if isinstance(boundary, dict) and inertial is not None and boundary.get("reference_frame") != inertial:
            errors.append(f"{field}: reference_frame must match inertial_frame")
    limits, max_coupling = inputs.get("acceptance_limits"), inputs.get("rider_max_coupling_force_n")
    if isinstance(limits, dict) and finite(max_coupling) and finite(limits.get("max_rider_force_n")) and limits["max_rider_force_n"] > max_coupling:
        errors.append("acceptance_limits: max_rider_force_n cannot exceed rider_max_coupling_force_n")

    structural = []
    if m.get("schema") != SCHEMA: structural.append("schema mismatch")
    if m.get("scenario_id") != SCENARIO_ID: structural.append("scenario_id mismatch")
    if m.get("architecture_mode") != "continuous_skyhook_ors": structural.append("architecture mismatch")
    if tuple(m.get("required_fields", ())) != FIELDS: structural.append("required_fields contract mismatch")
    if m.get("analysis_state") != "NotRun": structural.append("analysis_state must remain NotRun")
    if not isinstance(m.get("input_provenance"), dict): structural.append("input_provenance must be an object")
    if m.get("result_artifacts") != []: structural.append("result_artifacts must be empty")
    blocking_names = set(blocked)
    for error in errors:
        root = error.split(":", 1)[0].split(".", 1)[0]
        if root in FIELDS:
            blocking_names.add(root)
    expected = "Blocked" if blocking_names else "ReadyForIndependentReview"
    if m.get("declared_state") != expected: structural.append("declared_state inconsistent with completeness/type checks")
    return {"schema":"luminous-dynamics.orbital-ring-manifest-validation/v1",
            "validation_state":"passed_contract_checks" if not structural else "failed_contract_checks",
            "scenario_id":m.get("scenario_id"), "expected_gate_state":expected, "declared_gate_state":m.get("declared_state"),
            "analysis_state":m.get("analysis_state"), "required_input_count":len(FIELDS),
            "missing_or_unknown_fields":blocked, "type_errors":errors, "blocking_fields":sorted(blocking_names),
            "blocking_field_count":len(blocking_names),
            "structural_errors":structural, "physics_claim":"none; no simulation or equilibrium calculation is performed"}


def complete_fixture():
    return {"inertial_frame":"GCRF",
      "support_frame_transform":{"origin_position_m":[0,0,0],"origin_velocity_m_s":[0,0,0],"orientation_quaternion_xyzw":[0,0,0,1],"angular_velocity_rad_s":[0,0,0],"angular_acceleration_rad_s2":[0,0,0]},
      "ring_radius_m":7e6,"ring_plane_orientation":{"normal_unit_vector_in_inertial_frame":[0,0,1],"plane_reference":"declared inertial axes"},
      "initial_ring_position_m":[7e6,0,0],"initial_ring_velocity_m_s":[0,7500,0],"initial_equilibrium_evidence_ref":"fixture://equilibrium",
      "skyhook_support_locations_rad":[0,1,2],"support_load_before_n":[0,1e6,0],"support_load_after_n":[0,5e5,0],
      "load_change_profile":{"kind":"step","transition_duration_s":0,"profile_reference":"fixture://load-step"},"load_step_time_s":10,
      "rider_spacing_m":1e4,"rider_force_law":"fixture://rider-law","rider_stiffness_n_per_m":1e6,"rider_damping_n_s_per_m":1e4,
      "rider_max_coupling_force_n":1e8,"actuation_measurement_delay_s":0.1,"force_model_reference":"fixture://force-model",
      "energy_boundary":{"reference_frame":"GCRF","included_bodies":["ring","support","riders"],"external_work_terms":[],"boundary_flux_terms":[],"scope_complete":True},
      "momentum_boundary":{"reference_frame":"GCRF","included_bodies":["ring","support","riders"],"external_force_terms":[],"mass_momentum_flux_terms":[],"scope_complete":True},
      "acceptance_limits":{"max_ring_offset_m":100,"max_support_miss_distance_m":1000,"max_rider_force_n":1e8,"max_energy_residual_j":1e6,"max_momentum_residual_kg_m_s":1e6},
      "independent_reference_case":"fixture://reference"}


def fixture_provenance():
    return {name: {"classification":"test_fixture", "evidence_refs":[f"fixture://{name}"],
                   "rationale":"Deterministic validator-only fixture; not an engineering value.",
                   "review_state":"not_reviewed"} for name in FIELDS}


def self_test():
    tests=[]
    def check(name, passed): tests.append({"id":name,"status":"passed" if passed else "failed"})
    m={"schema":SCHEMA,"scenario_id":SCENARIO_ID,"architecture_mode":"continuous_skyhook_ors","declared_state":"Blocked","analysis_state":"NotRun","result_artifacts":[],"required_fields":list(FIELDS),"required_inputs":{k:None for k in FIELDS},"input_provenance":{k:None for k in FIELDS}}
    a=assess(m)
    check("23_required_fields",len(FIELDS)==23)
    check("template_contract_valid_but_blocked",a["validation_state"]=="passed_contract_checks" and a["expected_gate_state"]=="Blocked")
    check("template_missing_all_required_fields",a["blocking_field_count"]==23)
    check("not_run_is_not_a_physics_claim",a["analysis_state"]=="NotRun" and a["physics_claim"].startswith("none"))
    for field in FIELDS:
        c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"][field]=None; c["declared_state"]="Blocked"
        r=assess(c); check(f"missing_{field}_blocks",r["expected_gate_state"]=="Blocked" and not r["structural_errors"])
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["declared_state"]="ReadyForIndependentReview"
    r=assess(c); check("complete_fixture_review_ready_not_passed",r["expected_gate_state"]=="ReadyForIndependentReview" and r["analysis_state"]=="NotRun")
    for field in FIELDS:
        c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["input_provenance"][field]=None; c["declared_state"]="Blocked"
        r=assess(c); check(f"missing_provenance_{field}_blocks",r["expected_gate_state"]=="Blocked" and r["blocking_field_count"]==1)
    c["required_inputs"]["ring_radius_m"]="7000000"; c["declared_state"]="Blocked"; r=assess(c)
    check("wrong_numeric_type_blocks",r["expected_gate_state"]=="Blocked" and bool(r["type_errors"]) and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["rider_force_law"]="TBD"; c["declared_state"]="Blocked"; r=assess(c)
    check("tbd_force_law_blocks",r["expected_gate_state"]=="Blocked")
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["ring_plane_orientation"]["normal_unit_vector_in_inertial_frame"]=[0,0,2]; c["declared_state"]="Blocked"; r=assess(c)
    check("non_unit_plane_normal_blocks",r["expected_gate_state"]=="Blocked" and bool(r["type_errors"]) and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["load_change_profile"]["transition_duration_s"]=1; c["declared_state"]="Blocked"; r=assess(c)
    check("step_with_nonzero_duration_blocks",r["expected_gate_state"]=="Blocked" and bool(r["type_errors"]) and r["blocking_field_count"]==1)
    c=json.loads(json.dumps(m)); c["declared_state"]="ReadyForIndependentReview"; r=assess(c)
    check("premature_gate_upgrade_rejected",r["validation_state"]=="failed_contract_checks" and r["expected_gate_state"]=="Blocked")
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["initial_ring_position_m"]=[6e6,0,0]; c["declared_state"]="Blocked"; r=assess(c)
    check("position_radius_mismatch_blocks",r["expected_gate_state"]=="Blocked")
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["support_load_after_n"]=[0,2e6,0]; c["declared_state"]="Blocked"; r=assess(c)
    check("load_increase_in_reduction_scenario_blocks",r["expected_gate_state"]=="Blocked")
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["skyhook_support_locations_rad"]=[0,0,2]; c["declared_state"]="Blocked"; r=assess(c)
    check("duplicate_support_locations_block",r["expected_gate_state"]=="Blocked")
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["energy_boundary"]["reference_frame"]="ITRF"; c["declared_state"]="Blocked"; r=assess(c)
    check("energy_boundary_frame_mismatch_blocks",r["expected_gate_state"]=="Blocked")
    c=json.loads(json.dumps(m)); c["required_inputs"]=complete_fixture(); c["input_provenance"]=fixture_provenance(); c["required_inputs"]["acceptance_limits"]["max_rider_force_n"]=2e8; c["declared_state"]="Blocked"; r=assess(c)
    check("acceptance_limit_above_coupler_capacity_blocks",r["expected_gate_state"]=="Blocked")
    report={"schema":"luminous-dynamics.orbital-ring-manifest-self-test/v1","checks_passed":sum(t["status"]=="passed" for t in tests),"checks_failed":sum(t["status"]!="passed" for t in tests),"checks":tests,"interpretation":"Contract validation only; review-ready does not mean physically validated or safe."}
    print(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)); return 0 if report["checks_failed"]==0 else 1


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("manifest",nargs="?"); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: return self_test()
    if not a.manifest: p.error("manifest path required unless --self-test is used")
    try: m=json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc: print(json.dumps({"validation_state":"failed_to_read_manifest","error":str(exc)},sort_keys=True)); return 2
    r=assess(m); print(json.dumps(r,indent=2,sort_keys=True,allow_nan=False)); return 0 if r["validation_state"]=="passed_contract_checks" else 1

if __name__ == "__main__": raise SystemExit(main())
