#!/usr/bin/env python3
"""Independent stdlib-only qualifier for ROB-TOOL-001A."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "docs/release/evidence/rob-tool-001a-specialization-reference-v1.json"
EXPECTED_SHA256 = "e3244d59dd8d7ce385f21c7d79da06cd093fd6b79d9fd1ad5db770a84b319d2e"
EXPECTED_SCHEMA = "rob-tool-001a-specialization-reference-v1"
EXPECTED_AUTHORITY = "robot_tool_specialization_semantics_only_no_task_execution_or_safety_authority"
EXPECTED_BASE = {"branch": "main", "head": "eae17187e199e3a53d108b437c0215b5ff812261"}
EXPECTED_PARENT = {"rob_design_issue": 4795, "rob_spec_issue": 6033, "universal_embodiment_issue": 1274}
EXPECTED_ISSUE = 6038
EXPECTED_PROFILES = ["GenericGripper", "InspectionMetrologyHead", "PassiveMachineTendingAdapter"]
EXPECTED_QUAL_RESULTS = ["Pass", "Fail", "Blocked", "EnvironmentFailure"]
EXPECTED_EXTERNAL_ALIGNMENT = [
    {"role": "automatic_end_effector_exchange_vocabulary_reference", "standard": "ISO 11593", "version": "2022"},
    {"role": "industrial_robot_application_integration_context_reference", "standard": "ISO 10218-2", "version": "2025"},
]
EXPECTED_OUTCOMES = [
    "ToolSpecializationProfileAdmissible", "IdentityBlocked", "AttachmentBlocked", "InterfaceBlocked",
    "CalibrationBlocked", "FrameBlocked", "HostEnvelopeBlocked", "CollisionModelBlocked", "WorkspaceBlocked",
    "ProcessQualificationBoundaryBlocked", "ConsumableBlocked", "HealthCurrentnessBlocked",
    "ConfigurationCurrentnessBlocked", "TaskEvidenceBlocked", "AuthorityBoundaryBlocked",
    "ExchangeControllerFailed", "ExchangeEvidenceIncomplete", "ExchangeConfigurationBlocked",
    "ExchangeCalibrationBlocked", "ExchangeSemanticallyComplete",
]
EXPECTED_NONCLAIMS = [
    "actual_task_capability", "industrial_robot_safety_compliance", "collaborative_operation_approval",
    "construction_quality", "structural_fastening", "welding_quality", "manufacturing_process_qualification",
    "tool_certification", "procurement_or_resource_allocation", "physical_execution_authority",
]
EXPECTED_CASE_IDS = [
    "C01_complete_gripper_specialization",
    "C02_wrong_physical_tool_instance",
    "C03_attached_identity_unverified",
    "C04_attachment_not_physically_observed",
    "C05_mechanical_mount_mismatch",
    "C06_power_contract_mismatch",
    "C07_data_transport_matches_but_action_semantics_mismatch",
    "C08_calibration_expired",
    "C09_tcp_frame_missing",
    "C10_tool_mass_inertia_exceeds_host_envelope",
    "C11_collision_geometry_missing",
    "C12_task_target_unreachable",
    "C13_observed_trials_below_profile_requirement",
    "C14_scanner_field_task_with_stale_frame",
    "C15_drill_tool_missing_process_receipt",
    "C16_welding_tool_process_receipt_scope_mismatch",
    "C17_consumable_unavailable",
    "C18_tool_wear_requires_review",
    "C19_tool_repair_old_configuration_reused",
    "C20_same_tool_profile_other_host_exceeds_payload",
    "C21_planner_path_only_not_task_evidence",
    "C22_model_evidence_cannot_satisfy_field_requirement",
    "C23_task_evidence_scope_mismatch",
    "C24_execution_claim_requested_from_tool_profile",
    "C25_physical_authority_requested_from_tool_profile",
    "C26_exchange_controller_reports_failure",
    "C27_exchange_success_code_but_attachment_sensor_disagrees",
    "C28_exchange_new_tool_identity_unverified",
    "C29_exchange_mechanical_connection_unresolved",
    "C30_exchange_power_connection_unresolved",
    "C31_exchange_old_tool_detached_but_still_current",
    "C32_exchange_config_generation_not_updated",
    "C33_exchange_complete_but_calibration_stale",
    "C34_exchange_complete_but_tcp_frame_stale",
    "C35_exchange_complete_exact_semantic_thread",
    "C36_inspection_head_complete_field_specialization",
    "C37_passive_machine_tending_adapter_complete",
    "C38_process_tool_with_current_scoped_pass_receipt",
    "C39_process_receipt_fail_blocks_specialization",
    "C40_process_receipt_stale_blocks_specialization",
]
EXPECTED_TOP_KEYS = {
    "authority", "base", "cases", "exchange_defaults", "external_alignment", "field_vocabularies",
    "initial_profiles", "issue", "nonclaims", "outcomes", "parent", "qualification_result_vocabulary",
    "schema", "tool_defaults",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL_ROB_TOOL_001A_REFERENCE: {message}")


def canonical_bytes(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def derive_tool(s: dict) -> str:
    if s["physical_authority_requested"] or s["execution_claim_requested"]:
        return "AuthorityBoundaryBlocked"
    if s["physical_instance_identity"] != "Verified":
        return "IdentityBlocked"
    if s["attachment_state"] != "Attached":
        return "AttachmentBlocked"
    for key in ("mechanical_interface", "power_interface", "fluid_interface", "data_interface", "semantic_action_contract"):
        if s[key] not in ("Resolved", "NotApplicable"):
            return "InterfaceBlocked"
    if s["calibration_state"] not in ("Current", "NotRequired"):
        return "CalibrationBlocked"
    if s["tcp_frame_state"] not in ("Current", "NotApplicable"):
        return "FrameBlocked"
    if s["mass_inertia_state"] != "WithinHostEnvelope":
        return "HostEnvelopeBlocked"
    if s["collision_geometry_state"] != "Current":
        return "CollisionModelBlocked"
    if s["workspace_state"] != "Reachable":
        return "WorkspaceBlocked"
    if s["process_qualification_required"]:
        if not s["process_receipt_bound"]:
            return "ProcessQualificationBoundaryBlocked"
        if s["process_receipt_result"] != "Pass":
            return "ProcessQualificationBoundaryBlocked"
        if not s["process_receipt_current"]:
            return "ProcessQualificationBoundaryBlocked"
        if not s["process_receipt_scope_match"]:
            return "ProcessQualificationBoundaryBlocked"
    if s["consumable_state"] == "Unavailable":
        return "ConsumableBlocked"
    if s["wear_health_state"] != "Current":
        return "HealthCurrentnessBlocked"
    if not s["configuration_current"]:
        return "ConfigurationCurrentnessBlocked"
    if s["planner_or_controller_only"]:
        return "TaskEvidenceBlocked"
    if s["task_evidence_required_plane"] == "FIELD" and s["task_evidence_actual_plane"] != "FIELD":
        return "TaskEvidenceBlocked"
    if not s["task_evidence_current"] or not s["task_evidence_scope_match"]:
        return "TaskEvidenceBlocked"
    if s["observed_independent_task_trials"] < s["required_independent_task_trials"]:
        return "TaskEvidenceBlocked"
    return "ToolSpecializationProfileAdmissible"


def derive_exchange(s: dict) -> str:
    if s["physical_authority_requested"] or s["execution_claim_requested"]:
        return "AuthorityBoundaryBlocked"
    if s["exchange_controller_result"] != "Success":
        return "ExchangeControllerFailed"
    if not s["old_tool_detached_observed"]:
        return "ExchangeEvidenceIncomplete"
    if s["new_tool_attachment_state"] != "Attached":
        return "ExchangeEvidenceIncomplete"
    if s["new_tool_identity"] != "Verified":
        return "ExchangeEvidenceIncomplete"
    for key in ("mechanical_connection_state", "power_connection_state", "fluid_connection_state", "data_connection_state", "semantic_action_contract"):
        if s[key] not in ("Resolved", "NotApplicable"):
            return "ExchangeEvidenceIncomplete"
    if s["mass_inertia_state"] != "WithinHostEnvelope" or s["collision_geometry_state"] != "Current":
        return "ExchangeEvidenceIncomplete"
    if s["old_tool_still_current"]:
        return "ExchangeConfigurationBlocked"
    if not s["configuration_generation_updated"]:
        return "ExchangeConfigurationBlocked"
    if s["calibration_state"] != "Current" or s["tcp_frame_state"] != "Current":
        return "ExchangeCalibrationBlocked"
    return "ExchangeSemanticallyComplete"


def derive(kind: str, state: dict) -> str:
    if kind == "TOOL":
        return derive_tool(state)
    if kind == "EXCHANGE":
        return derive_exchange(state)
    fail(f"unknown case kind {kind!r}")


def validate_state(kind: str, state: dict, defaults: dict, vocab: dict) -> None:
    if set(state) != set(defaults):
        fail(f"expanded {kind} state keys differ from frozen defaults")
    for key, default in defaults.items():
        value = state[key]
        if key in vocab:
            if value not in vocab[key]:
                fail(f"{kind}.{key} has unknown enum value {value!r}")
            continue
        if isinstance(default, bool):
            if type(value) is not bool:
                fail(f"{kind}.{key} must be boolean")
        elif isinstance(default, int):
            if type(value) is not int or value < 0:
                fail(f"{kind}.{key} must be non-negative integer")
        else:
            fail(f"{kind}.{key} is neither frozen enum, bool nor int")


def hostile_self_tests(tool_defaults: dict, exchange_defaults: dict) -> None:
    def t(**changes):
        s = copy.deepcopy(tool_defaults); s.update(changes); return derive_tool(s)
    def x(**changes):
        s = copy.deepcopy(exchange_defaults); s.update(changes); return derive_exchange(s)

    checks = [
        (t(physical_authority_requested=True), "AuthorityBoundaryBlocked"),
        (t(physical_instance_identity="Unverified"), "IdentityBlocked"),
        (t(mechanical_interface="Mismatch"), "InterfaceBlocked"),
        (t(calibration_state="Stale"), "CalibrationBlocked"),
        (t(tcp_frame_state="Missing"), "FrameBlocked"),
        (t(mass_inertia_state="OutsideHostEnvelope"), "HostEnvelopeBlocked"),
        (t(collision_geometry_state="Missing"), "CollisionModelBlocked"),
        (t(workspace_state="Unreachable"), "WorkspaceBlocked"),
        (t(process_qualification_required=True), "ProcessQualificationBoundaryBlocked"),
        (t(process_qualification_required=True, process_receipt_bound=True, process_receipt_result="Fail", process_receipt_current=True, process_receipt_scope_match=True), "ProcessQualificationBoundaryBlocked"),
        (t(process_qualification_required=True, process_receipt_bound=True, process_receipt_result="Pass", process_receipt_current=True, process_receipt_scope_match=True), "ToolSpecializationProfileAdmissible"),
        (t(task_evidence_required_plane="FIELD"), "TaskEvidenceBlocked"),
        (t(observed_independent_task_trials=1), "TaskEvidenceBlocked"),
        (x(exchange_controller_result="Fail"), "ExchangeControllerFailed"),
        (x(new_tool_attachment_state="Detached"), "ExchangeEvidenceIncomplete"),
        (x(mechanical_connection_state="Unknown"), "ExchangeEvidenceIncomplete"),
        (x(old_tool_still_current=True), "ExchangeConfigurationBlocked"),
        (x(calibration_state="Stale"), "ExchangeCalibrationBlocked"),
        (x(), "ExchangeSemanticallyComplete"),
    ]
    for actual, expected in checks:
        if actual != expected:
            fail(f"hostile self-test derived {actual}, expected {expected}")


def main() -> None:
    raw = CORPUS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        fail(f"digest drift: {digest}")
    data = json.loads(raw)
    if raw != canonical_bytes(data):
        fail("corpus is not canonical compact sorted-key JSON + newline")
    if set(data) != EXPECTED_TOP_KEYS:
        fail("top-level key set drift")
    if data["schema"] != EXPECTED_SCHEMA or data["authority"] != EXPECTED_AUTHORITY:
        fail("schema/authority drift")
    if data["base"] != EXPECTED_BASE or data["parent"] != EXPECTED_PARENT or data["issue"] != EXPECTED_ISSUE:
        fail("base/parent/issue drift")
    if data["initial_profiles"] != EXPECTED_PROFILES:
        fail("initial profile vocabulary drift")
    if data["qualification_result_vocabulary"] != EXPECTED_QUAL_RESULTS:
        fail("qualification-result vocabulary drift")
    if data["external_alignment"] != EXPECTED_EXTERNAL_ALIGNMENT:
        fail("external alignment drift")
    if data["outcomes"] != EXPECTED_OUTCOMES or data["nonclaims"] != EXPECTED_NONCLAIMS:
        fail("outcome/nonclaim vocabulary drift")

    cases = data["cases"]
    if [c.get("id") for c in cases] != EXPECTED_CASE_IDS:
        fail("case identity/order drift")
    if len(cases) != 40:
        fail("case count drift")

    defaults_by_kind = {"TOOL": data["tool_defaults"], "EXCHANGE": data["exchange_defaults"]}
    vocab_by_kind = data["field_vocabularies"]
    if set(vocab_by_kind) != {"TOOL", "EXCHANGE"}:
        fail("field-vocabulary kind drift")

    # Frozen vocabularies may only describe actual fields, and every default enum must itself be legal.
    for kind in ("TOOL", "EXCHANGE"):
        defaults = defaults_by_kind[kind]
        vocab = vocab_by_kind[kind]
        if not set(vocab).issubset(defaults):
            fail(f"{kind} vocabulary names a non-default field")
        validate_state(kind, defaults, defaults, vocab)

    derived = []
    for case in cases:
        if set(case) != {"id", "kind", "expected", "overrides"}:
            fail(f"{case.get('id', '<unknown>')} case shape drift")
        kind = case["kind"]
        if kind not in defaults_by_kind:
            fail(f"{case['id']} unknown kind")
        if case["expected"] not in EXPECTED_OUTCOMES:
            fail(f"{case['id']} unknown expected disposition")
        overrides = case["overrides"]
        if type(overrides) is not dict:
            fail(f"{case['id']} overrides must be an object")
        unknown = set(overrides) - set(defaults_by_kind[kind])
        if unknown:
            fail(f"{case['id']} unknown override fields: {sorted(unknown)}")
        state = copy.deepcopy(defaults_by_kind[kind])
        state.update(overrides)
        validate_state(kind, state, defaults_by_kind[kind], vocab_by_kind[kind])
        result = derive(kind, state)
        if result != case["expected"]:
            fail(f"{case['id']} derived {result}, expected {case['expected']}")
        derived.append(result)

    census = Counter(derived)
    missing_outcomes = set(EXPECTED_OUTCOMES) - set(census)
    if missing_outcomes:
        fail(f"frozen disposition(s) unexercised: {sorted(missing_outcomes)}")

    hostile_self_tests(data["tool_defaults"], data["exchange_defaults"])
    print(
        "PASS_ROB_TOOL_001A_REFERENCE "
        f"digest={digest} cases={len(cases)} outcomes="
        + json.dumps(dict(sorted(census.items())), sort_keys=True)
    )


if __name__ == "__main__":
    main()
