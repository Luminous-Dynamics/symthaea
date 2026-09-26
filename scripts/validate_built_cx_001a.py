#!/usr/bin/env python3
import hashlib
import json
from collections import Counter
from pathlib import Path

CORPUS = Path("docs/release/evidence/built-cx-001a-commissioning-reference-v1.json")
EXPECTED_SHA256 = "f6ab3a0e66be5072ed479dc554728ce15a9471f1ead2ca10c221fdcedb441d71"
EXPECTED_SCHEMA = "built-cx-001a-commissioning-reference-v1"
EXPECTED_AUTHORITY = "commissioning_evidence_composition_only_no_safety_process_or_execution_authority"
EXPECTED_BASE = {"branch":"main","head":"eae17187e199e3a53d108b437c0215b5ff812261"}
EXPECTED_ISSUE = 6037
EXPECTED_PARENT = {"built_env_issue":6032,"se_vv_issue":3697}
EXPECTED_SCOPE_LEVELS = ["Component","Subsystem","System","IntegratedFacility","ProductionLine"]
EXPECTED_EVIDENCE_KINDS = ["Physical","Virtual"]
EXPECTED_OUTCOMES = [
    "InstallationBlocked","ConfigurationBlocked","CalibrationBlocked","PrerequisiteBlocked",
    "PhysicalResponseBlocked","FunctionalTestPending","FunctionalTestFailed",
    "CoverageBlocked","CommonModeDependencyBlocked","ComponentCommissioned",
    "IntegrationPending","IntegrationFailed","IntegratedCommissioned",
    "VirtualEvidencePending","VirtualEvidenceFailed","VirtualEvidenceOnly",
    "RecommissioningRequired","IssueClosureUnsubstantiated",
    "HistoryIntegrityBlocked","ProcessQualificationBoundaryBlocked","AuthorityBoundaryBlocked",
]
EXPECTED_NONCLAIMS = [
    "code_compliance","permit_or_occupancy_approval","fire_or_life_safety_approval",
    "equipment_certification","production_process_capability","product_conformity",
    "facility_safety","procurement_or_resource_allocation","physical_execution_authority",
]
EXPECTED_CASE_IDS = [
    "C01_delivered_not_installed",
    "C02_installed_wrong_configuration",
    "C03_installed_calibration_absent",
    "C04_powered_no_functional_test",
    "C05_command_accepted_no_physical_response",
    "C06_test_result_bound_to_stale_configuration",
    "C07_component_functionally_commissioned",
    "C08_component_passes_subsystem_integration_fails",
    "C09_two_subsystems_pass_timing_interaction_fails",
    "C10_redundancy_has_shared_upstream_common_mode",
    "C11_virtual_commissioning_pass_only",
    "C12_physical_pass_one_declared_mode_untested",
    "C13_issue_closed_without_retest",
    "C14_equipment_replaced_after_commissioning",
    "C15_control_software_revision_requires_retest",
    "C16_building_services_commissioned_process_separate",
    "C17_machines_pass_material_flow_integration_unresolved",
    "C18_robot_commissioned_cell_application_unresolved",
    "C19_utility_nominal_capacity_quality_unobserved",
    "C20_work_order_complete_without_functional_acceptance",
    "C21_ifc_ids_complete_without_commissioning",
    "C22_repair_changes_generation_old_commissioning_not_current",
    "C23_negative_history_deleted_after_repair",
    "C24_full_synthetic_building_route",
    "C25_full_synthetic_factory_line_commissioned_only",
    "C26_commissioning_promoted_to_process_qualification",
    "C27_commissioning_requests_physical_authority",
    "C28_functional_test_fails",
    "C29_system_functional_pass_integration_not_executed",
    "C30_virtual_commissioning_fails",
    "C31_virtual_commissioning_not_executed",
    "C32_issue_closed_with_retest_reference_but_retest_not_executed",
]
CASE_KEYS = {
    "calibration_state","command_accepted","common_mode_resolved","configuration_current",
    "configuration_match","delivered","evidence_kind","expected","functional_result",
    "history_retained","id","installed","integration_result","issue_closed",
    "lower_level_evidence_current","mode_coverage_complete","model_information_complete",
    "physical_authority_requested","physical_response_observed","prerequisites_resolved",
    "process_qualification_requested","retest_evidence","scope_level","virtual_result",
    "workflow_complete",
}
EXPECTED_CENSUS = {
    "InstallationBlocked":1,
    "ConfigurationBlocked":1,
    "CalibrationBlocked":1,
    "PrerequisiteBlocked":2,
    "PhysicalResponseBlocked":1,
    "FunctionalTestPending":4,
    "FunctionalTestFailed":1,
    "CoverageBlocked":1,
    "CommonModeDependencyBlocked":1,
    "ComponentCommissioned":1,
    "IntegrationPending":1,
    "IntegrationFailed":3,
    "IntegratedCommissioned":3,
    "VirtualEvidencePending":1,
    "VirtualEvidenceFailed":1,
    "VirtualEvidenceOnly":1,
    "RecommissioningRequired":4,
    "IssueClosureUnsubstantiated":1,
    "HistoryIntegrityBlocked":1,
    "ProcessQualificationBoundaryBlocked":1,
    "AuthorityBoundaryBlocked":1,
}

def fail(msg):
    raise SystemExit(f"FAIL_BUILT_CX_001A_REFERENCE: {msg}")

def derive(c):
    if c["physical_authority_requested"]:
        return "AuthorityBoundaryBlocked"
    if c["process_qualification_requested"]:
        return "ProcessQualificationBoundaryBlocked"
    if not c["history_retained"]:
        return "HistoryIntegrityBlocked"
    if not c["configuration_current"]:
        return "RecommissioningRequired"
    if c["issue_closed"] and not c["retest_evidence"]:
        return "IssueClosureUnsubstantiated"
    if c["evidence_kind"] == "Virtual":
        if c["virtual_result"] == "Pass":
            return "VirtualEvidenceOnly"
        if c["virtual_result"] == "Fail":
            return "VirtualEvidenceFailed"
        return "VirtualEvidencePending"
    if not c["delivered"] or not c["installed"]:
        return "InstallationBlocked"
    if not c["configuration_match"]:
        return "ConfigurationBlocked"
    if c["calibration_state"] not in {"Current","NotRequired"}:
        return "CalibrationBlocked"
    if not c["prerequisites_resolved"] or not c["lower_level_evidence_current"]:
        return "PrerequisiteBlocked"
    if c["command_accepted"] and not c["physical_response_observed"]:
        return "PhysicalResponseBlocked"
    if c["functional_result"] == "NotExecuted":
        return "FunctionalTestPending"
    if c["functional_result"] == "Fail":
        return "FunctionalTestFailed"
    if not c["mode_coverage_complete"]:
        return "CoverageBlocked"
    if not c["common_mode_resolved"]:
        return "CommonModeDependencyBlocked"
    if c["scope_level"] == "Component":
        return "ComponentCommissioned"
    if c["integration_result"] == "NotExecuted":
        return "IntegrationPending"
    if c["integration_result"] == "Fail":
        return "IntegrationFailed"
    if c["integration_result"] == "Pass":
        return "IntegratedCommissioned"
    return "IntegrationPending"

def validate_case_shape(c):
    if set(c) != CASE_KEYS:
        fail(f"{c.get('id','<unknown>')}: case keys drift")
    if c["evidence_kind"] not in EXPECTED_EVIDENCE_KINDS:
        fail(f"{c['id']}: invalid evidence_kind")
    if c["scope_level"] not in EXPECTED_SCOPE_LEVELS:
        fail(f"{c['id']}: invalid scope_level")
    if c["calibration_state"] not in {"Current","Missing","Stale","NotRequired"}:
        fail(f"{c['id']}: invalid calibration_state")
    if c["functional_result"] not in {"Pass","Fail","NotExecuted"}:
        fail(f"{c['id']}: invalid functional_result")
    if c["integration_result"] not in {"Pass","Fail","NotExecuted","NotApplicable"}:
        fail(f"{c['id']}: invalid integration_result")
    if c["virtual_result"] not in {"Pass","Fail","NotExecuted","NotApplicable"}:
        fail(f"{c['id']}: invalid virtual_result")
    bool_fields = [
        "command_accepted","common_mode_resolved","configuration_current","configuration_match",
        "delivered","history_retained","installed","issue_closed","lower_level_evidence_current",
        "mode_coverage_complete","model_information_complete","physical_authority_requested",
        "physical_response_observed","prerequisites_resolved","process_qualification_requested",
        "retest_evidence","workflow_complete",
    ]
    if any(type(c[k]) is not bool for k in bool_fields):
        fail(f"{c['id']}: non-bool boolean field")
    if c["expected"] not in EXPECTED_OUTCOMES:
        fail(f"{c['id']}: unknown outcome")

def hostile_self_tests():
    template = {
        "evidence_kind":"Physical","scope_level":"Component","delivered":True,"installed":True,
        "configuration_match":True,"calibration_state":"Current","prerequisites_resolved":True,
        "lower_level_evidence_current":True,"command_accepted":False,
        "physical_response_observed":True,"functional_result":"Pass",
        "integration_result":"NotApplicable","mode_coverage_complete":True,
        "common_mode_resolved":True,"configuration_current":True,"issue_closed":False,
        "retest_evidence":False,"history_retained":True,"virtual_result":"NotApplicable",
        "workflow_complete":False,"model_information_complete":False,
        "process_qualification_requested":False,"physical_authority_requested":False,
    }
    controls = [
        ("baseline", {}, "ComponentCommissioned"),
        ("authority", {"physical_authority_requested":True}, "AuthorityBoundaryBlocked"),
        ("process promotion", {"process_qualification_requested":True}, "ProcessQualificationBoundaryBlocked"),
        ("history loss", {"history_retained":False}, "HistoryIntegrityBlocked"),
        ("configuration drift", {"configuration_current":False}, "RecommissioningRequired"),
        ("issue closed no retest", {"issue_closed":True}, "IssueClosureUnsubstantiated"),
        ("virtual pending", {"evidence_kind":"Virtual","virtual_result":"NotExecuted"}, "VirtualEvidencePending"),
        ("virtual fail", {"evidence_kind":"Virtual","virtual_result":"Fail"}, "VirtualEvidenceFailed"),
        ("virtual pass", {"evidence_kind":"Virtual","virtual_result":"Pass"}, "VirtualEvidenceOnly"),
        ("installation", {"installed":False}, "InstallationBlocked"),
        ("configuration", {"configuration_match":False}, "ConfigurationBlocked"),
        ("calibration", {"calibration_state":"Stale"}, "CalibrationBlocked"),
        ("prerequisite", {"prerequisites_resolved":False}, "PrerequisiteBlocked"),
        ("response", {"command_accepted":True,"physical_response_observed":False,"functional_result":"NotExecuted"}, "PhysicalResponseBlocked"),
        ("functional pending", {"physical_response_observed":False,"functional_result":"NotExecuted"}, "FunctionalTestPending"),
        ("functional fail", {"functional_result":"Fail"}, "FunctionalTestFailed"),
        ("coverage", {"mode_coverage_complete":False}, "CoverageBlocked"),
        ("common mode", {"common_mode_resolved":False}, "CommonModeDependencyBlocked"),
        ("integration pending", {"scope_level":"System","integration_result":"NotExecuted"}, "IntegrationPending"),
        ("integration fail", {"scope_level":"System","integration_result":"Fail"}, "IntegrationFailed"),
        ("integrated", {"scope_level":"System","integration_result":"Pass"}, "IntegratedCommissioned"),
    ]
    for name, patch, expected in controls:
        c = dict(template)
        c.update(patch)
        got = derive(c)
        if got != expected:
            fail(f"hostile control {name}: {got} != {expected}")

def main():
    raw = CORPUS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
        fail("corpus sha256 drift")
    try:
        data = json.loads(raw)
    except Exception as e:
        fail(f"invalid json: {e}")
    canonical = (json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if raw != canonical:
        fail("corpus is not canonical compact sorted-key JSON + newline")
    if data.get("schema") != EXPECTED_SCHEMA:
        fail("schema drift")
    if data.get("authority") != EXPECTED_AUTHORITY:
        fail("authority drift")
    if data.get("base") != EXPECTED_BASE:
        fail("base drift")
    if data.get("issue") != EXPECTED_ISSUE:
        fail("issue drift")
    if data.get("parent") != EXPECTED_PARENT:
        fail("parent drift")
    if data.get("scope_levels") != EXPECTED_SCOPE_LEVELS:
        fail("scope level vocabulary drift")
    if data.get("evidence_kinds") != EXPECTED_EVIDENCE_KINDS:
        fail("evidence kind vocabulary drift")
    if data.get("outcomes") != EXPECTED_OUTCOMES:
        fail("outcome vocabulary drift")
    if data.get("nonclaims") != EXPECTED_NONCLAIMS:
        fail("nonclaim vocabulary drift")

    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 32:
        fail("expected exactly 32 cases")
    ids = [c.get("id") for c in cases]
    if ids != EXPECTED_CASE_IDS:
        fail("case identity/order drift")
    if len(ids) != len(set(ids)):
        fail("duplicate case ids")

    derived = []
    for c in cases:
        validate_case_shape(c)
        got = derive(c)
        derived.append(got)
        if got != c["expected"]:
            fail(f"{c['id']}: derived {got}, expected {c['expected']}")

    census = dict(Counter(derived))
    if census != EXPECTED_CENSUS:
        fail("derived disposition census drift")
    if set(census) != set(EXPECTED_OUTCOMES):
        fail("not every frozen disposition is exercised")

    by_id = {c["id"]: c for c in cases}
    anchors = {
        "C05_command_accepted_no_physical_response":"PhysicalResponseBlocked",
        "C11_virtual_commissioning_pass_only":"VirtualEvidenceOnly",
        "C20_work_order_complete_without_functional_acceptance":"FunctionalTestPending",
        "C21_ifc_ids_complete_without_commissioning":"FunctionalTestPending",
        "C25_full_synthetic_factory_line_commissioned_only":"IntegratedCommissioned",
        "C26_commissioning_promoted_to_process_qualification":"ProcessQualificationBoundaryBlocked",
        "C32_issue_closed_with_retest_reference_but_retest_not_executed":"FunctionalTestPending",
    }
    for cid, expected in anchors.items():
        if derive(by_id[cid]) != expected:
            fail(f"semantic anchor failed: {cid}")

    if not by_id["C20_work_order_complete_without_functional_acceptance"]["workflow_complete"]:
        fail("workflow-complete anchor lost")
    if not by_id["C21_ifc_ids_complete_without_commissioning"]["model_information_complete"]:
        fail("model-information anchor lost")
    if not by_id["C32_issue_closed_with_retest_reference_but_retest_not_executed"]["retest_evidence"]:
        fail("retest-reference anchor lost")

    hostile_self_tests()

    print(
        "PASS_BUILT_CX_001A_REFERENCE "
        f"digest={EXPECTED_SHA256} cases={len(cases)} "
        f"outcomes={json.dumps(dict(sorted(census.items())), sort_keys=True)}"
    )

if __name__ == "__main__":
    main()
