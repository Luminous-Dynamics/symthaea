#!/usr/bin/env python3
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

CORPUS = Path("docs/release/evidence/built-bim-001a-openbim-projection-reference-v1.json")
EXPECTED_SHA256 = "fcb2a5b33c284b8cb6f591180807d56bbcb231d3beb75edd0b76dae5aff99c01"
EXPECTED_SCHEMA = "built-bim-001a-openbim-projection-reference-v1"
EXPECTED_AUTHORITY = "external_model_projection_only_no_engineering_or_physical_authority"
EXPECTED_BASE = {"branch": "main", "head": "eae17187e199e3a53d108b437c0215b5ff812261"}
EXPECTED_ISSUE = 6036
EXPECTED_PARENT = {"built_env_issue": 6032, "se_semantics_issue": 3692}
EXPECTED_SUPPORTED = {
    "BCF": ["BCFWorkflowProjectionV1"],
    "IDS": ["IDS1.0"],
    "IFC": ["IFC4.3.2.0"],
}
EXPECTED_OUTCOMES = [
    "ProjectionAdmissible",
    "ProjectionAdmissibleWithLoss",
    "ArtifactValidityBlocked",
    "ArtifactIdentityBlocked",
    "AdapterIdentityBlocked",
    "SchemaProfileBlocked",
    "ExternalIdentityBlocked",
    "IdentityMappingBlocked",
    "UnitFrameBlocked",
    "CurrentnessBlocked",
    "ProjectionLossBlocked",
    "InformationRequirementSatisfied",
    "InformationRequirementSatisfiedPhysicalConflict",
    "InformationRequirementBlocked",
    "WorkflowStateOnly",
    "WorkflowStatePhysicalResolutionReferenced",
    "HistoryIntegrityBlocked",
    "EvidenceAuthorityBlocked",
    "AuthorityBoundaryBlocked",
]
EXPECTED_CASE_IDS = [
    "C01_ifc43_supported_minimal_projection",
    "C02_invalid_external_artifact",
    "C03_missing_external_artifact_identity",
    "C04_missing_projection_adapter_identity",
    "C05_unsupported_ifc_version",
    "C06_ifc_view_profile_mismatch",
    "C07_unknown_property_retained_with_loss",
    "C08_unit_mismatch",
    "C09_coordinate_frame_ambiguous",
    "C10_duplicate_labels_distinct_external_ids",
    "C11_external_object_identity_not_preserved",
    "C12_unresolved_canonical_owner_mapping",
    "C13_ambiguous_canonical_owner_mapping",
    "C14_model_revision_stale_vs_as_built",
    "C15_ids_information_requirement_pass",
    "C16_ids_pass_physical_observation_conflicts",
    "C17_ids_missing_required_field",
    "C18_ids_semantics_unsupported_by_adapter",
    "C19_bcf_closed_without_retest",
    "C20_bcf_closed_with_physical_resolution_ref",
    "C21_external_history_deleted",
    "C22_hidden_projection_loss_claimed_lossless",
    "C23_catalog_property_retained_as_source_declared",
    "C24_catalog_property_promoted_to_measurement",
    "C25_projection_requests_physical_authority",
]
EXPECTED_NONCLAIMS = [
    "architectural_correctness",
    "physical_as_built_conformity",
    "structural_adequacy",
    "code_compliance",
    "permit_or_occupancy_approval",
    "commissioning_completion",
    "physical_defect_resolution",
    "external_certification",
    "procurement_or_resource_allocation",
    "physical_execution_authority",
]
CASE_KEYS = {
    "adapter_identity_bound",
    "artifact_identity_bound",
    "artifact_valid",
    "canonical_mapping",
    "currentness",
    "evidence_promotion_requested",
    "expected",
    "external_identity_preserved",
    "history_retained",
    "id",
    "ids_result",
    "kind",
    "loss_reported",
    "physical_authority_requested",
    "physical_conflict",
    "physical_resolution_ref",
    "source_profile",
    "units_frames",
    "unsupported_semantics",
    "value_authority",
    "workflow_closed",
}
EXPECTED_CENSUS = {
    "ProjectionAdmissible": 3,
    "ProjectionAdmissibleWithLoss": 1,
    "ArtifactValidityBlocked": 1,
    "ArtifactIdentityBlocked": 1,
    "AdapterIdentityBlocked": 1,
    "SchemaProfileBlocked": 3,
    "ExternalIdentityBlocked": 1,
    "IdentityMappingBlocked": 2,
    "UnitFrameBlocked": 2,
    "CurrentnessBlocked": 1,
    "ProjectionLossBlocked": 1,
    "InformationRequirementSatisfied": 1,
    "InformationRequirementSatisfiedPhysicalConflict": 1,
    "InformationRequirementBlocked": 1,
    "WorkflowStateOnly": 1,
    "WorkflowStatePhysicalResolutionReferenced": 1,
    "HistoryIntegrityBlocked": 1,
    "EvidenceAuthorityBlocked": 1,
    "AuthorityBoundaryBlocked": 1,
}

def fail(msg):
    raise SystemExit(f"FAIL_BUILT_BIM_001A_REFERENCE: {msg}")

def derive(c, supported):
    if c["physical_authority_requested"]:
        return "AuthorityBoundaryBlocked"
    if c["evidence_promotion_requested"]:
        return "EvidenceAuthorityBlocked"
    if not c["history_retained"]:
        return "HistoryIntegrityBlocked"
    if not c["artifact_valid"]:
        return "ArtifactValidityBlocked"
    if not c["artifact_identity_bound"]:
        return "ArtifactIdentityBlocked"
    if not c["adapter_identity_bound"]:
        return "AdapterIdentityBlocked"
    if c["source_profile"] not in supported.get(c["kind"], []):
        return "SchemaProfileBlocked"
    if not c["external_identity_preserved"]:
        return "ExternalIdentityBlocked"
    if c["canonical_mapping"] != "Resolved":
        return "IdentityMappingBlocked"
    if c["units_frames"] != "Resolved":
        return "UnitFrameBlocked"
    if c["currentness"] != "Current":
        return "CurrentnessBlocked"
    if c["unsupported_semantics"] and not c["loss_reported"]:
        return "ProjectionLossBlocked"

    if c["kind"] == "IFC":
        return "ProjectionAdmissibleWithLoss" if c["unsupported_semantics"] else "ProjectionAdmissible"
    if c["kind"] == "IDS":
        if c["ids_result"] == "Pass":
            return "InformationRequirementSatisfiedPhysicalConflict" if c["physical_conflict"] else "InformationRequirementSatisfied"
        return "InformationRequirementBlocked"
    if c["kind"] == "BCF":
        if c["workflow_closed"] and c["physical_resolution_ref"]:
            return "WorkflowStatePhysicalResolutionReferenced"
        return "WorkflowStateOnly"
    return "SchemaProfileBlocked"

def validate_case_shape(c):
    if set(c) != CASE_KEYS:
        fail(f"{c.get('id','<unknown>')}: case keys drift")
    if c["kind"] not in {"IFC", "IDS", "BCF"}:
        fail(f"{c['id']}: invalid kind")
    if c["canonical_mapping"] not in {"Resolved", "Unresolved", "Ambiguous"}:
        fail(f"{c['id']}: invalid canonical_mapping")
    if c["units_frames"] not in {"Resolved", "Mismatch", "Ambiguous"}:
        fail(f"{c['id']}: invalid units_frames")
    if c["currentness"] not in {"Current", "Stale"}:
        fail(f"{c['id']}: invalid currentness")
    if c["ids_result"] not in {"Pass", "Fail", "NotApplicable", "NotEvaluated"}:
        fail(f"{c['id']}: invalid ids_result")
    if c["value_authority"] not in {"NotApplicable", "SourceDeclared", "PhysicalObservation"}:
        fail(f"{c['id']}: invalid value_authority")
    bool_fields = [
        "adapter_identity_bound","artifact_identity_bound","artifact_valid",
        "evidence_promotion_requested","external_identity_preserved","history_retained",
        "loss_reported","physical_authority_requested","physical_conflict",
        "physical_resolution_ref","unsupported_semantics","workflow_closed",
    ]
    if any(type(c[k]) is not bool for k in bool_fields):
        fail(f"{c['id']}: non-bool boolean field")
    if c["expected"] not in EXPECTED_OUTCOMES:
        fail(f"{c['id']}: unknown expected outcome")

def hostile_self_tests(supported):
    template = {
        "kind":"IFC","source_profile":"IFC4.3.2.0",
        "artifact_valid":True,"artifact_identity_bound":True,"adapter_identity_bound":True,
        "external_identity_preserved":True,"canonical_mapping":"Resolved",
        "units_frames":"Resolved","currentness":"Current","unsupported_semantics":False,
        "loss_reported":True,"ids_result":"NotApplicable","workflow_closed":False,
        "physical_resolution_ref":False,"physical_conflict":False,
        "value_authority":"NotApplicable","evidence_promotion_requested":False,
        "physical_authority_requested":False,"history_retained":True,
    }
    controls = [
        ("baseline", {}, "ProjectionAdmissible"),
        ("artifact invalid", {"artifact_valid":False}, "ArtifactValidityBlocked"),
        ("artifact identity absent", {"artifact_identity_bound":False}, "ArtifactIdentityBlocked"),
        ("adapter identity absent", {"adapter_identity_bound":False}, "AdapterIdentityBlocked"),
        ("unsupported profile", {"source_profile":"IFC2x3"}, "SchemaProfileBlocked"),
        ("external identity collapsed", {"external_identity_preserved":False}, "ExternalIdentityBlocked"),
        ("mapping ambiguous", {"canonical_mapping":"Ambiguous"}, "IdentityMappingBlocked"),
        ("unit mismatch", {"units_frames":"Mismatch"}, "UnitFrameBlocked"),
        ("stale", {"currentness":"Stale"}, "CurrentnessBlocked"),
        ("hidden loss", {"unsupported_semantics":True,"loss_reported":False}, "ProjectionLossBlocked"),
        ("evidence promotion", {"evidence_promotion_requested":True,"value_authority":"SourceDeclared"}, "EvidenceAuthorityBlocked"),
        ("physical authority", {"physical_authority_requested":True}, "AuthorityBoundaryBlocked"),
    ]
    for name, patch, expected in controls:
        c = dict(template)
        c.update(patch)
        got = derive(c, supported)
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
    if data.get("supported_profiles") != EXPECTED_SUPPORTED:
        fail("supported profile registry drift")
    if data.get("outcomes") != EXPECTED_OUTCOMES:
        fail("outcome vocabulary drift")
    if data.get("nonclaims") != EXPECTED_NONCLAIMS:
        fail("nonclaim vocabulary drift")

    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 25:
        fail("expected exactly 25 cases")
    ids = [c.get("id") for c in cases]
    if ids != EXPECTED_CASE_IDS:
        fail("case identity/order drift")
    if len(ids) != len(set(ids)):
        fail("duplicate case ids")

    derived = []
    for c in cases:
        validate_case_shape(c)
        got = derive(c, data["supported_profiles"])
        derived.append(got)
        if got != c["expected"]:
            fail(f"{c['id']}: derived {got}, expected {c['expected']}")

    if dict(Counter(derived)) != EXPECTED_CENSUS:
        fail("derived disposition census drift")

    by_id = {c["id"]: c for c in cases}
    anchors = {
        "C11_external_object_identity_not_preserved":"ExternalIdentityBlocked",
        "C16_ids_pass_physical_observation_conflicts":"InformationRequirementSatisfiedPhysicalConflict",
        "C20_bcf_closed_with_physical_resolution_ref":"WorkflowStatePhysicalResolutionReferenced",
        "C23_catalog_property_retained_as_source_declared":"ProjectionAdmissible",
        "C24_catalog_property_promoted_to_measurement":"EvidenceAuthorityBlocked",
        "C25_projection_requests_physical_authority":"AuthorityBoundaryBlocked",
    }
    for cid, expected in anchors.items():
        if derive(by_id[cid], data["supported_profiles"]) != expected:
            fail(f"semantic anchor failed: {cid}")

    if by_id["C23_catalog_property_retained_as_source_declared"]["value_authority"] != "SourceDeclared":
        fail("catalog/source-declared anchor lost")
    if by_id["C24_catalog_property_promoted_to_measurement"]["value_authority"] != "SourceDeclared":
        fail("evidence-promotion anchor lost source authority")
    if not by_id["C16_ids_pass_physical_observation_conflicts"]["physical_conflict"]:
        fail("IDS physical-conflict anchor lost")
    if not by_id["C20_bcf_closed_with_physical_resolution_ref"]["physical_resolution_ref"]:
        fail("BCF physical-resolution reference anchor lost")

    hostile_self_tests(data["supported_profiles"])

    print(
        "PASS_BUILT_BIM_001A_REFERENCE "
        f"digest={EXPECTED_SHA256} cases={len(cases)} "
        f"outcomes={json.dumps(dict(sorted(Counter(derived).items())), sort_keys=True)}"
    )

if __name__ == "__main__":
    main()
