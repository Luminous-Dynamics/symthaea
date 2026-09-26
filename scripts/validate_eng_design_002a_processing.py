#!/usr/bin/env python3
import hashlib, json, pathlib
from collections import Counter

PATH = pathlib.Path("docs/release/evidence/eng-design-002a-processing-reference-v1.json")
EXPECTED_SHA256 = "d0699938f93d401566109f9dc8ab9973c6dc971dcd6fa0096c1b505b6d2e8ec2"
EXPECTED_SCHEMA = "eng-design-002a-processing-reference-v1"
EXPECTED_AUTHORITY = "repository_design_process_orchestration_only_no_physical_execution_authority"
EXPECTED_IDS = [f"C{i:02d}" for i in range(1, 27)]
FORBIDDEN = {
    "readiness_score", "maturity_score", "design_score",
    "poetry_score", "beauty_score", "meaning_score", "universal_rank"
}

def fail(msg):
    raise SystemExit(f"FAIL_ENG_DESIGN_002A {msg}")

def canonical_bytes(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()

def validate_shape(obj):
    if obj.get("schema") != EXPECTED_SCHEMA:
        fail("schema")
    if obj.get("authority") != EXPECTED_AUTHORITY:
        fail("authority")
    allowed = set(obj.get("allowed_fields", []))
    defaults = obj.get("defaults", {})
    if set(defaults) != allowed:
        fail("defaults_allowed_fields")
    if FORBIDDEN & set(defaults):
        fail("forbidden_default_key")
    cases = obj.get("cases", [])
    if [c.get("id") for c in cases] != EXPECTED_IDS:
        fail("case_ids")
    dispositions = set(obj.get("dispositions", []))
    if not dispositions:
        fail("dispositions")
    for c in cases:
        ov = c.get("overrides")
        if not isinstance(ov, dict):
            fail(f"{c.get('id')}_overrides")
        unknown = set(ov) - allowed
        if unknown:
            fail(f"{c.get('id')}_unknown_fields={sorted(unknown)}")
        if FORBIDDEN & set(ov):
            fail(f"{c.get('id')}_forbidden_score")
        if c.get("expected") not in dispositions:
            fail(f"{c.get('id')}_unknown_expected")
    return allowed

def derive(c):
    if c["physical_execution_authority_requested"]:
        return "AuthorityBoundaryBlocked"
    if not c["history_append_only"] or not c["rejected_alternative_retained"]:
        return "HistoryIntegrityBlocked"
    if not c["intent_defined"] or not c["intended_use_defined"]:
        return "NeedsClarification"
    if not c["requirements_present"]:
        return "IntentStructured"
    if not c["strict_requirements_have_verification_method"]:
        return "RequirementsBlocked"
    if not c["architecture_defined"]:
        return "RequirementsBound"
    if not c["mandatory_interfaces_resolved"]:
        return "InterfaceBlocked"
    if not c["assumptions_current"]:
        return "AssumptionBlocked"
    if not c["risk_review_complete"] or c["blocking_risk_open"]:
        return "RiskBlocked"
    if c["preference_conflict_with_hard_requirement"]:
        return "PreferenceConflictBlocked"
    if not c["analysis_plan_complete"]:
        return "InterfacesBound"
    if c["unsupported_analysis_requested"]:
        return "AnalysisUnsupported"
    if c["proof_obligations_open"]:
        return "ProofObligationOpen"
    if not c["evidence_collection_started"]:
        return "AnalysisPlanned"
    if c["design_or_asbuilt_change"] and not c["requalification_complete"]:
        return "RequalificationRequired"
    if not c["configuration_current"]:
        return "ConfigurationStale"
    if c["field_evidence_required"] and not c["field_evidence_present"]:
        return "PhysicalEvidenceBlocked"
    if not c["verification_complete"]:
        return "EvidenceInProgress"
    if not c["verification_pass"]:
        return "VerificationFailed"
    if c["external_authority_required"] and not c["external_authority_current"]:
        return "ExternalAuthorityRequired"
    if c["validation_required"]:
        if not c["validation_complete"]:
            return "DesignVerified"
        if not c["validation_pass"]:
            return "ValidationFailed"
        if not c["release_evidence_complete"]:
            return "UseValidated"
    elif not c["release_evidence_complete"]:
        return "DesignVerified"
    return "ReleaseEvidenceEligible"

def evaluate(obj):
    got = []
    for case in obj["cases"]:
        c = dict(obj["defaults"])
        c.update(case["overrides"])
        out = derive(c)
        if out != case["expected"]:
            fail(f"{case['id']}_expected={case['expected']}_derived={out}")
        got.append(out)
    return Counter(got)

def hostile_tests(obj):
    allowed = set(obj["allowed_fields"])
    defaults = obj["defaults"]

    c = dict(defaults)
    c["physical_execution_authority_requested"] = True
    if derive(c) != "AuthorityBoundaryBlocked":
        fail("authority_hostile")

    c = dict(defaults)
    c["formal_proof_receipt_present"] = True
    c["field_evidence_required"] = True
    c["field_evidence_present"] = False
    if derive(c) != "PhysicalEvidenceBlocked":
        fail("proof_field_hostile")

    c = dict(defaults)
    c["poetic_intent_declared"] = True
    c["preference_conflict_with_hard_requirement"] = True
    if derive(c) != "PreferenceConflictBlocked":
        fail("preference_hostile")

    if not ({"future_unreviewed_field"} - allowed):
        fail("unknown_field_not_rejected")
    if not ({"readiness_score"} & FORBIDDEN):
        fail("score_not_forbidden")

def main():
    raw = PATH.read_bytes()
    try:
        obj = json.loads(raw)
    except Exception as e:
        fail(f"json={e}")
    if canonical_bytes(obj) != raw:
        fail("noncanonical_json")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        fail(f"digest={digest}")
    validate_shape(obj)
    census = evaluate(obj)
    hostile_tests(obj)
    print(
        "PASS_ENG_DESIGN_002A_REFERENCE "
        f"digest={digest} cases={len(obj['cases'])} "
        f"outcomes={json.dumps(dict(sorted(census.items())), sort_keys=True)}"
    )

if __name__ == "__main__":
    main()
