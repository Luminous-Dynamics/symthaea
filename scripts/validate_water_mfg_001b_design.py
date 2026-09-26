#!/usr/bin/env python3
"""Independent reference validator for WATER-MFG-001B ENG-DESIGN consumer packet.

This validator is intentionally stdlib-only and imports no Symthaea production code.
It establishes only faithful representation of the frozen synthetic source contract.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKET_PATH = ROOT / "docs/release/evidence/water-mfg-001b-eng-design-packet-v1.json"
DOC_PATH = ROOT / "docs/engineering/WATER_MFG_001B_ENG_DESIGN_PACKET.md"

EXPECTED_PACKET_SHA256 = "c9a611b11549cfa534757c43b59897356ac660d4de94084fe61358679faedc51"
EXPECTED_DOC_SHA256 = "c15a92b3119b84fad87c99103292553dcde5e21413c4457db6010b6c6428548f"
EXPECTED_SCHEMA = "water-mfg-001b-eng-design-packet-v1"
EXPECTED_AUTHORITY = "analysis_and_design_only_no_physical_execution_authority"
EXPECTED_SOURCE_BASE = "main@eae17187e199e3a53d108b437c0215b5ff812261"

EXPECTED_BINDINGS = {
    "eng_design": {
        "source_pr": 6006,
        "source_head": "0cc3a61871192e011299290b64514b4b341e5f1f",
        "qualifier_pr": 6009,
        "qualifier_head": "1273c3779313d321840c97ffdb2567406b0fe600",
        "qualification_requirement": "HostedExactHeadPassReceiptRequired",
        "qualification_receipt_ref": None,
    },
    "water_mfg": {
        "source_pr": 6001,
        "source_head": "3c8a439a4a837a535613b9bb09710c860dd28ff3",
        "qualifier_pr": 6003,
        "qualifier_head": "62d0d4eccdd7cff113070d8c68383f99bc12f8c5",
        "qualification_requirement": "HostedExactHeadPassReceiptRequired",
        "qualification_receipt_ref": None,
        "source_corpus_sha256": "75c74323afb879f0af4eb5917d74e620a01905beba7af50bbbb6301c757438c1",
    },
    "ind_comp": {
        "source_pr": 5990,
        "source_head": "b2346d0383567063893e37ce219ec15f631a9cb7",
        "qualifier_pr": 5995,
        "qualifier_head": "652b6f048ab8e49209df7eedfd2e4eed6d7f5562",
        "qualification_requirement": "HostedExactHeadPassReceiptRequired",
        "qualification_receipt_ref": None,
        "source_corpus_sha256": "11cd7e2eb13c22493341edfd29ea4e37905506637612880ffe21e7c8fb9ae799",
    },
    "prod_eqp": {
        "source_pr": 5988,
        "source_head": "95631725daed7963fa26511d79775373034d9592",
        "qualifier_pr": 5993,
        "qualifier_head": "8482bc226452267fcdc7f00feed0475ceefc8fbc",
        "qualification_requirement": "HostedExactHeadPassReceiptRequired",
        "qualification_receipt_ref": None,
        "source_corpus_sha256": "2cfcd09a899599e0bec8c7da7424e07c89324608d94365330010a6ac2e996ea7",
    },
}

METHODS = {"Analysis", "Inspection", "Demonstration", "Test"}
PLANES = {"SOURCE", "FIELD", "MIXED"}

EXPECTED_REVIEW_STATE = {
    "DIR": "SourceReady",
    "IRR": "SourceReady",
    "QRR": "BlockedPendingRequiredQualificationReceipts",
    "RER": "NotEntered",
    "overall": "QualificationPending",
}

EXPECTED_MUTATIONS = [
    "none_baseline",
    "remove_need",
    "remove_req_verification_method",
    "mark_claim_relevant_interface_unresolved",
    "claim_shared_common_mode_as_independent_redundancy",
    "use_MODEL_for_FIELD_required_component_function",
    "promote_unbound_upstream_qualifier_to_PASS_without_receipt",
    "stale_calibration_for_claim_bearing_observation",
    "configuration_changed_without_impact_review",
    "repair_without_requalification",
    "use_Mycelix_service_event_as_engineering_function_evidence",
    "generalize_nonpotable_profile_to_potable_claim",
    "request_physical_operation_authority",
]

FORBIDDEN_KEYS = {
    "priority_score",
    "readiness_score",
    "self_sufficiency_score",
    "closure_score",
    "resilience_score",
    "civilization_score",
    "execution_authority",
    "actuation_authority",
    "procurement_authority",
    "allocation_authority",
}


def fail(msg: str) -> None:
    raise AssertionError(msg)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(obj: object) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def walk_keys(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def exact_ids(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:03d}" for i in range(1, count + 1)]


def base_oracle_state() -> dict[str, bool]:
    return {
        "need_present": True,
        "all_verification_methods_present": True,
        "claim_interfaces_resolved": True,
        "common_modes_honest": True,
        "field_evidence_honest": True,
        "upstream_state_honest": True,
        "calibration_current": True,
        "configuration_impact_current": True,
        "repair_requalified": True,
        "operational_engineering_boundary_intact": True,
        "scope_bounded": True,
        "no_physical_authority": True,
        "required_qualification_receipts_bound": False,
        "verification_complete": False,
        "validation_complete": False,
    }


def apply_mutation(state: dict[str, bool], mutation: str) -> None:
    if mutation == "none_baseline":
        return
    if mutation == "remove_need":
        state["need_present"] = False
        return
    if mutation == "remove_req_verification_method":
        state["all_verification_methods_present"] = False
        return
    if mutation == "mark_claim_relevant_interface_unresolved":
        state["claim_interfaces_resolved"] = False
        return
    if mutation == "claim_shared_common_mode_as_independent_redundancy":
        state["common_modes_honest"] = False
        return
    if mutation == "use_MODEL_for_FIELD_required_component_function":
        state["field_evidence_honest"] = False
        return
    if mutation == "promote_unbound_upstream_qualifier_to_PASS_without_receipt":
        state["upstream_state_honest"] = False
        return
    if mutation == "stale_calibration_for_claim_bearing_observation":
        state["calibration_current"] = False
        return
    if mutation == "configuration_changed_without_impact_review":
        state["configuration_impact_current"] = False
        return
    if mutation == "repair_without_requalification":
        state["repair_requalified"] = False
        return
    if mutation == "use_Mycelix_service_event_as_engineering_function_evidence":
        state["operational_engineering_boundary_intact"] = False
        return
    if mutation == "generalize_nonpotable_profile_to_potable_claim":
        state["scope_bounded"] = False
        return
    if mutation == "request_physical_operation_authority":
        state["no_physical_authority"] = False
        return
    fail(f"unknown mutation: {mutation}")


def derive_disposition(state: dict[str, bool]) -> str:
    if not state["need_present"] or not state["all_verification_methods_present"]:
        return "DesignIntentBlocked"
    if not state["claim_interfaces_resolved"] or not state["common_modes_honest"]:
        return "InterfaceRiskBlocked"
    if (
        not state["field_evidence_honest"]
        or not state["upstream_state_honest"]
        or not state["calibration_current"]
    ):
        return "QualificationReadinessBlocked"
    if (
        not state["configuration_impact_current"]
        or not state["repair_requalified"]
        or not state["operational_engineering_boundary_intact"]
        or not state["scope_bounded"]
        or not state["no_physical_authority"]
    ):
        return "ReleaseEvidenceBlocked"
    if not state["required_qualification_receipts_bound"]:
        return "QualificationPending"
    if not state["verification_complete"]:
        return "VerificationPending"
    if not state["validation_complete"]:
        return "VerifiedOnly"
    return "ValidatedReleaseEligible"


def main() -> None:
    packet_raw = PACKET_PATH.read_bytes()
    doc_raw = DOC_PATH.read_bytes()

    if sha256(packet_raw) != EXPECTED_PACKET_SHA256:
        fail("packet SHA-256 drift")
    if sha256(doc_raw) != EXPECTED_DOC_SHA256:
        fail("design document SHA-256 drift")

    packet = json.loads(packet_raw)
    if canonical_bytes(packet) != packet_raw:
        fail("packet is not canonical compact sorted-key JSON + final newline")

    if packet.get("schema") != EXPECTED_SCHEMA:
        fail("schema drift")
    if packet.get("authority") != EXPECTED_AUTHORITY:
        fail("authority drift")
    if packet.get("source_base") != EXPECTED_SOURCE_BASE:
        fail("source base drift")
    if packet.get("status") != "QualificationPending":
        fail("baseline status must remain QualificationPending")
    if packet.get("bindings") != EXPECTED_BINDINGS:
        fail("exact upstream binding drift")

    if packet.get("review_state") != EXPECTED_REVIEW_STATE:
        fail("baseline review-state drift")

    need = packet.get("need")
    if not isinstance(need, dict) or need.get("id") != "NEED-WATER-001":
        fail("need identity drift")
    non_goals = set(need.get("non_goals", []))
    required_non_goals = {
        "potable-water claim",
        "public-health claim",
        "dosing or disinfection",
        "autonomous actuation",
    }
    if not required_non_goals.issubset(non_goals):
        fail("critical non-goal missing")

    reqs = packet.get("requirements", [])
    constraints = packet.get("constraints", [])
    assumptions = packet.get("assumptions", [])
    interfaces = packet.get("interfaces", [])
    decisions = packet.get("decisions", [])
    risks = packet.get("risks", [])
    verification = packet.get("verification", [])
    validation = packet.get("validation", [])
    mutations = packet.get("mutation_cases", [])

    if len(reqs) != 12:
        fail("requirement count drift")
    if len(constraints) != 5:
        fail("constraint count drift")
    if len(assumptions) != 4:
        fail("assumption count drift")
    if len(interfaces) != 8:
        fail("interface count drift")
    if len(decisions) != 4:
        fail("decision count drift")
    if len(risks) != 7:
        fail("risk count drift")
    if len(verification) != 12:
        fail("verification count drift")
    if len(validation) != 1:
        fail("validation count drift")
    if len(mutations) != 13:
        fail("mutation count drift")

    req_ids = [r.get("id") for r in reqs]
    if req_ids != exact_ids("REQ-WATER-", 12):
        fail("requirement identity/order drift")
    if any(r.get("need_refs") != ["NEED-WATER-001"] for r in reqs):
        fail("requirement-to-need trace drift")
    if any(r.get("method") not in METHODS for r in reqs):
        fail("verification method vocabulary drift")
    if any(r.get("required_evidence_plane") not in PLANES for r in reqs):
        fail("evidence-plane vocabulary drift")

    expected_ver_ids = exact_ids("VER-WATER-", 12)
    actual_ver_ids = [v.get("id") for v in verification]
    if actual_ver_ids != expected_ver_ids:
        fail("verification identity/order drift")
    for index, item in enumerate(verification, start=1):
        if item.get("requirement_ref") != f"REQ-WATER-{index:03d}":
            fail("verification-to-requirement trace drift")
        if item.get("result") != "NotExecuted" or item.get("currentness") != "Unqualified":
            fail("source packet may not claim executed verification")

    val = validation[0]
    if val.get("id") != "VAL-WATER-001":
        fail("validation identity drift")
    if val.get("result") != "PlannedQualificationPending":
        fail("validation must remain pending")
    if val.get("need_refs") != ["NEED-WATER-001"]:
        fail("validation-to-need trace drift")

    # Upstream qualification receipts are prerequisites only. The frozen
    # consumer itself has not executed verification or validation.
    if any(v.get("result") != "NotExecuted" for v in verification):
        fail("source packet cannot pre-claim verification completion")
    if val.get("result") == "Pass":
        fail("source packet cannot pre-claim validation completion")

    assumption_ids = [a.get("id") for a in assumptions]
    if assumption_ids != exact_ids("ASM-WATER-", 4):
        fail("assumption identity/order drift")
    if any(not a.get("invalidation_condition") for a in assumptions):
        fail("every assumption requires an invalidation condition")

    interface_ids = [i.get("id") for i in interfaces]
    if interface_ids != exact_ids("IFC-WATER-", 8):
        fail("interface identity/order drift")
    if any(i.get("authority") != "none" for i in interfaces):
        fail("interface may not mint authority")

    decision_ids = [d.get("id") for d in decisions]
    if decision_ids != exact_ids("DEC-WATER-", 4):
        fail("decision identity/order drift")
    if any(not d.get("rejected") for d in decisions):
        fail("rejected alternatives must remain retained")

    risk_ids = [r.get("id") for r in risks]
    if risk_ids != exact_ids("RISK-WATER-", 7):
        fail("risk identity/order drift")
    valid_ver_refs = set(expected_ver_ids)
    if any(r.get("verification_route") not in valid_ver_refs for r in risks):
        fail("risk verification route drift")

    req_by_id = {r["id"]: r for r in reqs}
    if req_by_id["REQ-WATER-003"]["required_evidence_plane"] != "FIELD":
        fail("component-function requirement must remain FIELD-bound")
    if req_by_id["REQ-WATER-005"]["required_evidence_plane"] != "FIELD":
        fail("claim-bearing measurement must remain FIELD-bound")

    if packet.get("change_impact", {}).get("id") != "CHG-WATER-001":
        fail("change-impact identity drift")
    triggers = set(packet["change_impact"].get("triggers", []))
    required_triggers = {
        "component substitution or repair",
        "sensor/calibration/configuration changes",
        "claim scope changes",
    }
    if not required_triggers.issubset(triggers):
        fail("change-impact trigger coverage drift")

    all_keys = set(walk_keys(packet))
    bad = sorted(all_keys & FORBIDDEN_KEYS)
    if bad:
        fail(f"forbidden authority/score keys present: {bad}")
    scoreish = sorted(
        key for key in all_keys
        if isinstance(key, str)
        and (key.endswith("_score") or key in {"rank", "ranking", "priority_rank"})
    )
    if scoreish:
        fail(f"forbidden scalar ranking keys present: {scoreish}")

    mutation_ids = [m.get("id") for m in mutations]
    if mutation_ids != exact_ids("WD-CASE-", 13):
        fail("mutation identity/order drift")
    mutation_names = [m.get("mutation") for m in mutations]
    if mutation_names != EXPECTED_MUTATIONS:
        fail("mutation vocabulary/order drift")

    derived = []
    for case in mutations:
        state = base_oracle_state()
        apply_mutation(state, case["mutation"])
        outcome = derive_disposition(state)
        derived.append(outcome)
        if outcome != case.get("expected"):
            fail(
                f"{case['id']} expected-field mismatch: "
                f"derived={outcome} corpus={case.get('expected')}"
            )

    census = Counter(derived)
    expected_census = Counter({
        "QualificationPending": 1,
        "DesignIntentBlocked": 2,
        "InterfaceRiskBlocked": 2,
        "QualificationReadinessBlocked": 3,
        "ReleaseEvidenceBlocked": 5,
    })
    if census != expected_census:
        fail(f"derived mutation census drift: {dict(census)}")

    text = doc_raw.decode("utf-8")
    anchors = [
        "Hosted execution results belong in external evidence/receipts bound to the exact qualifier head.",
        "MODEL evidence may support analysis; it cannot silently substitute for FIELD evidence.",
        "A mitigation written in this document is not a verified mitigation.",
        "Verification completion will not automatically imply validation completion.",
        "No silent evidence carry-forward.",
        "Do not create the production WATER adapter merely because this packet exists.",
    ]
    for anchor in anchors:
        if anchor not in text:
            fail(f"document anchor missing: {anchor}")

    print(
        "PASS_WATER_MFG_001B_SOURCE_VALIDATION "
        f"packet_sha256={EXPECTED_PACKET_SHA256} "
        f"mutations={len(mutations)} "
        f"baseline={packet['status']} "
        f"census={dict(sorted(census.items()))}"
    )


if __name__ == "__main__":
    main()
