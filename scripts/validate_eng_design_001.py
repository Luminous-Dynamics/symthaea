#!/usr/bin/env python3
"""Independent reference validator for ENG-DESIGN-001.

This validator intentionally imports no Symthaea production code. It validates the
exact frozen synthetic design-process corpus and independently derives every
expected review/release outcome.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

CORPUS = Path("docs/release/evidence/eng-design-001-process-reference-v1.json")
EXPECTED_SHA256 = "5767253ad7bbf722e3fbafb7033480515973871d06c4935c60f397b0fe1332cb"
EXPECTED_SCHEMA = "eng-design-001-process-reference-v1"
EXPECTED_AUTHORITY = "repository_process_semantics_only_no_physical_execution_authority"
EXPECTED_ISSUE_REF = "#6005"
EXPECTED_SOURCE_BASE = "main@eae17187e199e3a53d108b437c0215b5ff812261"

EXPECTED_PROCESS_IDS = [
    "NEED",
    "REQ",
    "CON",
    "ASM",
    "IFC",
    "DEC",
    "RISK",
    "VER",
    "VAL",
    "CFG",
    "EVD",
    "CHG",
]
EXPECTED_GATES = [
    "DesignIntentReview",
    "InterfaceRiskReview",
    "QualificationReadinessReview",
    "ReleaseEvidenceReview",
]
EXPECTED_METHODS = ["Analysis", "Inspection", "Demonstration", "Test"]
EXPECTED_PLANES = ["MODEL", "FIELD"]
EXPECTED_CASE_IDS = [f"EDP-{i:03d}" for i in range(1, 17)]
EXPECTED_FORBIDDEN = [
    "priority_score",
    "readiness_score",
    "self_sufficiency_score",
    "closure_score",
    "universal_rank",
]
EXPECTED_INVARIANTS = [
    "design_intent_is_not_requirement_satisfaction",
    "requirement_written_is_not_requirement_verified",
    "model_pass_is_not_field_verification",
    "verification_is_not_validation",
    "prototype_operation_is_not_process_capability",
    "one_tested_article_is_not_all_articles",
    "nominal_interface_compatibility_is_not_integrated_qualification",
    "configuration_change_invalidates_unrequalified_currentness",
    "paper_risk_mitigation_is_not_verified_mitigation",
    "repair_is_not_requalification",
    "operational_fact_is_not_engineering_qualification",
    "engineering_recommendation_is_not_physical_execution_authority",
]
EXPECTED_OUTCOME_COUNTS = {
    "DesignIntentBlocked": 3,
    "InterfaceRiskBlocked": 4,
    "QualificationReadinessBlocked": 2,
    "ReleaseEvidenceBlocked": 5,
    "VerifiedOnly": 1,
    "ValidatedReleaseEligible": 1,
}

EXPECTED_CASE_KEYS = {
    "case_id",
    "purpose",
    "need_defined",
    "requirements_traceable",
    "verification_methods_defined",
    "assumptions_current",
    "interfaces_resolved",
    "decision_history_preserved",
    "risk_mitigations_verified_or_open",
    "verification_complete",
    "verification_required_plane",
    "verification_evidence_plane",
    "validation_complete",
    "evidence_config_matches_live",
    "repair_requalified_or_not_applicable",
    "common_modes_resolved_or_not_claimed",
    "scope_not_overgeneralized",
    "cross_owner_boundary_clean",
    "execution_authority_requested",
    "expected",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL_ENG_DESIGN_001_REFERENCE: {message}")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_keys(child)


def derive(case: dict[str, Any]) -> str:
    # Review precedence is deliberate: later review states cannot launder an
    # earlier blocked design state into apparent readiness.
    if (
        not case["need_defined"]
        or not case["requirements_traceable"]
        or not case["verification_methods_defined"]
    ):
        return "DesignIntentBlocked"

    if (
        not case["assumptions_current"]
        or not case["interfaces_resolved"]
        or not case["decision_history_preserved"]
        or not case["risk_mitigations_verified_or_open"]
    ):
        return "InterfaceRiskBlocked"

    if (
        not case["verification_complete"]
        or case["verification_evidence_plane"] != case["verification_required_plane"]
    ):
        return "QualificationReadinessBlocked"

    if (
        not case["evidence_config_matches_live"]
        or not case["repair_requalified_or_not_applicable"]
        or not case["common_modes_resolved_or_not_claimed"]
        or not case["scope_not_overgeneralized"]
        or not case["cross_owner_boundary_clean"]
        or case["execution_authority_requested"]
    ):
        return "ReleaseEvidenceBlocked"

    if not case["validation_complete"]:
        return "VerifiedOnly"

    return "ValidatedReleaseEligible"


def require_case(cases: dict[str, dict[str, Any]], case_id: str, **expected: Any) -> None:
    case = cases[case_id]
    for key, value in expected.items():
        if case.get(key) != value:
            fail(f"{case_id}: expected {key}={value!r}, got {case.get(key)!r}")


def main() -> None:
    raw = CORPUS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        fail(f"digest drift: {digest}")

    data = json.loads(raw)
    if raw != canonical_bytes(data):
        fail("corpus is not canonical compact sorted-key JSON plus final newline")

    if data.get("schema") != EXPECTED_SCHEMA:
        fail("schema drift")
    if data.get("authority") != EXPECTED_AUTHORITY:
        fail("authority drift")
    if data.get("issue_ref") != EXPECTED_ISSUE_REF:
        fail("issue binding drift")
    if data.get("source_base") != EXPECTED_SOURCE_BASE:
        fail("source-base drift")

    if data.get("process_ids") != EXPECTED_PROCESS_IDS:
        fail("typed design-thread ID vocabulary drift")
    if data.get("review_gates") != EXPECTED_GATES:
        fail("review-gate vocabulary/order drift")
    if data.get("verification_methods") != EXPECTED_METHODS:
        fail("verification-method vocabulary drift")
    if data.get("evidence_planes") != EXPECTED_PLANES:
        fail("evidence-plane vocabulary drift")
    if data.get("forbidden_scalar_keys") != EXPECTED_FORBIDDEN:
        fail("forbidden scalar-key list drift")
    if data.get("invariants") != EXPECTED_INVARIANTS:
        fail("process invariant drift")

    observed_keys = set(walk_keys(data))
    leaked_forbidden = observed_keys.intersection(EXPECTED_FORBIDDEN)
    if leaked_forbidden:
        fail(f"forbidden scalar key present in corpus: {sorted(leaked_forbidden)}")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) != 16:
        fail("expected exactly 16 cases")

    cases: dict[str, dict[str, Any]] = {}
    outcomes: Counter[str] = Counter()
    for case in raw_cases:
        if not isinstance(case, dict):
            fail("case is not an object")
        if set(case) != EXPECTED_CASE_KEYS:
            fail(f"{case.get('case_id', '<unknown>')}: case shape drift")

        case_id = case["case_id"]
        if case_id in cases:
            fail(f"duplicate case ID: {case_id}")
        cases[case_id] = case

        if case["verification_required_plane"] not in EXPECTED_PLANES:
            fail(f"{case_id}: unknown required evidence plane")
        if case["verification_evidence_plane"] not in EXPECTED_PLANES:
            fail(f"{case_id}: unknown actual evidence plane")

        derived = derive(case)
        if case["expected"] != derived:
            fail(f"{case_id}: expected={case['expected']} independently-derived={derived}")
        outcomes[derived] += 1

    if list(cases) != EXPECTED_CASE_IDS:
        fail(f"case identity/order drift: {list(cases)}")
    if dict(outcomes) != EXPECTED_OUTCOME_COUNTS:
        fail(f"outcome census drift: {dict(outcomes)}")

    # Semantic anchors: these prevent a superficially self-consistent corpus
    # from erasing the design-process distinctions this benchmark exists to hold.
    require_case(
        cases,
        "EDP-010",
        verification_complete=True,
        verification_required_plane="FIELD",
        verification_evidence_plane="MODEL",
        expected="QualificationReadinessBlocked",
    )
    require_case(
        cases,
        "EDP-011",
        verification_complete=True,
        validation_complete=False,
        expected="VerifiedOnly",
    )
    require_case(
        cases,
        "EDP-012",
        evidence_config_matches_live=False,
        expected="ReleaseEvidenceBlocked",
    )
    require_case(
        cases,
        "EDP-013",
        repair_requalified_or_not_applicable=False,
        expected="ReleaseEvidenceBlocked",
    )
    require_case(
        cases,
        "EDP-014",
        common_modes_resolved_or_not_claimed=False,
        expected="ReleaseEvidenceBlocked",
    )
    require_case(
        cases,
        "EDP-015",
        scope_not_overgeneralized=False,
        expected="ReleaseEvidenceBlocked",
    )
    require_case(
        cases,
        "EDP-016",
        cross_owner_boundary_clean=False,
        execution_authority_requested=True,
        expected="ReleaseEvidenceBlocked",
    )

    eligible = [cid for cid, case in cases.items() if derive(case) == "ValidatedReleaseEligible"]
    if eligible != ["EDP-001"]:
        fail(f"unexpected validated-release-eligible cases: {eligible}")

    for case_id, case in cases.items():
        if derive(case) in {"VerifiedOnly", "ValidatedReleaseEligible"} and case["execution_authority_requested"]:
            fail(f"{case_id}: nonblocked design state requests physical execution authority")

    print(
        "PASS_ENG_DESIGN_001_REFERENCE "
        f"digest={digest} cases=16 outcomes={json.dumps(dict(outcomes), sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
