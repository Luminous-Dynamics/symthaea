#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ETK v1 discharge-receipt/currentness reference oracle.

Standard-library-only; imports no Symthaea code.

Core theorem:
    admitted evidence
    != discharge receipt
    != present-tense discharged obligation
    != typed current-discharge fact

A receipt records a historical authority transition against an issuance-time
obligation snapshot. Whether it still discharges an obligation is separately
derived from the current obligation and current engineering context. A typed
current-discharge fact is minted only for that positive present-tense theorem
and binds the exact historical receipt that witnesses it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from typing import Any

SCHEMA = "symthaea.etk-obligation-discharge-reference.v1"
OBLIGATION_SCHEMA = "symthaea.etk-proof-obligation-snapshot.v1"
OBLIGATION_DOMAIN = b"symthaea.etk-proof-obligation-snapshot.v1\x00"
RECEIPT_DOMAIN = b"symthaea.etk-obligation-discharge-receipt.v1\x00"
CURRENT_FACT_SCHEMA = "symthaea.etk-current-obligation-discharge-fact.v1"
CURRENT_FACT_DOMAIN = b"symthaea.etk-current-obligation-discharge-fact.v1\x00"

EXPECTED_CURRENT_FACT = (
    "sha256:ee766da9c94291c2c46579219d013684bdeef711ef075f9041e15d646e52397f"
)

TOP = {
    "schema",
    "issued_obligation",
    "admitted",
    "current_obligation",
    "current_context",
}
OBL = {"obligation_id", "claim", "expected_evidence_kind"}
ADMITTED = {
    "admitted_evidence_id",
    "candidate_artifact_id",
    "obligation_id",
    "obligation_revision",
    "subject_id",
    "twin_revision",
    "requirement_revision",
    "evidence_policy_id",
    "validity_domain_id",
    "currentness_proof_id",
}
CURRENT = {
    "subject_id",
    "twin_revision",
    "requirement_revision",
    "validity_domain_id",
    "currentness_proof_id",
}

ISSUANCE_ORDER = (
    "schema_mismatch",
    "unknown_top_level_field",
    "malformed_issued_obligation",
    "unknown_issued_obligation_field",
    "malformed_admitted_evidence",
    "unknown_admitted_field",
    "malformed_current_obligation",
    "unknown_current_obligation_field",
    "malformed_current_context",
    "unknown_current_field",
    "issued_not_simulation_obligation",
    "admitted_identity_incomplete",
    "admitted_obligation_id_mismatch",
    "admitted_obligation_revision_mismatch",
)
ISSUANCE_RANK = {reason: index for index, reason in enumerate(ISSUANCE_ORDER)}

CURRENTNESS_ORDER = (
    "obligation_id_changed",
    "obligation_revision_changed",
    "current_not_simulation_obligation",
    "subject_changed",
    "twin_revision_changed",
    "requirement_revision_changed",
    "validity_domain_changed",
    "currentness_proof_changed",
)
CURRENTNESS_RANK = {reason: index for index, reason in enumerate(CURRENTNESS_ORDER)}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def unknown(value: Any, allowed: set[str]) -> bool:
    return isinstance(value, dict) and bool(set(value) - allowed)


def domain_hash(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + canonical(value)).hexdigest()


def obligation_snapshot_id(obligation: dict[str, Any]) -> str:
    return domain_hash(
        OBLIGATION_DOMAIN,
        {
            "claim": obligation["claim"],
            "expected_evidence_kind": obligation["expected_evidence_kind"],
            "obligation_id": obligation["obligation_id"],
            "schema": OBLIGATION_SCHEMA,
        },
    )


def receipt_preimage(
    admitted: dict[str, Any], obligation_revision: str
) -> dict[str, Any]:
    return {
        "admitted_evidence_id": admitted["admitted_evidence_id"],
        "candidate_artifact_id": admitted["candidate_artifact_id"],
        "currentness_proof_id": admitted["currentness_proof_id"],
        "evidence_policy_id": admitted["evidence_policy_id"],
        "obligation_id": admitted["obligation_id"],
        "obligation_revision": obligation_revision,
        "requirement_revision": admitted["requirement_revision"],
        "subject_id": admitted["subject_id"],
        "twin_revision": admitted["twin_revision"],
        "validity_domain_id": admitted["validity_domain_id"],
    }


def current_fact_preimage(
    current_obligation: dict[str, Any],
    current_context: dict[str, Any],
    obligation_revision: str,
    witness_receipt_id: str,
) -> dict[str, Any]:
    return {
        "currentness_proof_id": current_context["currentness_proof_id"],
        "obligation_id": current_obligation["obligation_id"],
        "obligation_revision": obligation_revision,
        "requirement_revision": current_context["requirement_revision"],
        "schema": CURRENT_FACT_SCHEMA,
        "subject_id": current_context["subject_id"],
        "twin_revision": current_context["twin_revision"],
        "validity_domain_id": current_context["validity_domain_id"],
        "witness_receipt_id": witness_receipt_id,
    }


def deny_issuance(reasons: set[str]) -> dict[str, Any]:
    return {
        "decision": "DenyReceiptIssuance",
        "reasons": sorted(reasons, key=ISSUANCE_RANK.__getitem__),
    }


def validate_obligation(
    value: Any,
    malformed_reason: str,
    unknown_reason: str,
    reasons: set[str],
) -> bool:
    if not isinstance(value, dict):
        reasons.add(malformed_reason)
        return False
    if unknown(value, OBL):
        reasons.add(unknown_reason)
    if any(not nonempty(value.get(key)) for key in OBL):
        reasons.add(malformed_reason)
        return False
    return True


def evaluate(payload: Any) -> dict[str, Any]:
    reasons: set[str] = set()
    if not isinstance(payload, dict):
        return deny_issuance({"malformed_issued_obligation"})

    if payload.get("schema") != SCHEMA:
        reasons.add("schema_mismatch")
    if unknown(payload, TOP):
        reasons.add("unknown_top_level_field")

    issued_obligation = payload.get("issued_obligation")
    admitted = payload.get("admitted")
    current_obligation = payload.get("current_obligation")
    current_context = payload.get("current_context")

    issued_ok = validate_obligation(
        issued_obligation,
        "malformed_issued_obligation",
        "unknown_issued_obligation_field",
        reasons,
    )
    current_obligation_ok = validate_obligation(
        current_obligation,
        "malformed_current_obligation",
        "unknown_current_obligation_field",
        reasons,
    )

    if not isinstance(admitted, dict):
        reasons.add("malformed_admitted_evidence")
    else:
        if unknown(admitted, ADMITTED):
            reasons.add("unknown_admitted_field")
        if any(not nonempty(admitted.get(key)) for key in ADMITTED):
            reasons.add("admitted_identity_incomplete")

    if not isinstance(current_context, dict):
        reasons.add("malformed_current_context")
    else:
        if unknown(current_context, CURRENT):
            reasons.add("unknown_current_field")
        if any(not nonempty(current_context.get(key)) for key in CURRENT):
            reasons.add("malformed_current_context")

    if not all(
        isinstance(value, dict)
        for value in (issued_obligation, admitted, current_obligation, current_context)
    ):
        return deny_issuance(reasons)

    issued_snapshot = None
    if issued_ok:
        if issued_obligation["expected_evidence_kind"] != "Simulation":
            reasons.add("issued_not_simulation_obligation")
        issued_snapshot = obligation_snapshot_id(issued_obligation)
        if admitted.get("obligation_id") != issued_obligation["obligation_id"]:
            reasons.add("admitted_obligation_id_mismatch")
        if admitted.get("obligation_revision") != issued_snapshot:
            reasons.add("admitted_obligation_revision_mismatch")

    if reasons:
        return deny_issuance(reasons)

    assert issued_snapshot is not None
    receipt_id = domain_hash(
        RECEIPT_DOMAIN,
        receipt_preimage(admitted, issued_snapshot),
    )

    stale: set[str] = set()
    current_snapshot = obligation_snapshot_id(current_obligation)
    if current_obligation_ok:
        if current_obligation["obligation_id"] != issued_obligation["obligation_id"]:
            stale.add("obligation_id_changed")
        if current_snapshot != issued_snapshot:
            stale.add("obligation_revision_changed")
        if current_obligation["expected_evidence_kind"] != "Simulation":
            stale.add("current_not_simulation_obligation")

    if admitted["subject_id"] != current_context["subject_id"]:
        stale.add("subject_changed")
    if admitted["twin_revision"] != current_context["twin_revision"]:
        stale.add("twin_revision_changed")
    if admitted["requirement_revision"] != current_context["requirement_revision"]:
        stale.add("requirement_revision_changed")
    if admitted["validity_domain_id"] != current_context["validity_domain_id"]:
        stale.add("validity_domain_changed")
    if admitted["currentness_proof_id"] != current_context["currentness_proof_id"]:
        stale.add("currentness_proof_changed")

    common = {
        "receipt_id": receipt_id,
        "obligation_id": issued_obligation["obligation_id"],
        "obligation_revision": issued_snapshot,
        "admitted_evidence_id": admitted["admitted_evidence_id"],
    }
    if stale:
        return {
            "decision": "HistoricalReceipt",
            **common,
            "reasons": sorted(stale, key=CURRENTNESS_RANK.__getitem__),
        }

    fact_preimage = current_fact_preimage(
        current_obligation,
        current_context,
        current_snapshot,
        receipt_id,
    )
    return {
        "decision": "CurrentDischarge",
        **common,
        "current_discharge_fact_id": domain_hash(CURRENT_FACT_DOMAIN, fact_preimage),
    }


def fixture() -> dict[str, Any]:
    obligation = {
        "obligation_id": "11111111-2222-4333-8444-555555555555",
        "claim": "stress remains below allowable under service load",
        "expected_evidence_kind": "Simulation",
    }
    revision = obligation_snapshot_id(obligation)
    return {
        "schema": SCHEMA,
        "issued_obligation": copy.deepcopy(obligation),
        "admitted": {
            "admitted_evidence_id": "sha256:794475a988dbecf945306f051a54e000e03fe918c886ee1cc13a6f28b7ad9b10",
            "candidate_artifact_id": "solver-output:run-0007",
            "obligation_id": obligation["obligation_id"],
            "obligation_revision": revision,
            "subject_id": "bracket-alpha",
            "twin_revision": "design:G17",
            "requirement_revision": "REQ-STRESS:r5",
            "evidence_policy_id": "ETK-SIM-ADMISSION-V1",
            "validity_domain_id": "VD-static-G17-LC9",
            "currentness_proof_id": "currentness:design-G17:attestation-1",
        },
        "current_obligation": copy.deepcopy(obligation),
        "current_context": {
            "subject_id": "bracket-alpha",
            "twin_revision": "design:G17",
            "requirement_revision": "REQ-STRESS:r5",
            "validity_domain_id": "VD-static-G17-LC9",
            "currentness_proof_id": "currentness:design-G17:attestation-1",
        },
    }


def expect(payload: dict[str, Any], decision: str, reason: str | None = None) -> None:
    result = evaluate(payload)
    assert result["decision"] == decision, result
    if reason is not None:
        assert reason in result["reasons"], result


def self_test() -> tuple[str, str, str]:
    good = fixture()
    result = evaluate(good)
    assert result["decision"] == "CurrentDischarge", result
    assert result == evaluate(copy.deepcopy(good))

    expected_snapshot = good["admitted"]["obligation_revision"]
    expected_receipt = result["receipt_id"]
    expected_fact = result["current_discharge_fact_id"]
    assert expected_fact == EXPECTED_CURRENT_FACT

    cases: list[tuple[dict[str, Any], str, str]] = []

    payload = copy.deepcopy(good)
    payload["admitted"]["obligation_id"] = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    cases.append((payload, "DenyReceiptIssuance", "admitted_obligation_id_mismatch"))

    payload = copy.deepcopy(good)
    payload["admitted"]["obligation_revision"] = "sha256:stale"
    cases.append((payload, "DenyReceiptIssuance", "admitted_obligation_revision_mismatch"))

    payload = copy.deepcopy(good)
    payload["issued_obligation"]["expected_evidence_kind"] = "Telemetry"
    cases.append((payload, "DenyReceiptIssuance", "issued_not_simulation_obligation"))

    payload = copy.deepcopy(good)
    payload["current_obligation"]["claim"] += " with fatigue margin"
    cases.append((payload, "HistoricalReceipt", "obligation_revision_changed"))

    payload = copy.deepcopy(good)
    payload["current_obligation"]["obligation_id"] = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    cases.append((payload, "HistoricalReceipt", "obligation_id_changed"))

    payload = copy.deepcopy(good)
    payload["current_obligation"]["expected_evidence_kind"] = "Telemetry"
    cases.append((payload, "HistoricalReceipt", "current_not_simulation_obligation"))

    for field, reason in (
        ("subject_id", "subject_changed"),
        ("twin_revision", "twin_revision_changed"),
        ("requirement_revision", "requirement_revision_changed"),
        ("validity_domain_id", "validity_domain_changed"),
        ("currentness_proof_id", "currentness_proof_changed"),
    ):
        payload = copy.deepcopy(good)
        payload["current_context"][field] += ":new"
        cases.append((payload, "HistoricalReceipt", reason))

    payload = copy.deepcopy(good)
    payload["admitted"]["shadow_authority"] = True
    cases.append((payload, "DenyReceiptIssuance", "unknown_admitted_field"))

    payload = copy.deepcopy(good)
    payload["current_context"]["currentness_proof_id"] = ""
    cases.append((payload, "DenyReceiptIssuance", "malformed_current_context"))

    for payload, decision, reason in cases:
        expect(payload, decision, reason)

    payload = copy.deepcopy(good)
    payload["current_context"]["twin_revision"] = "design:G18"
    payload["current_context"]["currentness_proof_id"] = "currentness:design-G18:attestation-2"
    result = evaluate(payload)
    assert result["reasons"] == [
        "twin_revision_changed",
        "currentness_proof_changed",
    ], result
    assert "current_discharge_fact_id" not in result

    historical = copy.deepcopy(good)
    historical["current_obligation"]["claim"] += " with fatigue margin"
    historical["current_context"]["currentness_proof_id"] = (
        "currentness:design-G17:attestation-2"
    )
    historical_result = evaluate(historical)
    assert historical_result["decision"] == "HistoricalReceipt"
    assert historical_result["receipt_id"] == expected_receipt
    assert "current_discharge_fact_id" not in historical_result

    refreshed = copy.deepcopy(good)
    refreshed["admitted"]["currentness_proof_id"] = (
        "currentness:design-G17:attestation-2"
    )
    refreshed["current_context"]["currentness_proof_id"] = (
        "currentness:design-G17:attestation-2"
    )
    refreshed["admitted"]["admitted_evidence_id"] = "sha256:" + "ab" * 32
    refreshed_result = evaluate(refreshed)
    assert refreshed_result["decision"] == "CurrentDischarge"
    assert refreshed_result["current_discharge_fact_id"] != expected_fact

    return expected_snapshot, expected_receipt, expected_fact


def reject_constant(value: str) -> None:
    raise ValueError("non-standard JSON constant: " + value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        snapshot, receipt, fact = self_test()
        print("ok obligation_snapshot=" + snapshot)
        print("ok discharge_receipt=" + receipt)
        print("ok current_discharge_fact=" + fact)
        return 0

    try:
        payload = json.load(sys.stdin, parse_constant=reject_constant)
        print(canonical(evaluate(payload)).decode())
        return 0
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        print("invalid input: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
