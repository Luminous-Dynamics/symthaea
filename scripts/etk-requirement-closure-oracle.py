#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ETK requirement-decomposition / current-satisfaction oracle.

Standard-library-only; imports no Symthaea code.

Core theorem:
    requirement-obligation relationship
    != complete requirement decomposition
    != present-tense satisfied requirement

V1 deliberately supports only an explicit AllOf decomposition.  Every bound
obligation must be represented by a lower-layer present-discharge fact under
the exact current requirement revision.  Raw historical discharge receipts
are deliberately not accepted here.  Extra discharge facts never compensate
for a missing member.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from typing import Any

CONTRACT_SCHEMA = "symthaea.etk-requirement-verification-contract.v1"
CONTRACT_DOMAIN = b"symthaea.etk-requirement-verification-contract.v1\x00"
SATISFACTION_SCHEMA = "symthaea.etk-requirement-satisfaction-receipt.v1"
SATISFACTION_DOMAIN = b"symthaea.etk-requirement-satisfaction-receipt.v1\x00"

REQ_ID = "sha256:e340c5030eebc978c41443ffd64f340dc5febad31e376080340bacfceda60faa"
OBL_A = "sha256:743d2c13cfcc52bdfb4cfd9a4a836ed0806ace7b2a68b4c7284b1910d8863c29"
REL_A = "sha256:1cc54ee11ff9b7e99279d4c9a9d43f751809f7f30a43469cff8e18fe70eee408"

# Independent second obligation/relationship endpoint used to prove AllOf.
OBL_B = "sha256:9d66d0bf7385930b596052a466f13517b36a06ef24ee65039153b76b75f9ef50"
REL_B = "sha256:037a3f3c9f333b574cd7834a14ab01b59b9baa88a65b4e5bb1c08b60910927e0"

DECOMPOSITION_POLICY = "sha256:" + "61" * 32
DECOMPOSITION_ACCEPTANCE = "sha256:" + "62" * 32
CURRENTNESS_ASSERTION = "sha256:" + "63" * 32

class ClosureError(ValueError):
    pass

def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")

def domain_hash(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + canonical_json(value)).hexdigest()

def digest(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ClosureError(f"{field} must be a string")
    if not value.startswith("sha256:") or len(value) != 71:
        raise ClosureError(f"{field} must be canonical sha256:<64 lowercase hex>")
    h = value[7:]
    if any(ch not in "0123456789abcdef" for ch in h):
        raise ClosureError(f"{field} must be canonical sha256:<64 lowercase hex>")
    return value

def relationship(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ClosureError(f"{field} must be an object")
    allowed = {"relationship_id", "requirement_revision_id", "obligation_revision_id"}
    if set(value) != allowed:
        raise ClosureError(f"{field} has unknown/missing fields")
    return {
        "relationship_id": digest(value["relationship_id"], f"{field}.relationship_id"),
        "requirement_revision_id": digest(
            value["requirement_revision_id"], f"{field}.requirement_revision_id"
        ),
        "obligation_revision_id": digest(
            value["obligation_revision_id"], f"{field}.obligation_revision_id"
        ),
    }

def make_contract(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ClosureError("contract input must be an object")
    allowed = {
        "schema",
        "requirement_revision_id",
        "composition",
        "relationships",
        "decomposition_policy_revision_id",
        "decomposition_acceptance_record_digest",
    }
    if set(payload) != allowed:
        raise ClosureError("contract input has unknown/missing fields")
    if payload["schema"] != CONTRACT_SCHEMA:
        raise ClosureError("wrong contract schema")
    req = digest(payload["requirement_revision_id"], "requirement_revision_id")
    if payload["composition"] != "AllOf":
        raise ClosureError("V1 supports only AllOf composition")
    if not isinstance(payload["relationships"], list) or not payload["relationships"]:
        raise ClosureError("AllOf requires at least one relationship")

    relations = [relationship(v, f"relationships[{i}]") for i, v in enumerate(payload["relationships"])]
    relation_ids: set[str] = set()
    obligation_ids: set[str] = set()
    for rel in relations:
        if rel["requirement_revision_id"] != req:
            raise ClosureError("relationship targets a different requirement revision")
        if rel["relationship_id"] in relation_ids:
            raise ClosureError("duplicate relationship identity")
        if rel["obligation_revision_id"] in obligation_ids:
            raise ClosureError("duplicate obligation revision")
        relation_ids.add(rel["relationship_id"])
        obligation_ids.add(rel["obligation_revision_id"])

    canonical_relations = sorted(
        relations,
        key=lambda v: (v["obligation_revision_id"], v["relationship_id"]),
    )
    preimage = {
        "composition": "AllOf",
        "decomposition_acceptance_record_digest": digest(
            payload["decomposition_acceptance_record_digest"],
            "decomposition_acceptance_record_digest",
        ),
        "decomposition_policy_revision_id": digest(
            payload["decomposition_policy_revision_id"],
            "decomposition_policy_revision_id",
        ),
        "relationships": canonical_relations,
        "requirement_revision_id": req,
        "schema": CONTRACT_SCHEMA,
    }
    return {
        "verification_contract_id": domain_hash(CONTRACT_DOMAIN, preimage),
        **preimage,
    }

def evaluate_current_satisfaction(
    contract: Any,
    *,
    current_requirement_revision_id: str,
    current_discharge_facts: Any,
    currentness_assertion_id: str,
) -> dict[str, Any]:
    if not isinstance(contract, dict) or "verification_contract_id" not in contract:
        raise ClosureError("contract must be a constructed verification contract")
    req = digest(current_requirement_revision_id, "current_requirement_revision_id")
    currentness = digest(currentness_assertion_id, "currentness_assertion_id")
    if not isinstance(current_discharge_facts, list):
        raise ClosureError("current_discharge_facts must be a list")

    discharge_facts: dict[str, str] = {}
    for i, item in enumerate(current_discharge_facts):
        if not isinstance(item, dict) or set(item) != {
            "obligation_revision_id",
            "current_discharge_fact_id",
        }:
            raise ClosureError(f"current_discharge_facts[{i}] malformed")
        obligation = digest(
            item["obligation_revision_id"],
            f"current_discharge_facts[{i}].obligation_revision_id",
        )
        fact = digest(
            item["current_discharge_fact_id"],
            f"current_discharge_facts[{i}].current_discharge_fact_id",
        )
        if obligation in discharge_facts:
            raise ClosureError("duplicate current discharge fact for obligation revision")
        discharge_facts[obligation] = fact

    if req != contract["requirement_revision_id"]:
        return {
            "decision": "HistoricalVerificationContract",
            "verification_contract_id": contract["verification_contract_id"],
            "requirement_revision_id": contract["requirement_revision_id"],
            "reasons": ["RequirementRevisionChanged"],
        }

    required = [r["obligation_revision_id"] for r in contract["relationships"]]
    missing = sorted(ob for ob in required if ob not in discharge_facts)
    if missing:
        return {
            "decision": "RequirementUnsatisfied",
            "verification_contract_id": contract["verification_contract_id"],
            "requirement_revision_id": req,
            "missing_obligation_revision_ids": missing,
        }

    used = [
        {
            "obligation_revision_id": ob,
            "current_discharge_fact_id": discharge_facts[ob],
        }
        for ob in sorted(required)
    ]
    preimage = {
        "currentness_assertion_id": currentness,
        "current_discharge_facts": used,
        "requirement_revision_id": req,
        "schema": SATISFACTION_SCHEMA,
        "verification_contract_id": contract["verification_contract_id"],
    }
    return {
        "decision": "CurrentRequirementSatisfied",
        "requirement_satisfaction_receipt_id": domain_hash(SATISFACTION_DOMAIN, preimage),
        **preimage,
    }

def fixture_contract() -> dict[str, Any]:
    return make_contract(
        {
            "schema": CONTRACT_SCHEMA,
            "requirement_revision_id": REQ_ID,
            "composition": "AllOf",
            "relationships": [
                {
                    "relationship_id": REL_B,
                    "requirement_revision_id": REQ_ID,
                    "obligation_revision_id": OBL_B,
                },
                {
                    "relationship_id": REL_A,
                    "requirement_revision_id": REQ_ID,
                    "obligation_revision_id": OBL_A,
                },
            ],
            "decomposition_policy_revision_id": DECOMPOSITION_POLICY,
            "decomposition_acceptance_record_digest": DECOMPOSITION_ACCEPTANCE,
        }
    )

def self_test() -> None:
    contract = fixture_contract()

    # Ordering is semantic no-op.
    reordered = make_contract(
        {
            "schema": CONTRACT_SCHEMA,
            "requirement_revision_id": REQ_ID,
            "composition": "AllOf",
            "relationships": list(reversed(contract["relationships"])),
            "decomposition_policy_revision_id": DECOMPOSITION_POLICY,
            "decomposition_acceptance_record_digest": DECOMPOSITION_ACCEPTANCE,
        }
    )
    assert reordered["verification_contract_id"] == contract["verification_contract_id"]

    # One discharge cannot satisfy a two-obligation AllOf contract.
    partial = evaluate_current_satisfaction(
        contract,
        current_requirement_revision_id=REQ_ID,
        current_discharge_facts=[
            {"obligation_revision_id": OBL_A, "current_discharge_fact_id": "sha256:" + "71"*32}
        ],
        currentness_assertion_id=CURRENTNESS_ASSERTION,
    )
    assert partial["decision"] == "RequirementUnsatisfied"
    assert partial["missing_obligation_revision_ids"] == [OBL_B]

    # Extra unrelated discharges do not compensate and do not enter the receipt.
    complete = evaluate_current_satisfaction(
        contract,
        current_requirement_revision_id=REQ_ID,
        current_discharge_facts=[
            {"obligation_revision_id": OBL_B, "current_discharge_fact_id": "sha256:" + "72"*32},
            {"obligation_revision_id": "sha256:" + "73"*32, "current_discharge_fact_id": "sha256:" + "74"*32},
            {"obligation_revision_id": OBL_A, "current_discharge_fact_id": "sha256:" + "71"*32},
        ],
        currentness_assertion_id=CURRENTNESS_ASSERTION,
    )
    assert complete["decision"] == "CurrentRequirementSatisfied"
    assert [x["obligation_revision_id"] for x in complete["current_discharge_facts"]] == sorted([OBL_A, OBL_B])

    # Requirement mutation makes the exact decomposition historical.
    historical = evaluate_current_satisfaction(
        contract,
        current_requirement_revision_id="sha256:" + "75"*32,
        current_discharge_facts=[],
        currentness_assertion_id=CURRENTNESS_ASSERTION,
    )
    assert historical["decision"] == "HistoricalVerificationContract"

    # Currentness refresh changes satisfaction receipt but not decomposition identity.
    refreshed = evaluate_current_satisfaction(
        contract,
        current_requirement_revision_id=REQ_ID,
        current_discharge_facts=[
            {"obligation_revision_id": OBL_A, "current_discharge_fact_id": "sha256:" + "71"*32},
            {"obligation_revision_id": OBL_B, "current_discharge_fact_id": "sha256:" + "72"*32},
        ],
        currentness_assertion_id="sha256:" + "76"*32,
    )
    assert refreshed["requirement_satisfaction_receipt_id"] != complete["requirement_satisfaction_receipt_id"]
    assert contract["verification_contract_id"] == reordered["verification_contract_id"]

    # Duplicate relationship / duplicate obligation / cross-requirement / non-AllOf deny.
    bad_cases = []
    d = {
        "schema": CONTRACT_SCHEMA,
        "requirement_revision_id": REQ_ID,
        "composition": "AllOf",
        "relationships": [
            {"relationship_id": REL_A, "requirement_revision_id": REQ_ID, "obligation_revision_id": OBL_A},
            {"relationship_id": REL_A, "requirement_revision_id": REQ_ID, "obligation_revision_id": OBL_B},
        ],
        "decomposition_policy_revision_id": DECOMPOSITION_POLICY,
        "decomposition_acceptance_record_digest": DECOMPOSITION_ACCEPTANCE,
    }
    bad_cases.append(d)
    d2 = copy.deepcopy(d)
    d2["relationships"][1]["relationship_id"] = REL_B
    d2["relationships"][1]["obligation_revision_id"] = OBL_A
    bad_cases.append(d2)
    d3 = copy.deepcopy(d)
    d3["relationships"][1]["relationship_id"] = REL_B
    d3["relationships"][1]["requirement_revision_id"] = "sha256:" + "77"*32
    bad_cases.append(d3)
    d4 = copy.deepcopy(d)
    d4["composition"] = "AnyOf"
    d4["relationships"] = d4["relationships"][:1]
    bad_cases.append(d4)
    for bad in bad_cases:
        try:
            make_contract(bad)
        except ClosureError:
            pass
        else:
            raise AssertionError("expected contract denial")

    print("ok verification_contract_id=" + contract["verification_contract_id"])
    print("ok requirement_satisfaction_receipt_id=" + complete["requirement_satisfaction_receipt_id"])

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    payload = json.load(sys.stdin)
    try:
        contract = make_contract(payload["contract"])
        result = evaluate_current_satisfaction(
            contract,
            current_requirement_revision_id=payload["current_requirement_revision_id"],
            current_discharge_facts=payload["current_discharge_facts"],
            currentness_assertion_id=payload["currentness_assertion_id"],
        )
        print(json.dumps({"contract": contract, "result": result}, sort_keys=True, separators=(",", ":")))
        return 0
    except (ClosureError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"decision": "Deny", "reason": str(error)}, sort_keys=True))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
