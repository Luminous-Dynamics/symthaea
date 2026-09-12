#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ETK requirement -> proof-obligation relationship oracle.

Core theorem:
    requirement + obligation != established relationship

The oracle freezes two explicit relation forms:
- exact_restatement: statement must equal claim exactly and evidence kinds match;
- derived_safety_obligation: evidence kinds match and explicit derivation-policy,
  derivation-record, and relationship-acceptance identities are bound.

A successful binding is not discharge evidence and does not authenticate any
referenced derivation/acceptance record.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from typing import Any

REQ_DOMAIN = b"symthaea.etk-accepted-requirement.v1\x00"
OBLIGATION_DOMAIN = b"symthaea.etk-proof-obligation-snapshot.v1\x00"
BINDING_DOMAIN = b"symthaea.etk-requirement-obligation-binding.v1\x00"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
EVIDENCE_KINDS = {"FormalProof", "Simulation", "Test", "Telemetry", "Standard"}
DOMAINS = {
    "Civil", "Mechanical", "Electrical", "Aerospace", "ChemicalProcess",
    "Robotics", "Nuclear", "Materials", "Environmental", "Systems",
}
CRITICALITIES = {"Low", "Medium", "High", "Blocking"}


class Denied(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def domain_hash(domain: bytes, value: Any) -> str:
    h = hashlib.sha256()
    h.update(domain)
    h.update(canonical_json(value).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise Denied(f"invalid_text:{field}")
    return value


def digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Denied(f"invalid_digest:{field}")
    return value


def requirement_revision(payload: Any) -> str:
    required = {
        "logical_requirement_id", "domain", "statement", "criticality",
        "expected_evidence_kind", "structural_invariants", "acceptance_record_digest",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise Denied("requirement_fields")
    if payload["domain"] not in DOMAINS:
        raise Denied("invalid_domain")
    if payload["criticality"] not in CRITICALITIES:
        raise Denied("invalid_criticality")
    if payload["expected_evidence_kind"] not in EVIDENCE_KINDS:
        raise Denied("invalid_evidence_kind")
    invariants = payload["structural_invariants"]
    if not isinstance(invariants, list):
        raise Denied("malformed_invariants")
    normalized = [text(v, "invariant") for v in invariants]
    if len(set(normalized)) != len(normalized):
        raise Denied("duplicate_invariant")
    normalized.sort()
    return domain_hash(REQ_DOMAIN, {
        "acceptance_record_digest": digest(payload["acceptance_record_digest"], "acceptance"),
        "criticality": payload["criticality"],
        "domain": payload["domain"],
        "expected_evidence_kind": payload["expected_evidence_kind"],
        "logical_requirement_id": text(payload["logical_requirement_id"], "requirement_id"),
        "schema": "symthaea.etk-accepted-requirement.v1",
        "statement": text(payload["statement"], "statement"),
        "structural_invariants": normalized,
    })


def obligation_revision(payload: Any) -> str:
    if not isinstance(payload, dict) or set(payload) != {
        "obligation_id", "claim", "expected_evidence_kind"
    }:
        raise Denied("obligation_fields")
    if payload["expected_evidence_kind"] not in EVIDENCE_KINDS:
        raise Denied("invalid_obligation_evidence_kind")
    return domain_hash(OBLIGATION_DOMAIN, {
        "claim": text(payload["claim"], "obligation_claim"),
        "expected_evidence_kind": payload["expected_evidence_kind"],
        "obligation_id": text(payload["obligation_id"], "obligation_id"),
        "schema": "symthaea.etk-proof-obligation-snapshot.v1",
    })


def bind(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"requirement", "obligation", "relation"}:
        raise Denied("binding_fields")
    requirement = payload["requirement"]
    obligation = payload["obligation"]
    relation = payload["relation"]
    requirement_id = requirement_revision(requirement)
    obligation_id = obligation_revision(obligation)

    if requirement["expected_evidence_kind"] != obligation["expected_evidence_kind"]:
        raise Denied("evidence_kind_mismatch")
    if not isinstance(relation, dict) or "kind" not in relation:
        raise Denied("malformed_relation")

    kind = relation["kind"]
    if kind == "exact_restatement":
        if set(relation) != {"kind"}:
            raise Denied("exact_relation_fields")
        if requirement["statement"] != obligation["claim"]:
            raise Denied("exact_restatement_claim_mismatch")
        normalized_relation = {"kind": "exact_restatement"}
    elif kind == "derived_safety_obligation":
        if set(relation) != {
            "kind", "derivation_record_digest", "derivation_policy_revision_digest",
            "binding_acceptance_record_digest",
        }:
            raise Denied("derived_relation_fields")
        normalized_relation = {
            "binding_acceptance_record_digest": digest(
                relation["binding_acceptance_record_digest"], "binding_acceptance"
            ),
            "derivation_policy_revision_digest": digest(
                relation["derivation_policy_revision_digest"], "derivation_policy"
            ),
            "derivation_record_digest": digest(
                relation["derivation_record_digest"], "derivation_record"
            ),
            "kind": "derived_safety_obligation",
        }
    else:
        raise Denied("unsupported_relation_kind")

    preimage = {
        "obligation_id": obligation["obligation_id"],
        "obligation_revision_id": obligation_id,
        "relation": normalized_relation,
        "requirement_revision_id": requirement_id,
        "schema": "symthaea.etk-requirement-obligation-binding.v1",
    }
    return {
        "requirement_revision_id": requirement_id,
        "obligation_revision_id": obligation_id,
        "requirement_obligation_binding_id": domain_hash(BINDING_DOMAIN, preimage),
    }


def d(ch: str) -> str:
    return "sha256:" + ch * 64


def derived_fixture() -> dict[str, Any]:
    return {
        "requirement": {
            "logical_requirement_id": "REQ-STRESS",
            "domain": "Civil",
            "statement": "stress remains below allowable",
            "criticality": "Blocking",
            "expected_evidence_kind": "Simulation",
            "structural_invariants": ["stress <= 250 MPa"],
            "acceptance_record_digest": d("a"),
        },
        "obligation": {
            "obligation_id": "00000000-0000-4000-8000-000000000042",
            "claim": "stress remains below allowable under service load",
            "expected_evidence_kind": "Simulation",
        },
        "relation": {
            "kind": "derived_safety_obligation",
            "derivation_record_digest": d("7"),
            "derivation_policy_revision_digest": d("8"),
            "binding_acceptance_record_digest": d("9"),
        },
    }


def expect_denied(payload: dict[str, Any], reason: str) -> None:
    try:
        bind(payload)
    except Denied as error:
        if reason not in str(error):
            raise AssertionError((reason, str(error))) from error
        return
    raise AssertionError(f"expected denial: {reason}")


def self_test() -> dict[str, str]:
    payload = derived_fixture()
    ids = bind(payload)
    assert ids["requirement_revision_id"] == "sha256:e340c5030eebc978c41443ffd64f340dc5febad31e376080340bacfceda60faa"
    assert ids["obligation_revision_id"] == "sha256:743d2c13cfcc52bdfb4cfd9a4a836ed0806ace7b2a68b4c7284b1910d8863c29"
    assert ids["requirement_obligation_binding_id"] == "sha256:1cc54ee11ff9b7e99279d4c9a9d43f751809f7f30a43469cff8e18fe70eee408"

    exact_mismatch = copy.deepcopy(payload)
    exact_mismatch["relation"] = {"kind": "exact_restatement"}
    expect_denied(exact_mismatch, "exact_restatement_claim_mismatch")

    exact = copy.deepcopy(exact_mismatch)
    exact["requirement"]["statement"] = exact["obligation"]["claim"]
    exact_ids = bind(exact)
    assert exact_ids["requirement_obligation_binding_id"] != ids["requirement_obligation_binding_id"]

    evidence_mismatch = copy.deepcopy(payload)
    evidence_mismatch["obligation"]["expected_evidence_kind"] = "FormalProof"
    expect_denied(evidence_mismatch, "evidence_kind_mismatch")

    changed_derivation = copy.deepcopy(payload)
    changed_derivation["relation"]["derivation_record_digest"] = d("6")
    assert bind(changed_derivation)["requirement_obligation_binding_id"] != ids["requirement_obligation_binding_id"]

    changed_requirement = copy.deepcopy(payload)
    changed_requirement["requirement"]["statement"] = "stress remains below revised allowable"
    assert bind(changed_requirement)["requirement_obligation_binding_id"] != ids["requirement_obligation_binding_id"]

    shadow = copy.deepcopy(payload)
    shadow["relation"]["confidence"] = 1.0
    expect_denied(shadow, "derived_relation_fields")

    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.fixture:
        print(json.dumps(derived_fixture(), indent=2, sort_keys=True))
        return 0
    if args.self_test:
        ids = self_test()
        for key in sorted(ids):
            print(f"{key}={ids[key]}")
        return 0
    payload = json.load(sys.stdin)
    try:
        print(json.dumps({"decision": "Bound", "ids": bind(payload)}, sort_keys=True))
        return 0
    except Denied as error:
        print(json.dumps({"decision": "Deny", "reason": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
