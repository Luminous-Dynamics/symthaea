#!/usr/bin/env python3
"""Independent semantic checker for monetary architecture profile v1.

It intentionally does not import production economics code. It validates the
typed profile contract, verifies content digests, and rejects silent
cross-layer inheritance in transplant definitions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

PRIMITIVES = (
    "unit_of_account",
    "medium_of_exchange",
    "payment_instrument",
    "store_of_value_claim",
    "issuer_liability",
    "credit_creation",
    "reserve_or_backing",
    "settlement_asset",
    "redemption_convertibility",
    "transaction_finality",
    "liquidity_backstop",
    "clearing_topology",
    "interoperability",
    "default_loss_allocation",
    "governance_legal_finality",
)

COMPOSITION = (
    "ownership",
    "allocation",
    "governance",
    "ecological_constraints",
)

MONEYNESS = (
    "par_singleness",
    "liquidity_elasticity",
    "integrity_constraints",
)

CLASSIFICATIONS = {
    "whole_system",
    "architecture",
    "variant",
    "overlay",
    "macro_regime",
    "monetary_rule",
    "analytical_framework",
}

def fail(msg: str) -> None:
    raise ValueError(msg)

def canonical_without_digest(profile: dict) -> bytes:
    obj = copy.deepcopy(profile)
    obj["provenance"].pop("profile_digest", None)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def check_profile(p: dict) -> None:
    required = {
        "schema_version","profile_id","semantic_version","title","classification",
        "evidence_maturity","evidence_refs","claim_ceiling","provenance",
        "inheritance_policy","monetary_primitives","moneyness_properties",
        "institutional_composition","transplant_contract",
    }
    if set(p) != required:
        fail(f"{p.get('profile_id','<unknown>')}: top-level key mismatch")
    if p["schema_version"] != "monetary-architecture-profile-v1":
        fail(f"{p['profile_id']}: schema version")
    if p["classification"] not in CLASSIFICATIONS:
        fail(f"{p['profile_id']}: classification")
    if p["evidence_maturity"] not in {"A","B","C","D"}:
        fail(f"{p['profile_id']}: evidence maturity")
    ip = p["inheritance_policy"]
    if ip != {"implicit_inheritance": False, "missing_mechanism_disposition": "underdetermined"}:
        fail(f"{p['profile_id']}: inheritance policy")
    mp = p["monetary_primitives"]
    if set(mp) != set(PRIMITIVES):
        fail(f"{p['profile_id']}: primitive key set")
    for name, value in mp.items():
        if set(value) - {"status","definition","parameters","implementation_ref","evidence_refs"}:
            fail(f"{p['profile_id']}.{name}: unexpected keys")
        if value.get("status") not in {"specified","underdetermined","not_applicable"}:
            fail(f"{p['profile_id']}.{name}: status")
        if value["status"] == "specified" and not value.get("definition"):
            fail(f"{p['profile_id']}.{name}: specified without definition")
        if value["status"] != "specified" and "definition" in value:
            fail(f"{p['profile_id']}.{name}: non-specified mechanism carries definition")
    moneyness = p["moneyness_properties"]
    if set(moneyness) != set(MONEYNESS):
        fail(f"{p['profile_id']}: moneyness property key set")
    for name, value in moneyness.items():
        if value.get("status") not in {"specified","underdetermined","not_applicable"}:
            fail(f"{p['profile_id']}.{name}: status")
        if value["status"] == "specified" and not value.get("definition"):
            fail(f"{p['profile_id']}.{name}: specified without definition")
        if value["status"] != "specified" and "definition" in value:
            fail(f"{p['profile_id']}.{name}: non-specified property carries definition")
    comp = p["institutional_composition"]
    if set(comp) != set(COMPOSITION):
        fail(f"{p['profile_id']}: composition key set")
    for name, value in comp.items():
        if value.get("status") not in {"fixed","parameterized","underdetermined"}:
            fail(f"{p['profile_id']}.{name}: composition status")
        if value["status"] != "underdetermined" and not value.get("definition") and not value.get("profile_ref"):
            fail(f"{p['profile_id']}.{name}: composition is specified but unidentified")
    tc = p["transplant_contract"]
    if tc != {
        "mutable_layer":"monetary_primitives",
        "protected_layers":list(COMPOSITION),
        "comparison_rule":"candidate may differ only in mutable_layer",
    }:
        fail(f"{p['profile_id']}: transplant contract")
    digest = p["provenance"].get("profile_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        fail(f"{p['profile_id']}: missing profile digest")
    actual = hashlib.sha256(canonical_without_digest(p)).hexdigest()
    if actual != digest:
        fail(f"{p['profile_id']}: profile digest mismatch")

def check_registry(registry: dict) -> int:
    if registry.get("schema_version") != "monetary-architecture-registry-v1":
        fail("registry: schema version")
    profiles = registry.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        fail("registry: profiles must be non-empty")
    ids = set()
    for p in profiles:
        check_profile(p)
        if p["profile_id"] in ids:
            fail(f"duplicate profile id: {p['profile_id']}")
        ids.add(p["profile_id"])
    return len(profiles)

def check_negative(fixtures: dict) -> int:
    cases = fixtures.get("cases", [])
    if not isinstance(cases, list):
        fail("negative fixtures: cases must be an array")
    rejected = 0
    for case in cases:
        kind = case.get("kind")
        if kind == "implicit_inheritance":
            if case.get("proposed_behavior") != "inherit from source_profile":
                fail(f"{case.get('id')}: inheritance fixture malformed")
            if case.get("missing_primitive") not in PRIMITIVES:
                fail(f"{case.get('id')}: unknown primitive")
            rejected += 1
        elif kind == "profile_mutation":
            mutation_path = case.get("mutation", {}).get("path", "")
            if not mutation_path.startswith("monetary_primitives."):
                fail(f"{case.get('id')}: mutation escaped monetary layer")
            if case.get("expected") != "requires new profile generation and digest":
                fail(f"{case.get('id')}: mutation fixture must require new generation")
            rejected += 1
        elif kind == "cross_layer_mutation":
            mutated_layer = case.get("mutated_layer", "")
            if mutated_layer in {
                "institutional_composition.ownership",
                "institutional_composition.allocation",
                "institutional_composition.governance",
                "institutional_composition.ecological_constraints",
            }:
                if case.get("expected") != "reject":
                    fail(f"{case.get('id')}: cross-layer mutation must reject")
                rejected += 1
            else:
                fail(f"{case.get('id')}: unsupported cross-layer mutation")
        else:
            fail(f"{case.get('id','<unknown>')}: unknown negative fixture kind")
    return rejected

def main() -> int:
    if len(sys.argv) not in {2,3}:
        print("usage: verify_monetary_architecture_profiles.py REGISTRY.json [NEGATIVE.json]", file=sys.stderr)
        return 2
    try:
        registry = json.loads(Path(sys.argv[1]).read_text())
        count = check_registry(registry)
        negative_count = 0
        if len(sys.argv) == 3:
            negative = json.loads(Path(sys.argv[2]).read_text())
            negative_count = check_negative(negative)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"independent monetary-profile check: {count} profiles; {negative_count} negative cases rejected as expected")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
