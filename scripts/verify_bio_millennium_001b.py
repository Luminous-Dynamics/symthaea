#!/usr/bin/env python3
"""Verify BIO-MILLENNIUM-001B claim/evidence separation.

This verifier checks only local contract integrity. It does not interpret the
living Millennium Biology challenge criteria and cannot establish a scientific
result.
"""
from __future__ import annotations
import json
from pathlib import Path

MANIFEST = Path("docs/research/biology/millennium_problems_source_manifest_v1.json")
CONTRACT = Path("docs/research/biology/millennium_problems_claim_contract_v1.json")

REQUIRED_CLASSES = {
    "HypothesisProposal",
    "InSilicoPrediction",
    "SimulationResult",
    "RetrospectiveBenchmarkResult",
    "ProspectiveCommittedPrediction",
    "ExternalExperimentalObservation",
    "IndependentReplication",
    "OfficialChallengeCriterionEvidence",
}
FORBIDDEN_UPGRADES = {
    ("HypothesisProposal", "physical observation"),
    ("InSilicoPrediction", "physical observation"),
    ("SimulationResult", "physical observation"),
    ("RetrospectiveBenchmarkResult", "prospective prediction credit"),
    ("RetrospectiveBenchmarkResult", "physical observation"),
    ("ProspectiveCommittedPrediction", "physical observation"),
    ("ExternalExperimentalObservation", "independent replication"),
}

def fail(message: str) -> None:
    raise SystemExit(f"BIO-MILLENNIUM-001B FAIL: {message}")

def main() -> int:
    if not MANIFEST.is_file() or not CONTRACT.is_file():
        fail("required biology contract/manifest is missing")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    if manifest.get("schema") != "symthaea.bio-millennium.challenge-source-manifest.v1":
        fail("unexpected source manifest schema")
    if manifest.get("manifest_generation") != 1:
        fail("source manifest generation is not 1")
    if contract.get("schema") != "symthaea.bio-millennium.claim-contract.v1":
        fail("unexpected claim contract schema")
    if contract.get("contract_generation") != 1:
        fail("claim contract generation is not 1")

    parent = contract.get("parent_source_manifest", {})
    if parent.get("schema") != manifest["schema"]:
        fail("claim contract parent schema does not match source manifest")
    if parent.get("manifest_generation") != manifest["manifest_generation"]:
        fail("claim contract parent generation does not match source manifest")

    classes = {entry.get("id"): entry for entry in contract.get("claim_classes", [])}
    if set(classes) != REQUIRED_CLASSES:
        fail(f"claim class set mismatch: {sorted(classes)}")

    for entry in classes.values():
        for target in entry.get("cannot_satisfy", []):
            if (entry["id"], target) in FORBIDDEN_UPGRADES:
                continue
            if target == "official challenge criterion" and entry["id"] != "OfficialChallengeCriterionEvidence":
                continue
        if entry["id"] != "OfficialChallengeCriterionEvidence":
            if "official challenge criterion" not in entry.get("cannot_satisfy", []):
                fail(f"{entry['id']} lacks explicit official-criterion ceiling")

    roles = {entry.get("id"): entry for entry in contract.get("claim_roles", [])}
    required_roles = {
        "proposal", "model_prediction", "retrospective_result",
        "prospective_result", "experimental_observation", "official_criterion"
    }
    if set(roles) != required_roles:
        fail("claim-role set mismatch")

    if roles["official_criterion"]["accepted_classes"] != ["OfficialChallengeCriterionEvidence"]:
        fail("official criterion role must be isolated to source-bound official evidence")

    challenge_ids = [x["id"] for x in manifest["challenges"]]
    profile_ids = [x["id"] for x in contract["challenge_profiles"]]
    if profile_ids != challenge_ids:
        fail("challenge profile IDs/order diverge from frozen source manifest")

    if contract["authority"]["official_criterion_semantics_frozen"]:
        fail("001B must not freeze living official criterion semantics")
    if contract["authority"]["experimental_authority_granted"]:
        fail("001B cannot grant experimental authority")

    invariants = set(contract.get("global_invariants", []))
    required_invariants = {
        "proposal != prediction",
        "prediction != simulation result",
        "simulation result != physical observation",
        "retrospective result != prospective result",
        "model agreement != independent evidence",
        "formal proof of a model consequence != proof that the biological model is true",
    }
    missing = required_invariants - invariants
    if missing:
        fail(f"missing epistemic invariant(s): {sorted(missing)}")

    unresolved = contract.get("negative_space", {}).get("unresolved_until_source_bound", [])
    if len(unresolved) < 5:
        fail("negative-space list is too weak; official semantics must remain explicitly unresolved")

    print("BIO-MILLENNIUM-001B PASS: claim/evidence separation contract is internally consistent")
    print("official_criterion_semantics_frozen=false")
    print("experimental_authority_granted=false")
    print(f"challenge_profiles={len(profile_ids)}")
    print(f"claim_classes={len(classes)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
