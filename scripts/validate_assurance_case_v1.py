#!/usr/bin/env python3
"""Validate Assurance Case V1 and hostile review/composition mutations."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/formal/assurance_case_v1.json"
CHALLENGE_PATH = ROOT / "docs/formal/proof_challenge_solution_v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_NONCLAIMS = {
    "assurance case != theorem",
    "more evidence != automatic stronger claim",
    "risk tier != confidence score",
    "independent review != mathematical proof",
    "formal theorem != lifecycle assurance",
}

REQUIRED_FIELDS = {
    "id", "human_claim", "claim_type", "subject_sha256", "evidence",
    "trust_inventory", "gaps", "rebuttals", "claim_ceiling",
    "operational_prerequisites", "risk_tier", "status", "previous",
    "semantic_changes", "human_review_required", "independent_review_required",
}

ALLOWED_SOURCE_CLASSES = {
    "FormalCapsule": {
        "AbstractFormalTheorem", "ExtractedSourceRefinement",
        "DeductiveImplementationProof", "BoundedModelSafety",
        "TemporalModelEvidence", "BoundedTraceConformance",
    },
    "RuntimeQualification": {"RuntimeQualification"},
    "DifferentialTest": {"DifferentialConformance", "RuntimeQualification"},
    "Provenance": {"ProvenanceMetadata"},
    "VulnerabilityAnalysis": {"VulnerabilityAnalysis"},
    "OperationalControl": {"OperationalControl"},
    "IndependentReview": {"IndependentReview"},
    "ChallengeSolution": {"TrustTopologyMetadataOnly"},
}


def die(message: str) -> None:
    raise ValueError(message)


def node_digest(node: dict) -> str:
    encoded = json.dumps(node, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_case(node: dict, risk_tiers: set[str], relations: set[str], kinds: set[str], statuses: set[str]) -> None:
    missing = REQUIRED_FIELDS - set(node)
    if missing:
        die(f"assurance node missing fields: {sorted(missing)}")
    if not node["human_claim"] or not node["claim_type"]:
        die("assurance claim/type must be non-empty")
    if not HEX64.fullmatch(node["subject_sha256"]):
        die("assurance subject digest invalid")
    if node["risk_tier"] not in risk_tiers:
        die("unknown risk tier")
    if node["status"] not in statuses:
        die("unknown assurance status")

    for field in ("evidence", "trust_inventory", "gaps", "rebuttals", "claim_ceiling", "operational_prerequisites", "semantic_changes"):
        if not isinstance(node[field], list):
            die(f"{field} must be an array")
    if not node["evidence"]:
        die("assurance node requires supporting evidence")
    if not node["claim_ceiling"]:
        die("assurance claim ceiling must be explicit")

    blocking_child = False
    has_challenge_solution = False
    has_independent_review = False
    for evidence in node["evidence"]:
        for field in ("id", "kind", "source_class", "relation", "state"):
            if not evidence.get(field):
                die(f"evidence reference missing {field}")
        kind = evidence["kind"]
        if kind not in kinds:
            die("unknown evidence kind")
        if evidence["relation"] not in relations:
            die("unknown assurance relation")
        if evidence["source_class"] not in ALLOWED_SOURCE_CLASSES.get(kind, set()):
            die(f"evidence source class cannot be represented as {kind}")
        if evidence["state"] not in {"qualified", "review-required", "blocked", "superseded"}:
            die("unknown evidence state")
        if evidence["relation"] in {"Supports", "DependsOn", "Refines"} and evidence["state"] in {"blocked", "superseded"}:
            blocking_child = True
        if kind == "ChallengeSolution" and evidence["state"] == "qualified":
            has_challenge_solution = True
        if kind == "IndependentReview" and evidence["state"] == "qualified":
            has_independent_review = True

    if blocking_child and node["status"] not in {"review-required", "blocked", "superseded"}:
        die("required child blocked/superseded while assurance claim remains current")

    trust_identities = []
    for entry in node["trust_inventory"]:
        if not entry.get("category") or not entry.get("identity"):
            die("trust inventory entry incomplete")
        if not isinstance(entry.get("qualified"), bool):
            die("trust qualification flag must be boolean")
        trust_identities.append(entry["identity"])

    open_gap = any(gap.get("state") == "open" for gap in node["gaps"])
    open_rebuttal = any(rebuttal.get("state") == "open" for rebuttal in node["rebuttals"])
    if open_gap and node["status"] == "unconditional-current":
        die("unresolved gap hidden by unconditional status")
    if open_rebuttal and node["status"] == "unconditional-current":
        die("open rebuttal hidden by unconditional status")

    if not isinstance(node["human_review_required"], bool) or not isinstance(node["independent_review_required"], bool):
        die("review-required flags must be boolean")
    if node["semantic_changes"]:
        if node["risk_tier"] == "R0":
            die("R0 cannot carry semantic changes")
        if not node["human_review_required"]:
            die("semantic changes require human review")

    previous = node["previous"]
    for field in ("node_sha256", "trust_identities", "risk_tier"):
        if field not in previous:
            die(f"previous missing {field}")
    if previous["node_sha256"] is not None and not HEX64.fullmatch(previous["node_sha256"]):
        die("previous node digest invalid")
    if not isinstance(previous["trust_identities"], list):
        die("previous trust identities must be array")
    if previous["risk_tier"] is not None and previous["risk_tier"] not in risk_tiers:
        die("previous risk tier invalid")

    if previous["node_sha256"] is not None:
        added_trust = set(trust_identities) - set(previous["trust_identities"])
        if added_trust and node["risk_tier"] not in {"R3", "R4"}:
            die("new trust root requires R3/R4 escalation")
        if added_trust and not node["independent_review_required"]:
            die("new trust root requires independent review")

    if node["risk_tier"] == "R4":
        if not has_challenge_solution:
            die("R4 requires qualified challenge/solution evidence")
        if not has_independent_review:
            die("R4 requires qualified independent review evidence")
        if not node["human_review_required"] or not node["independent_review_required"]:
            die("R4 requires human and independent review")


def expect_invalid(node: dict, risk_tiers: set[str], relations: set[str], kinds: set[str], statuses: set[str]) -> None:
    try:
        validate_case(node, risk_tiers, relations, kinds, statuses)
    except ValueError:
        return
    die("negative control unexpectedly validated")


def main() -> int:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    challenge = json.loads(CHALLENGE_PATH.read_text(encoding="utf-8"))

    if contract.get("schema") != "symthaea.formal-verification.assurance-case.v1":
        die("wrong assurance-case schema")
    if contract.get("authority") != "AssuranceCompositionMetadataOnly":
        die("assurance-case authority escalated")
    if challenge.get("schema") != "symthaea.formal-verification.proof-challenge-solution.v1":
        die("parent challenge/solution contract missing or wrong")
    if contract.get("parent_contract") != challenge.get("schema"):
        die("assurance parent contract drifted")
    if set(contract.get("required_nonclaims", [])) != REQUIRED_NONCLAIMS:
        die("required nonclaims changed")

    risk_tiers = set(contract.get("risk_tiers", []))
    relations = set(contract.get("relations", []))
    kinds = set(contract.get("evidence_kinds", []))
    statuses = set(contract.get("statuses", []))
    if risk_tiers != {"R0", "R1", "R2", "R3", "R4"}:
        die("risk tier census changed")
    if relations != {"Supports", "DependsOn", "Refines", "Rebuts", "Gap", "Supersedes"}:
        die("relation census changed")
    if kinds != set(ALLOWED_SOURCE_CLASSES):
        die("evidence-kind census changed")
    if statuses != {"unconditional-current", "qualified-with-gaps", "review-required", "blocked", "superseded"}:
        die("status census changed")

    node = contract["synthetic_case"]
    validate_case(node, risk_tiers, relations, kinds, statuses)

    # Required child cannot become superseded while parent stays current.
    stale_child = copy.deepcopy(node)
    stale_child["evidence"][0]["state"] = "superseded"
    stale_child["status"] = "qualified-with-gaps"
    expect_invalid(stale_child, risk_tiers, relations, kinds, statuses)

    # A new trust root requires trust-boundary escalation and independent review.
    new_trust = copy.deepcopy(node)
    new_trust["previous"] = {
        "node_sha256": "6" * 64,
        "trust_identities": ["Lean4 kernel"],
        "risk_tier": "R2",
    }
    new_trust["risk_tier"] = "R2"
    new_trust["independent_review_required"] = False
    expect_invalid(new_trust, risk_tiers, relations, kinds, statuses)

    # Open rebuttal prevents an unconditional claim.
    rebutted = copy.deepcopy(node)
    rebutted["gaps"] = []
    rebutted["rebuttals"] = [{"id": "REB-1", "description": "counterexample", "state": "open"}]
    rebutted["status"] = "unconditional-current"
    expect_invalid(rebutted, risk_tiers, relations, kinds, statuses)

    # Runtime evidence cannot be relabeled as formal theorem evidence.
    relabeled = copy.deepcopy(node)
    relabeled["evidence"][2]["kind"] = "FormalCapsule"
    expect_invalid(relabeled, risk_tiers, relations, kinds, statuses)

    # R0 cannot hide semantic changes.
    fake_r0 = copy.deepcopy(node)
    fake_r0["risk_tier"] = "R0"
    expect_invalid(fake_r0, risk_tiers, relations, kinds, statuses)

    # R4 requires challenge/solution + independent review evidence and review flags.
    weak_r4 = copy.deepcopy(node)
    weak_r4["risk_tier"] = "R4"
    weak_r4["independent_review_required"] = True
    expect_invalid(weak_r4, risk_tiers, relations, kinds, statuses)

    # An unresolved gap cannot be hidden by unconditional status.
    hidden_gap = copy.deepcopy(node)
    hidden_gap["status"] = "unconditional-current"
    expect_invalid(hidden_gap, risk_tiers, relations, kinds, statuses)

    # Review-critical argument changes alter node identity.
    for label, mutate in (
        ("evidence", lambda n: n["evidence"].append({"id": "NEW", "kind": "Provenance", "source_class": "ProvenanceMetadata", "relation": "Supports", "state": "qualified"})),
        ("trust", lambda n: n["trust_inventory"].append({"category": "solver", "identity": "new-solver", "qualified": False})),
        ("gap", lambda n: n["gaps"].append({"id": "GAP-2", "description": "new gap", "state": "open"})),
        ("risk", lambda n: n.update({"risk_tier": "R3"})),
        ("ceiling", lambda n: n["claim_ceiling"].append("new ceiling")),
    ):
        candidate = copy.deepcopy(node)
        before = node_digest(candidate)
        mutate(candidate)
        after = node_digest(candidate)
        if before == after:
            die(f"{label} mutation did not alter assurance-node identity")

    print(f"ASSURANCE_CASE sha256={node_digest(node)}")
    print("ASSURANCE_CASE_V1_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"ASSURANCE_CASE_V1_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
