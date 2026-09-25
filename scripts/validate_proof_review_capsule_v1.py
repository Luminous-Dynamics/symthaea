#!/usr/bin/env python3
"""Validate Proof Review Capsule V1 and its semantic negative controls."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/formal/proof_review_capsule_schema_v1.json"
EVIDENCE_PATH = ROOT / "docs/formal/formal_verification_evidence_classes_v1.json"

HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_FIELDS = {
    "id",
    "human_claim",
    "formal_statement",
    "evidence_class",
    "status",
    "subject",
    "assumptions",
    "axioms",
    "trust_roots",
    "dependencies",
    "coverage",
    "receipts",
    "negative_controls",
    "claim_ceiling",
    "review_delta",
}

REQUIRED_NONCLAIMS = {
    "proof capsule != proof",
    "proof DAG consistency != theorem truth",
    "compact review != reduced verification",
    "finite basis != universal coverage without a coverage theorem",
    "many finite tests != infinite-domain proof",
    "composition metadata != a stronger evidence class",
}


def die(message: str) -> None:
    raise ValueError(message)


def statement_digest(statement: str) -> str:
    return hashlib.sha256(statement.encode("utf-8")).hexdigest()


def capsule_digest(capsule: dict) -> str:
    candidate = copy.deepcopy(capsule)
    candidate.pop("derived_digest", None)
    encoded = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def expect_invalid(capsule: dict, evidence_ids: set[str], coverage_kinds: set[str], scopes: set[str]) -> None:
    try:
        validate_capsule(capsule, evidence_ids, coverage_kinds, scopes)
    except ValueError:
        return
    die("negative control unexpectedly validated")


def validate_coverage(coverage: dict, coverage_kinds: set[str], scopes: set[str]) -> None:
    for field in ("kind", "scope", "domain", "basis", "coverage_theorem", "preservation_theorems", "exceptions"):
        if field not in coverage:
            die(f"coverage missing {field}")

    kind = coverage["kind"]
    scope = coverage["scope"]
    if kind not in coverage_kinds:
        die(f"unknown coverage kind: {kind}")
    if scope not in scopes:
        die(f"unknown coverage scope: {scope}")
    if not isinstance(coverage["basis"], list) or not isinstance(coverage["preservation_theorems"], list):
        die("coverage basis/preservation must be arrays")

    if kind == "FiniteCases" and scope == "exact-universal":
        die("FiniteCases cannot establish exact-universal coverage")

    if kind in {"Universal", "PartitionCover", "QuotientInvariant", "Bisimulation"}:
        if not coverage["coverage_theorem"]:
            die(f"{kind} requires a coverage theorem")

    if kind == "Induction":
        if not coverage["basis"]:
            die("Induction requires a base/basis case")
        if not coverage["preservation_theorems"]:
            die("Induction requires a preservation/step theorem")
        if not coverage["coverage_theorem"]:
            die("Induction requires constructor/well-founded coverage")
        if scope != "exact-universal":
            die("v1 Induction fixture must state exact-universal scope")

    if kind == "PartitionCover":
        if not coverage["preservation_theorems"]:
            die("PartitionCover requires branch theorem(s)")

    if kind == "GeneratorClosure":
        if not coverage["basis"]:
            die("GeneratorClosure requires a basis/generator set")
        if not coverage["preservation_theorems"]:
            die("GeneratorClosure requires closure theorem(s)")
        if not coverage["coverage_theorem"]:
            die("GeneratorClosure requires a generation/coverage theorem")

    if kind == "QuotientInvariant":
        if not coverage["basis"] or not coverage["preservation_theorems"]:
            die("QuotientInvariant requires representatives and invariance theorem(s)")

    if kind == "Bisimulation":
        if not coverage["preservation_theorems"]:
            die("Bisimulation requires next-step preservation")

    if kind == "LimitTransfer":
        if not coverage.get("convergence_theorem"):
            die("LimitTransfer requires convergence theorem")
        if not coverage.get("limit_transfer_theorem"):
            die("LimitTransfer requires property-transfer theorem")
        if scope not in {"asymptotic", "exact-universal"}:
            die("LimitTransfer scope must be asymptotic or exact-universal")


def validate_capsule(capsule: dict, evidence_ids: set[str], coverage_kinds: set[str], scopes: set[str]) -> None:
    missing = REQUIRED_FIELDS - set(capsule)
    if missing:
        die(f"capsule {capsule.get('id', '<unknown>')} missing fields: {sorted(missing)}")

    if capsule["evidence_class"] not in evidence_ids:
        die(f"unknown evidence class: {capsule['evidence_class']}")
    if not capsule["human_claim"] or not capsule["formal_statement"]:
        die("claim and formal statement must be non-empty")

    subject = capsule["subject"]
    if not HEX64.fullmatch(subject.get("sha256", "")):
        die("subject sha256 must be 64 lowercase hex characters")

    for array_field in ("assumptions", "axioms", "trust_roots", "dependencies", "receipts", "negative_controls", "claim_ceiling"):
        if not isinstance(capsule[array_field], list):
            die(f"{array_field} must be an array")
    if not capsule["claim_ceiling"]:
        die("claim ceiling must be explicit")
    if not capsule["receipts"]:
        die("at least one receipt is required")

    validate_coverage(capsule["coverage"], coverage_kinds, scopes)

    expected_statement_digest = statement_digest(capsule["formal_statement"])
    for receipt in capsule["receipts"]:
        if receipt.get("statement_sha256") != expected_statement_digest:
            die("receipt is not bound to the current formal statement")
        if not receipt.get("identity") or not receipt.get("kind"):
            die("receipt requires identity and kind")

    delta = capsule["review_delta"]
    if "previous_capsule" not in delta or "semantic_changes" not in delta:
        die("review_delta must include previous_capsule and semantic_changes")
    if not isinstance(delta["semantic_changes"], list):
        die("semantic_changes must be an array")


def main() -> int:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    if contract.get("schema") != "symthaea.formal-verification.proof-review-capsule.v1":
        die("wrong proof review capsule schema")
    if contract.get("authority") != "ReviewAndCompositionMetadataOnly":
        die("capsule contract authority escalated")
    if set(contract.get("required_nonclaims", [])) != REQUIRED_NONCLAIMS:
        die("required nonclaims changed")

    coverage_kinds = set(contract.get("coverage_kinds", []))
    scopes = set(contract.get("coverage_scopes", []))
    evidence_ids = {entry["id"] for entry in evidence.get("classes", [])}

    required_kinds = {
        "FiniteCases", "Universal", "Induction", "PartitionCover",
        "GeneratorClosure", "QuotientInvariant", "Bisimulation", "LimitTransfer"
    }
    if coverage_kinds != required_kinds:
        die("coverage kind census changed")

    required_relations = {
        "DependsOn", "Implies", "Conjunction", "Refines", "PartitionCover",
        "Induction", "GeneratorClosure", "QuotientInvariant", "Bisimulation", "LimitTransfer"
    }
    if set(contract.get("composition_relations", [])) != required_relations:
        die("composition relation census changed")

    capsules = contract.get("synthetic_capsules", [])
    if len(capsules) != 2:
        die("expected exactly two v1 synthetic capsules")
    for capsule in capsules:
        validate_capsule(capsule, evidence_ids, coverage_kinds, scopes)
        digest = capsule_digest(capsule)
        print(f"CAPSULE {capsule['id']} sha256={digest}")

    finite = next(c for c in capsules if c["id"] == "SYN-FINITE-001")
    infinite = next(c for c in capsules if c["id"] == "SYN-INFINITE-001")

    # Semantic mutation: stale receipt after theorem change must fail.
    stale_statement = copy.deepcopy(finite)
    stale_statement["formal_statement"] += " and True"
    expect_invalid(stale_statement, evidence_ids, coverage_kinds, scopes)

    # Infinite promotion from finite enumeration must fail.
    false_universal = copy.deepcopy(finite)
    false_universal["coverage"]["scope"] = "exact-universal"
    expect_invalid(false_universal, evidence_ids, coverage_kinds, scopes)

    # Induction without its base case must fail.
    no_base = copy.deepcopy(infinite)
    no_base["coverage"]["basis"] = []
    expect_invalid(no_base, evidence_ids, coverage_kinds, scopes)

    # Partition branches without coverage theorem must fail.
    no_partition_cover = copy.deepcopy(infinite)
    no_partition_cover["coverage"].update({
        "kind": "PartitionCover",
        "basis": ["D_i"],
        "preservation_theorems": ["forall i x, x in D_i -> P x"],
        "coverage_theorem": "",
    })
    expect_invalid(no_partition_cover, evidence_ids, coverage_kinds, scopes)

    # Limit reasoning needs both convergence and transfer.
    bad_limit = copy.deepcopy(infinite)
    bad_limit["coverage"].update({
        "kind": "LimitTransfer",
        "scope": "asymptotic",
        "basis": ["finite approximants"],
        "preservation_theorems": ["P holds on each approximant"],
        "coverage_theorem": "approximation family",
        "convergence_theorem": "approximants converge",
        "limit_transfer_theorem": "",
    })
    expect_invalid(bad_limit, evidence_ids, coverage_kinds, scopes)

    # Delta frontier: changing assumptions, dependencies, or claim ceiling must
    # change capsule identity even if the theorem text is untouched.
    for label, mutate in (
        ("assumption", lambda c: c["assumptions"].append("new assumption")),
        ("dependency", lambda c: c["dependencies"].append({"id": "CHILD", "capsule_sha256": "3" * 64, "relation": "DependsOn"})),
        ("claim-ceiling", lambda c: c["claim_ceiling"].append("stronger claim")),
    ):
        mutated = copy.deepcopy(infinite)
        before = capsule_digest(mutated)
        mutate(mutated)
        after = capsule_digest(mutated)
        if before == after:
            die(f"{label} semantic mutation did not change capsule identity")

    print("PROOF_REVIEW_CAPSULE_V1_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"PROOF_REVIEW_CAPSULE_V1_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
