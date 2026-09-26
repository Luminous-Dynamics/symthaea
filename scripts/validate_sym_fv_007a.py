#!/usr/bin/env python3
"""Fail-closed static contract for SYM-FV-007A.

This validator is deliberately not a proof checker. Lean owns theorem truth.
It protects the reviewed theorem/claim surface and hostile mutation contract.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/formal/sym-fv-007a-evidence-composition-v1.json"
PROOF = ROOT / "formal/lean/evidence/EvidenceComposition.lean"

PARENT = "89f31822c97596c35ca078071775035b74b6c92b"
PARENT_SCHEMA_BLOB = "26b966da5a4ee16eb48023033e7324d6f14a3707"
PROOF_BLOB = "4684326db9d980647914bca180c88cb7fad4ec7e"

EXPECTED_CLASSES = [
    "abstractFormalTheorem",
    "extractedSourceRefinement",
    "deductiveImplementationProof",
    "boundedModelSafety",
    "temporalModelEvidence",
    "boundedTraceConformance",
    "runtimeQualification",
]

EXPECTED_THEOREMS = [
    "compose_one_preserves_primary_class",
    "compose_one_preserves_subject",
    "compose_one_preserves_scope",
    "compose_one_preserves_root_assumptions",
    "compose_one_preserves_dependency_assumptions",
    "compose_many_preserves_primary_class",
    "compose_many_preserves_subject",
    "compose_many_preserves_scope",
    "compose_many_preserves_root_assumptions",
    "compose_many_preserves_head_dependency_assumptions",
]

REQUIRED_SOURCE_ANCHORS = [
    "primary := root.primary",
    "subject := root.subject",
    "scope := root.scope",
    "assumptions := root.assumptions ++ dependency.assumptions",
]

FORBIDDEN_RANKING_MARKERS = [
    "EvidenceClass.toNat",
    "EvidenceClass.rank",
    "strongestEvidence",
    "strongerEvidence",
    "maxEvidenceClass",
    "minEvidenceClass",
    "Ord EvidenceClass",
    "LT EvidenceClass",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def fail(msg: str) -> None:
    raise SystemExit(f"SYM-FV-007A contract failure: {msg}")


def validate_source(text: str) -> None:
    if re.search(r"\b(sorry|admit)\b", text):
        fail("proof hole marker present")
    if re.search(r"(?m)^\s*(axiom|opaque)\b", text):
        fail("user axiom/opaque declaration present")
    if "Mathlib" in text:
        fail("unexpected Mathlib dependency")

    for marker in REQUIRED_SOURCE_ANCHORS:
        if marker not in text:
            fail(f"missing composition anchor: {marker}")

    for marker in FORBIDDEN_RANKING_MARKERS:
        if marker in text:
            fail(f"synthetic evidence ranking introduced: {marker}")

    for theorem in EXPECTED_THEOREMS:
        if not re.search(rf"(?m)^theorem\s+{re.escape(theorem)}\b", text):
            fail(f"missing theorem: {theorem}")
        if f"#print axioms {theorem}" not in text:
            fail(f"missing axiom probe: {theorem}")

    constructors = re.findall(
        r"(?m)^\s*\|\s+(abstractFormalTheorem|extractedSourceRefinement|"
        r"deductiveImplementationProof|boundedModelSafety|temporalModelEvidence|"
        r"boundedTraceConformance|runtimeQualification)\b",
        text,
    )
    if constructors != EXPECTED_CLASSES:
        fail(f"evidence-class census drifted: {constructors!r}")


def validate_manifest(data: dict) -> None:
    if data.get("schema") != "symthaea.formal.sym-fv-007a-evidence-composition.v1":
        fail("schema identity drift")
    if data.get("tracking_issue") != 5961 or data.get("parent_issue") != 5945:
        fail("issue lineage drift")
    if data.get("evidence_class") != "AbstractFormalTheorem":
        fail("primary evidence class drift")
    if data.get("authority") != "EvidenceOnly":
        fail("authority escalation")

    parent = data.get("parent_receipt_contract", {})
    if parent.get("commit") != PARENT or parent.get("schema_blob") != PARENT_SCHEMA_BLOB:
        fail("parent receipt binding drift")

    proof = data.get("proof_subject", {})
    if proof.get("path") != "formal/lean/evidence/EvidenceComposition.lean":
        fail("proof path drift")
    if proof.get("blob") != PROOF_BLOB:
        fail("proof blob drift")

    if data.get("evidence_classes") != EXPECTED_CLASSES:
        fail("manifest evidence-class census drift")
    if data.get("theorems") != EXPECTED_THEOREMS:
        fail("manifest theorem census drift")

    contract = data.get("composition_contract", {})
    expected_contract = {
        "primary_class": "root-preserved",
        "subject": "root-preserved",
        "scope": "root-preserved",
        "assumptions": "monotone-accumulation",
        "class_ordering": "none",
    }
    if contract != expected_contract:
        fail("composition contract drift")

    nonclaims = set(data.get("nonclaims", []))
    required_nonclaims = {
        "not-truth-of-underlying-dependency-claims",
        "not-json-or-python-validator-refinement",
        "not-freshness-or-invalidation-theorem",
        "not-dag-acyclicity-theorem",
        "not-total-ordering-of-evidence-quality",
        "not-runtime-authorization",
    }
    if not required_nonclaims.issubset(nonclaims):
        fail("required claim ceilings missing")


def expect_source_rejection(name: str, mutant: str) -> None:
    try:
        validate_source(mutant)
    except SystemExit:
        return
    fail(f"hostile mutant unexpectedly accepted: {name}")


def hostile_self_tests(source: str) -> None:
    mutants = {
        "dependency-class-replaces-root": source.replace(
            "primary := root.primary", "primary := dependency.primary", 1
        ),
        "dependency-subject-replaces-root": source.replace(
            "subject := root.subject", "subject := dependency.subject", 1
        ),
        "dependency-scope-replaces-root": source.replace(
            "scope := root.scope", "scope := dependency.scope", 1
        ),
        "drop-dependency-assumptions": source.replace(
            "assumptions := root.assumptions ++ dependency.assumptions",
            "assumptions := root.assumptions",
            1,
        ),
        "drop-root-assumptions": source.replace(
            "assumptions := root.assumptions ++ dependency.assumptions",
            "assumptions := dependency.assumptions",
            1,
        ),
        "synthetic-ranking": source + "\ndef EvidenceClass.rank : EvidenceClass → Nat := fun _ => 0\n",
        "proof-hole": source.replace("  rfl", "  sorry", 1),
    }
    for name, mutant in mutants.items():
        if mutant == source:
            fail(f"hostile mutant did not alter source: {name}")
        expect_source_rejection(name, mutant)


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = PROOF.read_text(encoding="utf-8")

    validate_manifest(data)
    validate_source(source)

    if git("rev-parse", f"HEAD:{PROOF.relative_to(ROOT)}") != PROOF_BLOB:
        fail("checked-out proof blob does not match frozen subject")
    if git("rev-parse", f"{PARENT}:docs/formal/formal_evidence_receipt_v1.schema.json") != PARENT_SCHEMA_BLOB:
        fail("parent receipt schema blob does not match frozen subject")

    hostile_self_tests(source)
    print("SYM-FV-007A static contract: PASS")


if __name__ == "__main__":
    main()
