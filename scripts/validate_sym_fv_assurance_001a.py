#!/usr/bin/env python3
"""Static admission gate for SYM-FV-ASSURANCE-001A.

Lean itself remains the proof authority. This script binds the intended theorem
surface, canonical disposition mapping, repaired parent lineage, and hostile
source mutations so the workflow fails closed before invoking the prover.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/formal/sym-fv-assurance-001a-closure-calculus-v1.json"
PROOF_PATH = ROOT / "formal/lean/assurance/Closure.lean"

EXPECTED_SCHEMA = "symthaea.formal.sym-fv-assurance-001a-closure-calculus.v1"
EXPECTED_ISSUE = 6073
PARENT_ISSUE = 5801
PARENT_HEAD = "a4d4506c57902f96b6c260e7e9c56f9fc2e654c3"
PARENT_PROOF_PATH = "formal/lean/hdc/BinaryHVBind.lean"
PARENT_PROOF_BLOB = "259ba64888d8492bafc123b72d67b70da9282c36"

REQUIRED_TYPES = (
    "EvidenceClass",
    "PropertyKind",
    "EvidenceValidity",
    "CanonicalResult",
    "OutcomeDisposition",
    "EvidenceReceipt",
    "Obligation",
    "RefinementWitness",
    "ClaimEntailmentWitness",
    "IndependenceWitness",
    "ReachabilityWitness",
    "DependencyPath",
)

REQUIRED_DEFS = (
    "disposition",
    "AssumptionsAllowed",
    "SubjectTransfer",
    "ClaimTransfer",
    "Supports",
    "ReachabilitySatisfied",
    "Conflicting",
    "ConflictFree",
    "Closed",
    "IndependentSupport",
)

REQUIRED_THEOREMS = (
    "blocked_cannot_support",
    "environment_failure_cannot_support",
    "superseded_cannot_support",
    "invalidated_cannot_support",
    "evidence_class_non_amplification",
    "safety_cannot_close_liveness",
    "mismatched_subject_requires_refinement",
    "mismatched_claim_requires_entailment",
    "exact_subject_support_preserves_assumptions",
    "same_provenance_cannot_count_as_independent",
    "required_reachability_empty_rejected",
    "closed_requires_every_mandatory_obligation",
    "contradictory_current_evidence_prevents_closure",
    "dependency_path_decreases",
    "dependency_cycles_rejected",
    "support_non_amplification_summary",
)

EXPECTED_DISPOSITION_ARMS = (
    ".SemanticSuccess => .Pass",
    ".SemanticCounterexample => .Fail",
    ".ProofOrQualificationFailure => .Fail",
    ".UnsupportedBoundary => .Blocked",
    ".InsufficientBound => .Blocked",
    ".ResourceExhaustion => .Blocked",
    ".MissingPrerequisite => .Blocked",
    ".StaleSubjectOrDependency => .Blocked",
    ".AmbiguousOrUnknownOutcome => .Blocked",
    ".UnclassifiedToolCrash => .Blocked",
    ".EnvironmentUnavailable => .EnvironmentFailure",
    ".ToolInstallationFailure => .EnvironmentFailure",
    ".RunnerInfrastructureFailure => .EnvironmentFailure",
)

REQUIRED_NONCLAIMS = {
    "closure-calculus-is-not-soundness-of-imported-verifiers",
    "closure-calculus-does-not-prove-truth-of-imported-premises",
    "closure-calculus-is-not-production-validator-refinement",
    "closure-calculus-is-not-xenia-pulse-or-finance-system-proof",
    "closure-calculus-is-not-runtime-authorization",
    "closure-calculus-is-not-legal-or-safety-certification",
    "source-authored-theorem-is-not-executed-qualified-theorem",
}


class ContractError(RuntimeError):
    pass


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def validate_contract(manifest: dict, proof: str) -> None:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ContractError("wrong schema")
    if manifest.get("issue") != EXPECTED_ISSUE:
        raise ContractError("wrong issue binding")
    if manifest.get("parent_issue") != PARENT_ISSUE:
        raise ContractError("wrong repaired parent issue")
    if manifest.get("parent_repaired_head") != PARENT_HEAD:
        raise ContractError("wrong repaired parent head")
    if manifest.get("evidence_class") != "AbstractFormalTheorem":
        raise ContractError("wrong evidence class")

    parent = manifest.get("parent_proof", {})
    if parent.get("path") != PARENT_PROOF_PATH:
        raise ContractError("wrong parent proof path")
    if parent.get("blob") != PARENT_PROOF_BLOB:
        raise ContractError("wrong parent proof blob")
    if parent.get("relationship") != "qualification-pattern-and-formal-substrate-parent-not-semantic-premise":
        raise ContractError("parent relationship overclaims semantic premise")

    subject = manifest.get("proof_subject", {})
    if subject.get("path") != str(PROOF_PATH.relative_to(ROOT)):
        raise ContractError("wrong proof subject path")
    if subject.get("requires_mathlib") is not False:
        raise ContractError("unexpected Mathlib requirement")
    if subject.get("requires_user_axioms") is not False:
        raise ContractError("user axioms unexpectedly permitted")
    if subject.get("proof_holes_permitted") is not False:
        raise ContractError("proof holes unexpectedly permitted")

    if re.search(r"\b(sorry|admit)\b", proof):
        raise ContractError("proof-hole token present")
    if re.search(r"(?m)^\s*(axiom|opaque)\s+", proof):
        raise ContractError("user-declared axiom/opaque declaration present")
    if re.search(r"\bunsafe\b", proof):
        raise ContractError("unsafe declaration/token present")
    if "import Mathlib" in proof:
        raise ContractError("unexpected Mathlib dependency")
    if "namespace Symthaea.Formal.Assurance" not in proof:
        raise ContractError("canonical assurance namespace missing")

    for type_name in REQUIRED_TYPES:
        if not re.search(rf"\b(?:inductive|structure)\s+{re.escape(type_name)}\b", proof):
            raise ContractError(f"required type missing: {type_name}")

    for def_name in REQUIRED_DEFS:
        if not re.search(rf"\bdef\s+{re.escape(def_name)}\b", proof):
            raise ContractError(f"required definition missing: {def_name}")

    for theorem in REQUIRED_THEOREMS:
        if not re.search(rf"\btheorem\s+{re.escape(theorem)}\b", proof):
            raise ContractError(f"required theorem missing: {theorem}")
        if f"#print axioms {theorem}" not in proof:
            raise ContractError(f"axiom census directive missing: {theorem}")

    for arm in EXPECTED_DISPOSITION_ARMS:
        if arm not in proof:
            raise ContractError(f"canonical disposition arm missing/drifted: {arm}")

    # Structural non-amplification controls.
    supports_match = re.search(
        r"def\s+Supports\b(?P<body>.*?)(?=\n/-- Positive reachability)",
        proof,
        re.S,
    )
    if not supports_match:
        raise ContractError("Supports body not found")
    supports = supports_match.group("body")
    required_support_terms = (
        "e.validity = .Current",
        "disposition e.result = .Pass",
        "e.evidenceClass = o.requiredClass",
        "e.propertyKind = o.propertyKind",
        "SubjectTransfer refinements e o",
        "ClaimTransfer entailments e o",
    )
    for term in required_support_terms:
        if term not in supports:
            raise ContractError(f"Supports lost mandatory term: {term}")

    if "rank prerequisite < rank dependent" not in proof:
        raise ContractError("rank-decreasing dependency edge missing")
    if "Nat.lt_irrefl" not in proof:
        raise ContractError("dependency cycle rejection theorem missing irreflexivity proof")

    independence_match = re.search(
        r"def\s+IndependentSupport\b(?P<body>.*?)(?=\n/--\nDependency justification)",
        proof,
        re.S,
    )
    if not independence_match:
        raise ContractError("IndependentSupport body not found")
    independence = independence_match.group("body")
    for term in (
        "left.evidenceId ≠ right.evidenceId",
        "left.provenanceRoot ≠ right.provenanceRoot",
        "w.basisDeclared = true",
    ):
        if term not in independence:
            raise ContractError(f"independence anti-double-counting term missing: {term}")

    kernel = manifest.get("semantic_kernel", {})
    if kernel.get("dependency_discipline") != "strictly rank-decreasing DependencyPath":
        raise ContractError("manifest dependency discipline drift")
    if set(kernel.get("canonical_dispositions", [])) != {
        "Pass", "Fail", "Blocked", "EnvironmentFailure"
    }:
        raise ContractError("manifest disposition vocabulary drift")

    nonclaims = set(manifest.get("nonclaims", []))
    if not REQUIRED_NONCLAIMS.issubset(nonclaims):
        raise ContractError("required claim ceiling missing")


def expect_reject(label: str, manifest: dict, proof: str) -> None:
    try:
        validate_contract(manifest, proof)
    except ContractError:
        print(f"negative control {label}: REJECTED")
        return
    raise ContractError(f"negative control false-green: {label}")


def main() -> int:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        proof = PROOF_PATH.read_text(encoding="utf-8")

        if git("merge-base", "--is-ancestor", PARENT_HEAD, "HEAD", check=False).returncode != 0:
            raise ContractError("current branch is not descended from repaired SYM-FV-002 head")

        observed_parent_blob = git("rev-parse", f"{PARENT_HEAD}:{PARENT_PROOF_PATH}").stdout.strip()
        if observed_parent_blob != PARENT_PROOF_BLOB:
            raise ContractError("repaired parent theorem blob does not match manifest")

        validate_contract(manifest, proof)
        print("SYM-FV-ASSURANCE-001A static contract: PASS")

        mutant = copy.deepcopy(manifest)
        mutant["issue"] = 6065
        expect_reject("issue-rebinding", mutant, proof)

        mutant = copy.deepcopy(manifest)
        mutant["parent_repaired_head"] = "0" * 40
        expect_reject("parent-lineage-rebinding", mutant, proof)

        expect_reject(
            "blocked-promoted-to-pass",
            manifest,
            proof.replace(".ResourceExhaustion => .Blocked", ".ResourceExhaustion => .Pass", 1),
        )
        expect_reject(
            "safety-liveness-gate-deleted",
            manifest,
            proof.replace("e.propertyKind = o.propertyKind ∧", "True ∧", 1),
        )
        expect_reject(
            "evidence-class-gate-deleted",
            manifest,
            proof.replace("e.evidenceClass = o.requiredClass ∧", "True ∧", 1),
        )
        expect_reject(
            "independence-provenance-deleted",
            manifest,
            proof.replace("left.provenanceRoot ≠ right.provenanceRoot ∧", "True ∧", 1),
        )
        expect_reject(
            "dependency-rank-decrease-deleted",
            manifest,
            proof.replace("rank prerequisite < rank dependent", "True", 1),
        )
        expect_reject(
            "theorem-deletion",
            manifest,
            proof.replace("theorem dependency_cycles_rejected", "theorem dependency_cycles_rejected_missing", 1),
        )
        expect_reject("proof-hole", manifest, proof + "\nexample : True := by sorry\n")

        print("SYM-FV-ASSURANCE-001A negative controls: PASS")
        return 0
    except (ContractError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"SYM-FV-ASSURANCE-001A validation: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
