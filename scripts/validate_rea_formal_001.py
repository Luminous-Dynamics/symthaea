#!/usr/bin/env python3
"""Static admission checks for the REA provider-observation Lean adapter."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/formal/rea-provider-observation-v1.json"
CLOSURE = ROOT / "formal/lean/assurance/Closure.lean"
PROOF = ROOT / "formal/lean/assurance/REAProviderObservation.lean"

PARENT_HEAD = "e0172f5440e917641bfd124c4ccfc8458819883b"
PARENT_CLOSURE_BLOB = "3b7e9edca79b01cc005b86ca6ae0c2afdc2e26f4"

THEOREMS = (
    "provider_observation_support_binds_full_request",
    "changed_derivation_digest_cannot_support",
    "changed_claim_id_cannot_support",
    "changed_subject_digest_cannot_support",
    "changed_claim_digest_cannot_support",
    "changed_request_schema_cannot_support",
    "changed_result_schema_cannot_support",
    "changed_rule_generation_cannot_support",
    "changed_rule_id_cannot_support",
    "changed_platform_cannot_support",
    "changed_platform_version_cannot_support",
    "changed_environment_cannot_support",
    "provider_observation_is_a_distinct_evidence_class",
    "provider_observation_support_has_exact_predicate",
    "provider_observation_cannot_promote_to_other_predicate",
    "provider_observation_cannot_close_member_integrity",
    "provider_observation_cannot_close_producer_authenticity",
    "provider_observation_cannot_close_signature_validity",
    "provider_observation_cannot_close_signer_identity",
    "provider_observation_cannot_close_signer_trust",
    "provider_observation_cannot_close_package_authorization",
    "provider_observation_cannot_close_archive_completeness",
    "provider_observation_cannot_close_platform_acceptance",
    "provider_observation_cannot_close_security_control",
    "confidence_authority_are_non_amplifying",
    "provider_observation_refutation_remains_meta_claim",
    "provider_aliases_do_not_create_independence",
)


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def validate() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    closure = CLOSURE.read_text(encoding="utf-8")
    proof = PROOF.read_text(encoding="utf-8")

    require(
        manifest.get("schema") == "symthaea.formal.rea-provider-observation.v1",
        "wrong REA formal schema",
    )
    require(manifest.get("issue") == 10782, "wrong issue binding")
    require(manifest.get("parent_issue") == 6073, "wrong formal parent issue")
    require(manifest.get("parent_pr") == 6077, "wrong formal parent PR")
    require(manifest.get("parent_formal_head") == PARENT_HEAD, "wrong formal parent head")
    require(
        manifest.get("evidence_class") == "AbstractFormalTheorem",
        "wrong formal evidence class",
    )
    require(
        manifest.get("proof_subject", {}).get("path")
        == str(PROOF.relative_to(ROOT)),
        "wrong Lean proof subject path",
    )
    require(
        manifest.get("proof_subject", {}).get("imports") == ["assurance.Closure"],
        "adapter must import the existing closure calculus",
    )
    require(
        manifest.get("proof_subject", {}).get("requires_mathlib") is False,
        "Mathlib is not authorized in this adapter",
    )
    require(
        manifest.get("proof_subject", {}).get("requires_user_axioms") is False,
        "user axioms are not authorized",
    )
    require(
        manifest.get("proof_subject", {}).get("proof_holes_permitted") is False,
        "proof holes are not authorized",
    )

    parent_blob = git("rev-parse", f"{PARENT_HEAD}:formal/lean/assurance/Closure.lean").stdout.strip()
    require(parent_blob == PARENT_CLOSURE_BLOB, "pinned parent closure blob drifted")
    ancestor = git("merge-base", "--is-ancestor", PARENT_HEAD, "HEAD", check=False)
    require(ancestor.returncode == 0, "pinned formal parent is not an ancestor of this subject")

    require(
        re.search(r"\binductive\s+EvidenceClass\b", closure) is not None,
        "generic closure EvidenceClass is missing",
    )
    require(
        re.search(r"(?m)^\s*\|\s*ProviderObservation\s*$", closure) is not None,
        "ProviderObservation is not a distinct constructor in the existing closure kernel",
    )
    require(
        "import assurance.Closure" in proof,
        "REA adapter must import the existing closure kernel",
    )
    require(
        "namespace Symthaea.Formal.REA" in proof,
        "REA formal namespace missing",
    )
    require(
        re.search(r"\bstructure\s+ClaimBinding\b", proof) is not None,
        "exact ClaimBinding is missing",
    )
    for term in (
        "claimRequestDigest",
        "derivationDigest",
        "requestSchemaGeneration",
        "resultSchemaGeneration",
        "evaluatorRuleGeneration",
        "ruleId",
        "evaluatorVersion",
        "claimId",
        "subjectSha256",
        "operation",
        "platform",
        "platformVersion",
        "environment",
        "predicate",
    ):
        require(term in proof, f"claim binding field missing: {term}")

    require(
        re.search(r"\binductive\s+ObservationConfidence\b", proof) is not None,
        "typed confidence descriptor missing",
    )
    require(
        re.search(r"\binductive\s+ObservationAuthority\b", proof) is not None,
        "typed authority descriptor missing",
    )
    require(
        not re.search(r"inductive\s+Observation(?:Confidence|Authority)\b[\s\S]{0,500}deriving[^\n]*\b(?:Ord|PartialOrd)\b", proof),
        "confidence or authority must not define an ordering",
    )
    require(
        "evidenceClass = .ProviderObservation" in proof,
        "positive adapter support must require ProviderObservation",
    )
    require(
        "ProviderReportedObservation" in proof,
        "v0 meta-claim predicate is missing",
    )
    require(
        "receipt.binding = obligation.binding" in proof,
        "exact full request binding is not required",
    )
    require(
        "receipt.binding.evaluatorRuleGeneration = receipt.base.generation" in proof
        and "obligation.binding.evaluatorRuleGeneration = obligation.base.generation" in proof,
        "evaluator rule generation is not tied to generic closure generation",
    )
    require(
        "semanticPolarity receipt.base.result = .SemanticCounterevidence" in proof,
        "provider-report contradiction is not typed as meta-claim counterevidence",
    )
    require(
        "same_provenance_cannot_count_as_independent" in proof,
        "provider aliases must reuse the generic provenance independence rule",
    )

    for theorem in THEOREMS:
        require(
            re.search(rf"\btheorem\s+{re.escape(theorem)}\b", proof) is not None,
            f"required theorem missing: {theorem}",
        )
        require(
            f"#print axioms {theorem}" in proof,
            f"axiom census directive missing: {theorem}",
        )
    require(
        manifest.get("theorem_targets") == list(THEOREMS),
        "manifest theorem targets differ from the source contract",
    )

    for forbidden in (r"\b(sorry|admit)\b", r"(?m)^\s*(axiom|opaque)\s+", r"\bunsafe\b"):
        require(re.search(forbidden, proof) is None, f"forbidden Lean source pattern: {forbidden}")
    require("import Mathlib" not in proof, "Mathlib import is not permitted")

    require(
        all(item[0] == "ProviderReportedObservation" for item in manifest["forbidden_promotions"]),
        "forbidden-promotion matrix must originate at provider observation",
    )
    allowed = {name for name in re.findall(r"\|\s*(\w+)\s*$", proof, re.M)}
    for _, target in manifest["forbidden_promotions"]:
        require(target in allowed, f"forbidden-promotion target is not in ClaimPredicate: {target}")

    print("REA-FORMAL-001 static contract: PASS")


if __name__ == "__main__":
    try:
        validate()
    except (ContractError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"REA-FORMAL-001 static contract: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
