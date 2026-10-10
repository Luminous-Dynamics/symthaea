#!/usr/bin/env python3
"""Static admission gate for SYM-FV-041A authority-kernel theorem subject."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/formal/sym-fv-041a-authority-kernel-v1.json"
PROOF_PATH = ROOT / "formal/lean/authority/AuthorityKernel.lean"

EXPECTED_SCHEMA = "symthaea.formal.sym-fv-041a-authority-kernel.v1"
EXPECTED_ISSUE = 5802
EXPECTED_PARENT = 5794
REQUIRED_FIELDS = (
    "subject", "action", "resource", "policy", "evidence", "authority", "context",
    "authorityEpoch", "policyEpoch", "evidenceEpoch",
)
REQUIRED_THEOREMS = (
    "trace_nonamplification",
    "nonamplification_violation_implies_promoting",
    "evidence_only_nonamplifying",
    "binding_exact_accepts",
    "subject_mismatch_reject",
    "action_mismatch_reject",
    "resource_mismatch_reject",
    "policy_mismatch_reject",
    "evidence_mismatch_reject",
    "authority_mismatch_reject",
    "context_mismatch_reject",
    "authority_epoch_binding_mismatch_reject",
    "policy_epoch_binding_mismatch_reject",
    "evidence_epoch_binding_mismatch_reject",
    "authority_epoch_currentness_mismatch_reject",
    "policy_epoch_currentness_mismatch_reject",
    "evidence_epoch_currentness_mismatch_reject",
)


class ContractError(RuntimeError):
    pass


def validate(manifest: dict, proof: str) -> None:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ContractError("wrong schema")
    if manifest.get("issue") != EXPECTED_ISSUE or manifest.get("parent_issue") != EXPECTED_PARENT:
        raise ContractError("wrong issue lineage")
    if manifest.get("evidence_class") != "AbstractFormalTheorem":
        raise ContractError("wrong evidence class")
    subject = manifest.get("proof_subject", {})
    if subject.get("authority_structure") != "Preorder":
        raise ContractError("authority algebra strengthened or changed")
    if subject.get("requires_mathlib") is not False:
        raise ContractError("unexpected Mathlib dependency")

    fields = manifest.get("abstract_binding_fields")
    if fields != list(REQUIRED_FIELDS):
        raise ContractError("binding field census drifted")
    theorems = manifest.get("theorems")
    if theorems != list(REQUIRED_THEOREMS):
        raise ContractError("theorem census drifted")

    if re.search(r"\b(sorry|admit)\b", proof):
        raise ContractError("proof hole token present")
    if re.search(r"(?m)^\s*(axiom|opaque)\s+", proof):
        raise ContractError("user-declared axiom/opaque present")
    if "import Mathlib" in proof:
        raise ContractError("Mathlib import present")
    if "[Preorder Authority]" not in proof:
        raise ContractError("Preorder authority boundary missing")
    if "inductive NonPromotingTrace" not in proof:
        raise ContractError("finite trace subject missing")
    if "EvidenceOnly" in proof:
        raise ContractError("unexpected capitalized evidence predicate spelling")
    if "evidenceOnly e → nonPromoting e" not in proof:
        raise ContractError("evidence non-promoting refinement obligation missing")

    binding_match = re.search(
        r"structure\s+Binding\s+\(Id\s*:\s*Type\s+u\)\s+where(?P<body>.*?)\n\n/-- Abstract admission",
        proof,
        re.S,
    )
    if not binding_match:
        raise ContractError("Binding structure not found")
    body = binding_match.group("body")
    for field in REQUIRED_FIELDS:
        if not re.search(rf"(?m)^\s*{re.escape(field)}\s*:\s*", body):
            raise ContractError(f"binding field missing: {field}")

    for theorem in REQUIRED_THEOREMS:
        if not re.search(rf"\btheorem\s+{re.escape(theorem)}\b", proof):
            raise ContractError(f"theorem missing: {theorem}")
        if f"#print axioms {theorem}" not in proof:
            raise ContractError(f"axiom census missing: {theorem}")

    required_nonclaims = {
        "abstract-theorem-is-not-concrete-subsystem-refinement",
        "finite-trace-safety-is-not-distributed-currentness",
        "currentness-equality-is-not-revocation-linearizability",
        "promotion-classification-is-not-promotion-authorization",
    }
    if not required_nonclaims.issubset(set(manifest.get("nonclaims", []))):
        raise ContractError("claim ceiling weakened")


def expect_reject(label: str, manifest: dict, proof: str) -> None:
    try:
        validate(manifest, proof)
    except ContractError:
        print(f"negative control {label}: REJECTED")
        return
    raise ContractError(f"negative control false-green: {label}")


def main() -> int:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        proof = PROOF_PATH.read_text(encoding="utf-8")
        validate(manifest, proof)
        print("SYM-FV-041A static contract: PASS")

        expect_reject(
            "theorem-deletion",
            manifest,
            proof.replace("theorem trace_nonamplification", "theorem trace_nonamplification_missing", 1),
        )
        expect_reject("proof-hole", manifest, proof + "\nexample : True := by sorry\n")
        expect_reject("binding-field-deletion", manifest, proof.replace("  evidence : Id\n", "", 1))
        expect_reject(
            "evidence-classification-deletion",
            manifest,
            proof.replace("evidenceOnly e → nonPromoting e", "evidenceOnly e → True", 1),
        )
        print("SYM-FV-041A negative controls: PASS")
        return 0
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        print(f"SYM-FV-041A validation: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
