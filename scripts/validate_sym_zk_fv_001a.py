#!/usr/bin/env python3
"""Static admission gate for SYM-ZK-FV-001A.

Lean typechecking is authoritative for the theorem. This gate binds the theorem
subject, production refinement targets, claim ceiling, and hostile mutants.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/formal/sym-zk-fv-001a-balance-relation-v1.json"
PROOF = ROOT / "formal/lean/zk/BalanceRelation.lean"

EXPECTED_SCHEMA = "symthaea.formal.sym-zk-fv-001a-balance-relation.v1"
EXPECTED_ISSUE = 5849
SOURCE_COMMIT = "2dfddf6027d8eaf62221f8a71be2bb6d1d7bd9a9"
TARGETS = {
    "crates/core/symthaea-zkproof/core/src/lib.rs": "07de3583896eaaab5b68994c2244acaca72152ac",
    "crates/core/symthaea-zkproof/methods/guest/src/main.rs": "8b57a56ca9e3051bbbf83107b916d645e90099c7",
}
REQUIRED_THEOREMS = (
    "relation_holds",
    "sufficient_eq_true_iff",
    "sufficient_true_of_ge",
    "sufficient_false_of_not_ge",
    "threshold_preserved",
    "nonce_preserved",
    "relation_deterministic",
)
EXPECTED_CLAIM_CEILING = {
    "statement-relation-only",
    "not-risc0-guest-refinement",
    "not-receipt-or-image-id-binding",
    "not-proof-system-soundness",
    "not-zero-knowledge",
    "not-side-channel-privacy",
    "not-runtime-authorization",
}


class ContractError(RuntimeError):
    pass


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate(manifest: dict, proof: str, observed: dict[str, str]) -> None:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ContractError("wrong schema")
    if manifest.get("issue") != EXPECTED_ISSUE:
        raise ContractError("wrong issue")
    if manifest.get("evidence_class") != "AbstractFormalTheorem":
        raise ContractError("wrong evidence class")

    subject = manifest.get("subject", {})
    if subject.get("name") != "BalanceRelation":
        raise ContractError("wrong statement name")
    if subject.get("value_domain") != "Fin (2^64)":
        raise ContractError("wrong value domain")
    if subject.get("private_witness_fields") != ["balance"]:
        raise ContractError("private witness census drift")
    if subject.get("public_input_fields") != ["requiredMinimum", "nonce"]:
        raise ContractError("public input census drift")
    if subject.get("public_output_fields") != ["sufficient", "requiredMinimum", "nonce"]:
        raise ContractError("public output census drift")

    targets = manifest.get("production_refinement_targets", [])
    by_path = {t.get("path"): t for t in targets}
    if set(by_path) != set(TARGETS):
        raise ContractError("production target path census drift")
    for path, blob in TARGETS.items():
        target = by_path[path]
        if target.get("commit") != SOURCE_COMMIT:
            raise ContractError(f"wrong source commit for {path}")
        if target.get("blob") != blob:
            raise ContractError(f"wrong source blob for {path}")
        if target.get("relationship") != "future-refinement-target-not-proved-here":
            raise ContractError(f"claim escalation for {path}")
        if observed.get(path) != blob:
            raise ContractError(f"HEAD source drift for {path}")

    if re.search(r"\b(sorry|admit)\b", proof):
        raise ContractError("proof hole token present")
    if re.search(r"(?m)^\s*(axiom|opaque)\s+", proof):
        raise ContractError("user axiom/opaque present")
    if "import Mathlib" in proof:
        raise ContractError("unexpected Mathlib dependency")
    if not re.search(r"abbrev\s+U64\s*:\s*Type\s*:=\s*Fin\s*\(2\s*\^\s*64\)", proof):
        raise ContractError("exact U64 domain missing")
    if "decide (i.requiredMinimum ≤ i.balance)" not in proof:
        raise ContractError("exact balance comparison missing")
    if not re.search(r"def\s+BalanceRelation\b", proof):
        raise ContractError("BalanceRelation definition missing")

    for theorem in REQUIRED_THEOREMS:
        if not re.search(rf"\btheorem\s+{re.escape(theorem)}\b", proof):
            raise ContractError(f"required theorem missing: {theorem}")
        if f"#print axioms {theorem}" not in proof:
            raise ContractError(f"axiom census missing: {theorem}")

    if set(manifest.get("required_theorems", [])) != set(REQUIRED_THEOREMS):
        raise ContractError("theorem manifest drift")
    if set(manifest.get("claim_ceiling", [])) != EXPECTED_CLAIM_CEILING:
        raise ContractError("claim ceiling drift/escalation")

    coverage = manifest.get("coverage", {})
    if coverage.get("kind") != "Universal":
        raise ContractError("coverage must remain Universal")
    if coverage.get("rust_u64_refinement") != "unproved-successor-obligation":
        raise ContractError("Rust u64 refinement was silently promoted")


def expect_reject(label: str, manifest: dict, proof: str, observed: dict[str, str]) -> None:
    try:
        validate(manifest, proof, observed)
    except ContractError:
        print(f"negative control {label}: REJECTED")
        return
    raise ContractError(f"negative control false-green: {label}")


def main() -> int:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        proof = PROOF.read_text(encoding="utf-8")
        observed: dict[str, str] = {}
        for path, expected in TARGETS.items():
            historical = git("rev-parse", f"{SOURCE_COMMIT}:{path}")
            if historical != expected:
                raise ContractError(f"historical binding mismatch for {path}")
            observed[path] = git("rev-parse", f"HEAD:{path}")

        validate(manifest, proof, observed)
        print("SYM-ZK-FV-001A static contract: PASS")

        expect_reject(
            "comparison-mutation",
            manifest,
            proof.replace("decide (i.requiredMinimum ≤ i.balance)", "decide (i.requiredMinimum < i.balance)", 1),
            observed,
        )
        expect_reject(
            "nonce-theorem-deletion",
            manifest,
            proof.replace("theorem nonce_preserved", "theorem nonce_preserved_missing", 1),
            observed,
        )
        expect_reject(
            "threshold-theorem-deletion",
            manifest,
            proof.replace("theorem threshold_preserved", "theorem threshold_preserved_missing", 1),
            observed,
        )
        expect_reject("proof-hole", manifest, proof + "\nexample : True := by sorry\n", observed)

        mutant = copy.deepcopy(manifest)
        mutant["production_refinement_targets"][0]["blob"] = "0" * 40
        expect_reject("source-rebinding", mutant, proof, observed)

        mutant = copy.deepcopy(manifest)
        mutant["claim_ceiling"].append("guest-refinement-proved")
        expect_reject("claim-escalation", mutant, proof, observed)

        print("SYM-ZK-FV-001A negative controls: PASS")
        return 0
    except (ContractError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"SYM-ZK-FV-001A validation: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
