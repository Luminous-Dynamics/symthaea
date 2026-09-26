#!/usr/bin/env python3
"""Static admission gate for SYM-HDC-CRYPTO-FV-001B.

This gate binds the common-mask theorem to the exact inherited HDC algebra and
exact quarantined production FHE source subject. Lean typechecking remains a
separate qualification step; this validator prevents subject/claim drift and
runs hostile source/manifest controls without third-party dependencies.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/formal/sym-hdc-crypto-fv-001b-common-mask-v1.json"
PROOF_PATH = ROOT / "formal/lean/hdc/HdcCryptoCommonMask.lean"
FHE_PATH = ROOT / "crates/core/symthaea-hdc-crypto/src/fhe.rs"

EXPECTED_SCHEMA = "symthaea.formal.hdc-crypto-fv-001b-common-mask.v1"
EXPECTED_ISSUE = 5844
EXPECTED_PARENT = "9d1d279bdfad2dde43c4cde85ccfc98880f51dd1"
EXPECTED_HDC_PROOF_BLOB = "f724618781b2660d1e76e45a69dcf796a792d76d"
EXPECTED_ATTACK_PROOF_BLOB = "df18652dad946dcc1d22da335eba4ee78838b07e"
EXPECTED_FHE_BLOB = "bd67fdc42dc9ed598b2c93f80995324db066c9a7"
EXPECTED_FHE_PATH = "crates/core/symthaea-hdc-crypto/src/fhe.rs"
REQUIRED_THEOREMS = (
    "common_mask_preserves_pairwise_xor",
    "common_mask_leaks_pairwise_relation",
    "common_mask_preserves_pairwise_xor_swapped",
)
FORBIDDEN_REDEFINITIONS = ("BinaryHV", "HdcDimension", "BitIndex", "bind", "zero")


class ContractError(RuntimeError):
    pass


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_contract(manifest: dict, proof: str, fhe_source: str, observed_fhe_blob: str) -> None:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ContractError("wrong schema")
    if manifest.get("tracking_issue") != EXPECTED_ISSUE:
        raise ContractError("wrong tracking issue")
    if manifest.get("authority") != "EvidenceOnly":
        raise ContractError("authority must remain EvidenceOnly")
    if manifest.get("evidence_class") != "AbstractFormalTheorem":
        raise ContractError("wrong evidence class")

    parent = manifest.get("exact_parent", {})
    if parent.get("commit") != EXPECTED_PARENT:
        raise ContractError("parent commit drift")
    if parent.get("abstract_hdc_proof", {}).get("blob") != EXPECTED_HDC_PROOF_BLOB:
        raise ContractError("abstract HDC proof binding drift")
    if parent.get("attack_proof", {}).get("blob") != EXPECTED_ATTACK_PROOF_BLOB:
        raise ContractError("001A attack proof binding drift")

    target = manifest.get("production_future_refinement_target", {})
    expected_target = {
        "commit": EXPECTED_PARENT,
        "path": EXPECTED_FHE_PATH,
        "blob": EXPECTED_FHE_BLOB,
        "relationship": "future-refinement-target-not-proved-here",
    }
    for key, value in expected_target.items():
        if target.get(key) != value:
            raise ContractError(f"production target {key} drift")
    if observed_fhe_blob != EXPECTED_FHE_BLOB:
        raise ContractError("production FHE source blob drifted at child head")

    # Bind the theorem to the actual quarantined source shape without claiming refinement.
    source_needles = (
        "//! **QUARANTINED:**",
        "ciphertext: plaintext.bind(mask)",
        "When both are encrypted with the **same mask**, returns the true similarity.",
        "The pool below deliberately reuses one mask, exposing pairwise plaintext",
    )
    for needle in source_needles:
        if needle not in fhe_source:
            raise ContractError(f"production source semantic marker missing: {needle}")

    formal = manifest.get("formal_subject", {})
    if formal.get("reuses_parent_binaryhv") is not True:
        raise ContractError("child must reuse parent BinaryHV algebra")
    if formal.get("defines_hamming_distance") is not False:
        raise ContractError("001B must not claim its own Hamming definition")
    if tuple(formal.get("theorems", ())) != REQUIRED_THEOREMS:
        raise ContractError("theorem census drift")

    if re.search(r"\b(sorry|admit)\b", proof):
        raise ContractError("proof hole token present")
    if re.search(r"(?m)^\s*(axiom|opaque)\s+", proof):
        raise ContractError("user-declared axiom/opaque declaration present")

    for name in FORBIDDEN_REDEFINITIONS:
        if re.search(rf"(?m)^\s*(?:def|abbrev|structure|inductive)\s+{re.escape(name)}\b", proof):
            raise ContractError(f"child illegally redefines inherited algebra: {name}")

    if not re.search(
        r"theorem\s+common_mask_preserves_pairwise_xor\s*\n?\s*\(x y mask : BinaryHV\).*?"
        r"bind\s*\(maskWith x mask\)\s*\(maskWith y mask\)\s*=\s*bind x y",
        proof,
        re.S,
    ):
        raise ContractError("exact common-mask pairwise-XOR theorem shape missing")

    for theorem in REQUIRED_THEOREMS:
        if not re.search(rf"\btheorem\s+{re.escape(theorem)}\b", proof):
            raise ContractError(f"required theorem missing: {theorem}")
        if f"#print axioms {theorem}" not in proof:
            raise ContractError(f"axiom census directive missing: {theorem}")

    deferred = manifest.get("deferred_corollaries", [])
    if not deferred or deferred[0].get("target") != "SYM-FV-004":
        raise ContractError("Hamming corollary must remain explicitly deferred to SYM-FV-004")

    nonclaims = set(manifest.get("nonclaims", []))
    required_nonclaims = {
        "pairwise-xor-theorem-is-not-yet-a-hamming-distance-theorem",
        "abstract-theorem-is-not-production-rust-refinement",
        "abstract-theorem-is-not-proof-that-all-hdc-cryptography-is-insecure",
        "abstract-theorem-is-not-a-semantic-security-theorem",
        "abstract-theorem-is-not-a-rustc-llvm-or-native-binary-proof",
        "abstract-theorem-grants-no-runtime-authority",
    }
    if not required_nonclaims.issubset(nonclaims):
        raise ContractError("required claim ceiling missing")

    claim = manifest.get("primary_claim", "")
    if "pairwise XOR relation" not in claim or "Hamming" in claim:
        raise ContractError("primary claim must remain pairwise-XOR-only")


def expect_reject(label: str, manifest: dict, proof: str, fhe_source: str, blob: str) -> None:
    try:
        validate_contract(manifest, proof, fhe_source, blob)
    except ContractError:
        print(f"negative control {label}: REJECTED")
        return
    raise ContractError(f"negative control false-green: {label}")


def main() -> int:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        proof = PROOF_PATH.read_text(encoding="utf-8")
        fhe_source = FHE_PATH.read_text(encoding="utf-8")

        historical_fhe = git("rev-parse", f"{EXPECTED_PARENT}:{EXPECTED_FHE_PATH}")
        observed_fhe = git("rev-parse", f"HEAD:{EXPECTED_FHE_PATH}")
        if historical_fhe != EXPECTED_FHE_BLOB:
            raise ContractError("historical FHE source no longer resolves to expected blob")

        historical_hdc = git("rev-parse", f"{EXPECTED_PARENT}:formal/lean/hdc/BinaryHVBind.lean")
        historical_attack = git("rev-parse", f"{EXPECTED_PARENT}:formal/lean/hdc/HdcCryptoAttacks.lean")
        if historical_hdc != EXPECTED_HDC_PROOF_BLOB:
            raise ContractError("historical HDC proof blob drift")
        if historical_attack != EXPECTED_ATTACK_PROOF_BLOB:
            raise ContractError("historical 001A attack proof blob drift")

        validate_contract(manifest, proof, fhe_source, observed_fhe)
        print("SYM-HDC-CRYPTO-FV-001B static contract: PASS")

        # Common-mask is essential: replacing the second mask by another mask
        # must no longer satisfy the exact required theorem shape.
        expect_reject(
            "different-mask-on-second-operand",
            manifest,
            proof.replace(
                "bind (maskWith x mask) (maskWith y mask) = bind x y",
                "bind (maskWith x mask) (maskWith y otherMask) = bind x y",
                1,
            ),
            fhe_source,
            observed_fhe,
        )

        expect_reject(
            "wrong-rhs",
            manifest,
            proof.replace("= bind x y := by", "= zero := by", 1),
            fhe_source,
            observed_fhe,
        )

        expect_reject(
            "theorem-deletion",
            manifest,
            proof.replace(
                "theorem common_mask_preserves_pairwise_xor",
                "theorem common_mask_preserves_pairwise_xor_missing",
                1,
            ),
            fhe_source,
            observed_fhe,
        )
        expect_reject("proof-hole", manifest, proof + "\nexample : True := by sorry\n", fhe_source, observed_fhe)
        expect_reject(
            "parent-algebra-redefinition",
            manifest,
            proof + "\ndef bind (a b : BinaryHV) : BinaryHV := a\n",
            fhe_source,
            observed_fhe,
        )

        mutant = copy.deepcopy(manifest)
        mutant["production_future_refinement_target"]["blob"] = "0" * 40
        expect_reject("production-source-rebinding", mutant, proof, fhe_source, observed_fhe)

        mutant = copy.deepcopy(manifest)
        mutant["primary_claim"] = "Common-mask reuse preserves exact Hamming distance in production Rust."
        expect_reject("claim-escalation", mutant, proof, fhe_source, observed_fhe)

        print("SYM-HDC-CRYPTO-FV-001B negative controls: PASS")
        return 0
    except (ContractError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"SYM-HDC-CRYPTO-FV-001B validation: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
