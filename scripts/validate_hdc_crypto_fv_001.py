#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/formal/hdc-crypto-fv-001-negative-theorems-v1.json"
PROOF = ROOT / "formal/lean/hdc/HdcCryptoNegative.lean"
SOURCE_PATH = "crates/core/symthaea-hdc-crypto/src/crypto.rs"
EXPECTED_SOURCE_COMMIT = "2dfddf6027d8eaf62221f8a71be2bb6d1d7bd9a9"
EXPECTED_SOURCE_BLOB = "59c0ea6bc92fd1d67a9f764bb396918fd823331a"
REQUIRED = (
    "known_pair_recovers_effective_mask",
    "one_pair_forges_arbitrary_message",
    "one_share_recovers_secret",
)


class ContractError(RuntimeError):
    pass


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate(manifest: dict, proof: str, observed_blob: str) -> None:
    if manifest.get("schema") != "symthaea.formal.hdc-crypto-fv-001.v1":
        raise ContractError("wrong schema")
    if manifest.get("issue") != 5864 or manifest.get("parent_issue") != 5862:
        raise ContractError("wrong issue binding")
    if manifest.get("evidence_class") != "AbstractFormalTheorem":
        raise ContractError("wrong evidence class")

    source = manifest.get("source", {})
    if source.get("commit") != EXPECTED_SOURCE_COMMIT:
        raise ContractError("wrong source commit")
    if source.get("path") != SOURCE_PATH or source.get("blob") != EXPECTED_SOURCE_BLOB:
        raise ContractError("wrong source binding")
    if source.get("relationship") != "abstract-negative-theorem-target-not-rust-refinement":
        raise ContractError("source relationship overclaims refinement")
    if observed_blob != EXPECTED_SOURCE_BLOB:
        raise ContractError("current source blob drifted")

    if re.search(r"\b(sorry|admit)\b", proof):
        raise ContractError("proof hole token present")
    if re.search(r"(?m)^\s*(axiom|opaque)\s+", proof):
        raise ContractError("user axiom/opaque present")
    if "import Mathlib" in proof:
        raise ContractError("unexpected Mathlib dependency")
    if not re.search(r"def\s+HdcDimension\s*:\s*Nat\s*:=\s*16_384\b", proof):
        raise ContractError("dimension declaration missing")
    if "def mac (message effectiveMask : BinaryHV)" not in proof:
        raise ContractError("MAC abstraction missing")
    if "def recoverEffectiveMask" not in proof or "def forgeTag" not in proof:
        raise ContractError("forgery construction missing")
    if "structure Share" not in proof or "def recoverOne" not in proof:
        raise ContractError("share recovery model missing")

    for theorem in REQUIRED:
        if not re.search(rf"\btheorem\s+{re.escape(theorem)}\b", proof):
            raise ContractError(f"required theorem missing: {theorem}")
        if f"#print axioms {theorem}" not in proof:
            raise ContractError(f"axiom census missing: {theorem}")

    nonclaims = set(manifest.get("nonclaims", []))
    required_nonclaims = {
        "not-rust-refinement",
        "not-positive-mac-security",
        "not-positive-threshold-security",
    }
    if not required_nonclaims.issubset(nonclaims):
        raise ContractError("claim ceiling missing")


def expect_reject(label: str, manifest: dict, proof: str, observed_blob: str) -> None:
    try:
        validate(manifest, proof, observed_blob)
    except ContractError:
        print(f"negative control {label}: REJECTED")
        return
    raise ContractError(f"negative control false-green: {label}")


def main() -> int:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        proof = PROOF.read_text(encoding="utf-8")
        historical = git("rev-parse", f"{EXPECTED_SOURCE_COMMIT}:{SOURCE_PATH}")
        observed = git("rev-parse", f"HEAD:{SOURCE_PATH}")
        if historical != EXPECTED_SOURCE_BLOB:
            raise ContractError("historical source no longer resolves to expected blob")

        validate(manifest, proof, observed)
        print("HDC-CRYPTO-FV-001 static contract: PASS")

        mutant = copy.deepcopy(manifest)
        mutant["source"]["blob"] = "0" * 40
        expect_reject("source-rebinding", mutant, proof, observed)

        expect_reject(
            "forgery-theorem-deletion",
            manifest,
            proof.replace("theorem one_pair_forges_arbitrary_message", "theorem removed_forgery", 1),
            observed,
        )
        expect_reject(
            "share-theorem-deletion",
            manifest,
            proof.replace("theorem one_share_recovers_secret", "theorem removed_recovery", 1),
            observed,
        )
        expect_reject("proof-hole", manifest, proof + "\nexample : True := by sorry\n", observed)

        mutant = copy.deepcopy(manifest)
        mutant["nonclaims"] = [x for x in mutant["nonclaims"] if x != "not-positive-mac-security"]
        expect_reject("claim-escalation", mutant, proof, observed)

        print("HDC-CRYPTO-FV-001 negative controls: PASS")
        return 0
    except (ContractError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"HDC-CRYPTO-FV-001 validation: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
