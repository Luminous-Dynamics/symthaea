#!/usr/bin/env python3
"""Static admission gate for SYM-HDC-CRYPTO-FV-001A.

This child must reuse the exact SYM-FV-002 BinaryHV algebra rather than defining
another model. Lean typechecking is performed separately by the dedicated lane.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/formal/sym-hdc-crypto-fv-001a-v1.json"
CHILD_PATH = ROOT / "formal/lean/hdc/HdcCryptoAttacks.lean"
PARENT_PATH = ROOT / "formal/lean/hdc/BinaryHVBind.lean"

EXPECTED_SCHEMA = "symthaea.formal.sym-hdc-crypto-fv-001a.v1"
EXPECTED_ISSUE = 5857
PARENT_COMMIT = "2199cf04634e81d85592ed0479a5c3d2d0fb6ac8"
PARENT_BLOB = "f724618781b2660d1e76e45a69dcf796a792d76d"
SOURCE_COMMIT = "2dfddf6027d8eaf62221f8a71be2bb6d1d7bd9a9"
SOURCE_TARGETS = {
    "crates/core/symthaea-hdc-crypto/src/crypto.rs": "59c0ea6bc92fd1d67a9f764bb396918fd823331a",
    "crates/core/symthaea-core/src/hdc/hdc_crypto.rs": "a232b0b699f5d0301fe052b0e013f289194c13b6",
}
REQUIRED_THEOREMS = (
    "recover_derived_from_known_pair",
    "universal_known_pair_forgery",
    "forged_tag_is_accepted",
    "one_share_recovers_secret",
)
EXPECTED_CLAIM_CEILING = {
    "abstract-attack-algebra-only",
    "not-production-rust-refinement",
    "not-proof-all-hdc-cryptography-insecure",
    "not-commitment-context-or-fhe-proof-yet",
    "not-secure-replacement",
}


class ContractError(RuntimeError):
    pass


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate(manifest: dict, child: str, parent_blob: str, observed: dict[str, str]) -> None:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ContractError("wrong schema")
    if manifest.get("issue") != EXPECTED_ISSUE:
        raise ContractError("wrong issue binding")
    if manifest.get("evidence_class") != "AbstractFormalTheorem":
        raise ContractError("wrong evidence class")

    parent = manifest.get("stack_parent", {})
    if parent.get("commit") != PARENT_COMMIT:
        raise ContractError("wrong parent theorem commit")
    if parent.get("blob") != PARENT_BLOB:
        raise ContractError("wrong parent theorem blob in manifest")
    if parent.get("path") != "formal/lean/hdc/BinaryHVBind.lean":
        raise ContractError("wrong parent theorem path")
    if parent.get("relationship") != "reused-exact-abstract-xor-algebra":
        raise ContractError("parent relationship drift")
    if parent_blob != PARENT_BLOB:
        raise ContractError("checked-out parent theorem blob drift")

    targets = manifest.get("production_refinement_targets", [])
    by_path = {t.get("path"): t for t in targets}
    if set(by_path) != set(SOURCE_TARGETS):
        raise ContractError("production source target census drift")
    for path, expected_blob in SOURCE_TARGETS.items():
        target = by_path[path]
        if target.get("commit") != SOURCE_COMMIT:
            raise ContractError(f"wrong source commit for {path}")
        if target.get("blob") != expected_blob:
            raise ContractError(f"wrong source blob for {path}")
        if observed.get(path) != expected_blob:
            raise ContractError(f"checked-out production source drift: {path}")

    if re.search(r"\b(sorry|admit)\b", child):
        raise ContractError("proof hole token present")
    if re.search(r"(?m)^\s*(axiom|opaque)\s+", child):
        raise ContractError("user axiom/opaque declaration present")
    if "import Mathlib" in child:
        raise ContractError("unexpected Mathlib dependency")

    # This is a child theorem: it must reuse the parent model, not redefine it.
    forbidden_model_defs = (
        r"\babbrev\s+BinaryHV\b",
        r"\bdef\s+BinaryHV\b",
        r"\bdef\s+bind\b",
        r"\bdef\s+zero\b",
        r"\bdef\s+HdcDimension\b",
    )
    for pattern in forbidden_model_defs:
        if re.search(pattern, child):
            raise ContractError(f"duplicate HDC model definition detected: {pattern}")

    required_fragments = (
        "open Symthaea.Formal.HDC",
        "def macTag",
        "def recoverDerived",
        "def forgeTag",
        "def makeShare",
        "def recoverOne",
    )
    for fragment in required_fragments:
        if fragment not in child:
            raise ContractError(f"required attack definition missing: {fragment}")

    for theorem in REQUIRED_THEOREMS:
        if not re.search(rf"\btheorem\s+{re.escape(theorem)}\b", child):
            raise ContractError(f"required theorem missing: {theorem}")
        if f"#print axioms {theorem}" not in child:
            raise ContractError(f"axiom census missing: {theorem}")

    if set(manifest.get("theorem_subjects", [])) != set(REQUIRED_THEOREMS):
        raise ContractError("theorem manifest drift")
    if set(manifest.get("claim_ceiling", [])) != EXPECTED_CLAIM_CEILING:
        raise ContractError("claim ceiling drift/escalation")
    coverage = manifest.get("coverage", {})
    if coverage.get("kind") != "Universal":
        raise ContractError("attack coverage must remain Universal")


def expect_reject(label: str, manifest: dict, child: str, parent_blob: str, observed: dict[str, str]) -> None:
    try:
        validate(manifest, child, parent_blob, observed)
    except ContractError:
        print(f"negative control {label}: REJECTED")
        return
    raise ContractError(f"negative control false-green: {label}")


def main() -> int:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        child = CHILD_PATH.read_text(encoding="utf-8")
        parent_blob = git("rev-parse", "HEAD:formal/lean/hdc/BinaryHVBind.lean")
        historical_parent = git("rev-parse", f"{PARENT_COMMIT}:formal/lean/hdc/BinaryHVBind.lean")
        if historical_parent != PARENT_BLOB:
            raise ContractError("historical parent theorem no longer resolves to expected blob")

        observed: dict[str, str] = {}
        for path, expected_blob in SOURCE_TARGETS.items():
            historical = git("rev-parse", f"{SOURCE_COMMIT}:{path}")
            if historical != expected_blob:
                raise ContractError(f"historical production binding mismatch: {path}")
            observed[path] = git("rev-parse", f"HEAD:{path}")

        validate(manifest, child, parent_blob, observed)
        print("SYM-HDC-CRYPTO-FV-001A static contract: PASS")

        mutant = copy.deepcopy(manifest)
        mutant["stack_parent"]["blob"] = "0" * 40
        expect_reject("parent-algebra-drift", mutant, child, parent_blob, observed)

        expect_reject(
            "duplicate-model-injection",
            manifest,
            child + "\nabbrev BinaryHV : Type := Bool\n",
            parent_blob,
            observed,
        )
        expect_reject(
            "forgery-theorem-deletion",
            manifest,
            child.replace("theorem universal_known_pair_forgery", "theorem universal_known_pair_forgery_missing", 1),
            parent_blob,
            observed,
        )
        expect_reject(
            "one-share-theorem-deletion",
            manifest,
            child.replace("theorem one_share_recovers_secret", "theorem one_share_recovers_secret_missing", 1),
            parent_blob,
            observed,
        )
        expect_reject("proof-hole", manifest, child + "\nexample : True := by sorry\n", parent_blob, observed)

        mutant = copy.deepcopy(manifest)
        mutant["production_refinement_targets"][0]["blob"] = "0" * 40
        expect_reject("source-rebinding", mutant, child, parent_blob, observed)

        mutant = copy.deepcopy(manifest)
        mutant["claim_ceiling"].append("production-hdc-mac-refinement-proved")
        expect_reject("claim-escalation", mutant, child, parent_blob, observed)

        print("SYM-HDC-CRYPTO-FV-001A negative controls: PASS")
        return 0
    except (ContractError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"SYM-HDC-CRYPTO-FV-001A validation: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
