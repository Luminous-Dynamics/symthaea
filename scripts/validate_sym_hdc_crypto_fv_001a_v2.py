#!/usr/bin/env python3
"""Static admission gate for SYM-HDC-CRYPTO-FV-001A v2.

This gate validates immutable parent/source bindings, exact child scope, theorem
presence, non-duplication of the inherited HDC model, and hostile mutations.
Lean elaboration is a separate mandatory step in the dedicated workflow.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_COMMIT = "a4d4506c57902f96b6c260e7e9c56f9fc2e654c3"
PARENT_PATH = "formal/lean/hdc/BinaryHVBind.lean"
PARENT_BLOB = "259ba64888d8492bafc123b72d67b70da9282c36"
INVALIDATED_V1_PARENT_BLOB = "f724618781b2660d1e76e45a69dcf796a792d76d"
CHILD_PATH = ROOT / "formal/lean/hdc/HdcCryptoAttacks.lean"
MANIFEST_PATH = ROOT / "docs/formal/sym-hdc-crypto-fv-001a-v2.json"
EXPECTED_SCHEMA = "symthaea.formal.sym-hdc-crypto-fv-001a.v2"
EXPECTED_FILES = {
    ".github/workflows/sym-hdc-crypto-fv-001a-v2.yml",
    "docs/formal/sym-hdc-crypto-fv-001a-v2.json",
    "formal/lean/hdc/HdcCryptoAttacks.lean",
    "scripts/validate_sym_hdc_crypto_fv_001a_v2.py",
}
SOURCE_TARGETS = {
    "crates/core/symthaea-hdc-crypto/src/crypto.rs": "59c0ea6bc92fd1d67a9f764bb396918fd823331a",
    "crates/core/symthaea-core/src/hdc/hdc_crypto.rs": "a232b0b699f5d0301fe052b0e013f289194c13b6",
}
REQUIRED_THEOREMS = {
    "recover_derived_from_known_pair",
    "universal_known_pair_forgery",
    "forged_tag_is_accepted",
    "one_share_recovers_secret",
}
FORBIDDEN_REDEFS = ("HdcDimension", "BitIndex", "BinaryHV", "bind", "zero")
FORBIDDEN_PROOF_HOLES = ("sorry", "admit")
REQUIRED_CEILING = {
    "abstract-attack-algebra-only",
    "not-production-rust-refinement",
    "not-proof-all-hdc-cryptography-insecure",
    "not-commitment-context-or-fhe-proof-yet",
    "not-secure-replacement",
    "not-compiler-or-native-binary-proof",
    "not-runtime-authority",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def fail(msg: str) -> None:
    raise AssertionError(msg)


def theorem_names(text: str) -> set[str]:
    return set(re.findall(r"(?m)^theorem\s+([A-Za-z0-9_']+)", text))


def validate_child_text(text: str) -> None:
    names = theorem_names(text)
    missing = REQUIRED_THEOREMS - names
    if missing:
        fail(f"missing required theorem(s): {sorted(missing)}")
    for name in REQUIRED_THEOREMS:
        if f"#print axioms {name}" not in text:
            fail(f"missing axiom census for {name}")
    for symbol in FORBIDDEN_REDEFS:
        if re.search(rf"(?m)^(?:def|abbrev|structure|inductive)\s+{re.escape(symbol)}\b", text):
            fail(f"child illegally redefines inherited semantic root {symbol}")
    lowered = text.lower()
    for token in FORBIDDEN_PROOF_HOLES:
        if re.search(rf"\b{token}\b", lowered):
            fail(f"proof hole token present: {token}")
    if re.search(r"(?m)^axiom\s+", text):
        fail("child introduces an axiom")


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        fail("manifest schema drift")
    parent = manifest.get("stack_parent", {})
    if parent.get("commit") != PARENT_COMMIT or parent.get("blob") != PARENT_BLOB:
        fail("manifest parent binding drift")
    if parent.get("blob") == INVALIDATED_V1_PARENT_BLOB:
        fail("invalidated v1 parent theorem blob reintroduced")
    targets = {x["path"]: x["blob"] for x in manifest.get("production_refinement_targets", [])}
    if targets != SOURCE_TARGETS:
        fail("production source bindings drift")
    if set(manifest.get("claim_ceiling", [])) != REQUIRED_CEILING:
        fail("claim ceiling drift")
    qc = manifest.get("qualification_contract", {})
    for key in (
        "exact_pr_head_checkout",
        "known_bad_lean_subject_must_fail",
        "prover_exit_must_propagate",
        "sorryAx_forbidden",
        "lean_error_markers_forbidden",
        "repository_postflight_clean",
    ):
        if qc.get(key) is not True:
            fail(f"qualification invariant disabled: {key}")


def main() -> int:
    child = CHILD_PATH.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if PARENT_BLOB == INVALIDATED_V1_PARENT_BLOB:
        fail("repaired parent blob unexpectedly equals invalidated v1 blob")
    if git("rev-parse", f"{PARENT_COMMIT}:{PARENT_PATH}") != PARENT_BLOB:
        fail("immutable parent theorem blob mismatch")
    if git("rev-parse", f"HEAD:{PARENT_PATH}") != PARENT_BLOB:
        fail("child does not inherit exact repaired parent theorem blob")
    for path, blob in SOURCE_TARGETS.items():
        if git("rev-parse", f"{PARENT_COMMIT}:{path}") != blob:
            fail(f"immutable production blob mismatch: {path}")
        if git("rev-parse", f"HEAD:{path}") != blob:
            fail(f"child production source drift: {path}")

    changed = set(filter(None, git("diff", "--name-only", PARENT_COMMIT, "HEAD").splitlines()))
    if changed != EXPECTED_FILES:
        fail(f"unexpected child diff scope: {sorted(changed ^ EXPECTED_FILES)}")

    validate_child_text(child)
    validate_manifest(manifest)

    mutants = {
        "forgery theorem deletion": re.sub(
            r"(?s)theorem universal_known_pair_forgery.*?(?=\n/-- Abstract verifier)", "", child, count=1
        ),
        "proof hole injection": child.replace("exact unbind_right secret mask", "sorry", 1),
        "duplicate BinaryHV model": "abbrev BinaryHV : Type := Nat\n" + child,
    }
    for label, mutant in mutants.items():
        try:
            validate_child_text(mutant)
        except AssertionError:
            pass
        else:
            fail(f"hostile control unexpectedly admitted: {label}")

    mutant_manifest = json.loads(json.dumps(manifest))
    mutant_manifest["stack_parent"]["blob"] = "0" * 40
    try:
        validate_manifest(mutant_manifest)
    except AssertionError:
        pass
    else:
        fail("hostile parent-binding mutation unexpectedly admitted")

    stale_manifest = json.loads(json.dumps(manifest))
    stale_manifest["stack_parent"]["blob"] = INVALIDATED_V1_PARENT_BLOB
    try:
        validate_manifest(stale_manifest)
    except AssertionError:
        pass
    else:
        fail("invalidated v1 lineage unexpectedly admitted")

    print("SYM-HDC-CRYPTO-FV-001A v2 static admission: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SYM-HDC-CRYPTO-FV-001A v2 static admission: FAIL: {exc}", file=sys.stderr)
        raise
