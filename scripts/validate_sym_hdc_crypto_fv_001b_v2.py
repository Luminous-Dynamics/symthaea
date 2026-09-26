#!/usr/bin/env python3
"""Static admission gate for SYM-HDC-CRYPTO-FV-001B v2."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT_COMMIT = "b485a5713e5e8d971342a93f1a6e514512052b9e"
STALE_PARENT_COMMIT = "604388ab2d3b59cccf34e456342a7376f000ae8a"
INHERITED = {
    "formal/lean/hdc/BinaryHVBind.lean": "259ba64888d8492bafc123b72d67b70da9282c36",
    "formal/lean/hdc/HdcCryptoAttacks.lean": "67d44281d03252101eb01d6ac9713ea7638044de",
}
FHE_PATH = "crates/core/symthaea-hdc-crypto/src/fhe.rs"
FHE_BLOB = "bd67fdc42dc9ed598b2c93f80995324db066c9a7"
CHILD_REL = "formal/lean/hdc/HdcCryptoCommonMask.lean"
CHILD_BLOB = "f100281fd520cbb8014a48561c7ccd24f9b968b8"
CHILD_PATH = ROOT / CHILD_REL
MANIFEST_PATH = ROOT / "docs/formal/sym-hdc-crypto-fv-001b-v2.json"
EXPECTED_SCHEMA = "symthaea.formal.sym-hdc-crypto-fv-001b.v2"
EXPECTED_FILES = {
    ".github/workflows/sym-hdc-crypto-fv-001b-v2.yml",
    "docs/formal/sym-hdc-crypto-fv-001b-v2.json",
    CHILD_REL,
    "scripts/validate_sym_hdc_crypto_fv_001b_v2.py",
}
REQUIRED_THEOREMS = {
    "common_mask_preserves_pairwise_xor",
    "common_mask_leaks_pairwise_relation",
    "common_mask_preserves_pairwise_xor_swapped",
}
REQUIRED_CEILING = {
    "abstract-common-mask-pairwise-xor-only",
    "not-hamming-theorem-in-this-line",
    "not-production-rust-refinement",
    "not-semantic-security-proof",
    "not-proof-all-hdc-cryptography-insecure",
    "not-compiler-or-native-binary-proof",
    "not-runtime-authority",
}
FORBIDDEN_REDEFS = ("HdcDimension", "BitIndex", "BinaryHV", "bind", "zero")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def fail(msg: str) -> None:
    raise AssertionError(msg)


def validate_text(text: str) -> None:
    names = set(re.findall(r"(?m)^theorem\s+([A-Za-z0-9_']+)", text))
    missing = REQUIRED_THEOREMS - names
    if missing:
        fail(f"missing theorem(s): {sorted(missing)}")
    for name in REQUIRED_THEOREMS:
        if f"#print axioms {name}" not in text:
            fail(f"missing axiom census for {name}")
    for symbol in FORBIDDEN_REDEFS:
        if re.search(rf"(?m)^(?:def|abbrev|structure|inductive)\s+{re.escape(symbol)}\b", text):
            fail(f"semantic-root redefinition: {symbol}")
    if re.search(r"\b(?:sorry|admit)\b", text.lower()):
        fail("proof-hole token present")
    if re.search(r"(?m)^\s*(?:axiom|opaque)\s+", text):
        fail("child introduces axiom/opaque authority")
    if "import Mathlib" in text:
        fail("unexpected Mathlib dependency")
    if "maskWith x mask) (maskWith y mask)" not in text:
        fail("common-mask theorem no longer uses the same mask twice")
    if "= bind x y" not in text:
        fail("pairwise-XOR theorem RHS drift")


def validate_manifest(m: dict) -> None:
    if m.get("schema") != EXPECTED_SCHEMA:
        fail("schema drift")
    stack = m.get("stack_parent", {})
    if stack.get("pr") != 5905 or stack.get("commit") != PARENT_COMMIT:
        fail("stack parent drift")
    if stack.get("superseded_commit") != STALE_PARENT_COMMIT:
        fail("superseded parent census drift")
    if stack.get("commit") == STALE_PARENT_COMMIT:
        fail("stale stack-parent head reintroduced")
    inherited = {x["path"]: x["blob"] for x in m.get("inherited_subjects", [])}
    if inherited != INHERITED:
        fail("inherited subject binding drift")
    theorem_file = m.get("theorem_file", {})
    if theorem_file.get("path") != CHILD_REL or theorem_file.get("blob") != CHILD_BLOB:
        fail("child theorem-file binding drift")
    target = m.get("production_refinement_target", {})
    if target.get("commit") != PARENT_COMMIT:
        fail("FHE source commit drift")
    if target.get("path") != FHE_PATH or target.get("blob") != FHE_BLOB:
        fail("FHE source binding drift")
    if set(m.get("claim_ceiling", [])) != REQUIRED_CEILING:
        fail("claim ceiling drift")
    metric = m.get("canonical_metric_dependency", {})
    if metric.get("issue") != 5717 or metric.get("pr") != 5903:
        fail("canonical metric dependency drift")
    qc = m.get("qualification_contract", {})
    for key in (
        "exact_pr_head_checkout",
        "exact_stack_parent_head_binding",
        "known_bad_lean_subject_must_fail",
        "prover_exit_must_propagate",
        "sorryAx_forbidden",
        "lean_error_markers_forbidden",
        "repository_postflight_clean",
    ):
        if qc.get(key) is not True:
            fail(f"qualification invariant disabled: {key}")


def expect_reject(label: str, fn) -> None:
    try:
        fn()
    except AssertionError:
        print(f"negative control {label}: REJECTED")
        return
    fail(f"hostile control unexpectedly admitted: {label}")


def main() -> int:
    text = CHILD_PATH.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if git("merge-base", PARENT_COMMIT, "HEAD") != PARENT_COMMIT:
        fail("exact strengthened stack parent is not an ancestor of child HEAD")
    for path, blob in INHERITED.items():
        if git("rev-parse", f"{PARENT_COMMIT}:{path}") != blob:
            fail(f"parent blob mismatch: {path}")
        if git("rev-parse", f"HEAD:{path}") != blob:
            fail(f"child inherited blob drift: {path}")
    if git("rev-parse", f"{PARENT_COMMIT}:{FHE_PATH}") != FHE_BLOB:
        fail("parent FHE blob mismatch")
    if git("rev-parse", f"HEAD:{FHE_PATH}") != FHE_BLOB:
        fail("child FHE source drift")
    if git("rev-parse", f"HEAD:{CHILD_REL}") != CHILD_BLOB:
        fail("child theorem blob drift")

    changed = set(filter(None, git("diff", "--name-only", PARENT_COMMIT, "HEAD").splitlines()))
    if changed != EXPECTED_FILES:
        fail(f"unexpected child diff scope: {sorted(changed ^ EXPECTED_FILES)}")

    validate_text(text)
    validate_manifest(manifest)

    text_mutants = {
        "different second mask": text.replace("maskWith y mask)", "maskWith y otherMask)", 1),
        "wrong RHS": text.replace("= bind x y := by", "= zero := by", 1),
        "theorem deletion": re.sub(
            r"(?s)theorem common_mask_preserves_pairwise_xor\n.*?(?=\n/-- Equality of pairwise XOR relations)",
            "",
            text,
            count=1,
        ),
        "proof hole": text.replace("exact common_mask_preserves_pairwise_xor y x mask", "sorry", 1),
        "semantic root duplicate": "abbrev BinaryHV : Type := Nat\n" + text,
    }
    for label, mutant in text_mutants.items():
        expect_reject(label, lambda mutant=mutant: validate_text(mutant))

    stale_manifest = json.loads(json.dumps(manifest))
    stale_manifest["stack_parent"]["commit"] = STALE_PARENT_COMMIT
    expect_reject("stale stack parent", lambda: validate_manifest(stale_manifest))

    theorem_manifest = json.loads(json.dumps(manifest))
    theorem_manifest["theorem_file"]["blob"] = "0" * 40
    expect_reject("theorem file rebinding", lambda: validate_manifest(theorem_manifest))

    source_manifest = json.loads(json.dumps(manifest))
    source_manifest["production_refinement_target"]["blob"] = "0" * 40
    expect_reject("source rebinding", lambda: validate_manifest(source_manifest))

    claim_manifest = json.loads(json.dumps(manifest))
    claim_manifest["claim_ceiling"].append("production-rust-refinement-proved")
    expect_reject("claim escalation", lambda: validate_manifest(claim_manifest))

    print("SYM-HDC-CRYPTO-FV-001B v2 static admission: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SYM-HDC-CRYPTO-FV-001B v2 static admission: FAIL: {exc}", file=sys.stderr)
        raise
