#!/usr/bin/env python3
"""Fail-closed static contract for SYM-FV-004A.

This validator does not prove Lean theorems. It freezes subject identity, claim
scope, dependency boundaries, and hostile source mutations before the dedicated
real-Lean/proof-audit lane executes.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

BASE = "a607dc2611dd04b60b91e3f6b74ffd726d5666a9"
PARENT_THEOREM_BLOB = "f724618781b2660d1e76e45a69dcf796a792d76d"
PRODUCTION_BLOB = "22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056"
THEOREM = Path("formal/lean/hdc/BinaryHVHamming.lean")
AUDIT_TEST = Path("crates/bridges/symthaea-lean-bridge/tests/binaryhv_hamming_formal_audit.rs")
CONTRACT = Path("docs/formal/sym-fv-004a-hamming-core-v1.json")
WORKFLOW = Path(".github/workflows/sym-fv-004a-hamming.yml")
SELF = Path("scripts/validate_sym_fv_004a.py")
ALLOWED = {str(THEOREM), str(AUDIT_TEST), str(CONTRACT), str(WORKFLOW), str(SELF)}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_theorem_text(source: str) -> None:
    for forbidden in ("sorry", "admit", "axiom ", "import Mathlib"):
        require(forbidden not in source, f"forbidden Lean construct/dependency: {forbidden!r}")

    independent = "def mismatchVector (a b : BinaryHV) : BinaryHV :=\n  fun i => a i != b i"
    require(independent in source, "mismatchVector must remain independently defined by Bool inequality")

    required = [
        "theorem mismatch_bit",
        "theorem mismatchVector_eq_bind",
        "theorem hamming_eq_bitCount_bind",
        "theorem hamming_symm",
        "theorem mismatch_bind_right_invariant",
        "theorem hamming_bind_right_invariant",
        "theorem hamming_bind_left_invariant",
    ]
    for marker in required:
        require(marker in source, f"missing required theorem marker: {marker}")

    require(
        "mismatchVector (bind a mask) (bind b mask) = mismatchVector a b" in source,
        "common-mask mismatch invariant must use the same exact mask on both operands",
    )
    require(
        "hammingDistance (bind a mask) (bind b mask) = hammingDistance a b" in source,
        "common-mask Hamming isometry statement drifted",
    )
    require(
        "def bitCount (v : BinaryHV) : Nat :=\n  Fin.foldl HdcDimension" in source,
        "bitCount must remain a core-Lean Fin.foldl over the canonical HDC dimension",
    )

    for forbidden_claim in ("theorem similarity", "def similarity", "Float32", " f32"):
        require(forbidden_claim not in source, f"004A may not acquire deferred similarity/floating claim: {forbidden_claim}")


def hostile_controls(source: str) -> None:
    mutants = [
        source.replace("fun i => a i != b i", "fun i => Bool.xor (a i) (b i)", 1),
        source.replace(
            "mismatchVector (bind a mask) (bind b mask) = mismatchVector a b",
            "mismatchVector (bind a mask) (bind b a) = mismatchVector a b",
            1,
        ),
        source.replace("by\n  cases ha", "by\n  sorry\n  cases ha", 1),
        "import Mathlib\n" + source,
    ]
    for idx, mutant in enumerate(mutants, start=1):
        try:
            validate_theorem_text(mutant)
        except AssertionError:
            continue
        raise AssertionError(f"hostile control {idx} produced a false green")


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], check=True)
    changed = set(git("diff", "--name-only", f"{BASE}..HEAD").splitlines())
    require(changed == ALLOWED, f"child scope drift: expected {sorted(ALLOWED)}, got {sorted(changed)}")

    require(
        git("rev-parse", "HEAD:formal/lean/hdc/BinaryHVBind.lean") == PARENT_THEOREM_BLOB,
        "inherited SYM-FV-002 theorem blob drifted",
    )
    require(
        git("rev-parse", "HEAD:crates/core/symthaea-core/src/hdc/binary_hv.rs") == PRODUCTION_BLOB,
        "future production Hamming refinement target drifted",
    )

    source = THEOREM.read_text(encoding="utf-8")
    validate_theorem_text(source)
    hostile_controls(source)

    test = AUDIT_TEST.read_text(encoding="utf-8")
    require(test.count("TheoremAuditCase {") == 8, "expected seven theorem cases plus one hostile mutant")
    require("AxiomPolicy::constitutional()" in test, "Hamming theorem set must use constitutional axiom policy")
    require("create_new(true)" in test, "combined Lean subject must use create-new semantics")
    require("hamming_bind_right_invariant" in test, "audit test must cover common-mask Hamming isometry")
    require("hammingDistance a mask" in test, "audit test must retain wrong-statement hostile control")

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    require(contract["evidence_class"] == "AbstractFormalTheorem", "evidence class drift")
    require(contract["lean_dependency_class"] == "core-lean-only", "dependency class drift")
    require(contract["mathlib_required"] is False, "004A must not silently acquire Mathlib")
    require(contract["stack_parent_head"] == BASE, "stack parent binding drift")
    require(contract["semantic_parent_theorem"]["blob"] == PARENT_THEOREM_BLOB, "parent theorem contract drift")
    require(contract["production_future_refinement_target"]["blob"] == PRODUCTION_BLOB, "production target contract drift")
    require(len(contract["theorems"]) == 7, "theorem census drift")
    require("not-production-rust-refinement" in contract["nonclaims"], "Rust-refinement nonclaim missing")
    require("not-floating-point-similarity-proof" in contract["nonclaims"], "floating-point nonclaim missing")

    print("SYM-FV-004A static contract: PASS")
    print("hostile controls: PASS")
    print(f"exact child paths: {len(changed)}")


if __name__ == "__main__":
    main()
