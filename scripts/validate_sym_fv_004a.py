#!/usr/bin/env python3
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROOF = ROOT / "formal/lean/hdc/HammingSemantics.lean"
PARENT = ROOT / "formal/lean/hdc/BinaryHVBind.lean"
MANIFEST = ROOT / "docs/formal/sym-fv-004a-hamming-semantics-v1.json"
SOURCE = ROOT / "crates/core/symthaea-core/src/hdc/binary_hv.rs"

EXPECTED_PARENT_BLOB = "259ba64888d8492bafc123b72d67b70da9282c36"
EXPECTED_SOURCE_BLOB = "22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056"
BASE = "a4d4506c57902f96b6c260e7e9c56f9fc2e654c3"
EXPECTED_CHILD_PATHS = {
    ".github/workflows/sym-fv-004a-hamming-semantics.yml",
    "docs/formal/sym-fv-004a-hamming-semantics-v1.json",
    "formal/lean/hdc/HammingSemantics.lean",
    "scripts/validate_sym_fv_004a.py",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def main() -> None:
    text = PROOF.read_text(encoding="utf-8")
    parent = PARENT.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    require(git("rev-parse", f"HEAD:{PARENT.relative_to(ROOT)}") == EXPECTED_PARENT_BLOB,
            "parent theorem blob drift")
    require(git("rev-parse", f"{BASE}:{SOURCE.relative_to(ROOT)}") == EXPECTED_SOURCE_BLOB,
            "production scalar source blob drift")

    changed = set(git("diff", "--name-only", BASE, "HEAD").splitlines())
    require(changed == EXPECTED_CHILD_PATHS,
            f"unexpected 004A diff scope: {sorted(changed)}")

    for forbidden in ("sorry", "admit", "axiom ", "opaque "):
        require(forbidden not in text, f"forbidden proof escape present: {forbidden!r}")

    required = [
        "def bitCount",
        "def bitAtNat",
        "def popcountPrefix",
        "def popcount",
        "def hammingDistance",
        "theorem hamming_eq_popcount_bind",
        "theorem popcount_zero",
        "theorem hamming_self",
        "theorem hamming_comm",
        "theorem common_left_bind_pairwise_xor",
        "theorem hamming_common_left_bind_invariant",
        "theorem hamming_common_right_bind_invariant",
        "hammingDistance a b = popcount (bind a b)",
        "hammingDistance (bind mask a) (bind mask b) = hammingDistance a b",
    ]
    for marker in required:
        require(marker in text, f"missing theorem/definition marker: {marker}")

    # Do not allow this child to shadow the inherited semantic root.
    for marker in ("def HdcDimension", "abbrev BitIndex", "abbrev BinaryHV", "def bind", "def zero"):
        require(marker not in text, f"004A must not redefine inherited subject: {marker}")
        require(marker in parent, f"parent subject unexpectedly lacks: {marker}")

    # Ensure the production facts that motivate later refinement still exist,
    # without claiming this abstract theorem proves those Rust functions.
    for marker in (
        "pub fn hamming_distance_scalar",
        "(a ^ b).count_ones()",
        "pub fn similarity_scalar",
        "matching_bits as f32 / Self::DIM as f32",
    ):
        require(marker in source, f"production scalar semantics marker missing: {marker}")

    require(manifest["evidence_class"] == "AbstractFormalTheorem", "wrong evidence class")
    require(manifest["parent"]["theorem_blob"] == EXPECTED_PARENT_BLOB, "manifest parent drift")
    require(manifest["production_future_refinement_target"]["blob"] == EXPECTED_SOURCE_BLOB,
            "manifest production blob drift")
    require(manifest["dependency_policy"]["mathlib_required"] is False,
            "004A unexpectedly requires Mathlib")

    # Hostile mutations: each must violate a structural contract checked above.
    mutants = {
        "different-mask": text.replace(
            "hammingDistance (bind mask a) (bind mask b) = hammingDistance a b",
            "hammingDistance (bind mask a) (bind zero b) = hammingDistance a b",
            1,
        ),
        "wrong-rhs": text.replace(
            "hammingDistance a b = popcount (bind a b)",
            "hammingDistance a b = 0",
            1,
        ),
        "proof-hole": text.replace("by\n  rfl", "by\n  sorry", 1),
    }
    require(required[-1] not in mutants["different-mask"], "different-mask mutant was not effective")
    require(required[-2] not in mutants["wrong-rhs"], "wrong-RHS mutant was not effective")
    require("sorry" in mutants["proof-hole"], "proof-hole mutant was not effective")

    print("SYM-FV-004A static contract: PASS")
    print("SYM-FV-004A hostile controls: PASS")


if __name__ == "__main__":
    main()
