#!/usr/bin/env python3
"""Static admission gate for SYM-FV-002 BinaryHV bind theorem subject.

This validates subject identity and proof-source structure. Lean typechecking is
performed separately by the dedicated workflow and is the authoritative proof check.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/formal/sym-fv-002-binaryhv-bind-v1.json"
PROOF_PATH = ROOT / "formal/lean/hdc/BinaryHVBind.lean"

EXPECTED_SCHEMA = "symthaea.formal.sym-fv-002-binaryhv-bind.v1"
EXPECTED_ISSUE = 5715
EXPECTED_SOURCE_COMMIT = "458c7b98d81c64b9361e252f85ef9d45132e6682"
EXPECTED_SOURCE_PATH = "crates/core/symthaea-core/src/hdc/binary_hv.rs"
EXPECTED_SOURCE_BLOB = "22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056"
EXPECTED_DIMENSION = 16384
EXPECTED_BYTES = 2048
REQUIRED_THEOREMS = (
    "bit_bind",
    "bind_zero_right",
    "bind_zero_left",
    "bind_self_inverse",
    "bind_comm",
    "bind_assoc",
    "unbind_right",
    "unbind_left",
)


class ContractError(RuntimeError):
    pass


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_contract(manifest: dict, proof: str, observed_source_blob: str) -> None:
    if manifest.get("schema") != EXPECTED_SCHEMA:
        raise ContractError("wrong schema")
    if manifest.get("issue") != EXPECTED_ISSUE:
        raise ContractError("wrong issue binding")
    if manifest.get("evidence_class") != "AbstractFormalTheorem":
        raise ContractError("wrong evidence class")

    source = manifest.get("production_refinement_target", {})
    expected_source = {
        "commit": EXPECTED_SOURCE_COMMIT,
        "path": EXPECTED_SOURCE_PATH,
        "blob": EXPECTED_SOURCE_BLOB,
        "function": "BinaryHV::bind_scalar",
    }
    for key, value in expected_source.items():
        if source.get(key) != value:
            raise ContractError(f"wrong production source {key}")
    if source.get("relationship") != "future-refinement-target-not-proved-here":
        raise ContractError("source relationship overclaims refinement")
    if observed_source_blob != EXPECTED_SOURCE_BLOB:
        raise ContractError("production source blob drifted")

    subject = manifest.get("abstract_subject", {})
    if subject.get("dimension_bits") != EXPECTED_DIMENSION:
        raise ContractError("wrong abstract bit dimension")
    if subject.get("production_bytes") != EXPECTED_BYTES:
        raise ContractError("wrong production byte count")
    if subject.get("representation") != "Fin 16384 -> Bool":
        raise ContractError("wrong abstract representation")
    if subject.get("binding") != "pointwise Bool.xor":
        raise ContractError("wrong abstract binding semantics")

    if re.search(r"\b(sorry|admit)\b", proof):
        raise ContractError("proof hole token present")
    if re.search(r"(?m)^\s*(axiom|opaque)\s+", proof):
        raise ContractError("user-declared axiom/opaque declaration present")
    if "import Mathlib" in proof:
        raise ContractError("unexpected Mathlib dependency")
    if not re.search(r"def\s+HdcDimension\s*:\s*Nat\s*:=\s*16_384\b", proof):
        raise ContractError("exact HDC dimension declaration missing")
    if not re.search(r"abbrev\s+BinaryHV\s*:\s*Type\s*:=\s*BitIndex\s*→\s*Bool", proof):
        raise ContractError("BinaryHV abstract representation missing")
    if not re.search(r"def\s+bind\s*\(a b : BinaryHV\).*Bool\.xor", proof, re.S):
        raise ContractError("pointwise XOR bind definition missing")

    for theorem in REQUIRED_THEOREMS:
        if not re.search(rf"\btheorem\s+{re.escape(theorem)}\b", proof):
            raise ContractError(f"required theorem missing: {theorem}")
        if f"#print axioms {theorem}" not in proof:
            raise ContractError(f"axiom census directive missing: {theorem}")

    nonclaims = set(manifest.get("nonclaims", []))
    required_nonclaims = {
        "abstract-theorem-is-not-rust-refinement",
        "abstract-theorem-is-not-simd-proof",
        "abstract-theorem-is-not-compiler-or-binary-proof",
        "abstract-hdc-law-is-not-empirical-cognitive-evidence",
    }
    if not required_nonclaims.issubset(nonclaims):
        raise ContractError("required claim ceiling missing")


def expect_reject(label: str, manifest: dict, proof: str, observed_source_blob: str) -> None:
    try:
        validate_contract(manifest, proof, observed_source_blob)
    except ContractError:
        print(f"negative control {label}: REJECTED")
        return
    raise ContractError(f"negative control false-green: {label}")


def main() -> int:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        proof = PROOF_PATH.read_text(encoding="utf-8")
        observed = git("rev-parse", f"HEAD:{EXPECTED_SOURCE_PATH}")
        historical = git("rev-parse", f"{EXPECTED_SOURCE_COMMIT}:{EXPECTED_SOURCE_PATH}")
        if historical != EXPECTED_SOURCE_BLOB:
            raise ContractError("historical source binding no longer resolves to expected blob")

        validate_contract(manifest, proof, observed)
        print("SYM-FV-002 static contract: PASS")

        mutant = copy.deepcopy(manifest)
        mutant["abstract_subject"]["dimension_bits"] = 8192
        expect_reject("dimension-drift", mutant, proof, observed)

        mutant = copy.deepcopy(manifest)
        mutant["production_refinement_target"]["blob"] = "0" * 40
        expect_reject("source-rebinding", mutant, proof, observed)

        expect_reject(
            "theorem-deletion",
            manifest,
            proof.replace("theorem bind_assoc", "theorem bind_assoc_missing", 1),
            observed,
        )
        expect_reject("proof-hole", manifest, proof + "\nexample : True := by sorry\n", observed)
        print("SYM-FV-002 negative controls: PASS")
        return 0
    except (ContractError, json.JSONDecodeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"SYM-FV-002 validation: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
