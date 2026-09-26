#!/usr/bin/env python3
import copy
import json
import subprocess
from pathlib import Path

PARENT = "a4d4506c57902f96b6c260e7e9c56f9fc2e654c3"
PARENT_SUBJECT = "formal/lean/hdc/BinaryHVBind.lean"
PARENT_BLOB = "259ba64888d8492bafc123b72d67b70da9282c36"
CORE_TARGET = "crates/core/symthaea-core/src/hdc/binary_hv.rs"
CORE_BLOB = "22a56cfaedf5b0cb7d8ff8b73c41ed4caf8d7056"
CRYPTO_TARGET = "crates/core/symthaea-hdc-crypto/src/binary_hv.rs"
CRYPTO_BLOB = "9fe5409c3edfcdc6112a61b125c1b278f9eb0d4a"
LEAN = "formal/lean/hdc/PermutationAction.lean"
MANIFEST = "docs/formal/sym-fv-002p-a-permutation-action-v1.json"
VALIDATOR = "scripts/validate_sym_fv_002p_a.py"
WORKFLOW = ".github/workflows/sym-fv-002p-a-permutation-action.yml"
EXPECTED_DIFF = {LEAN, MANIFEST, VALIDATOR, WORKFLOW}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def validate_contract(lean: str, manifest: dict, core_source: str, crypto_source: str) -> None:
    require("sorry" not in lean and "admit" not in lean, "proof hole present")
    for forbidden in [
        "abbrev BinaryHV",
        "def BinaryHV",
        "def bind (",
        "def zero : BinaryHV",
        "def HdcDimension",
    ]:
        require(forbidden not in lean, f"inherited semantic root redefined: {forbidden}")

    required = [
        "structure IndexPermutation where",
        "left_inv : ∀ i, inverse (forward i) = i",
        "right_inv : ∀ i, forward (inverse i) = i",
        "def permuteBy",
        "theorem permuteBy_identity",
        "theorem permuteBy_zero",
        "theorem permuteBy_bind",
        "theorem permuteBy_compose",
        "theorem inverse_after_permute",
        "theorem permute_after_inverse",
        "theorem permuteBy_injective",
        "theorem permuteBy_surjective",
        "theorem permuteBy_bijective",
        "#print axioms permuteBy_bijective",
    ]
    for token in required:
        require(token in lean, f"missing theorem/contract token: {token}")

    require(manifest["evidence_class"] == "AbstractFormalTheorem", "evidence class escalated")
    require(manifest["semantic_layer"] == "generic-index-permutation-action", "cyclic layer collapsed into generic proof")
    require(manifest["stack_parent_head"] == PARENT, "parent head drift")
    require(manifest["inherited_subject"]["blob"] == PARENT_BLOB, "parent theorem blob drift")
    require(manifest["cyclic_instantiation_issue"] == 5937, "cyclic instantiation dependency lost")

    targets = {entry["path"]: entry for entry in manifest["future_production_targets"]}
    require(targets[CORE_TARGET]["blob"] == CORE_BLOB, "core production target rebound")
    require(targets[CRYPTO_TARGET]["blob"] == CRYPTO_BLOB, "crypto production target rebound")
    require(
        targets[CORE_TARGET]["relationship"] == "future-refinement-target-not-proved-here",
        "core target promoted to refinement",
    )
    require(
        targets[CRYPTO_TARGET]["relationship"]
        == "separate-future-refinement-target-not-proved-equivalent",
        "duplicate implementation equivalence overclaimed",
    )

    nonclaims = set(manifest["nonclaims"])
    for item in {
        "not-cyclic-offset-instantiation",
        "not-production-rust-refinement",
        "not-equivalence-of-production-implementations",
        "not-cryptographic-security",
    }:
        require(item in nonclaims, f"missing nonclaim: {item}")

    require("pub fn permute" in core_source, "core production permutation target disappeared")
    require("pub fn permute" in crypto_source, "crypto production permutation target disappeared")


def expect_reject(name: str, lean: str, manifest: dict, core_source: str, crypto_source: str) -> None:
    try:
        validate_contract(lean, manifest, core_source, crypto_source)
    except (AssertionError, KeyError):
        return
    raise AssertionError(f"hostile mutation unexpectedly accepted: {name}")


def main() -> None:
    require(git("merge-base", PARENT, "HEAD") == PARENT, "repaired SYM-FV-002 head is not ancestor")
    changed = set(filter(None, git("diff", "--name-only", f"{PARENT}...HEAD").splitlines()))
    require(changed == EXPECTED_DIFF, f"unexpected child scope: {sorted(changed)}")
    require(git("rev-parse", f"HEAD:{PARENT_SUBJECT}") == PARENT_BLOB, "inherited BinaryHV theorem blob drift")
    require(git("rev-parse", f"HEAD:{CORE_TARGET}") == CORE_BLOB, "core BinaryHV source blob drift")
    require(git("rev-parse", f"HEAD:{CRYPTO_TARGET}") == CRYPTO_BLOB, "crypto BinaryHV source blob drift")

    lean = Path(LEAN).read_text(encoding="utf-8")
    manifest = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
    core_source = Path(CORE_TARGET).read_text(encoding="utf-8")
    crypto_source = Path(CRYPTO_TARGET).read_text(encoding="utf-8")
    validate_contract(lean, manifest, core_source, crypto_source)

    expect_reject(
        "remove-left-inverse",
        lean.replace("left_inv : ∀ i, inverse (forward i) = i", "left_inv_REMOVED : ∀ i, inverse (forward i) = i", 1),
        manifest,
        core_source,
        crypto_source,
    )
    expect_reject(
        "delete-xor-homomorphism",
        lean.replace("theorem permuteBy_bind", "theorem permuteBy_bind_REMOVED", 1),
        manifest,
        core_source,
        crypto_source,
    )
    expect_reject(
        "delete-bijectivity",
        lean.replace("theorem permuteBy_bijective", "theorem permuteBy_bijective_REMOVED", 1),
        manifest,
        core_source,
        crypto_source,
    )
    expect_reject(
        "redefine-binaryhv",
        "abbrev BinaryHV : Type := BitIndex → Bool\n" + lean,
        manifest,
        core_source,
        crypto_source,
    )

    hostile = copy.deepcopy(manifest)
    hostile["semantic_layer"] = "cyclic-fin-16384-production-refinement"
    expect_reject("cyclic-layer-collapse", lean, hostile, core_source, crypto_source)

    hostile = copy.deepcopy(manifest)
    hostile["future_production_targets"][0]["relationship"] = "proved-rust-refinement"
    expect_reject("production-refinement-escalation", lean, hostile, core_source, crypto_source)

    hostile = copy.deepcopy(manifest)
    hostile["future_production_targets"][1]["relationship"] = "proved-equivalent-to-core"
    expect_reject("duplicate-equivalence-escalation", lean, hostile, core_source, crypto_source)

    hostile = copy.deepcopy(manifest)
    hostile["inherited_subject"]["blob"] = "0" * 40
    expect_reject("parent-blob-rebind", lean, hostile, core_source, crypto_source)

    print("SYM-FV-002P-A static contract and hostile controls: PASS")


if __name__ == "__main__":
    main()
