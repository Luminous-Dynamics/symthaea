#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

PARENT = "b485a5713e5e8d971342a93f1a6e514512052b9e"
THEOREM = Path("formal/lean/hdc/HdcThresholdShareLeakage.lean")
MANIFEST = Path("docs/formal/sym-hdc-crypto-fv-001c-v1.json")
EXPECTED_CHANGED = {
    ".github/workflows/sym-hdc-crypto-fv-001c.yml",
    "docs/formal/sym-hdc-crypto-fv-001c-v1.json",
    "formal/lean/hdc/HdcThresholdShareLeakage.lean",
    "scripts/validate_sym_hdc_crypto_fv_001c.py",
}
EXPECTED_BLOBS = {
    "formal/lean/hdc/BinaryHVBind.lean": "259ba64888d8492bafc123b72d67b70da9282c36",
    "formal/lean/hdc/HdcCryptoAttacks.lean": "67d44281d03252101eb01d6ac9713ea7638044de",
    "formal/lean/hdc/HdcThresholdShareLeakage.lean": "0b063aa74447560176ff14db17d5e2fd3e234eab",
    "crates/core/symthaea-hdc-crypto/src/crypto.rs": "59c0ea6bc92fd1d67a9f764bb396918fd823331a",
    "crates/core/symthaea-core/src/hdc/hdc_crypto.rs": "a232b0b699f5d0301fe052b0e013f289194c13b6",
}
REQUIRED_THEOREMS = {
    "indexed_share_recovers_secret",
    "one_share_recovery_ignores_threshold_declaration",
    "threshold_gt_one_still_allows_one_share_recovery",
    "share_index_is_irrelevant_to_recovery",
}
FORBIDDEN_LOCAL_REDEFINITIONS = (
    "abbrev BinaryHV",
    "def bind ",
    "structure AbstractShare",
    "def makeShare ",
    "def recoverOne ",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    require(git("merge-base", PARENT, "HEAD") == PARENT, "exact #5905 parent is not an ancestor")

    changed = set(filter(None, git("diff", "--name-only", PARENT, "HEAD").splitlines()))
    require(changed == EXPECTED_CHANGED, f"unexpected child diff scope: {sorted(changed)}")

    for path, expected in EXPECTED_BLOBS.items():
        actual = git("rev-parse", f"HEAD:{path}")
        require(actual == expected, f"blob drift for {path}: {actual} != {expected}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest["schema"] == "symthaea.formal.sym-hdc-crypto-fv-001c.v1", "wrong schema")
    require(manifest["issue"] == 6126, "wrong issue")
    require(manifest["parent_issue"] == 5844, "wrong parent issue")
    require(manifest["evidence_class"] == "AbstractFormalTheorem", "wrong evidence class")
    require(manifest["stack_parent"]["commit"] == PARENT, "manifest parent drift")
    require(manifest["theorem_file"]["blob"] == EXPECTED_BLOBS[str(THEOREM)], "theorem blob not frozen")
    require(set(manifest["theorem_file"]["required_theorems"]) == REQUIRED_THEOREMS, "theorem inventory drift")
    require(manifest["production_refinement_target"]["blob"] == EXPECTED_BLOBS["crates/core/symthaea-hdc-crypto/src/crypto.rs"], "standalone source drift")
    require(manifest["compatibility_inventory"]["blob"] == EXPECTED_BLOBS["crates/core/symthaea-core/src/hdc/hdc_crypto.rs"], "compat source drift")
    require("no-automatic-proof-transfer" in manifest["compatibility_inventory"]["relationship"], "compatibility copy must remain non-authoritative")

    source = THEOREM.read_text(encoding="utf-8")
    for name in REQUIRED_THEOREMS:
        require(f"theorem {name}" in source, f"missing theorem {name}")
        require(f"#print axioms {name}" in source, f"missing axiom census for {name}")

    lowered = source.lower()
    require("sorry" not in lowered, "forbidden sorry in theorem source")
    require("admit" not in lowered, "forbidden admit in theorem source")
    require("axiom " not in lowered, "forbidden user axiom in theorem source")
    for marker in FORBIDDEN_LOCAL_REDEFINITIONS:
        require(marker not in source, f"forbidden local redefinition: {marker}")

    require("open Symthaea.Formal.HDC.CryptoAttacks" in source, "must reuse inherited attack/share algebra")
    require("splitAdmissible" in source and "k % 2 = 1" in source, "split precondition not modeled")
    require("1 < k" in source, "k>1 theorem sensitivity missing")

    prod = Path("crates/core/symthaea-hdc-crypto/src/crypto.rs").read_text(encoding="utf-8")
    for marker in (
        "pub struct HdcShare",
        "pub index: usize",
        "pub share: BinaryHV",
        "pub mask: BinaryHV",
        "assert!(k >= 1",
        "assert!(k <= n",
        "assert!(k % 2 == 1",
        "let share = secret.bind(&mask)",
    ):
        require(marker in prod, f"production binding marker missing: {marker}")

    compat = Path("crates/core/symthaea-core/src/hdc/hdc_crypto.rs").read_text(encoding="utf-8")
    require("pub struct HdcShare" in compat and "pub mask: BinaryHV" in compat, "compatibility inventory no longer matches recorded duplicate shape")

    nonclaims = set(manifest["nonclaims"])
    for needed in (
        "not-production-rust-refinement",
        "not-proof-all-threshold-cryptography-is-insecure",
        "not-compatibility-copy-refinement",
        "not-secure-replacement",
        "not-production-cryptographic-authority",
    ):
        require(needed in nonclaims, f"missing nonclaim {needed}")

    print("SYM-HDC-CRYPTO-FV-001C static contract: PASS")


if __name__ == "__main__":
    main()
