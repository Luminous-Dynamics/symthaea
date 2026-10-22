#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

PARENT = "fbd7a754ea8389ca93f7680d93ed8b48553e6376"
THEOREM = Path("formal/lean/security/XeniaV2Nonce.lean")
MANIFEST = Path("docs/formal/sym-fv-xenia-001-v1.json")
EXPECTED_CHANGED = {
    ".github/workflows/sym-fv-xenia-001.yml",
    "docs/formal/sym-fv-xenia-001-v1.json",
    "formal/lean/security/XeniaV2Nonce.lean",
    "scripts/validate_sym_fv_xenia_001.py",
}
EXPECTED_THEOREM_BLOB = "423b1e7865b107f6e9321f7bbcdd662353979d2d"
REQUIRED_THEOREMS = {
    "xor_fixed_injective",
    "pad64_injective",
    "xor_nonce_fixed_injective",
    "nonce_from_sequence_injective",
    "distinct_sequences_distinct_nonces",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    require(git("merge-base", PARENT, "HEAD") == PARENT, "exact main parent is not an ancestor")
    changed = set(filter(None, git("diff", "--name-only", PARENT, "HEAD").splitlines()))
    require(changed == EXPECTED_CHANGED, f"unexpected child diff scope: {sorted(changed)}")

    actual_blob = git("rev-parse", f"HEAD:{THEOREM}")
    require(actual_blob == EXPECTED_THEOREM_BLOB, f"theorem blob drift: {actual_blob}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest["schema"] == "symthaea.formal.sym-fv-xenia-001.v1", "wrong schema")
    require(manifest["issue"] == 6141, "wrong issue")
    require(manifest["evidence_class"] == "AbstractFormalTheorem", "wrong evidence class")
    require(manifest["exact_parent"]["commit"] == PARENT, "parent drift in manifest")
    require(manifest["theorem_file"]["blob"] == EXPECTED_THEOREM_BLOB, "manifest theorem blob drift")
    require(set(manifest["theorem_file"]["required_theorems"]) == REQUIRED_THEOREMS, "theorem inventory drift")
    target = manifest["cross_repository_design_target"]
    require(target["repository"] == "Luminous-Dynamics/xenia-wire" and target["issue"] == 78, "wrong Xenia design target")
    require("no-production-source-refinement-yet" in target["relationship"], "must not imply production refinement")

    source = THEOREM.read_text(encoding="utf-8")
    lowered = source.lower()
    require("sorry" not in lowered, "forbidden sorry in theorem source")
    require("admit" not in lowered, "forbidden admit in theorem source")
    require("axiom " not in lowered, "forbidden user axiom in theorem source")
    require("symthaea.formal.hdc" not in lowered, "must not import/reuse HDC XOR semantics")

    for marker in (
        "abbrev Bits32 : Type := Bits 32",
        "abbrev Bits64 : Type := Bits 64",
        "structure Nonce96",
        "high32 : Bits32",
        "low64 : Bits64",
        "def pad64",
        "def xorNonce",
        "def nonceFromSequence",
    ):
        require(marker in source, f"missing width/shape marker: {marker}")

    for name in REQUIRED_THEOREMS:
        require(f"theorem {name}" in source, f"missing theorem {name}")
        require(f"#print axioms {name}" in source, f"missing axiom census for {name}")

    nonclaims = set(manifest["nonclaims"])
    for needed in (
        "not-proof-hkdf-outputs-differ",
        "not-byte-order-or-serialization-refinement",
        "not-allocator-no-wrap-proof",
        "not-exclusive-sender-ownership-proof",
        "not-reset-or-recovery-safety-proof",
        "not-aead-primitive-security-proof",
        "not-production-xenia-v2-qualification",
    ):
        require(needed in nonclaims, f"missing nonclaim {needed}")

    require(manifest["abstract_subject"]["sequence_width_bits"] == 64, "wrong sequence width")
    require(manifest["abstract_subject"]["nonce_width_bits"] == 96, "wrong nonce width")
    require(manifest["abstract_subject"]["iv_scope"] == "one exact fixed-IV traffic context", "IV scope escalation")

    print("SYM-FV-XENIA-001 static contract: PASS")


if __name__ == "__main__":
    main()
