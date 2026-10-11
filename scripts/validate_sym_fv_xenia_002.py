#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

PARENT = "fbd7a754ea8389ca93f7680d93ed8b48553e6376"
THEOREM = Path("formal/lean/security/XeniaV2SequenceAllocator.lean")
MANIFEST = Path("docs/formal/sym-fv-xenia-002-v1.json")
EXPECTED_CHANGED = {
    ".github/workflows/sym-fv-xenia-002.yml",
    "docs/formal/sym-fv-xenia-002-v1.json",
    "formal/lean/security/XeniaV2SequenceAllocator.lean",
    "scripts/validate_sym_fv_xenia_002.py",
}
EXPECTED_THEOREM_BLOB = "c9c4320876bc5653de294ae7cd10e745d1e3c8d0"
REQUIRED_THEOREMS = {
    "allocate_success_shape",
    "allocate_success_in_range",
    "allocate_success_returns_current",
    "allocate_success_advances_exactly_once",
    "allocate_refuses_when_exhausted",
    "allocate_refuses_at_exact_limit",
    "allocator_state_strictly_monotone_on_success",
    "consecutive_successors",
    "consecutive_successes_are_distinct",
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
    require(manifest["schema"] == "symthaea.formal.sym-fv-xenia-002.v1", "wrong schema")
    require(manifest["issue"] == 6144, "wrong issue")
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
    require("wrapping" not in source[source.find("def allocate"):source.find("theorem allocate_success_shape")].lower(), "authoritative allocator must not use wrapping semantics")

    for marker in (
        "def SequenceLimit : Nat := 2 ^ 64",
        "structure AllocatorState",
        "next : Nat",
        "if s.next < SequenceLimit then",
        "some (s.next, { next := s.next + 1 })",
        "else\n    none",
    ):
        require(marker in source, f"missing allocator marker: {marker}")

    for name in REQUIRED_THEOREMS:
        require(f"theorem {name}" in source, f"missing theorem {name}")
        require(f"#print axioms {name}" in source, f"missing axiom census for {name}")

    require(manifest["abstract_subject"]["sequence_domain"] == "Nat", "abstract sequence domain must remain Nat")
    require(manifest["abstract_subject"]["sequence_limit"] == "2^64", "sequence limit drift")
    require(manifest["future_composition"]["sibling_issue"] == 6141, "wrong sibling proof link")
    require(manifest["future_composition"]["sibling_pr"] == 6142, "wrong sibling PR link")

    nonclaims = set(manifest["nonclaims"])
    for needed in (
        "not-rust-u64-refinement",
        "not-compiled-overflow-or-panic-proof",
        "not-key-iv-reset-lifecycle-proof",
        "not-exclusive-sender-ownership-proof",
        "not-aead-usage-limit-analysis",
        "not-production-xenia-v2-qualification",
    ):
        require(needed in nonclaims, f"missing nonclaim {needed}")

    print("SYM-FV-XENIA-002 static contract: PASS")


if __name__ == "__main__":
    main()
