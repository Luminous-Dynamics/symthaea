#!/usr/bin/env python3
"""Validate the machine-readable Symthaea formal-verification architecture v1.

This validator intentionally proves only contract shape and required claim ceilings.
It does not execute Lean, Aeneas, Verus, TLC, Alloy, or any product code.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "docs/formal/formal_verification_evidence_classes_v1.json"

EXPECTED_CLASSES = [
    "AbstractFormalTheorem",
    "ExtractedSourceRefinement",
    "DeductiveImplementationProof",
    "BoundedModelSafety",
    "TemporalModelEvidence",
    "BoundedTraceConformance",
    "RuntimeQualification",
]

REQUIRED_NON_EQUIVALENCES = {
    "AbstractFormalTheorem != implementation theorem",
    "Aeneas extraction != compiler verification",
    "Aeneas extraction != native binary verification",
    "generated Lean != Lean proof",
    "Lean proof != intended-theorem conformance",
    "Lean proof != axiom-policy acceptance",
    "Verus PASS != rustc correctness",
    "Verus PASS != LLVM correctness",
    "TLC bounded PASS != unbounded proof",
    "Alloy UNSAT != theorem outside the declared scope",
    "crosswalk != behavioral refinement",
    "bounded trace conformance != full refinement",
    "formal-model refinement != Holochain/runtime refinement",
    "formal verification != permission for external effects",
    "proof receipt != runtime authority",
}

EXPECTED_STANDARD_TOOLS = {
    "abstract_math": ["Lean4", "Mathlib4"],
    "rust_to_lean": ["Charon", "Aeneas"],
    "rust_deductive": ["Verus"],
    "distributed_temporal": ["TLA+", "TLC"],
    "structural": ["Alloy"],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FORMAL_VERIFICATION_ARCHITECTURE_V1_FAIL: {message}")


def main() -> None:
    data = json.loads(PROFILE.read_text(encoding="utf-8"))

    require(
        data.get("schema") == "symthaea.formal-verification.evidence-classes.v1",
        "unexpected schema",
    )
    require(data.get("tracking_issue") == 5712, "unexpected tracking issue")
    require(data.get("authority") == "ArchitectureOnly", "authority must remain ArchitectureOnly")

    classes = data.get("classes")
    require(isinstance(classes, list), "classes must be a list")
    class_ids = [entry.get("id") for entry in classes if isinstance(entry, dict)]
    require(class_ids == EXPECTED_CLASSES, f"evidence class order/census drift: {class_ids!r}")
    require(len(set(class_ids)) == len(class_ids), "duplicate evidence class")

    for entry in classes:
        require(isinstance(entry.get("establishes"), str) and entry["establishes"].strip(), f"{entry.get('id')} missing establishes")
        nonclaims = entry.get("does_not_establish")
        require(isinstance(nonclaims, list) and nonclaims, f"{entry.get('id')} missing nonclaims")
        require(len(nonclaims) == len(set(nonclaims)), f"{entry.get('id')} has duplicate nonclaims")

    observed_non_eq = data.get("mandatory_non_equivalences")
    require(isinstance(observed_non_eq, list), "mandatory_non_equivalences must be a list")
    require(len(observed_non_eq) == len(set(observed_non_eq)), "duplicate mandatory non-equivalence")
    missing = REQUIRED_NON_EQUIVALENCES.difference(observed_non_eq)
    require(not missing, f"missing required non-equivalences: {sorted(missing)!r}")

    require(data.get("standard_tools") == EXPECTED_STANDARD_TOOLS, "standard tool ownership drift")

    experimental = data.get("experimental_tools")
    require(isinstance(experimental, dict), "experimental_tools must be an object")
    require(set(experimental) == {"Quint", "Creusot", "Hax"}, "experimental tool census drift")

    serialized = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    print("FORMAL_VERIFICATION_ARCHITECTURE_V1_PASS")
    print(f"profile_sha256={hashlib.sha256(PROFILE.read_bytes()).hexdigest()}")
    print(f"canonical_semantic_sha256={hashlib.sha256(serialized).hexdigest()}")
    print(f"evidence_classes={len(class_ids)}")
    print(f"mandatory_non_equivalences={len(observed_non_eq)}")


if __name__ == "__main__":
    main()
