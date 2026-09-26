#!/usr/bin/env python3
"""Fail-closed validator for the verification closure matrix."""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/formal/verification-closure-matrix-v1.json"

EXPECTED_STATES = {
    "Closed",
    "Open",
    "NotApplicable",
    "ImportedAssurance",
    "BoundedOnly",
    "Unknown",
}
EXPECTED_RESULTS = {"Pass", "Fail", "Blocked", "EnvironmentFailure"}
EXPECTED_DIMENSIONS = {
    "mathematical_spec",
    "source_refinement",
    "bounded_implementation_safety",
    "distributed_model",
    "compiler_assurance",
    "dependency_closure",
    "build_identity",
    "artifact_identity",
    "release_identity",
    "runtime_identity",
    "cross_repo_subject_closure",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate(matrix: dict) -> None:
    require(matrix["schema"] == "symthaea.formal.verification-closure-matrix.v1", "schema drift")
    require(matrix["tracking_issue"] == 6027, "tracking issue drift")
    require(matrix["authority"] == "EvidenceOnly", "coverage matrix may not acquire runtime authority")
    require(set(matrix["closure_states"]) == EXPECTED_STATES, "closure-state vocabulary drift")
    require(set(matrix["dimensions"]) == EXPECTED_DIMENSIONS, "closure-dimension census drift")

    seen: set[str] = set()
    for obligation in matrix["obligations"]:
        oid = obligation["id"]
        require(oid not in seen, f"duplicate obligation id: {oid}")
        seen.add(oid)

        result = obligation["current_qualification"]["result"]
        require(result in EXPECTED_RESULTS, f"noncanonical qualification result for {oid}: {result}")
        closure = obligation["closure"]
        require(set(closure) == EXPECTED_DIMENSIONS, f"closure vector incomplete for {oid}")
        for dimension, state in closure.items():
            require(state in EXPECTED_STATES, f"invalid closure state {oid}.{dimension}={state}")

        closed = [d for d, state in closure.items() if state == "Closed"]
        if closed:
            require(result == "Pass", f"Closed dimensions require current Pass evidence: {oid}: {closed}")
            require(obligation["evidence_refs"], f"Closed dimensions require evidence references: {oid}")

        # Bounded model evidence must remain visibly bounded and must never be
        # used as a spelling alias for universal closure.
        if closure["bounded_implementation_safety"] == "BoundedOnly":
            require("bounded" in " ".join(obligation.get("assumptions", [])).lower()
                    or any("bounded" in claim.lower() for claim in obligation.get("nonclaims", [])),
                    f"BoundedOnly state must expose its bound/nonclaim: {oid}")

        # Cross-repository closure is never inferred from source/refinement state.
        if closure["cross_repo_subject_closure"] == "Closed":
            require(any("repo" in ref.lower() or "subject" in ref.lower()
                        for ref in obligation["evidence_refs"]),
                    f"cross-repo closure requires explicit subject/repository evidence: {oid}")

        require(obligation.get("nonclaims"), f"nonclaims required: {oid}")


def expect_reject(label: str, mutant: dict) -> None:
    try:
        validate(mutant)
    except Exception:
        return
    raise AssertionError(f"hostile mutant unexpectedly accepted: {label}")


def self_test(matrix: dict) -> None:
    # A blocked candidate cannot claim mathematical closure.
    mutant = json.loads(json.dumps(matrix))
    mutant["obligations"][0]["closure"]["mathematical_spec"] = "Closed"
    expect_reject("blocked-evidence-closes-math", mutant)

    # Tool-specific result strings are not canonical evidence states.
    mutant = json.loads(json.dumps(matrix))
    mutant["obligations"][0]["current_qualification"]["result"] = "LeanTypechecked"
    expect_reject("tool-outcome-as-result", mutant)

    # Missing dimensions are not silently interpreted as Open.
    mutant = json.loads(json.dumps(matrix))
    del mutant["obligations"][0]["closure"]["source_refinement"]
    expect_reject("missing-dimension", mutant)

    # Coverage metadata never gains action/runtime authority.
    mutant = json.loads(json.dumps(matrix))
    mutant["authority"] = "RuntimeAuthority"
    expect_reject("authority-escalation", mutant)

    # Duplicate obligation IDs fail closed.
    mutant = json.loads(json.dumps(matrix))
    mutant["obligations"].append(json.loads(json.dumps(mutant["obligations"][0])))
    expect_reject("duplicate-obligation", mutant)


def main() -> int:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    validate(matrix)
    self_test(matrix)
    print("verification_closure_matrix=PASS")
    print(f"obligations={len(matrix['obligations'])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"verification_closure_matrix=FAIL: {exc}", file=sys.stderr)
        raise
