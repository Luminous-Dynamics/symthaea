#!/usr/bin/env python3
"""Fail-closed validator for Symthaea formal closure-vector v1.

This validator checks the semantic state contract without treating a closure
manifest as proof. It intentionally has no third-party dependencies.
"""

from __future__ import annotations

import copy
import json
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs/formal/formal_closure_vector_v1.schema.json"
EXAMPLES_PATH = ROOT / "docs/formal/formal_closure_vector_v1.examples.json"

SCHEMA = "symthaea.formal.closure-vector.v1"
STATES = {"Closed", "Open", "NotApplicable", "ImportedAssurance", "BoundedOnly", "Unknown"}
CORE_DIMENSIONS = [
    "TheoremStatementIdentity",
    "MathematicalSpecification",
    "CheckerIndependence",
    "SourceRefinement",
    "OptimizedImplementationRelation",
    "CompilerSemantics",
    "DependencyClosure",
    "BuildIdentity",
    "ArtifactIdentity",
    "ReleaseIdentity",
    "RuntimeIdentity",
    "EnvironmentRealization",
    "DistributedAssumptions",
    "CrossRepositorySubjectClosure",
]
EXTENSION_NAME = re.compile(r"^[A-Z][A-Za-z0-9]+$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_SCORE_KEYS = {
    "coverage_percent",
    "verified_percent",
    "verification_score",
    "formal_score",
    "assurance_score",
    "overall_score",
}
HETEROGENEOUS_DIMENSIONS = {
    "MathematicalSpecification",
    "SourceRefinement",
    "CompilerSemantics",
    "BuildIdentity",
    "RuntimeIdentity",
    "CrossRepositorySubjectClosure",
}


class ClosureError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureError(message)


def nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def reject_score_keys(node: Any, path: str = "$") -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            require(key not in FORBIDDEN_SCORE_KEYS, f"{path}: forbidden aggregate score key {key!r}")
            reject_score_keys(value, f"{path}.{key}")
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            reject_score_keys(value, f"{path}[{idx}]")


def validate_support(support: Any, path: str) -> set[str]:
    require(isinstance(support, list) and support, f"{path}: support must be a non-empty list")
    identities: set[str] = set()
    for idx, item in enumerate(support):
        p = f"{path}[{idx}]"
        require(isinstance(item, dict), f"{p}: support entry must be an object")
        require(set(item) <= {"kind", "identity", "evidence_class", "claim_ceiling"}, f"{p}: unsupported support field")
        require(nonempty_text(item.get("kind")), f"{p}: kind required")
        require(nonempty_text(item.get("identity")), f"{p}: identity required")
        identities.add(item["identity"])
        if "evidence_class" in item:
            require(nonempty_text(item["evidence_class"]), f"{p}: evidence_class must be non-empty")
        if "claim_ceiling" in item:
            ceiling = item["claim_ceiling"]
            require(isinstance(ceiling, list) and all(nonempty_text(x) for x in ceiling), f"{p}: invalid claim_ceiling")
            require(len(ceiling) == len(set(ceiling)), f"{p}: duplicate claim_ceiling entry")
    return identities


def validate_dimension(name: str, dim: Any, path: str) -> set[str]:
    require(isinstance(dim, dict), f"{path}: dimension must be an object")
    allowed = {"state", "support", "obligation", "followup_issue", "reason", "bound", "provider", "profile", "claim_ceiling"}
    require(set(dim) <= allowed, f"{path}: unsupported fields: {sorted(set(dim) - allowed)}")
    state = dim.get("state")
    require(state in STATES, f"{path}: invalid state {state!r}")

    support_ids: set[str] = set()
    if "support" in dim:
        support_ids = validate_support(dim["support"], f"{path}.support")

    if state == "Closed":
        require(support_ids, f"{path}: Closed requires exact support receipt/capsule identity")
        require("obligation" not in dim, f"{path}: Closed may not retain an unresolved obligation")
    elif state == "Open":
        require(nonempty_text(dim.get("obligation")), f"{path}: Open requires an obligation")
        require(not support_ids, f"{path}: Open may cite context elsewhere but may not carry closing support")
    elif state == "NotApplicable":
        require(nonempty_text(dim.get("reason")), f"{path}: NotApplicable requires reason")
        require(not support_ids, f"{path}: NotApplicable may not carry closing support")
    elif state == "ImportedAssurance":
        require(nonempty_text(dim.get("provider")), f"{path}: ImportedAssurance requires provider")
        require(nonempty_text(dim.get("profile")), f"{path}: ImportedAssurance requires profile/version identity")
        ceiling = dim.get("claim_ceiling")
        require(isinstance(ceiling, list) and ceiling and all(nonempty_text(x) for x in ceiling),
                f"{path}: ImportedAssurance requires non-empty claim_ceiling")
    elif state == "BoundedOnly":
        require(nonempty_text(dim.get("bound")), f"{path}: BoundedOnly requires exact bound/profile")
        require(support_ids, f"{path}: BoundedOnly requires exact supporting evidence")
    elif state == "Unknown":
        require(nonempty_text(dim.get("reason")), f"{path}: Unknown requires reason")
        require(not support_ids, f"{path}: Unknown may not carry closing support")

    if "followup_issue" in dim:
        require(isinstance(dim["followup_issue"], int) and dim["followup_issue"] > 0,
                f"{path}: followup_issue must be a positive integer")
    return support_ids


def validate_record(record: Any) -> None:
    require(isinstance(record, dict), "record must be an object")
    reject_score_keys(record)
    require(set(record) == {"schema", "claim_id", "subject", "claim_ceiling", "dimensions"},
            "record top-level fields must be exact")
    require(record["schema"] == SCHEMA, "schema drift")
    require(nonempty_text(record["claim_id"]), "claim_id required")

    subject = record["subject"]
    require(isinstance(subject, dict), "subject must be object")
    require(set(subject) == {"repository", "commit", "semantic_subject"}, "subject fields must be exact")
    require(nonempty_text(subject["repository"]), "subject.repository required")
    require(isinstance(subject["commit"], str) and HEX40.fullmatch(subject["commit"]) is not None,
            "subject.commit must be lowercase 40-hex Git commit")
    require(nonempty_text(subject["semantic_subject"]), "subject.semantic_subject required")

    ceiling = record["claim_ceiling"]
    require(isinstance(ceiling, list) and ceiling and all(nonempty_text(x) for x in ceiling), "claim_ceiling required")
    require(len(ceiling) == len(set(ceiling)), "claim_ceiling must be unique")

    dimensions = record["dimensions"]
    require(isinstance(dimensions, dict), "dimensions must be object")
    require(all(name in dimensions for name in CORE_DIMENSIONS), "all core closure dimensions are mandatory")
    require(set(dimensions) <= set(CORE_DIMENSIONS) | {"extensions"}, "unknown core dimension")

    support_by_dimension: dict[str, set[str]] = {}
    for name in CORE_DIMENSIONS:
        support_by_dimension[name] = validate_dimension(name, dimensions[name], f"dimensions.{name}")

    extensions = dimensions.get("extensions", {})
    require(isinstance(extensions, dict), "dimensions.extensions must be object")
    for name, dim in extensions.items():
        require(EXTENSION_NAME.fullmatch(name) is not None, f"invalid extension dimension name: {name}")
        validate_dimension(name, dim, f"dimensions.extensions.{name}")

    # Anti-greenwashing ratchet: one receipt identity may not be repeated across
    # multiple heterogeneous closure boundaries as if it closed all of them.
    closed_heterogeneous = [
        name for name in HETEROGENEOUS_DIMENSIONS
        if dimensions[name]["state"] == "Closed"
    ]
    if len(closed_heterogeneous) >= 2:
        union = set().union(*(support_by_dimension[name] for name in closed_heterogeneous))
        require(len(union) >= 2,
                "one support identity may not close multiple heterogeneous assurance dimensions")


def validate_examples(payload: Any) -> None:
    require(isinstance(payload, dict), "examples payload must be object")
    require(payload.get("schema") == "symthaea.formal.closure-vector-examples.v1", "examples schema drift")
    records = payload.get("records")
    require(isinstance(records, list) and len(records) >= 3, "at least three seeded examples required")
    ids: set[str] = set()
    for record in records:
        validate_record(record)
        require(record["claim_id"] not in ids, f"duplicate claim_id: {record['claim_id']}")
        ids.add(record["claim_id"])


def synthetic_base() -> dict[str, Any]:
    dims = {name: {"state": "NotApplicable", "reason": f"synthetic {name} not applicable"} for name in CORE_DIMENSIONS}
    return {
        "schema": SCHEMA,
        "claim_id": "synthetic.validator.fixture",
        "subject": {
            "repository": "Luminous-Dynamics/symthaea",
            "commit": "0" * 40,
            "semantic_subject": "synthetic",
        },
        "claim_ceiling": ["synthetic validator fixture only"],
        "dimensions": dims,
    }


def expect_reject(name: str, record: dict[str, Any]) -> None:
    try:
        validate_record(record)
    except ClosureError:
        return
    raise AssertionError(f"hostile mutant unexpectedly accepted: {name}")


def self_test() -> None:
    base = synthetic_base()
    validate_record(base)

    # Exercise each state positively.
    for state, extra in [
        ("Open", {"obligation": "prove the missing relation"}),
        ("Unknown", {"reason": "not yet classified"}),
        ("ImportedAssurance", {"provider": "provider", "profile": "v1@sha256:fixture", "claim_ceiling": ["provider claim only"]}),
        ("Closed", {"support": [{"kind": "Receipt", "identity": "sha256:closed-fixture"}]}),
        ("BoundedOnly", {"bound": "2^8 states", "support": [{"kind": "Receipt", "identity": "sha256:bounded-fixture"}]}),
    ]:
        candidate = synthetic_base()
        candidate["dimensions"]["TheoremStatementIdentity"] = {"state": state, **extra}
        validate_record(candidate)

    mutant = synthetic_base()
    mutant["coverage_percent"] = 100
    expect_reject("coverage-percent", mutant)

    mutant = synthetic_base()
    mutant["dimensions"]["MathematicalSpecification"] = {"state": "Closed"}
    expect_reject("closed-without-support", mutant)

    mutant = synthetic_base()
    mutant["dimensions"]["RuntimeIdentity"] = {"state": "BoundedOnly", "support": [{"kind": "Receipt", "identity": "x"}]}
    expect_reject("bounded-without-bound", mutant)

    mutant = synthetic_base()
    mutant["dimensions"]["DependencyClosure"] = {"state": "ImportedAssurance", "provider": "vendor"}
    expect_reject("import-without-profile-ceiling", mutant)

    mutant = synthetic_base()
    mutant["dimensions"]["RuntimeIdentity"] = {"state": "NotApplicable"}
    expect_reject("not-applicable-without-reason", mutant)

    # One theorem receipt cannot simultaneously close source, compiler, build,
    # and runtime boundaries merely by being copied into every field.
    mutant = synthetic_base()
    same = {"state": "Closed", "support": [{"kind": "Receipt", "identity": "sha256:one-green-receipt"}]}
    mutant["dimensions"]["MathematicalSpecification"] = copy.deepcopy(same)
    mutant["dimensions"]["SourceRefinement"] = copy.deepcopy(same)
    expect_reject("single-receipt-all-green", mutant)


def main() -> int:
    # Schema presence/parseability is part of the contract even though this
    # dependency-free validator owns the state-dependent semantic checks.
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    require(schema.get("title") == "Symthaea Formal Closure Vector V1", "schema document drift")
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    validate_examples(examples)
    self_test()
    print("formal_closure_vector_v1=PASS")
    print(f"seeded_records={len(examples['records'])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"formal_closure_vector_v1=FAIL: {exc}", file=sys.stderr)
        raise
