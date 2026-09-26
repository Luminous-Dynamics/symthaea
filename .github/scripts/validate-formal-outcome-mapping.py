#!/usr/bin/env python3
"""Validate the canonical formal verifier outcome-to-result mapping.

This is a semantic contract validator, not a verifier-correctness proof. It
ensures the repository's reviewed disposition classes cannot silently map
resource exhaustion, unsupported boundaries, ambiguous outcomes, or
infrastructure failures to canonical Pass/Fail states that overclaim what was
actually observed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

MAPPING = Path("docs/formal/sym-fv-infra-003b-outcome-mapping-v1.json")
CANONICAL = {"Pass", "Fail", "Blocked", "EnvironmentFailure"}

EXPECTED = {
    "SemanticSuccess": "Pass",
    "SemanticCounterexample": "Fail",
    "ProofOrQualificationFailure": "Fail",
    "UnsupportedBoundary": "Blocked",
    "InsufficientBound": "Blocked",
    "ResourceExhaustion": "Blocked",
    "MissingPrerequisite": "Blocked",
    "StaleSubjectOrDependency": "Blocked",
    "AmbiguousOrUnknownOutcome": "Blocked",
    "UnclassifiedToolCrash": "Blocked",
    "EnvironmentUnavailable": "EnvironmentFailure",
    "ToolInstallationFailure": "EnvironmentFailure",
    "RunnerInfrastructureFailure": "EnvironmentFailure",
}


class MappingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MappingError(message)


def validate(data: dict) -> None:
    require(data.get("schema") == "symthaea.formal.outcome-mapping.v1", "wrong schema")
    require(data.get("authority") == "EvidenceOnly", "mapping may only carry EvidenceOnly authority")
    require(set(data.get("canonical_results", [])) == CANONICAL, "canonical result vocabulary drift")
    require(data.get("closed_world_default") == "Blocked", "unknown outcomes must default to Blocked")

    dispositions = data.get("dispositions")
    require(isinstance(dispositions, dict), "dispositions must be an object")
    require(set(dispositions) == set(EXPECTED), "disposition set drift")
    require(dispositions == EXPECTED, "canonical disposition mapping drift")

    for disposition, result in dispositions.items():
        require(result in CANONICAL, f"{disposition}: noncanonical result {result!r}")

    hostile = data.get("hostile_control_requirement", {})
    require(hostile.get("accepted_only_on_expected_semantic_failure") is True, "hostile controls need semantic-failure evidence")
    require(hostile.get("resource_exhaustion_must_not_count_as_rejection") is True, "resource exhaustion cannot satisfy hostile control")
    require(hostile.get("insufficient_bound_must_not_count_as_rejection") is True, "insufficient bounds cannot satisfy hostile control")
    require(hostile.get("environment_failure_must_not_count_as_rejection") is True, "environment failure cannot satisfy hostile control")

    fields = data.get("receipt_fields", [])
    required_fields = {
        "result",
        "outcome_disposition",
        "raw_tool_outcome",
        "mapping_schema_version",
        "mapping_schema_digest",
    }
    require(set(fields) == required_fields, "receipt mapping fields drift")

    examples = data.get("examples", {})
    for tool in ("kani", "aeneas", "lean", "quint"):
        require(tool in examples, f"missing example taxonomy for {tool}")
        for raw, disposition in examples[tool].items():
            require(disposition in dispositions, f"{tool}/{raw}: unknown disposition {disposition}")

    nonclaims = set(data.get("nonclaims", []))
    for required in (
        "VerifierCorrectness",
        "TheoremTruth",
        "OutcomeTaxonomyCompleteness",
        "ImplementationRefinement",
        "EvidenceClassPromotion",
        "RuntimeAuthority",
    ):
        require(required in nonclaims, f"missing nonclaim {required}")


def expect_rejected(label: str, mutant: dict) -> None:
    try:
        validate(mutant)
    except MappingError:
        return
    raise AssertionError(f"hostile outcome-mapping mutant admitted: {label}")


def self_test(base: dict) -> None:
    validate(base)

    mutants: list[tuple[str, dict]] = []

    m = copy.deepcopy(base)
    m["dispositions"]["InsufficientBound"] = "Fail"
    mutants.append(("insufficient bound becomes fail", m))

    m = copy.deepcopy(base)
    m["dispositions"]["ResourceExhaustion"] = "Pass"
    mutants.append(("resource exhaustion becomes pass", m))

    m = copy.deepcopy(base)
    m["dispositions"]["UnsupportedBoundary"] = "Pass"
    mutants.append(("unsupported boundary becomes pass", m))

    m = copy.deepcopy(base)
    m["dispositions"]["EnvironmentUnavailable"] = "Fail"
    mutants.append(("environment unavailable becomes fail", m))

    m = copy.deepcopy(base)
    m["closed_world_default"] = "Pass"
    mutants.append(("unknown defaults to pass", m))

    m = copy.deepcopy(base)
    m["hostile_control_requirement"]["resource_exhaustion_must_not_count_as_rejection"] = False
    mutants.append(("resource exhaustion accepted as hostile rejection", m))

    m = copy.deepcopy(base)
    m["canonical_results"].append("LeanTypechecked")
    mutants.append(("tool-specific canonical state", m))

    for label, mutant in mutants:
        expect_rejected(label, mutant)


if __name__ == "__main__":
    data = json.loads(MAPPING.read_text(encoding="utf-8"))
    self_test(data)
    print("formal_outcome_mapping=PASS")
    print("closed_world_default=Blocked")
    for disposition, result in EXPECTED.items():
        print(f"mapping={disposition}->{result}")
