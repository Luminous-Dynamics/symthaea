#!/usr/bin/env python3
"""Validate the canonical formal verifier outcome mapping.

This is a semantic contract validator, not a verifier-correctness proof. It
keeps operational result (`Pass|Fail|Blocked|EnvironmentFailure`) separate from
semantic polarity so an operational failure cannot silently become evidence
that the underlying system property is false.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

MAPPING = Path("docs/formal/sym-fv-infra-003b-outcome-mapping-v1.json")
CANONICAL = {"Pass", "Fail", "Blocked", "EnvironmentFailure"}
POLARITIES = {
    "PositiveSupport",
    "SemanticCounterevidence",
    "QualificationNegative",
    "NoSemanticConclusion",
}

EXPECTED_RESULT = {
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

EXPECTED_POLARITY = {
    "SemanticSuccess": "PositiveSupport",
    "SemanticCounterexample": "SemanticCounterevidence",
    "ProofOrQualificationFailure": "QualificationNegative",
    "UnsupportedBoundary": "NoSemanticConclusion",
    "InsufficientBound": "NoSemanticConclusion",
    "ResourceExhaustion": "NoSemanticConclusion",
    "MissingPrerequisite": "NoSemanticConclusion",
    "StaleSubjectOrDependency": "NoSemanticConclusion",
    "AmbiguousOrUnknownOutcome": "NoSemanticConclusion",
    "UnclassifiedToolCrash": "NoSemanticConclusion",
    "EnvironmentUnavailable": "NoSemanticConclusion",
    "ToolInstallationFailure": "NoSemanticConclusion",
    "RunnerInfrastructureFailure": "NoSemanticConclusion",
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
    require(set(data.get("semantic_polarity_states", [])) == POLARITIES, "semantic polarity vocabulary drift")
    require(data.get("closed_world_default") == "Blocked", "unknown outcomes must default to Blocked")
    require(
        data.get("closed_world_semantic_default") == "NoSemanticConclusion",
        "unknown outcomes must carry no semantic conclusion",
    )

    dispositions = data.get("dispositions")
    require(isinstance(dispositions, dict), "dispositions must be an object")
    require(set(dispositions) == set(EXPECTED_RESULT), "disposition set drift")
    require(dispositions == EXPECTED_RESULT, "canonical disposition mapping drift")

    polarity = data.get("semantic_polarity")
    require(isinstance(polarity, dict), "semantic_polarity must be an object")
    require(set(polarity) == set(EXPECTED_POLARITY), "semantic polarity disposition set drift")
    require(polarity == EXPECTED_POLARITY, "canonical semantic polarity mapping drift")

    for disposition, result in dispositions.items():
        require(result in CANONICAL, f"{disposition}: noncanonical result {result!r}")
    for disposition, semantic in polarity.items():
        require(semantic in POLARITIES, f"{disposition}: unknown semantic polarity {semantic!r}")

    # The crucial epistemic boundary: workflow failure and semantic refutation
    # are distinct. Only an explicit SemanticCounterexample carries refutation
    # polarity in this mapping generation.
    require(
        dispositions["ProofOrQualificationFailure"] == "Fail"
        and polarity["ProofOrQualificationFailure"] == "QualificationNegative",
        "proof/qualification failure must be operational Fail without semantic refutation",
    )
    require(
        dispositions["SemanticCounterexample"] == "Fail"
        and polarity["SemanticCounterexample"] == "SemanticCounterevidence",
        "semantic counterexample must retain explicit counterevidence polarity",
    )
    require(
        polarity["SemanticSuccess"] == "PositiveSupport",
        "semantic success must be the only positive-support mapping",
    )
    require(
        [k for k, v in polarity.items() if v == "PositiveSupport"] == ["SemanticSuccess"],
        "only SemanticSuccess may provide PositiveSupport",
    )
    require(
        [k for k, v in polarity.items() if v == "SemanticCounterevidence"] == ["SemanticCounterexample"],
        "only SemanticCounterexample may provide SemanticCounterevidence",
    )

    hostile = data.get("hostile_control_requirement", {})
    require(hostile.get("accepted_only_on_expected_semantic_failure") is True, "hostile controls need semantic-failure evidence")
    require(hostile.get("semantic_counterevidence_required_for_refutation") is True, "refutation needs semantic counterevidence")
    require(hostile.get("qualification_failure_must_not_count_as_counterevidence") is True, "qualification failure cannot refute property")
    require(hostile.get("resource_exhaustion_must_not_count_as_rejection") is True, "resource exhaustion cannot satisfy hostile control")
    require(hostile.get("insufficient_bound_must_not_count_as_rejection") is True, "insufficient bounds cannot satisfy hostile control")
    require(hostile.get("environment_failure_must_not_count_as_rejection") is True, "environment failure cannot satisfy hostile control")

    fields = data.get("receipt_fields", [])
    required_fields = {
        "result",
        "outcome_disposition",
        "semantic_polarity",
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
            require(disposition in polarity, f"{tool}/{raw}: disposition lacks semantic polarity")

    rules = set(data.get("semantic_rules", []))
    required_rules = {
        "operational-fail-does-not-by-itself-refute-property",
        "only-semantic-counterevidence-may-participate-in-refutation",
        "qualification-negative-prevents-positive-admission-without-implying-property-false",
        "no-semantic-conclusion-carries-neither-positive-support-nor-refutation-authority",
    }
    require(required_rules.issubset(rules), "semantic rule set incomplete")

    nonclaims = set(data.get("nonclaims", []))
    for required in (
        "VerifierCorrectness",
        "TheoremTruth",
        "OutcomeTaxonomyCompleteness",
        "ImplementationRefinement",
        "EvidenceClassPromotion",
        "RefutationWithoutExplicitClaimRelation",
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
    m["semantic_polarity"]["ProofOrQualificationFailure"] = "SemanticCounterevidence"
    mutants.append(("qualification failure becomes counterevidence", m))

    m = copy.deepcopy(base)
    m["semantic_polarity"]["ResourceExhaustion"] = "SemanticCounterevidence"
    mutants.append(("resource exhaustion becomes counterevidence", m))

    m = copy.deepcopy(base)
    m["semantic_polarity"]["SemanticCounterexample"] = "QualificationNegative"
    mutants.append(("semantic counterexample loses refutation polarity", m))

    m = copy.deepcopy(base)
    m["semantic_polarity"]["UnsupportedBoundary"] = "PositiveSupport"
    mutants.append(("unsupported boundary becomes positive semantic evidence", m))

    m = copy.deepcopy(base)
    m["closed_world_default"] = "Pass"
    mutants.append(("unknown defaults to pass", m))

    m = copy.deepcopy(base)
    m["closed_world_semantic_default"] = "SemanticCounterevidence"
    mutants.append(("unknown defaults to counterevidence", m))

    m = copy.deepcopy(base)
    m["hostile_control_requirement"]["qualification_failure_must_not_count_as_counterevidence"] = False
    mutants.append(("qualification failure accepted as refutation", m))

    m = copy.deepcopy(base)
    m["hostile_control_requirement"]["resource_exhaustion_must_not_count_as_rejection"] = False
    mutants.append(("resource exhaustion accepted as hostile rejection", m))

    m = copy.deepcopy(base)
    m["receipt_fields"].remove("semantic_polarity")
    mutants.append(("semantic polarity omitted from receipt binding", m))

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
    print("closed_world_semantic_default=NoSemanticConclusion")
    for disposition, result in EXPECTED_RESULT.items():
        print(f"result_mapping={disposition}->{result}")
        print(f"semantic_mapping={disposition}->{EXPECTED_POLARITY[disposition]}")
