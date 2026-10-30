#!/usr/bin/env python3
"""Static structure/coverage audit for a versioned rubric specification.

This checks declared rubric completeness and planned test coverage only. It
cannot prove construct validity, empirical dimensionality, preference fit,
judge robustness or adversarial resistance.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_TEST_TYPES = ("positive", "negative", "edge_case", "adversarial")
TOP_FIELDS = {
    "schema", "purpose", "rubric_id", "rubric_source_revision", "rubric_sha256",
    "hard_constraint_policy", "score_scale", "criteria", "dependency_rules", "tests",
}
CRITERION_FIELDS = {
    "criterion_id", "construct_dimension", "definition", "applicability",
    "required_observable_evidence", "insufficient_evidence", "exclusions",
    "priority", "hard_constraint", "veto_condition", "score_anchors",
    "counterexamples", "planned_test_ids",
}
TEST_FIELDS = {
    "test_id", "test_type", "criterion_ids", "expected_judge_behavior",
    "candidate_pair_sha256",
}
DEPENDENCY_FIELDS = {"from_criterion_id", "to_criterion_id", "relation", "rationale"}


def _exact_fields(row: dict[str, Any], expected: set[str], context: str, issues: list[str]) -> None:
    actual = set(row)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        issues.append(f"{context} missing fields: {','.join(missing)}")
    if extra:
        issues.append(f"{context} unknown fields: {','.join(extra)}")


def audit_rubric_specification(spec: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if not isinstance(spec, dict):
        raise ValueError("rubric specification must be a JSON object")
    _exact_fields(spec, TOP_FIELDS, "rubric", issues)

    if spec.get("schema") != "symthaea.science_bench_refresh_rubric_specification.v1":
        issues.append("unsupported rubric specification schema")
    purpose = spec.get("purpose")
    if not isinstance(purpose, str) or purpose not in {"example_only", "qualified_rubric"}:
        issues.append("purpose must be example_only or qualified_rubric")
    example = purpose == "example_only"

    rubric_id = spec.get("rubric_id")
    if not isinstance(rubric_id, str) or not rubric_id.strip() or len(rubric_id) > 128:
        issues.append("rubric_id must be non-empty and at most 128 characters")
    revision = spec.get("rubric_source_revision")
    if not isinstance(revision, str) or not revision.strip():
        issues.append("rubric_source_revision must be non-empty")
    elif not example and (HEX40.fullmatch(revision) is None or set(revision) == {"0"}):
        issues.append("qualified rubric requires a non-zero lowercase 40-character source revision")
    digest = spec.get("rubric_sha256")
    if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
        issues.append("rubric_sha256 must be 64 lowercase hexadecimal characters")
    elif not example and set(digest) == {"0"}:
        issues.append("qualified rubric cannot use a zero rubric digest")

    if spec.get("hard_constraint_policy") != "veto_not_average":
        issues.append("hard_constraint_policy must be veto_not_average")

    scale = spec.get("score_scale")
    if not isinstance(scale, dict) or set(scale) != {"minimum_millionths", "maximum_millionths"}:
        issues.append("score_scale must contain exactly minimum_millionths and maximum_millionths")
    elif scale != {"minimum_millionths": 0, "maximum_millionths": 1_000_000}:
        issues.append("v1 score scale must be 0..1,000,000 millionths")

    criteria = spec.get("criteria")
    tests = spec.get("tests")
    dependencies = spec.get("dependency_rules")
    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 64:
        issues.append("criteria must contain 2..64 rows")
        criteria = criteria if isinstance(criteria, list) else []
    if not isinstance(tests, list) or not 1 <= len(tests) <= 10_000:
        issues.append("tests must contain 1..10,000 rows")
        tests = tests if isinstance(tests, list) else []
    if not isinstance(dependencies, list) or len(dependencies) > 512:
        issues.append("dependency_rules must be an array of at most 512 rows")
        dependencies = dependencies if isinstance(dependencies, list) else []

    criteria_by_id: dict[str, dict[str, Any]] = {}
    for index, criterion in enumerate(criteria):
        context = f"criterion[{index}]"
        if not isinstance(criterion, dict):
            issues.append(f"{context} must be an object")
            continue
        _exact_fields(criterion, CRITERION_FIELDS, context, issues)
        criterion_id = criterion.get("criterion_id")
        if not isinstance(criterion_id, str) or not criterion_id.strip() or len(criterion_id) > 128:
            issues.append(f"{context} has invalid criterion_id")
            continue
        if criterion_id in criteria_by_id:
            issues.append(f"duplicate criterion_id {criterion_id}")
        criteria_by_id[criterion_id] = criterion

        for field, minimum in (
            ("required_observable_evidence", 2),
            ("insufficient_evidence", 2),
            ("exclusions", 2),
            ("counterexamples", 2),
            ("planned_test_ids", 4),
        ):
            values = criterion.get(field)
            if not isinstance(values, list) or len(values) < minimum:
                issues.append(f"criterion {criterion_id} requires at least {minimum} {field}")
            elif any(not isinstance(value, str) or not value.strip() for value in values):
                issues.append(f"criterion {criterion_id} contains empty {field} entries")
            elif field == "planned_test_ids" and len(values) != len(set(values)):
                issues.append(f"criterion {criterion_id} has duplicate planned_test_ids")

        for field, maximum in (("definition", 4000), ("applicability", 2000)):
            value = criterion.get(field)
            if not isinstance(value, str) or len(value.strip()) < 20 or len(value) > maximum:
                issues.append(f"criterion {criterion_id} has inadequate {field}")

        dimension = criterion.get("construct_dimension")
        if not isinstance(dimension, str) or not dimension.strip():
            issues.append(f"criterion {criterion_id} requires a construct_dimension")

        priority = criterion.get("priority")
        if not isinstance(priority, int) or isinstance(priority, bool) or not 1 <= priority <= 100:
            issues.append(f"criterion {criterion_id} priority must be an integer in 1..100")

        hard = criterion.get("hard_constraint")
        veto = criterion.get("veto_condition")
        if not isinstance(hard, bool):
            issues.append(f"criterion {criterion_id} hard_constraint must be boolean")
        elif hard and (not isinstance(veto, str) or len(veto.strip()) < 8):
            issues.append(f"hard criterion {criterion_id} requires an explicit veto_condition")
        elif not hard and veto is not None:
            issues.append(f"non-hard criterion {criterion_id} must have null veto_condition")

        anchors = criterion.get("score_anchors")
        if not isinstance(anchors, list) or len(anchors) != 3:
            issues.append(f"criterion {criterion_id} requires exactly three score_anchors")
        else:
            values = [item.get("score_millionths") if isinstance(item, dict) else None for item in anchors]
            if values != [0, 500_000, 1_000_000]:
                issues.append(f"criterion {criterion_id} score anchors must be 0, 500000 and 1000000 millionths")
            if any(not isinstance(item, dict) or not isinstance(item.get("description"), str) or len(item["description"].strip()) < 8 for item in anchors):
                issues.append(f"criterion {criterion_id} has an empty score-anchor description")

    tests_by_id: dict[str, dict[str, Any]] = {}
    coverage: dict[str, dict[str, int]] = {
        criterion_id: {test_type: 0 for test_type in REQUIRED_TEST_TYPES}
        for criterion_id in criteria_by_id
    }
    for index, test in enumerate(tests):
        context = f"test[{index}]"
        if not isinstance(test, dict):
            issues.append(f"{context} must be an object")
            continue
        _exact_fields(test, TEST_FIELDS, context, issues)
        test_id = test.get("test_id")
        if not isinstance(test_id, str) or not test_id.strip() or len(test_id) > 128:
            issues.append(f"{context} has invalid test_id")
            continue
        if test_id in tests_by_id:
            issues.append(f"duplicate test_id {test_id}")
        tests_by_id[test_id] = test
        test_type = test.get("test_type")
        if not isinstance(test_type, str) or test_type not in {"positive", "negative", "edge_case", "adversarial", "meaning_preserving", "meaning_changing"}:
            issues.append(f"test {test_id} has unsupported test_type")
            test_type = ""
        candidate_digest = test.get("candidate_pair_sha256")
        if not isinstance(candidate_digest, str) or HEX64.fullmatch(candidate_digest) is None:
            issues.append(f"test {test_id} has invalid candidate_pair_sha256")
        elif not example and set(candidate_digest) == {"0"}:
            issues.append(f"qualified test {test_id} cannot use a zero candidate-pair digest")
        criterion_ids = test.get("criterion_ids")
        if not isinstance(criterion_ids, list) or not criterion_ids:
            issues.append(f"test {test_id} requires criterion_ids")
            continue
        if any(not isinstance(item, str) for item in criterion_ids):
            issues.append(f"test {test_id} criterion_ids must be strings")
            continue
        if len(criterion_ids) != len(set(criterion_ids)):
            issues.append(f"test {test_id} repeats a criterion_id")
        for criterion_id in criterion_ids:
            if criterion_id not in criteria_by_id:
                issues.append(f"test {test_id} references unknown criterion {criterion_id}")
            elif test_type in coverage[criterion_id]:
                coverage[criterion_id][test_type] += 1
        behavior = test.get("expected_judge_behavior")
        if not isinstance(behavior, str) or behavior not in {
            "accept_supported", "reject_unsupported",
            "abstain_or_mark_scope_ambiguous", "no_unearned_score_uplift",
            "directional_criterion_sensitivity",
        }:
            issues.append(f"test {test_id} has unsupported expected_judge_behavior")

    for criterion_id, criterion in criteria_by_id.items():
        planned = criterion.get("planned_test_ids", [])
        if not isinstance(planned, list):
            continue
        for test_id in planned:
            if not isinstance(test_id, str):
                continue
            test = tests_by_id.get(test_id)
            if test is None:
                issues.append(f"criterion {criterion_id} references missing planned test {test_id}")
            elif criterion_id not in test.get("criterion_ids", []):
                issues.append(f"planned test {test_id} does not map back to criterion {criterion_id}")
        for test_id, test in tests_by_id.items():
            if criterion_id in test.get("criterion_ids", []) and test_id not in planned:
                issues.append(f"criterion {criterion_id} omits mapped test {test_id} from planned_test_ids")
        for test_type in REQUIRED_TEST_TYPES:
            if coverage[criterion_id][test_type] < 1:
                issues.append(f"criterion {criterion_id} lacks {test_type} test coverage")

    seen_dependencies: set[tuple[str, str, str]] = set()
    for index, dependency in enumerate(dependencies):
        context = f"dependency[{index}]"
        if not isinstance(dependency, dict):
            issues.append(f"{context} must be an object")
            continue
        _exact_fields(dependency, DEPENDENCY_FIELDS, context, issues)
        source = dependency.get("from_criterion_id")
        target = dependency.get("to_criterion_id")
        relation = dependency.get("relation")
        if not isinstance(source, str) or not isinstance(target, str) or not isinstance(relation, str):
            issues.append(f"{context} source, target and relation must be strings")
            continue
        if source not in criteria_by_id or target not in criteria_by_id:
            issues.append(f"{context} references an unknown criterion")
        if source == target:
            issues.append(f"{context} cannot refer to itself")
        if relation not in {"requires", "conflicts_with", "overrides"}:
            issues.append(f"{context} has unsupported relation")
        key = (source, target, relation)
        if key in seen_dependencies:
            issues.append(f"{context} duplicates a dependency rule")
        seen_dependencies.add(key)
        rationale = dependency.get("rationale")
        if not isinstance(rationale, str) or len(rationale.strip()) < 12:
            issues.append(f"{context} requires a substantive rationale")

    dimensions = {
        criterion.get("construct_dimension")
        for criterion in criteria_by_id.values()
        if isinstance(criterion.get("construct_dimension"), str)
    }
    hard_count = sum(1 for criterion in criteria_by_id.values() if criterion.get("hard_constraint") is True)
    report = {
        "schema": "symthaea.science_bench_refresh_rubric_specification_audit.v1",
        "rubric_id": rubric_id if isinstance(rubric_id, str) else "",
        "valid": not issues,
        "criterion_count": len(criteria_by_id),
        "declared_dimension_count": len(dimensions),
        "hard_constraint_count": hard_count,
        "test_count": len(tests_by_id),
        "coverage_by_criterion": [
            {"criterion_id": criterion_id, **coverage[criterion_id]}
            for criterion_id in sorted(coverage)
        ],
        "uncovered_criterion_count": sum(
            1 for row in coverage.values()
            if any(row.get(test_type, 0) == 0 for test_type in REQUIRED_TEST_TYPES)
        ),
        "hard_constraints_veto_not_average": spec.get("hard_constraint_policy") == "veto_not_average",
        "empirical_dimensionality_estimated": False,
        "construct_validity_proven": False,
        "preference_fit_estimated": False,
        "adversarial_robustness_proven": False,
        "judge_execution_proven": False,
        "human_adjudication_proven": False,
        "scoring_eligible_declared": False,
        "claim_boundary": "Static structural completeness and declared test coverage only; not construct validity, empirical dimensionality, human preference fit, judge behavior, or adversarial robustness.",
        "issues": issues,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rubric_spec", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        spec = json.loads(args.rubric_spec.read_text(encoding="utf-8"))
        report = audit_rubric_specification(spec)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(json.dumps({"valid": False, "issues": [str(error)]}, indent=2))
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["valid"] else 11


if __name__ == "__main__":
    raise SystemExit(main())
