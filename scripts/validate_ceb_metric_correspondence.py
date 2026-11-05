#!/usr/bin/env python3
"""Validate CEB metric-to-external-observation correspondence.

Offline data-contract validation only: this does not run candidate code,
benchmark tasks, graders or external comparisons.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error


def semantic_errors(crosswalk: dict, matrix: dict, registry: dict) -> list[str]:
    errors: list[str] = []
    expected = {
        (lane["lane_id"], metric["metric_id"]): metric
        for lane in matrix["lanes"] for metric in lane["primary_metrics"]
    }
    mappings = crosswalk["metric_mappings"]
    actual_keys = [(row["lane_id"], row["ceb_metric_id"]) for row in mappings]
    if len(actual_keys) != len(set(actual_keys)):
        errors.append("duplicate CEB metric mapping")
    if set(actual_keys) != set(expected):
        missing = sorted(set(expected) - set(actual_keys))
        extra = sorted(set(actual_keys) - set(expected))
        errors.append(f"metric coverage mismatch; missing={missing}; extra={extra}")

    observations = {row["observation_id"]: row for row in registry["observations"]}
    comparators = {row["comparator_id"]: row for row in registry["comparators"]}

    for row in mappings:
        key = (row["lane_id"], row["ceb_metric_id"])
        metric = expected.get(key)
        if metric is None:
            continue
        if row["ceb_metric_definition"] != metric["definition"]:
            errors.append(f"{key}: CEB metric definition drifted from measurement matrix")
        if row["target_statement"] != metric["prospective_target"]:
            errors.append(f"{key}: target statement drifted from measurement matrix")
        if row["direct_superiority_claim_allowed"] is not False:
            errors.append(f"{key}: direct superiority claim must remain disabled")
        exact = row["correspondence_class"] == "exact_semantics"
        if row["metric_semantics_eligible_for_matched_test"] != exact:
            errors.append(f"{key}: matched-test eligibility does not follow correspondence class")
        if exact and row["comparison_use"] != "replication_candidate":
            errors.append(f"{key}: exact semantics must be a replication candidate")
        if not row["external_observation_ids"]:
            if row["correspondence_class"] != "not_comparable" or row["comparison_use"] != "not_comparable":
                errors.append(f"{key}: empty mapping must be explicitly not_comparable")
        elif row["comparison_use"] == "not_comparable":
            errors.append(f"{key}: non-empty mapping cannot be marked not_comparable")

        external_metrics = []
        for observation_id in row["external_observation_ids"]:
            observation = observations.get(observation_id)
            if observation is None:
                errors.append(f"{key}: unknown external observation {observation_id}")
                continue
            comparator = comparators.get(observation["comparator_id"])
            if comparator is None:
                errors.append(f"{key}: observation {observation_id} has unknown comparator")
                continue
            if comparator["lane_id"] != row["lane_id"]:
                errors.append(f"{key}: observation {observation_id} belongs to {comparator['lane_id']}, not {row['lane_id']}")
            external_metrics.append(observation["metric"]["metric_id"])

        if row["ceb_metric_id"] == "recall_at_10" and any(value != "recall" for value in external_metrics):
            errors.append("CEB-CONTEXT Recall@10 may only map to a recall observation, not Context F1 or Pass@1")
        if row["ceb_metric_id"] == "bug_recall_at_k":
            if any(value != "bug_recall" for value in external_metrics):
                errors.append("CEB-DEBUG bug_recall_at_k may only map to PDB bug_recall observations")
            if row["correspondence_class"] == "exact_semantics":
                errors.append("CEB-DEBUG Recall@K cannot be exact semantics for aggregate PDB bug recall")
        if row["ceb_metric_id"] == "joint_secure_functional_pass" and any(value != "secure_pass" for value in external_metrics):
            errors.append("CEB-SECURE joint secure-functional pass cannot be mapped to FuncPass")
        agent_expected = {
            "paired_act_abstain_accuracy": ("agentabstain-gemini-paired", "paired_accuracy"),
            "act_side_accuracy": ("agentabstain-gemini-act", "act_accuracy"),
            "abstain_side_accuracy": ("agentabstain-gemini-abstain", "abstain_accuracy"),
            "conditioned_abstention_rate": ("agentabstain-gemini-car", "conditioned_abstention_rate"),
        }
        if row["ceb_metric_id"] in agent_expected:
            expected_id, expected_metric = agent_expected[row["ceb_metric_id"]]
            if row["external_observation_ids"] != [expected_id]:
                errors.append(f"{key}: AgentAbstain metric must bind only to {expected_id}")
            if external_metrics != [expected_metric] or not exact:
                errors.append(f"{key}: AgentAbstain paired metric semantics were altered")

    return errors


def validate(crosswalk: dict, schema: dict, matrix: dict, registry: dict) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(crosswalk), key=lambda error: list(error.absolute_path))
    if schema_errors:
        return [
            f"schema: {'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in schema_errors
        ]
    return semantic_errors(crosswalk, matrix, registry)


def self_test(crosswalk: dict, schema: dict, matrix: dict, registry: dict) -> None:
    def must_reject(mutant: dict, label: str) -> None:
        if not validate(mutant, schema, matrix, registry):
            raise SystemExit(f"metric correspondence negative control unexpectedly passed: {label}")

    mutant = copy.deepcopy(crosswalk)
    mutant["metric_mappings"] = mutant["metric_mappings"][1:]
    must_reject(mutant, "missing CEB metric mapping")

    mutant = copy.deepcopy(crosswalk)
    mutant["metric_mappings"][1]["ceb_metric_id"] = mutant["metric_mappings"][0]["ceb_metric_id"]
    must_reject(mutant, "duplicate CEB metric mapping")

    mutant = copy.deepcopy(crosswalk)
    row = next(x for x in mutant["metric_mappings"] if x["ceb_metric_id"] == "recall_at_10")
    row["external_observation_ids"] = ["contextbench-codenib-f1"]
    must_reject(mutant, "Recall@10 incorrectly mapped to Context F1")

    mutant = copy.deepcopy(crosswalk)
    row = next(x for x in mutant["metric_mappings"] if x["ceb_metric_id"] == "bug_recall_at_k")
    row["correspondence_class"] = "exact_semantics"
    row["metric_semantics_eligible_for_matched_test"] = True
    row["comparison_use"] = "replication_candidate"
    must_reject(mutant, "Recall@5 incorrectly promoted to exact PDB bug-recall semantics")

    mutant = copy.deepcopy(crosswalk)
    row = next(x for x in mutant["metric_mappings"] if x["ceb_metric_id"] == "joint_secure_functional_pass")
    row["external_observation_ids"].append("als-fable-funcpass")
    must_reject(mutant, "joint secure pass mapped to FuncPass")

    mutant = copy.deepcopy(crosswalk)
    row = next(x for x in mutant["metric_mappings"] if x["ceb_metric_id"] == "paired_act_abstain_accuracy")
    row["correspondence_class"] = "related_but_distinct"
    must_reject(mutant, "AgentAbstain paired semantics silently downgraded")

    mutant = copy.deepcopy(crosswalk)
    mutant["metric_mappings"][0]["direct_superiority_claim_allowed"] = True
    must_reject(mutant, "direct SOTA claim enabled")

    mutant = copy.deepcopy(crosswalk)
    mutant["metric_mappings"][0]["external_observation_ids"] = ["agentabstain-gemini-paired"]
    must_reject(mutant, "cross-lane observation substituted")

    print("CEB_METRIC_CORRESPONDENCE_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base = root / "benchmarks" / "ceb"
    try:
        crosswalk = load(base / "metric-correspondence.v0.json")
        schema = load(base / "metric-correspondence.schema.json")
        matrix = load(base / "measurement-matrix.v0.json")
        registry = load(base / "external-observations.v0.json")
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        print(f"CEB_METRIC_CORRESPONDENCE_ERROR: {error}", file=sys.stderr)
        return 2
    errors = validate(crosswalk, schema, matrix, registry)
    if errors:
        for error in errors:
            print(f"CEB_METRIC_CORRESPONDENCE_FAIL: {error}", file=sys.stderr)
        return 1
    if args.self_test:
        self_test(crosswalk, schema, matrix, registry)
    print(
        "CEB_METRIC_CORRESPONDENCE_PASS: "
        f"metrics={len(crosswalk['metric_mappings'])} "
        f"exact={sum(x['correspondence_class']=='exact_semantics' for x in crosswalk['metric_mappings'])} "
        f"close={sum(x['correspondence_class']=='close_semantics' for x in crosswalk['metric_mappings'])} "
        f"related={sum(x['correspondence_class']=='related_but_distinct' for x in crosswalk['metric_mappings'])} "
        f"not_comparable={sum(x['correspondence_class']=='not_comparable' for x in crosswalk['metric_mappings'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
