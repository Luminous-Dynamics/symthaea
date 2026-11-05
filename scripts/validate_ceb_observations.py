#!/usr/bin/env python3
"""Validate CEB's normalized external comparator observation registry.

This is a deterministic, offline data-contract check. It does not fetch sources,
execute benchmark tasks, run candidate code, or qualify a benchmark runner.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error


def semantic_errors(payload: dict) -> list[str]:
    errors: list[str] = []
    sources = {row["source_id"]: row for row in payload["sources"]}
    comparators = {row["comparator_id"]: row for row in payload["comparators"]}
    observations = payload["observations"]

    if len(sources) != len(payload["sources"]):
        errors.append("duplicate source_id")
    if len({row["source_url"] for row in payload["sources"]}) != len(payload["sources"]):
        errors.append("duplicate source_url")
    if len(comparators) != len(payload["comparators"]):
        errors.append("duplicate comparator_id")
    if len({row["observation_id"] for row in observations}) != len(observations):
        errors.append("duplicate observation_id")

    used_sources: set[str] = set()
    used_comparators: set[str] = set()
    metric_system_keys: set[tuple[str, str, str]] = set()

    for comparator in payload["comparators"]:
        source_ids = [comparator["source_id"], *comparator["supporting_source_ids"]]
        for source_id in source_ids:
            source = sources.get(source_id)
            if source is None:
                errors.append(f"{comparator['comparator_id']}: unknown source {source_id}")
                continue
            used_sources.add(source_id)
            if comparator["lane_id"] not in source["supports"]:
                errors.append(
                    f"{comparator['comparator_id']}: source {source_id} does not support "
                    f"{comparator['lane_id']}"
                )
        denom = comparator["sample_denominator"]
        if denom["status"] == "reported" and denom["value"] is None:
            errors.append(f"{comparator['comparator_id']}: reported denominator has no value")
        if denom["status"] == "not_reported" and denom["value"] is not None:
            errors.append(f"{comparator['comparator_id']}: not_reported denominator has a value")

    for observation in observations:
        comparator = comparators.get(observation["comparator_id"])
        if comparator is None:
            errors.append(
                f"{observation['observation_id']}: unknown comparator "
                f"{observation['comparator_id']}"
            )
            continue
        used_comparators.add(comparator["comparator_id"])
        metric = observation["metric"]
        uncertainty = metric["uncertainty"]
        if uncertainty["applies_to_metric_id"] != metric["metric_id"]:
            errors.append(
                f"{observation['observation_id']}: uncertainty is attached to a different metric"
            )
        if uncertainty["applies_to_task_count"] != comparator["sample_denominator"]["value"]:
            errors.append(
                f"{observation['observation_id']}: uncertainty denominator differs from comparator denominator"
            )
        if uncertainty["status"] == "reported":
            if uncertainty["margin"] is None:
                errors.append(f"{observation['observation_id']}: reported uncertainty has no margin")
        elif uncertainty["margin"] is not None:
            errors.append(f"{observation['observation_id']}: unreported uncertainty has a margin")

        key = (
            comparator["comparator_id"],
            metric["metric_id"],
            observation["system"]["system_id"],
        )
        if key in metric_system_keys:
            errors.append(
                f"{observation['observation_id']}: duplicate metric/system observation within comparator"
            )
        metric_system_keys.add(key)

        if metric["value_status"] == "unlabeled" and metric["metric_id"] != "unlabeled_leaderboard_score":
            errors.append(f"{observation['observation_id']}: unlabeled value assigned a named metric")
        if metric["value_status"] == "qualitative" and metric["metric_id"] != "verifier_lifecycle_policy":
            errors.append(f"{observation['observation_id']}: qualitative finding assigned a numeric metric")

    if used_comparators != set(comparators):
        missing = sorted(set(comparators) - used_comparators)
        errors.append("comparators without observations: " + ", ".join(missing))
    if used_sources != set(sources):
        missing = sorted(set(sources) - used_sources)
        errors.append("registered sources not used by any comparator: " + ", ".join(missing))

    # These two splits must remain separate benchmark populations.
    full = comparators.get("swe-pro-v2-full")
    hard = comparators.get("swe-pro-v2-hard")
    if full and hard:
        if full["sample_denominator"]["value"] != 642:
            errors.append("SWE-Bench Pro V2 Full denominator drifted from 642")
        if hard["sample_denominator"]["value"] != 51:
            errors.append("SWE-Bench Pro V2 HARD denominator drifted from 51")
        if full["sample_denominator"]["value"] == hard["sample_denominator"]["value"]:
            errors.append("SWE-Bench Pro V2 Full and HARD denominators were collapsed")

    # Do not allow the best functional and best secure scores to be synthesized
    # from different configurations. Both real configurations must remain explicit.
    als = [
        row for row in observations
        if row["comparator_id"] == "agent-security-league"
    ]
    by_system: dict[str, dict[str, float]] = {}
    for row in als:
        by_system.setdefault(row["system"]["system_id"], {})[
            row["metric"]["metric_id"]
        ] = row["metric"]["value"]
    muse = by_system.get("muse-wonder-muse-code", {})
    fable = by_system.get("fable51-claude-code", {})
    if muse.get("functional_pass") != 91.6 or muse.get("secure_pass") != 25.1:
        errors.append("Muse Code/Muse Wonder functional and secure observations drifted or were mixed")
    if fable.get("functional_pass") != 86.6 or fable.get("secure_pass") != 36.9:
        errors.append("Claude Code/Fable 5.1 functional and secure observations drifted or were mixed")

    agent = [
        row for row in observations
        if row["comparator_id"] == "agentabstain"
        and row["system"]["system_id"] == "gemini31-google-adk-agentabstain"
    ]
    agent_metrics = {row["metric"]["metric_id"]: row["metric"]["value"] for row in agent}
    if agent_metrics != {
        "paired_accuracy": 59.5,
        "act_accuracy": 90.5,
        "abstain_accuracy": 65.4,
        "conditioned_abstention_rate": 65.7,
    }:
        errors.append("AgentAbstain paired/act/abstain/CAR observations drifted or were conflated")

    hil = [
        row for row in observations
        if row["comparator_id"] == "hil-bench"
        and row["observation_id"] == "hil-muse-unlabeled-score"
    ]
    if hil and (
        hil[0]["metric"]["value_status"] != "unlabeled"
        or hil[0]["metric"]["metric_id"] != "unlabeled_leaderboard_score"
    ):
        errors.append("HiL-Bench unlabeled score was assigned named metric semantics")

    return errors


def validate_payload(payload: dict, schema: dict, format_checker: FormatChecker) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=format_checker)
    schema_errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    if schema_errors:
        return [
            f"schema: {'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in schema_errors
        ]
    return semantic_errors(payload)


def self_test(payload: dict, schema: dict, format_checker: FormatChecker) -> None:
    def must_reject(mutant: dict, name: str) -> None:
        if not validate_payload(mutant, schema, format_checker):
            raise SystemExit(f"negative control unexpectedly passed: {name}")

    # A score row cannot carry multiple system configurations.
    mutant = copy.deepcopy(payload)
    mutant["observations"][0]["systems"] = [mutant["observations"][0]["system"]]
    must_reject(mutant, "mixed system configurations")

    # A source reference must resolve and support the comparator's lane.
    mutant = copy.deepcopy(payload)
    mutant["comparators"][0]["source_id"] = "missing-source"
    must_reject(mutant, "unknown comparator source")

    # A benchmark split cannot borrow another split's denominator.
    mutant = copy.deepcopy(payload)
    mutant["comparators"][1]["sample_denominator"]["value"] = 642
    must_reject(mutant, "collapsed SWE-Bench Pro Full/HARD denominator")

    # A reported interval must bind to its own metric and exact denominator.
    mutant = copy.deepcopy(payload)
    interval = next(row for row in mutant["observations"] if row["observation_id"] == "pdb-single-opus-precision")
    interval["metric"]["uncertainty"]["applies_to_metric_id"] = "bug_recall"
    must_reject(mutant, "confidence interval attached to a different metric")

    mutant = copy.deepcopy(payload)
    interval = next(row for row in mutant["observations"] if row["observation_id"] == "pdb-single-opus-precision")
    interval["metric"]["uncertainty"]["applies_to_task_count"] = 484
    must_reject(mutant, "confidence interval attached to a different denominator")

    # An unlabeled number cannot be relabeled as ASK-F1 or Pass@3.
    mutant = copy.deepcopy(payload)
    hil = next(row for row in mutant["observations"] if row["observation_id"] == "hil-muse-unlabeled-score")
    hil["metric"]["metric_id"] = "ask_f1"
    must_reject(mutant, "unlabeled HiL score assigned a named metric")

    # A synthetic composite of unrelated maxima must not be introduced.
    mutant = copy.deepcopy(payload)
    mutant["observations"][0]["derived_from_observation_ids"] = [
        "swepro-full-opus-resolve", "contextbench-openjiuwen-f1"
    ]
    must_reject(mutant, "composite score synthesized from unrelated maxima")

    # Observation identities and source dates are part of the evidence record.
    mutant = copy.deepcopy(payload)
    mutant["observations"][1]["observation_id"] = mutant["observations"][0]["observation_id"]
    must_reject(mutant, "duplicate observation identity")

    mutant = copy.deepcopy(payload)
    mutant["sources"][0]["accessed_on"] = "not-a-date"
    must_reject(mutant, "missing/invalid source retrieval date")

    # The security benchmark's functional and secure maxima must stay tied to
    # their actual, different configurations.
    mutant = copy.deepcopy(payload)
    row = next(row for row in mutant["observations"] if row["observation_id"] == "als-fable-secpass")
    row["system"]["system_id"] = "muse-wonder-muse-code"
    must_reject(mutant, "functional/security maxima mixed across configurations")

    # A matched/SOTA claim is not available from a contextual external record.
    mutant = copy.deepcopy(payload)
    mutant["comparators"][0]["comparison_class"] = "beats_sota"
    must_reject(mutant, "unsupported direct superiority claim")

    print("CEB_EXTERNAL_OBSERVATION_SELF_TEST_PASS")


def validate_pilot_payload(pilot: dict, schema: dict, format_checker: FormatChecker) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=format_checker)
    schema_errors = sorted(validator.iter_errors(pilot), key=lambda error: list(error.absolute_path))
    if schema_errors:
        return [
            f"pilot schema: {'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in schema_errors
        ]
    errors: list[str] = []
    families = pilot["population"]["root_cause_families"]
    if len({row["id"] for row in families}) != len(families):
        errors.append("pilot root-cause family IDs are not unique")
    if sum(row["target_tasks"] for row in families) != pilot["population"]["target_task_count"]:
        errors.append("pilot root-cause family task targets do not sum to the target task count")
    custody = pilot["task_custody"]
    if custody["development_partition_tasks"] + custody["heldout_partition_tasks"] != pilot["population"]["target_task_count"]:
        errors.append("pilot development and holdout partitions do not sum to the target task count")
    if pilot["statistics"]["independent_seeds_minimum"] < 3:
        errors.append("pilot minimum independent seed count is below three")
    if pilot["status"] != "design_only_not_executed":
        errors.append("pilot must remain design_only_not_executed")
    if pilot["execution_gate"]["candidate_controlled_execution"] != "blocked_until_issue_10064_qualified":
        errors.append("candidate-controlled execution gate was weakened")
    if pilot["execution_gate"]["independent_regrade_claim"] != "blocked_until_issue_10369_qualified":
        errors.append("independent regrade trust gate was weakened")
    return errors


def pilot_self_test(pilot: dict, schema: dict, format_checker: FormatChecker) -> None:
    def must_reject(mutant: dict, label: str) -> None:
        if not validate_pilot_payload(mutant, schema, format_checker):
            raise SystemExit(f"pilot negative control unexpectedly passed: {label}")

    mutant = copy.deepcopy(pilot)
    mutant["population"]["root_cause_families"][0]["target_tasks"] += 1
    must_reject(mutant, "root-cause strata do not sum to 48")

    mutant = copy.deepcopy(pilot)
    mutant["task_custody"]["heldout_partition_tasks"] = 15
    must_reject(mutant, "development/holdout partition is incomplete")

    mutant = copy.deepcopy(pilot)
    mutant["status"] = "qualified"
    must_reject(mutant, "design-only pilot marked qualified")

    mutant = copy.deepcopy(pilot)
    mutant["execution_gate"]["candidate_controlled_execution"] = "allowed"
    must_reject(mutant, "OS isolation gate weakened")

    mutant = copy.deepcopy(pilot)
    mutant["statistics"]["independent_seeds_minimum"] = 2
    must_reject(mutant, "insufficient independent seed count")

    mutant = copy.deepcopy(pilot)
    mutant["statistics"]["pilot_is_exploratory"] = False
    must_reject(mutant, "pilot converted into an unsupported superiority study")

    print("CEB_RUST_PILOT_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base = root / "benchmarks" / "ceb"
    try:
        payload = load_json(base / "external-observations.v0.json")
        schema = load_json(base / "external-observations.schema.json")
        pilot = load_json(base / "rust-pilot-preregistration.v0.json")
        pilot_schema = load_json(base / "rust-pilot-preregistration.schema.json")
        Draft202012Validator.check_schema(schema)
        Draft202012Validator.check_schema(pilot_schema)
    except Exception as error:
        print(f"CEB_EXTERNAL_OBSERVATION_ERROR: {error}", file=sys.stderr)
        return 2
    format_checker = FormatChecker()
    if args.self_test:
        self_test(payload, schema, format_checker)
        pilot_self_test(pilot, pilot_schema, format_checker)
        return 0
    errors = validate_payload(payload, schema, format_checker)
    pilot_errors = validate_pilot_payload(pilot, pilot_schema, format_checker)
    if errors or pilot_errors:
        for error in errors:
            print(f"CEB_EXTERNAL_OBSERVATION_FAIL: {error}", file=sys.stderr)
        for error in pilot_errors:
            print(f"CEB_RUST_PILOT_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "CEB_EXTERNAL_OBSERVATION_PASS: "
        f"sources={len(payload['sources'])} "
        f"comparators={len(payload['comparators'])} "
        f"observations={len(payload['observations'])}"
    )
    print(
        "CEB_RUST_PILOT_PASS: "
        f"tasks={pilot['population']['target_task_count']} "
        f"families={len(pilot['population']['root_cause_families'])} "
        f"arms={len(pilot['baseline_arms'])} "
        f"holdout={pilot['task_custody']['heldout_partition_tasks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
