#!/usr/bin/env python3
"""CI-only validation for the shared CMAQ pre-run freeze contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "frozen-run-manifest.schema.json"
EXAMPLE_PATH = ROOT / "examples" / "cmaq-018-development-manifest.example.json"


def fail_closed(validator: Draft202012Validator, instance: dict) -> bool:
    return not list(validator.iter_errors(instance))


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    errors = sorted(validator.iter_errors(example), key=lambda error: list(error.path))
    if errors:
        raise SystemExit(
            "example manifest failed schema validation:\n"
            + "\n".join(f"  {error.json_path}: {error.message}" for error in errors)
        )

    baseline_ids = [baseline["id"] for baseline in example["baseline_registry"]]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise SystemExit("duplicate baseline IDs")

    baseline_set = set(baseline_ids)
    for comparison in example["comparisons"]:
        if comparison["treatment_id"] not in baseline_set:
            raise SystemExit("comparison treatment ID is not registered")
        if comparison["control_id"] not in baseline_set:
            raise SystemExit("comparison control ID is not registered")

    for baseline in example["baseline_registry"]:
        if (
            baseline["cost_accounting"] == "candidate_count_only"
            and baseline["eligibility"] == "eligible_for_primary_contrast"
        ):
            raise SystemExit("candidate-count-only baseline cannot be eligible as compute matched")
        if baseline["kind"] == "oracle" and baseline["eligibility"] != "oracle_only":
            raise SystemExit("oracle baseline must remain oracle-only")
        if baseline["eligibility"] == "eligible_for_primary_contrast" and baseline["ineligibility_reason"] is not None:
            raise SystemExit("eligible baseline cannot carry an ineligibility reason")

    if example["candidate"]["source_commit"] != example["repository"]["commit"]:
        raise SystemExit("candidate source commit differs from frozen repository commit")
    if example["generator"]["source_commit"] != example["repository"]["commit"]:
        raise SystemExit("generator source commit differs from frozen repository commit")
    if example["metrics"]["bundle_sha256"] != example["evaluator"]["metric_bundle_sha256"]:
        raise SystemExit("metric bundle digest differs between metrics and evaluator")
    if example["budgets"]["matching_method"] == "candidate_budget_only":
        if example["budgets"]["measurement_status"] == "measured_hardware_costs":
            raise SystemExit("candidate-count-only matching cannot claim measured hardware matching")
    if example["qualification"]["candidate_truth_access"] is not False:
        raise SystemExit("candidate truth access must remain disabled")
    if example["metrics"]["cross_track_pooling"] is not False:
        raise SystemExit("cross-track pooling must remain disabled")

    hostile_cases = []

    mutated = copy.deepcopy(example)
    mutated["schema_version"] = "cmaq-frozen-run-manifest-v999"
    hostile_cases.append(("unknown schema version", mutated))

    mutated = copy.deepcopy(example)
    mutated["repository"]["dirty_tree"] = True
    hostile_cases.append(("dirty repository", mutated))

    mutated = copy.deepcopy(example)
    mutated["candidate"]["config_sha256"] = "not-a-sha256"
    hostile_cases.append(("malformed config digest", mutated))

    mutated = copy.deepcopy(example)
    mutated["qualification"]["candidate_truth_access"] = True
    hostile_cases.append(("candidate can access hidden truth", mutated))

    mutated = copy.deepcopy(example)
    mutated["metrics"]["cross_track_pooling"] = True
    hostile_cases.append(("cross-track scalar pooling", mutated))

    mutated = copy.deepcopy(example)
    mutated["task_contract"]["simulator_steps_per_action"] = 0
    hostile_cases.append(("invalid action cadence", mutated))

    mutated = copy.deepcopy(example)
    mutated["budgets"]["matching_method"] = "strictly_compute_matched"
    hostile_cases.append(("unknown compute matching class", mutated))

    mutated = copy.deepcopy(example)
    mutated["baseline_registry"][0]["eligibility"] = "eligible_for_primary_contrast"
    mutated["baseline_registry"][0]["cost_accounting"] = "candidate_count_only"
    mutated["baseline_registry"][0]["ineligibility_reason"] = None
    hostile_cases.append(("candidate count mislabeled as eligible matched baseline", mutated))

    mutated = copy.deepcopy(example)
    mutated["baseline_registry"][-1]["eligibility"] = "eligible_for_primary_contrast"
    mutated["baseline_registry"][-1]["ineligibility_reason"] = None
    hostile_cases.append(("oracle admitted as primary deployable baseline", mutated))

    for label, hostile in hostile_cases:
        if fail_closed(validator, hostile):
            raise SystemExit(f"hostile manifest was accepted: {label}")

    print(
        json.dumps(
            {
                "schema_version": schema["properties"]["schema_version"]["const"],
                "example_valid": True,
                "cross_reference_checks": 7,
                "hostile_cases_rejected": len(hostile_cases),
                "ci_only": True,
                "performance_result_present": example["claim_boundary"]["performance_result_present"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
