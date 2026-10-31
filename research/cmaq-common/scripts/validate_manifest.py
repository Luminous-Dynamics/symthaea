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


def semantic_errors(instance: dict) -> list[str]:
    errors: list[str] = []
    baseline_ids = [baseline["id"] for baseline in instance["baseline_registry"]]
    if len(baseline_ids) != len(set(baseline_ids)):
        errors.append("duplicate baseline IDs")

    baseline_set = set(baseline_ids)
    baseline_by_id = {baseline["id"]: baseline for baseline in instance["baseline_registry"]}
    for comparison in instance["comparisons"]:
        if comparison["treatment_id"] not in baseline_set:
            errors.append("comparison treatment ID is not registered")
        if comparison["control_id"] not in baseline_set:
            errors.append("comparison control ID is not registered")
        if comparison["status"] == "qualified":
            required_match = {"episodes", "seeds", "observations", "actions", "horizon"}
            comparison_baselines = (comparison["treatment_id"], comparison["control_id"])
            if any(baseline_id not in baseline_set for baseline_id in comparison_baselines):
                continue
            for baseline_id in comparison_baselines:
                baseline = baseline_by_id[baseline_id]
                if baseline["eligibility"] != "eligible_for_primary_contrast":
                    errors.append("qualified contrast uses an ineligible baseline")
                if baseline["kind"] == "oracle":
                    errors.append("qualified contrast cannot use evaluator oracle as a treatment/control")
            if not required_match.issubset(set(comparison["matched_dimensions"])):
                errors.append("qualified contrast is missing a core matched dimension")
            if "compute" in comparison["matched_dimensions"]:
                method = instance["budgets"]["matching_method"]
                if method not in {"measured_flops", "measured_wallclock", "matched_multi_resource"}:
                    errors.append("qualified compute contrast lacks a measured matching method")
                for baseline_id in (comparison["treatment_id"], comparison["control_id"]):
                    baseline = baseline_by_id.get(baseline_id)
                    if baseline is None or baseline["cost_accounting"] != "measured":
                        errors.append("qualified compute contrast includes an unmeasured baseline")

    build_status = instance["candidate"]["build_status"]
    artifact_digest = instance["candidate"]["artifact_sha256"]
    if build_status == "source_only_not_compiled" and artifact_digest is not None:
        errors.append("source-only candidate cannot claim a built artifact digest")
    if build_status in {"built_and_hashed", "external_frozen_artifact"} and artifact_digest is None:
        errors.append("built or external candidate must carry an artifact digest")

    if instance["qualification"]["stage"] == "qualification":
        qualification = instance["qualification"]
        if not (
            qualification["candidate_frozen"]
            and qualification["evaluator_frozen"]
            and qualification["thresholds_frozen"]
            and qualification["qualification_data_sealed"]
        ):
            errors.append("qualification stage requires frozen candidate/evaluator/thresholds and sealed data")

    split = instance["sample"]["mechanism_family_split"]
    if split["family_disjoint"]:
        development = set(split["development"])
        calibration = set(split["calibration"])
        qualification = set(split["qualification"])
        if development & calibration or development & qualification or calibration & qualification:
            errors.append("mechanism-family splits overlap despite family_disjoint=true")

    for source in instance["data_sources"]:
        if source["visibility"] == "evaluator_only" and source["candidate_access"]:
            errors.append("evaluator-only data source cannot expose candidate-access fields")
        if source["visibility"] == "candidate_visible" and not source["candidate_access"]:
            errors.append("candidate-visible data source must declare its visible fields")

    if instance["candidate"]["source_commit"] != instance["repository"]["commit"]:
        errors.append("candidate source commit differs from frozen repository commit")
    if instance["generator"]["source_commit"] != instance["repository"]["commit"]:
        errors.append("generator source commit differs from frozen repository commit")
    if instance["metrics"]["bundle_sha256"] != instance["evaluator"]["metric_bundle_sha256"]:
        errors.append("metric bundle digest differs between metrics and evaluator")
    if (
        instance["budgets"]["matching_method"] == "candidate_budget_only"
        and instance["budgets"]["measurement_status"] == "measured_hardware_costs"
    ):
        errors.append("candidate-count-only matching cannot claim measured hardware matching")
    if instance["qualification"]["candidate_truth_access"] is not False:
        errors.append("candidate truth access must remain disabled")
    if instance["metrics"]["cross_track_pooling"] is not False:
        errors.append("cross-track pooling must remain disabled")

    for baseline in instance["baseline_registry"]:
        if (
            baseline["cost_accounting"] == "candidate_count_only"
            and baseline["eligibility"] == "eligible_for_primary_contrast"
        ):
            errors.append("candidate-count-only baseline cannot be eligible as compute matched")
        if baseline["kind"] == "oracle" and baseline["eligibility"] != "oracle_only":
            errors.append("oracle baseline must remain oracle-only")
        if (
            baseline["eligibility"] == "eligible_for_primary_contrast"
            and baseline.get("ineligibility_reason") is not None
        ):
            errors.append("eligible baseline cannot carry an ineligibility reason")

    return errors


def fail_closed(validator: Draft202012Validator, instance: dict) -> bool:
    return not list(validator.iter_errors(instance)) and not semantic_errors(instance)


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

    semantic_failures = semantic_errors(example)
    if semantic_failures:
        raise SystemExit("example failed semantic checks: " + "; ".join(semantic_failures))

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

    mutated = copy.deepcopy(example)
    mutated["comparisons"][0]["status"] = "qualified"
    hostile_cases.append(("qualified contrast missing core matched dimensions", mutated))

    mutated = copy.deepcopy(example)
    mutated["comparisons"][0]["status"] = "qualified"
    mutated["comparisons"][0]["matched_dimensions"].append("horizon")
    mutated["comparisons"][0]["matched_dimensions"].append("compute")
    mutated["budgets"]["matching_method"] = "candidate_budget_only"
    hostile_cases.append(("qualified compute claim uses candidate-count-only budget", mutated))

    mutated = copy.deepcopy(example)
    mutated["qualification"]["stage"] = "qualification"
    mutated["qualification"]["qualification_data_sealed"] = False
    hostile_cases.append(("qualification stage without sealed data", mutated))

    mutated = copy.deepcopy(example)
    mutated["candidate"]["build_status"] = "built_and_hashed"
    mutated["candidate"]["artifact_sha256"] = None
    hostile_cases.append(("built candidate without artifact digest", mutated))

    mutated = copy.deepcopy(example)
    mutated["data_sources"][0]["visibility"] = "evaluator_only"
    mutated["data_sources"][0]["candidate_access"] = ["hidden_truth"]
    hostile_cases.append(("evaluator-only dataset exposes fields to candidate", mutated))

    mutated = copy.deepcopy(example)
    mutated["comparisons"][0]["status"] = "qualified"
    mutated["comparisons"][0]["treatment_id"] = "unregistered_model"
    hostile_cases.append(("qualified comparison references unknown baseline", mutated))

    for label, hostile in hostile_cases:
        if fail_closed(validator, hostile):
            raise SystemExit(f"hostile manifest was accepted: {label}")

    print(
        json.dumps(
            {
                "schema_version": schema["properties"]["schema_version"]["const"],
                "example_valid": True,
                "semantic_invariants_checked": "passed",
                "hostile_cases_rejected": len(hostile_cases),
                "ci_only": True,
                "performance_result_present": example["claim_boundary"]["performance_result_present"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
