#!/usr/bin/env python3
"""Validate CEB's design-only benchmark-integrity and V4V control catalog.

This is a deterministic, offline contract check. It does not execute candidate
code, reference verifiers, build scripts, graders, or attack payloads.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


REQUIRED_V4V_CONTROLS = {
    "BI-N29": ("always_accept_verifier", "reject", "negative"),
    "BI-N30": ("always_reject_verifier", "reject", "negative"),
    "BI-N31": ("exact_match_only_verifier", "reject", "negative"),
    "BI-N32": ("mutation_insensitive_verifier", "reject", "negative"),
    "BI-N33": ("contradictory_identical_input_labels", "unresolved", "negative"),
    "BI-N34": ("reference_verifier_replay_mismatch", "reject", "negative"),
    "BI-N35": ("no_confirmed_positive_before_verifier_freeze", "unresolved", "negative"),
    "BI-N36": ("specification_composition_coverage_gap", "unresolved", "negative"),
    "BI-P09": ("valid_semantic_alternative_acceptance", "qualifying", "positive"),
    "BI-P10": ("reference_verifier_mutation_sensitivity", "qualifying", "positive"),
}

REQUIRED_EVAL_GATES = {
    "reference_verifier_replay_matches_adjudicated_positive_and_negative_labels",
    "repeatable_verifier_decisions_from_frozen_reset_state",
    "identical verifier-visible inputs cannot carry contradictory labels without adjudicated hidden-state distinction",
    "always_accept_always_reject_and_exact_match_only_verifier_baselines_must_fail",
    "necessary verifier mutations must change at least one preregistered decision",
    "at least one known-valid positive solution must be accepted before verifier freeze",
}

REQUIRED_EVAL_ARTIFACTS = {
    "reference_verifier_replay_matrix",
    "verifier_positive_negative_control_set",
    "verifier_mutation_results",
    "input_label_consistency_audit",
    "trivial_baseline_results",
    "confirmed_positive_verification_record",
}

REQUIRED_OPT_CONTROLS = {
    "BI-N37": ("benchmark_input_memorization", "reject", "negative"),
    "BI-N38": ("benchmark_harness_detection_or_bypass", "reject", "negative"),
    "BI-N39": ("workload_distribution_overfit", "reject", "negative"),
    "BI-P11": ("contract_valid_cache_memoization", "qualifying", "positive"),
}

REQUIRED_OPT_GATES = {
    "candidate must not detect benchmark invocation or alter, bypass or short-circuit the measurement harness",
    "held-out inputs cover preregistered in-spec sizes and distributions not disclosed to the candidate",
    "memoization and caching are accepted only when permitted by contract and correct under invalidation, collision, cold-start, unseen-input and resource controls",
    "raw optimization measurements and integrity-adjudicated status must be reported separately",
}

REQUIRED_OPT_ARTIFACTS = {
    "frozen_challenge_workload_distribution_digest",
    "benchmark_harness_integrity_review",
    "optimization_cache_semantics_test_evidence",
    "unseen_input_generalization_results",
    "optimization_integrity_adjudication",
}

REQUIRED_OPT_BASELINES = {
    "known_input_memorization_negative_control",
    "benchmark_harness_detection_negative_control",
    "legitimate_cache_memoization_positive_control",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def semantic_errors(catalog: dict[str, Any], matrix: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if catalog.get("schema_version") != 0 or catalog.get("benchmark_id") != "CEB":
        errors.append("catalog must remain CEB schema version 0")
    if catalog.get("catalog_id") != "CEB-BI-ATTACKS-001":
        errors.append("unexpected integrity catalog identity")
    if catalog.get("execution_status") != "design_only":
        errors.append("integrity catalog must remain design_only")
    if catalog.get("release") != "v0-design-only":
        errors.append("integrity catalog release must remain v0-design-only")

    controls = catalog.get("controls")
    if not isinstance(controls, list) or not controls:
        errors.append("controls must be a non-empty array")
        return errors

    ids: list[str] = []
    negative_count = 0
    positive_count = 0
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(controls):
        if not isinstance(row, dict):
            errors.append(f"controls[{index}] must be an object")
            continue
        control_id = row.get("id")
        if not isinstance(control_id, str) or not re.fullmatch(r"BI-[NP][0-9]{2}", control_id):
            errors.append(f"controls[{index}] has an invalid control ID")
            continue
        ids.append(control_id)
        by_id[control_id] = row
        polarity = row.get("polarity")
        if polarity == "negative":
            negative_count += 1
            if row.get("expected_verdict") not in {"reject", "unresolved"}:
                errors.append(f"{control_id}: negative control cannot have a qualifying expected verdict")
            if row.get("expected_qualification") not in {"reject", "unresolved"}:
                errors.append(f"{control_id}: negative control cannot be qualification-positive")
        elif polarity == "positive":
            positive_count += 1
            if row.get("expected_verdict") != "qualifying":
                errors.append(f"{control_id}: positive control must have qualifying expected verdict")
            if row.get("expected_qualification") != "qualifying":
                errors.append(f"{control_id}: positive control must have qualifying expected qualification")
        else:
            errors.append(f"{control_id}: polarity must be negative or positive")

        if row.get("status") != "design_only_not_executed":
            errors.append(f"{control_id}: control must remain design_only_not_executed")
        for field in ("threat_class", "target", "scenario", "required_evidence", "execution_profile"):
            if not isinstance(row.get(field), str) or len(row[field].strip()) < 3:
                errors.append(f"{control_id}: required descriptive field {field} is missing")

    if len(ids) != len(set(ids)):
        errors.append("duplicate integrity control IDs")

    minimum = catalog.get("minimum_initial_corpus", {})
    if not isinstance(minimum, dict):
        errors.append("minimum_initial_corpus must be an object")
    else:
        if negative_count < int(minimum.get("negative_controls", 20)):
            errors.append("negative control count is below the declared minimum")
        if positive_count < int(minimum.get("positive_controls", 5)):
            errors.append("positive control count is below the declared minimum")
        if minimum.get("authored_negative_controls") != negative_count:
            errors.append("authored_negative_controls does not match actual negative control count")
        if minimum.get("authored_positive_controls") != positive_count:
            errors.append("authored_positive_controls does not match actual positive control count")

    production = catalog.get("production_qualification_targets", {})
    if not isinstance(production, dict):
        errors.append("production qualification targets must be an object")
    elif production.get("distinct_negative_fault_families", 0) < negative_count:
        errors.append("distinct negative fault-family target must cover the authored negative corpus")

    for control_id, (threat, verdict, polarity) in REQUIRED_V4V_CONTROLS.items():
        row = by_id.get(control_id)
        if row is None:
            errors.append(f"missing required V4V control {control_id}")
            continue
        if row.get("threat_class") != threat:
            errors.append(f"{control_id}: V4V threat class changed")
        if row.get("expected_verdict") != verdict or row.get("expected_qualification") != verdict:
            errors.append(f"{control_id}: V4V expected disposition changed")
        if row.get("polarity") != polarity:
            errors.append(f"{control_id}: V4V polarity changed")

    if matrix.get("status") != "design_only":
        errors.append("measurement matrix must remain design_only")
    lanes = matrix.get("lanes")
    if not isinstance(lanes, list):
        errors.append("measurement matrix lanes must be an array")
        return errors
    eval_lanes = [lane for lane in lanes if isinstance(lane, dict) and lane.get("lane_id") == "CEB-EVAL"]
    if len(eval_lanes) != 1:
        errors.append("measurement matrix must contain exactly one CEB-EVAL lane")
        return errors
    eval_lane = eval_lanes[0]
    hard_gates = set(eval_lane.get("hard_gates", []))
    artifacts = set(eval_lane.get("evidence_contract", {}).get("required_artifacts", []))
    baseline_arms = set(eval_lane.get("baseline_arms", []))
    for gate in REQUIRED_EVAL_GATES:
        if gate not in hard_gates:
            errors.append(f"CEB-EVAL is missing V4V hard gate: {gate}")
    for artifact in REQUIRED_EVAL_ARTIFACTS:
        if artifact not in artifacts:
            errors.append(f"CEB-EVAL is missing V4V evidence artifact: {artifact}")
    required_baselines = {
        "always_accept_verifier_negative_control",
        "always_reject_verifier_negative_control",
        "exact_match_only_verifier_negative_control",
        "semantically_equivalent_alternative_solution_positive_control",
        "mutation_sensitivity_ablation",
    }
    for baseline in required_baselines - baseline_arms:
        errors.append(f"CEB-EVAL is missing V4V baseline arm: {baseline}")

    opt_lanes = [lane for lane in lanes if isinstance(lane, dict) and lane.get("lane_id") == "CEB-OPT"]
    if len(opt_lanes) != 1:
        errors.append("measurement matrix must contain exactly one CEB-OPT lane")
        return errors
    opt_lane = opt_lanes[0]
    opt_gates = set(opt_lane.get("hard_gates", []))
    opt_artifacts = set(opt_lane.get("evidence_contract", {}).get("required_artifacts", []))
    opt_baselines = set(opt_lane.get("baseline_arms", []))
    for gate in REQUIRED_OPT_GATES:
        if gate not in opt_gates:
            errors.append(f"CEB-OPT is missing optimization-integrity hard gate: {gate}")
    for artifact in REQUIRED_OPT_ARTIFACTS:
        if artifact not in opt_artifacts:
            errors.append(f"CEB-OPT is missing optimization-integrity evidence artifact: {artifact}")
    for baseline in REQUIRED_OPT_BASELINES - opt_baselines:
        errors.append(f"CEB-OPT is missing optimization-integrity baseline: {baseline}")

    for control_id, (threat, verdict, polarity) in REQUIRED_OPT_CONTROLS.items():
        row = by_id.get(control_id)
        if row is None:
            errors.append(f"missing required optimization-integrity control {control_id}")
            continue
        if row.get("threat_class") != threat:
            errors.append(f"{control_id}: optimization-integrity threat class changed")
        if row.get("expected_verdict") != verdict or row.get("expected_qualification") != verdict:
            errors.append(f"{control_id}: optimization-integrity expected disposition changed")
        if row.get("polarity") != polarity:
            errors.append(f"{control_id}: optimization-integrity polarity changed")

    return errors


def validate(catalog: dict[str, Any], matrix: dict[str, Any]) -> list[str]:
    return semantic_errors(catalog, matrix)


def self_test(catalog: dict[str, Any], matrix: dict[str, Any]) -> None:
    def must_reject(mutated_catalog: dict[str, Any], mutated_matrix: dict[str, Any], label: str) -> None:
        if not validate(mutated_catalog, mutated_matrix):
            raise SystemExit(f"integrity catalog negative control unexpectedly passed: {label}")

    mutant = copy.deepcopy(catalog)
    mutant["controls"][1]["id"] = mutant["controls"][0]["id"]
    must_reject(mutant, copy.deepcopy(matrix), "duplicate control identity")

    mutant = copy.deepcopy(catalog)
    mutant["execution_status"] = "qualified"
    must_reject(mutant, copy.deepcopy(matrix), "design-only catalog marked qualified")

    mutant = copy.deepcopy(catalog)
    row = next(item for item in mutant["controls"] if item["id"] == "BI-N30")
    row["expected_verdict"] = "qualifying"
    row["expected_qualification"] = "qualifying"
    must_reject(mutant, copy.deepcopy(matrix), "always-reject negative control relabeled positive")

    mutant = copy.deepcopy(catalog)
    mutant["controls"] = [item for item in mutant["controls"] if item["id"] != "BI-P09"]
    must_reject(mutant, copy.deepcopy(matrix), "semantic-alternative positive control removed")

    mutant_matrix = copy.deepcopy(matrix)
    lane = next(item for item in mutant_matrix["lanes"] if item["lane_id"] == "CEB-EVAL")
    lane["hard_gates"].remove(next(iter(REQUIRED_EVAL_GATES)))
    must_reject(copy.deepcopy(catalog), mutant_matrix, "V4V hard gate removed")

    mutant = copy.deepcopy(catalog)
    mutant["controls"] = [item for item in mutant["controls"] if item["id"] != "BI-P11"]
    must_reject(mutant, copy.deepcopy(matrix), "legitimate cache positive control removed")

    mutant_matrix = copy.deepcopy(matrix)
    opt_lane = next(item for item in mutant_matrix["lanes"] if item["lane_id"] == "CEB-OPT")
    opt_lane["hard_gates"].remove(next(iter(REQUIRED_OPT_GATES)))
    must_reject(copy.deepcopy(catalog), mutant_matrix, "optimization-integrity hard gate removed")

    print("CEB_INTEGRITY_CATALOG_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    base = root / "benchmarks" / "ceb"
    try:
        catalog = load_json(base / "integrity" / "attack-catalog.v0.json")
        matrix = load_json(base / "measurement-matrix.v0.json")
    except Exception as error:
        print(f"CEB_INTEGRITY_CATALOG_ERROR: {error}", file=sys.stderr)
        return 2

    errors = validate(catalog, matrix)
    if errors:
        for error in errors:
            print(f"CEB_INTEGRITY_CATALOG_FAIL: {error}", file=sys.stderr)
        return 1
    if args.self_test:
        self_test(catalog, matrix)
    negatives = sum(row.get("polarity") == "negative" for row in catalog["controls"])
    positives = sum(row.get("polarity") == "positive" for row in catalog["controls"])
    print(
        "CEB_INTEGRITY_CATALOG_PASS: "
        f"negative_controls={negatives} positive_controls={positives} "
        "v4v_controls=10 optimization_controls=4 eval_lane=CEB-EVAL opt_lane=CEB-OPT"
    )
    print("CEB_CLAIM_BOUNDARY: static catalog validation is not verifier execution or runner qualification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
