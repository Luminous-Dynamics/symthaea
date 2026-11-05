#!/usr/bin/env python3
"""Validate CEB comparative claims and statistical qualification gates.

Offline contract checks only. This tool does not run benchmark tasks, candidate
code, graders, models, statistical analyses, or claim-producing experiments.
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


def semantic_errors(claim: dict, crosswalk: dict) -> list[str]:
    errors: list[str] = []
    arms = claim["arms"]
    roles = [arm["role"] for arm in arms]
    if roles.count("treatment") != 1 or roles.count("control") != 1:
        errors.append("exactly one treatment and one control arm are required")

    # A comparison is not matched when its underlying evaluation inputs differ.
    digest_fields = (
        "task_set_sha256", "prompt_set_sha256", "source_set_sha256",
        "environment_sha256", "tool_policy_sha256", "resource_policy_sha256",
        "grader_sha256", "rubric_sha256",
    )
    for field in digest_fields:
        if arms[0][field] != arms[1][field]:
            errors.append(f"matched arms differ on {field}")

    design = claim["design"]
    stats = claim["statistics"]
    gates = claim["hard_gates"]
    claim_type = claim["claim_type"]
    confirmatory = claim_type in ("noninferiority", "superiority", "pareto_dominance")
    matched = claim_type in ("matched_comparison", "noninferiority", "superiority", "pareto_dominance")

    if matched and (not design["paired_tasks"] or not design["same_task_set"]):
        errors.append("matched claims require paired outcomes on the same task set")
    if design["effective_task_count"] != design["task_count"]:
        errors.append("repeated seeds cannot inflate the independent task denominator")
    if design["unit_of_analysis"] != "task_id":
        errors.append("the independent sampling unit must be task_id, not seed/run")

    if confirmatory:
        preregistered = (
            design["primary_endpoint_preregistered"]
            and design["one_primary_endpoint"]
            and design["primary_contrast_preregistered"]
        )
        if not preregistered:
            errors.append("confirmatory claim requires one preregistered endpoint and contrast")
        if not design["heldout_access_frozen"]:
            errors.append("confirmatory claim requires a frozen holdout")
        if design["repository_cluster_count"] <= 12:
            errors.append("12 or fewer repository clusters are exploratory under CEB V0")
        if stats["power_target"] < 0.90 or stats["estimated_power"] < 0.90 or not stats["power_analysis_sha256"]:
            errors.append("confirmatory claim requires a digest-bound >=90% power analysis")
        if stats["cluster_uncertainty_method"] not in (
            "pairs_cluster_bootstrap", "wild_cluster_restricted_bootstrap", "randomization_inference"
        ):
            errors.append("confirmatory claim requires cluster-aware or randomization-based uncertainty")
        if stats["secondary_metrics"] and stats["multiple_testing_adjustment"] != "holm":
            errors.append("confirmatory secondary metrics require Holm adjustment")
        if (
            not gates["isolation_qualified"]
            or not gates["trust_qualified"]
            or not gates["independent_regrade_qualified"]
        ):
            errors.append("qualified claims require isolation, trust-root and independent-regrade qualification")
        if gates["reference_patch_control"] != "pass" or gates["empty_patch_control"] != "fail":
            errors.append("qualified claims require reference-patch PASS and empty-patch FAIL")
        if gates["critical_security_failures"] > 0 or gates["critical_evaluator_failures"] > 0:
            errors.append("critical security/evaluator failures are non-compensable hard-gate failures")

    if stats["outcome_type"] == "binary":
        if stats["discordant_pairs"] < 25 and stats["binary_test"] != "mcnemar_exact":
            errors.append("fewer than 25 discordant pairs require exact McNemar/binomial inference")
        if stats["discordant_pairs"] >= 25 and stats["binary_test"] == "not_applicable":
            errors.append("paired binary outcomes require an explicit McNemar analysis")
    elif stats["binary_test"] != "not_applicable":
        errors.append("non-binary outcomes must not claim a McNemar test")

    if claim_type == "superiority":
        if stats["ci_lower"] <= stats["practical_margin"]:
            errors.append("superiority requires the lower confidence bound to exceed the practical margin")
        if claim["claim_status"] != "qualified_superiority" or claim["disposition"]["outcome"] != "superior":
            errors.append("superiority claim status/disposition is inconsistent")

    if claim_type == "noninferiority":
        margin = stats["noninferiority_margin"]
        if margin is None or stats["ci_lower"] <= -margin:
            errors.append("non-inferiority requires a preregistered margin and lower bound above its negative")
        if claim["claim_status"] != "qualified_noninferiority" or claim["disposition"]["outcome"] != "noninferior":
            errors.append("non-inferiority claim status/disposition is inconsistent")

    if claim_type == "pareto_dominance":
        vector = claim["pareto_vector"]
        if len(vector) < 2:
            errors.append("Pareto dominance requires at least two preregistered dimensions")
        if any(row["ci_lower"] <= -row["noninferiority_margin"] for row in vector):
            errors.append("Pareto dominance requires non-inferiority on every dimension")
        if not any(row["ci_lower"] > 0 for row in vector):
            errors.append("Pareto dominance requires strict improvement on at least one dimension")
        if claim["claim_status"] != "qualified_pareto" or claim["disposition"]["outcome"] != "pareto_dominant":
            errors.append("Pareto claim status/disposition is inconsistent")

    if claim_type == "exploratory":
        if claim["claim_status"] != "exploratory" or claim["disposition"]["outcome"] != "exploratory":
            errors.append("exploratory claim must remain explicitly exploratory")
    if claim_type == "descriptive":
        if claim["claim_status"] != "descriptive" or claim["disposition"]["outcome"] != "descriptive_only":
            errors.append("descriptive claim cannot be promoted to inferential status")
    if claim_type == "matched_comparison" and (
        claim["claim_status"] != "matched_only" or claim["disposition"]["outcome"] != "matched_only"
    ):
        errors.append("matched-comparison disposition must not imply superiority")

    if claim["comparison_scope"] == "external_benchmark_comparison" and matched:
        metric_id = claim["metric_identity"]["metric_id"]
        mapping = next(
            (row for row in crosswalk["metric_mappings"] if row["ceb_metric_id"] == metric_id),
            None,
        )
        if mapping is None or mapping["correspondence_class"] != "exact_semantics" or mapping["comparison_use"] != "replication_candidate":
            errors.append("external matched claim requires exact metric semantics and replication-candidate mapping")
        if claim["metric_identity"]["correspondence_class"] != "exact_semantics":
            errors.append("external matched claim cannot use a related/close/not-comparable metric")
        if claim["metric_identity"]["comparison_use"] != "replication_candidate":
            errors.append("external matched claim must be a replication candidate")
        cited = set(claim["metric_identity"]["external_observation_ids"])
        if not cited:
            errors.append("external comparison must identify matched external observation(s)")
        elif mapping is not None and not cited.issubset(set(mapping["external_observation_ids"])):
            errors.append("claim cites an external observation not registered for this metric")

    return errors


def validate_registry(registry: dict, registry_schema: dict) -> list[str]:
    validator = Draft202012Validator(registry_schema, format_checker=FormatChecker())
    issues = list(validator.iter_errors(registry))
    if issues:
        return [
            f"registry schema {'/'.join(map(str, issue.absolute_path)) or '<root>'}: {issue.message}"
            for issue in issues
        ]
    if registry["claims"]:
        return ["V0 claim registry must remain empty until qualified empirical evidence exists"]
    return []


def valid_superiority_fixture() -> dict:
    digest = "a" * 64

    def arm(role: str, system: str) -> dict:
        return {
            "role": role, "system_id": system, "model": "same pinned model family",
            "harness": "same pinned harness", "configuration": "matched budget",
            "task_set_sha256": digest, "prompt_set_sha256": "b" * 64,
            "source_set_sha256": "c" * 64, "environment_sha256": "d" * 64,
            "tool_policy_sha256": "e" * 64, "resource_policy_sha256": "f" * 64,
            "grader_sha256": "1" * 64, "rubric_sha256": "2" * 64,
        }

    return {
        "schema_version": 0,
        "claim_id": "ceb-claim-synthetic-superiority",
        "claim_type": "superiority",
        "claim_status": "qualified_superiority",
        "comparison_scope": "external_benchmark_comparison",
        "benchmark_identity": {
            "benchmark_id": "AgentAbstain", "version": "synthetic-matched-v1",
            "split": "synthetic-test-split", "task_count": 480, "task_set_sha256": digest,
        },
        "metric_identity": {
            "metric_id": "paired_act_abstain_accuracy",
            "correspondence_class": "exact_semantics",
            "comparison_use": "replication_candidate",
            "external_observation_ids": ["agentabstain-gemini-paired"],
        },
        "arms": [arm("treatment", "symthaea-treatment"), arm("control", "baseline-control")],
        "design": {
            "paired_tasks": True, "same_task_set": True, "unit_of_analysis": "task_id",
            "task_count": 480, "effective_task_count": 480,
            "repository_cluster_count": 16, "root_cause_cluster_count": 32,
            "seeds_per_task": 3, "primary_endpoint_preregistered": True,
            "one_primary_endpoint": True, "primary_contrast_preregistered": True,
            "heldout_access_frozen": True, "preregistration_sha256": "3" * 64,
        },
        "statistics": {
            "outcome_type": "binary", "binary_test": "mcnemar_exact", "discordant_pairs": 20,
            "cluster_uncertainty_method": "wild_cluster_restricted_bootstrap",
            "effect_estimate": 0.10, "ci_level": 0.95, "ci_lower": 0.06, "ci_upper": 0.14,
            "practical_margin": 0.05, "noninferiority_margin": None,
            "power_target": 0.90, "estimated_power": 0.92, "power_analysis_sha256": "4" * 64,
            "secondary_metrics": [], "multiple_testing_adjustment": "none",
        },
        "hard_gates": {
            "isolation_qualified": True, "trust_qualified": True,
            "independent_regrade_qualified": True, "reference_patch_control": "pass",
            "empty_patch_control": "fail", "critical_security_failures": 0,
            "critical_evaluator_failures": 0,
        },
        "pareto_vector": [],
        "claim_statement": "Synthetic validator fixture only; not an actual benchmark result or a CEB claim.",
        "disposition": {
            "outcome": "superior",
            "reason": "Synthetic positive fixture to validate claim-gate logic only.",
        },
    }


def self_test(registry: dict, registry_schema: dict, claim_schema: dict, crosswalk: dict) -> None:
    if validate_registry(registry, registry_schema):
        raise SystemExit("valid design-only claim registry unexpectedly failed")

    fixture = valid_superiority_fixture()
    errors = semantic_errors(fixture, crosswalk)
    schema_errors = list(Draft202012Validator(
        claim_schema, format_checker=FormatChecker()
    ).iter_errors(fixture))
    if errors or schema_errors:
        messages = errors + [issue.message for issue in schema_errors]
        raise SystemExit("valid synthetic superiority fixture failed: " + "; ".join(messages))

    def must_reject(mutant: dict, label: str) -> None:
        invalid_schema = list(Draft202012Validator(
            claim_schema, format_checker=FormatChecker()
        ).iter_errors(mutant))
        invalid_semantics = semantic_errors(mutant, crosswalk)
        if not invalid_schema and not invalid_semantics:
            raise SystemExit(f"comparative claim negative control unexpectedly passed: {label}")

    mutant = copy.deepcopy(fixture)
    mutant["design"]["repository_cluster_count"] = 6
    must_reject(mutant, "six repository pilot promoted to confirmatory superiority")

    mutant = copy.deepcopy(fixture)
    mutant["design"]["paired_tasks"] = False
    must_reject(mutant, "unpaired task comparison")

    mutant = copy.deepcopy(fixture)
    mutant["design"]["effective_task_count"] = 1440
    must_reject(mutant, "seeds counted as independent tasks")

    mutant = copy.deepcopy(fixture)
    mutant["design"]["primary_endpoint_preregistered"] = False
    must_reject(mutant, "post-hoc primary endpoint")

    mutant = copy.deepcopy(fixture)
    mutant["statistics"]["secondary_metrics"] = ["cost_per_accepted_repair"]
    mutant["statistics"]["multiple_testing_adjustment"] = "none"
    must_reject(mutant, "unadjusted secondary metric claim")

    mutant = copy.deepcopy(fixture)
    mutant["metric_identity"]["correspondence_class"] = "related_but_distinct"
    mutant["metric_identity"]["comparison_use"] = "context_only"
    must_reject(mutant, "external superiority claim on a contextual metric")

    mutant = copy.deepcopy(fixture)
    mutant["hard_gates"]["critical_security_failures"] = 1
    must_reject(mutant, "critical security failure compensated by performance")

    mutant = copy.deepcopy(fixture)
    mutant["statistics"]["ci_lower"] = 0.03
    must_reject(mutant, "confidence bound fails practical superiority margin")

    mutant = copy.deepcopy(fixture)
    mutant["statistics"]["discordant_pairs"] = 12
    mutant["statistics"]["binary_test"] = "mcnemar_asymptotic"
    must_reject(mutant, "asymptotic McNemar with sparse discordant pairs")

    mutant = copy.deepcopy(fixture)
    mutant["hard_gates"]["trust_qualified"] = False
    must_reject(mutant, "independent claim before trust qualification")

    exploratory = copy.deepcopy(fixture)
    exploratory["comparison_scope"] = "internal_component_ablation"
    exploratory["metric_identity"]["correspondence_class"] = "related_but_distinct"
    exploratory["metric_identity"]["comparison_use"] = "context_only"
    exploratory["design"]["repository_cluster_count"] = 6
    exploratory["claim_type"] = "exploratory"
    exploratory["claim_status"] = "exploratory"
    exploratory["disposition"]["outcome"] = "exploratory"
    if semantic_errors(exploratory, crosswalk):
        raise SystemExit("valid six-repository exploratory fixture unexpectedly failed")

    print("CEB_COMPARATIVE_CLAIM_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    base = args.repo_root.resolve() / "benchmarks" / "ceb"
    try:
        registry = load(base / "comparative-claim-registry.v0.json")
        registry_schema = load(base / "comparative-claim-registry.schema.json")
        claim_schema = load(base / "comparative-claim.schema.json")
        crosswalk = load(base / "metric-correspondence.v0.json")
        Draft202012Validator.check_schema(registry_schema)
        Draft202012Validator.check_schema(claim_schema)
    except Exception as error:
        print(f"CEB_COMPARATIVE_CLAIM_ERROR: {error}", file=sys.stderr)
        return 2

    errors = validate_registry(registry, registry_schema)
    if errors:
        for error in errors:
            print(f"CEB_COMPARATIVE_CLAIM_FAIL: {error}", file=sys.stderr)
        return 1
    if args.self_test:
        self_test(registry, registry_schema, claim_schema, crosswalk)

    print(
        "CEB_COMPARATIVE_CLAIM_PASS: "
        f"registered_claims={len(registry['claims'])} status={registry['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
