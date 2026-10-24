#!/usr/bin/env python3
"""Replay and independently score a saved synthetic casting scenario policy sweep.

This checker validates the saved fidelity ledger against the recorded analytic
scenario model, re-derives budget/promotion counts, and recomputes 2D hypervolume
without importing the policy runner's scorer. It does not establish external SOTA
or physical casting validity.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

from multifidelity_scenario_suite import (
    MultiFidelityScenario,
    SUPPORTED_DIMENSIONS,
    SUPPORTED_SCENARIOS,
)

DEFAULT_POLICY_MANIFEST = Path("docs/research/CASTING_MULTIFIDELITY_POLICY_SCENARIOS_V1.json")
DEFAULT_SCENARIO_MANIFEST = Path("docs/research/CASTING_MULTIFIDELITY_SCENARIO_SUITE_V1.json")
SOURCE_PATHS = (
    "tools/casting-benchmark/multifidelity_stress.py",
    "tools/casting-benchmark/multifidelity_scenario_suite.py",
    "tools/casting-benchmark/multifidelity_scenario_policy_bakeoff.py",
    "tools/casting-benchmark/score_scenario_policy_bakeoff.py",
    "docs/research/CASTING_MULTIFIDELITY_SCENARIO_SUITE_V1.json",
    "docs/research/CASTING_MULTIFIDELITY_POLICY_SCENARIOS_V1.json",
)
REFERENCE_POINT = (1.2, 1.2)
TOLERANCE = 1e-9

TOP_LEVEL_KEYS = {
    "benchmark_id", "synthetic_fixture_only", "scenario_ids", "dimensions", "severities",
    "seeds", "policies", "design_count", "cost_budget", "fidelity_costs", "run_count",
    "summary_count", "paired_comparison_count", "summaries", "paired_comparisons_vs_high_only",
    "paired_runs", "bootstrap_samples", "warning",
}
RUN_KEYS = {
    "benchmark_id", "synthetic_fixture_only", "policy", "scenario_id", "dimension", "severity",
    "seed", "design_count", "fidelity_costs", "cost_budget", "cost_spent", "unused_cost",
    "evaluation_count", "fidelity_evaluations", "fidelity_cost_units_by_level",
    "high_fidelity_call_count", "true_finite_pool_hypervolume",
    "observed_high_fidelity_hypervolume", "normalized_hypervolume_fraction",
    "audited_promotion_count", "predicted_feasible_promotion_count",
    "actual_feasible_promotion_count", "false_promotion_count", "false_promotion_rate",
    "promotion_records", "evaluations",
}
EVALUATION_KEYS = {
    "candidate_id", "fidelity", "objectives", "constraint_margin", "cost_units"
}
PROMOTION_KEYS = {
    "candidate_id", "predicted_margin", "method", "observed_high_margin"
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def loads_no_duplicate_keys(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_digests(repo_root: Path) -> dict[str, str]:
    """Hash all source and manifest inputs used to replay the benchmark."""
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = repo_root / relative
        hashes[relative] = sha256_bytes(path.read_bytes())
    return hashes


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _close(actual: Any, expected: float, label: str, errors: list[str]) -> None:
    if not _finite_number(actual) or abs(float(actual) - expected) > TOLERANCE:
        errors.append(f"{label} mismatch: expected {expected!r}, observed {actual!r}")


def _exact_keys(value: Any, expected: set[str], label: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return False
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        errors.append(f"{label} missing keys: {missing}")
    if unknown:
        errors.append(f"{label} has unknown keys: {unknown}")
    return not missing and not unknown


def hypervolume_2d(points: list[tuple[float, float]], reference: tuple[float, float] = REFERENCE_POINT) -> float:
    """Independent exact 2D minimization hypervolume implementation."""
    if len(reference) != 2 or not all(math.isfinite(v) for v in reference):
        raise ValueError("reference must contain two finite values")
    unique = sorted(set((float(x), float(y)) for x, y in points))
    front: list[tuple[float, float]] = []
    best_y = math.inf
    for x, y in unique:
        if x >= reference[0] or y >= reference[1]:
            continue
        if y < best_y:
            front.append((x, y))
            best_y = y
    area = 0.0
    for index, (x, y) in enumerate(front):
        next_x = front[index + 1][0] if index + 1 < len(front) else reference[0]
        area += max(0.0, next_x - x) * max(0.0, reference[1] - y)
    if not math.isfinite(area):
        raise ArithmeticError("non-finite hypervolume")
    return area


def _percentile(values: list[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("percentile requires non-empty values and q in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_interval(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values or samples < 100:
        raise ValueError("bootstrap needs values and at least 100 samples")
    rng = random.Random(seed)
    n = len(values)
    means = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    ]
    return _percentile(means, 0.025), _percentile(means, 0.975)


def _check_metric(run: dict, key: str, expected: Any, errors: list[str]) -> None:
    if expected is None:
        if run.get(key) is not None:
            errors.append(f"run {key} mismatch: expected None, observed {run.get(key)!r}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        _close(run.get(key), float(expected), f"run {key}", errors)
    elif run.get(key) != expected:
        errors.append(f"run {key} mismatch: expected {expected!r}, observed {run.get(key)!r}")


def _validate_and_score_run(
    run: Any,
    *,
    scenario_manifest: dict,
    allowed_scenarios: set[str],
    allowed_dimensions: set[int],
    allowed_severities: set[float],
    allowed_seeds: set[int],
    allowed_policies: set[str],
    cost_budget: int,
    fidelity_costs: dict[str, int],
) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    label = "run"
    if not _exact_keys(run, RUN_KEYS, label, errors):
        return None, errors
    if run["benchmark_id"] != "CASTING-MF-POLICY-SCENARIOS-001":
        errors.append("run benchmark_id is not CASTING-MF-POLICY-SCENARIOS-001")
    if run["synthetic_fixture_only"] is not True:
        errors.append("run must declare synthetic_fixture_only=true")
    scenario_id = run["scenario_id"]
    dimension = run["dimension"]
    severity = run["severity"]
    seed = run["seed"]
    policy = run["policy"]
    if not isinstance(scenario_id, str) or scenario_id not in allowed_scenarios or scenario_id not in SUPPORTED_SCENARIOS:
        errors.append(f"unsupported scenario_id: {scenario_id!r}")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension not in allowed_dimensions or dimension not in SUPPORTED_DIMENSIONS:
        errors.append(f"unsupported dimension: {dimension!r}")
    if not _finite_number(severity) or float(severity) not in allowed_severities:
        errors.append(f"unregistered severity: {severity!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in allowed_seeds:
        errors.append(f"unregistered seed: {seed!r}")
    if not isinstance(policy, str) or policy not in allowed_policies:
        errors.append(f"unregistered policy: {policy!r}")
    if isinstance(run["design_count"], bool) or not isinstance(run["design_count"], int) or run["design_count"] < 75:
        errors.append("design_count must be an integer at least 75")
    if run["cost_budget"] != cost_budget:
        errors.append("run cost_budget differs from top-level budget")
    if run["fidelity_costs"] != fidelity_costs:
        errors.append("run fidelity_costs differs from top-level cost map")

    if errors:
        return None, errors

    try:
        world = MultiFidelityScenario(
            scenario_id=scenario_id,
            dimension=dimension,
            severity=float(severity),
            seed=seed,
            fidelity_costs=fidelity_costs,
            design_count=int(run["design_count"]),
        )
    except (TypeError, ValueError) as exc:
        return None, [f"cannot construct registered scenario: {exc}"]

    evaluations = run["evaluations"]
    if not isinstance(evaluations, list):
        return None, ["run evaluations must be a list"]
    if len(evaluations) != run["evaluation_count"]:
        errors.append("evaluation_count differs from trace row count")

    counts = {name: 0 for name in ("low", "medium", "high")}
    costs = {name: 0 for name in ("low", "medium", "high")}
    latest_by_candidate_fidelity: dict[tuple[int, str], dict] = {}
    seen: set[tuple[int, str]] = set()
    for index, evaluation in enumerate(evaluations):
        row_label = f"evaluation[{index}]"
        if not _exact_keys(evaluation, EVALUATION_KEYS, row_label, errors):
            continue
        candidate_id = evaluation["candidate_id"]
        fidelity = evaluation["fidelity"]
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int) or candidate_id not in world.candidate_ids:
            errors.append(f"{row_label} has invalid candidate_id {candidate_id!r}")
            continue
        if not isinstance(fidelity, str) or fidelity not in counts:
            errors.append(f"{row_label} has unsupported fidelity {fidelity!r}")
            continue
        if not isinstance(evaluation["objectives"], list) or len(evaluation["objectives"]) != 2 or not all(_finite_number(v) for v in evaluation["objectives"]):
            errors.append(f"{row_label} objectives must be a finite two-element list")
            continue
        if not _finite_number(evaluation["constraint_margin"]):
            errors.append(f"{row_label} constraint_margin must be finite")
            continue
        if (
            isinstance(evaluation["cost_units"], bool)
            or not isinstance(evaluation["cost_units"], int)
            or evaluation["cost_units"] != fidelity_costs[fidelity]
        ):
            errors.append(f"{row_label} cost_units does not match the locked fidelity cost")
            continue
        key = (candidate_id, fidelity)
        if key in seen:
            errors.append(f"duplicate candidate/fidelity evaluation: {key}")
            continue
        seen.add(key)

        expected = world.evaluate(candidate_id, fidelity)
        for j in range(2):
            if abs(float(evaluation["objectives"][j]) - expected.objectives[j]) > TOLERANCE:
                errors.append(f"{row_label} objective {j} differs from scenario replay")
        if abs(float(evaluation["constraint_margin"]) - expected.constraint_margin) > TOLERANCE:
            errors.append(f"{row_label} constraint margin differs from scenario replay")
        if evaluation["cost_units"] != expected.cost_units:
            errors.append(f"{row_label} fidelity cost differs from scenario replay")
        counts[fidelity] += 1
        costs[fidelity] += int(evaluation["cost_units"])
        latest_by_candidate_fidelity[key] = evaluation

    if sum(counts.values()) != len(evaluations):
        errors.append("not all evaluation rows could be validated")
    if run["fidelity_evaluations"] != counts:
        errors.append(f"fidelity_evaluations mismatch: expected {counts}, observed {run['fidelity_evaluations']!r}")
    if run["fidelity_cost_units_by_level"] != costs:
        errors.append(f"fidelity_cost_units_by_level mismatch: expected {costs}, observed {run['fidelity_cost_units_by_level']!r}")
    cost_spent = sum(costs.values())
    if cost_spent > cost_budget:
        errors.append(f"budget overrun: spent {cost_spent}, budget {cost_budget}")
    _check_metric(run, "cost_spent", cost_spent, errors)
    _check_metric(run, "unused_cost", cost_budget - cost_spent, errors)
    _check_metric(run, "high_fidelity_call_count", counts["high"], errors)

    truth = [world.truth_for_audit_only(cid) for cid in world.candidate_ids]
    truth_feasible = [obs for obs in truth if obs.constraint_margin >= 0.0]
    true_hv = hypervolume_2d([obs.objectives for obs in truth_feasible])
    observed_high = [
        row for (candidate_id, fidelity), row in latest_by_candidate_fidelity.items()
        if fidelity == "high" and float(row["constraint_margin"]) >= 0.0
    ]
    observed_hv = hypervolume_2d([tuple(float(v) for v in row["objectives"]) for row in observed_high])
    normalized_hv = observed_hv / true_hv if true_hv > 0.0 else 0.0
    _check_metric(run, "true_finite_pool_hypervolume", true_hv, errors)
    _check_metric(run, "observed_high_fidelity_hypervolume", observed_hv, errors)
    _check_metric(run, "normalized_hypervolume_fraction", normalized_hv, errors)

    high_by_id = {
        candidate_id: row
        for (candidate_id, fidelity), row in latest_by_candidate_fidelity.items()
        if fidelity == "high"
    }
    promotions = run["promotion_records"]
    if not isinstance(promotions, list):
        errors.append("promotion_records must be a list")
        promotions = []
    promotion_ids: set[int] = set()
    predicted_feasible = 0
    actual_feasible = 0
    false_promotions = 0
    for index, promotion in enumerate(promotions):
        label = f"promotion[{index}]"
        if not _exact_keys(promotion, PROMOTION_KEYS, label, errors):
            continue
        candidate_id = promotion["candidate_id"]
        if isinstance(candidate_id, bool) or not isinstance(candidate_id, int) or candidate_id in promotion_ids:
            errors.append(f"{label} has invalid or duplicate candidate_id")
            continue
        promotion_ids.add(candidate_id)
        if candidate_id not in high_by_id:
            errors.append(f"{label} has no corresponding high-fidelity evaluation")
            continue
        predicted_margin = promotion["predicted_margin"]
        recorded_actual = promotion["observed_high_margin"]
        actual_trace_margin = float(high_by_id[candidate_id]["constraint_margin"])
        if not _finite_number(predicted_margin) or not _finite_number(recorded_actual):
            errors.append(f"{label} margins must be finite")
            continue
        if abs(float(recorded_actual) - actual_trace_margin) > TOLERANCE:
            errors.append(f"{label} observed_high_margin differs from high-fidelity trace")
        predicted_ok = float(predicted_margin) >= 0.0
        actual_ok = actual_trace_margin >= 0.0
        predicted_feasible += int(predicted_ok)
        actual_feasible += int(actual_ok)
        false_promotions += int(predicted_ok and not actual_ok)
        if not isinstance(promotion["method"], str) or not promotion["method"]:
            errors.append(f"{label} method must be a non-empty string")
    _check_metric(run, "audited_promotion_count", len(promotions), errors)
    _check_metric(run, "predicted_feasible_promotion_count", predicted_feasible, errors)
    _check_metric(run, "actual_feasible_promotion_count", actual_feasible, errors)
    _check_metric(run, "false_promotion_count", false_promotions, errors)
    false_rate = false_promotions / predicted_feasible if predicted_feasible else None
    _check_metric(run, "false_promotion_rate", false_rate, errors)

    scored = {
        "scenario_id": scenario_id,
        "dimension": dimension,
        "severity": float(severity),
        "seed": seed,
        "policy": policy,
        "normalized_hypervolume_fraction": normalized_hv,
        "true_finite_pool_hypervolume": true_hv,
        "observed_high_fidelity_hypervolume": observed_hv,
        "cost_spent": cost_spent,
        "unused_cost": cost_budget - cost_spent,
        "fidelity_evaluations": counts,
        "high_fidelity_call_count": counts["high"],
        "audited_promotion_count": len(promotions),
        "predicted_feasible_promotion_count": predicted_feasible,
        "actual_feasible_promotion_count": actual_feasible,
        "false_promotion_count": false_promotions,
        "false_promotion_rate": false_rate,
    }
    return scored, errors


def score_document(
    document: Any,
    scenario_manifest: dict,
    policy_manifest: dict,
    source_hashes: dict[str, str],
    input_sha256: str,
) -> tuple[dict, list[str]]:
    """Recompute each run metric and aggregate summaries from the raw nested ledger."""
    errors: list[str] = []
    if not _exact_keys(document, TOP_LEVEL_KEYS, "result document", errors):
        return {}, errors
    if document["benchmark_id"] != "CASTING-MF-POLICY-SCENARIOS-001":
        errors.append("unexpected benchmark_id")
    if document["synthetic_fixture_only"] is not True:
        errors.append("result must declare synthetic_fixture_only=true")
    if scenario_manifest.get("benchmark_id") != "CASTING-MF-SCENARIOS-001":
        errors.append("scenario manifest benchmark id mismatch")
    if policy_manifest.get("benchmark_id") != "CASTING-MF-POLICY-SCENARIOS-001":
        errors.append("policy manifest benchmark id mismatch")

    scenarios = document["scenario_ids"]
    dimensions = document["dimensions"]
    severities = document["severities"]
    seeds = document["seeds"]
    policies = document["policies"]
    if (
        not isinstance(scenarios, list) or not scenarios
        or any(not isinstance(value, str) or not value for value in scenarios)
        or len(set(scenarios)) != len(scenarios)
    ):
        errors.append("scenario_ids must be a non-empty unique list of strings")
        scenarios = []
    if set(scenarios) - set(scenario_manifest.get("default_sweep", {}).get("scenarios", [])):
        errors.append("scenario_ids contains scenarios absent from the locked manifest")
    if (
        not isinstance(dimensions, list) or not dimensions
        or any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions)
        or len(set(dimensions)) != len(dimensions)
    ):
        errors.append("dimensions must be a non-empty unique integer list")
        dimensions = []
    if set(dimensions) - set(scenario_manifest.get("design_space", {}).get("dimensions_supported", [])):
        errors.append("dimensions contains values absent from the locked manifest")
    if (
        not isinstance(severities, list) or not severities
        or any(not _finite_number(v) or v < 0 for v in severities)
        or len(set(severities)) != len(severities)
    ):
        errors.append("severities must be non-empty, unique, finite non-negative values")
        severities = []
    if (
        not isinstance(seeds, list) or not seeds
        or any(isinstance(v, bool) or not isinstance(v, int) for v in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        errors.append("seeds must be a non-empty unique integer list")
        seeds = []
    required_policies = [item["id"] for item in policy_manifest.get("policies", [])]
    if policies != required_policies:
        errors.append("policies must exactly match the frozen policy order")
        policies = []
    design_count = document["design_count"]
    cost_budget = document["cost_budget"]
    fidelity_costs = document["fidelity_costs"]
    if isinstance(design_count, bool) or not isinstance(design_count, int) or design_count < 75:
        errors.append("top-level design_count must be an integer >=75")
        design_count = 75
    if isinstance(cost_budget, bool) or not isinstance(cost_budget, int) or cost_budget < 1:
        errors.append("top-level cost_budget must be a positive integer")
        cost_budget = 1
    if not isinstance(fidelity_costs, dict) or set(fidelity_costs) != {"low", "medium", "high"} or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (fidelity_costs.values() if isinstance(fidelity_costs, dict) else [])):
        errors.append("top-level fidelity_costs must define positive integer low/medium/high costs")
        fidelity_costs = {"low": 1, "medium": 5, "high": 25}
    if isinstance(document["bootstrap_samples"], bool) or not isinstance(document["bootstrap_samples"], int) or document["bootstrap_samples"] < 100:
        errors.append("bootstrap_samples must be an integer >=100")

    expected_run_count = len(scenarios) * len(dimensions) * len(severities) * len(seeds) * len(policies)
    expected_summary_count = len(scenarios) * len(dimensions) * len(severities) * len(policies)
    expected_comparison_count = len(scenarios) * len(dimensions) * len(severities) * max(0, len(policies) - 1)
    _check_metric(document, "run_count", expected_run_count, errors)
    _check_metric(document, "summary_count", expected_summary_count, errors)
    _check_metric(document, "paired_comparison_count", expected_comparison_count, errors)
    if not isinstance(document["summaries"], list) or len(document["summaries"]) != expected_summary_count:
        errors.append("summaries list length does not match the registered design; receipt summaries are rebuilt from raw runs")
    if not isinstance(document["paired_comparisons_vs_high_only"], list) or len(document["paired_comparisons_vs_high_only"]) != expected_comparison_count:
        errors.append("paired comparisons list length does not match the registered design; receipt comparisons are rebuilt from raw runs")
    if not isinstance(document["paired_runs"], list) or len(document["paired_runs"]) != expected_run_count:
        errors.append("paired_runs length does not match the registered design")
        runs = document["paired_runs"] if isinstance(document["paired_runs"], list) else []
    else:
        runs = document["paired_runs"]

    scored_runs: list[dict] = []
    seen_keys: set[tuple] = set()
    for index, run in enumerate(runs):
        scored, run_errors = _validate_and_score_run(
            run,
            scenario_manifest=scenario_manifest,
            allowed_scenarios=set(scenarios),
            allowed_dimensions=set(dimensions),
            allowed_severities={float(v) for v in severities},
            allowed_seeds=set(seeds),
            allowed_policies=set(policies),
            cost_budget=cost_budget,
            fidelity_costs=fidelity_costs,
        )
        errors.extend(f"paired_runs[{index}]: {error}" for error in run_errors)
        if scored is None:
            continue
        key = (scored["scenario_id"], scored["dimension"], scored["severity"], scored["seed"], scored["policy"])
        if key in seen_keys:
            errors.append(f"duplicate policy-run key: {key}")
        seen_keys.add(key)
        scored_runs.append(scored)

    expected_keys = {
        (scenario, dimension, float(severity), seed, policy)
        for scenario in scenarios for dimension in dimensions
        for severity in severities for seed in seeds for policy in policies
    }
    if seen_keys != expected_keys:
        errors.append(
            f"run key coverage mismatch; missing={len(expected_keys - seen_keys)}, extra={len(seen_keys - expected_keys)}"
        )
    if errors:
        return {
            "receipt_schema_version": "1.0.0",
            "status": "REJECTED_TRACE_OR_CONTRACT_MISMATCH",
            "input_sha256": input_sha256,
            "source_sha256": source_hashes,
            "error_count": len(errors),
        }, errors

    indexed = {
        (row["scenario_id"], row["dimension"], row["severity"], row["seed"], row["policy"]): row
        for row in scored_runs
    }
    summaries: list[dict] = []
    comparisons: list[dict] = []
    bootstrap_samples = document["bootstrap_samples"]
    for scenario in scenarios:
        for dimension in dimensions:
            for severity in severities:
                for policy in policies:
                    group = [
                        indexed[(scenario, dimension, float(severity), seed, policy)]
                        for seed in seeds
                    ]
                    quality = [float(row["normalized_hypervolume_fraction"]) for row in group]
                    false_rates = [float(row["false_promotion_rate"]) for row in group if row["false_promotion_rate"] is not None]
                    summaries.append({
                        "scenario_id": scenario,
                        "dimension": dimension,
                        "severity": float(severity),
                        "policy": policy,
                        "paired_seed_count": len(group),
                        "median_normalized_hypervolume_fraction": statistics.median(quality),
                        "q25_normalized_hypervolume_fraction": _percentile(quality, 0.25),
                        "q75_normalized_hypervolume_fraction": _percentile(quality, 0.75),
                        "mean_false_promotion_rate": statistics.fmean(false_rates) if false_rates else None,
                        "mean_cost_spent": statistics.fmean(float(row["cost_spent"]) for row in group),
                        "mean_unused_cost": statistics.fmean(float(row["unused_cost"]) for row in group),
                        "mean_high_fidelity_calls": statistics.fmean(float(row["high_fidelity_call_count"]) for row in group),
                    })
                for policy in policies:
                    if policy == "high_only":
                        continue
                    differences = [
                        float(indexed[(scenario, dimension, float(severity), seed, policy)]["normalized_hypervolume_fraction"])
                        - float(indexed[(scenario, dimension, float(severity), seed, "high_only")]["normalized_hypervolume_fraction"])
                        for seed in seeds
                    ]
                    seed_tag = 991 + sum(ord(c) for c in scenario) + dimension * 17 + int(round(float(severity) * 10000)) + sum(ord(c) for c in policy)
                    ci_low, ci_high = _bootstrap_interval(differences, bootstrap_samples, seed_tag)
                    comparisons.append({
                        "scenario_id": scenario,
                        "dimension": dimension,
                        "severity": float(severity),
                        "reference_policy": "high_only",
                        "policy": policy,
                        "paired_seed_count": len(differences),
                        "mean_paired_hypervolume_fraction_delta": statistics.fmean(differences),
                        "median_paired_hypervolume_fraction_delta": statistics.median(differences),
                        "bootstrap_95_ci_mean_delta": {"lower": ci_low, "upper": ci_high},
                        "wins": sum(value > 1e-12 for value in differences),
                        "ties": sum(abs(value) <= 1e-12 for value in differences),
                        "losses": sum(value < -1e-12 for value in differences),
                        "uncertainty_note": "Exploratory unadjusted paired-seed interval; not proof of superiority.",
                    })

    receipt = {
        "receipt_schema_version": "1.0.0",
        "status": "SYNTHETIC_TRACE_REPLAYED_NOT_SOTA_EVIDENCE",
        "benchmark_id": "CASTING-MF-POLICY-SCENARIOS-001",
        "synthetic_fixture_only": True,
        "input_sha256": input_sha256,
        "source_sha256": source_hashes,
        "source_hash_semantics": "SHA-256 of exact local bytes at scoring time; these digests record provenance but do not constitute a digital signature or pre-run preregistration.",
        "scoring_method": "Post-run scenario replay, per-call cost/count reconciliation, exact independently implemented 2D HV recomputation, promotion-label recomputation, then summary/paired-statistic recomputation from scored run records.",
        "run_count": len(scored_runs),
        "summary_count_recomputed": len(summaries),
        "paired_comparison_count_recomputed": len(comparisons),
        "policy_comparison_scope": "Repository-internal policies only; external baselines and physical validation absent.",
        "scored_runs": scored_runs,
        "recomputed_summaries": summaries,
        "recomputed_paired_comparisons_vs_high_only": comparisons,
        "warning": "A valid replay proves trace consistency with the locked synthetic scenario code only. It is not evidence of external SOTA, foundry physics, or industrial benefit.",
    }
    return receipt, []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, help="saved JSON emitted by multifidelity_scenario_policy_bakeoff.py")
    parser.add_argument("--receipt", type=Path, default=None, help="write recomputed receipt JSON to this path")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        raw_bytes = args.result_json.read_bytes()
        document = loads_no_duplicate_keys(raw_bytes.decode("utf-8"))
        scenario_manifest = loads_no_duplicate_keys((repo_root / DEFAULT_SCENARIO_MANIFEST).read_text(encoding="utf-8"))
        policy_manifest = loads_no_duplicate_keys((repo_root / DEFAULT_POLICY_MANIFEST).read_text(encoding="utf-8"))
        hashes = source_digests(repo_root)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"SCORE_INPUT_INVALID: {exc}", file=sys.stderr)
        return 2
    receipt, errors = score_document(
        document,
        scenario_manifest,
        policy_manifest,
        hashes,
        sha256_bytes(raw_bytes),
    )
    output = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.receipt is not None:
        try:
            args.receipt.write_text(output, encoding="utf-8")
        except OSError as exc:
            print(f"RECEIPT_WRITE_FAILED: {exc}", file=sys.stderr)
            return 2
    if errors:
        for error in errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"ERROR: {len(errors)-100} more errors omitted", file=sys.stderr)
        print(f"TRACE_REJECTED: {len(errors)} integrity/contract error(s)", file=sys.stderr)
        return 3
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
