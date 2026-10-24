#!/usr/bin/env python3
"""Run internal casting proposal policies across paired adversarial fidelity scenarios.

This is a synthetic policy bake-off, not an external SOTA comparison or casting simulator.
It uses existing policy implementations and the same candidate pool within each scenario-seed.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Iterable

from multifidelity_stress import (
    FIDELITY_COSTS,
    FIDELITIES,
    POLICIES,
    PolicyEvaluator,
    _paired_bootstrap_mean_interval,
    hypervolume_2d,
    normalize_fidelity_costs,
    required_policy_costs,
)
from multifidelity_scenario_suite import (
    DEFAULT_DESIGN_COUNT,
    DEFAULT_DIMENSIONS,
    SUPPORTED_SCENARIOS,
    DEFAULT_SEEDS,
    DEFAULT_SEVERITIES,
    MultiFidelityScenario,
    SUPPORTED_DIMENSIONS,
)


def run_policy_on_scenario(
    policy_name: str,
    scenario_id: str,
    dimension: int,
    severity: float,
    seed: int,
    cost_budget: int = 200,
    fidelity_costs: dict[str, int] | None = None,
    design_count: int = DEFAULT_DESIGN_COUNT,
) -> dict:
    """Run one policy, then audit its selected high-fidelity archive against hidden truth."""
    if policy_name not in POLICIES:
        raise ValueError(f"unknown policy: {policy_name}")
    costs = normalize_fidelity_costs(fidelity_costs)
    if cost_budget < required_policy_costs(costs)[policy_name]:
        raise ValueError(
            f"cost_budget is below the frozen template minimum for policy {policy_name}"
        )
    world = MultiFidelityScenario(
        scenario_id=scenario_id,
        dimension=dimension,
        severity=severity,
        seed=seed,
        fidelity_costs=costs,
        design_count=design_count,
    )
    evaluator = PolicyEvaluator(world, cost_budget=cost_budget)
    POLICIES[policy_name](evaluator, seed)

    # Only after the policy has finished do we query analytic truth for independent scoring.
    truth = [world.truth_for_audit_only(cid) for cid in evaluator.candidate_ids]
    feasible_truth = [obs for obs in truth if obs.constraint_margin >= 0.0]
    true_hv = hypervolume_2d([obs.objectives for obs in feasible_truth])
    observed_high = [obs for obs in evaluator.high_observations if obs.constraint_margin >= 0.0]
    observed_hv = hypervolume_2d([obs.objectives for obs in observed_high])

    call_counts = {fidelity: 0 for fidelity in FIDELITIES}
    cost_totals = {fidelity: 0 for fidelity in FIDELITIES}
    for record in evaluator.records:
        call_counts[record.fidelity] += 1
        cost_totals[record.fidelity] += record.cost_units

    basis = evaluator.promotion_basis
    high_by_id = {obs.candidate_id: obs for obs in evaluator.high_observations}
    audited_promotions = [
        (candidate_id, predicted_margin, method, high_by_id[candidate_id])
        for candidate_id, (predicted_margin, method) in basis.items()
        if candidate_id in high_by_id
    ]
    false_promotions = sum(
        predicted_margin >= 0.0 and actual.constraint_margin < 0.0
        for _, predicted_margin, _, actual in audited_promotions
    )
    predicted_feasible_promotions = sum(
        predicted_margin >= 0.0
        for _, predicted_margin, _, _ in audited_promotions
    )
    actual_feasible_promotions = sum(
        actual.constraint_margin >= 0.0
        for _, _, _, actual in audited_promotions
    )

    return {
        "benchmark_id": "CASTING-MF-POLICY-SCENARIOS-001",
        "synthetic_fixture_only": True,
        "policy": policy_name,
        "scenario_id": scenario_id,
        "dimension": dimension,
        "severity": severity,
        "seed": seed,
        "design_count": design_count,
        "fidelity_costs": costs,
        "cost_budget": cost_budget,
        "cost_spent": evaluator.cost_spent,
        "unused_cost": cost_budget - evaluator.cost_spent,
        "evaluation_count": len(evaluator.records),
        "fidelity_evaluations": call_counts,
        "fidelity_cost_units_by_level": cost_totals,
        "high_fidelity_call_count": call_counts["high"],
        "true_finite_pool_hypervolume": true_hv,
        "observed_high_fidelity_hypervolume": observed_hv,
        "normalized_hypervolume_fraction": observed_hv / true_hv if true_hv > 0.0 else 0.0,
        "audited_promotion_count": len(audited_promotions),
        "predicted_feasible_promotion_count": predicted_feasible_promotions,
        "actual_feasible_promotion_count": actual_feasible_promotions,
        "false_promotion_count": false_promotions,
        "false_promotion_rate": (
            false_promotions / predicted_feasible_promotions
            if predicted_feasible_promotions else None
        ),
        "evaluations": [record.__dict__ if hasattr(record, "__dict__") else {
            "candidate_id": record.candidate_id,
            "fidelity": record.fidelity,
            "objectives": record.objectives,
            "constraint_margin": record.constraint_margin,
            "cost_units": record.cost_units,
        } for record in evaluator.records],
    }


def _percentile(values: list[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("percentile requires a non-empty list and q in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    fraction = position - lo
    return ordered[lo] * (1.0 - fraction) + ordered[hi] * fraction


def run_policy_sweep(
    scenario_ids: Iterable[str] = SUPPORTED_SCENARIOS,
    dimensions: Iterable[int] = DEFAULT_DIMENSIONS,
    severities: Iterable[float] = DEFAULT_SEVERITIES,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    cost_budget: int | None = None,
    fidelity_costs: dict[str, int] | None = None,
    design_count: int = DEFAULT_DESIGN_COUNT,
    bootstrap_samples: int = 1000,
) -> dict:
    scenarios = tuple(scenario_ids)
    dims = tuple(dimensions)
    severity_values = tuple(float(v) for v in severities)
    seed_values = tuple(seeds)
    costs = normalize_fidelity_costs(fidelity_costs)
    if not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenario_ids must be non-empty and unique")
    if set(scenarios) - set(SUPPORTED_SCENARIOS):
        raise ValueError("scenario_ids contains an unsupported scenario")
    if not dims or len(set(dims)) != len(dims) or set(dims) - set(SUPPORTED_DIMENSIONS):
        raise ValueError(f"dimensions must be unique members of {SUPPORTED_DIMENSIONS}")
    if not severity_values or any(not math.isfinite(v) or v < 0.0 for v in severity_values):
        raise ValueError("severities must be non-empty, finite and non-negative")
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be non-empty and unique")
    if design_count < 75:
        raise ValueError("design_count must be at least 75 for all frozen proposal policies")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")

    required = required_policy_costs(costs)
    minimum_budget = max(required.values())
    if cost_budget is None:
        cost_budget = minimum_budget
    if cost_budget < minimum_budget:
        raise ValueError(f"cost_budget must be at least {minimum_budget} for all frozen policy templates")

    runs = [
        run_policy_on_scenario(
            policy_name=policy,
            scenario_id=scenario,
            dimension=dimension,
            severity=severity,
            seed=seed,
            cost_budget=cost_budget,
            fidelity_costs=costs,
            design_count=design_count,
        )
        for scenario in scenarios
        for dimension in dims
        for severity in severity_values
        for seed in seed_values
        for policy in POLICIES
    ]

    indexed = {
        (row["scenario_id"], row["dimension"], row["severity"], row["seed"], row["policy"]): row
        for row in runs
    }
    summaries: list[dict] = []
    comparisons: list[dict] = []
    for scenario in scenarios:
        for dimension in dims:
            for severity in severity_values:
                for policy in POLICIES:
                    group = [
                        indexed[(scenario, dimension, severity, seed, policy)]
                        for seed in seed_values
                    ]
                    values = [float(row["normalized_hypervolume_fraction"]) for row in group]
                    promotion_rates = [
                        float(row["false_promotion_rate"]) for row in group
                        if row["false_promotion_rate"] is not None
                    ]
                    summaries.append({
                        "scenario_id": scenario,
                        "dimension": dimension,
                        "severity": severity,
                        "policy": policy,
                        "paired_seed_count": len(group),
                        "median_normalized_hypervolume_fraction": statistics.median(values),
                        "q25_normalized_hypervolume_fraction": _percentile(values, 0.25),
                        "q75_normalized_hypervolume_fraction": _percentile(values, 0.75),
                        "mean_false_promotion_rate": (
                            statistics.fmean(promotion_rates) if promotion_rates else None
                        ),
                        "mean_cost_spent": statistics.fmean(float(row["cost_spent"]) for row in group),
                        "mean_unused_cost": statistics.fmean(float(row["unused_cost"]) for row in group),
                        "mean_high_fidelity_calls": statistics.fmean(
                            float(row["high_fidelity_call_count"]) for row in group
                        ),
                    })
                for policy in POLICIES:
                    if policy == "high_only":
                        continue
                    differences = [
                        float(indexed[(scenario, dimension, severity, seed, policy)]["normalized_hypervolume_fraction"])
                        - float(indexed[(scenario, dimension, severity, seed, "high_only")]["normalized_hypervolume_fraction"])
                        for seed in seed_values
                    ]
                    bootstrap_seed = (
                        8803 + sum(ord(c) for c in scenario) + dimension * 11
                        + int(round(severity * 1_000_000)) + sum(ord(c) for c in policy)
                    )
                    ci_low, ci_high = _paired_bootstrap_mean_interval(
                        differences, bootstrap_samples=bootstrap_samples, seed=bootstrap_seed
                    )
                    comparisons.append({
                        "scenario_id": scenario,
                        "dimension": dimension,
                        "severity": severity,
                        "reference_policy": "high_only",
                        "policy": policy,
                        "paired_seed_count": len(differences),
                        "mean_paired_hypervolume_fraction_delta": statistics.fmean(differences),
                        "median_paired_hypervolume_fraction_delta": statistics.median(differences),
                        "bootstrap_95_ci_mean_delta": {"lower": ci_low, "upper": ci_high},
                        "wins": sum(v > 1e-12 for v in differences),
                        "ties": sum(abs(v) <= 1e-12 for v in differences),
                        "losses": sum(v < -1e-12 for v in differences),
                        "uncertainty_note": (
                            "Exploratory paired-seed percentile-bootstrap interval; not adjusted "
                            "for multiple comparisons and not evidence of external SOTA or physical validity."
                        ),
                    })

    return {
        "benchmark_id": "CASTING-MF-POLICY-SCENARIOS-001",
        "synthetic_fixture_only": True,
        "scenario_ids": list(scenarios),
        "dimensions": list(dims),
        "severities": list(severity_values),
        "seeds": list(seed_values),
        "policies": list(POLICIES),
        "design_count": design_count,
        "cost_budget": cost_budget,
        "fidelity_costs": costs,
        "run_count": len(runs),
        "summary_count": len(summaries),
        "paired_comparison_count": len(comparisons),
        "summaries": summaries,
        "paired_comparisons_vs_high_only": comparisons,
        "paired_runs": runs,
        "bootstrap_samples": bootstrap_samples,
        "warning": (
            "Synthetic internal-policy comparison only. This does not execute BoTorch/COCO baselines, "
            "does not establish SOTA, and does not validate foundry physics."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=",".join(SUPPORTED_SCENARIOS))
    parser.add_argument("--dimensions", default=",".join(str(v) for v in DEFAULT_DIMENSIONS))
    parser.add_argument("--severities", default=",".join(str(v) for v in DEFAULT_SEVERITIES))
    parser.add_argument("--seeds", default=",".join(str(v) for v in DEFAULT_SEEDS))
    parser.add_argument("--cost-budget", type=int, default=None)
    parser.add_argument("--fidelity-costs", default="low=1,medium=5,high=25")
    parser.add_argument("--design-count", type=int, default=DEFAULT_DESIGN_COUNT)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    parsed_costs: dict[str, int] = {}
    for entry in args.fidelity_costs.split(","):
        if "=" not in entry:
            raise ValueError("each fidelity cost must use name=integer format")
        name, value = entry.split("=", 1)
        name = name.strip()
        if name in parsed_costs:
            raise ValueError(f"duplicate fidelity cost: {name}")
        parsed_costs[name] = int(value.strip())
    result = run_policy_sweep(
        scenario_ids=[v.strip() for v in args.scenarios.split(",") if v.strip()],
        dimensions=[int(v.strip()) for v in args.dimensions.split(",") if v.strip()],
        severities=[float(v.strip()) for v in args.severities.split(",") if v.strip()],
        seeds=[int(v.strip()) for v in args.seeds.split(",") if v.strip()],
        cost_budget=args.cost_budget,
        fidelity_costs=parsed_costs,
        design_count=args.design_count,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
