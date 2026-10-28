#!/usr/bin/env python3
"""Compositional multi-fidelity stress bake-off (synthetic only).

The objective-discrepancy family and feasibility-discrepancy family are crossed
independently. Off-diagonal cells create error combinations not seen in the
single-family benchmark. The runner exposes no audit-truth call to a policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import asdict
from typing import Iterable

from multifidelity_stress import (
    POLICIES as EXISTING_POLICIES,
    PolicyEvaluator,
    REFERENCE_POINT,
    hypervolume_2d,
    normalize_fidelity_costs,
    required_policy_costs,
)
from multifidelity_decision_value_policy import (
    POLICY_ID as DECISION_VALUE_POLICY_ID,
    decision_value_of_information_policy,
)
from multifidelity_scenario_suite import (
    DEFAULT_DESIGN_COUNT,
    DEFAULT_DIMENSIONS,
    DEFAULT_SEEDS,
    DEFAULT_SEVERITIES,
    FIDELITY_COSTS,
    Observation,
    MultiFidelityScenario,
    SUPPORTED_DIMENSIONS,
    SUPPORTED_SCENARIOS,
)

BENCHMARK_ID = "CASTING-MF-COMPOSITION-001"
DEFAULT_COMPOSITION_SEVERITIES = (0.0, *DEFAULT_SEVERITIES)
POLICIES = {
    **EXISTING_POLICIES,
    DECISION_VALUE_POLICY_ID: decision_value_of_information_policy,
}


def _required_policy_costs(costs: dict[str, int]) -> dict[str, int]:
    required = required_policy_costs(costs)
    required[DECISION_VALUE_POLICY_ID] = (
        60 * costs["low"] + 8 * costs["medium"] + 4 * costs["high"]
    )
    return required


SOURCE_PATHS = (
    "tools/casting-benchmark/multifidelity_compositional_bakeoff.py",
    "tools/casting-benchmark/multifidelity_decision_value_policy.py",
    "tools/casting-benchmark/score_compositional_bakeoff.py",
    "tools/casting-benchmark/multifidelity_scenario_suite.py",
    "tools/casting-benchmark/multifidelity_stress.py",
    "docs/research/CASTING_MULTIFIDELITY_SCENARIO_SUITE_V1.json",
    "docs/research/CASTING_COMPOSITIONAL_BAKEOFF_V1.json",
)


def source_lock() -> dict[str, str]:
    """Capture source/manifest bytes before any optimizer evaluations occur."""
    root = Path(__file__).resolve().parents[2]
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in SOURCE_PATHS
    }


class CompositionalScenarioWorld:
    """Compose objective errors from one family and margin errors from another."""

    def __init__(
        self,
        objective_family: str,
        constraint_family: str,
        dimension: int,
        severity: float,
        seed: int,
        fidelity_costs: dict[str, int],
        design_count: int,
    ) -> None:
        self.objective_family = objective_family
        self.constraint_family = constraint_family
        self.dimension = dimension
        self.severity = severity
        self.seed = seed
        self._costs = dict(fidelity_costs)
        kwargs = {
            "dimension": dimension,
            "severity": severity,
            "seed": seed,
            "fidelity_costs": fidelity_costs,
            "design_count": design_count,
        }
        self._objective_world = MultiFidelityScenario(objective_family, **kwargs)
        self._constraint_world = MultiFidelityScenario(constraint_family, **kwargs)
        self._truth_world = MultiFidelityScenario("smooth_stationary", **kwargs)

    @property
    def fidelity_costs(self) -> dict[str, int]:
        return dict(self._costs)

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return self._objective_world.candidate_ids

    def design(self, candidate_id: int):
        return self._objective_world.design(candidate_id)

    def _response(self, candidate_id: int, fidelity: str) -> Observation:
        # The policy sees the objective family's objective values and the
        # constraint family's margin. At high fidelity both equal analytic truth.
        objective = self._objective_world.evaluate(candidate_id, fidelity)
        constraint = self._constraint_world.evaluate(candidate_id, fidelity)
        return Observation(
            candidate_id=candidate_id,
            fidelity=fidelity,
            objectives=objective.objectives,
            constraint_margin=constraint.constraint_margin,
            cost_units=self._costs[fidelity],
        )

    def truth_for_audit_only(self, candidate_id: int) -> Observation:
        """Audit-only truth; call only after the policy has stopped."""
        return self._truth_world.truth_for_audit_only(candidate_id)


def _digest_candidate_pool(world: CompositionalScenarioWorld) -> str:
    payload = [
        [design.candidate_id, list(design.x)]
        for design in (world.design(cid) for cid in world.candidate_ids)
    ]
    raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def run_one(
    objective_family: str,
    constraint_family: str,
    dimension: int,
    severity: float,
    seed: int,
    policy: str,
    cost_budget: int,
    fidelity_costs: dict[str, int],
    design_count: int,
) -> dict:
    if objective_family not in SUPPORTED_SCENARIOS or constraint_family not in SUPPORTED_SCENARIOS:
        raise ValueError("unsupported objective or constraint mismatch family")
    if dimension not in SUPPORTED_DIMENSIONS:
        raise ValueError(f"dimension must be one of {SUPPORTED_DIMENSIONS}")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if design_count < 75:
        raise ValueError("design_count must be at least 75 for the frozen policy templates")
    costs = normalize_fidelity_costs(fidelity_costs)
    if cost_budget < _required_policy_costs(costs)[policy]:
        raise ValueError(f"cost_budget is below the frozen minimum for {policy}")

    world = CompositionalScenarioWorld(
        objective_family, constraint_family, dimension, severity, seed, costs, design_count
    )
    evaluator = PolicyEvaluator(world, cost_budget=cost_budget)
    POLICIES[policy](evaluator, seed)

    # Query truth only after the policy has completed. All designs—not only
    # queried designs—form the fixed finite-pool oracle for normalization.
    truth = [world.truth_for_audit_only(cid) for cid in evaluator.candidate_ids]
    true_hv = hypervolume_2d(
        [obs.objectives for obs in truth if obs.constraint_margin >= 0.0],
        reference=REFERENCE_POINT,
    )
    observed_high = [
        obs for obs in evaluator.high_observations if obs.constraint_margin >= 0.0
    ]
    observed_hv = hypervolume_2d(
        [obs.objectives for obs in observed_high], reference=REFERENCE_POINT
    )

    counts = {fidelity: 0 for fidelity in ("low", "medium", "high")}
    level_costs = {fidelity: 0 for fidelity in ("low", "medium", "high")}
    for record in evaluator.records:
        counts[record.fidelity] += 1
        level_costs[record.fidelity] += record.cost_units

    high_by_id = {obs.candidate_id: obs for obs in evaluator.high_observations}
    promotions = []
    for candidate_id, (predicted_margin, method) in evaluator.promotion_basis.items():
        actual = high_by_id.get(candidate_id)
        if actual is not None:
            promotions.append({
                "candidate_id": candidate_id,
                "predicted_margin": predicted_margin,
                "method": method,
                "observed_high_margin": actual.constraint_margin,
            })
    false_promotions = sum(
        p["predicted_margin"] >= 0.0 and p["observed_high_margin"] < 0.0
        for p in promotions
    )
    predicted_feasible = sum(p["predicted_margin"] >= 0.0 for p in promotions)

    return {
        "benchmark_id": BENCHMARK_ID,
        "synthetic_fixture_only": True,
        "objective_family": objective_family,
        "constraint_family": constraint_family,
        "dimension": dimension,
        "severity": severity,
        "seed": seed,
        "policy": policy,
        "design_count": design_count,
        "candidate_pool_sha256": _digest_candidate_pool(world),
        "fidelity_costs": costs,
        "cost_budget": cost_budget,
        "cost_spent": evaluator.cost_spent,
        "unused_cost": evaluator.remaining_budget,
        "evaluation_count": len(evaluator.records),
        "fidelity_evaluations": counts,
        "fidelity_cost_units_by_level": level_costs,
        "high_fidelity_call_count": counts["high"],
        "true_finite_pool_hypervolume": true_hv,
        "observed_high_fidelity_hypervolume": observed_hv,
        "normalized_hypervolume_fraction": observed_hv / true_hv if true_hv > 0.0 else 0.0,
        "promotion_records": promotions,
        "decision_value_trace": getattr(evaluator, "decision_value_trace", []),
        "predicted_feasible_promotion_count": predicted_feasible,
        "false_promotion_count": false_promotions,
        "false_promotion_rate": false_promotions / predicted_feasible if predicted_feasible else None,
        "evaluations": [asdict(record) for record in evaluator.records],
    }


def run_sweep(
    objective_families: Iterable[str] = SUPPORTED_SCENARIOS,
    constraint_families: Iterable[str] = SUPPORTED_SCENARIOS,
    dimensions: Iterable[int] = DEFAULT_DIMENSIONS,
    severities: Iterable[float] = DEFAULT_COMPOSITION_SEVERITIES,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    policies: Iterable[str] | None = None,
    cost_budget: int = 200,
    fidelity_costs: dict[str, int] | None = None,
    design_count: int = DEFAULT_DESIGN_COUNT,
) -> dict:
    objectives = tuple(objective_families)
    constraints = tuple(constraint_families)
    dims = tuple(dimensions)
    sev = tuple(float(x) for x in severities)
    seed_values = tuple(seeds)
    policy_values = tuple(POLICIES) if policies is None else tuple(policies)
    costs = normalize_fidelity_costs(fidelity_costs)
    locked_sources = source_lock()
    for values, name in (
        (objectives, "objective_families"),
        (constraints, "constraint_families"),
        (dims, "dimensions"),
        (sev, "severities"),
        (seed_values, "seeds"),
        (policy_values, "policies"),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be non-empty and unique")
    if set(objectives) - set(SUPPORTED_SCENARIOS) or set(constraints) - set(SUPPORTED_SCENARIOS):
        raise ValueError("unknown mismatch family")
    if set(dims) - set(SUPPORTED_DIMENSIONS):
        raise ValueError(f"dimensions must be drawn from {SUPPORTED_DIMENSIONS}")
    if set(policy_values) - set(POLICIES):
        raise ValueError("unknown policy")
    if any(not isinstance(s, int) or isinstance(s, bool) for s in seed_values):
        raise ValueError("seeds must be integers")
    if any(s < 0 for s in sev):
        raise ValueError("severities must be non-negative")
    if design_count < 75:
        raise ValueError("design_count must be at least 75")
    required = _required_policy_costs(costs)
    if cost_budget < max(required[p] for p in policy_values):
        raise ValueError("cost_budget must satisfy every selected policy's frozen minimum")

    runs = [
        run_one(obj, con, dim, severity, seed, policy, cost_budget, costs, design_count)
        for obj in objectives
        for con in constraints
        for dim in dims
        for severity in sev
        for seed in seed_values
        for policy in policy_values
    ]
    return {
        "benchmark_id": BENCHMARK_ID,
        "synthetic_fixture_only": True,
        "objective_families": list(objectives),
        "constraint_families": list(constraints),
        "dimensions": list(dims),
        "severities": list(sev),
        "seeds": list(seed_values),
        "policies": list(policy_values),
        "design_count": design_count,
        "cost_budget": cost_budget,
        "fidelity_costs": costs,
        "source_sha256": locked_sources,
        "run_count": len(runs),
        "paired_runs": runs,
        "warning": (
            "Compositional synthetic mismatch diagnostic only. No external optimizer, "
            "COCO result, validated casting model, or physical foundry evidence is included."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective-families", default=",".join(SUPPORTED_SCENARIOS))
    parser.add_argument("--constraint-families", default=",".join(SUPPORTED_SCENARIOS))
    parser.add_argument("--dimensions", default=",".join(map(str, DEFAULT_DIMENSIONS)))
    parser.add_argument("--severities", default=",".join(map(str, DEFAULT_COMPOSITION_SEVERITIES)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--cost-budget", type=int, default=200)
    parser.add_argument("--fidelity-costs", default="low=1,medium=5,high=25")
    parser.add_argument("--design-count", type=int, default=DEFAULT_DESIGN_COUNT)
    args = parser.parse_args()
    costs = {}
    for item in args.fidelity_costs.split(","):
        if "=" not in item:
            raise ValueError("fidelity costs must use name=integer")
        key, value = item.split("=", 1)
        if key in costs:
            raise ValueError(f"duplicate fidelity cost: {key}")
        costs[key.strip()] = int(value.strip())
    result = run_sweep(
        objective_families=[v.strip() for v in args.objective_families.split(",") if v.strip()],
        constraint_families=[v.strip() for v in args.constraint_families.split(",") if v.strip()],
        dimensions=[int(v) for v in args.dimensions.split(",") if v.strip()],
        severities=[float(v) for v in args.severities.split(",") if v.strip()],
        seeds=[int(v) for v in args.seeds.split(",") if v.strip()],
        policies=[v.strip() for v in args.policies.split(",") if v.strip()],
        cost_budget=args.cost_budget,
        fidelity_costs=costs,
        design_count=args.design_count,
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
