#!/usr/bin/env python3
"""Deterministic multi-fidelity casting-like stress benchmark (synthetic only).

Standard library only; benchmark/test tooling, not a production dependency.
Policy code receives a restricted evaluator interface; the separate audit path owns truth.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Iterable

FIDELITY_COSTS = {"low": 1, "medium": 5, "high": 25}
FIDELITY_NAMES = ("low", "medium", "high")
OBJECTIVE_COUNT = 2
REFERENCE_POINT = (1.2, 1.2)
IDEAL_POINT = (0.0, 0.0)
DEFAULT_COST_BUDGET = 200
DEFAULT_GRID_SIZE = 21
MIN_POLICY_CANDIDATES = 75


@dataclass(frozen=True)
class Design:
    candidate_id: int
    x: float
    y: float


@dataclass(frozen=True)
class Observation:
    candidate_id: int
    fidelity: str
    objectives: tuple[float, float]
    constraint_margin: float
    cost_units: int

    @property
    def predicted_feasible(self) -> bool:
        return self.constraint_margin >= 0.0


@dataclass(frozen=True)
class EvaluationRecord:
    candidate_id: int
    fidelity: str
    objectives: tuple[float, float]
    constraint_margin: float
    cost_units: int


class SyntheticCastingWorld:
    """Finite deterministic design pool with three deliberately mismatched fidelities."""

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        severity: float = 1.0,
        fidelity_costs: dict[str, int] | None = None,
    ) -> None:
        if grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        if not math.isfinite(severity) or severity < 0.0:
            raise ValueError("severity must be finite and non-negative")
        self.grid_size = grid_size
        self.severity = severity
        self._fidelity_costs = normalize_fidelity_costs(fidelity_costs)
        self._designs = tuple(
            Design(i * grid_size + j, i / (grid_size - 1), j / (grid_size - 1))
            for i in range(grid_size)
            for j in range(grid_size)
        )
        self._by_id = {d.candidate_id: d for d in self._designs}

    @property
    def designs(self) -> tuple[Design, ...]:
        return self._designs

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return tuple(d.candidate_id for d in self._designs)

    @property
    def fidelity_costs(self) -> dict[str, int]:
        return dict(self._fidelity_costs)

    def design(self, candidate_id: int) -> Design:
        try:
            return self._by_id[candidate_id]
        except KeyError as exc:
            raise ValueError(f"unknown candidate_id: {candidate_id}") from exc

    def _high_truth(self, design: Design) -> tuple[tuple[float, float], float]:
        x, y = design.x, design.y
        ripple0 = 0.015 * math.sin(4.0 * math.pi * x) * math.cos(2.0 * math.pi * y)
        ripple1 = 0.015 * math.cos(2.0 * math.pi * x) * math.sin(4.0 * math.pi * y)
        f0 = 0.15 + (x - 0.22) ** 2 + 0.24 * y + ripple0
        f1 = 0.15 + (x - 0.78) ** 2 + 0.24 * (1.0 - y) + ripple1
        # Nonlinear feasible region; margin >= 0 is feasible.
        margin = 0.20 - (x - 0.55) ** 2 - (y - 0.45) ** 2
        return (f0, f1), margin

    def _response(self, candidate_id: int, fidelity: str) -> Observation:
        if fidelity not in self._fidelity_costs:
            raise ValueError(f"unsupported fidelity: {fidelity}")
        design = self.design(candidate_id)
        (hf0, hf1), hfm = self._high_truth(design)
        x, y = design.x, design.y
        s = self.severity

        # Deterministic heteroscedastic perturbations: no hidden RNG or unlogged noise seed.
        noise0 = s * (0.004 + 0.035 * x) * math.sin(17.0 * x + 13.0 * y)
        noise1 = s * (0.006 + 0.030 * y) * math.cos(11.0 * x - 19.0 * y)

        if fidelity == "high":
            objectives, margin = (hf0, hf1), hfm
        elif fidelity == "medium":
            objectives = (
                hf0 + s * (0.035 * x + 0.025 * math.sin(2.0 * math.pi * (x + y))) + noise0 * 0.30,
                hf1 - s * (0.030 * y - 0.018 * math.cos(2.0 * math.pi * x)) + noise1 * 0.30,
            )
            margin = hfm + s * (0.055 * math.sin(3.0 * math.pi * x + 2.0 * math.pi * y) - 0.006)
        else:
            # Larger, non-monotone discrepancy and boundary inversions on purpose.
            objectives = (
                hf0 + s * (0.15 * x + 0.075 * math.sin(4.0 * math.pi * y)) + noise0,
                hf1 + s * (0.12 * (1.0 - y) + 0.080 * math.cos(3.0 * math.pi * x) * math.sin(2.0 * math.pi * y)) + noise1,
            )
            margin = hfm + s * (0.16 * math.sin(2.0 * math.pi * x) * math.cos(2.0 * math.pi * y) - 0.035)

        if not all(math.isfinite(v) for v in (*objectives, margin)):
            raise ArithmeticError("synthetic evaluator emitted a non-finite value")
        return Observation(candidate_id, fidelity, objectives, margin, self._fidelity_costs[fidelity])

    def truth_for_audit_only(self, candidate_id: int) -> Observation:
        """Benchmark auditor only. Do not give this object to candidate policies."""
        return self._response(candidate_id, "high")


class PolicyEvaluator:
    """Capability-limited view exposed to policies; it has no all-truth operation."""

    def __init__(self, world: SyntheticCastingWorld, cost_budget: int, max_high_calls: int | None = None) -> None:
        if cost_budget < 1:
            raise ValueError("cost_budget must be positive")
        if max_high_calls is not None and max_high_calls < 0:
            raise ValueError("max_high_calls must be non-negative")
        self._world = world
        self._cost_budget = cost_budget
        self._max_high_calls = max_high_calls
        self._cost_spent = 0
        self._high_calls = 0
        self._records: list[EvaluationRecord] = []
        self._latest: dict[int, Observation] = {}
        self._promotion_basis: dict[int, tuple[float, str]] = {}

    @property
    def candidate_ids(self) -> tuple[int, ...]:
        return self._world.candidate_ids

    @property
    def cost_spent(self) -> int:
        return self._cost_spent

    @property
    def high_calls(self) -> int:
        return self._high_calls

    @property
    def records(self) -> tuple[EvaluationRecord, ...]:
        return tuple(self._records)

    @property
    def high_observations(self) -> tuple[Observation, ...]:
        return tuple(record_to_observation(r) for r in self._records if r.fidelity == "high")

    @property
    def promotion_basis(self) -> dict[int, tuple[float, str]]:
        return dict(self._promotion_basis)

    def design(self, candidate_id: int) -> Design:
        return self._world.design(candidate_id)

    def evaluate(self, candidate_id: int, fidelity: str) -> Observation:
        cost = self._world.fidelity_costs.get(fidelity)
        if cost is None:
            raise ValueError(f"unsupported fidelity: {fidelity}")
        if self._cost_spent + cost > self._cost_budget:
            raise BudgetExceeded(f"fidelity={fidelity} would exceed cost budget")
        if fidelity == "high" and self._max_high_calls is not None and self._high_calls >= self._max_high_calls:
            raise BudgetExceeded("high-fidelity call cap exhausted")
        obs = self._world._response(candidate_id, fidelity)
        self._cost_spent += cost
        if fidelity == "high":
            self._high_calls += 1
        self._records.append(EvaluationRecord(
            obs.candidate_id, obs.fidelity, obs.objectives, obs.constraint_margin, obs.cost_units
        ))
        self._latest[candidate_id] = obs
        return obs

    def record_promotion_basis(self, candidate_id: int, predicted_margin: float, method: str) -> None:
        if not math.isfinite(predicted_margin) or not method:
            raise ValueError("promotion basis requires finite margin and non-empty method")
        self._promotion_basis[candidate_id] = (predicted_margin, method)


class BudgetExceeded(RuntimeError):
    pass


def normalize_fidelity_costs(costs: dict[str, int] | None = None) -> dict[str, int]:
    if costs is None:
        return dict(FIDELITY_COSTS)
    if set(costs) != set(FIDELITY_NAMES):
        raise ValueError("fidelity costs must define exactly low, medium, and high")
    normalized: dict[str, int] = {}
    for name in FIDELITY_NAMES:
        value = costs[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"fidelity cost for {name} must be a positive integer")
        normalized[name] = value
    return normalized


def required_policy_costs(costs: dict[str, int] | None = None) -> dict[str, int]:
    c = normalize_fidelity_costs(costs)
    return {
        "high_only": 8 * c["high"],
        "low_screen": 75 * c["low"] + 5 * c["high"],
        "fixed_top_k": 50 * c["low"] + 10 * c["medium"] + 4 * c["high"],
        "discrepancy_aware": 55 * c["low"] + 4 * c["medium"] + 5 * c["high"],
    }


def record_to_observation(record: EvaluationRecord) -> Observation:
    return Observation(record.candidate_id, record.fidelity, record.objectives, record.constraint_margin, record.cost_units)


def deterministic_sample(candidate_ids: Iterable[int], count: int, seed: int) -> list[int]:
    items = list(candidate_ids)
    if count < 0 or count > len(items):
        raise ValueError("sample count out of range")
    state = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(len(items) - 1, 0, -1):
        state ^= (state >> 12) & 0xFFFFFFFFFFFFFFFF
        state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
        state ^= (state >> 27) & 0xFFFFFFFFFFFFFFFF
        value = (state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF
        j = value % (i + 1)
        items[i], items[j] = items[j], items[i]
    return items[:count]


def _scalar_score(obs: Observation, weight: float) -> float:
    return weight * obs.objectives[0] + (1.0 - weight) * obs.objectives[1]


def select_diverse(
    observations: Iterable[Observation],
    count: int,
    weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    corrected: dict[int, tuple[float, float, float]] | None = None,
) -> list[tuple[int, float]]:
    """Return candidate IDs paired with the margin used by the promotion decision."""
    by_id = {obs.candidate_id: obs for obs in observations}
    if count < 0 or count > len(by_id):
        raise ValueError("selection count out of range")

    def candidate_values(candidate_id: int) -> tuple[tuple[float, float], float]:
        if corrected and candidate_id in corrected:
            f0, f1, margin = corrected[candidate_id]
            return (f0, f1), margin
        obs = by_id[candidate_id]
        return obs.objectives, obs.constraint_margin

    selected: list[tuple[int, float]] = []
    for weight in weights:
        eligible = [
            candidate_id for candidate_id in by_id
            if candidate_id not in {item[0] for item in selected}
            and candidate_values(candidate_id)[1] >= 0.0
        ]
        if not eligible:
            eligible = [
                candidate_id for candidate_id in by_id
                if candidate_id not in {item[0] for item in selected}
            ]
        if not eligible:
            break
        winner = min(eligible, key=lambda cid: _scalar_score(Observation(
            cid, "low", candidate_values(cid)[0], candidate_values(cid)[1], FIDELITY_COSTS["low"]
        ), weight))
        selected.append((winner, candidate_values(winner)[1]))

    if len(selected) < count:
        remaining = [cid for cid in by_id if cid not in {item[0] for item in selected}]
        remaining.sort(key=lambda cid: (
            candidate_values(cid)[1] < 0.0,
            abs(candidate_values(cid)[1]) if candidate_values(cid)[1] < 0.0 else
            sum(candidate_values(cid)[0]),
        ))
        selected.extend((cid, candidate_values(cid)[1]) for cid in remaining[: count - len(selected)])
    return selected[:count]


def pareto_front(
    points: Iterable[tuple[float, float]],
    reference: tuple[float, float] = REFERENCE_POINT,
) -> list[tuple[float, float]]:
    if len(reference) != 2 or not all(math.isfinite(v) for v in reference):
        raise ValueError("reference point must contain two finite values")
    unique = sorted(set((float(a), float(b)) for a, b in points))
    front: list[tuple[float, float]] = []
    best_y = math.inf
    for x, y in unique:
        if x >= reference[0] or y >= reference[1]:
            continue
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def hypervolume_2d(
    points: Iterable[tuple[float, float]],
    reference: tuple[float, float] = REFERENCE_POINT,
) -> float:
    front = pareto_front(points, reference=reference)
    area = 0.0
    for i, (x, y) in enumerate(front):
        next_x = front[i + 1][0] if i + 1 < len(front) else reference[0]
        area += max(0.0, next_x - x) * max(0.0, reference[1] - y)
    if not math.isfinite(area):
        raise ArithmeticError("hypervolume became non-finite")
    return area


def _evaluate_sample(evaluator: PolicyEvaluator, candidate_ids: list[int], fidelity: str) -> list[Observation]:
    results = []
    for candidate_id in candidate_ids:
        results.append(evaluator.evaluate(candidate_id, fidelity))
    return results


def high_only_policy(evaluator: PolicyEvaluator, seed: int) -> None:
    picks = deterministic_sample(evaluator.candidate_ids, 8, seed)
    for candidate_id in picks:
        evaluator.evaluate(candidate_id, "high")


def low_screen_policy(evaluator: PolicyEvaluator, seed: int) -> None:
    picks = deterministic_sample(evaluator.candidate_ids, 75, seed)
    low = _evaluate_sample(evaluator, picks, "low")
    promotions = select_diverse(low, 5)
    for candidate_id, margin in promotions:
        evaluator.record_promotion_basis(candidate_id, margin, "low_predicted_margin")
        evaluator.evaluate(candidate_id, "high")


def fixed_top_k_policy(evaluator: PolicyEvaluator, seed: int) -> None:
    picks = deterministic_sample(evaluator.candidate_ids, 50, seed)
    low = _evaluate_sample(evaluator, picks, "low")
    medium_ids = [candidate_id for candidate_id, _ in select_diverse(low, 10)]
    medium = _evaluate_sample(evaluator, medium_ids, "medium")
    promotions = select_diverse(medium, 4)
    for candidate_id, margin in promotions:
        evaluator.record_promotion_basis(candidate_id, margin, "medium_top_k")
        evaluator.evaluate(candidate_id, "high")


def discrepancy_aware_policy(evaluator: PolicyEvaluator, seed: int) -> None:
    picks = deterministic_sample(evaluator.candidate_ids, 55, seed)
    low = _evaluate_sample(evaluator, picks, "low")
    # Query medium where predicted feasibility is uncertain, plus both objective extremes.
    by_margin = sorted(low, key=lambda obs: abs(obs.constraint_margin))
    low_f0 = min(low, key=lambda obs: obs.objectives[0])
    low_f1 = min(low, key=lambda obs: obs.objectives[1])
    medium_ids: list[int] = []
    for candidate_id in [by_margin[0].candidate_id, by_margin[1].candidate_id,
                         low_f0.candidate_id, low_f1.candidate_id]:
        if candidate_id not in medium_ids:
            medium_ids.append(candidate_id)
    if len(medium_ids) < 4:
        medium_ids.extend(cid for cid in picks if cid not in medium_ids and len(medium_ids) < 4)
    medium = _evaluate_sample(evaluator, medium_ids, "medium")
    medium_by_id = {obs.candidate_id: obs for obs in medium}

    # Two deliberately diverse high-fidelity anchors estimate medium->high discrepancy.
    anchors = select_diverse(medium, 2)
    high_anchors: list[Observation] = []
    for candidate_id, margin in anchors:
        evaluator.record_promotion_basis(candidate_id, margin, "discrepancy_anchor")
        high_anchors.append(evaluator.evaluate(candidate_id, "high"))

    delta_f0 = sum(h.objectives[0] - medium_by_id[h.candidate_id].objectives[0] for h in high_anchors) / len(high_anchors)
    delta_f1 = sum(h.objectives[1] - medium_by_id[h.candidate_id].objectives[1] for h in high_anchors) / len(high_anchors)
    delta_margin = sum(h.constraint_margin - medium_by_id[h.candidate_id].constraint_margin for h in high_anchors) / len(high_anchors)

    # Chain the paired low->medium discrepancy and the anchor-derived medium->high correction.
    mean_d0 = sum(m.objectives[0] - next(l for l in low if l.candidate_id == m.candidate_id).objectives[0] for m in medium) / len(medium)
    mean_d1 = sum(m.objectives[1] - next(l for l in low if l.candidate_id == m.candidate_id).objectives[1] for m in medium) / len(medium)
    mean_dm = sum(m.constraint_margin - next(l for l in low if l.candidate_id == m.candidate_id).constraint_margin for m in medium) / len(medium)
    corrected: dict[int, tuple[float, float, float]] = {}
    for obs in low:
        corrected[obs.candidate_id] = (
            obs.objectives[0] + mean_d0 + delta_f0,
            obs.objectives[1] + mean_d1 + delta_f1,
            obs.constraint_margin + mean_dm + delta_margin,
        )

    anchor_ids = {obs.candidate_id for obs in high_anchors}
    candidates = [obs for obs in low if obs.candidate_id not in anchor_ids]
    promotions = select_diverse(candidates, 3, weights=(0.0, 0.5, 1.0), corrected=corrected)
    for candidate_id, margin in promotions:
        evaluator.record_promotion_basis(candidate_id, margin, "discrepancy_corrected")
        evaluator.evaluate(candidate_id, "high")


POLICIES = {
    "high_only": high_only_policy,
    "low_screen": low_screen_policy,
    "fixed_top_k": fixed_top_k_policy,
    "discrepancy_aware": discrepancy_aware_policy,
}


def _validate_run_configuration(
    cost_budget: int,
    grid_size: int,
    severity: float,
    fidelity_costs: dict[str, int] | None = None,
    required_budget: int | None = None,
) -> None:
    costs = normalize_fidelity_costs(fidelity_costs)
    minimum = required_budget if required_budget is not None else max(required_policy_costs(costs).values())
    if cost_budget < minimum:
        raise ValueError(f"cost_budget must be at least {minimum} for the frozen policy templates and fidelity costs")
    if grid_size < 9 or grid_size * grid_size < MIN_POLICY_CANDIDATES:
        raise ValueError(f"grid_size must expose at least {MIN_POLICY_CANDIDATES} candidate designs")
    if not math.isfinite(severity) or severity < 0.0:
        raise ValueError("severity must be finite and non-negative")


def run_policy(
    policy_name: str,
    seed: int = 17,
    cost_budget: int = DEFAULT_COST_BUDGET,
    grid_size: int = DEFAULT_GRID_SIZE,
    severity: float = 1.0,
    fidelity_costs: dict[str, int] | None = None,
) -> dict:
    if policy_name not in POLICIES:
        raise ValueError(f"unknown policy: {policy_name}")
    configured_costs = normalize_fidelity_costs(fidelity_costs)
    _validate_run_configuration(
        cost_budget, grid_size, severity, configured_costs,
        required_policy_costs(configured_costs)[policy_name],
    )
    world = SyntheticCastingWorld(grid_size=grid_size, severity=severity, fidelity_costs=configured_costs)
    evaluator = PolicyEvaluator(world, cost_budget=cost_budget)
    POLICIES[policy_name](evaluator, seed)

    # Ground truth is exposed only to the separate audit/scoring path, after the policy returns.
    high = [world.truth_for_audit_only(cid) for cid in evaluator.candidate_ids]
    truth_feasible = [obs for obs in high if obs.constraint_margin >= 0.0]
    true_hv = hypervolume_2d([obs.objectives for obs in truth_feasible])
    observed_high = [obs for obs in evaluator.high_observations if obs.constraint_margin >= 0.0]
    observed_hv = hypervolume_2d([obs.objectives for obs in observed_high])

    counts = {fidelity: 0 for fidelity in FIDELITY_COSTS}
    cost_totals = {fidelity: 0 for fidelity in FIDELITY_COSTS}
    for record in evaluator.records:
        counts[record.fidelity] += 1
        cost_totals[record.fidelity] += record.cost_units

    basis = evaluator.promotion_basis
    high_by_id = {obs.candidate_id: obs for obs in evaluator.high_observations}
    audited_promotions = [
        (cid, margin, method, high_by_id[cid])
        for cid, (margin, method) in basis.items() if cid in high_by_id
    ]
    false_promotions = sum(
        1 for _, predicted_margin, _, observed in audited_promotions
        if predicted_margin >= 0.0 and observed.constraint_margin < 0.0
    )
    predicted_feasible_promotions = sum(1 for _, margin, _, _ in audited_promotions if margin >= 0.0)
    actual_feasible_promotions = sum(1 for _, _, _, obs in audited_promotions if obs.constraint_margin >= 0.0)

    return {
        "synthetic_fixture_only": True,
        "policy": policy_name,
        "seed": seed,
        "grid_size": grid_size,
        "severity": severity,
        "fidelity_costs": configured_costs,
        "cost_budget": cost_budget,
        "cost_spent": evaluator.cost_spent,
        "unused_cost": cost_budget - evaluator.cost_spent,
        "evaluation_count": len(evaluator.records),
        "fidelity_evaluations": counts,
        "fidelity_cost_units_by_level": cost_totals,
        "high_fidelity_call_count": counts["high"],
        "true_finite_pool_hypervolume": true_hv,
        "observed_high_fidelity_hypervolume": observed_hv,
        "normalized_hypervolume_fraction": observed_hv / true_hv if true_hv > 0.0 else 0.0,
        "audited_promotion_count": len(audited_promotions),
        "predicted_feasible_promotion_count": predicted_feasible_promotions,
        "actual_feasible_promotion_count": actual_feasible_promotions,
        "false_promotion_count": false_promotions,
        "false_promotion_rate": (
            false_promotions / predicted_feasible_promotions if predicted_feasible_promotions else None
        ),
        "evaluations": [asdict(record) for record in evaluator.records],
    }


def run_suite(seed: int = 17, cost_budget: int | None = None, grid_size: int = DEFAULT_GRID_SIZE,
              severity: float = 1.0, fidelity_costs: dict[str, int] | None = None) -> dict:
    costs = normalize_fidelity_costs(fidelity_costs)
    if cost_budget is None:
        cost_budget = max(required_policy_costs(costs).values())
    _validate_run_configuration(cost_budget, grid_size, severity, costs)
    results = [
        run_policy(name, seed=seed, cost_budget=cost_budget, grid_size=grid_size, severity=severity, fidelity_costs=costs)
        for name in POLICIES
    ]
    return {
        "benchmark_id": "CASTING-MF-STRESS-001",
        "synthetic_fixture_only": True,
        "seed": seed,
        "cost_budget": cost_budget,
        "grid_size": grid_size,
        "severity": severity,
        "fidelity_costs": costs,
        "policy_results": results,
        "warning": "Synthetic known-truth stress fixture only; no foundry/physical or SOTA claim.",
    }



def _percentile(values: list[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("percentile requires non-empty values and q in [0,1]")
    ordered = sorted(values)
    location = (len(ordered) - 1) * q
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def run_sweep(
    seeds: Iterable[int] = (17, 23, 29, 31, 37, 41, 43, 47, 53, 59),
    severities: Iterable[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
    cost_budget: int | None = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    fidelity_costs: dict[str, int] | None = None,
) -> dict:
    """Paired seed x severity sweep; all policies see the same deterministic candidate pool."""
    seed_values = tuple(seeds)
    severity_values = tuple(float(v) for v in severities)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("sweep seeds must be non-empty and unique")
    if not severity_values or any(not math.isfinite(v) or v < 0.0 for v in severity_values):
        raise ValueError("sweep severities must be non-empty finite non-negative values")
    costs = normalize_fidelity_costs(fidelity_costs)
    if cost_budget is None:
        cost_budget = max(required_policy_costs(costs).values())
    _validate_run_configuration(cost_budget, grid_size, max(severity_values), costs)

    runs: list[dict] = []
    for severity in severity_values:
        for seed in seed_values:
            suite = run_suite(seed=seed, cost_budget=cost_budget, grid_size=grid_size, severity=severity, fidelity_costs=costs)
            runs.extend(suite["policy_results"])

    summaries = []
    for severity in severity_values:
        for policy in POLICIES:
            group = [
                row for row in runs
                if row["severity"] == severity and row["policy"] == policy
            ]
            quality = [float(row["normalized_hypervolume_fraction"]) for row in group]
            promotion_rates = [
                float(row["false_promotion_rate"])
                for row in group if row["false_promotion_rate"] is not None
            ]
            summaries.append({
                "policy": policy,
                "severity": severity,
                "seed_count": len(group),
                "median_normalized_hypervolume_fraction": statistics.median(quality),
                "q25_normalized_hypervolume_fraction": _percentile(quality, 0.25),
                "q75_normalized_hypervolume_fraction": _percentile(quality, 0.75),
                "mean_false_promotion_rate": (
                    statistics.fmean(promotion_rates) if promotion_rates else None
                ),
                "total_cost_units_mean": statistics.fmean(float(row["cost_spent"]) for row in group),
                "mean_high_fidelity_calls": statistics.fmean(
                    float(row["high_fidelity_call_count"]) for row in group
                ),
            })

    return {
        "benchmark_id": "CASTING-MF-STRESS-001",
        "synthetic_fixture_only": True,
        "cost_budget": cost_budget,
        "fidelity_costs": costs,
        "grid_size": grid_size,
        "seeds": list(seed_values),
        "severities": list(severity_values),
        "summary_by_policy_and_severity": summaries,
        "paired_runs": runs,
        "warning": (
            "Synthetic finite-pool stress test only. Policy comparisons are descriptive until "
            "repeated-seed uncertainty analysis and locked MF-HVKG/MOMF baselines are included."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cost-budget", type=int, default=None,
                        help="maximum cost units; defaults to the maximum cost of the fixed policy templates")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--fidelity-costs", type=str, default="low=1,medium=5,high=25",
                        help="comma-separated positive integer costs, e.g. low=1,medium=10,high=50")
    parser.add_argument("--sweep-seeds", type=str, default=None,
                        help="comma-separated paired seeds; enables severity sweep")
    parser.add_argument("--sweep-severities", type=str, default="0,0.5,1,1.5,2",
                        help="comma-separated non-negative discrepancy severity values")
    args = parser.parse_args()
    parsed_costs: dict[str, int] = {}
    for entry in args.fidelity_costs.split(","):
        if "=" not in entry:
            raise ValueError("each fidelity cost must use name=integer format")
        name, raw_value = entry.split("=", 1)
        if name.strip() in parsed_costs:
            raise ValueError(f"duplicate fidelity cost: {name.strip()}")
        parsed_costs[name.strip()] = int(raw_value.strip())
    costs = normalize_fidelity_costs(parsed_costs)
    if args.sweep_seeds is not None:
        seeds = [int(value.strip()) for value in args.sweep_seeds.split(",") if value.strip()]
        severities = [float(value.strip()) for value in args.sweep_severities.split(",") if value.strip()]
        result = run_sweep(seeds, severities, args.cost_budget, args.grid_size, costs)
    else:
        result = run_suite(args.seed, args.cost_budget, args.grid_size, args.severity, costs)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
