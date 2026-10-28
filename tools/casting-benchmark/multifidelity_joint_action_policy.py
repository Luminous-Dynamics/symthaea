#!/usr/bin/env python3
"""Adaptive joint candidate/fidelity acquisition for casting-like synthetic tests.

The policy is a transparent heuristic research candidate, not Bayesian VOI or an
implementation of any cited paper. It selects each (design, fidelity) action
online and prices the expected deferred high-fidelity validation cost of cheap
observations. All choices depend only on the prior evaluation ledger.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from multifidelity_stress import PolicyEvaluator

POLICY_ID = "joint_action_cost_aware"
BOOTSTRAP_LOW_CALLS = 6
INFORMATION_FRACTION = {"low": 0.25, "medium": 0.60, "high": 1.0}
FIDELITY_ORDER = {"low": 0, "medium": 1, "high": 2}
SCORE_WEIGHTS = {
    "feasible_hypervolume_gain": 0.70,
    "constraint_boundary_information": 0.20,
    "frontier_objective_information": 0.10,
}
REFERENCE_POINT = (1.2, 1.2)
MIN_SIGMA = 1e-6


def _field(row: Any, key: str) -> Any:
    return row[key] if isinstance(row, dict) else getattr(row, key)


def _row(row: Any) -> dict[str, Any]:
    return {
        "candidate_id": int(_field(row, "candidate_id")),
        "fidelity": str(_field(row, "fidelity")),
        "objectives": tuple(float(v) for v in _field(row, "objectives")),
        "constraint_margin": float(_field(row, "constraint_margin")),
        "cost_units": int(_field(row, "cost_units")),
    }


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("coordinate dimensions differ")
    return math.sqrt(statistics.fmean((x - y) ** 2 for x, y in zip(a, b)))


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _robust_scale(values: list[float]) -> float:
    if not values:
        return 0.12
    center = _median(values)
    mad = _median([abs(value - center) for value in values])
    mean = statistics.fmean(values)
    variance = statistics.fmean((value - mean) ** 2 for value in values)
    return max(MIN_SIGMA, 1.4826 * mad, 0.5 * math.sqrt(max(0.0, variance)))


def _local_delta(
    candidate_id: int,
    pairs: list[tuple[int, tuple[float, float, float]]],
    coordinates: dict[int, tuple[float, ...]],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not pairs:
        return (0.0, 0.0, 0.0), (0.12, 0.12, 0.12)
    target = coordinates[candidate_id]
    ranked = sorted(
        ((_distance(target, coordinates[other]), other, delta) for other, delta in pairs),
        key=lambda item: (item[0], item[1]),
    )[: min(4, len(pairs))]
    raw_weights = [1.0 / ((distance + 0.02) ** 2) for distance, _, _ in ranked]
    total = sum(raw_weights)
    weights = [value / total for value in raw_weights]
    means = tuple(
        sum(weight * row[2][j] for weight, row in zip(weights, ranked))
        for j in range(3)
    )
    global_scales = [
        _robust_scale([delta[j] for _, delta in pairs]) for j in range(3)
    ]
    nearest_distance = ranked[0][0]
    scales = []
    for j in range(3):
        local_variance = sum(
            weight * (row[2][j] - means[j]) ** 2
            for weight, row in zip(weights, ranked)
        )
        scales.append(max(
            MIN_SIGMA,
            math.sqrt(
                local_variance
                + (0.35 * global_scales[j]) ** 2
                + (nearest_distance * global_scales[j]) ** 2
            ),
        ))
    return means, tuple(scales)


def _pareto_front(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    front: list[tuple[float, float]] = []
    best_y = math.inf
    for x, y in sorted(set(points)):
        if x >= REFERENCE_POINT[0] or y >= REFERENCE_POINT[1]:
            continue
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def _hypervolume(points: Iterable[tuple[float, float]]) -> float:
    front = _pareto_front(points)
    area = 0.0
    for i, (x, y) in enumerate(front):
        next_x = front[i + 1][0] if i + 1 < len(front) else REFERENCE_POINT[0]
        area += max(0.0, next_x - x) * max(0.0, REFERENCE_POINT[1] - y)
    return area


def _prediction(
    candidate_id: int,
    by_id: dict[int, dict[str, dict[str, Any]]],
    coordinates: dict[int, tuple[float, ...]],
) -> tuple[tuple[float, float], float, tuple[float, float, float], float]:
    low_medium = []
    medium_high = []
    for cid, observations in by_id.items():
        low, medium, high = observations.get("low"), observations.get("medium"), observations.get("high")
        if low is not None and medium is not None:
            low_medium.append((cid, (
                medium["objectives"][0] - low["objectives"][0],
                medium["objectives"][1] - low["objectives"][1],
                medium["constraint_margin"] - low["constraint_margin"],
            )))
        if medium is not None and high is not None:
            medium_high.append((cid, (
                high["objectives"][0] - medium["objectives"][0],
                high["objectives"][1] - medium["objectives"][1],
                high["constraint_margin"] - medium["constraint_margin"],
            )))
    lm_mean, lm_sigma = _local_delta(candidate_id, low_medium, coordinates)
    mh_mean, mh_sigma = _local_delta(candidate_id, medium_high, coordinates)
    observations = by_id.get(candidate_id, {})
    spatial_distance = 0.0
    if observations.get("medium") is not None:
        base = observations["medium"]
        objectives = tuple(base["objectives"][j] + mh_mean[j] for j in range(2))
        margin = base["constraint_margin"] + mh_mean[2]
        sigma = mh_sigma
    elif observations.get("low") is not None:
        base = observations["low"]
        objectives = tuple(base["objectives"][j] + lm_mean[j] + mh_mean[j] for j in range(2))
        margin = base["constraint_margin"] + lm_mean[2] + mh_mean[2]
        sigma = tuple(math.sqrt(lm_sigma[j] ** 2 + mh_sigma[j] ** 2) for j in range(3))
    else:
        sampled = [cid for cid, levels in by_id.items() if levels]
        if not sampled:
            return (0.5, 0.5), 0.0, (0.20, 0.20, 0.20), 1.0
        source_id = min(
            sampled,
            key=lambda other: (_distance(coordinates[candidate_id], coordinates[other]), other),
        )
        source_levels = by_id[source_id]
        source_distance = _distance(coordinates[candidate_id], coordinates[source_id])
        spatial_distance = source_distance
        if "high" in source_levels:
            base = source_levels["high"]
            objectives = tuple(base["objectives"])
            margin = base["constraint_margin"]
            sigma = (0.08, 0.08, 0.08)
        elif "medium" in source_levels:
            base = source_levels["medium"]
            objectives = tuple(base["objectives"][j] + mh_mean[j] for j in range(2))
            margin = base["constraint_margin"] + mh_mean[2]
            sigma = tuple(math.sqrt(mh_sigma[j] ** 2 + (0.08 * source_distance) ** 2) for j in range(3))
        else:
            base = source_levels["low"]
            objectives = tuple(base["objectives"][j] + lm_mean[j] + mh_mean[j] for j in range(2))
            margin = base["constraint_margin"] + lm_mean[2] + mh_mean[2]
            sigma = tuple(
                math.sqrt(lm_sigma[j] ** 2 + mh_sigma[j] ** 2 + (0.08 * source_distance) ** 2)
                for j in range(3)
            )
    if spatial_distance > 0.0:
        sigma = tuple(math.sqrt(sigma[j] ** 2 + (0.08 * spatial_distance) ** 2) for j in range(3))
    return objectives, margin, sigma, spatial_distance


def _impact(
    objectives: tuple[float, float],
    margin: float,
    sigma: tuple[float, float, float],
    high_archive: list[dict[str, Any]],
) -> float:
    sigma0, sigma1, sigma_margin = sigma
    feasible_probability = 0.5 * (
        1.0 + math.erf(margin / (max(sigma_margin, MIN_SIGMA) * math.sqrt(2.0)))
    )
    points = [
        tuple(row["objectives"]) for row in high_archive
        if row["constraint_margin"] >= 0.0
    ]
    gain = max(0.0, _hypervolume(points + [objectives]) - _hypervolume(points)) / (1.2 * 1.2)
    boundary_information = min(1.0, sigma_margin / 0.08) * math.exp(
        -abs(margin) / max(sigma_margin, MIN_SIGMA)
    )
    front = _pareto_front(points)
    front_distance = (
        min(math.sqrt((objectives[0] - x) ** 2 + (objectives[1] - y) ** 2) for x, y in front)
        if front else 0.0
    )
    objective_information = min(1.0, (sigma0 + sigma1) / 0.20) * math.exp(-front_distance / 0.10)
    return (
        SCORE_WEIGHTS["feasible_hypervolume_gain"] * feasible_probability * gain
        + SCORE_WEIGHTS["constraint_boundary_information"] * boundary_information
        + SCORE_WEIGHTS["frontier_objective_information"] * objective_information
    )


def choose_joint_action(
    history: Iterable[Any],
    candidate_ids: Iterable[int],
    coordinates: dict[int, tuple[float, ...]],
    fidelity_costs: dict[str, int],
    remaining_budget: int,
    seed: int,
) -> dict[str, Any] | None:
    """Pure action selector. Returns None only when no legal action is affordable."""
    rows = [_row(row) for row in history]
    by_id: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(row["candidate_id"], {})[row["fidelity"]] = row
    low_rows = [row for row in rows if row["fidelity"] == "low"]
    medium_count = sum(row["fidelity"] == "medium" for row in rows)
    high_rows = [row for row in rows if row["fidelity"] == "high"]
    high_count = len(high_rows)
    candidate_values = tuple(candidate_ids)

    # Cold-start only: establish six spatially spread low-fidelity anchors. After
    # this, every call is selected by the joint candidate/fidelity score.
    if len(low_rows) < min(BOOTSTRAP_LOW_CALLS, len(candidate_values)):
        seen = {row["candidate_id"] for row in low_rows}
        candidates = [cid for cid in candidate_values if cid not in seen]
        if not candidates or fidelity_costs["low"] > remaining_budget:
            return None
        def separation(cid: int) -> float:
            return min(
                (_distance(coordinates[cid], coordinates[row["candidate_id"]]) for row in low_rows),
                default=1.0,
            )
        winner = min(candidates, key=lambda cid: (-separation(cid), cid))
        distance = separation(winner)
        return {
            "candidate_id": winner,
            "fidelity": "low",
            "score": distance,
            "effective_cost": fidelity_costs["low"],
            "deferred_validation_cost": 0.0,
            "decision_impact": 0.0,
            "eligible_action_count": len(candidates),
            "selection_mode": "bootstrap_space_filling",
        }

    high_archive = high_rows
    predictions = {}
    impacts = {}
    diversities = {}
    for cid in candidate_values:
        levels = by_id.get(cid, {})
        if "high" in levels:
            continue
        predicted, margin, sigma, _ = _prediction(cid, by_id, coordinates)
        predictions[cid] = (predicted, margin, sigma)
        impacts[cid] = _impact(predicted, margin, sigma, high_archive)
        sampled_low_ids = [row["candidate_id"] for row in low_rows]
        diversities[cid] = min(
            (_distance(coordinates[cid], coordinates[other]) for other in sampled_low_ids),
            default=1.0,
        )
    max_impact = max(impacts.values(), default=MIN_SIGMA)
    actions: list[dict[str, Any]] = []
    for cid, (predicted, margin, sigma) in predictions.items():
        levels = by_id.get(cid, {})
        if "low" not in levels:
            eligible = ("low",)
        elif "medium" not in levels:
            eligible = ("medium", "high")
        else:
            eligible = ("high",)
        impact = impacts[cid]
        validation_probability = min(0.95, max(0.05, impact / max(max_impact, MIN_SIGMA)))
        for fidelity in eligible:
            call_cost = fidelity_costs[fidelity]
            if call_cost > remaining_budget:
                continue
            deferred = validation_probability * fidelity_costs["high"] if fidelity != "high" else 0.0
            effective_cost = call_cost + deferred
            info = INFORMATION_FRACTION[fidelity]
            score = (impact * info + 0.02 * diversities[cid] * info) / effective_cost
            if high_count == 0:
                if fidelity == "high":
                    score *= 1.25 if "medium" in levels else 0.65
                elif fidelity == "medium" and medium_count < 2:
                    score *= 1.40
            actions.append({
                "candidate_id": cid,
                "fidelity": fidelity,
                "score": score,
                "effective_cost": effective_cost,
                "deferred_validation_cost": deferred,
                "decision_impact": impact,
                "eligible_action_count": 0,
                "selection_mode": "joint_acquisition",
            })
    if not actions:
        return None
    actions.sort(key=lambda item: (
        -item["score"], item["candidate_id"], FIDELITY_ORDER[item["fidelity"]]
    ))
    winner = dict(actions[0])
    winner["eligible_action_count"] = len(actions)
    return winner


def joint_action_cost_aware_policy(evaluator: PolicyEvaluator, seed: int) -> None:
    """Spend budget by adaptively choosing both candidate and fidelity per call."""
    costs = evaluator.fidelity_costs
    coordinates = {
        cid: tuple(evaluator.design(cid).x) for cid in evaluator.candidate_ids
    }
    evaluator.joint_action_trace = []
    while True:
        budget_before = evaluator.remaining_budget
        choice = choose_joint_action(
            evaluator.records,
            evaluator.candidate_ids,
            coordinates,
            costs,
            budget_before,
            seed,
        )
        if choice is None:
            break
        cid, fidelity = choice["candidate_id"], choice["fidelity"]
        if fidelity == "high":
            predicted, margin, _, _ = _prediction(
                cid,
                {
                    candidate: {
                        obs["fidelity"]: obs
                        for obs in [_row(record) for record in evaluator.records]
                        if obs["candidate_id"] == candidate
                    }
                    for candidate in evaluator.candidate_ids
                },
                coordinates,
            )
            evaluator.record_promotion_basis(cid, margin, "joint_action_cost_aware")
        evaluator.evaluate(cid, fidelity)
        evaluator.joint_action_trace.append({
            "step": len(evaluator.joint_action_trace) + 1,
            "budget_before": budget_before,
            "eligible_action_count": choice["eligible_action_count"],
            "selected_candidate_id": cid,
            "selected_fidelity": fidelity,
            "selected_score": choice["score"],
            "selected_effective_cost": choice["effective_cost"],
            "selected_deferred_validation_cost": choice["deferred_validation_cost"],
            "selected_decision_impact": choice["decision_impact"],
            "selection_mode": choice["selection_mode"],
        })
