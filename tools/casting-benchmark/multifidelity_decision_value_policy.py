#!/usr/bin/env python3
"""Decision-impact-weighted multi-fidelity acquisition heuristic.

This is an interpretable research baseline, not exact Bayesian value of
information. It estimates the decision impact of an expensive confirmation from
predicted feasible hypervolume gain, uncertainty near the feasibility boundary,
and objective uncertainty near the predicted trade-off frontier.
"""
from __future__ import annotations

import math
import statistics
from typing import Callable, Iterable

from multifidelity_stress import (
    Observation,
    PolicyEvaluator,
    deterministic_sample,
    hypervolume_2d,
    pareto_front,
    select_diverse,
)

POLICY_ID = "decision_value_of_information"
LOW_CALLS = 60
MEDIUM_CALLS = 8
HIGH_CALLS = 4
SCORE_WEIGHTS = {
    "expected_hypervolume_gain": 0.70,
    "constraint_boundary_information": 0.20,
    "frontier_objective_information": 0.10,
}
HYPERVOLUME_AREA = 1.2 * 1.2
MIN_SIGMA = 1e-6


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _robust_sigma(values: list[float]) -> float:
    """MAD-based scale with a small numerical floor."""
    if not values:
        return MIN_SIGMA
    center = _median(values)
    mad = _median([abs(value - center) for value in values])
    # With tiny paired samples MAD can be zero despite disagreement; retain a
    # standard-deviation fallback so uncertainty does not collapse artificially.
    mean = statistics.fmean(values)
    variance = statistics.fmean((value - mean) ** 2 for value in values)
    return max(MIN_SIGMA, 1.4826 * mad, math.sqrt(max(0.0, variance)) * 0.5)


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("candidate coordinate dimensions differ")
    return math.sqrt(statistics.fmean((a - b) ** 2 for a, b in zip(left, right)))


def _weighted_local(
    candidate_id: int,
    references: list[Observation],
    deltas: dict[int, tuple[float, ...]],
    evaluator: PolicyEvaluator,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Estimate multi-output fidelity deltas and uncertainty from paired probes."""
    if not references:
        width = len(next(iter(deltas.values()))) if deltas else 3
        return (tuple(0.0 for _ in range(width)), tuple(0.1 for _ in range(width)), 1.0)
    target = evaluator.design(candidate_id).x
    ranked = sorted(
        ((_distance(target, evaluator.design(obs.candidate_id).x), obs.candidate_id) for obs in references),
        key=lambda item: (item[0], item[1]),
    )
    nearest = ranked[: min(4, len(ranked))]
    # Distance weighting makes corrections local, but the offset prevents one
    # nearly identical input from erasing all uncertainty.
    raw_weights = [1.0 / ((distance + 0.02) ** 2) for distance, _ in nearest]
    total_weight = sum(raw_weights)
    weights = [weight / total_weight for weight in raw_weights]
    width = len(next(iter(deltas.values())))
    means = tuple(
        sum(weight * deltas[candidate][j] for weight, (_, candidate) in zip(weights, nearest))
        for j in range(width)
    )
    global_scales = [
        _robust_sigma([delta[j] for delta in deltas.values()])
        for j in range(width)
    ]
    local_scales = []
    for j in range(width):
        local_var = sum(
            weight * (deltas[candidate][j] - means[j]) ** 2
            for weight, (_, candidate) in zip(weights, nearest)
        )
        # Penalize extrapolation away from paired observations. Distances are
        # RMS-normalized by input dimensionality, so this term is comparable
        # across the supported dimensions.
        nearest_distance = nearest[0][0]
        local_scales.append(
            max(MIN_SIGMA, math.sqrt(local_var + (0.35 * global_scales[j]) ** 2
                                     + (nearest_distance * global_scales[j]) ** 2))
        )
    return means, tuple(local_scales), nearest[0][0]


def _normal_cdf(value: float, sigma: float) -> float:
    return 0.5 * (1.0 + math.erf(value / (max(sigma, MIN_SIGMA) * math.sqrt(2.0))))


def _frontier_distance(
    objectives: tuple[float, float], feasible_archive: list[Observation]
) -> float:
    front = pareto_front([obs.objectives for obs in feasible_archive])
    if not front:
        return 0.0
    return min(math.sqrt((objectives[0] - x) ** 2 + (objectives[1] - y) ** 2) for x, y in front)


def _predict(
    candidate_id: int,
    low_by_id: dict[int, Observation],
    medium: list[Observation],
    high_anchors: list[Observation],
    evaluator: PolicyEvaluator,
) -> tuple[tuple[float, float], float, tuple[float, float, float]]:
    medium_by_id = {obs.candidate_id: obs for obs in medium}
    paired_low = [obs for obs in medium if obs.candidate_id in low_by_id]
    low_to_medium: dict[int, tuple[float, ...]] = {
        obs.candidate_id: (
            obs.objectives[0] - low_by_id[obs.candidate_id].objectives[0],
            obs.objectives[1] - low_by_id[obs.candidate_id].objectives[1],
            obs.constraint_margin - low_by_id[obs.candidate_id].constraint_margin,
        )
        for obs in paired_low
    }
    high_to_medium: dict[int, tuple[float, ...]] = {
        obs.candidate_id: (
            next(h for h in high_anchors if h.candidate_id == obs.candidate_id).objectives[0] - obs.objectives[0],
            next(h for h in high_anchors if h.candidate_id == obs.candidate_id).objectives[1] - obs.objectives[1],
            next(h for h in high_anchors if h.candidate_id == obs.candidate_id).constraint_margin - obs.constraint_margin,
        )
        for obs in medium if any(h.candidate_id == obs.candidate_id for h in high_anchors)
    }
    first_mean, first_sigma, _ = _weighted_local(candidate_id, paired_low, low_to_medium, evaluator)
    anchor_refs = [medium_by_id[obs.candidate_id] for obs in high_anchors]
    second_mean, second_sigma, _ = _weighted_local(candidate_id, anchor_refs, high_to_medium, evaluator)
    low = low_by_id[candidate_id]
    predicted = (
        low.objectives[0] + first_mean[0] + second_mean[0],
        low.objectives[1] + first_mean[1] + second_mean[1],
    )
    margin = low.constraint_margin + first_mean[2] + second_mean[2]
    sigma = (
        math.sqrt(first_sigma[0] ** 2 + second_sigma[0] ** 2),
        math.sqrt(first_sigma[1] ** 2 + second_sigma[1] ** 2),
        math.sqrt(first_sigma[2] ** 2 + second_sigma[2] ** 2),
    )
    return predicted, margin, sigma


def _decision_impact_score(
    candidate_id: int,
    predicted_objectives: tuple[float, float],
    predicted_margin: float,
    sigma: tuple[float, float, float],
    feasible_archive: list[Observation],
) -> float:
    sigma_f0, sigma_f1, sigma_margin = sigma
    p_feasible = _normal_cdf(predicted_margin, sigma_margin)
    current_hv = hypervolume_2d([obs.objectives for obs in feasible_archive])
    proposed_hv = hypervolume_2d([obs.objectives for obs in feasible_archive] + [predicted_objectives])
    normalized_gain = max(0.0, proposed_hv - current_hv) / HYPERVOLUME_AREA
    boundary_uncertainty = min(1.0, sigma_margin / 0.08)
    boundary_information = boundary_uncertainty * math.exp(
        -abs(predicted_margin) / max(sigma_margin, MIN_SIGMA)
    )
    objective_uncertainty = min(1.0, (sigma_f0 + sigma_f1) / 0.20)
    distance = _frontier_distance(predicted_objectives, feasible_archive)
    frontier_proximity = math.exp(-distance / 0.10)
    objective_information = objective_uncertainty * frontier_proximity
    return (
        SCORE_WEIGHTS["expected_hypervolume_gain"] * p_feasible * normalized_gain
        + SCORE_WEIGHTS["constraint_boundary_information"] * boundary_information
        + SCORE_WEIGHTS["frontier_objective_information"] * objective_information
    )


def decision_value_of_information_policy(evaluator: PolicyEvaluator, seed: int) -> None:
    """Spend 60 low + 8 medium + 4 high calls under the standard 200-unit budget.

    The acquisition is a transparent decision-impact heuristic, not an exact
    posterior expected value of information. The four high calls are two
    discrepancy anchors and two sequential confirmations.
    """
    costs = evaluator.fidelity_costs
    required = LOW_CALLS * costs["low"] + MEDIUM_CALLS * costs["medium"] + HIGH_CALLS * costs["high"]
    if evaluator.cost_budget < required:
        raise ValueError(
            f"{POLICY_ID} requires at least {required} cost units for its frozen call template"
        )
    low_ids = deterministic_sample(evaluator.candidate_ids, LOW_CALLS, seed)
    low = [evaluator.evaluate(candidate_id, "low") for candidate_id in low_ids]
    low_by_id = {obs.candidate_id: obs for obs in low}

    # Ensure medium probes span active constraints, objective extremes and
    # scalarization trade-offs rather than only sampling the predicted winner.
    selected: list[int] = []
    boundary = sorted(low, key=lambda obs: (abs(obs.constraint_margin), obs.candidate_id))
    for candidate_id in [obs.candidate_id for obs in boundary[:2]]:
        if candidate_id not in selected:
            selected.append(candidate_id)
    for obj_index in (0, 1):
        best = min(low, key=lambda obs: (obs.objectives[obj_index], obs.candidate_id))
        if best.candidate_id not in selected:
            selected.append(best.candidate_id)
    for candidate_id, _ in select_diverse(
        low, len(low), weights=(0.0, 0.25, 0.5, 0.75, 1.0)
    ):
        if candidate_id not in selected and len(selected) < MEDIUM_CALLS:
            selected.append(candidate_id)
    if len(selected) < MEDIUM_CALLS:
        selected.extend(
            cid for cid in low_ids if cid not in selected and len(selected) < MEDIUM_CALLS
        )
    selected = selected[:MEDIUM_CALLS]
    medium = [evaluator.evaluate(candidate_id, "medium") for candidate_id in selected]
    medium_by_id = {obs.candidate_id: obs for obs in medium}

    # One anchor is deliberately near the predicted feasibility boundary; the
    # other expands objective-space/design-space coverage.
    anchor_first = min(medium, key=lambda obs: (abs(obs.constraint_margin), obs.candidate_id))
    remaining_medium = [obs for obs in medium if obs.candidate_id != anchor_first.candidate_id]
    anchor_second = max(
        remaining_medium,
        key=lambda obs: (
            _distance(
                evaluator.design(obs.candidate_id).x,
                evaluator.design(anchor_first.candidate_id).x,
            ),
            obs.candidate_id,
        ),
    )
    anchors = [anchor_first, anchor_second]
    high_anchors = []
    for medium_obs in anchors:
        evaluator.record_promotion_basis(
            medium_obs.candidate_id,
            medium_obs.constraint_margin,
            "decision_impact_anchor",
        )
        high_anchors.append(evaluator.evaluate(medium_obs.candidate_id, "high"))

    while evaluator.remaining_budget >= costs["high"] and len(evaluator.high_observations) < HIGH_CALLS:
        high_ids = {obs.candidate_id for obs in evaluator.high_observations}
        feasible_archive = [obs for obs in evaluator.high_observations if obs.constraint_margin >= 0.0]
        options = []
        for candidate_id in low_ids:
            if candidate_id in high_ids:
                continue
            predicted_objectives, predicted_margin, sigma = _predict(
                candidate_id, low_by_id, medium, high_anchors, evaluator
            )
            score = _decision_impact_score(
                candidate_id, predicted_objectives, predicted_margin, sigma, feasible_archive
            )
            options.append((score, candidate_id, predicted_margin))
        if not options:
            break
        _, winner, predicted_margin = max(options, key=lambda item: (item[0], -item[1]))
        evaluator.record_promotion_basis(
            winner, predicted_margin, "decision_impact_confirmation"
        )
        evaluator.evaluate(winner, "high")


POLICY = decision_value_of_information_policy
