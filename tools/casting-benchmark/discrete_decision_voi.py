#!/usr/bin/env python3
"""Exact discrete Bayesian value-of-information reference calculation.

The policy receives a prior, decision-loss table, and predictive observation
likelihoods. It does not receive the evaluator's realized world or sealed
holdout outcomes. This module computes VOI for finite known distributions; it
does not establish that an upstream predictive model is calibrated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence


TOL = 1e-10


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string without edge whitespace")
    return value


def _distribution(values: Sequence[object], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{label} must be a non-empty numeric sequence")
    numbers = tuple(_finite_number(value, f"{label}[{index}]") for index, value in enumerate(values))
    if any(value < 0.0 or value > 1.0 for value in numbers):
        raise ValueError(f"{label} probabilities must be in [0, 1]")
    total = sum(numbers)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=TOL):
        raise ValueError(f"{label} probabilities must sum to one")
    return tuple(value / total for value in numbers)


@dataclass(frozen=True)
class ObservationAction:
    """An observation option with P(observation | latent world) and known cost.

    likelihoods[world_index][outcome_index] contains a predictive probability,
    not a realized holdout result. Rows must be valid categorical distributions.
    """

    action_id: str
    estimated_cost_units: float
    likelihoods: tuple[tuple[float, ...], ...]

    def validate(self, world_count: int) -> tuple[tuple[float, ...], ...]:
        _nonblank(self.action_id, "action_id")
        cost = _finite_number(self.estimated_cost_units, "estimated_cost_units")
        if cost <= 0.0:
            raise ValueError("estimated_cost_units must be strictly positive")
        if isinstance(self.likelihoods, (str, bytes)) or not isinstance(self.likelihoods, Sequence):
            raise ValueError("likelihoods must be a sequence of world rows")
        if len(self.likelihoods) != world_count:
            raise ValueError("likelihoods must have one row per latent world")
        normalized_rows: list[tuple[float, ...]] = []
        outcome_count: int | None = None
        for index, row in enumerate(self.likelihoods):
            normalized = _distribution(row, f"likelihoods[{index}]")
            if outcome_count is None:
                outcome_count = len(normalized)
            elif len(normalized) != outcome_count:
                raise ValueError("likelihood rows must have the same number of outcomes")
            normalized_rows.append(normalized)
        return tuple(normalized_rows)


@dataclass(frozen=True)
class ActionVOIScore:
    action_id: str
    current_decision_index: int
    current_bayes_risk: float
    expected_posterior_bayes_risk: float
    expected_value_of_information: float
    estimated_cost_units: float
    voi_per_cost: float
    observation_probabilities: tuple[float, ...]
    posterior_decision_indices: tuple[int | None, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _bayes_decision_and_risk(
    prior: Sequence[float],
    losses: Sequence[Sequence[float]],
) -> tuple[int, float]:
    expected_losses = [
        sum(prior[world] * losses[decision][world] for world in range(len(prior)))
        for decision in range(len(losses))
    ]
    best_index = min(range(len(expected_losses)), key=lambda i: (expected_losses[i], i))
    return best_index, expected_losses[best_index]


def evaluate_action(
    prior: Sequence[object],
    decision_losses: Sequence[Sequence[object]],
    action: ObservationAction,
) -> ActionVOIScore:
    """Compute exact finite-state expected VOI and VOI per evaluation cost.

    decision_losses[d][w] is the loss of taking decision d when world w is true.
    The selected decision minimizes expected loss. A query's VOI is current
    Bayes risk minus expected posterior Bayes risk across all possible outcomes.
    """
    normalized_prior = _distribution(prior, "prior")
    world_count = len(normalized_prior)
    if isinstance(decision_losses, (str, bytes)) or not isinstance(decision_losses, Sequence) or not decision_losses:
        raise ValueError("decision_losses must contain at least one decision row")

    loss_rows: list[tuple[float, ...]] = []
    for index, row in enumerate(decision_losses):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != world_count:
            raise ValueError(f"decision_losses[{index}] must contain one loss per world")
        losses = tuple(_finite_number(value, f"decision_losses[{index}][{world}]") for world, value in enumerate(row))
        if any(value < 0.0 for value in losses):
            raise ValueError("decision losses must be non-negative")
        loss_rows.append(losses)

    normalized_losses = tuple(loss_rows)
    current_decision, current_risk = _bayes_decision_and_risk(normalized_prior, normalized_losses)
    likelihoods = action.validate(world_count)
    outcome_count = len(likelihoods[0])

    observation_probabilities: list[float] = []
    posterior_decisions: list[int | None] = []
    expected_posterior_risk = 0.0

    for outcome_index in range(outcome_count):
        outcome_probability = sum(
            normalized_prior[world] * likelihoods[world][outcome_index]
            for world in range(world_count)
        )
        observation_probabilities.append(outcome_probability)
        if outcome_probability <= TOL:
            posterior_decisions.append(None)
            continue

        posterior = tuple(
            normalized_prior[world] * likelihoods[world][outcome_index] / outcome_probability
            for world in range(world_count)
        )
        decision_index, posterior_risk = _bayes_decision_and_risk(posterior, normalized_losses)
        posterior_decisions.append(decision_index)
        expected_posterior_risk += outcome_probability * posterior_risk

    voi = current_risk - expected_posterior_risk
    if voi < -TOL:
        raise ArithmeticError("expected posterior Bayes risk exceeded current Bayes risk")
    voi = max(0.0, voi)
    cost = float(action.estimated_cost_units)
    return ActionVOIScore(
        action_id=action.action_id,
        current_decision_index=current_decision,
        current_bayes_risk=current_risk,
        expected_posterior_bayes_risk=expected_posterior_risk,
        expected_value_of_information=voi,
        estimated_cost_units=cost,
        voi_per_cost=voi / cost,
        observation_probabilities=tuple(observation_probabilities),
        posterior_decision_indices=tuple(posterior_decisions),
    )


def rank_actions(
    prior: Sequence[object],
    decision_losses: Sequence[Sequence[object]],
    actions: Iterable[ObservationAction],
    *,
    remaining_cost_units: float,
    minimum_voi_per_cost: float = 0.0,
) -> list[ActionVOIScore]:
    """Rank affordable actions by exact discrete VOI per cost, deterministically."""
    budget = _finite_number(remaining_cost_units, "remaining_cost_units")
    threshold = _finite_number(minimum_voi_per_cost, "minimum_voi_per_cost")
    if budget < 0.0:
        raise ValueError("remaining_cost_units must be non-negative")
    if threshold < 0.0:
        raise ValueError("minimum_voi_per_cost must be non-negative")

    action_list = list(actions)
    seen: set[str] = set()
    for action in action_list:
        if not isinstance(action, ObservationAction):
            raise ValueError("actions must contain only ObservationAction values")
        _nonblank(action.action_id, "action_id")
        if action.action_id in seen:
            raise ValueError(f"duplicate action_id: {action.action_id}")
        seen.add(action.action_id)
        # Validate even unaffordable actions to reject malformed benchmark input.
        action.validate(len(prior))

    ranked = [
        evaluate_action(prior, decision_losses, action)
        for action in action_list
        if float(action.estimated_cost_units) <= budget
    ]
    ranked = [score for score in ranked if score.voi_per_cost > threshold]
    ranked.sort(key=lambda score: (-score.voi_per_cost, score.estimated_cost_units, score.action_id))
    return ranked


def select_best_action(
    prior: Sequence[object],
    decision_losses: Sequence[Sequence[object]],
    actions: Iterable[ObservationAction],
    *,
    remaining_cost_units: float,
    minimum_voi_per_cost: float = 0.0,
) -> ActionVOIScore | None:
    """Return the best affordable action or None when no action clears threshold."""
    ranked = rank_actions(
        prior,
        decision_losses,
        actions,
        remaining_cost_units=remaining_cost_units,
        minimum_voi_per_cost=minimum_voi_per_cost,
    )
    return ranked[0] if ranked else None
