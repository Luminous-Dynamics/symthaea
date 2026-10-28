#!/usr/bin/env python3
"""Deterministic decision gate for casting multi-fidelity experiments.

This is a transparent decision proxy, not a calibrated probabilistic model.
Callers must estimate decision-loss reduction and constraint-risk reduction
from evidence genuinely available to the policy, never from sealed holdout truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


ACTION_KINDS = frozenset({"evaluate", "escalate", "replicate"})
DISPOSITIONS = frozenset({"evaluate", "escalate", "replicate", "stop", "abstain"})


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class DecisionAction:
    """One permissible next action and policy-side estimates for that action.

    decision_loss_reduction is a non-negative normalized proxy loss reduction.
    normalized_constraint_risk_reduction is a [0, 1] decision proxy; it is
    NOT a probability, confidence level, or calibrated risk bound.
    estimated_cost_units must include the marginal cost visible to the policy.
    """

    action_id: str
    candidate_id: str
    kind: str
    fidelity: str
    estimated_cost_units: float
    decision_loss_reduction: float
    normalized_constraint_risk_reduction: float = 0.0

    def validate(self) -> None:
        _nonblank(self.action_id, "action_id")
        _nonblank(self.candidate_id, "candidate_id")
        _nonblank(self.fidelity, "fidelity")
        if self.kind not in ACTION_KINDS:
            raise ValueError(f"kind must be one of {sorted(ACTION_KINDS)}")
        cost = _finite_number(self.estimated_cost_units, "estimated_cost_units")
        value = _finite_number(self.decision_loss_reduction, "decision_loss_reduction")
        risk_reduction = _finite_number(
            self.normalized_constraint_risk_reduction,
            "normalized_constraint_risk_reduction",
        )
        if cost <= 0.0:
            raise ValueError("estimated_cost_units must be strictly positive")
        if value < 0.0:
            raise ValueError("decision_loss_reduction must be non-negative")
        if not 0.0 <= risk_reduction <= 1.0:
            raise ValueError("normalized_constraint_risk_reduction must be in [0, 1]")


@dataclass(frozen=True)
class GateDecision:
    """Auditable disposition; estimated cost is reserved, not realized cost."""

    disposition: str
    reason_code: str
    action_id: str | None
    candidate_id: str | None
    fidelity: str | None
    estimated_cost_units: float
    reserved_cost_units: float
    decision_loss_reduction: float | None
    value_per_cost: float | None
    quality_target_met: bool
    risk_gate_satisfied: bool
    remaining_cost_units: float
    decision_proxy_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def choose_next_step(
    actions: Iterable[DecisionAction],
    *,
    remaining_cost_units: float,
    quality_target_met: bool,
    risk_gate_satisfied: bool,
    minimum_value_per_cost: float = 0.0,
    minimum_risk_reduction: float = 0.05,
) -> GateDecision:
    """Choose an affordable next action or explicitly stop/abstain.

    When the risk gate is unresolved, only actions that meet the declared
    positive risk-reduction proxy are eligible. If none does, the policy
    abstains rather than declaring the design safe or silently stopping.

    Among eligible actions, maximize decision-loss-reduction per estimated
    cost. Stable tie-breaking is by greater risk-reduction proxy, then
    lexicographically smaller action_id. No hidden truth is read here.
    """
    if not isinstance(quality_target_met, bool) or not isinstance(risk_gate_satisfied, bool):
        raise ValueError("quality_target_met and risk_gate_satisfied must be booleans")

    budget = _finite_number(remaining_cost_units, "remaining_cost_units")
    threshold = _finite_number(minimum_value_per_cost, "minimum_value_per_cost")
    min_risk = _finite_number(minimum_risk_reduction, "minimum_risk_reduction")
    if budget < 0.0:
        raise ValueError("remaining_cost_units must be non-negative")
    if threshold < 0.0:
        raise ValueError("minimum_value_per_cost must be non-negative")
    if not 0.0 <= min_risk <= 1.0:
        raise ValueError("minimum_risk_reduction must be in [0, 1]")

    validated = list(actions)
    seen_ids: set[str] = set()
    for action in validated:
        if not isinstance(action, DecisionAction):
            raise ValueError("actions must contain only DecisionAction values")
        action.validate()
        if action.action_id in seen_ids:
            raise ValueError(f"duplicate action_id: {action.action_id}")
        seen_ids.add(action.action_id)

    if quality_target_met and risk_gate_satisfied:
        return GateDecision(
            disposition="stop",
            reason_code="quality_target_and_risk_gate_satisfied",
            action_id=None,
            candidate_id=None,
            fidelity=None,
            estimated_cost_units=0.0,
            reserved_cost_units=0.0,
            decision_loss_reduction=None,
            value_per_cost=None,
            quality_target_met=quality_target_met,
            risk_gate_satisfied=risk_gate_satisfied,
            remaining_cost_units=budget,
        )

    affordable = [
        action for action in validated
        if float(action.estimated_cost_units) <= budget
    ]
    if not risk_gate_satisfied:
        affordable = [
            action for action in affordable
            if float(action.normalized_constraint_risk_reduction) > 0.0
            and float(action.normalized_constraint_risk_reduction) >= min_risk
        ]

    scored: list[tuple[float, float, str, DecisionAction]] = []
    for action in affordable:
        score = float(action.decision_loss_reduction) / float(action.estimated_cost_units)
        if float(action.decision_loss_reduction) > 0.0 and score > threshold:
            scored.append((
                score,
                float(action.normalized_constraint_risk_reduction),
                action.action_id,
                action,
            ))

    if not scored:
        if not risk_gate_satisfied:
            disposition = "abstain"
            reason = "risk_gate_unresolved_no_qualifying_affordable_action"
        elif not affordable:
            disposition = "stop"
            reason = "no_affordable_action_above_value_threshold"
        else:
            disposition = "stop"
            reason = "marginal_decision_value_below_threshold"
        return GateDecision(
            disposition=disposition,
            reason_code=reason,
            action_id=None,
            candidate_id=None,
            fidelity=None,
            estimated_cost_units=0.0,
            reserved_cost_units=0.0,
            decision_loss_reduction=None,
            value_per_cost=None,
            quality_target_met=quality_target_met,
            risk_gate_satisfied=risk_gate_satisfied,
            remaining_cost_units=budget,
        )

    # Deterministic tie handling: score descending, risk reduction descending,
    # then action_id ascending. Keep the raw score in the receipt.
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    score, _, _, selected = scored[0]
    return GateDecision(
        disposition=selected.kind,
        reason_code="highest_decision_value_per_cost",
        action_id=selected.action_id,
        candidate_id=selected.candidate_id,
        fidelity=selected.fidelity,
        estimated_cost_units=float(selected.estimated_cost_units),
        reserved_cost_units=float(selected.estimated_cost_units),
        decision_loss_reduction=float(selected.decision_loss_reduction),
        value_per_cost=score,
        quality_target_met=quality_target_met,
        risk_gate_satisfied=risk_gate_satisfied,
        remaining_cost_units=budget,
    )


if __name__ == "__main__":
    # Compact deterministic smoke example, useful without external packages.
    sample = [
        DecisionAction("screen-a", "candidate-a", "evaluate", "low", 1, 0.2, 0.0),
        DecisionAction("verify-a", "candidate-a", "escalate", "high", 5, 0.8, 0.4),
    ]
    print(choose_next_step(
        sample,
        remaining_cost_units=5,
        quality_target_met=False,
        risk_gate_satisfied=False,
        minimum_value_per_cost=0.05,
        minimum_risk_reduction=0.1,
    ).to_dict())
