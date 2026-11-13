"""MCR² v15 synthetic causal-attribution benchmark.

Deterministic scalar-margin reference. This is not a physical plant model,
controller, safety proof, or deployment benchmark.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Arm(str, Enum):
    NO_ACTION = "no_action"
    REACTIVE = "reactive_threshold"
    PREDICTIVE = "predictive_reflex"
    SHAM = "shuffled_precursor_control"
    ORACLE = "evaluator_oracle"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    horizon_ticks: int
    initial_margin_milli: int
    hazard_start_tick: int | None
    loss_per_tick_milli: int
    mitigation_per_tick_milli: int
    observed_precursor_tick: int | None
    sham_precursor_tick: int | None
    time_to_boundary_lower_ticks: int | None
    predictive_latency_ticks: int
    reactive_latency_ticks: int
    reserve_ticks: int
    reactive_threshold_milli: int
    compute_cost_units: int
    compute_budget_units: int
    intervention_cost_units: int
    evidence_qualified: bool
    task_context_matches: bool

    def validate(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")
        ints = (
            self.horizon_ticks, self.initial_margin_milli,
            self.loss_per_tick_milli, self.mitigation_per_tick_milli,
            self.predictive_latency_ticks, self.reactive_latency_ticks,
            self.reserve_ticks, self.reactive_threshold_milli,
            self.compute_cost_units, self.compute_budget_units,
            self.intervention_cost_units,
        )
        if any(type(x) is not int or x < 0 for x in ints):
            raise ValueError("integer fields must be non-negative integers")
        if self.horizon_ticks < 1:
            raise ValueError("horizon_ticks must be positive")
        if self.hazard_start_tick is not None and (
            type(self.hazard_start_tick) is not int
            or not 0 <= self.hazard_start_tick < self.horizon_ticks
        ):
            raise ValueError("hazard_start_tick is outside episode")
        for name, value in (
            ("observed_precursor_tick", self.observed_precursor_tick),
            ("sham_precursor_tick", self.sham_precursor_tick),
        ):
            if value is not None and (
                type(value) is not int or not 0 <= value < self.horizon_ticks
            ):
                raise ValueError(f"{name} is outside episode")
        if self.time_to_boundary_lower_ticks is not None and (
            type(self.time_to_boundary_lower_ticks) is not int
            or self.time_to_boundary_lower_ticks < 0
        ):
            raise ValueError("time_to_boundary_lower_ticks must be null or non-negative")
        if type(self.evidence_qualified) is not bool or type(self.task_context_matches) is not bool:
            raise ValueError("qualification flags must be booleans")


@dataclass(frozen=True)
class EpisodeResult:
    scenario_id: str
    arm: str
    safe: bool
    final_margin_milli: int
    boundary_crossing_tick: int | None
    trigger_tick: int | None
    effect_tick: int | None
    intervention_count: int
    false_intervention: bool
    compute_cost_units: int
    intervention_cost_units: int
    disposition: str


def _no_action_fails(s: Scenario) -> bool:
    if s.hazard_start_tick is None:
        return False
    margin = s.initial_margin_milli
    for tick in range(s.horizon_ticks):
        if tick >= s.hazard_start_tick:
            margin -= s.loss_per_tick_milli
        if margin <= 0:
            return True
    return False


def _eligible(s: Scenario, signal_tick: int | None) -> tuple[bool, str]:
    if signal_tick is None:
        return False, "no_precursor_signal"
    if not s.evidence_qualified:
        return False, "precursor_evidence_unqualified"
    if not s.task_context_matches:
        return False, "task_context_mismatch"
    if s.time_to_boundary_lower_ticks is None:
        return False, "time_to_boundary_unknown"
    if s.compute_cost_units > s.compute_budget_units:
        return False, "compute_budget_exceeded"
    if s.time_to_boundary_lower_ticks <= s.predictive_latency_ticks + s.reserve_ticks:
        return False, "no_positive_effect_entry_margin"
    return True, "eligible_for_independent_projection"


def simulate(s: Scenario, arm: Arm) -> EpisodeResult:
    """Replay one arm against the scenario's frozen exogenous disturbance."""
    s.validate()
    margin = s.initial_margin_milli
    trigger_tick = effect_tick = crossing_tick = None
    intervention_active = False
    compute_cost = action_cost = 0
    disposition = "no_action"
    signal_tick = (
        s.observed_precursor_tick if arm is Arm.PREDICTIVE
        else s.sham_precursor_tick if arm is Arm.SHAM
        else None
    )
    eligible = False
    if arm in (Arm.PREDICTIVE, Arm.SHAM):
        compute_cost = s.compute_cost_units
        eligible, disposition = _eligible(s, signal_tick)

    for tick in range(s.horizon_ticks):
        if effect_tick == tick:
            intervention_active = True
        if arm in (Arm.PREDICTIVE, Arm.SHAM) and eligible and trigger_tick is None:
            if signal_tick == tick:
                trigger_tick = tick
                effect_tick = tick + s.predictive_latency_ticks
                action_cost = s.intervention_cost_units
                disposition = "scheduled_for_independent_projection"
        if arm is Arm.REACTIVE and trigger_tick is None and margin <= s.reactive_threshold_milli:
            trigger_tick = tick
            effect_tick = tick + s.reactive_latency_ticks
            action_cost = s.intervention_cost_units
            disposition = "reactive_threshold_triggered"
        if arm is Arm.ORACLE and trigger_tick is None and s.hazard_start_tick is not None:
            trigger_tick = max(0, s.hazard_start_tick - s.predictive_latency_ticks)
            effect_tick = trigger_tick + s.predictive_latency_ticks
            if tick == trigger_tick:
                action_cost = s.intervention_cost_units
                disposition = "evaluator_oracle_only"
        if s.hazard_start_tick is not None and tick >= s.hazard_start_tick:
            loss = s.loss_per_tick_milli
            if intervention_active:
                loss = max(0, loss - s.mitigation_per_tick_milli)
            margin -= loss
        if margin <= 0:
            crossing_tick = tick
            break

    safe = crossing_tick is None
    false_intervention = trigger_tick is not None and (
        s.hazard_start_tick is None or not _no_action_fails(s)
    )
    return EpisodeResult(
        s.scenario_id, arm.value, safe, margin, crossing_tick,
        trigger_tick, effect_tick, int(trigger_tick is not None),
        false_intervention, compute_cost, action_cost, disposition,
    )


def compare(scenarios: list[Scenario]) -> dict[str, Any]:
    """Paired scenario-level comparison; oracle is evaluator-only."""
    arms = (Arm.NO_ACTION, Arm.REACTIVE, Arm.PREDICTIVE, Arm.SHAM, Arm.ORACLE)
    rows = {arm.value: [simulate(s, arm) for s in scenarios] for arm in arms}

    def summary(results: list[EpisodeResult]) -> dict[str, int | float]:
        n = len(results)
        return {
            "episodes": n,
            "safe_episodes": sum(r.safe for r in results),
            "safe_success_rate": sum(r.safe for r in results) / n if n else 0.0,
            "interventions": sum(r.intervention_count for r in results),
            "false_interventions": sum(r.false_intervention for r in results),
            "total_compute_cost_units": sum(r.compute_cost_units for r in results),
            "total_intervention_cost_units": sum(r.intervention_cost_units for r in results),
        }

    def paired(a: list[EpisodeResult], b: list[EpisodeResult]) -> dict[str, int | float]:
        wins = sum(x.safe and not y.safe for x, y in zip(a, b))
        losses = sum(not x.safe and y.safe for x, y in zip(a, b))
        return {
            "treatment_only_safe": wins,
            "control_only_safe": losses,
            "net_paired_safe_lift": (wins - losses) / len(a) if a else 0.0,
            "paired_episodes": len(a),
        }

    no_action = rows[Arm.NO_ACTION.value]
    predictive = rows[Arm.PREDICTIVE.value]
    preventable = sum(not r.safe for r in no_action)
    intercepted = sum((not n.safe) and p.safe for n, p in zip(no_action, predictive))
    return {
        "schema_version": "mcr2-causal-reflex-report-v0.1",
        "scenario_count": len(scenarios),
        "independent_unit": "scenario_episode",
        "arms": {name: summary(results) for name, results in rows.items()},
        "predictive_vs_reactive": paired(predictive, rows[Arm.REACTIVE.value]),
        "predictive_vs_sham": paired(predictive, rows[Arm.SHAM.value]),
        "predictive_preventable_event_interception": {
            "preventable_events_without_action": preventable,
            "prevented_by_predictive_arm": intercepted,
            "interception_rate": intercepted / preventable if preventable else None,
        },
        "oracle_is_evaluator_only": True,
        "claim_boundary": "synthetic_deterministic_reference_only",
    }


def from_dict(raw: dict[str, Any]) -> Scenario:
    scenario = Scenario(**raw)
    scenario.validate()
    return scenario
