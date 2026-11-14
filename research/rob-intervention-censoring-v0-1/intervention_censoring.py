"""MCR² v16 robotics intervention-censoring reference semantics.

This module classifies what is actually observable for two distinct estimands:
(1) untreated natural-history event time T0, and
(2) policy-specific fixed-horizon outcomes under randomized policy assignment.

It intentionally does not implement a survival estimator or causal estimator.
All units and fixtures are synthetic. No actuator authority is represented.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class T0Disposition(str, Enum):
    OBSERVED_BOUNDARY_EVENT = "BoundaryEventObservedBeforeIntervention"
    ADMIN_RIGHT_CENSOR = "AdministrativeRightCensorBeforeIntervention"
    INTERVENTION_COUNTERFACTUAL_UNOBSERVED = "PreventiveInterventionBeforeBoundary"
    POST_INTERVENTION_EVENT_ONLY = "PostInterventionBoundaryEvent"
    COMPETING_TERMINAL_EVENT = "CompetingTerminalEvent"
    OBSERVATION_LOSS = "ObservationOrTelemetryLoss"
    INSTRUMENTATION_INVALID = "InstrumentationInvalid"
    TIME_EVIDENCE_INVALID = "TimeEvidenceInvalid"
    DELIVERY_UNKNOWN = "UnknownDeliveryOrEffect"
    PROTOCOL_DEVIATION = "ProtocolDeviationOrAbort"
    ORDER_AMBIGUOUS = "EventInterventionOrderAmbiguous"
    UNKNOWN = "UnknownOutcome"


@dataclass(frozen=True)
class Episode:
    episode_id: str
    assigned_policy: str
    assignment_design: str
    followup_end: int
    assignment_precedes_time_zero: bool = False
    randomization_receipt_valid: bool = False
    assignment_block_id: str | None = None
    event_time: int | None = None
    event_kind: str | None = None
    intervention_time: int | None = None
    intervention_delivery_known: bool = True
    time_zero_valid: bool = True
    clock_valid: bool = True
    instrumentation_valid: bool = True
    followup_end_reason: str = "scheduled_admin"
    policy_endpoint: str | None = None
    policy_followup_complete: bool = False
    censor_model_qualified: bool = False
    censoring_positivity_supported: bool = False
    censoring_weights_supported: bool = False

    def validate(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not isinstance(self.assigned_policy, str) or not self.assigned_policy:
            raise ValueError("assigned_policy must be non-empty")
        if self.assignment_design not in {"randomized_blocked", "randomized_unblocked", "observational"}:
            raise ValueError("unknown assignment_design")
        for name in ("followup_end", "event_time", "intervention_time"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be null or a non-negative integer")
        if self.followup_end <= 0:
            raise ValueError("followup_end must be positive")
        if self.event_kind not in {None, "boundary", "competing"}:
            raise ValueError("event_kind must be null, boundary, or competing")
        if self.event_time is None and self.event_kind is not None:
            raise ValueError("event_kind requires event_time")
        if self.event_time is not None and self.event_kind is None:
            raise ValueError("event_time requires event_kind")
        if self.event_time is not None and self.event_time > self.followup_end:
            raise ValueError("event_time cannot occur after followup_end")
        if self.intervention_time is not None and self.intervention_time > self.followup_end:
            raise ValueError("intervention_time cannot occur after followup_end")
        bool_fields = (
            "intervention_delivery_known", "time_zero_valid", "clock_valid",
            "instrumentation_valid", "policy_followup_complete",
            "assignment_precedes_time_zero", "randomization_receipt_valid",
            "censor_model_qualified", "censoring_positivity_supported",
            "censoring_weights_supported",
        )
        for name in bool_fields:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.followup_end_reason not in {"scheduled_admin", "telemetry_loss", "abort", "unknown"}:
            raise ValueError("unknown followup_end_reason")
        if self.policy_endpoint not in {None, "safe_success", "boundary_event", "task_failure", "competing_event"}:
            raise ValueError("unknown policy_endpoint")
        if self.policy_followup_complete and self.policy_endpoint is None:
            raise ValueError("complete policy follow-up requires policy_endpoint")
        if self.assignment_block_id is not None and (not isinstance(self.assignment_block_id, str) or not self.assignment_block_id):
            raise ValueError("assignment_block_id must be null or non-empty text")
        if self.assignment_design == "randomized_blocked" and not self.assignment_block_id:
            raise ValueError("randomized_blocked episodes require assignment_block_id")


def classify_untreated_event_time(ep: Episode) -> T0Disposition:
    """Classify evidence for T0. Never infer untreated event time from treated paths."""
    ep.validate()
    if not ep.time_zero_valid or not ep.clock_valid:
        return T0Disposition.TIME_EVIDENCE_INVALID
    if not ep.instrumentation_valid:
        return T0Disposition.INSTRUMENTATION_INVALID
    if not ep.intervention_delivery_known:
        return T0Disposition.DELIVERY_UNKNOWN

    if ep.event_time is not None and ep.intervention_time is not None:
        if ep.event_time == ep.intervention_time:
            return T0Disposition.ORDER_AMBIGUOUS
        if ep.event_time < ep.intervention_time:
            return (T0Disposition.OBSERVED_BOUNDARY_EVENT
                    if ep.event_kind == "boundary"
                    else T0Disposition.COMPETING_TERMINAL_EVENT)
        return (T0Disposition.POST_INTERVENTION_EVENT_ONLY
                if ep.event_kind == "boundary"
                else T0Disposition.COMPETING_TERMINAL_EVENT)

    if ep.event_time is not None:
        return (T0Disposition.OBSERVED_BOUNDARY_EVENT
                if ep.event_kind == "boundary"
                else T0Disposition.COMPETING_TERMINAL_EVENT)

    if ep.intervention_time is not None:
        return T0Disposition.INTERVENTION_COUNTERFACTUAL_UNOBSERVED

    if ep.followup_end_reason == "telemetry_loss":
        return T0Disposition.OBSERVATION_LOSS
    if ep.followup_end_reason == "abort":
        return T0Disposition.PROTOCOL_DEVIATION
    if ep.followup_end_reason == "unknown":
        return T0Disposition.UNKNOWN
    return T0Disposition.ADMIN_RIGHT_CENSOR


def censor_method_admissible(ep: Episode) -> tuple[bool, str]:
    """Minimum receipt gate for #9690-style IPCW use; not a theorem or estimator."""
    disposition = classify_untreated_event_time(ep)
    if disposition is not T0Disposition.ADMIN_RIGHT_CENSOR:
        return False, f"not_administrative_right_censor:{disposition.value}"
    if not ep.censor_model_qualified:
        return False, "censor_model_not_qualified"
    if not ep.censoring_positivity_supported:
        return False, "censoring_positivity_not_supported"
    if not ep.censoring_weights_supported:
        return False, "censoring_weight_receipt_missing_or_unsupported"
    return True, "eligible_for_separately_qualified_censor_adjustment"


def intention_to_treat_summary(
    episodes: Iterable[Episode],
    control_policy: str,
    candidate_policy: str,
) -> dict[str, Any]:
    """Descriptive fixed-horizon ITT summary for two randomized safe policies.

    Event of interest is boundary_event or task_failure. Competing events remain
    separately counted. All assigned episodes must have complete fixed-horizon
    follow-up; censor-adjusted policy estimators are out of scope.
    """
    rows = list(episodes)
    if not rows:
        return {"estimable": False, "reason": "empty_cohort"}
    selected = [e for e in rows if e.assigned_policy in {control_policy, candidate_policy}]
    if not selected:
        return {"estimable": False, "reason": "no_selected_episodes"}
    if any(e.assignment_design not in {"randomized_blocked", "randomized_unblocked"} for e in selected):
        return {"estimable": False, "reason": "assignment_not_randomized"}
    if any(not e.assignment_precedes_time_zero for e in selected):
        return {"estimable": False, "reason": "assignment_not_pre_episode"}
    if any(not e.randomization_receipt_valid for e in selected):
        return {"estimable": False, "reason": "randomization_receipt_invalid_or_missing"}
    if any(not e.policy_followup_complete for e in selected):
        return {"estimable": False, "reason": "incomplete_fixed_horizon_followup_requires_separate_censor_method"}
    if any(e.policy_endpoint is None for e in selected):
        return {"estimable": False, "reason": "missing_policy_endpoint"}
    groups: dict[str, list[Episode]] = {
        control_policy: [e for e in selected if e.assigned_policy == control_policy],
        candidate_policy: [e for e in selected if e.assigned_policy == candidate_policy],
    }
    if not groups[control_policy] or not groups[candidate_policy]:
        return {"estimable": False, "reason": "one_policy_arm_empty"}
    matched_block_count = 0
    if any(e.assignment_design == "randomized_blocked" for e in selected):
        control_blocks = {e.assignment_block_id for e in groups[control_policy]}
        candidate_blocks = {e.assignment_block_id for e in groups[candidate_policy]}
        if control_blocks != candidate_blocks:
            return {"estimable": False, "reason": "blocked_randomization_missing_arm_within_block"}
        matched_block_count = len(control_blocks)

    def summarize(group: list[Episode]) -> dict[str, Any]:
        n = len(group)
        adverse = sum(e.policy_endpoint in {"boundary_event", "task_failure"} for e in group)
        safe = sum(e.policy_endpoint == "safe_success" for e in group)
        competing = sum(e.policy_endpoint == "competing_event" for e in group)
        return {
            "assigned_n": n,
            "adverse_n": adverse,
            "adverse_risk": adverse / n,
            "safe_success_n": safe,
            "competing_event_n": competing,
        }

    control = summarize(groups[control_policy])
    candidate = summarize(groups[candidate_policy])
    return {
        "estimable": True,
        "estimand": "fixed_horizon_intention_to_treat_adverse_event_risk_difference",
        "control_policy": control_policy,
        "candidate_policy": candidate_policy,
        "control": control,
        "candidate": candidate,
        "candidate_minus_control_risk_difference": candidate["adverse_risk"] - control["adverse_risk"],
        "individual_counterfactuals_identified": False,
        "randomization_integrity": {
            "assignment_precedes_time_zero": all(e.assignment_precedes_time_zero for e in selected),
            "randomization_receipt_valid": all(e.randomization_receipt_valid for e in selected),
            "matched_block_count": matched_block_count,
        },
        "uncertainty_interval": None,
        "note": "Descriptive synthetic point estimate only; use preregistered episode/article-clustered inference for confirmatory claims.",
    }


def cohort_identifiability(episodes: Iterable[Episode]) -> dict[str, Any]:
    """Describe whether an untreated T0 reference stratum is present."""
    rows = list(episodes)
    classes = [classify_untreated_event_time(e) for e in rows]
    eligible_reference = sum(
        d in {T0Disposition.OBSERVED_BOUNDARY_EVENT, T0Disposition.ADMIN_RIGHT_CENSOR}
        and e.intervention_time is None
        for e, d in zip(rows, classes)
    )
    treated = sum(e.intervention_time is not None for e in rows)
    return {
        "episodes": len(rows),
        "untreated_reference_episodes": eligible_reference,
        "episodes_with_intervention": treated,
        "t0_identifiability": "reference_stratum_present" if eligible_reference else "not_identifiable_from_this_cohort_alone",
        "warning": "Individual T0 is not recovered for an episode whose intervention changed its trajectory.",
    }


def from_dict(raw: dict[str, Any]) -> Episode:
    episode = Episode(**raw)
    episode.validate()
    return episode
