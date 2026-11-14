"""MCR² v17 deterministic known-truth DGP and evaluation diagnostics.

Synthetic-only benchmark scaffold. It is not a survival estimator, causal estimator,
robot simulator, controller, or actuator interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable

POLICIES = ("reference_safe", "anticipatory_safe")
SCENARIOS = {
    "complete_followup": {"high": 1.0, "low": 1.0},
    "mcar_endpoint_loss": {"high": 0.90, "low": 0.90},
    "conditional_endpoint_loss": {"high": 0.48, "low": 0.96},
    "positivity_failure": {"high": 0.0, "low": 0.96},
}
FORBIDDEN_OBSERVED_KEYS = {
    "reference_event_time", "candidate_event_time", "reference_potential_outcome",
    "candidate_potential_outcome", "potential_outcomes", "oracle_truth",
    "counterfactual_event_time", "unassigned_policy_outcome",
}


@dataclass(frozen=True)
class DGPConfig:
    seed: int = 20261113
    blocks: int = 10_000
    horizon: float = 20.0
    reference_hazard: tuple[tuple[str, float], ...] = (("high", 0.12), ("low", 0.02))
    candidate_high_hazard: float = 0.06
    competing_hazard: float = 0.004
    positivity_floor: float = 0.05

    def validate(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(self.blocks) is not int or self.blocks < 2 or self.blocks % 2:
            raise ValueError("blocks must be an even integer >= 2")
        if not isinstance(self.horizon, (int, float)) or isinstance(self.horizon, bool) or self.horizon <= 0:
            raise ValueError("horizon must be positive")
        hazards = dict(self.reference_hazard)
        if set(hazards) != {"high", "low"} or any(v <= 0 for v in hazards.values()):
            raise ValueError("reference_hazard must define positive high and low rates")
        if not (0 < self.candidate_high_hazard < hazards["high"]):
            raise ValueError("candidate_high_hazard must be positive and below high reference hazard")
        if self.competing_hazard < 0:
            raise ValueError("competing_hazard must be non-negative")
        if not (0 < self.positivity_floor <= 1):
            raise ValueError("positivity_floor must be in (0, 1]")


def _exponential_time(rng: random.Random, hazard: float) -> float:
    u = max(rng.random(), 1e-15)
    return -math.log(u) / hazard


def _terminal_outcome(event_time: float, competing_time: float, horizon: float) -> str:
    if event_time <= horizon and event_time < competing_time:
        return "boundary_event"
    if competing_time <= horizon and competing_time < event_time:
        return "competing_event"
    return "event_free"


def generate_scenario(
    scenario: str,
    config: DGPConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    """Return (observed_records, evaluator_only_truth, censor probabilities).

    Each randomized block contains one episode assigned to each safe policy. The
    latent reference and candidate event times are generated for every episode,
    but only the outcome under the assigned policy enters the observed artifact.
    Endpoint loss occurs before the horizon endpoint is ascertained; it is *not*
    silently converted into an event-free outcome.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    cfg = config or DGPConfig()
    cfg.validate()
    rng = random.Random(cfg.seed + list(SCENARIOS).index(scenario) * 10_007)
    censor = dict(SCENARIOS[scenario])
    hazards = dict(cfg.reference_hazard)
    observed: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []

    for block_index in range(cfg.blocks):
        stratum = "high" if block_index < cfg.blocks // 2 else "low"
        candidate_slot = rng.randrange(2)
        for slot in range(2):
            episode_id = f"{scenario}-b{block_index:05d}-e{slot}"
            assigned = "anticipatory_safe" if slot == candidate_slot else "reference_safe"
            # Common random numbers make the potential path comparison paired within episode.
            u_event = max(rng.random(), 1e-15)
            t_reference = -math.log(u_event) / hazards[stratum]
            intervention_applied = assigned == "anticipatory_safe" and stratum == "high"
            candidate_hazard = cfg.candidate_high_hazard if stratum == "high" else hazards[stratum]
            t_candidate = -math.log(u_event) / candidate_hazard
            t_competing = _exponential_time(rng, cfg.competing_hazard) if cfg.competing_hazard > 0 else math.inf
            y_reference = _terminal_outcome(t_reference, t_competing, cfg.horizon)
            y_candidate = _terminal_outcome(t_candidate, t_competing, cfg.horizon)
            g = censor[stratum]
            complete = rng.random() < g
            missing_reason = None
            if not complete:
                missing_reason = "administrative_endpoint_not_obtained" if rng.random() < 0.5 else "telemetry_loss_before_horizon"
            assigned_outcome = y_candidate if assigned == "anticipatory_safe" else y_reference
            assigned_time = t_candidate if assigned == "anticipatory_safe" else t_reference
            observed_outcome = assigned_outcome if complete else None
            observed_event_time = (
                assigned_time if complete and assigned_outcome == "boundary_event"
                else t_competing if complete and assigned_outcome == "competing_event"
                else None
            )
            observed.append({
                "episode_id": episode_id,
                "block_id": f"{scenario}-b{block_index:05d}",
                "assignment_design": "randomized_blocked",
                "assignment_precedes_time_zero": True,
                "randomization_receipt_valid": True,
                "assignment_time": -1.0,
                "time_zero": 0.0,
                "assigned_policy": assigned,
                "pre_assignment_stratum": stratum,
                "intervention_applied": intervention_applied,
                "intervention_time": 0.0 if intervention_applied else None,
                "followup_complete_by_horizon": complete,
                "followup_status": "complete" if complete else missing_reason,
                "policy_outcome": observed_outcome,
                "policy_event_time_observed": observed_event_time,
                "horizon": cfg.horizon,
            })
            truth.append({
                "episode_id": episode_id,
                "block_id": f"{scenario}-b{block_index:05d}",
                "assigned_policy": assigned,
                "pre_assignment_stratum": stratum,
                "reference_event_time": t_reference,
                "candidate_event_time": t_candidate,
                "competing_event_time": None if math.isinf(t_competing) else t_competing,
                "reference_potential_outcome": y_reference,
                "candidate_potential_outcome": y_candidate,
                "intervention_applied_under_assigned_policy": intervention_applied,
                "endpoint_observed_probability": g,
                "followup_complete_by_horizon": complete,
                "assigned_policy_outcome": assigned_outcome,
            })

    validate_artifact_separation(observed, truth)
    return observed, truth, censor


def validate_artifact_separation(
    observed: Iterable[dict[str, Any]], truth: Iterable[dict[str, Any]]
) -> None:
    obs = list(observed)
    oracle = list(truth)
    if len(obs) != len(oracle):
        raise ValueError("observed and oracle artifact row counts differ")
    truth_ids = [r.get("episode_id") for r in oracle]
    obs_ids = [r.get("episode_id") for r in obs]
    if obs_ids != truth_ids or len(set(obs_ids)) != len(obs_ids):
        raise ValueError("episode identity/order mismatch or duplicate IDs")
    for row in obs:
        forbidden = FORBIDDEN_OBSERVED_KEYS.intersection(row)
        if forbidden:
            raise ValueError(f"observed artifact leaks oracle-only keys: {sorted(forbidden)}")
        if row["assigned_policy"] not in POLICIES:
            raise ValueError("unknown assigned policy")
        if row["assignment_design"] != "randomized_blocked":
            raise ValueError("v17 requires blocked randomized assignment")
        if row["assignment_time"] >= row["time_zero"] or not row["assignment_precedes_time_zero"]:
            raise ValueError("assignment must precede time zero")
        if not row["randomization_receipt_valid"]:
            raise ValueError("randomization receipt invalid")
        if row["followup_complete_by_horizon"] != (row["policy_outcome"] is not None):
            raise ValueError("incomplete endpoint must remain Unknown; complete endpoint needs an outcome")
        if row["policy_outcome"] not in {None, "boundary_event", "competing_event", "event_free"}:
            raise ValueError("unknown terminal policy outcome")
        if row["policy_event_time_observed"] is not None and row["policy_outcome"] not in {"boundary_event", "competing_event"}:
            raise ValueError("event time can only accompany a terminal event")
    blocks: dict[str, list[dict[str, Any]]] = {}
    for row in obs:
        blocks.setdefault(row["block_id"], []).append(row)
    for block, rows in blocks.items():
        if len(rows) != 2 or {r["assigned_policy"] for r in rows} != set(POLICIES):
            raise ValueError(f"block {block} does not contain exactly one episode per arm")
        if len({r["pre_assignment_stratum"] for r in rows}) != 1:
            raise ValueError(f"block {block} is not matched on pre-assignment stratum")


def _risk(outcomes: Iterable[str | None]) -> tuple[float | None, int, int, int]:
    rows = list(outcomes)
    known = [x for x in rows if x is not None]
    if not known:
        return None, 0, 0, 0
    boundary = sum(x == "boundary_event" for x in known)
    competing = sum(x == "competing_event" for x in known)
    event_free = sum(x == "event_free" for x in known)
    return boundary / len(known), len(known), competing, event_free


def evaluate_scenario(
    scenario: str,
    observed: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    censor_probabilities: dict[str, float],
    config: DGPConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DGPConfig()
    cfg.validate()
    validate_artifact_separation(observed, truth)
    truth_by_id = {r["episode_id"]: r for r in truth}
    ref_obs = [r for r in observed if r["assigned_policy"] == "reference_safe"]
    cand_obs = [r for r in observed if r["assigned_policy"] == "anticipatory_safe"]
    ref_truth = [truth_by_id[r["episode_id"]] for r in ref_obs]
    cand_truth = [truth_by_id[r["episode_id"]] for r in cand_obs]

    true_ref_risk = sum(r["reference_potential_outcome"] == "boundary_event" for r in ref_truth) / len(ref_truth)
    true_candidate_risk = sum(r["candidate_potential_outcome"] == "boundary_event" for r in cand_truth) / len(cand_truth)
    complete_ref = [r for r in ref_obs if r["followup_complete_by_horizon"]]
    cc_risk, cc_n, cc_comp, cc_free = _risk(r["policy_outcome"] for r in complete_ref)
    min_support = min(censor_probabilities.values())
    if min_support < cfg.positivity_floor:
        ipw = {"estimable": False, "reason": "positivity_failure_or_below_predeclared_support_floor", "minimum_observation_probability": min_support}
    else:
        weighted_events = sum(
            (1.0 if r["policy_outcome"] == "boundary_event" else 0.0)
            / censor_probabilities[r["pre_assignment_stratum"]]
            for r in complete_ref
        )
        ipw_risk = weighted_events / len(ref_obs)
        ipw = {
            "estimable": True,
            "risk": ipw_risk,
            "n_assigned_reference": len(ref_obs),
            "n_endpoint_observed": len(complete_ref),
            "minimum_observation_probability": min_support,
            "diagnostic": "oracle-known observation probabilities; benchmark diagnostic only, not an operational estimator",
        }

    # These quantities are oracle truth for the assigned arms, available only to the evaluator.
    true_rd = true_candidate_risk - true_ref_risk
    candidate_complete = [r for r in cand_obs if r["followup_complete_by_horizon"]]
    cand_cc_risk, cand_cc_n, cand_cc_comp, cand_cc_free = _risk(r["policy_outcome"] for r in candidate_complete)
    unadjusted_itt_admissible = all(r["followup_complete_by_horizon"] for r in observed)
    randomized_itt = None
    if unadjusted_itt_admissible:
        r0, n0, comp0, free0 = _risk(r["policy_outcome"] for r in ref_obs)
        r1, n1, comp1, free1 = _risk(r["policy_outcome"] for r in cand_obs)
        randomized_itt = {
            "estimable": True,
            "control_boundary_risk": r0,
            "candidate_boundary_risk": r1,
            "candidate_minus_control_risk_difference": r1 - r0,
            "control_n": n0,
            "candidate_n": n1,
            "control_competing_n": comp0,
            "candidate_competing_n": comp1,
            "uncertainty_interval": None,
            "note": "Synthetic complete-follow-up point estimate; no inferential interval or population claim.",
        }
    else:
        randomized_itt = {
            "estimable": False,
            "reason": "incomplete_fixed_horizon_outcomes_require_separately_qualified_missingness_or_censoring_method",
            "assigned_denominator_preserved": len(observed),
            "missing_outcomes": sum(not r["followup_complete_by_horizon"] for r in observed),
        }

    # Naive as-treated comparison: receipt is selected by pre-assignment risk stratum.
    received = [r for r in observed if r["intervention_applied"] and r["policy_outcome"] is not None]
    not_received = [r for r in observed if not r["intervention_applied"] and r["policy_outcome"] is not None]
    received_risk, received_n, received_comp, _ = _risk(r["policy_outcome"] for r in received)
    not_received_risk, not_received_n, not_received_comp, _ = _risk(r["policy_outcome"] for r in not_received)
    as_treated = {
        "received_intervention_boundary_risk": received_risk,
        "received_intervention_n": received_n,
        "no_intervention_received_boundary_risk": not_received_risk,
        "no_intervention_received_n": not_received_n,
        "naive_received_minus_not_received_risk_difference": (
            received_risk - not_received_risk
            if received_risk is not None and not_received_risk is not None else None
        ),
        "is_randomized_itt": False,
        "selection_note": "Receipt is conditioned on high pre-assignment risk; this contrast is not the randomized policy effect.",
    }

    return {
        "scenario": scenario,
        "seed": cfg.seed + list(SCENARIOS).index(scenario) * 10_007,
        "blocks": cfg.blocks,
        "episodes_assigned": len(observed),
        "horizon": cfg.horizon,
        "censor_probabilities_by_pre_assignment_stratum": censor_probabilities,
        "reference_policy_forecast_lane": {
            "oracle_reference_arm_boundary_risk": true_ref_risk,
            "complete_case_boundary_risk": cc_risk,
            "complete_case_n": cc_n,
            "complete_case_competing_n": cc_comp,
            "complete_case_event_free_n": cc_free,
            "complete_case_absolute_error": abs(cc_risk - true_ref_risk) if cc_risk is not None else None,
            "oracle_ipw_diagnostic": ipw,
            "oracle_ipw_absolute_error": abs(ipw["risk"] - true_ref_risk) if ipw["estimable"] else None,
        },
        "policy_value_lane": {
            "oracle_reference_arm_boundary_risk": true_ref_risk,
            "oracle_candidate_arm_boundary_risk": true_candidate_risk,
            "oracle_candidate_minus_reference_risk_difference": true_rd,
            "randomized_fixed_horizon_itt": randomized_itt,
            "candidate_complete_case_boundary_risk": cand_cc_risk,
            "candidate_complete_case_n": cand_cc_n,
            "candidate_complete_case_competing_n": cand_cc_comp,
            "naive_as_treated": as_treated,
            "individual_counterfactuals_identified_from_observed_episode": False,
        },
        "evidence_accounting": {
            "reference_arm_assigned_n": len(ref_obs),
            "candidate_arm_assigned_n": len(cand_obs),
            "complete_followup_n": sum(r["followup_complete_by_horizon"] for r in observed),
            "unobserved_endpoint_n": sum(not r["followup_complete_by_horizon"] for r in observed),
            "administrative_endpoint_not_obtained_n": sum(r["followup_status"] == "administrative_endpoint_not_obtained" for r in observed),
            "telemetry_loss_before_horizon_n": sum(r["followup_status"] == "telemetry_loss_before_horizon" for r in observed),
            "oracle_reference_competing_n": sum(r["reference_potential_outcome"] == "competing_event" for r in ref_truth),
            "oracle_candidate_competing_n": sum(r["candidate_potential_outcome"] == "competing_event" for r in cand_truth),
            "assignment_precedes_time_zero": all(r["assignment_precedes_time_zero"] for r in observed),
            "valid_randomization_receipts": all(r["randomization_receipt_valid"] for r in observed),
            "exactly_one_episode_per_policy_per_block": True,
        },
        "claim_boundary": "Synthetic known-truth benchmark only; no survival estimator, causal estimator, SOTA, physical safety, real-time, simulator, HIL, or production-control claim.",
    }


def run_all(config: DGPConfig | None = None) -> tuple[dict[str, Any], dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]]:
    cfg = config or DGPConfig()
    cfg.validate()
    reports: dict[str, Any] = {}
    data: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for name in SCENARIOS:
        observed, truth, censor = generate_scenario(name, cfg)
        reports[name] = evaluate_scenario(name, observed, truth, censor, cfg)
        data[name] = (observed, truth)
    return {
        "benchmark": "MCR² v17 known-truth intervention and endpoint-ascertainment DGP",
        "version": "0.1",
        "synthetic_only": True,
        "configuration": {
            "seed": cfg.seed,
            "blocks_per_scenario": cfg.blocks,
            "episodes_per_scenario": cfg.blocks * 2,
            "horizon": cfg.horizon,
            "reference_hazard": dict(cfg.reference_hazard),
            "candidate_high_hazard": cfg.candidate_high_hazard,
            "competing_hazard": cfg.competing_hazard,
            "positivity_floor": cfg.positivity_floor,
        },
        "scenarios": reports,
        "interpretation": {
            "observed_vs_oracle": "Observed records contain only the assigned policy outcome. Potential outcomes and reference event times remain in a separate evaluator-only oracle artifact.",
            "ipw_scope": "Known-probability fixed-horizon endpoint diagnostic only; not a survival estimator or a replacement for #9690.",
            "positivity": "A zero or sub-floor observation probability blocks point identification for the target population; no extrapolation is produced.",
            "causal_scope": "Randomized assignment supports an arm-level ITT estimand when the complete fixed-horizon outcomes are observed; it never identifies each episode's individual counterfactual from its observed record.",
            "competing_events": "Competing terminal events are counted separately, not converted to event-free or independent-censor observations.",
        },
    }, data
