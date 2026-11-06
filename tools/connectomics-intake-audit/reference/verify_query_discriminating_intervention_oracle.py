#!/usr/bin/env python3
"""Independent standard-library oracle for query-discriminating intervention fixtures.

Recomputes predictive entropy, expected posterior query loss, nuisance-robust
decision value per cost, and entropy-greedy selection from the raw fixture.
This is a contract oracle, not a biological intervention model.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def entropy(probabilities: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probabilities if p > 0.0)


def baseline_bayes_risk(
    worlds: list[dict[str, Any]], query: dict[str, Any]
) -> float:
    p_true = sum(w["prior_probability"] for w in worlds if w["query_decision"])
    p_false = 1.0 - p_true
    loss_if_false = query["loss_false_true"]
    loss_if_true = query["loss_true_false"]
    return min(p_true * loss_if_false, p_false * loss_if_true)


def evaluate_action(
    action_id: str,
    worlds: list[dict[str, Any]],
    nuisance_ids: list[str],
    prediction_index: dict[tuple[str, str, str], dict[str, float]],
    query: dict[str, Any],
) -> tuple[float, float]:
    """Return (worst-case expected posterior query loss, worst-case EIG)."""
    risk_by_nuisance: list[float] = []
    eig_by_nuisance: list[float] = []

    for nuisance_id in nuisance_ids:
        outcome_ids = sorted(
            {
                outcome
                for world in worlds
                for outcome in prediction_index[(action_id, world["world_id"], nuisance_id)]
            }
        )
        prior_predictive: dict[str, float] = {}
        conditional_entropy = 0.0

        for world in worlds:
            distribution = prediction_index[(action_id, world["world_id"], nuisance_id)]
            total = sum(distribution.values())
            require(abs(total - 1.0) <= 1e-9, f"{action_id}/{world['world_id']}/{nuisance_id}: outcome probabilities do not sum to 1")
            conditional_entropy += world["prior_probability"] * entropy(list(distribution.values()))
            for outcome, probability in distribution.items():
                prior_predictive[outcome] = prior_predictive.get(outcome, 0.0) + world["prior_probability"] * probability

        eig = entropy([prior_predictive.get(outcome, 0.0) for outcome in outcome_ids]) - conditional_entropy
        eig_by_nuisance.append(max(0.0, eig))

        expected_posterior_loss = 0.0
        for outcome in outcome_ids:
            outcome_mass = prior_predictive.get(outcome, 0.0)
            if outcome_mass <= 0.0:
                continue
            posterior_true = 0.0
            posterior_false = 0.0
            for world in worlds:
                likelihood = prediction_index[(action_id, world["world_id"], nuisance_id)].get(outcome, 0.0)
                mass = world["prior_probability"] * likelihood
                if world["query_decision"]:
                    posterior_true += mass
                else:
                    posterior_false += mass
            posterior_true /= outcome_mass
            posterior_false /= outcome_mass
            posterior_loss = min(
                posterior_true * query["loss_false_true"],
                posterior_false * query["loss_true_false"],
            )
            expected_posterior_loss += outcome_mass * posterior_loss
        risk_by_nuisance.append(expected_posterior_loss)

    return max(risk_by_nuisance), min(eig_by_nuisance)


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(data["schema_version"] == 1, "unsupported schema version")
    require(data["claim_scope"] == "synthetic_known_truth", "this starter oracle only accepts the synthetic known-truth fixture")
    require(data["dataset_binding"]["evidence_tier"] == "SYNTHETIC", "synthetic scope must use SYNTHETIC evidence tier")
    require(valid_sha(data["dataset_binding"]["dataset_revision_sha256"]), "dataset digest malformed")
    require(valid_sha(data["dataset_binding"]["source_access_receipt_sha256"]), "source-access receipt malformed")
    require(valid_sha(data["query"]["query_spec_sha256"]), "query digest malformed")
    require(valid_sha(data["query"]["decision_rule_sha256"]), "query decision-rule digest malformed")

    worlds = sorted(data["worlds"], key=lambda row: row["world_id"])
    interventions = sorted(data["interventions"], key=lambda row: row["intervention_id"])
    nuisance_models = sorted(data["nuisance_models"], key=lambda row: row["nuisance_model_id"])
    require(len(worlds) >= 2, "at least two plausible worlds required")
    require(len({w["world_id"] for w in worlds}) == len(worlds), "duplicate world IDs")
    require(len({a["intervention_id"] for a in interventions}) == len(interventions), "duplicate intervention IDs")
    require(len({m["nuisance_model_id"] for m in nuisance_models}) == len(nuisance_models), "duplicate nuisance IDs")

    prior_total = sum(w["prior_probability"] for w in worlds)
    require(abs(prior_total - 1.0) <= 1e-9, "world priors must sum to one")
    require(all(w["biological_disposition"] == "ADMISSIBLE" for w in worlds), "only independently admissible worlds may enter the prior")
    for row in worlds:
        require(valid_sha(row["graph_sha256"]), "world graph digest malformed")
        require(valid_sha(row["source_evidence_sha256"]), "world source digest malformed")

    nuisance_ids = [m["nuisance_model_id"] for m in nuisance_models]
    intervention_ids = [a["intervention_id"] for a in interventions]
    world_ids = [w["world_id"] for w in worlds]
    nuisance_map = {m["nuisance_model_id"]: m for m in nuisance_models}
    intervention_map = {a["intervention_id"]: a for a in interventions}

    for model in nuisance_models:
        require(valid_sha(model["model_sha256"]), "nuisance-model digest malformed")
        require(valid_sha(model["observation_model_sha256"]), "observation-model digest malformed")
    for action in interventions:
        require(valid_sha(action["feasibility_receipt_sha256"]), "feasibility receipt malformed")
        require(valid_sha(action["safety_profile_sha256"]), "safety-profile digest malformed")
        require(action["cost_units"] > 0, "intervention cost must be positive")

    expected_keys = {
        (action_id, world_id, nuisance_id)
        for action_id in intervention_ids
        for world_id in world_ids
        for nuisance_id in nuisance_ids
    }
    prediction_index: dict[tuple[str, str, str], dict[str, float]] = {}
    trajectory_digests: set[str] = set()
    for row in data["predictions"]:
        key = (row["intervention_id"], row["world_id"], row["nuisance_model_id"])
        require(key in expected_keys, f"prediction has unknown reference: {key}")
        require(key not in prediction_index, f"duplicate prediction record: {key}")
        require(valid_sha(row["trajectory_sha256"]), "trajectory digest malformed")
        trajectory_digests.add(row["trajectory_sha256"])
        outcomes: dict[str, float] = {}
        for outcome in row["observations"]:
            require(outcome["outcome_id"] not in outcomes, f"duplicate outcome ID in {key}")
            outcomes[outcome["outcome_id"]] = outcome["probability"]
        require(abs(sum(outcomes.values()) - 1.0) <= 1e-9, f"outcome probabilities do not sum to one: {key}")
        prediction_index[key] = outcomes
    require(set(prediction_index) == expected_keys, "prediction matrix is not a complete intervention × world × nuisance Cartesian product")

    split = data["split_manifest"]
    require(valid_sha(split["split_sha256"]), "split digest malformed")
    train = set(split["train_units"])
    validation = set(split["validation_units"])
    lockbox = set(split["lockbox_units"])
    require(train and validation and lockbox, "train, validation and lockbox groups must all be non-empty")
    require(not (train & validation or train & lockbox or validation & lockbox), "split leakage across independent units")

    p_true = sum(w["prior_probability"] for w in worlds if w["query_decision"])
    prior_decision = p_true >= 0.5
    require(prior_decision == data["query"]["baseline_decision"], "declared baseline decision is not the Bayes decision under the frozen prior")
    baseline_risk = baseline_bayes_risk(worlds, data["query"])
    require(data["selection_objective"] == "WORST_CASE_EXPECTED_POSTERIOR_0_1_LOSS_REDUCTION_PER_COST", "unsupported selection objective")

    computed: dict[str, dict[str, float]] = {}
    for action_id in intervention_ids:
        risk, eig = evaluate_action(action_id, worlds, nuisance_ids, prediction_index, data["query"])
        action = intervention_map[action_id]
        value_per_cost = max(0.0, baseline_risk - risk) / action["cost_units"]
        computed[action_id] = {
            "worst_case_expected_posterior_query_loss": risk,
            "worst_case_eig_bits": eig,
            "decision_value_per_cost": value_per_cost,
        }

    feasible_ids = sorted(
        action_id for action_id in intervention_ids
        if intervention_map[action_id]["feasibility_status"] == "FEASIBLE"
    )
    require(feasible_ids, "no feasible intervention candidates")
    expected_selected = sorted(
        feasible_ids,
        key=lambda action_id: (-computed[action_id]["decision_value_per_cost"], action_id),
    )[0]
    expected_entropy_greedy = sorted(
        feasible_ids,
        key=lambda action_id: (-computed[action_id]["worst_case_eig_bits"], action_id),
    )[0]

    oracle = data["oracle_expectations"]
    require(abs(oracle["baseline_expected_query_loss"] - baseline_risk) <= 1e-9, "baseline expected query loss mismatch")
    for action_id in intervention_ids:
        for field in (
            "worst_case_expected_posterior_query_loss",
            "worst_case_eig_bits",
            "decision_value_per_cost",
        ):
            require(action_id in oracle[field], f"oracle expectation missing {field}/{action_id}")
            require(abs(oracle[field][action_id] - computed[action_id][field]) <= 1e-8, f"oracle mismatch for {field}/{action_id}")
    require(expected_selected == oracle["expected_selected_intervention_id"], "robust decision objective selected the wrong intervention")
    require(expected_entropy_greedy == oracle["expected_entropy_greedy_intervention_id"], "entropy-greedy baseline selected the wrong intervention")
    require(intervention_map[expected_selected]["feasibility_status"] == "FEASIBLE", "selected intervention is not feasible")
    require(expected_selected != "infeasible-perfect", "disallowed perfect intervention was selected")
    require(computed[expected_selected]["worst_case_expected_posterior_query_loss"] < baseline_risk, "selected action does not reduce worst-case query loss")

    return {
        "baseline_expected_query_loss": baseline_risk,
        "selected_intervention_id": expected_selected,
        "entropy_greedy_intervention_id": expected_entropy_greedy,
        "per_intervention": computed,
        "world_count": len(worlds),
        "nuisance_model_count": len(nuisance_ids),
        "intervention_count": len(intervention_ids),
        "prediction_record_count": len(prediction_index),
        "claim_scope": data["claim_scope"],
        "status": "SYNTHETIC_CONTRACT_VALIDATED",
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_query_discriminating_intervention_oracle.py FIXTURE.json", file=sys.stderr)
        return 2
    try:
        raw = Path(sys.argv[1]).read_bytes()
        data = json.loads(raw)
        report = verify(data)
        print(json.dumps(report, indent=2, sort_keys=True))
        print("independent query-discriminating intervention oracle: PASS")
        return 0
    except Exception as exc:
        print(f"query-discriminating intervention oracle: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
