#!/usr/bin/env python3
"""Independent finite oracle for response-model adequacy challenges; standard-library only."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def unique(values: list[str], label: str) -> None:
    require(len(values) == len(set(values)), f"duplicate {label}")


def baseline_risk(worlds: list[dict[str, Any]], query: dict[str, Any]) -> tuple[float, bool]:
    p_true = sum(w["prior_probability"] for w in worlds if w["query_decision"])
    p_false = 1.0 - p_true
    true_decision_loss = p_false * query["loss_false_true"]
    false_decision_loss = p_true * query["loss_true_false"]
    decision = true_decision_loss <= false_decision_loss
    return min(true_decision_loss, false_decision_loss), decision


def model_action_risk(action_id: str, model_id: str, worlds: list[dict[str, Any]],
                      predictions: dict[tuple[str, str, str], dict[str, float]],
                      query: dict[str, Any]) -> float:
    outcome_ids: set[str] = set()
    for world in worlds:
        outcome_ids.update(predictions[(action_id, world["world_id"], model_id)])
    total = 0.0
    for outcome_id in outcome_ids:
        joint_true = 0.0
        joint_false = 0.0
        for world in worlds:
            joint = world["prior_probability"] * predictions[
                (action_id, world["world_id"], model_id)
            ].get(outcome_id, 0.0)
            if world["query_decision"]:
                joint_true += joint
            else:
                joint_false += joint
        total += min(
            joint_false * query["loss_false_true"],
            joint_true * query["loss_true_false"],
        )
    return total


def objective_for_envelope(model_ids: list[str], actions: list[dict[str, Any]],
                            worlds: list[dict[str, Any]],
                            predictions: dict[tuple[str, str, str], dict[str, float]],
                            query: dict[str, Any], prior_risk: float
                            ) -> tuple[dict[str, float], dict[str, float], str | None]:
    robust_loss: dict[str, float] = {}
    value_per_cost: dict[str, float] = {}
    for action in actions:
        action_id = action["intervention_id"]
        robust = max(model_action_risk(action_id, model_id, worlds, predictions, query)
                     for model_id in model_ids)
        robust_loss[action_id] = robust
        value_per_cost[action_id] = max(0.0, prior_risk - robust) / action["cost_units"]
    feasible = [a for a in actions if a["feasibility_status"] == "FEASIBLE"]
    ranked = sorted(feasible, key=lambda a: (-value_per_cost[a["intervention_id"]], a["intervention_id"]))
    if not ranked or value_per_cost[ranked[0]["intervention_id"]] <= 1e-12:
        return robust_loss, value_per_cost, None
    return robust_loss, value_per_cost, ranked[0]["intervention_id"]


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(data["schema_version"] == 1, "unsupported schema version")
    require(data["claim_scope"] == "synthetic_known_truth", "reference oracle only accepts synthetic fixture")
    require(data["dataset_binding"]["evidence_tier"] == "SYNTHETIC", "synthetic claim requires synthetic data tier")
    for field in ("revision_sha256", "source_access_receipt_sha256"):
        require(valid_sha(data["dataset_binding"][field]), f"malformed dataset digest: {field}")

    worlds, actions, models, challenges, query = (
        data["worlds"], data["interventions"], data["models"], data["challenges"], data["query"]
    )
    require(len(worlds) >= 2, "at least two worlds are required")
    unique([w["world_id"] for w in worlds], "world identity")
    unique([a["intervention_id"] for a in actions], "intervention identity")
    unique([m["model_id"] for m in models], "model identity")
    unique([c["challenge_id"] for c in challenges], "challenge identity")
    require(all(w["biological_disposition"] == "ADMISSIBLE" for w in worlds),
            "non-admissible world cannot enter probability envelope")
    require(all(math.isfinite(w["prior_probability"]) and 0 <= w["prior_probability"] <= 1 for w in worlds),
            "invalid world prior")
    require(abs(sum(w["prior_probability"] for w in worlds) - 1.0) <= 1e-12,
            "world priors must sum to one")
    for field in ("query_spec_sha256", "decision_rule_sha256"):
        require(valid_sha(query[field]), f"malformed query digest: {field}")
    require(query["loss_false_true"] >= 0 and query["loss_true_false"] >= 0, "negative query loss")

    prior_loss, prior_decision = baseline_risk(worlds, query)
    require(prior_decision == query["baseline_decision"], "declared baseline decision is not the Bayes decision")

    model_index = {m["model_id"]: m for m in models}
    require(data["baseline_model_id"] in model_index, "baseline model missing")
    require(model_index[data["baseline_model_id"]]["role"] == "BASELINE", "baseline model role mismatch")
    require(sum(m["role"] == "BASELINE" for m in models) == 1, "exactly one baseline model is required")
    challenge_index = {c["challenge_id"]: c for c in challenges}
    for model in models:
        for field in ("model_sha256", "observation_kernel_sha256"):
            require(valid_sha(model[field]), f"malformed {field} for {model['model_id']}")
        if model["role"] == "BASELINE":
            require(model["challenge_id"] is None, "baseline model cannot reference a challenge")
        else:
            require(model["challenge_id"] in challenge_index, "challenger has no challenge record")
            require(challenge_index[model["challenge_id"]]["model_id"] == model["model_id"],
                    "challenge/model identity mismatch")

    split = data["split_manifest"]
    model_fit, challenge_units, lockbox = (
        set(split["model_fit_units"]), set(split["challenge_units"]), set(split["policy_lockbox_units"])
    )
    require(model_fit and challenge_units and lockbox, "all split partitions must be non-empty")
    require(not (model_fit & challenge_units or model_fit & lockbox or challenge_units & lockbox),
            "model-fit/challenge/lockbox leakage")
    require(valid_sha(split["split_sha256"]), "malformed split digest")

    for challenge in challenges:
        require(challenge["disposition"] in {"ADMITTED", "REJECTED", "DISPUTED", "INDETERMINATE", "INACCESSIBLE"},
                "unknown challenge disposition")
        require(challenge["model_id"] is None or challenge["model_id"] in model_index,
                "challenge references an unknown model")
        require(set(challenge["challenge_unit_ids"]).issubset(challenge_units),
                "challenge unit is outside the frozen challenge partition")
        for field in ("raw_evidence_sha256", "challenge_report_sha256", "adjudication_receipt_sha256"):
            require(valid_sha(challenge[field]), f"malformed challenge digest: {field}")
        if challenge["disposition"] == "ADMITTED":
            require(challenge["model_id"] is not None, "admitted challenge needs a model")
        if challenge["disposition"] == "REJECTED":
            require(challenge["model_id"] is None or model_index[challenge["model_id"]]["role"] == "CHALLENGER",
                    "rejected challenge linkage is invalid")

    for action in actions:
        require(math.isfinite(action["cost_units"]) and action["cost_units"] > 0,
                "intervention cost must be finite and positive")
        require(action["feasibility_status"] in {"FEASIBLE", "DISALLOWED", "INACCESSIBLE", "NOT_ASSESSED"},
                "unknown intervention feasibility")
        for field in ("feasibility_receipt_sha256", "safety_profile_sha256"):
            require(valid_sha(action[field]), f"malformed intervention digest: {field}")

    expected = {(a["intervention_id"], w["world_id"], m["model_id"])
                for a in actions for w in worlds for m in models}
    prediction_index: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in data["predictions"]:
        key = (row["intervention_id"], row["world_id"], row["model_id"])
        require(key in expected, f"unknown prediction reference: {key}")
        require(key not in prediction_index, f"duplicate prediction record: {key}")
        outcomes: dict[str, float] = {}
        for outcome in row["outcomes"]:
            require(outcome["outcome_id"] not in outcomes, "duplicate outcome identity")
            p = outcome["probability"]
            require(math.isfinite(p) and 0 <= p <= 1, "outcome probability outside [0,1]")
            outcomes[outcome["outcome_id"]] = p
        require(abs(sum(outcomes.values()) - 1.0) <= 1e-12,
                f"outcome probabilities do not sum to one: {key}")
        prediction_index[key] = outcomes
    require(set(prediction_index) == expected, "prediction matrix is incomplete")

    base_ids = [data["baseline_model_id"]]
    admitted_ids = [
        m["model_id"] for m in models
        if m["role"] == "CHALLENGER"
        and challenge_index[m["challenge_id"]]["disposition"] == "ADMITTED"
    ]
    unresolved = sum(c["disposition"] in {"DISPUTED", "INDETERMINATE", "INACCESSIBLE"}
                     for c in challenges)
    base_losses, base_values, base_best = objective_for_envelope(
        base_ids, actions, worlds, prediction_index, query, prior_loss
    )
    expanded_ids = base_ids + admitted_ids
    expanded_losses, expanded_values, expanded_best = objective_for_envelope(
        expanded_ids, actions, worlds, prediction_index, query, prior_loss
    )
    for action in actions:
        action_id = action["intervention_id"]
        require(expanded_losses[action_id] + 1e-12 >= base_losses[action_id],
                "model expansion reduced robust posterior loss")
        require(expanded_values[action_id] <= base_values[action_id] + 1e-12,
                "model expansion increased value per cost")
    require(max(expanded_values.values()) <= max(base_values.values()) + 1e-12,
            "best robust value increased after model expansion")

    baseline_ancestry = set(model_index[data["baseline_model_id"]]["ancestry_group_ids"])
    ancestry_overlaps = {
        model_id: sorted(baseline_ancestry & set(model_index[model_id]["ancestry_group_ids"]))
        for model_id in admitted_ids
    }
    oracle = data["oracle_expectations"]
    require(abs(oracle["baseline_expected_query_loss"] - prior_loss) <= 1e-12,
            "baseline expected query loss mismatch")
    require(oracle["baseline_envelope_best_intervention_id"] == base_best,
            "baseline-envelope optimum mismatch")
    require(oracle["expanded_envelope_best_intervention_id"] == expanded_best,
            "expanded-envelope optimum mismatch")
    for action in actions:
        action_id = action["intervention_id"]
        require(abs(oracle["baseline_envelope_robust_loss"][action_id] - base_losses[action_id]) <= 1e-12,
                f"baseline risk mismatch: {action_id}")
        require(abs(oracle["expanded_envelope_robust_loss"][action_id] - expanded_losses[action_id]) <= 1e-12,
                f"expanded risk mismatch: {action_id}")

    status = "NOT_PROMOTABLE_UNRESOLVED_CHALLENGE" if unresolved else "SYNTHETIC_CONTRACT_VALIDATED"
    require(oracle["status"] == status, "oracle status mismatch")
    return {
        "baseline_expected_query_loss": prior_loss,
        "baseline_envelope_best_intervention_id": base_best,
        "expanded_envelope_best_intervention_id": expanded_best,
        "baseline_envelope_robust_loss": base_losses,
        "expanded_envelope_robust_loss": expanded_losses,
        "baseline_envelope_value_per_cost": base_values,
        "expanded_envelope_value_per_cost": expanded_values,
        "admitted_challenger_ids": admitted_ids,
        "ancestry_overlaps": ancestry_overlaps,
        "unresolved_challenge_count": unresolved,
        "status": status,
    }


def self_test() -> None:
    worlds = [
        {"query_decision": True, "prior_probability": 0.9},
        {"query_decision": False, "prior_probability": 0.1},
    ]
    loss, decision = baseline_risk(worlds, {"loss_false_true": 1.0, "loss_true_false": 1.0})
    require(abs(loss - 0.1) <= 1e-12 and decision is True, "binary Bayes-risk control failed")
    print("response-model adequacy oracle self-test: PASS")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        try:
            self_test()
            return 0
        except Exception as exc:
            print(f"oracle self-test: FAIL: {exc}", file=sys.stderr)
            return 1
    if len(sys.argv) != 2:
        print("usage: verify_response_model_adequacy_oracle.py FIXTURE.json | --self-test", file=sys.stderr)
        return 2
    try:
        data = json.loads(Path(sys.argv[1]).read_bytes())
        print(json.dumps(verify(data), indent=2, sort_keys=True))
        print("independent response-model adequacy oracle: PASS")
        return 0
    except Exception as exc:
        print(f"response-model adequacy oracle: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
