#!/usr/bin/env python3
"""Exact finite-horizon oracle for static-model robust connectome experiments.

This oracle exhaustively enumerates one shared non-anticipative policy tree.
The static objective keeps one model vertex fixed for the entire trajectory.
The rectangular relaxation allows the second-stage kernel vertex to vary by
observed first-stage history. It does not call any production planner.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

DECISIONS = ("SUPPORT", "REFUTE")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return abs(left - right) <= tolerance


def unique(values: list[str], label: str) -> None:
    require(all(isinstance(value, str) and value.strip() for value in values), f"{label} contains an empty identity")
    require(len(values) == len(set(values)), f"{label} contains duplicate identities")


def probabilities(rows: list[dict[str, Any]], label: str) -> dict[str, float]:
    ids = [row.get("outcome_id", row.get("world_id")) for row in rows]
    unique(ids, f"{label} IDs")
    result = {row.get("outcome_id", row.get("world_id")): row["probability"] for row in rows}
    require(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in result.values()), f"{label} probability outside [0,1]")
    require(close(sum(result.values()), 1.0), f"{label} probabilities do not sum to one")
    return result


def validate_input(data: dict[str, Any]) -> dict[str, Any]:
    require(data["schema_version"] == 1, "unsupported schema version")
    require(data["claim_scope"] == "synthetic_known_truth", "V1 oracle only supports synthetic known-truth fixtures")
    query = data["query"]
    require(valid_sha(query["query_spec_sha256"]), "query specification digest malformed")
    require(valid_sha(query["decision_rule_sha256"]), "decision-rule digest malformed")
    require(0.0 <= query["certificate_max_expected_loss"] <= 1.0, "certificate tolerance outside [0,1]")

    worlds_rows = data["worlds"]
    actions_rows = data["actions"]
    models_rows = data["model_vertices"]
    world_ids = [row["world_id"] for row in worlds_rows]
    action_ids = [row["action_id"] for row in actions_rows]
    model_ids = [row["model_vertex_id"] for row in models_rows]
    unique(world_ids, "world")
    unique(action_ids, "action")
    unique(model_ids, "model vertex")
    worlds = {row["world_id"]: row for row in worlds_rows}
    actions = {row["action_id"]: row for row in actions_rows}
    require(len(worlds) == 2, "starter V1 fixture requires exactly two worlds")
    require(all(row["query_decision"] in (True, False) for row in worlds_rows), "query decisions must be Boolean")
    for world in worlds_rows:
        require(valid_sha(world["source_evidence_sha256"]), "world evidence digest malformed")

    for action in actions_rows:
        require(action["feasibility_status"] in {"FEASIBLE", "DISALLOWED", "INACCESSIBLE", "NOT_ASSESSED"}, "unknown action feasibility")
        require(valid_sha(action["feasibility_receipt_sha256"]), "action feasibility receipt malformed")
        require(math.isfinite(action["stage_one_cost_units"]) and action["stage_one_cost_units"] > 0, "invalid stage-one cost")
        if action["feasibility_status"] != "FEASIBLE":
            require(action["action_id"] != data["policy_space"]["root_action_id"], "infeasible root action")
        if action["one_shot_eligible"]:
            require(action["stage_one_available"], "one-shot action is not available at stage one")
    policy_space = data["policy_space"]
    root_id = policy_space["root_action_id"]
    require(root_id in actions and actions[root_id]["stage_one_available"], "root action unavailable at stage one")
    histories = policy_space["first_stage_outcome_ids"]
    unique(histories, "first-stage history")
    require(len(histories) == 2, "V1 starter contract requires two first-stage histories")
    second_outcomes = policy_space["second_stage_outcome_ids"]
    unique(second_outcomes, "second-stage outcome")
    require(len(second_outcomes) == 2, "V1 starter contract requires two second-stage outcomes")

    history_options_rows = policy_space["second_stage_options_by_first_outcome"]
    unique([row["first_outcome_id"] for row in history_options_rows], "history option")
    options_by_history = {}
    for row in history_options_rows:
        history = row["first_outcome_id"]
        require(history in histories, f"unknown first-stage history {history}")
        option_ids = [option["action_id"] for option in row["options"]]
        unique(option_ids, f"{history} second-stage action")
        require(len(option_ids) >= 1, f"{history} has no second-stage actions")
        for option in row["options"]:
            require(option["action_id"] in actions, f"unknown second-stage action {option['action_id']}")
            require(actions[option["action_id"]]["feasibility_status"] == "FEASIBLE", "infeasible action entered policy option set")
            require(math.isfinite(option["cost_units"]) and option["cost_units"] > 0, "invalid second-stage cost")
        options_by_history[history] = {option["action_id"]: option["cost_units"] for option in row["options"]}
    require(set(options_by_history) == set(histories), "second-stage option history coverage is incomplete")

    models = {}
    global_root_histories = set()
    for model in models_rows:
        model_id = model["model_vertex_id"]
        require(valid_sha(model["model_sha256"]), f"{model_id} model digest malformed")
        unique(model["dependence_group_ids"], f"{model_id} dependence groups")
        prior = probabilities(model["prior"], f"{model_id} prior")
        require(set(prior) == set(worlds), f"{model_id} prior does not cover the exact world set")
        require(len(model["stage_one_kernels"]) == len(actions) * len(worlds), f"{model_id} stage-one kernel matrix has wrong size")
        stage_one = {}
        for kernel in model["stage_one_kernels"]:
            key = (kernel["action_id"], kernel["world_id"])
            require(key[0] in actions and key[1] in worlds, f"{model_id} stage-one kernel has unknown reference {key}")
            require(key not in stage_one, f"{model_id} duplicate stage-one kernel {key}")
            stage_one[key] = probabilities(kernel["outcomes"], f"{model_id} stage-one {key}")
        expected_stage_one = {(action_id, world_id) for action_id in actions for world_id in worlds}
        require(set(stage_one) == expected_stage_one, f"{model_id} stage-one kernel matrix is incomplete")
        observed_root_histories = {
            outcome_id
            for world_id in worlds
            for outcome_id in stage_one[(root_id, world_id)]
        }
        require(
            observed_root_histories.issubset(set(histories)),
            f"{model_id} root-action emits an undeclared history category",
        )
        global_root_histories.update(observed_root_histories)

        stage_two = {}
        expected_stage_two = {
            (history, action_id, world_id)
            for history in histories
            for action_id in options_by_history[history]
            for world_id in worlds
        }
        for kernel in model["stage_two_kernels"]:
            key = (kernel["first_outcome_id"], kernel["action_id"], kernel["world_id"])
            require(key in expected_stage_two, f"{model_id} stage-two kernel has unknown reference {key}")
            require(key not in stage_two, f"{model_id} duplicate stage-two kernel {key}")
            stage_two[key] = probabilities(kernel["outcomes"], f"{model_id} stage-two {key}")
        require(set(stage_two) == expected_stage_two, f"{model_id} stage-two kernel matrix is incomplete")
        observed_second_outcomes = {
            outcome_id
            for distribution in stage_two.values()
            for outcome_id in distribution
        }
        require(
            observed_second_outcomes == set(second_outcomes),
            f"{model_id} stage-two outcomes do not match the frozen outcome alphabet",
        )
        models[model_id] = {"row": model, "prior": prior, "stage_one": stage_one, "stage_two": stage_two}

    require(
        global_root_histories == set(histories),
        "union of model-vertex root-action outcomes does not match the frozen history alphabet",
    )

    split = data["split_manifest"]
    require(valid_sha(split["split_sha256"]), "split digest malformed")
    train, validation, lockbox = set(split["train_units"]), set(split["validation_units"]), set(split["lockbox_units"])
    require(train and validation and lockbox, "train, validation and lockbox must be non-empty")
    require(not (train & validation or train & lockbox or validation & lockbox), "split leakage")
    return {
        "worlds": worlds,
        "actions": actions,
        "models": models,
        "root_id": root_id,
        "histories": histories,
        "second_outcomes": second_outcomes,
        "options_by_history": options_by_history,
        "history_options_rows": history_options_rows,
    }


def risk_for_static_model(
    model: dict[str, Any],
    validated: dict[str, Any],
    action_map: dict[str, str],
    decision_map: dict[str, str],
) -> float:
    total = 0.0
    for world_id, world in validated["worlds"].items():
        p_world = model["prior"][world_id]
        first_distribution = model["stage_one"][(validated["root_id"], world_id)]
        for first_outcome, p_first in first_distribution.items():
            second_action = action_map[first_outcome]
            second_distribution = model["stage_two"][(first_outcome, second_action, world_id)]
            for second_outcome, p_second in second_distribution.items():
                decision = decision_map[f"{first_outcome}|{second_outcome}"]
                wrong = (decision == "SUPPORT") != world["query_decision"]
                total += p_world * p_first * p_second * float(wrong)
    return total


def rectangular_risk_for_policy(
    root_model: dict[str, Any],
    validated: dict[str, Any],
    action_map: dict[str, str],
    decision_map: dict[str, str],
) -> float:
    total = 0.0
    for first_outcome in validated["histories"]:
        posterior_mass = 0.0
        world_joint = {}
        for world_id, world in validated["worlds"].items():
            mass = root_model["prior"][world_id] * root_model["stage_one"][(validated["root_id"], world_id)].get(first_outcome, 0.0)
            world_joint[world_id] = mass
            posterior_mass += mass
        if posterior_mass <= 0.0:
            continue
        action_id = action_map[first_outcome]
        worst_conditional = 0.0
        for second_model in validated["models"].values():
            conditional_loss = 0.0
            for world_id, world in validated["worlds"].items():
                p_world_given_history = world_joint[world_id] / posterior_mass
                distribution = second_model["stage_two"][(first_outcome, action_id, world_id)]
                for second_outcome, p_second in distribution.items():
                    decision = decision_map[f"{first_outcome}|{second_outcome}"]
                    wrong = (decision == "SUPPORT") != world["query_decision"]
                    conditional_loss += p_world_given_history * p_second * float(wrong)
            worst_conditional = max(worst_conditional, conditional_loss)
        total += posterior_mass * worst_conditional
    return total


def static_policy_search(validated: dict[str, Any]) -> dict[str, Any]:
    histories = validated["histories"]
    options = validated["options_by_history"]
    models = list(validated["models"].values())
    decisions_keys = [f"{history}|{outcome}" for history in histories for outcome in validated["second_outcomes"]]
    candidates = []
    sorted_actions = {history: sorted(options[history]) for history in histories}
    for action_choices in itertools.product(*(sorted_actions[history] for history in histories)):
        action_map = dict(zip(histories, action_choices))
        worst_branch_cost = validated["actions"][validated["root_id"]]["stage_one_cost_units"] + max(
            options[history][action_map[history]] for history in histories
        )
        if worst_branch_cost > validated["max_cost"] + 1e-12:
            continue
        for decision_choices in itertools.product(DECISIONS, repeat=len(decisions_keys)):
            decision_map = dict(zip(decisions_keys, decision_choices))
            per_model = {
                model["row"]["model_vertex_id"]: risk_for_static_model(model, validated, action_map, decision_map)
                for model in models
            }
            robust_risk = max(per_model.values())
            decision_order = tuple(DECISIONS.index(decision_map[key]) for key in decisions_keys)
            action_order = tuple(action_map[history] for history in histories)
            key = (robust_risk, worst_branch_cost, action_order, decision_order)
            candidates.append((key, action_map, decision_map, per_model))
    require(candidates, "no feasible static policy trees")
    key, action_map, decision_map, per_model = min(candidates, key=lambda item: item[0])
    return {
        "robust_risk": key[0],
        "worst_branch_cost": key[1],
        "action_map": action_map,
        "decision_map": decision_map,
        "per_model_risks": per_model,
    }


def rectangular_policy_search(validated: dict[str, Any]) -> dict[str, Any]:
    histories = validated["histories"]
    options = validated["options_by_history"]
    models = list(validated["models"].values())
    decisions_keys = [f"{history}|{outcome}" for history in histories for outcome in validated["second_outcomes"]]
    candidates = []
    sorted_actions = {history: sorted(options[history]) for history in histories}
    for action_choices in itertools.product(*(sorted_actions[history] for history in histories)):
        action_map = dict(zip(histories, action_choices))
        worst_branch_cost = validated["actions"][validated["root_id"]]["stage_one_cost_units"] + max(
            options[history][action_map[history]] for history in histories
        )
        if worst_branch_cost > validated["max_cost"] + 1e-12:
            continue
        for decision_choices in itertools.product(DECISIONS, repeat=len(decisions_keys)):
            decision_map = dict(zip(decisions_keys, decision_choices))
            per_root_model = {
                root_model["row"]["model_vertex_id"]: rectangular_risk_for_policy(
                    root_model, validated, action_map, decision_map
                )
                for root_model in models
            }
            robust_risk = max(per_root_model.values())
            decision_order = tuple(DECISIONS.index(decision_map[key]) for key in decisions_keys)
            action_order = tuple(action_map[history] for history in histories)
            key = (robust_risk, worst_branch_cost, action_order, decision_order)
            candidates.append((key, action_map, decision_map, per_root_model))
    require(candidates, "no feasible rectangular policy trees")
    key, action_map, decision_map, per_root_model = min(candidates, key=lambda item: item[0])
    return {
        "robust_risk": key[0],
        "worst_branch_cost": key[1],
        "action_map": action_map,
        "decision_map": decision_map,
        "per_root_model_risks": per_root_model,
    }


def one_shot_robust_loss(
    action_id: str,
    validated: dict[str, Any],
) -> tuple[float, dict[str, str]]:
    models = list(validated["models"].values())
    all_outcomes = sorted({
        outcome
        for model in models
        for world_id in validated["worlds"]
        for outcome in model["stage_one"][(action_id, world_id)]
    })
    candidates = []
    for decisions in itertools.product(DECISIONS, repeat=len(all_outcomes)):
        mapping = dict(zip(all_outcomes, decisions))
        per_model = {}
        for model in models:
            risk = 0.0
            for world_id, world in validated["worlds"].items():
                for outcome, probability in model["stage_one"][(action_id, world_id)].items():
                    wrong = (mapping[outcome] == "SUPPORT") != world["query_decision"]
                    risk += model["prior"][world_id] * probability * float(wrong)
            per_model[model["row"]["model_vertex_id"]] = risk
        candidates.append((max(per_model.values()), tuple(DECISIONS.index(mapping[outcome]) for outcome in all_outcomes), mapping))
    risk, _, mapping = min(candidates, key=lambda item: (item[0], item[1]))
    return risk, mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        raw = args.fixture.read_bytes()
        data = json.loads(raw)
        validated = validate_input(data)
        validated["max_cost"] = data["policy_space"]["max_worst_branch_cost_units"]
        static = static_policy_search(validated)
        rectangular = rectangular_policy_search(validated)
        rectangular_static_policy_risk = max(
            rectangular_risk_for_policy(model, validated, static["action_map"], static["decision_map"])
            for model in validated["models"].values()
        )
        one_shot = {
            action_id: one_shot_robust_loss(action_id, validated)[0]
            for action_id, action in validated["actions"].items()
            if action["one_shot_eligible"] and action["feasibility_status"] == "FEASIBLE"
        }
        best_one_shot = min(one_shot.values())
        tolerance = data["query"]["certificate_max_expected_loss"]
        static_certificate = static["robust_risk"] <= tolerance
        rectangular_certificate = rectangular["robust_risk"] <= tolerance
        expected = data["oracle_expectations"]

        checks = {
            "best_one_shot_robust_loss": best_one_shot,
            "static_robust_loss": static["robust_risk"],
            "rectangular_risk_of_static_policy": rectangular_static_policy_risk,
            "rectangular_minimax_loss": rectangular["robust_risk"],
            "static_rectangular_gap": rectangular["robust_risk"] - static["robust_risk"],
            "static_gain_over_one_shot": best_one_shot - static["robust_risk"],
            "static_worst_branch_cost": static["worst_branch_cost"],
            "rectangular_worst_branch_cost": rectangular["worst_branch_cost"],
            "static_certificate": static_certificate,
            "rectangular_certificate": rectangular_certificate,
            "expected_static_second_action_by_history": static["action_map"],
            "expected_static_terminal_decision_by_history": static["decision_map"],
            "expected_rectangular_terminal_decision_by_history": rectangular["decision_map"],
        }
        for field, actual in checks.items():
            wanted = expected[field]
            if isinstance(actual, dict):
                require(actual == wanted, f"known-answer mismatch for {field}: {actual!r}")
            elif isinstance(actual, bool):
                require(actual is wanted, f"known-answer mismatch for {field}: {actual!r}")
            else:
                require(close(actual, wanted), f"known-answer mismatch for {field}: actual={actual}, expected={wanted}")
        require(one_shot == expected["one_shot_robust_loss_by_action"], "one-shot risk mapping mismatch")
        require(static["robust_risk"] <= rectangular["robust_risk"] + 1e-12, "static risk must not exceed rectangular relaxation")
        require(rectangular_static_policy_risk >= static["robust_risk"] - 1e-12, "rectangular evaluation of static policy became better than static evaluation")
        require(static["action_map"]["CAL-A"] == "ACTION-A" and static["action_map"]["CAL-B"] == "ACTION-B", "calibration-driven action map not recovered")
        require(static["decision_map"]["CAL-A|POSITIVE"] == "SUPPORT", "static policy terminal mapping incorrect")

        report = {
            "schema_version": 1,
            "tool": "verify-static-model-sequential-connectome-policy",
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "run_id": data["run_id"],
            "claim_scope": data["claim_scope"],
            "world_count": len(validated["worlds"]),
            "model_vertex_count": len(validated["models"]),
            "action_count": len(validated["actions"]),
            "static_robust_risk": static["robust_risk"],
            "static_worst_branch_cost": static["worst_branch_cost"],
            "static_policy": {
                "root_action_id": validated["root_id"],
                "second_action_by_first_history": static["action_map"],
                "terminal_decision_by_history": static["decision_map"],
                "per_model_risk": static["per_model_risks"],
            },
            "rectangular_minimax_risk": rectangular["robust_risk"],
            "rectangular_worst_branch_cost": rectangular["worst_branch_cost"],
            "rectangular_policy": {
                "root_action_id": validated["root_id"],
                "second_action_by_first_history": rectangular["action_map"],
                "terminal_decision_by_history": rectangular["decision_map"],
                "per_root_model_risk": rectangular["per_root_model_risks"],
            },
            "rectangular_risk_of_static_optimal_policy": rectangular_static_policy_risk,
            "static_rectangular_gap": rectangular["robust_risk"] - static["robust_risk"],
            "one_shot_robust_risk_by_action": one_shot,
            "best_one_shot_robust_risk": best_one_shot,
            "static_gain_over_one_shot": best_one_shot - static["robust_risk"],
            "static_robust_certificate": static_certificate,
            "rectangular_robust_certificate": rectangular_certificate,
            "model_ancestry_overlaps": [
                {
                    "model_vertex_a": left["model_vertex_id"],
                    "model_vertex_b": right["model_vertex_id"],
                    "shared_group_ids": sorted(set(left["dependence_group_ids"]) & set(right["dependence_group_ids"])),
                }
                for index, left in enumerate(sorted(data["model_vertices"], key=lambda row: row["model_vertex_id"]))
                for right in sorted(data["model_vertices"], key=lambda row: row["model_vertex_id"])[index + 1:]
                if set(left["dependence_group_ids"]) & set(right["dependence_group_ids"])
            ],
            "model_independence_disposition": "NOT_ESTABLISHED_BY_ANCESTRY_LABELS",
            "status": "SYNTHETIC_STATIC_MODEL_SEQUENTIAL_POLICY_VALIDATED",
            "scope_note": "Exact finite policy-tree results are conditional on the supplied worlds and history-conditioned kernels. Rectangular model switching is a conservative relaxation, not a claim that biological mechanisms change between stages."
        }
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            args.report.write_text(rendered, encoding="utf-8")
            print(f"static-model sequential policy oracle: PASS; report_sha256={hashlib.sha256(rendered.encode()).hexdigest()}")
        else:
            print(rendered, end="")
            print("static-model sequential policy oracle: PASS")
        return 0
    except Exception as exc:
        print(f"static-model sequential policy oracle: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
