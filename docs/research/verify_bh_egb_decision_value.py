#!/usr/bin/env python3
"""Exact finite-state control separating information gain from decision value."""
import argparse
import copy
import json
import sys
from fractions import Fraction as F
from pathlib import Path


def same(actual, expected, label):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def evaluate(p):
    states = p["states"]
    ids = [s["state_id"] for s in states]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("state identifiers must be non-empty and unique")
    prior = {s["state_id"]: F(s["probability"]) for s in states}
    classes = {s["state_id"]: s["decision_class"] for s in states}
    if any(v <= 0 for v in prior.values()) or sum(prior.values(), F(0)) != 1:
        raise ValueError("state probabilities must be positive and sum to one")
    decision_classes = set(classes.values())
    if len(decision_classes) < 2:
        raise ValueError("at least two decision classes are required")
    if p["loss"] != {"kind": "zero_one_decision_classification", "loss_for_wrong_decision": "1", "loss_for_correct_decision": "0"}:
        raise ValueError("unsupported or altered loss contract")

    class_mass = {c: sum((prior[s] for s in ids if classes[s] == c), F(0)) for c in decision_classes}
    pre_risk = F(1) - max(class_mass.values())
    same(pre_risk, F(p["expected"]["pre_action_bayes_risk"]), "pre-action Bayes risk")

    results = []
    for action in p["actions"]:
        mapping = action["outcome_by_state"]
        if set(mapping) != set(ids):
            raise ValueError(f"{action['action_id']} must define exactly one outcome per state")
        cost = F(action["cost"])
        if cost < 0:
            raise ValueError("action cost must be non-negative")
        outcome_mass = {}
        joint = {}
        for state_id, outcome in mapping.items():
            outcome_mass[outcome] = outcome_mass.get(outcome, F(0)) + prior[state_id]
            key = (outcome, classes[state_id])
            joint[key] = joint.get(key, F(0)) + prior[state_id]
        if len(outcome_mass) != 2 or any(mass != F(1, 2) for mass in outcome_mass.values()):
            raise ValueError("fixture requires two equiprobable outcomes for its exact one-bit control")

        # With deterministic action outputs and two equiprobable outcomes,
        # I(State; Outcome) = H(Outcome) = exactly one bit.
        information_gain_bits = F(1)
        expected_posterior_risk = F(0)
        for outcome, mass in outcome_mass.items():
            posterior_class_mass = {
                c: joint.get((outcome, c), F(0)) / mass for c in decision_classes
            }
            conditional_risk = F(1) - max(posterior_class_mass.values())
            expected_posterior_risk += mass * conditional_risk

        gross_value = pre_risk - expected_posterior_risk
        net_value = gross_value - cost
        expected = action["expected"]
        same(information_gain_bits, F(expected["information_gain_bits"]), f"{action['action_id']} information gain")
        same(expected_posterior_risk, F(expected["expected_posterior_bayes_risk"]), f"{action['action_id']} posterior risk")
        same(gross_value, F(expected["gross_decision_value"]), f"{action['action_id']} gross decision value")
        same(net_value, F(expected["net_value_of_computation"]), f"{action['action_id']} net VoC")
        results.append({
            "action_id": action["action_id"],
            "information_gain_bits": information_gain_bits,
            "expected_posterior_bayes_risk": expected_posterior_risk,
            "gross_decision_value": gross_value,
            "cost": cost,
            "net_value_of_computation": net_value,
        })

    if len(results) != 2:
        raise ValueError("this known-answer fixture expects exactly two actions")
    same(results[0]["information_gain_bits"], results[1]["information_gain_bits"], "equal information gain")
    best = max(results, key=lambda r: (r["net_value_of_computation"], r["action_id"]))
    same(best["action_id"], p["expected"]["best_action"], "selected action")
    same(best["net_value_of_computation"], F(p["expected"]["best_net_value_of_computation"]), "best net VoC")
    same(results[1]["gross_decision_value"] - results[0]["gross_decision_value"], F(p["expected"]["decision_value_gap_between_actions"]), "decision value gap")

    rejected = []
    def must_reject(candidate, label):
        try:
            evaluate(candidate)
        except (ValueError, AssertionError, KeyError, TypeError):
            rejected.append(label)
            return
        raise AssertionError(f"hostile mutation accepted: {label}")

    bad = copy.deepcopy(p)
    bad["actions"][0]["cost"] = "0"
    must_reject(bad, "action_cost_mutation")
    bad = copy.deepcopy(p)
    bad["states"][0]["probability"] = "1/2"
    must_reject(bad, "prior_mutation")
    bad = copy.deepcopy(p)
    bad["states"][1]["decision_class"] = "B"
    must_reject(bad, "decision_class_mutation")
    bad = copy.deepcopy(p)
    bad["actions"][1]["outcome_by_state"] = copy.deepcopy(bad["actions"][0]["outcome_by_state"])
    must_reject(bad, "outcome_mapping_mutation")

    return {
        "status": "PASS_EXACT_FINITE_DECISION_VALUE_CONTROL_ONLY",
        "pre_action_bayes_risk": str(pre_risk),
        "action_results": [{k: str(v) if isinstance(v, F) else v for k, v in row.items()} for row in results],
        "equal_information_gain_bits": "1",
        "selected_action": best["action_id"],
        "decision_value_gap": str(F(p["expected"]["decision_value_gap_between_actions"])),
        "negative_controls_passed": len(rejected),
        "negative_controls": rejected,
        "claim_ceiling": p["claim_ceiling"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", nargs="?", type=Path, default=Path(__file__).with_name("BH_EGB_DECISION_VALUE_CONTROL_V0_1.json"))
    args = parser.parse_args()
    try:
        data = json.loads(args.fixture.read_text(encoding="utf-8"))
        report = evaluate(data)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
