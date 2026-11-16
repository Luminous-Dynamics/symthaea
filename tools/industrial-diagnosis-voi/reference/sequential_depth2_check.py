#!/usr/bin/env python3
"""Independent standard-library reference for the frozen depth-two VoI case.

This does not invoke or import the Rust implementation. It independently
enumerates every first measurement and each branch's admissible second-test
choices for the synthetic binary-outcome pump/valve fixture.
"""

from __future__ import annotations

import json
from typing import Any

HYPOTHESES = (
    "DANGEROUS_PROCESS_DEGRADATION",
    "METROLOGY_CALIBRATION_DRIFT",
    "NORMAL_OR_INNOCUOUS_VARIATION",
)
PRIOR = (0.10, 0.30, 0.60)
DECISIONS = ("CONTINUE", "STOP_AND_QUARANTINE", "RECALIBRATE_AND_RETEST")
LOSS = (
    (100.0, 12.0, 60.0),
    (20.0, 12.0, 2.0),
    (0.0, 12.0, 8.0),
)
EXISTING_ORIGINS = {"LIVE-PRESSURE-TAG-07"}
MEASUREMENTS = (
    {
        "id": "CALIBRATION_REFERENCE_ARTIFACT",
        "cost": 1.0,
        "likelihood": (0.10, 0.95, 0.05),
        "origin": "INDEPENDENT-CAL-ARTIFACT",
        "calibration": True,
        "applicability": True,
        "authority": True,
    },
    {
        "id": "INDEPENDENT_PROCESS_SAFETY_TEST",
        "cost": 2.0,
        "likelihood": (0.90, 0.15, 0.10),
        "origin": "INDEPENDENT-PROCESS-TEST",
        "calibration": True,
        "applicability": True,
        "authority": True,
    },
    {
        "id": "GENERAL_VIBRATION_SCREEN",
        "cost": 0.5,
        "likelihood": (0.55, 0.65, 0.35),
        "origin": "INDEPENDENT-VIBRATION-SENSOR",
        "calibration": True,
        "applicability": True,
        "authority": True,
    },
)
BUDGET = 4.0
EPSILON = 1.0e-9


def min_decision(posterior: tuple[float, ...]) -> tuple[str, float]:
    candidates = [
        (sum(posterior[i] * LOSS[i][j] for i in range(len(HYPOTHESES))), DECISIONS[j])
        for j in range(len(DECISIONS))
    ]
    value, decision = min(candidates, key=lambda item: (item[0], item[1]))
    return decision, value


def outcome_posterior(
    prior: tuple[float, ...],
    likelihood: tuple[float, ...],
    positive: bool,
) -> tuple[float, tuple[float, ...] | None]:
    factors = likelihood if positive else tuple(1.0 - value for value in likelihood)
    probability = sum(prior[i] * factors[i] for i in range(len(prior)))
    if probability <= 0.0:
        return probability, None
    return probability, tuple(prior[i] * factors[i] / probability for i in range(len(prior)))


def test_value(
    prior: tuple[float, ...],
    measurement: dict[str, Any],
) -> tuple[float, float, dict[str, Any]]:
    p_positive, positive_posterior = outcome_posterior(prior, measurement["likelihood"], True)
    p_negative, negative_posterior = outcome_posterior(prior, measurement["likelihood"], False)
    positive = min_decision(positive_posterior)[1] if positive_posterior is not None else 0.0
    negative = min_decision(negative_posterior)[1] if negative_posterior is not None else 0.0
    expected_terminal_loss = p_positive * positive + p_negative * negative
    current_loss = min_decision(prior)[1]
    net_value = current_loss - expected_terminal_loss - measurement["cost"]
    return expected_terminal_loss, net_value, {
        "positive_probability": p_positive,
        "negative_probability": p_negative,
        "positive_posterior": positive_posterior,
        "negative_posterior": negative_posterior,
    }


def admissible(measurement: dict[str, Any], used_origin: str | None) -> bool:
    return (
        measurement["cost"] >= 0.0
        and measurement["calibration"]
        and measurement["applicability"]
        and measurement["authority"]
        and measurement["origin"] not in EXISTING_ORIGINS
        and measurement["origin"] != used_origin
    )


def main() -> None:
    baseline_decision, baseline_loss = min_decision(PRIOR)

    greedy_candidates = []
    for measurement in MEASUREMENTS:
        if not admissible(measurement, None) or measurement["cost"] > BUDGET:
            continue
        terminal_loss, net_value, _ = test_value(PRIOR, measurement)
        if net_value > EPSILON:
            greedy_candidates.append((net_value, measurement["id"], terminal_loss, measurement["cost"]))

    if greedy_candidates:
        _, greedy_id, greedy_terminal, greedy_cost = sorted(
            greedy_candidates, key=lambda item: (-item[0], item[1])
        )[0]
        greedy_total = greedy_terminal + greedy_cost
    else:
        greedy_id = "NO_TEST"
        greedy_terminal = baseline_loss
        greedy_cost = 0.0
        greedy_total = baseline_loss

    policies = []
    for first in MEASUREMENTS:
        if not admissible(first, None) or first["cost"] > BUDGET:
            continue
        _, _, first_outcomes = test_value(PRIOR, first)
        branches = []
        expected_terminal = 0.0
        expected_cost = first["cost"]

        for outcome_name, probability, posterior in (
            ("POSITIVE", first_outcomes["positive_probability"], first_outcomes["positive_posterior"]),
            ("NEGATIVE", first_outcomes["negative_probability"], first_outcomes["negative_posterior"]),
        ):
            if probability <= 0.0 or posterior is None:
                continue

            stop_decision, current_loss = min_decision(posterior)
            remaining_budget = BUDGET - first["cost"]
            second_candidates = []
            for second in MEASUREMENTS:
                if second["id"] == first["id"] or not admissible(second, first["origin"]):
                    continue
                if second["cost"] > remaining_budget + EPSILON:
                    continue
                second_terminal, second_net, second_outcomes = test_value(posterior, second)
                if second_net > EPSILON:
                    second_candidates.append((
                        second_net,
                        second["id"],
                        second_terminal,
                        second["cost"],
                        second_outcomes,
                    ))

            if second_candidates:
                _, second_id, terminal_loss, second_cost, second_outcomes = sorted(
                    second_candidates, key=lambda item: (-item[0], item[1])
                )[0]
                positive_decision_loss = (
                    min_decision(second_outcomes["positive_posterior"])[1]
                    if second_outcomes["positive_posterior"] is not None
                    else None
                )
                negative_decision_loss = (
                    min_decision(second_outcomes["negative_posterior"])[1]
                    if second_outcomes["negative_posterior"] is not None
                    else None
                )
                positive_decision = (
                    min_decision(second_outcomes["positive_posterior"])[0]
                    if second_outcomes["positive_posterior"] is not None
                    else None
                )
                negative_decision = (
                    min_decision(second_outcomes["negative_posterior"])[0]
                    if second_outcomes["negative_posterior"] is not None
                    else None
                )
                second_positive_probability = second_outcomes["positive_probability"]
                second_negative_probability = second_outcomes["negative_probability"]
                positive_leaf = (
                    first["cost"] + second_cost + positive_decision_loss
                    if positive_decision_loss is not None else None
                )
                negative_leaf = (
                    first["cost"] + second_cost + negative_decision_loss
                    if negative_decision_loss is not None else None
                )
                leaf_losses = [loss for loss in (positive_leaf, negative_leaf) if loss is not None]
                worst_case_leaf = max(leaf_losses)
                stop = None
            else:
                second_id = "STOP"
                terminal_loss = current_loss
                second_cost = 0.0
                positive_decision = None
                negative_decision = None
                second_positive_probability = None
                second_negative_probability = None
                positive_leaf = None
                negative_leaf = None
                worst_case_leaf = first["cost"] + current_loss
                stop = stop_decision

            branch_total = first["cost"] + terminal_loss + second_cost
            expected_terminal += probability * terminal_loss
            expected_cost += probability * second_cost
            branches.append({
                "first_outcome": outcome_name,
                "probability": probability,
                "second_measurement": second_id,
                "stop_decision": stop,
                "second_positive_decision": positive_decision,
                "second_negative_decision": negative_decision,
                "second_positive_probability": second_positive_probability,
                "second_negative_probability": second_negative_probability,
                "second_positive_leaf_total_loss": positive_leaf,
                "second_negative_leaf_total_loss": negative_leaf,
                "terminal_loss": terminal_loss,
                "second_cost": second_cost,
                "branch_total_loss": branch_total,
                "worst_case_leaf_total_loss": worst_case_leaf,
            })

        total = expected_terminal + expected_cost
        if baseline_loss - total > EPSILON:
            policies.append({
                "first_measurement": first["id"],
                "branches": branches,
                "expected_terminal_loss": expected_terminal,
                "expected_measurement_cost": expected_cost,
                "expected_total_loss": total,
                "worst_case_leaf_total_loss": max(branch["worst_case_leaf_total_loss"] for branch in branches),
                "worst_case_regret_vs_no_test": max(0.0, max(branch["worst_case_leaf_total_loss"] for branch in branches) - baseline_loss),
                "improvement_over_no_test": baseline_loss - total,
            })

    if not policies:
        raise RuntimeError("reference fixture did not produce a positive-value adaptive policy")
    adaptive = sorted(
        policies,
        key=lambda policy: (policy["expected_total_loss"], policy["first_measurement"]),
    )[0]

    result = {
        "reference": "PUMP-VALVE-VOI-SYNTHETIC-001",
        "policy_model": "exhaustive_binary_outcome_depth_2_conditional_independence",
        "budget_units": BUDGET,
        "no_test_decision": baseline_decision,
        "no_test_total_loss": baseline_loss,
        "greedy_one_step": {
            "first_measurement": greedy_id,
            "expected_terminal_loss": greedy_terminal,
            "expected_measurement_cost": greedy_cost,
            "expected_total_loss": greedy_total,
        },
        "adaptive_depth_two": adaptive,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
