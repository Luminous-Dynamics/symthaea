#!/usr/bin/env python3
"""Finite-action own-design power ceiling and shared-nuisance sequence oracle."""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_CEILING, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-adaptive-falsification-budget-v0.1.json"
getcontext().prec = 60

Interval = tuple[Fraction, Fraction]


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def as_decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def bernoulli_kl(p: Fraction, q_value: Fraction) -> Decimal:
    if not 0 <= p <= 1 or not 0 < q_value < 1:
        raise ValueError("Bernoulli probabilities are outside supported domain")
    if p == q_value:
        return Decimal(0)
    p_d = as_decimal(p)
    q_d = as_decimal(q_value)
    complement_p = Decimal(1) - p_d
    complement_q = Decimal(1) - q_d
    first = Decimal(0) if p == 0 else p_d * (p_d / q_d).ln()
    second = (
        Decimal(0)
        if p == 1
        else complement_p * (complement_p / complement_q).ln()
    )
    return first + second


def design_divergence(
    statistical: dict[str, Any],
    proportions: dict[str, dict[str, str]],
) -> Decimal:
    truth = {
        action: q(parameters["success_probability"])
        for action, parameters in statistical["truth"].items()
    }
    nulls = statistical["null_parameters"]
    divergences: list[Decimal] = []
    for null_parameters in nulls.values():
        total = Decimal(0)
        for action in statistical["action_ids"]:
            weight = q(proportions[action])
            null_probability = q(
                null_parameters[action]["success_probability"]
            )
            total += as_decimal(weight) * bernoulli_kl(
                truth[action],
                null_probability,
            )
        divergences.append(total)
    if not divergences:
        raise ValueError("working null family cannot be empty")
    return min(divergences)


def close_decimal(left: Decimal, right: Decimal, tolerance: Decimal) -> bool:
    return abs(left - right) <= tolerance


def interval(lower: dict[str, str], upper: dict[str, str]) -> Interval:
    lo, hi = q(lower), q(upper)
    if lo > hi:
        raise ValueError("interval lower endpoint exceeds upper endpoint")
    return lo, hi


def intersect(left: Interval, right: Interval) -> Interval | None:
    lo = max(left[0], right[0])
    hi = min(left[1], right[1])
    return (lo, hi) if lo <= hi else None


def feasible_shared_calibration(
    sequence: dict[str, Any],
    candidate_name: str,
    action_subset: list[str],
) -> Interval | None:
    prior = interval(
        sequence["calibration"]["lower"],
        sequence["calibration"]["upper"],
    )
    feasible: Interval | None = prior
    candidate = sequence["candidate_models"][candidate_name]
    for action in action_subset:
        observed = q(sequence["observation_values"][action])
        predicted = q(candidate["predictions"][action])
        error = interval(
            sequence["measurement_error"]["lower"],
            sequence["measurement_error"]["upper"],
        )
        # y = prediction + one shared xi + eta; solve exactly for xi.
        required_xi = (
            observed - predicted - error[1],
            observed - predicted - error[0],
        )
        if feasible is None:
            return None
        feasible = intersect(feasible, required_xi)
    return feasible


def feasible_with_action_local_calibration(
    sequence: dict[str, Any],
    candidate_name: str,
    action_subset: list[str],
) -> bool:
    # Deliberately incorrect counterfactual: a fresh nuisance coordinate per action.
    for action in action_subset:
        if feasible_shared_calibration(sequence, candidate_name, [action]) is None:
            return False
    return True


def validate_no_power_promotion(requested_claim: str) -> None:
    forbidden = {
        "GUARANTEED_POWER",
        "ACHIEVED_POWER",
        "VALID_ANYTIME_TEST",
        "MODEL_ADEQUACY_CERTIFICATE",
    }
    if requested_claim in forbidden:
        raise ValueError("a necessary power ceiling is not an achieved detector guarantee")


def validate_sequence_claim(
    surviving_models: list[str],
    requested_claim: str,
) -> None:
    if not surviving_models and requested_claim == "ALTERNATIVE_MODEL_PROVEN_TRUE":
        raise ValueError("rejecting the entire represented class proves no alternative")
    if surviving_models and requested_claim == "MODEL_CLASS_FALSIFIED":
        raise ValueError("a nonempty candidate intersection is not a falsification")


def run_negative_controls(data: dict[str, Any]) -> None:
    statistical = data["statistical"]
    designs = {item["id"]: item["proportions"] for item in statistical["designs"]}
    assert design_divergence(statistical, designs["A_only"]) == 0
    assert design_divergence(statistical, designs["B_only"]) == 0

    try:
        validate_no_power_promotion("GUARANTEED_POWER")
    except ValueError:
        pass
    else:
        raise AssertionError("positive own-design ceiling was promoted to guaranteed power")

    try:
        validate_no_power_promotion("ACHIEVED_POWER")
    except ValueError:
        pass
    else:
        raise AssertionError("necessary sample-size condition was promoted to achieved power")

    sequence = data["deterministic_sequence"]
    actions = sequence["action_ids"]
    candidates = list(sequence["candidate_models"])
    for action in actions:
        single_survivors = [
            name for name in candidates
            if feasible_shared_calibration(sequence, name, [action]) is not None
        ]
        assert single_survivors == ["H1", "H2"]

    joint_survivors = [
        name for name in candidates
        if feasible_shared_calibration(sequence, name, actions) is not None
    ]
    assert joint_survivors == []

    local_reset_survivors = [
        name for name in candidates
        if feasible_with_action_local_calibration(sequence, name, actions)
    ]
    assert local_reset_survivors == ["H1", "H2"]

    try:
        validate_sequence_claim(["H1"], "MODEL_CLASS_FALSIFIED")
    except ValueError:
        pass
    else:
        raise AssertionError("one surviving model was ignored")

    try:
        validate_sequence_claim([], "ALTERNATIVE_MODEL_PROVEN_TRUE")
    except ValueError:
        pass
    else:
        raise AssertionError("an alternative model was proven by rejection of the listed family")


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.adaptive_falsification_budget.v0_1"
    assert data["schema_version"] == "0.1.0"

    statistical = data["statistical"]
    designs = {item["id"]: item["proportions"] for item in statistical["designs"]}
    d_a = design_divergence(statistical, designs["A_only"])
    d_b = design_divergence(statistical, designs["B_only"])
    d_balanced = design_divergence(statistical, designs["balanced"])

    alpha = q(statistical["test_target"]["alpha"])
    target_power = q(statistical["test_target"]["power"])
    beta = q(statistical["test_target"]["beta"])
    assert target_power + beta == 1

    threshold = bernoulli_kl(target_power, alpha)
    required_horizon = threshold / d_balanced
    minimum_integer_horizon = int(
        required_horizon.to_integral_value(rounding=ROUND_CEILING)
    )
    t4_product = Decimal(4) * d_balanced
    t5_product = Decimal(5) * d_balanced

    expected = statistical["expected"]
    tolerance = Decimal("1e-50")
    assert d_a == as_decimal(q(expected["D_A_only"])) == 0
    assert d_b == as_decimal(q(expected["D_B_only"])) == 0
    assert close_decimal(
        d_balanced,
        Decimal(expected["D_balanced_decimal"]),
        tolerance,
    )
    assert close_decimal(
        threshold,
        Decimal(expected["binary_kl_power_vs_alpha_decimal"]),
        tolerance,
    )
    assert close_decimal(
        t4_product,
        Decimal(expected["T4_times_D_decimal"]),
        tolerance,
    )
    assert close_decimal(
        t5_product,
        Decimal(expected["T5_times_D_decimal"]),
        tolerance,
    )
    assert close_decimal(
        required_horizon,
        Decimal(expected["necessary_horizon_lower_bound_decimal"]),
        tolerance,
    )
    assert minimum_integer_horizon == expected["first_integer_horizon_not_ruled_out"] == 5
    assert t4_product < threshold
    assert t5_product >= threshold
    assert expected["T4_fails_necessary_condition"] is True
    assert expected["T5_meets_necessary_condition_not_guarantee"] is True

    sequence = data["deterministic_sequence"]
    require_hard = sequence["measurement_error"]["semantics"] == "deterministic_hard_exact"
    require_shared = (
        sequence["calibration"]["scope"] == "same_single_coordinate_across_A_and_B"
    )
    assert require_hard and require_shared
    assert sequence["calibration"]["nuisance_id"] != ""

    actions = sequence["action_ids"]
    candidates = list(sequence["candidate_models"])
    expected_sequence = sequence["expected"]
    for action in actions:
        survivors = [
            name for name in candidates
            if feasible_shared_calibration(sequence, name, [action]) is not None
        ]
        assert survivors == expected_sequence[f"action_{action}_candidates"]

    joint_shared = [
        name for name in candidates
        if feasible_shared_calibration(sequence, name, actions) is not None
    ]
    joint_local = [
        name for name in candidates
        if feasible_with_action_local_calibration(sequence, name, actions)
    ]
    assert joint_shared == expected_sequence["joint_shared_calibration_candidates"]
    assert joint_local == (
        expected_sequence["joint_action_local_calibration_counterfactual_candidates"]
    )
    assert expected_sequence["each_single_action_leaves_both_candidates"] is True
    assert expected_sequence["joint_shared_sequence_falsifies_all_candidates"] is True

    run_negative_controls(data)

    print("AIRROAD adaptive falsification budget reference: PASS")
    print(f"D(A-only) = {d_a}")
    print(f"D(B-only) = {d_b}")
    print(f"D(balanced) = 3/10*ln(4) = {d_balanced}")
    print(f"kl(0.8 || 0.05) = {threshold}")
    print(f"necessary horizon = {required_horizon}; integer lower threshold = {minimum_integer_horizon}")
    print("T=4 violates the necessary power condition; T=5 does not violate it but is not a guarantee")
    print(f"single-action survivors: A={expected_sequence['action_A_candidates']}, B={expected_sequence['action_B_candidates']}")
    print(f"joint shared-calibration survivors: {joint_shared}")
    print(f"incorrect action-local calibration survivors: {joint_local}")
    print("power-claim and shared-calibration negative controls: PASS")


if __name__ == "__main__":
    main()
