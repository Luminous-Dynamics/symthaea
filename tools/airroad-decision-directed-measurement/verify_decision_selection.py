#!/usr/bin/env python3
"""Exact-rational verifier for AIRROAD's decision-directed action discriminator."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-decision-directed-measurement-v0.1.json"


def q(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def margin_lower(data: dict[str, Any], upper_bounds: dict[str, Fraction]) -> Fraction:
    result = q(data["margin"]["constant"])
    coefficients = data["margin"]["negative_coefficients"]
    assert set(coefficients) == set(upper_bounds)
    for coordinate_id, coefficient in coefficients.items():
        result -= q(coefficient) * upper_bounds[coordinate_id]
    return result


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.decision_directed_measurement.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_hard_set_policy_discriminator"
    assert data["uncertainty_semantics"] == "deterministic_hard_set"

    initial = data["initial_uncertainty"]
    coordinate_ids = list(initial)
    assert coordinate_ids == ["x", "y"]
    lower = {key: q(value["lower"]) for key, value in initial.items()}
    upper = {key: q(value["upper"]) for key, value in initial.items()}
    assert all(lower[key] <= upper[key] for key in coordinate_ids)

    initial_margin = margin_lower(data, upper)
    assert initial_margin == Fraction(-7, 10)
    assert initial_margin == q(data["expected"]["initial_margin_lower"])

    widths = {key: upper[key] - lower[key] for key in coordinate_ids}
    for key in coordinate_ids:
        assert widths[key] == q(data["expected"]["widths"][key])

    actions = data["actions"]
    assert len(actions) == 2
    budget = q(data["risk_budget"])
    results: dict[str, dict[str, Fraction]] = {}

    for action in actions:
        action_id = action["action_id"]
        assert action["action_kind"] == "synthetic_guaranteed_contraction"
        assert action["update_basis"] == "stipulated_synthetic_bound"
        cost = q(action["cost"])
        risk = q(action["risk"])
        assert cost > 0
        assert Fraction(0) <= risk <= budget

        updated = dict(upper)
        updates = action["upper_updates"]
        assert set(updates) <= set(coordinate_ids)
        assert action["target_coordinate_id"] in updates
        for coordinate_id, encoded_upper in updates.items():
            new_upper = q(encoded_upper)
            assert lower[coordinate_id] <= new_upper <= upper[coordinate_id]
            updated[coordinate_id] = new_upper

        new_margin = margin_lower(data, updated)
        improvement = new_margin - initial_margin
        results[action_id] = {
            "margin_lower": new_margin,
            "margin_improvement": improvement,
            "decision_value_per_cost": improvement / cost,
        }

        expected = data["expected"]["action_results"][action_id]
        assert new_margin == q(expected["margin_lower"])
        assert improvement == q(expected["margin_improvement"])

    # Width-only selection uses the largest original coordinate width.
    widest_coordinate = max(coordinate_ids, key=lambda key: widths[key])
    width_action = next(
        action["action_id"]
        for action in actions
        if action["target_coordinate_id"] == widest_coordinate
    )
    assert width_action == data["expected"]["width_only_action"] == "refine-x"

    # Decision selection maximizes exact worst-case margin improvement per cost.
    # Python's max preserves the first item on a tie; the input action order is frozen.
    decision_action = max(
        (action["action_id"] for action in actions),
        key=lambda action_id: results[action_id]["decision_value_per_cost"],
    )
    assert decision_action == data["expected"]["decision_value_action"] == "calibrate-y"
    selected_margin = results[decision_action]["margin_lower"]
    assert selected_margin == q(data["expected"]["selected_action_margin_lower"])
    assert (selected_margin > 0) is data["expected"]["selected_action_margin_is_positive"]

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls == {
        "unproved-hard-set-contraction": "reject-no-guaranteed-progress",
        "statistical-covariance-as-hard-set": "reject-claim-class",
        "stale-solver-witness": "reject-action-ranking",
        "unauthorized-physical-excitation": "reject-action",
        "incomplete-outcome-partition": "not-established",
    }

    assert data["claim_ceiling"] == [
        "Synthetic exact-rational two-coordinate decision-value discriminator only.",
        "Action contractions are stipulated and are not observed or physically validated measurements.",
        "No IQC solve, actual excitation, FRF qualification, aircraft stability or certification is established.",
    ]

    print("AIRROAD decision-directed measurement exact-rational reference: PASS")
    print(f"initial robust-margin lower bound: {initial_margin}")
    print(f"largest-width policy: {width_action} (width={widths[widest_coordinate]})")
    print(
        "decision-value policy: "
        f"{decision_action} (gain={results[decision_action]['margin_improvement']}, "
        f"new lower margin={selected_margin})"
    )
    print("statistical and physical-excitation claims: not evaluated by this fixture")


if __name__ == "__main__":
    main()
