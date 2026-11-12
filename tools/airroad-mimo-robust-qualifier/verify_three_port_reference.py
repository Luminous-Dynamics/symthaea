#!/usr/bin/env python3
"""Independent exact-rational oracle for AIRROAD's three-port robust reference."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-three-port-robust-v0.1.json"
BOUNDARY = Fraction(1, 8)


def q(record: dict[str, str]) -> Fraction:
    value = Fraction(int(record["numerator"]), int(record["denominator"]))
    return value


def expected_disposition(lower: Fraction, upper: Fraction) -> str:
    if lower <= 0 or lower > upper:
        raise ValueError("gamma interval must be positive and ordered")
    if upper < BOUNDARY:
        return "ROBUSTLY_STABLE"
    if lower > BOUNDARY:
        return "ROBUSTLY_UNSTABLE"
    return "NOT_ESTABLISHED"


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_three_port_robust_reference.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_reference"

    model = data["model"]
    matrix = model["transfer_matrix"]["numerator_matrix"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert model["transfer_matrix"]["port_ids"] == ["bus-a", "bus-b", "bus-c"]
    assert model["transfer_matrix"]["common_denominator"] == "s + 1"
    assert model["transfer_matrix"]["feedback_convention"] == "negative_unity_feedback"

    # All diagonal SISO projections are identically zero.
    assert [matrix[i][i] for i in range(3)] == [
        {"numerator": "0", "denominator": "1"}
    ] * 3

    # Preserve the complete directed cycle and its parameter dependence.
    assert matrix[0][1] == {"numerator": "2", "denominator": "1"}
    assert matrix[1][2] == {"numerator": "2", "denominator": "1"}
    assert matrix[2][0] == {"expression": "-2*gamma"}

    # det((s+1)I + M(gamma)) = (s+1)^3 - 8 gamma.
    # Expand with exact coefficients: s^3 + 3s^2 + 3s + 1 - 8 gamma.
    assert model["return_ratio_determinant"] == "((s + 1)^3 - 8*gamma) / (s + 1)^3"
    assert model["closed_loop_characteristic_coefficients_descending"] == [
        "1", "3", "3", "1 - 8*gamma"
    ]
    assert model["open_loop_poles"] == [
        {"real": "-1", "imag": "0", "multiplicity": 3}
    ]
    assert model["exact_stability_boundary"] == {"numerator": "1", "denominator": "8"}

    # Exact set-level classification; no frequency sweep and no floating roots.
    for case in data["uncertainty_sets"]:
        lower = q(case["gamma_lower"])
        upper = q(case["gamma_upper"])
        actual = expected_disposition(lower, upper)
        assert actual == case["expected"], (case["id"], actual, case["expected"])
        assert lower > 0 and upper >= lower

        # The cubic Routh-Hurwitz conditions reduce to gamma < 1/8.
        # For gamma > 1/8, (s+1)^3 = 8 gamma has one positive real root.
        if actual == "ROBUSTLY_STABLE":
            assert upper < BOUNDARY
        elif actual == "ROBUSTLY_UNSTABLE":
            assert lower > BOUNDARY
        else:
            assert lower <= BOUNDARY <= upper

        print(f"PASS {case['id']}: [{lower}, {upper}] -> {actual}")

    # Invalid intervals must be rejected by the oracle's admission rule.
    for lower, upper in (
        (Fraction(0, 1), Fraction(1, 16)),
        (Fraction(1, 4), Fraction(1, 8)),
    ):
        try:
            expected_disposition(lower, upper)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid interval accepted: [{lower}, {upper}]")

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["diagonal-siso-only"] == "must_not_claim_network_stability"
    assert controls["uncertainty-set-touches-boundary"] == "NOT_ESTABLISHED"
    assert controls["uncertainty-set-crosses-boundary"] == "NOT_ESTABLISHED"
    assert controls["stable-set-labelled-unstable"] == "reject"
    assert controls["invalid-non-positive-lower-bound"] == "reject"
    assert controls["reversed-interval"] == "reject"

    print("PASS all six negative-control declarations")
    print("PASS exact three-port robust-reference oracle")


if __name__ == "__main__":
    main()
