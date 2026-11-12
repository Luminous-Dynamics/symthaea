#!/usr/bin/env python3
"""Independent exact-rational oracle for AIRROAD's three-port robust reference."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-three-port-robust-v0.1.json"


def q(record: dict[str, str]) -> Fraction:
    return Fraction(int(record["numerator"]), int(record["denominator"]))


# Sparse bivariate polynomials: keys are (power of s, power of gamma).
Polynomial = dict[tuple[int, int], Fraction]


def clean(poly: Polynomial) -> Polynomial:
    return {key: value for key, value in poly.items() if value}


def add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, Fraction(0)) + value
    return clean(result)


def negate(poly: Polynomial) -> Polynomial:
    return {key: -value for key, value in poly.items()}


def multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for (s_left, g_left), a in left.items():
        for (s_right, g_right), b in right.items():
            key = (s_left + s_right, g_left + g_right)
            result[key] = result.get(key, Fraction(0)) + a * b
    return clean(result)


def parse_matrix_entry(entry: dict[str, str]) -> Polynomial:
    if "expression" in entry:
        if entry["expression"] == "-2*gamma":
            return {(0, 1): Fraction(-2)}
        raise ValueError(f"unsupported symbolic matrix entry: {entry['expression']}")
    value = q(entry)
    return {(0, 0): value} if value else {}


def determinant_3(matrix: list[list[Polynomial]]) -> Polynomial:
    # Determinant via all six permutations, independently of the expected
    # characteristic polynomial stored in the fixture.
    permutations = (
        ((0, 1, 2), 1),
        ((0, 2, 1), -1),
        ((1, 0, 2), -1),
        ((1, 2, 0), 1),
        ((2, 0, 1), 1),
        ((2, 1, 0), -1),
    )
    result: Polynomial = {}
    for columns, sign in permutations:
        term = {(0, 0): Fraction(sign)}
        for row, column in enumerate(columns):
            term = multiply(term, matrix[row][column])
        result = add(result, term)
    return result


def expected_disposition(
    lower: Fraction, upper: Fraction, boundary: Fraction
) -> str:
    if lower <= 0 or lower > upper:
        raise ValueError("gamma interval must be positive and ordered")
    if upper < boundary:
        return "ROBUSTLY_STABLE"
    if lower > boundary:
        return "ROBUSTLY_UNSTABLE"
    return "NOT_ESTABLISHED"


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_three_port_robust_reference.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_reference"

    model = data["model"]
    raw_matrix = model["transfer_matrix"]["numerator_matrix"]
    assert len(raw_matrix) == 3 and all(len(row) == 3 for row in raw_matrix)
    assert model["transfer_matrix"]["port_ids"] == ["bus-a", "bus-b", "bus-c"]
    assert model["transfer_matrix"]["common_denominator"] == "s + 1"
    assert model["transfer_matrix"]["feedback_convention"] == "negative_unity_feedback"

    # Parse the matrix from the fixture, then independently form
    # (s+1)I + M(gamma) and calculate its exact symbolic determinant.
    coupling = [[parse_matrix_entry(entry) for entry in row] for row in raw_matrix]
    assert [coupling[i][i] for i in range(3)] == [{}, {}, {}]

    s_plus_one = {(1, 0): Fraction(1), (0, 0): Fraction(1)}
    characteristic_matrix = [
        [
            add(coupling[row][column], s_plus_one if row == column else {})
            for column in range(3)
        ]
        for row in range(3)
    ]
    characteristic = determinant_3(characteristic_matrix)
    expected_characteristic = {
        (3, 0): Fraction(1),
        (2, 0): Fraction(3),
        (1, 0): Fraction(3),
        (0, 0): Fraction(1),
        (0, 1): Fraction(-8),
    }
    assert characteristic == expected_characteristic, characteristic

    # The characteristic polynomial is cubic in s with constant term
    # 1 - 8*gamma. Derive the exact stability boundary from that term.
    constant = characteristic.get((0, 0), Fraction(0))
    gamma_coefficient = characteristic.get((0, 1), Fraction(0))
    assert gamma_coefficient != 0
    boundary = -constant / gamma_coefficient
    assert boundary == Fraction(1, 8)
    assert model["return_ratio_determinant"] == (
        "((s + 1)^3 - 8*gamma) / (s + 1)^3"
    )
    assert model["closed_loop_characteristic_coefficients_descending"] == [
        "1", "3", "3", "1 - 8*gamma"
    ]
    assert model["open_loop_poles"] == [
        {"real": "-1", "imag": "0", "multiplicity": 3}
    ]

    # This cubic's Routh-Hurwitz conditions reduce to positive constant term:
    # a0 = 1 - 8*gamma > 0, while a1=a2=3 and a2*a1 > a3*a0.
    assert characteristic[(3, 0)] == 1
    assert characteristic[(2, 0)] == 3
    assert characteristic[(1, 0)] == 3
    assert 3 * 3 > 1  # 9 > max(1 - 8*gamma) for gamma > 0 in the stable domain.

    # Exact set-level classification; no frequency sweep and no floating roots.
    for case in data["uncertainty_sets"]:
        lower = q(case["gamma_lower"])
        upper = q(case["gamma_upper"])
        actual = expected_disposition(lower, upper, boundary)
        assert actual == case["expected"], (case["id"], actual, case["expected"])
        assert lower > 0 and upper >= lower

        a0_at_upper_gamma = Fraction(1) - 8 * upper
        a0_at_lower_gamma = Fraction(1) - 8 * lower
        if actual == "ROBUSTLY_STABLE":
            assert upper < boundary
            # Every member has positive cubic coefficients and satisfies
            # a2*a1 > a3*a0, the remaining cubic Routh-Hurwitz condition.
            assert a0_at_upper_gamma > 0
            assert Fraction(9) > Fraction(1) * a0_at_lower_gamma
        elif actual == "ROBUSTLY_UNSTABLE":
            assert lower > boundary
            # The Routh first column has signs +,+,+,-: exactly one RHP root.
            assert a0_at_lower_gamma < 0
            assert (Fraction(9) - a0_at_lower_gamma) / 3 > 0
        else:
            assert lower <= boundary <= upper
            assert a0_at_upper_gamma <= 0 <= a0_at_lower_gamma

        print(f"PASS {case['id']}: [{lower}, {upper}] -> {actual}")

    # Invalid intervals are rejected by exact admission logic.
    for lower, upper in (
        (Fraction(0), Fraction(1, 16)),
        (Fraction(1, 4), Fraction(1, 8)),
    ):
        try:
            expected_disposition(lower, upper, boundary)
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
    print("PASS exact three-port determinant, boundary, and robust-reference oracle")


if __name__ == "__main__":
    main()
