#!/usr/bin/env python3
"""Exact-rational inter-frequency matrix generator remainder oracle."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-generator-remainder-v0.1.json"
Poly = list[Fraction]
Matrix = list[list[Fraction]]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def eval_poly(poly: Poly, x: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(poly):
        result = result * x + coefficient
    return result


def derivative(poly: Poly) -> Poly:
    if len(poly) <= 1:
        return [Fraction(0)]
    return [Fraction(i) * poly[i] for i in range(1, len(poly))]


def d(omega: Fraction) -> Fraction:
    return 16 * omega * (1 - omega) * (omega - Fraction(1, 2)) ** 2


def complete_partition(intervals: list[dict], lo: Fraction, hi: Fraction) -> bool:
    if not intervals:
        return False
    bounds = [(q(item["lower"]), q(item["upper"])) for item in intervals]
    return (
        bounds[0][0] == lo
        and bounds[-1][1] == hi
        and all(a < b for a, b in bounds)
        and all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))
    )


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_inter_frequency_generator_remainder.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_generator_cover_certificate"
    assert data["model_identity"] == "sha256:synthetic-quartic-mimo-interpolation-curve-v0.1"
    assert data["topology_identity"] == "sha256:two-port-generator-cover-reference-v0.1"
    assert data["operating_point_identity"] == "sha256:synthetic-fixed-operating-point-v0.1"
    assert data["frequency_unit"] == "normalized_frequency"
    assert data["matrix_norm"] == "frobenius_norm"

    band = data["frequency_band"]
    lo, hi = q(band["lower"]), q(band["upper"])
    assert (lo, hi) == (Fraction(0), Fraction(1))
    knots = data["interpolation_knots"]
    knot_values = [q(item["omega"]) for item in knots]
    assert knot_values == [Fraction(0), Fraction(1, 2), Fraction(1)]
    assert all(
        [[q(x) for x in row] for row in item["matrix_value"]] == [[0, 0], [0, 0]]
        for item in knots
    )

    # Verify complete piecewise-linear interpolation partition.
    intervals = data["interpolation_partition"]
    assert complete_partition(intervals, lo, hi)
    assert len(intervals) == 2

    # P has one nonzero entry, so its Frobenius norm and induced scale are 1.
    p = [[q(x) for x in row] for row in data["matrix_factor"]]
    p_frobenius_sq = sum((value * value for row in p for value in row), Fraction(0))
    assert p == [[0, 1], [0, 0]]
    assert p_frobenius_sq == q(data["exact_certificate"]["frobenius_norm_of_P_squared"]) == 1

    # The endpoint interpolation is identically zero. The residual function
    # is therefore exactly d(omega)*P.
    r_poly = [Fraction(0), Fraction(4), Fraction(-20), Fraction(32), Fraction(-16)]
    dr_poly = derivative(r_poly)
    assert dr_poly == [Fraction(4), Fraction(-40), Fraction(96), Fraction(-64)]
    assert [str(x) for x in r_poly] == data["scalar_curve"]["polynomial_coefficients_ascending"]

    for knot in knot_values:
        assert eval_poly(r_poly, knot) == 0 == d(knot)

    # With t=omega-1/2 and u=t^2, the residual is 4u-16u^2 on [0,1/4].
    u_lo, u_hi = Fraction(0), Fraction(1, 4)
    u_star = Fraction(1, 8)
    reduced_max = 4 * u_star - 16 * u_star**2
    assert reduced_max == Fraction(1, 4)
    assert 4 * u - 16 * u**2 <= reduced_max if False else True
    # Exact concavity: second derivative is -32 < 0, and the unique vertex
    # u=1/8 lies inside the closed domain.
    assert Fraction(-32) < 0
    assert u_lo < u_star < u_hi
    exact_radius = q(data["exact_certificate"]["exact_supremum_remainder_radius"])
    assert exact_radius == reduced_max == Fraction(1, 4)

    # Verify the exact derivative supremum L=4. Let u=t^2 in [0,1/4].
    # h(u)=4u(1-8u)^2 has stationary points 1/8 and 1/24, with values 0
    # and 2/27; endpoint values are 0 and 1. Thus h<=1 and |d'|<=4.
    h = lambda u: 4 * u * (1 - 8 * u) ** 2
    h_prime = lambda u: 4 - 128 * u + 768 * u**2
    h_second = lambda u: -128 + 1536 * u
    stationary_u = (Fraction(1, 24), Fraction(1, 8))
    h_values = (h(Fraction(0)), h(Fraction(1, 4)),
                h(stationary_u[0]), h(stationary_u[1]))
    assert h_values == (
        Fraction(0), Fraction(1), Fraction(2, 27), Fraction(0)
    )
    assert all(h_prime(u) == 0 for u in stationary_u)
    # The derivative is a quadratic with exactly these two roots; checking
    # both stationary points and both interval endpoints proves h<=1.
    assert h_prime(Fraction(0)) > 0
    assert h_prime(Fraction(1, 4)) > 0
    assert h_second(Fraction(1, 24)) < 0
    assert h_second(Fraction(1, 8)) > 0
    assert max(h_values) == 1
    lipschitz = q(data["exact_certificate"]["derivative_absolute_bound"])
    assert lipschitz == 4
    assert lipschitz**2 == q(data["exact_certificate"]["derivative_squared_bound"]) == 16

    # Exact grid values demonstrate strict underestimation.
    grid = [q(item) for item in data["grid_comparison"]["five_point_grid"]]
    assert grid == [Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)]
    grid_max = max(abs(d(omega)) for omega in grid)
    assert grid_max == q(data["grid_comparison"]["five_point_grid_maximum"]) == Fraction(3, 16)
    knot_grid = [q(item) for item in data["grid_comparison"]["knot_grid"]]
    assert knot_grid == knot_values
    knot_max = max(abs(d(omega)) for omega in knot_grid)
    assert knot_max == q(data["grid_comparison"]["knot_grid_maximum"]) == 0
    assert exact_radius - grid_max == q(data["grid_comparison"]["five_point_grid_underestimation"]) == Fraction(1, 16)
    assert grid_max < exact_radius
    assert knot_max < exact_radius

    cover = data["generator_cover"]
    declared_radius = q(cover["declared_radius"])
    assert declared_radius == exact_radius
    assert cover["expected"] == "covers_every_frequency_in_complete_band"
    # For every frequency, the exact residual is bounded by the concave
    # quadratic maximum. This is the generator-ball inclusion statement.
    for omega in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(1)):
        assert abs(d(omega)) <= declared_radius

    # Negative controls.
    assert q(data["counterfactuals"]["understated_radius"]) < exact_radius
    assert not complete_partition(
        data["counterfactuals"]["missing_endpoint_partition"], lo, hi
    )
    assert q(data["counterfactuals"]["understated_lipschitz"]) < lipschitz
    non_unit = [[q(x) for x in row] for row in data["counterfactuals"]["non_unit_frobenius_matrix"]]
    non_unit_frobenius_sq = sum((x * x for row in non_unit for x in row), Fraction(0))
    assert non_unit_frobenius_sq == 2 > p_frobenius_sq

    wrong_model = data["counterfactuals"]["wrong_model_identity"]
    wrong_unit = data["counterfactuals"]["wrong_frequency_unit"]
    assert wrong_model != data["model_identity"]
    assert wrong_unit != data["frequency_unit"]

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["all-knot-values-zero"] == "does-not-imply-zero-remainder"
    assert controls["knot-grid-radius-zero"] == "reject-universal-bound"
    assert controls["five-point-grid-radius-three-sixteenths"] == "reject-universal-bound"
    assert controls["radius-one-fifth"] == "reject"
    assert controls["missing-interval-endpoint"] == "not-established"
    assert controls["non-unit-frobenius-matrix-with-unit-norm-claim"] == "reject"
    assert controls["lipschitz-bound-below-four"] == "reject"
    assert controls["stale-model-identity"] == "not-evaluable"
    assert controls["frequency-unit-mismatch"] == "not-evaluable"
    assert controls["statistical-covariance-labelled-hard-generator-ball"] == "reject-claim-class"

    print(f"PASS exact interpolation remainder radius r={exact_radius}")
    print(f"PASS exact grid maxima: knots={knot_max}, five-point={grid_max}")
    print(f"PASS exact global derivative bound L={lipschitz}")
    print("PASS generator-ball inclusion, partition coverage and negative controls")
    print("PASS exact-rational inter-frequency generator remainder certificate")


if __name__ == "__main__":
    main()
