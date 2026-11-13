#!/usr/bin/env python3
"""Exact-rational continuous-band discrepancy certificate for AIRROAD MIMO."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-continuous-envelope-v0.1.json"
Poly = list[Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def p_add(a: Poly, b: Poly) -> Poly:
    out = [Fraction(0)] * max(len(a), len(b))
    for i, value in enumerate(a):
        out[i] += value
    for i, value in enumerate(b):
        out[i] += value
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def p_mul(a: Poly, b: Poly) -> Poly:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, left in enumerate(a):
        for j, right in enumerate(b):
            out[i + j] += left * right
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def p_scale(a: Poly, scalar: Fraction) -> Poly:
    return [scalar * value for value in a]


def p_eval(poly: Poly, x: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(poly):
        result = result * x + coefficient
    return result


def p_derivative(poly: Poly) -> Poly:
    if len(poly) <= 1:
        return [Fraction(0)]
    return [Fraction(i) * poly[i] for i in range(1, len(poly))]


def squared_delta(x: Fraction) -> Fraction:
    return (
        Fraction(1_000_000)
        * (x - 1) ** 2
        * (x - 4) ** 2
        * (16 - x) ** 2
        / (25 + x) ** 7
    )


def derive_stationary_polynomial() -> Poly:
    # d(log f)/dx =
    # 2/(x-1) + 2/(x-4) - 2/(16-x) - 7/(25+x).
    xm1 = [Fraction(-1), Fraction(1)]
    xm4 = [Fraction(-4), Fraction(1)]
    sixteen_minus_x = [Fraction(16), Fraction(-1)]
    twentyfive_plus_x = [Fraction(25), Fraction(1)]

    terms = [
        p_scale(p_mul(p_mul(xm4, sixteen_minus_x), twentyfive_plus_x), Fraction(2)),
        p_scale(p_mul(p_mul(xm1, sixteen_minus_x), twentyfive_plus_x), Fraction(2)),
        p_scale(p_mul(p_mul(xm1, xm4), twentyfive_plus_x), Fraction(-2)),
        p_scale(p_mul(p_mul(xm1, xm4), sixteen_minus_x), Fraction(-7)),
    ]
    result = [Fraction(0)]
    for term in terms:
        result = p_add(result, term)
    return result


def interval_has_zero(lo: Fraction, hi: Fraction, roots: tuple[Fraction, ...]) -> bool:
    return any(lo <= root <= hi for root in roots)


def valid_isolating_interval(
    lo: Fraction, hi: Fraction, band_lo: Fraction, band_hi: Fraction, poly: Poly
) -> bool:
    return (
        band_lo <= lo < hi <= band_hi
        and p_eval(poly, lo) > 0
        and p_eval(poly, hi) < 0
    )


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_continuous_rational_discrepancy_envelope.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_continuous_band_certificate"
    assert data["model_set_identity"] == "sha256:finite-knot-hidden-crossport-model-pair-v0.1"
    assert data["topology_identity"] == "sha256:two-port-bus-a-b-reference-v0.1"
    assert data["frequency_unit"] == "rad_per_second"
    assert data["norm"] == "spectral_norm"
    assert data["matrix_error_structure"]["sole_nonzero_entry"] == {
        "row": 0,
        "column": 1,
        "observed_port_id": "bus-a",
        "driven_port_id": "bus-b",
    }

    band = data["frequency_band"]
    omega_lo, omega_hi = q(band["omega_lower"]), q(band["omega_upper"])
    x_lo, x_hi = q(band["x_lower"]), q(band["x_upper"])
    assert (omega_lo, omega_hi) == (Fraction(5, 2), Fraction(7, 2))
    assert (x_lo, x_hi) == (omega_lo**2, omega_hi**2)

    # Derive the stationary polynomial from the rational function. Do not
    # trust the coefficient declaration in the fixture.
    p = derive_stationary_polynomial()
    expected_p = [Fraction(-4648), Fraction(2520), Fraction(-213), Fraction(1)]
    assert p == expected_p
    assert p == [Fraction(v) for v in data["exact_reduction"]["stationary_polynomial_coefficients_ascending"]]

    p1, p2 = p_derivative(p), p_derivative(p_derivative(p))
    assert p_eval(p2, x_hi) == q(data["critical_root_isolation"]["second_derivative_at_band_upper"])
    assert p_eval(p2, x_hi) < 0  # p'' is increasing but still negative at upper band edge
    assert p_eval(p1, x_lo) < 0  # p' decreases over band; already negative at lower edge
    assert p_eval(p, x_lo) > 0
    assert p_eval(p, x_hi) < 0
    assert (p_eval(p, x_lo), p_eval(p, x_hi)) == (
        q(data["critical_root_isolation"]["polynomial_at_band_lower"]),
        q(data["critical_root_isolation"]["polynomial_at_band_upper"]),
    )
    assert p_eval(p1, x_lo) == q(data["critical_root_isolation"]["first_derivative_at_band_lower"])

    # p'<0 throughout because p''<0 throughout; hence p is strictly
    # decreasing and has exactly one root in the band.
    root = data["critical_root_isolation"]
    root_lo, root_hi = q(root["lower"]), q(root["upper"])
    assert valid_isolating_interval(root_lo, root_hi, x_lo, x_hi, p)
    assert p_eval(p, root_lo) == q(root["polynomial_at_lower"])
    assert p_eval(p, root_hi) == q(root["polynomial_at_upper"])
    assert root_lo == Fraction(1017, 100)
    assert root_hi == Fraction(1018, 100)

    # f rises then falls, so its minimum is one of the two endpoints.
    left_value, right_value = squared_delta(x_lo), squared_delta(x_hi)
    minimum = min(left_value, right_value)
    assert left_value < right_value
    expected_min = data["expected_enclosures"]["minimum_squared_discrepancy"]
    assert minimum == q(expected_min["lower"]) == q(expected_min["upper"])
    assert minimum > 0

    # The unique maximum lies in the root interval. Bound each monotone
    # positive factor at its endpoint to obtain a rational global upper.
    maximum_upper = (
        Fraction(1_000_000)
        * (root_hi - 1) ** 2
        * (root_hi - 4) ** 2
        * (16 - root_lo) ** 2
        / (25 + root_lo) ** 7
    )
    sample_omega = Fraction(319, 100)
    sample_x = sample_omega**2
    assert root_lo < sample_x < root_hi
    maximum_lower = squared_delta(sample_x)
    expected_bounds = data["expected_enclosures"]
    assert maximum_lower == q(expected_bounds["supremum_squared_lower_bound"])
    assert maximum_upper == q(expected_bounds["supremum_squared_upper_bound"])
    assert maximum_lower <= maximum_upper

    ratio = q(expected_bounds["upper_to_lower_ratio_limit"])
    assert ratio == Fraction(1_005_435, 1_000_000)
    assert maximum_upper * ratio.denominator < maximum_lower * ratio.numerator

    # A coarse grid is demonstrably below a known unsampled value. It cannot
    # provide the universal supremum certificate.
    grid = [q(item) for item in data["coarse_grid_control"]["sampled_omega"]]
    assert grid == [Fraction(5, 2), Fraction(3), Fraction(7, 2)]
    grid_values = [squared_delta(omega**2) for omega in grid]
    grid_max = max(grid_values)
    expected_grid_max = q(data["coarse_grid_control"]["sampled_max_squared"])
    assert grid_max == expected_grid_max
    assert grid_max < maximum_lower
    assert q(data["coarse_grid_control"]["unsampled_lower_witness_omega"]) == sample_omega
    assert q(data["coarse_grid_control"]["unsampled_value_squared"]) == maximum_lower

    # Counterfactual band crosses the exact zero at omega=2: a positive
    # uniform separation statement must be rejected.
    zero_band = (Fraction(3, 2), Fraction(5, 2))
    assert interval_has_zero(*zero_band, roots=(Fraction(1), Fraction(2), Fraction(4)))
    assert not interval_has_zero(omega_lo, omega_hi, roots=(Fraction(1), Fraction(2), Fraction(4)))

    # Hostile root brackets and identities are checked against actual rules.
    bad_root = (Fraction(1016, 100), Fraction(1017, 100))
    assert not valid_isolating_interval(*bad_root, x_lo, x_hi, p)
    outside_root = (Fraction(20), Fraction(21))
    assert not valid_isolating_interval(*outside_root, x_lo, x_hi, p)

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["finite-grid-promoted-to-universal-bound"] == "reject"
    assert controls["frequency-band-crosses-delta-zero-at-2-rad-s"] == "uniform-positive-separation-not-established"
    assert controls["bad-critical-root-isolation"] == "reject"
    assert controls["root-interval-outside-frequency-band"] == "reject"
    assert controls["stale-model-set-identity"] == "not-evaluable"
    assert controls["frequency-unit-mismatch"] == "not-evaluable"
    assert controls["port-row-column-swapped"] == "not-evaluable"

    print(f"PASS exact stationary polynomial: {p}")
    print(f"PASS unique critical point isolated in [{root_lo}, {root_hi}]")
    print(f"PASS continuous-band minimum squared discrepancy: {minimum}")
    print(f"PASS H-infinity squared enclosure: [{maximum_lower}, {maximum_upper}]")
    print(f"PASS squared enclosure ratio < {ratio}")
    print("PASS coarse-grid counterexample and all negative controls")
    print("PASS exact-rational continuous-band discrepancy certificate")


if __name__ == "__main__":
    main()
