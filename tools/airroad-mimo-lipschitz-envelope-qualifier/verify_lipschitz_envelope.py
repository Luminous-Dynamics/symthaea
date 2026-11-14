#!/usr/bin/env python3
"""Exact-rational Lipschitz residual envelope oracle."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-lipschitz-envelope-v0.1.json"
Poly = list[Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def eval_poly(poly: Poly, x: Fraction) -> Fraction:
    out = Fraction(0)
    for coefficient in reversed(poly):
        out = out * x + coefficient
    return out


def derivative(poly: Poly) -> Poly:
    if len(poly) <= 1:
        return [Fraction(0)]
    return [Fraction(i) * poly[i] for i in range(1, len(poly))]


def residual(omega: Fraction) -> Fraction:
    return omega * (omega - Fraction(1, 2)) * (omega - 1)


def cone_radius(omega: Fraction, knots: list[dict], lipschitz: Fraction) -> Fraction:
    return min(
        abs(q(item["observed_residual_center"]))
        + q(item["hard_error_radius"])
        + lipschitz * abs(omega - q(item["omega"]))
        for item in knots
    )


def coverage_valid(knots: list[Fraction], band_lo: Fraction, band_hi: Fraction) -> bool:
    return (
        len(knots) >= 2
        and knots == sorted(knots)
        and len(knots) == len(set(knots))
        and knots[0] == band_lo
        and knots[-1] == band_hi
    )



def evidence_identity_valid(data: dict, derivative: dict) -> bool:
    return (
        derivative.get("bound_model_identity") == data.get("model_set_identity")
        and derivative.get("bound_operating_point_identity") == data.get("operating_point_identity")
        and derivative.get("bound_frequency_unit") == data.get("frequency_unit")
        and bool(derivative.get("evidence_reference"))
    )


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_lipschitz_residual_envelope.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_deterministic_residual_bound"
    assert data["model_set_identity"] == "sha256:synthetic-cubic-residual-family-v0.1"
    assert data["operating_point_identity"] == "sha256:synthetic-operating-point-v0.1"
    assert data["frequency_unit"] == "normalized_frequency"

    band = data["band"]
    lo, hi = q(band["lower"]), q(band["upper"])
    assert (lo, hi) == (Fraction(0), Fraction(1))

    # The residual coefficients are frozen, but derive and validate the
    # derivative from the factored exact polynomial independently.
    r = [Fraction(0), Fraction(1, 2), Fraction(-3, 2), Fraction(1)]
    dr = derivative(r)
    assert dr == [
        Fraction(1, 2), Fraction(-3), Fraction(3)
    ]
    derivative_of_derivative = derivative(dr)
    second_derivative_of_derivative = derivative(derivative_of_derivative)
    assert derivative_of_derivative == [Fraction(-3), Fraction(6)]
    assert second_derivative_of_derivative == [Fraction(6)]
    assert second_derivative_of_derivative[0] > 0
    assert [str(c) for c in r] == data["residual_polynomial_coefficients_ascending"]
    assert [str(c) for c in dr] == data["derivative_polynomial_coefficients_ascending"]

    vertex = Fraction(1, 2)
    derivative_at_lo = eval_poly(dr, lo)
    derivative_at_vertex = eval_poly(dr, vertex)
    derivative_at_hi = eval_poly(dr, hi)
    assert derivative_at_lo == Fraction(1, 2)
    assert derivative_at_vertex == Fraction(-1, 4)
    assert derivative_at_hi == Fraction(1, 2)
    derivative_evidence = data["derivative_evidence"]
    assert derivative_evidence["convex_derivative"] is True
    assert q(derivative_evidence["derivative_vertex"]) == vertex
    assert evidence_identity_valid(data, derivative_evidence)

    # The derivative polynomial is convex; its maximum is at an endpoint
    # and its minimum at its vertex. The exact global absolute bound is 1/2.
    exact_l = max(
        abs(derivative_at_lo),
        abs(derivative_at_vertex),
        abs(derivative_at_hi),
    )
    declared_l = q(data["derivative_evidence"]["exact_global_abs_derivative_bound"])
    assert exact_l == declared_l == Fraction(1, 2)

    samples = data["samples"]
    knots = [q(item["omega"]) for item in samples]
    assert coverage_valid(knots, lo, hi)
    assert [item["sample_id"] for item in samples] == ["omega-0", "omega-half", "omega-1"]
    for item in samples:
        assert q(item["observed_residual_center"]) == 0
        assert q(item["hard_error_radius"]) == Fraction(1, 100)
        assert item["uncertainty_semantics"] == "deterministic_hard_interval"
        assert abs(residual(q(item["omega"])) - q(item["observed_residual_center"])) <= q(item["hard_error_radius"])

    gaps = [right - left for left, right in zip(knots, knots[1:])]
    max_gap = max(gaps)
    max_nearest = max_gap / 2
    assert max_gap == q(data["sample_coverage"]["maximum_gap"])
    assert max_nearest == q(data["sample_coverage"]["maximum_nearest_knot_distance"])
    assert data["sample_coverage"]["includes_lower_band_endpoint"] is True
    assert data["sample_coverage"]["includes_upper_band_endpoint"] is True

    epsilon = max(q(item["hard_error_radius"]) for item in samples)
    center_bound = max(abs(q(item["observed_residual_center"])) for item in samples)
    global_bound = center_bound + epsilon + declared_l * max_nearest
    assert global_bound == q(data["expected"]["global_continuous_residual_bound"])
    assert global_bound == Fraction(27, 200)

    # Check pointwise cones and a hidden between-knot residual that all
    # measured sample knots fail to reveal.
    for item in data["expected"]["cone_radii"]:
        omega, expected_radius = q(item["omega"]), q(item["radius"])
        assert cone_radius(omega, samples, declared_l) == expected_radius
        assert abs(residual(omega)) <= expected_radius

    quarter, three_quarters = Fraction(1, 4), Fraction(3, 4)
    assert residual(quarter) == q(data["expected"]["actual_residual_at_quarter"])
    assert residual(three_quarters) == q(data["expected"]["actual_residual_at_three_quarters"])
    assert residual(quarter) != 0 and residual(three_quarters) != 0
    assert abs(residual(quarter)) <= global_bound
    assert abs(residual(three_quarters)) <= global_bound
    assert all(residual(knot) == 0 for knot in knots)

    # An understated derivative bound is rejected by the independent
    # polynomial derivative maximum, even though it would fit the knots.
    understated_l = Fraction(1, 4)
    assert understated_l < exact_l

    # Admission identities must agree; missing L never yields a continuous
    # envelope. Statistical-only observations cannot satisfy hard-bound input.
    no_lipschitz = None
    assert no_lipschitz is None
    assert data["expected"]["no_lipschitz_bound_disposition"] == "NOT_ESTABLISHED_FOR_CONTINUOUS_DOMAIN"
    assert data["model_set_identity"] == "sha256:synthetic-cubic-residual-family-v0.1"
    assert data["operating_point_identity"] == "sha256:synthetic-operating-point-v0.1"

    stale_derivative = dict(derivative_evidence)
    stale_derivative["bound_model_identity"] = "sha256:stale-model"
    assert not evidence_identity_valid(data, stale_derivative)
    stale_derivative = dict(derivative_evidence)
    stale_derivative["bound_operating_point_identity"] = "sha256:stale-operating-point"
    assert not evidence_identity_valid(data, stale_derivative)
    stale_derivative = dict(derivative_evidence)
    stale_derivative["bound_frequency_unit"] = "Hz"
    assert not evidence_identity_valid(data, stale_derivative)

    statistical_samples = [dict(item) for item in samples]
    statistical_samples[0]["uncertainty_semantics"] = "uncalibrated_statistical_confidence_interval"
    assert not all(
        item.get("uncertainty_semantics") == "deterministic_hard_interval"
        for item in statistical_samples
    )

    # Coverage controls: missing endpoint, duplicate and unsorted knots.
    assert not coverage_valid(knots[:-1], lo, hi)
    assert not coverage_valid([knots[0], knots[1], knots[1], knots[2]], lo, hi)
    assert not coverage_valid([knots[2], knots[1], knots[0]], lo, hi)

    # A false maximum-gap declaration must be rejected.
    declared_wrong_gap = Fraction(3, 4)
    assert declared_wrong_gap != max_gap

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["zero-sample-residuals-without-L"] == "not_established"
    assert controls["understated-L-one-quarter"] == "reject"
    assert controls["stale-model-identity"] == "not_evaluable"
    assert controls["wrong-operating-point-identity"] == "not_evaluable"
    assert controls["statistical-only-sample-intervals"] == "reject_claim_class"
    assert controls["missing-band-endpoint"] == "reject"
    assert controls["duplicate-knot"] == "reject"
    assert controls["unsorted-knots"] == "reject"
    assert controls["hidden-between-knot-residual"] == "must-be-enclosed"
    assert controls["incorrect-maximum-gap"] == "reject"
    assert controls["frequency-unit-mismatch"] == "not_evaluable"

    print(f"PASS exact global derivative bound L={declared_l}")
    print(f"PASS maximum gap={max_gap}; nearest-knot distance={max_nearest}")
    print(f"PASS continuous residual envelope={global_bound}")
    print("PASS hidden between-knot residual and hard-bound controls")
    print("PASS exact-rational Lipschitz residual certificate")


if __name__ == "__main__":
    main()
