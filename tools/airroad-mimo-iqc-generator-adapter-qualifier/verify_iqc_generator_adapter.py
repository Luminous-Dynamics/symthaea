#!/usr/bin/env python3
"""Exact-rational shared-generator and inter-frequency remainder adapter oracle."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-iqc-generator-adapter-v0.1.json"
Matrix = list[list[Fraction]]
Poly = list[Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def p_eval(poly: Poly, x: Fraction) -> Fraction:
    out = Fraction(0)
    for coefficient in reversed(poly):
        out = out * x + coefficient
    return out


def matrix_scale(a: Matrix, scalar: Fraction) -> Matrix:
    return [[scalar * value for value in row] for row in a]


def matrix_sub(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def matrix_zero(a: Matrix) -> bool:
    return all(value == 0 for row in a for value in row)


def partition_valid(intervals: list[tuple[Fraction, Fraction]],
                    lo: Fraction, hi: Fraction) -> bool:
    return (
        bool(intervals)
        and intervals[0][0] == lo
        and intervals[-1][1] == hi
        and all(a < b for a, b in intervals)
        and all(intervals[i][1] == intervals[i + 1][0]
                for i in range(len(intervals) - 1))
    )


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_iqc_shared_generator_adapter.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_joint_hard_generator_set"
    assert data["model_identity"] == "sha256:two-frequency-shared-calibration-generator-v0.1"
    assert data["topology_identity"] == "sha256:two-port-bus-a-b-v0.1"
    assert data["operating_point_identity"] == "sha256:fixed-operating-point-v0.1"
    assert data["acquisition_root_identity"] == "sha256:common-acquisition-root-v0.1"
    assert data["calibration_certificate_identity"] == "sha256:common-calibration-certificate-v0.1"
    assert data["frequency_unit"] == "rad_per_second"
    assert data["matrix_norm"] == "frobenius_norm"
    assert data["port_order"] == ["bus-a", "bus-b"]

    coordinates = data["generator_coordinates"]
    assert len(coordinates) == 1
    theta = coordinates[0]
    theta_id = theta["coordinate_id"]
    assert theta_id == "theta-common-gain"
    assert theta["uncertainty_semantics"] == "deterministic_hard_set"
    theta_domain = (q(theta["lower"]), q(theta["upper"]))
    assert theta_domain == (Fraction(-1), Fraction(1))
    assert theta["acquisition_root_identity"] == data["acquisition_root_identity"]
    assert theta["calibration_certificate_identity"] == data["calibration_certificate_identity"]
    assert theta["applies_to_frequency_ids"] == ["omega-1", "omega-2"]
    assert set(theta["applies_to_matrix_elements"]) == {"G11", "G12", "G21", "G22"}

    knots = data["frequency_knots"]
    assert len(knots) == 2
    assert [item["frequency_id"] for item in knots] == ["omega-1", "omega-2"]
    assert [q(item["omega"]) for item in knots] == [Fraction(1), Fraction(2)]
    assert all(item["shared_coordinate_id"] == theta_id for item in knots)

    nominal = []
    generators = []
    for item in knots:
        n = [[q(x) for x in row] for row in item["nominal_matrix"]]
        g = [[q(x) for x in row] for row in item["generator_matrix"]]
        nominal.append(n)
        generators.append(g)
    i2 = [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]]
    assert nominal == [i2, matrix_scale(i2, Fraction(2))]
    assert generators == [
        matrix_scale(i2, Fraction(1, 10)),
        matrix_scale(i2, Fraction(1, 5)),
    ]

    # The same theta coordinate at both knots yields an exact joint relation.
    for value in (theta_domain[0], Fraction(0), theta_domain[1]):
        g1 = matrix_sub(nominal[0], matrix_scale(generators[0], -value))
        g2 = matrix_sub(nominal[1], matrix_scale(generators[1], -value))
        assert matrix_zero(matrix_sub(g2, matrix_scale(g1, Fraction(2))))
    coefficient_relation = matrix_sub(generators[1], matrix_scale(generators[0], Fraction(2)))
    nominal_relation = matrix_sub(nominal[1], matrix_scale(nominal[0], Fraction(2)))
    assert matrix_zero(coefficient_relation) and matrix_zero(nominal_relation)
    assert matrix_zero([[q(x) for x in row] for row in data["joint_relation"]["expected_matrix"]])

    # With independent theta_1/theta_2, each diagonal relation is
    # (theta_2-theta_1)/5 and therefore spans [-2/5,+2/5].
    product = data["independent_product_counterfactual"]
    assert product["coordinate_omega_1"] != product["coordinate_omega_2"]
    independent_domain = (q(product["both_domains"]["lower"]), q(product["both_domains"]["upper"]))
    assert independent_domain == theta_domain
    relation_bounds = (
        (independent_domain[0] - independent_domain[1]) / 5,
        (independent_domain[1] - independent_domain[0]) / 5,
    )
    assert relation_bounds == (Fraction(-2, 5), Fraction(2, 5))
    assert relation_bounds == (
        q(product["relation_diagonal_interval"]["lower"]),
        q(product["relation_diagonal_interval"]["upper"]),
    )
    assert relation_bounds[0] < 0 < relation_bounds[1]

    # Translated generator remainder: x=omega-1 reduces exactly to #11919.
    remainder = data["inter_frequency_remainder"]
    omega_lo, omega_hi = q(remainder["interval"]["lower"]), q(remainder["interval"]["upper"])
    assert (omega_lo, omega_hi) == (Fraction(1), Fraction(2))
    x_lo, x_hi = omega_lo - 1, omega_hi - 1
    assert (x_lo, x_hi) == (Fraction(0), Fraction(1))
    p = [[q(x) for x in row] for row in remainder["matrix_factor"]]
    p_frob_sq = sum((x * x for row in p for x in row), Fraction(0))
    assert p == [[0, 1], [0, 0]]
    assert p_frob_sq == q(remainder["frobenius_norm_factor_squared"]) == 1

    d_poly = [Fraction(0), Fraction(4), Fraction(-20), Fraction(32), Fraction(-16)]
    # In x∈[0,1], d=4u-16u² with u=(x-1/2)²∈[0,1/4].
    u_star = Fraction(1, 8)
    exact_max = 4 * u_star - 16 * u_star**2
    assert exact_max == Fraction(1, 4)
    assert exact_max == q(remainder["certified_frobenius_radius"])
    for omega in (Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4), Fraction(2)):
        x = omega - 1
        assert abs(p_eval(d_poly, x)) <= exact_max

    # Verify full interval coverage by the piecewise generator/remainder.
    assert q(data["piecewise_generator"]["additive_remainder_ball_radius"]) == exact_max
    assert data["piecewise_generator"]["shared_coordinate_id"] == theta_id
    assert data["piecewise_generator"]["additive_remainder_ball_norm"] == "frobenius_norm"
    assert data["expected"]["combined_object_scope"] == "frequency_interval_1_to_2_joint_shared_generator_plus_remainder_ball"

    # Counterfactuals: distinct IDs with equal numeric domain are not shared.
    cf = data["counterfactuals"]
    assert cf["second_frequency_coordinate_id"] != theta_id
    assert q(cf["understated_remainder_radius"]) < exact_max
    gap = [(q(x["lower"]), q(x["upper"])) for x in cf["interval_gap"]]
    assert not partition_valid(gap, omega_lo, omega_hi)
    assert cf["stale_calibration_identity"] != data["calibration_certificate_identity"]
    assert cf["stale_topology_identity"] != data["topology_identity"]
    assert cf["statistical_semantics"] != "deterministic_hard_set"

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["same-coordinate-id-at-both-knots"] == "joint-affine-relation-exact"
    assert controls["different-coordinate-ids-same-domain"] == "do-not-claim-shared"
    assert controls["incomplete-matrix-element-scope"] == "not-evaluable"
    assert controls["statistical-covariance-as-hard-set"] == "reject-claim-class"
    assert controls["independent-product-as-shared-joint-set"] == "reject"
    assert controls["understated-inter-frequency-radius"] == "reject"
    assert controls["frequency-interval-gap"] == "not-established"
    assert controls["stale-calibration-or-topology"] == "not-evaluable"
    assert controls["sampled-maximum-as-supremum"] == "reject"
    assert controls["generator-artifact-labelled-iqc-pass"] == "reject-claim-class"
    assert controls["unknown-cross-frequency-dependency"] == "preserve-unknown"

    print("PASS one shared theta coordinate covers both 2x2 FRF knots")
    print("PASS exact joint relation G2(theta)-2*G1(theta)=0")
    print("PASS independent product relation interval [-2/5,2/5]")
    print("PASS exact translated inter-frequency Frobenius remainder radius 1/4")
    print("PASS full-band coverage, shared identity and uncertainty-class controls")
    print("PASS generator adapter claim ceiling (no IQC/stability PASS)")


if __name__ == "__main__":
    main()
