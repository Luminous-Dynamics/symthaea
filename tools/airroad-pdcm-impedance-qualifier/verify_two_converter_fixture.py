#!/usr/bin/env python3
"""Independent exact-rational verifier for the two-converter shared-source fixture."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-pdcm-two-converter-shared-source-v0.1.json"


def q(value: dict[str, Any]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def determinant_2(matrix: list[list[Fraction]]) -> Fraction:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def determinant_3(matrix: list[list[Fraction]]) -> Fraction:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_analytical_reference"
    p = data["model"]["parameters"]
    eq = data["equilibrium"]

    vs, rs = q(p["Vs"]), q(p["Rs"])
    c0 = q(p["C0"])
    r1, r2 = q(p["R1"]), q(p["R2"])
    c1, c2 = q(p["C1"]), q(p["C2"])
    p1, p2 = q(p["P1"]), q(p["P2"])
    v0, v1, v2 = q(eq["V0"]), q(eq["V1"]), q(eq["V2"])

    gs, g1, g2 = 1 / rs, 1 / r1, 1 / r2
    i_source = (vs - v0) / rs
    i1 = (v0 - v1) / r1
    i2 = (v0 - v2) / r2

    assert i_source == q(eq["source_current"])
    assert i1 == q(eq["branch_current_1"])
    assert i2 == q(eq["branch_current_2"])
    assert i1 * v1 == p1
    assert i2 * v2 == p2
    assert i_source == i1 + i2

    source_loss = i_source * i_source * rs
    feeder_1_loss = i1 * i1 * r1
    feeder_2_loss = i2 * i2 * r2
    source_power, load_power = vs * i_source, p1 + p2
    assert source_loss == q(eq["source_resistor_loss"])
    assert feeder_1_loss == q(eq["feeder_1_loss"])
    assert feeder_2_loss == q(eq["feeder_2_loss"])
    assert source_power == q(eq["source_power"])
    assert load_power == q(eq["total_load_power"])
    assert source_power == load_power + source_loss + feeder_1_loss + feeder_2_loss

    isolated_1 = (-g1 + p1 / (v1 * v1)) / c1
    isolated_2 = (-g2 + p2 / (v2 * v2)) / c2
    assert isolated_1 == q(data["isolated_branch_reference"]["eigenvalue_1"])
    assert isolated_2 == q(data["isolated_branch_reference"]["eigenvalue_2"])
    assert isolated_1 < 0 and isolated_2 < 0

    jacobian = [
        [-(gs + g1 + g2) / c0, g1 / c0, g2 / c0],
        [g1 / c1, (-g1 + p1 / (v1 * v1)) / c1, Fraction(0)],
        [g2 / c2, Fraction(0), (-g2 + p2 / (v2 * v2)) / c2],
    ]
    observed = [[q(value) for value in row]
                for row in data["coupled_linearization"]["jacobian"]]
    assert jacobian == observed

    trace = sum(jacobian[index][index] for index in range(3))
    principal_minors = (
        determinant_2([[jacobian[0][0], jacobian[0][1]],
                       [jacobian[1][0], jacobian[1][1]]])
        + determinant_2([[jacobian[0][0], jacobian[0][2]],
                         [jacobian[2][0], jacobian[2][2]]])
        + determinant_2([[jacobian[1][1], jacobian[1][2]],
                         [jacobian[2][1], jacobian[2][2]]])
    )
    det = determinant_3(jacobian)
    coefficients = [Fraction(1), -trace, principal_minors, -det]
    expected = [q(value) for value in
                data["coupled_linearization"]["characteristic_polynomial_coefficients_descending"]]
    assert coefficients == expected

    common = [[q(value) for value in row]
              for row in data["coupled_linearization"]["common_mode_jacobian"]]
    assert common == [[Fraction(-3), Fraction(2)],
                      [Fraction(1), Fraction(-1, 2)]]
    common_det = determinant_2(common)
    assert common_det == q(data["coupled_linearization"]["common_mode_determinant"])
    assert common_det < 0  # One positive and one negative real eigenvalue.

    factorized = [Fraction(1), Fraction(4), Fraction(5, 4), Fraction(-1, 4)]
    assert coefficients == factorized
    assert 57 > 7 * 7  # sqrt(57) > 7; (-7 + sqrt(57))/4 is positive.

    # Reconstruct the symmetric common-mode SISO source/load interconnection:
    # Yi(s) = Gi*(Ci*s - hi)/(Ci*s + Gi - hi)
    # Zs(s) = 1/(Gs + C0*s), L(s) = Zs(s)*(Y1(s) + Y2(s)).
    h1, h2 = p1 / (v1 * v1), p2 / (v2 * v2)
    branch_num_1 = [g1 * c1, -g1 * h1]
    branch_den_1 = [c1, g1 - h1]
    branch_num_2 = [g2 * c2, -g2 * h2]
    branch_den_2 = [c2, g2 - h2]
    expected_num_1 = [q(value) for value in
                      data["frequency_domain"]["branch_admittance_numerator_coefficients_descending"]]
    expected_den_1 = [q(value) for value in
                      data["frequency_domain"]["branch_admittance_denominator_coefficients_descending"]]
    assert branch_num_1 == branch_num_2 == expected_num_1
    assert branch_den_1 == branch_den_2 == expected_den_1

    source_den = [c0, gs]
    expected_source_den = [q(value) for value in
                           data["frequency_domain"]["source_impedance_denominator_coefficients_descending"]]
    assert source_den == expected_source_den

    # Identical branch admittances in parallel: 2*Yi(s).
    # Multiply by Zs(s), and check its exact numerator/denominator coefficients.
    return_num = [2 * value for value in branch_num_1]
    return_den = [
        source_den[0] * branch_den_1[0],
        source_den[0] * branch_den_1[1] + source_den[1] * branch_den_1[0],
        source_den[1] * branch_den_1[1],
    ]
    expected_return_num = [q(value) for value in
                           data["frequency_domain"]["common_mode_return_ratio_numerator_coefficients_descending"]]
    expected_return_den = [q(value) for value in
                           data["frequency_domain"]["common_mode_return_ratio_denominator_coefficients_descending"]]
    assert return_num == expected_return_num
    assert return_den == expected_return_den

    # 1 + L(s) = 0: add numerator to denominator after degree alignment.
    closed_loop = [
        return_den[0],
        return_den[1] + return_num[0],
        return_den[2] + return_num[1],
    ]
    expected_closed = [q(value) for value in
                       data["frequency_domain"]["closed_loop_characteristic_polynomial_coefficients_descending"]]
    assert closed_loop == expected_closed == [Fraction(1), Fraction(7, 2), Fraction(-1, 2)]
    common_trace = common[0][0] + common[1][1]
    common_characteristic = [Fraction(1), -common_trace, common_det]
    assert closed_loop == common_characteristic  # excludes stable differential mode

    print("PASS exact equilibrium and power closure")
    print("PASS both isolated converter branches have stable local poles")
    print("PASS exact coupled Jacobian and characteristic polynomial")
    print("PASS source/load branch admittances and source impedance")
    print("PASS common-mode return ratio and closed-loop characteristic polynomial")
    print("PASS shared-source common mode has one positive real pole")
    print("PASS per-converter stability does not imply network stability")


if __name__ == "__main__":
    main()
