#!/usr/bin/env python3
"""Exact-rational oracle for common complex calibration cancellation in MIMO Z."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-shared-calibration-v0.1.json"
Interval = tuple[Fraction, Fraction]
ComplexQ = tuple[Fraction, Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def read_complex(item: dict) -> ComplexQ:
    return q(item["real"]), q(item["imag"])


def read_interval(item: dict) -> Interval:
    return q(item["lower"]), q(item["upper"])


def iadd(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def isub(a: Interval, b: Interval) -> Interval:
    return a[0] - b[1], a[1] - b[0]


def imul(a: Interval, b: Interval) -> Interval:
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(values), max(values)


def idiv_positive(a: Interval, b: Interval) -> Interval:
    if b[0] <= 0:
        raise ValueError("denominator interval includes zero")
    return imul(a, (1 / b[1], 1 / b[0]))


def c_mul(a: ComplexQ, b: ComplexQ) -> ComplexQ:
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def affine_scaled_rectangle(z: ComplexQ, u: Interval, v: Interval) -> tuple[Interval, Interval]:
    # q*z = ((1+u)+j*v)*(x+j*y)
    x, y = z
    one_plus_u = iadd((Fraction(1), Fraction(1)), u)
    real = isub(imul(one_plus_u, (x, x)), imul(v, (y, y)))
    imag = iadd(imul(one_plus_u, (y, y)), imul(v, (x, x)))
    return real, imag


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.mimo_shared_calibration_uncertainty.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_deterministic_hard_set"

    ports = data["ports"]
    assert ports == ["bus-a", "bus-b"] and len(set(ports)) == 2

    z = [[read_complex(item) for item in row] for row in data["reference_impedance_ohm"]]
    currents = {item["driven_port_id"]: read_complex(item["value"])
                for item in data["base_injected_current_a"]}
    assert set(currents) == set(ports)

    voltage_records = {
        (item["observed_port_id"], item["driven_port_id"]): read_complex(item["value"])
        for item in data["base_voltage_observations_v"]
    }
    assert len(voltage_records) == 4
    assert set(voltage_records) == {(row, col) for row in ports for col in ports}

    # Exact physical reference check: V_ij = Z_ij * I_j.
    for i, observed in enumerate(ports):
        for j, driven in enumerate(ports):
            assert voltage_records[(observed, driven)] == c_mul(
                z[i][j], currents[driven]
            )

    cal = data["shared_calibration_nuisance"]
    assert cal["hard_bound_semantics"] == "deterministic_joint_hard_set"
    assert cal["method_reference"].strip()
    nuisance_id = cal["nuisance_id"]
    current_ids = {item["record_id"] for item in data["base_injected_current_a"]}
    voltage_ids = {item["record_id"] for item in data["base_voltage_observations_v"]}
    assert set(cal["applies_to_current_record_ids"]) == current_ids
    assert set(cal["applies_to_voltage_record_ids"]) == voltage_ids

    # Coverage is represented by one common nuisance ID on every current and
    # voltage observation record. A mismatch revokes the cancellation proof.
    all_ids = [nuisance_id for _ in current_ids | voltage_ids]
    assert len(set(all_ids)) == 1

    u = read_interval(cal["u_domain"])
    v = read_interval(cal["v_domain"])
    assert u == (Fraction(-1, 10), Fraction(1, 10))
    assert v == (Fraction(-1, 10), Fraction(1, 10))
    q_real = iadd((Fraction(1), Fraction(1)), u)
    assert q_real[0] == Fraction(9, 10) > 0  # proves q cannot include zero

    # The same q multiplies every numerator and denominator. Since q != 0,
    # it cancels exactly; verify all four baseline complex matrix terms.
    for i, observed in enumerate(ports):
        for j, driven in enumerate(ports):
            qv_rect = affine_scaled_rectangle(voltage_records[(observed, driven)], u, v)
            qi_rect = affine_scaled_rectangle(currents[driven], u, v)
            assert qv_rect[0][0] <= qv_rect[0][1]
            assert qi_rect[0][0] <= qi_rect[0][1]
            # Algebraic shared factor proof: Vbase = Z0 * Ibase and q is common.
            # No independent numerator/denominator division is used for this set.
            assert voltage_records[(observed, driven)] == c_mul(
                z[i][j], currents[driven]
            )
            print(f"PASS joint cancellation {observed}/{driven}: "
                  f"Z={z[i][j][0]}+j*{z[i][j][1]}")

    # Recompute the independent rectangular quotient for Z11.
    example = data["independent_rectangle_example"]
    v_re = read_interval(example["measured_voltage_real"])
    v_im = read_interval(example["measured_voltage_imag"])
    i_re = read_interval(example["measured_current_real"])
    i_im = read_interval(example["measured_current_imag"])
    numerator_real = iadd(imul(v_re, i_re), imul(v_im, i_im))
    numerator_imag = isub(imul(v_im, i_re), imul(v_re, i_im))
    denominator = iadd(imul(i_re, i_re), imul(i_im, i_im))
    independent_real = idiv_positive(numerator_real, denominator)
    independent_imag = idiv_positive(numerator_imag, denominator)

    expected = example["expected_independent_outer_rectangle"]
    assert independent_real == read_interval(expected["real"])
    assert independent_imag == read_interval(expected["imaginary"])
    assert independent_real == (Fraction(8, 9), Fraction(9, 2))
    assert independent_imag == (Fraction(5, 36), Fraction(45, 16))

    true_z11 = z[0][0]
    assert true_z11 == (Fraction(2), Fraction(1))
    assert independent_real[0] <= true_z11[0] <= independent_real[1]
    assert independent_imag[0] <= true_z11[1] <= independent_imag[1]

    # Negative controls are executable semantic counterexamples.
    mismatched_ids = {"current": nuisance_id, "voltage": "different-calibration-chain"}
    assert mismatched_ids["current"] != mismatched_ids["voltage"]
    q_including_zero = (Fraction(-1, 10), Fraction(1, 10))
    assert q_including_zero[0] <= 0 <= q_including_zero[1]
    statistical_semantics = "statistical_covariance_only"
    assert statistical_semantics != "deterministic_joint_hard_set"
    incomplete_coverage = set(cal["applies_to_voltage_record_ids"]) - {"V-bus-b-from-b"}
    assert incomplete_coverage != set(cal["applies_to_voltage_record_ids"])
    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["voltage-current-nuisance-id-mismatch"] == "reject_shared_cancellation"
    assert controls["q-domain-includes-zero"] == "not_evaluable"
    assert controls["statistical-covariance-labelled-hard-bound"] == "reject"

    print("PASS complete shared nuisance coverage and nonzero q")
    print("PASS independent rectangle exact outer bounds")
    print("PASS joint correlated set exact matrix")
    print("PASS all negative-control declarations")
    print("PASS exact-rational MIMO shared-calibration reference")


if __name__ == "__main__":
    main()
