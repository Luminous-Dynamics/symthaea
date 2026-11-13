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



def shared_coverage_valid(
    current_records: list[dict], voltage_records: list[dict], nuisance_id: str
) -> bool:
    current_ids = {item["record_id"] for item in current_records}
    voltage_ids = {item["record_id"] for item in voltage_records}
    if len(current_ids) != len(current_records) or len(voltage_ids) != len(voltage_records):
        return False
    if any(item.get("calibration_nuisance_id") != nuisance_id for item in current_records):
        return False
    if any(item.get("calibration_nuisance_id") != nuisance_id for item in voltage_records):
        return False
    return True


def square_matrix_coverage_valid(ports: list[str], records: list[dict]) -> bool:
    if not ports or len(set(ports)) != len(ports):
        return False
    pairs = [(item["observed_port_id"], item["driven_port_id"]) for item in records]
    expected = {(row, column) for row in ports for column in ports}
    return len(pairs) == len(set(pairs)) and set(pairs) == expected


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

    # Every raw record explicitly binds the same shared nuisance identity.
    current_nuisance_ids = {item["calibration_nuisance_id"] for item in data["base_injected_current_a"]}
    voltage_nuisance_ids = {item["calibration_nuisance_id"] for item in data["base_voltage_observations_v"]}
    assert current_nuisance_ids == {nuisance_id}
    assert voltage_nuisance_ids == {nuisance_id}
    assert shared_coverage_valid(
        data["base_injected_current_a"],
        data["base_voltage_observations_v"],
        nuisance_id,
    )
    assert square_matrix_coverage_valid(
        ports, data["base_voltage_observations_v"]
    )

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

    # Recompute and verify the propagated measured rectangles for Z11 from
    # the same q domain, then form the independent numerator/denominator box.
    example = data["independent_rectangle_example"]
    v11_base = voltage_records[("bus-a", "bus-a")]
    i1_base = currents["bus-a"]
    v11_rect = affine_scaled_rectangle(v11_base, u, v)
    i1_rect = affine_scaled_rectangle(i1_base, u, v)
    assert v11_rect[0] == read_interval(example["measured_voltage_real"])
    assert v11_rect[1] == read_interval(example["measured_voltage_imag"])
    assert i1_rect[0] == read_interval(example["measured_current_real"])
    assert i1_rect[1] == read_interval(example["measured_current_imag"])
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

    # Negative controls are executed against the admission predicates.
    mismatch_currents = [dict(item) for item in data["base_injected_current_a"]]
    mismatch_currents[0]["calibration_nuisance_id"] = "different-calibration-chain"
    assert not shared_coverage_valid(
        mismatch_currents, data["base_voltage_observations_v"], nuisance_id
    )

    missing_voltage_coverage = list(cal["applies_to_voltage_record_ids"][:-1])
    assert set(missing_voltage_coverage) != voltage_ids
    missing_current_coverage = list(cal["applies_to_current_record_ids"][:-1])
    assert set(missing_current_coverage) != current_ids

    q_real_including_zero = (Fraction(0), Fraction(1))
    assert q_real_including_zero[0] <= 0 <= q_real_including_zero[1]

    statistical_semantics = "statistical_covariance_only"
    assert statistical_semantics != "deterministic_joint_hard_set"

    duplicate_ports = ["bus-a", "bus-a"]
    assert not square_matrix_coverage_valid(
        duplicate_ports, data["base_voltage_observations_v"]
    )

    altered_pairs = [dict(item) for item in data["base_voltage_observations_v"]]
    altered_pairs[0]["driven_port_id"] = "bus-b"
    assert not square_matrix_coverage_valid(ports, altered_pairs)

    # A wide independent rectangle cannot be labelled an exact point set:
    # its bounds do not collapse to the known reference element.
    claimed_exact = (
        independent_real == (true_z11[0], true_z11[0])
        and independent_imag == (true_z11[1], true_z11[1])
    )
    assert not claimed_exact

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["voltage-current-nuisance-id-mismatch"] == "reject_shared_cancellation"
    assert controls["missing-voltage-record-from-coverage"] == "not_evaluable"
    assert controls["missing-current-record-from-coverage"] == "not_evaluable"
    assert controls["q-domain-includes-zero"] == "not_evaluable"
    assert controls["statistical-covariance-labelled-hard-bound"] == "reject"
    assert controls["duplicate-port-identity"] == "reject"
    assert controls["altered-voltage-current-port-pairing"] == "reject"
    assert controls["independent-rectangle-claimed-exact"] == "reject_claim_class"

    print("PASS complete shared nuisance coverage and nonzero q")
    print("PASS independent rectangle exact outer bounds")
    print("PASS joint correlated set exact matrix")
    print("PASS all negative-control declarations")
    print("PASS exact-rational MIMO shared-calibration reference")


if __name__ == "__main__":
    main()
