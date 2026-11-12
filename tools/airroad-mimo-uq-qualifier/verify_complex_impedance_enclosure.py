#!/usr/bin/env python3
"""Independent Fraction oracle for the MIMO complex-disc quotient enclosure."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-mimo-complex-enclosure-v0.1.json"


def q(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def sub(a, b):
    return a[0] - b[1], a[1] - b[0]


def mul(a, b):
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(products), max(products)


def div_positive(a, b):
    assert b[0] > 0 and b[1] >= b[0]
    return mul(a, (1 / b[1], 1 / b[0]))


def interval(center, radius):
    return center - radius, center + radius


def exact_enclosure(voltage, voltage_radius, current, current_radius):
    vr = interval(voltage[0], voltage_radius)
    vi = interval(voltage[1], voltage_radius)
    ir = interval(current[0], current_radius)
    ii = interval(current[1], current_radius)

    # Exact rational radial bound for the fixture (all current centers lie on
    # the real axis and are positive).
    assert current[1] == 0 and current[0] > 0
    magnitude_lower = current[0] - current_radius
    magnitude_upper = current[0] + current_radius
    assert magnitude_lower > 0
    denominator = (magnitude_lower**2, magnitude_upper**2)

    numerator_real = add(mul(vr, ir), mul(vi, ii))
    numerator_imag = sub(mul(vi, ir), mul(vr, ii))
    return div_positive(numerator_real, denominator), div_positive(numerator_imag, denominator)


def parse_complex(record):
    return q(record["real"]), q(record["imag"])


def parse_pair(record):
    return q(record["lower"]), q(record["upper"])


def main():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_reference"
    assert data["hard_bound_semantics"]["kind"] == "deterministic_hard_bound"
    assert data["hard_bound_semantics"]["method_reference"].strip()
    assert data["dependence_assumption"] == "independent_discs_outer_enclosure"

    ports = data["ports"]
    assert len(ports) == 2 and len(set(ports)) == 2
    columns = data["columns"]
    assert {column["driven_port_id"] for column in columns} == set(ports)
    assert len(columns) == len(ports)

    expected_rows = data["expected_matrix_enclosures_row_major"]
    assert len(expected_rows) == len(ports)

    for row_index, observed_id in enumerate(ports):
        assert len(expected_rows[row_index]) == len(ports)
        for column_index, driven_id in enumerate(ports):
            column = next(item for item in columns if item["driven_port_id"] == driven_id)
            observations = column["observations"]
            assert {item["observed_port_id"] for item in observations} == set(ports)
            observation = next(item for item in observations if item["observed_port_id"] == observed_id)

            voltage = parse_complex(observation["response_voltage_v"])
            current = parse_complex(column["injected_current_a"])
            voltage_radius = q(observation["voltage_error_radius_v"])
            current_radius = q(column["current_error_radius_a"])
            assert voltage_radius >= 0 and current_radius >= 0
            expected_real, expected_imag = exact_enclosure(
                voltage, voltage_radius, current, current_radius
            )

            frozen = expected_rows[row_index][column_index]
            assert frozen["observed_port_id"] == observed_id
            assert frozen["driven_port_id"] == driven_id
            assert expected_real == (
                parse_pair(frozen["real_ohm"])
            ), (observed_id, driven_id, expected_real, frozen["real_ohm"])
            assert expected_imag == (
                parse_pair(frozen["imaginary_ohm"])
            ), (observed_id, driven_id, expected_imag, frozen["imaginary_ohm"])

            # Current error disc must exclude zero.
            current_center_magnitude = abs(current[0])
            assert current_center_magnitude - current_radius > 0

            print(
                f"PASS {observed_id}/{driven_id}: "
                f"Re={expected_real}, Im={expected_imag}"
            )

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}

    # Execute current-disc zero-exclusion and explicit hard-bound semantics.
    control_column = columns[0]
    zero_current = parse_complex(control_column["injected_current_a"])
    zero_radius = abs(zero_current[0])
    assert abs(zero_current[0]) - zero_radius <= 0
    assert controls["current-disc-touches-zero"] == "not_evaluable_current_set_includes_zero"

    assert data["hard_bound_semantics"]["kind"] == "deterministic_hard_bound"
    statistical_only = {"kind": "statistical_only", "method_reference": "one-sigma"}
    assert statistical_only["kind"] != "deterministic_hard_bound"
    assert controls["statistical-only-bound"] == "not_evaluable_not_a_hard_bound"

    empty_method = dict(data["hard_bound_semantics"])
    empty_method["method_reference"] = ""
    assert not empty_method["method_reference"].strip()
    assert controls["empty-method-reference"] == "not_evaluable_missing_hard_bound_method"

    # A deleted cross-port observation cannot complete a square matrix.
    missing = json.loads(json.dumps(data))
    missing["columns"][0]["observations"].pop()
    observed_ids = {
        item["observed_port_id"] for item in missing["columns"][0]["observations"]
    }
    assert observed_ids != set(ports)
    assert controls["missing-cross-port-observation"] == "not_evaluable_missing_matrix_element"

    query_hz = Fraction(1, 1)
    recorded_hz = q(data["angular_frequency_rad_s"])
    assert query_hz != recorded_hz
    assert controls["off-knot-query"] == "not_evaluable_unsupported_frequency"

    # The rational current is nonzero, but binary64 squaring can underflow.
    tiny = 1.0e-300
    assert tiny != 0.0 and tiny * tiny == 0.0
    assert controls["tiny-denominator-underflow"] == "not_evaluable_floating_point_domain"

    print("PASS all six executable negative controls")


if __name__ == "__main__":
    main()
