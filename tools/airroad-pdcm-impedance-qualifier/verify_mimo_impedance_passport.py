#!/usr/bin/env python3
"""Independent exact-rational verifier for the MIMO impedance passport fixture."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-pdcm-mimo-impedance-passport-v0.1.json"


def q(obj):
    return Fraction(int(obj["numerator"]), int(obj["denominator"]))


def cdiv(a, b):
    den = b[0] * b[0] + b[1] * b[1]
    if den <= 0:
        raise AssertionError("zero complex excitation")
    return (
        (a[0] * b[0] + a[1] * b[1]) / den,
        (a[1] * b[0] - a[0] * b[1]) / den,
    )


def cp(obj):
    return q(obj["real"]), q(obj["imag"])


def main():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_reference"

    identity = data["sweep_identity"]
    for field in ("configuration_digest", "operating_point_digest", "sweep_digest"):
        assert identity[field].strip()
    assert identity["origin"]["kind"] == "analytical_reference"
    assert identity["origin"]["derivation_digest"].strip()

    ports = data["ports"]
    ids = [port["port_id"] for port in ports]
    assert len(ids) == len(set(ids)) == 2
    assert all(port["voltage_positive_reference"] for port in ports)
    assert all(port["current_positive_direction"] for port in ports)

    previous_frequency = None
    for slice_record in data["frequency_slices"]:
        omega = q(slice_record["angular_frequency_rad_s"])
        assert omega >= 0
        if previous_frequency is not None:
            assert omega > previous_frequency
        previous_frequency = omega

        columns = slice_record["columns"]
        assert len(columns) == len(ids)
        by_driven = {}
        for column in columns:
            driven = column["driven_port_id"]
            assert driven in ids and driven not in by_driven
            assert column["excitation_digest"].strip()
            assert column["column_payload_digest"].strip()
            injected = cp(column["injected_current_a"])
            assert injected != (Fraction(0), Fraction(0))
            assert q(column["absolute_error_bound_a"]) >= 0

            observed = {}
            for output in column["voltage_observations"]:
                observed_id = output["observed_port_id"]
                assert observed_id in ids and observed_id not in observed
                assert cp(output["response_voltage_v"])
                assert q(output["absolute_error_bound_v"]) >= 0
                observed[observed_id] = cp(output["response_voltage_v"])
            assert set(observed) == set(ids)
            by_driven[driven] = (injected, observed)

        assert set(by_driven) == set(ids)
        computed = []
        for observed_id in ids:
            row = []
            for driven_id in ids:
                injected, responses = by_driven[driven_id]
                row.append(cdiv(responses[observed_id], injected))
            computed.append(row)

        expected = [
            [cp(value) for value in row]
            for row in slice_record["expected_impedance_matrix_ohm"]
        ]
        assert computed == expected
        assert computed == [
            [(Fraction(2), Fraction(1)), (Fraction(1), Fraction(2))],
            [(Fraction(1, 2), Fraction(-1)), (Fraction(3), Fraction(-1))],
        ]

    # Structural negative-control expectations are normative, not inferred.
    controls = {control["id"]: control["expected"] for control in data["negative_controls"]}
    assert controls["missing-cross-port-observation"] == "not_evaluable_missing_matrix_element"
    assert controls["duplicate-driven-port"] == "not_evaluable_duplicate_input_column"
    assert controls["measured-without-calibration"] == "not_evaluable_missing_calibration"
    assert controls["off-knot-query"] == "not_evaluable_unsupported_frequency"
    assert controls["zero-excitation"] == "not_evaluable_invalid_excitation"

    print("PASS exact 2x2 matrix point estimates at both recorded knots")
    print("PASS complete excitation columns and output response rows")
    print("PASS port direction and operating/configuration identity fields")
    print("PASS uncertainty bounds retained and non-negative")
    print("PASS missing/duplicate/calibration/off-knot/zero-current negative controls")


if __name__ == "__main__":
    main()
