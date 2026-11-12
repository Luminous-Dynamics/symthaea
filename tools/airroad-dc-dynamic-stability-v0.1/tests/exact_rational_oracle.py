#!/usr/bin/env python3
"""Independent exact-rational oracle for AIRROAD DC stability test vectors."""

from __future__ import annotations

import json
from fractions import Fraction
from math import isqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "docs/research/fixtures/airroad-dc-dynamic-stability-v0.1.json"


def rational(obj: dict[str, str]) -> Fraction:
    return Fraction(int(obj["numerator"]), int(obj["denominator"]))


def exact_sqrt(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("square root of negative rational")
    numerator = isqrt(value.numerator)
    denominator = isqrt(value.denominator)
    assert numerator * numerator == value.numerator, value
    assert denominator * denominator == value.denominator, value
    return Fraction(numerator, denominator)


def decimal_fraction(value: str) -> Fraction:
    return Fraction(value)


def verify_case(default_model: dict, case: dict) -> None:
    parameters = case.get("model_parameters", default_model["parameters"])
    vs = rational(parameters["Vs"]["exact_rational"])
    rs = rational(parameters["Rs"]["exact_rational"])
    capacitance = rational(parameters["C"]["exact_rational"])
    power = rational(case["P"]["exact_rational"])

    delta = vs * vs - 4 * rs * power
    assert delta == rational(case["delta"]["exact_rational"]), case["id"]

    critical_power = vs * vs / (4 * rs)
    loading_fraction = power / critical_power
    margin = delta / (vs * vs)
    assert loading_fraction == rational(case["power_loading_fraction"])
    assert margin == rational(case["normalized_static_fold_margin"])

    expected_roots = case["roots"]
    if delta < 0:
        assert expected_roots == [], case["id"]
        assert case["disposition"] == "no_equilibrium_non_pass"
    else:
        root_gap = exact_sqrt(delta)
        roots = (
            [(vs + root_gap) / 2, (vs - root_gap) / 2]
            if delta > 0 else [vs / 2]
        )
        assert len(roots) == len(expected_roots), case["id"]

        for index, (voltage, expected) in enumerate(zip(roots, expected_roots)):
            current = power / voltage
            eigenvalue = (1 / capacitance) * (
                -1 / rs + power / (voltage * voltage)
            )
            assert voltage == decimal_fraction(expected["voltage"]), case["id"]
            assert current == decimal_fraction(expected["current"]), case["id"]
            assert eigenvalue == rational(expected["eigenvalue"]), case["id"]
            assert (vs - voltage) / rs == current, case["id"]

        if delta == 0:
            assert case["disposition"] == "critical_fold_non_pass"
        else:
            assert case["disposition"] == "two_equilibria"

    # Recompute the phase-line sign directly from the ODE, not root labels.
    for probe in case["phase_line_probes"]:
        voltage = decimal_fraction(probe["voltage_v"])
        derivative = ((vs - voltage) / rs - power / voltage) / capacitance
        observed = (
            "increasing" if derivative > 0
            else "decreasing" if derivative < 0
            else "equilibrium"
        )
        assert observed == probe["expected_sign"], (
            case["id"], voltage, derivative, probe["expected_sign"]
        )


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_analytical_reference"
    cases = data["cases"]
    assert len(cases) == 5
    for case in cases:
        verify_case(data["model"], case)
    print(f"exact-rational oracle: {len(cases)} cases passed")


if __name__ == "__main__":
    main()
