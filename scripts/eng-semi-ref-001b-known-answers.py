#!/usr/bin/env python3
"""Generate/verify ENG-SEMI-REF-001B high-precision known answers.

Independent of the Rust implementation. Uses Python Decimal with 80 digits and
exact SI constants. Synthetic parameters only; no TCAD, datasheet, or physical
measurement fitting.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
import hashlib
import json
from pathlib import Path

getcontext().prec = 80

K_B = Decimal("1.380649e-23")
Q_E = Decimal("1.602176634e-19")


def thermal_voltage(t_kelvin: str) -> Decimal:
    return K_B * Decimal(t_kelvin) / Q_E


def diode_current(t_kelvin: str, i_s: str, n: str, voltage: str) -> Decimal:
    vt = thermal_voltage(t_kelvin)
    exponent = Decimal(voltage) / (Decimal(n) * vt)
    return Decimal(i_s) * (exponent.exp() - Decimal(1))


def diode_conductance(t_kelvin: str, i_s: str, n: str, voltage: str) -> Decimal:
    scale = Decimal(n) * thermal_voltage(t_kelvin)
    return Decimal(i_s) * (Decimal(voltage) / scale).exp() / scale


PROFILES = [
    {
        "id": "p300_n1_is1e-12",
        "temperature_kelvin": "300",
        "saturation_current_amps": "1e-12",
        "ideality_factor": "1",
        "voltage_min_volts": "-0.2",
        "voltage_max_volts": "0.8",
    },
    {
        "id": "p250_n1_is1e-12",
        "temperature_kelvin": "250",
        "saturation_current_amps": "1e-12",
        "ideality_factor": "1",
        "voltage_min_volts": "-0.2",
        "voltage_max_volts": "0.8",
    },
    {
        "id": "p350_n1_is1e-12",
        "temperature_kelvin": "350",
        "saturation_current_amps": "1e-12",
        "ideality_factor": "1",
        "voltage_min_volts": "-0.2",
        "voltage_max_volts": "0.8",
    },
    {
        "id": "p300_n1p5_is1e-12",
        "temperature_kelvin": "300",
        "saturation_current_amps": "1e-12",
        "ideality_factor": "1.5",
        "voltage_min_volts": "-0.2",
        "voltage_max_volts": "0.8",
    },
]


def numeric(case_id: str, operation: str, expected: Decimal, **fields: str) -> dict[str, str]:
    return {
        "id": case_id,
        "operation": operation,
        **fields,
        "expected": format(expected, "f"),
        "abs_tolerance": "1e-18",
        "rel_tolerance": "5e-13",
    }


def build_fixture() -> dict:
    cases: list[dict] = [
        numeric("thermal_250", "thermal_voltage", thermal_voltage("250"), temperature_kelvin="250"),
        numeric("thermal_300", "thermal_voltage", thermal_voltage("300"), temperature_kelvin="300"),
        numeric("thermal_350", "thermal_voltage", thermal_voltage("350"), temperature_kelvin="350"),
    ]

    p = PROFILES[0]
    for voltage in ["-0.2", "0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.8"]:
        cases.append(
            numeric(
                "current_" + voltage.replace("-", "m").replace(".", "p"),
                "current",
                diode_current(p["temperature_kelvin"], p["saturation_current_amps"], p["ideality_factor"], voltage),
                profile_id=p["id"],
                voltage_volts=voltage,
            )
        )

    for voltage in ["0", "0.2", "0.5"]:
        cases.append(
            numeric(
                "conductance_" + voltage.replace(".", "p"),
                "conductance",
                diode_conductance(p["temperature_kelvin"], p["saturation_current_amps"], p["ideality_factor"], voltage),
                profile_id=p["id"],
                voltage_volts=voltage,
            )
        )

    p250, p350, p15 = PROFILES[1], PROFILES[2], PROFILES[3]
    cases.extend(
        [
            numeric("temp250_current_0p3", "current", diode_current(p250["temperature_kelvin"], p250["saturation_current_amps"], p250["ideality_factor"], "0.3"), profile_id=p250["id"], voltage_volts="0.3"),
            numeric("temp350_current_0p3", "current", diode_current(p350["temperature_kelvin"], p350["saturation_current_amps"], p350["ideality_factor"], "0.3"), profile_id=p350["id"], voltage_volts="0.3"),
            numeric("ideality1p5_current_0p4", "current", diode_current(p15["temperature_kelvin"], p15["saturation_current_amps"], p15["ideality_factor"], "0.4"), profile_id=p15["id"], voltage_volts="0.4"),
            {"id": "thermal_zero_temp", "operation": "thermal_voltage", "temperature_kelvin": "0", "expected_error": "NonPositiveTemperature"},
            {"id": "thermal_nan", "operation": "thermal_voltage", "temperature_kelvin": "NaN", "expected_error": "NonFiniteTemperature"},
            {"id": "current_out_of_domain", "operation": "current", "profile_id": "p300_n1_is1e-12", "voltage_volts": "0.9", "expected_error": "VoltageOutsideDomain"},
            {"id": "current_nan_voltage", "operation": "current", "profile_id": "p300_n1_is1e-12", "voltage_volts": "NaN", "expected_error": "NonFiniteVoltage"},
        ]
    )

    return {
        "schema": "eng-semi-ref-001b-known-answers-v1",
        "purpose": "Independent high-precision known-answer corpus for symthaea-semiconductor analytical diode V1. Synthetic parameters only; not fit to TCAD, datasheets, or physical measurements.",
        "generator": {
            "language": "Python decimal",
            "precision_digits": 80,
            "constants": {
                "boltzmann_j_per_k": "1.380649e-23",
                "elementary_charge_c": "1.602176634e-19",
            },
            "equations": {
                "thermal_voltage": "k_B*T/q",
                "current": "I_s*(exp(V/(n*V_T))-1)",
                "conductance": "I_s*exp(V/(n*V_T))/(n*V_T)",
            },
        },
        "profiles": PROFILES,
        "cases": cases,
    }


def canonical_bytes() -> bytes:
    return (json.dumps(build_fixture(), indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path, help="verify an existing fixture byte-for-byte")
    args = parser.parse_args()

    expected = canonical_bytes()
    digest = hashlib.sha256(expected).hexdigest()

    if args.check is None:
        print(expected.decode(), end="")
        return 0

    actual = args.check.read_bytes()
    if actual != expected:
        print(f"mismatch expected_sha256={digest} actual_sha256={hashlib.sha256(actual).hexdigest()}")
        return 1

    print(f"ok cases={len(build_fixture()['cases'])} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
