#!/usr/bin/env python3
"""Exact-rational oracle for an isolated underdamped DC-link RLC discharge."""

from __future__ import annotations

import json
from fractions import Fraction
from math import factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-dc-capacitor-rlc-v0.1.json"
Interval = tuple[Fraction, Fraction]


def q(record: dict[str, str]) -> Fraction:
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def series_exp_neg(x: Fraction, order: int) -> Fraction:
    return sum(
        (Fraction((-1) ** k) * x**k / factorial(k) for k in range(order + 1)),
        Fraction(0),
    )


def series_sin(x: Fraction, order: int) -> Fraction:
    return sum(
        (Fraction((-1) ** k) * x ** (2 * k + 1) / factorial(2 * k + 1)
         for k in range(order + 1)),
        Fraction(0),
    )


def series_cos(x: Fraction, order: int) -> Fraction:
    return sum(
        (Fraction((-1) ** k) * x ** (2 * k) / factorial(2 * k)
         for k in range(order + 1)),
        Fraction(0),
    )


def alternating_bounds(x: Fraction, lower_order: int, upper_order: int, kind: str) -> Interval:
    if not (0 <= x <= 1):
        raise ValueError("Taylor enclosure supports arguments only in [0,1]")
    if lower_order % 2 != 1 or upper_order % 2 != 0:
        raise ValueError("odd lower/even upper truncation required")
    functions = {
        "exp": series_exp_neg,
        "sin": series_sin,
        "cos": series_cos,
    }
    fn = functions[kind]
    lower = fn(x, lower_order)
    upper = fn(x, upper_order)
    if lower > upper:
        raise ArithmeticError("invalid alternating-series interval")
    return lower, upper


def add(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Interval, b: Interval) -> Interval:
    return a[0] - b[1], a[1] - b[0]


def scale(a: Interval, c: Fraction) -> Interval:
    return (a[0] * c, a[1] * c) if c >= 0 else (a[1] * c, a[0] * c)


def mul(a: Interval, b: Interval) -> Interval:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(products), max(products)


def overlap(a: Interval, b: Interval) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def compute(v0: Fraction, resistance: Fraction, inductance: Fraction,
            capacitance: Fraction, omega_d: Fraction, evaluation_time: Fraction):
    if min(v0, resistance, inductance, capacitance, omega_d, evaluation_time) <= 0:
        raise ValueError("all RLC inputs and evaluation time must be positive")
    alpha = resistance / (2 * inductance)
    if inductance * capacitance * (omega_d**2 + alpha**2) != 1:
        raise ValueError("omegaD is inconsistent with the declared underdamped RLC parameters")
    if alpha * evaluation_time > Fraction(1, 2):
        raise ValueError("alpha*T exceeds the frozen Taylor domain")
    if omega_d * evaluation_time > Fraction(1, 4):
        raise ValueError("omegaD*T exceeds the frozen Taylor domain")

    k_gain = v0 / (inductance * omega_d)
    exp_alpha = alternating_bounds(alpha * evaluation_time, 13, 12, "exp")
    exp_twice_alpha = alternating_bounds(2 * alpha * evaluation_time, 13, 12, "exp")
    sin_theta = alternating_bounds(omega_d * evaluation_time, 7, 6, "sin")
    cos_theta = alternating_bounds(omega_d * evaluation_time, 7, 6, "cos")
    sin_double = alternating_bounds(2 * omega_d * evaluation_time, 7, 6, "sin")
    cos_double = alternating_bounds(2 * omega_d * evaluation_time, 7, 6, "cos")

    current = scale(mul(exp_alpha, sin_theta), k_gain)
    cap_voltage = scale(
        mul(exp_alpha, add(cos_theta, scale(sin_theta, alpha / omega_d))),
        v0,
    )

    a = 2 * alpha
    b = 2 * omega_d
    integral_a = scale(sub((Fraction(1), Fraction(1)), exp_twice_alpha), 1 / a)
    integral_inner = add(scale(cos_double, -a), scale(sin_double, b))
    integral_b = scale(
        add((Fraction(a), Fraction(a)), mul(exp_twice_alpha, integral_inner)),
        1 / (a * a + b * b),
    )
    i2t = scale(sub(integral_a, integral_b), k_gain * k_gain / 2)
    if i2t[0] < 0 or i2t[1] < i2t[0]:
        raise ArithmeticError("invalid I2t interval")

    resistor_energy = scale(i2t, resistance)
    inductor_energy = scale((current[0] ** 2, current[1] ** 2), inductance / 2)
    capacitor_energy = scale(
        (cap_voltage[0] ** 2, cap_voltage[1] ** 2),
        capacitance / 2,
    )
    initial_energy = (
        capacitance * v0 * v0 / 2,
        capacitance * v0 * v0 / 2,
    )
    accounted = add(add(resistor_energy, inductor_energy), capacitor_energy)
    if not overlap(initial_energy, accounted):
        raise ArithmeticError("initial/final energy intervals do not overlap")

    return {
        "alpha": alpha,
        "current": current,
        "capacitor_voltage": cap_voltage,
        "i2t": i2t,
        "resistor_energy": resistor_energy,
        "inductor_energy": inductor_energy,
        "capacitor_energy": capacitor_energy,
        "initial_energy": initial_energy,
        "exp_alpha": exp_alpha,
        "exp_twice_alpha": exp_twice_alpha,
        "sin_theta": sin_theta,
        "cos_theta": cos_theta,
        "sin_double": sin_double,
        "cos_double": cos_double,
    }


def expected_pair(data: dict) -> Interval:
    return q(data["lower"]), q(data["upper"])


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.dc_capacitor_rlc_discharge.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_analytical_reference"
    m = data["model"]
    assert q(m["initial_current_a"]) == 0
    assert m["source_contribution"] == "none_isolated_capacitor_discharge_reference"

    v0 = q(m["initial_capacitor_voltage_v"])
    resistance = q(m["series_resistance_ohm"])
    inductance = q(m["series_inductance_h"])
    capacitance = q(m["capacitance_f"])
    omega_d = q(m["omega_d_rad_s"])
    evaluation_time = q(m["evaluation_time_s"])

    actual = compute(v0, resistance, inductance, capacitance, omega_d, evaluation_time)
    assert actual["alpha"] == q(m["alpha_per_s"])
    assert omega_d**2 + actual["alpha"]**2 == q(m["omega0_rad_s"])**2
    assert actual["initial_energy"] == expected_pair(data["expected_enclosures"]["initial_capacitor_energy_j"])

    bounds = data["taylor_bounds"]
    for field, key in (
        ("exp_alpha", "exp_minus_alpha_t"),
        ("exp_twice_alpha", "exp_minus_two_alpha_t"),
        ("sin_theta", "sin_omega_t"),
        ("cos_theta", "cos_omega_t"),
        ("sin_double", "sin_two_omega_t"),
        ("cos_double", "cos_two_omega_t"),
    ):
        assert actual[field] == (
            q(bounds[key]["lower"]),
            q(bounds[key]["upper"]),
        )

    expected = data["expected_enclosures"]
    result_pairs = {
        "current_at_t_a": actual["current"],
        "capacitor_voltage_at_t_v": actual["capacitor_voltage"],
        "branch_i2t_a2s": actual["i2t"],
        "resistor_energy_j": actual["resistor_energy"],
        "inductor_energy_j": actual["inductor_energy"],
        "remaining_capacitor_energy_j": actual["capacitor_energy"],
        "initial_capacitor_energy_j": actual["initial_energy"],
    }
    for name, pair in result_pairs.items():
        assert pair == expected_pair(expected[name]), (name, pair, expected_pair(expected[name]))
        print(f"PASS {name}: [{float(pair[0]):.12f}, {float(pair[1]):.12f}]")

    # Exact amplitude scaling: current and voltage scale linearly; energies
    # and I2t scale quadratically when V0 is doubled.
    doubled = compute(2 * v0, resistance, inductance, capacitance, omega_d, evaluation_time)
    assert doubled["current"] == scale(actual["current"], 2)
    assert doubled["capacitor_voltage"] == scale(actual["capacitor_voltage"], 2)
    assert doubled["i2t"] == scale(actual["i2t"], 4)

    # Shorter time in the initial rising-current interval must have lower
    # current and accumulated I2t for this fixed fixture.
    shorter = compute(v0, resistance, inductance, capacitance, omega_d, evaluation_time / 2)
    assert shorter["current"][1] < actual["current"][0]
    assert shorter["i2t"][1] < actual["i2t"][0]

    invalid = [
        (Fraction(0), resistance, inductance, capacitance, omega_d, evaluation_time),
        (v0, Fraction(0), inductance, capacitance, omega_d, evaluation_time),
        (v0, resistance, Fraction(0), capacitance, omega_d, evaluation_time),
        (v0, resistance, inductance, Fraction(0), omega_d, evaluation_time),
        (v0, resistance, inductance, capacitance, Fraction(0), evaluation_time),
        (v0, resistance, inductance, capacitance, omega_d, Fraction(0)),
        (v0, resistance, inductance, capacitance, Fraction(1), evaluation_time),
        (v0, resistance, inductance, capacitance, omega_d, evaluation_time * 2),
    ]
    for values in invalid:
        try:
            compute(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid RLC input was accepted: {values}")

    assert len(data["negative_controls"]) == 8
    print("PASS energy-conservation interval overlap")
    print("PASS exact amplitude scaling and initial-time monotonicity")
    print("PASS underdamped / numerical-domain negative controls")
    print("PASS exact-rational capacitor RLC reference")


if __name__ == "__main__":
    main()
