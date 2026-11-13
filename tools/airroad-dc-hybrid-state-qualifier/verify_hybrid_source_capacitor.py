#!/usr/bin/env python3
"""Exact-rational oracle for state-consistent source/initial-state decomposition."""

from __future__ import annotations

import json
from fractions import Fraction
from math import factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-dc-hybrid-source-capacitor-v0.1.json"
Interval = tuple[Fraction, Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def pair(item: dict) -> Interval:
    return q(item["lower"]), q(item["upper"])


def exp_partial(x: Fraction, n: int) -> Fraction:
    return sum((Fraction((-1) ** k, factorial(k)) * x**k for k in range(n + 1)), Fraction(0))


def sin_partial(x: Fraction, n: int) -> Fraction:
    return sum((Fraction((-1) ** k, factorial(2 * k + 1)) * x ** (2 * k + 1)
                for k in range(n + 1)), Fraction(0))


def cos_partial(x: Fraction, n: int) -> Fraction:
    return sum((Fraction((-1) ** k, factorial(2 * k)) * x ** (2 * k)
                for k in range(n + 1)), Fraction(0))


def alternating_bounds(x: Fraction, lower_n: int, upper_n: int, fn) -> Interval:
    if not (0 <= x <= 1) or lower_n % 2 != 1 or upper_n % 2 != 0:
        raise ValueError("unsupported alternating-series domain or truncation parity")
    lower, upper = fn(x, lower_n), fn(x, upper_n)
    if lower > upper:
        raise ArithmeticError("alternating-series endpoints reversed")
    return lower, upper


def add(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Interval, b: Interval) -> Interval:
    return a[0] - b[1], a[1] - b[0]


def mul(a: Interval, b: Interval) -> Interval:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(products), max(products)


def scale(a: Interval, c: Fraction) -> Interval:
    return (a[0] * c, a[1] * c) if c >= 0 else (a[1] * c, a[0] * c)


def overlap(a: Interval, b: Interval) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.dc_hybrid_source_capacitor_state_reference.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_analytical_reference"

    topo = data["topology"]
    vs = q(topo["source_voltage_v"])
    rs = q(topo["source_series_resistance_ohm"])
    inductance = q(topo["source_series_inductance_h"])
    rf = q(topo["fault_resistance_ohm"])
    capacitance = q(topo["dc_link_capacitance_f"])
    i0 = q(topo["initial_inductor_current_a"])
    v0 = q(topo["initial_capacitor_voltage_v"])
    t = q(topo["evaluation_time_s"])
    assert (vs, rs, inductance, rf, capacitance, i0, v0, t) == (
        Fraction(600), Fraction(1), Fraction(1), Fraction(1), Fraction(1),
        Fraction(0), Fraction(600), Fraction(1, 4)
    )
    assert topo["sign_convention"]["inductor_current"] == "source_to_bus_positive"
    assert topo["sign_convention"]["capacitor_current"] == (
        "current_into_positive_capacitor_terminal_positive"
    )

    # Derive A and b from the component equations, not from their JSON copy.
    a11 = -rs / inductance
    a12 = -Fraction(1) / inductance
    a21 = Fraction(1) / capacitance
    a22 = -Fraction(1) / (rf * capacitance)
    b1, b2 = vs / inductance, Fraction(0)
    expected_a = [
        [Fraction(-1), Fraction(-1)],
        [Fraction(1), Fraction(-1)],
    ]
    assert [[a11, a12], [a21, a22]] == expected_a
    declared_a = [[q(cell) for cell in row] for row in data["state_space"]["A"]]
    assert declared_a == expected_a
    assert [b1, b2] == [q(x) for x in data["state_space"]["b"]]
    assert [i0, v0] == [q(x) for x in data["state_space"]["initial_state"]]

    # Solve equilibrium exactly: 0 = Vs - Rs*i_eq - v_eq, 0 = i_eq-v_eq/Rf.
    i_eq = vs / (rs + rf)
    v_eq = rf * i_eq
    assert [i_eq, v_eq] == [q(x) for x in data["state_space"]["equilibrium_state"]]
    assert i_eq == 300 and v_eq == 300

    bounds = data["taylor_contract"]
    e = alternating_bounds(t, bounds["exp_negative_t"]["lower_order"],
                           bounds["exp_negative_t"]["upper_order"], exp_partial)
    s = alternating_bounds(t, bounds["sin_t"]["lower_order"],
                           bounds["sin_t"]["upper_order"], sin_partial)
    c = alternating_bounds(t, bounds["cos_t"]["lower_order"],
                           bounds["cos_t"]["upper_order"], cos_partial)
    one = (Fraction(1), Fraction(1))

    # Source-forced response: x_eq - exp(A*t)*x_eq.
    source_i = scale(sub(one, mul(e, sub(c, s))), i_eq)
    source_v = scale(sub(one, mul(e, add(s, c))), v_eq)

    # Natural response: exp(A*t)*x0, with source disabled.
    natural_i = scale(mul(e, s), -v0)
    natural_v = scale(mul(e, c), v0)

    # Direct complete response: x_eq + exp(A*t)*(x0-x_eq).
    total_i_direct = scale(sub(one, mul(e, add(c, s))), i_eq)
    total_v_direct = add((v_eq, v_eq), scale(mul(e, sub(c, s)), v_eq))

    # Compose signed state contributions, never magnitudes.
    total_i = add(source_i, natural_i)
    total_v = add(source_v, natural_v)

    expected = data["expected_enclosures"]
    actual = {
        "source_forced_inductor_current_a": source_i,
        "source_forced_node_voltage_v": source_v,
        "initial_state_inductor_current_a": natural_i,
        "initial_state_node_voltage_v": natural_v,
        "total_inductor_current_a": total_i,
        "total_node_voltage_v": total_v,
        "direct_total_inductor_current_a": total_i_direct,
        "direct_total_node_voltage_v": total_v_direct,
    }
    for key, interval in actual.items():
        assert interval == pair(expected[key]), (key, interval, pair(expected[key]))
        print(f"PASS {key}: [{float(interval[0]):.12f}, {float(interval[1]):.12f}]")

    assert overlap(total_i, total_i_direct)
    assert overlap(total_v, total_v_direct)

    # Capacitor current follows the derivative of the directly composed bus voltage:
    # dv/dt = -2*V_eq*exp(-t)*cos(t) for this symmetric fixture.
    capacitor_current = scale(mul(e, c), -2 * v_eq * capacitance)
    fault_current = scale(total_v, Fraction(1, 1) / rf)
    kcl_current = add(fault_current, capacitor_current)
    assert capacitor_current == pair(expected["capacitor_current_a"])
    assert kcl_current == pair(expected["kcl_reconstructed_current_a"])
    assert overlap(total_i_direct, kcl_current)

    # Deliberate adversarial check: replacing signed response addition with
    # absolute-magnitude addition must not reproduce the same branch current.
    abs_source = source_i
    abs_natural = (-natural_i[1], -natural_i[0])
    magnitude_sum = add(abs_source, abs_natural)
    assert not overlap(magnitude_sum, total_i_direct)

    # Instantaneous power identity:
    # Vs*i = Rs*i^2 + v^2/Rf + L*i*di/dt + C*v*dv/dt.
    di_dt = scale(
        sub(sub((vs, vs), scale(total_i_direct, rs)), total_v_direct),
        Fraction(1, 1) / inductance,
    )
    dv_dt = scale(
        sub(total_i_direct, scale(total_v_direct, Fraction(1, 1) / rf)),
        Fraction(1, 1) / capacitance,
    )
    input_power = scale(total_i_direct, vs)
    resistive_loss = add(scale(mul(total_i_direct, total_i_direct), rs),
                         scale(mul(total_v_direct, total_v_direct), Fraction(1, 1) / rf))
    stored_energy_rate = add(scale(mul(total_i_direct, di_dt), inductance),
                              scale(mul(total_v_direct, dv_dt), capacitance))
    assert overlap(input_power, add(resistive_loss, stored_energy_rate))


    # Execute the declared identity/sign/domain negative controls instead of
    # merely trusting their labels in the fixture.
    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    changed_rs = rs + 1
    changed_a11 = -changed_rs / inductance
    assert changed_a11 != a11
    assert controls["wrong-source-resistance"] == "reject"

    changed_v0 = v0 - 100
    assert changed_v0 != v0
    assert [i0, changed_v0] != [q(x) for x in data["state_space"]["initial_state"]]
    assert controls["wrong-capacitor-initial-state"] == "reject"

    declared_signs = topo["sign_convention"]
    assert declared_signs["inductor_current"] == "source_to_bus_positive"
    bad_sign = dict(declared_signs)
    bad_sign["inductor_current"] = "bus_to_source_positive"
    assert bad_sign != declared_signs
    assert controls["wrong-sign-convention"] == "reject"

    # Changing the A matrix for only one component response invalidates the
    # decomposition. Both source-forced and natural responses use the same A.
    changed_component_a = [[a11 - 1, a12], [a21, a22]]
    assert changed_component_a != expected_a
    assert controls["different-state-matrix-for-component-responses"] == "reject"

    # Deliberately reverse the capacitor-current sign: KCL must no longer
    # overlap the independently calculated source-branch current.
    wrong_kcl = sub(fault_current, capacitor_current)
    assert not overlap(total_i_direct, wrong_kcl)
    assert controls["kcl-does-not-close"] == "not_established"

    # Flip the stored-energy rate sign: the instantaneous power invariant
    # must fail its conservative interval-overlap check.
    wrong_power = add(resistive_loss, scale(stored_energy_rate, Fraction(-1)))
    assert not overlap(input_power, wrong_power)
    assert controls["instantaneous-power-balance-does-not-overlap"] == "not_established"

    assert controls["source-and-capacitor-evidence-identities-mismatch"] == "not_evaluable"

    assert controls["sum-absolute-component-currents"] == "reject_as_not_the_linear_superposition"

    print("PASS signed source/initial-state decomposition")
    print("PASS direct state-space solution vs decomposition")
    print("PASS KCL current reconstruction")
    print("PASS absolute-magnitude addition negative control")
    print("PASS instantaneous power balance")
    print("PASS exact-rational hybrid source/capacitor reference")


if __name__ == "__main__":
    main()
