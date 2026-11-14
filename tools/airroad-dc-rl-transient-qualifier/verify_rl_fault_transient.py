#!/usr/bin/env python3
"""Exact-rational oracle for AIRROAD's single-path RL fault transient."""

from __future__ import annotations

import json
from fractions import Fraction
from math import factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-dc-rl-fault-transient-v0.1.json"


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def encode(value: Fraction) -> dict[str, str]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def series_exp_neg(x: Fraction, order: int) -> Fraction:
    return sum(
        (Fraction((-1) ** k) * x**k / factorial(k) for k in range(order + 1)),
        Fraction(0),
    )


def exp_neg_bounds(x: Fraction, lower_order: int, upper_order: int) -> tuple[Fraction, Fraction]:
    if not (0 <= x <= 1):
        raise ValueError("alternating Taylor contract requires 0 <= x <= 1")
    if lower_order % 2 != 1 or upper_order % 2 != 0 or upper_order != lower_order - 1 and upper_order != lower_order + 1:
        raise ValueError("expected adjacent odd-lower / even-upper Taylor orders")
    lower = series_exp_neg(x, lower_order)
    upper = series_exp_neg(x, upper_order)
    if lower > upper:
        raise ArithmeticError("invalid alternating-series bounds")
    return lower, upper


def pair(record: dict[str, dict[str, str]]) -> tuple[Fraction, Fraction]:
    return q(record["lower"]), q(record["upper"])


def compute(vs: Fraction, rs: Fraction, rf: Fraction, inductance: Fraction, tc: Fraction):
    if min(vs, rs, rf, inductance, tc) <= 0:
        raise ValueError("Vs, Rs, Rf, L and tc must be positive")
    resistance = rs + rf
    tau = inductance / resistance
    x = tc / tau
    if not (0 < x <= Fraction(1, 2)):
        raise ValueError("V0.1 admits only 0 < tc/tau <= 1/2")

    exp_x_lo, exp_x_hi = exp_neg_bounds(x, 13, 12)
    exp_2x_lo, exp_2x_hi = exp_neg_bounds(2 * x, 13, 14)

    i_inf = vs / resistance
    i_clear = (i_inf * (1 - exp_x_hi), i_inf * (1 - exp_x_lo))

    constant = tc - Fraction(3, 2) * tau
    q_lo = i_inf**2 * (constant + 2 * tau * exp_x_lo - tau * Fraction(1, 2) * exp_2x_hi)
    q_hi = i_inf**2 * (constant + 2 * tau * exp_x_hi - tau * Fraction(1, 2) * exp_2x_lo)
    if q_lo < 0 or q_hi < q_lo:
        raise ArithmeticError("invalid I2t enclosure")
    i2t = (q_lo, q_hi)

    fault_energy = (rf * q_lo, rf * q_hi)
    source_resistor_energy = (rs * q_lo, rs * q_hi)
    inductor_energy = (
        Fraction(1, 2) * inductance * i_clear[0]**2,
        Fraction(1, 2) * inductance * i_clear[1]**2,
    )
    source_energy = (
        vs * i_inf * (tc - tau + tau * exp_x_lo),
        vs * i_inf * (tc - tau + tau * exp_x_hi),
    )
    return {
        "tau": tau,
        "x": x,
        "i_inf": i_inf,
        "exp_x": (exp_x_lo, exp_x_hi),
        "exp_2x": (exp_2x_lo, exp_2x_hi),
        "clearing_current": i_clear,
        "i2t": i2t,
        "fault_energy": fault_energy,
        "source_resistor_energy": source_resistor_energy,
        "inductor_energy": inductor_energy,
        "source_energy": source_energy,
    }


def overlap(left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.dc_rl_fault_transient.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_analytical_reference"

    m = data["model"]
    vs = q(m["source_voltage_v"])
    rs = q(m["source_feeder_resistance_ohm"])
    rf = q(m["fault_resistance_ohm"])
    resistance = q(m["total_resistance_ohm"])
    inductance = q(m["series_inductance_h"])
    initial_current = q(m["initial_current_a"])
    tc = q(m["clearing_time_s"])

    assert initial_current == 0
    assert rs + rf == resistance
    tau = inductance / resistance
    assert tau == q(m["time_constant_s"])
    assert tc / tau == q(m["normalized_clearing_time"])
    assert vs / resistance == q(m["prospective_steady_current_a"])
    assert m["clearing_time_provenance"] == "synthetic_fixture_only_not_protection_evidence"

    computed = compute(vs, rs, rf, inductance, tc)

    eb = data["exponential_bounds"]
    assert computed["exp_x"] == (
        q(eb["exp_minus_half"]["lower"]), q(eb["exp_minus_half"]["upper"])
    )
    assert computed["exp_2x"] == (
        q(eb["exp_minus_one"]["lower"]), q(eb["exp_minus_one"]["upper"])
    )

    expected = data["expected_enclosures"]
    actual_pairs = {
        "clearing_current_a": computed["clearing_current"],
        "branch_i2t_a2s": computed["i2t"],
        "fault_resistor_energy_j": computed["fault_energy"],
        "source_feeder_resistor_energy_j": computed["source_resistor_energy"],
        "inductor_stored_energy_j": computed["inductor_energy"],
        "source_delivered_energy_j": computed["source_energy"],
    }
    for name, actual in actual_pairs.items():
        assert actual == pair(expected[name]), (name, actual, pair(expected[name]))
        print(f"PASS {name}: [{float(actual[0]):.12f}, {float(actual[1]):.12f}]")

    # Energy balance: source-delivered interval must overlap the independently
    # enclosed sum of the resistor losses and stored magnetic energy.
    resistive_total = (
        computed["i2t"][0] * resistance,
        computed["i2t"][1] * resistance,
    )
    total_accounted = (
        resistive_total[0] + computed["inductor_energy"][0],
        resistive_total[1] + computed["inductor_energy"][1],
    )
    assert overlap(computed["source_energy"], total_accounted)

    # Exact scaling: doubling Vs doubles current and quadruples I2t/energy.
    doubled = compute(2 * vs, rs, rf, inductance, tc)
    assert doubled["clearing_current"] == tuple(2 * x for x in computed["clearing_current"])
    assert doubled["i2t"] == tuple(4 * x for x in computed["i2t"])

    # At fixed R,L, a longer admitted clearing interval cannot lower the
    # monotone current or accumulated I2t.
    shorter = compute(vs, rs, rf, inductance, tc / 2)
    assert shorter["clearing_current"][1] < computed["clearing_current"][0]
    assert shorter["i2t"][1] < computed["i2t"][0]

    # Invalid domains are rejected.
    invalid_inputs = [
        (Fraction(0), rs, rf, inductance, tc),
        (vs, Fraction(0), rf, inductance, tc),
        (vs, rs, rf, Fraction(0), tc),
        (vs, rs, rf, inductance, Fraction(0)),
        (vs, rs, rf, inductance, tc * 2),
    ]
    for values in invalid_inputs:
        try:
            compute(*values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid input was admitted: {values}")

    assert all(
        item["expected"] in {
            "reject",
            "not_evaluable_for_protection_join",
            "not_evaluable",
        }
        for item in data["negative_controls"]
    )
    assert len(data["negative_controls"]) == 8

    print("PASS energy-balance interval overlap")
    print("PASS exact voltage scaling and clearing-time monotonicity")
    print("PASS all invalid-domain controls")
    print("PASS exact-rational RL transient reference")


if __name__ == "__main__":
    main()
