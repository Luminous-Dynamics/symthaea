#!/usr/bin/env python3
"""Exact-rational oracle for state-continuous ideal source-contactor opening."""

from __future__ import annotations

import json
from fractions import Fraction
from math import factorial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-dc-contactor-event-state-v0.1.json"
Interval = tuple[Fraction, Fraction]


def q(item: dict[str, str]) -> Fraction:
    return Fraction(int(item["numerator"]), int(item["denominator"]))


def read_pair(item: dict) -> Interval:
    return q(item["lower"]), q(item["upper"])


def exp_partial(x: Fraction, n: int) -> Fraction:
    return sum((Fraction((-1) ** k, factorial(k)) * x**k for k in range(n + 1)), Fraction(0))


def exp_bounds(x: Fraction, lower_n: int, upper_n: int) -> Interval:
    if not (0 <= x <= 1) or lower_n % 2 != 1 or upper_n % 2 != 0:
        raise ValueError("unsupported exponential Taylor domain/parity")
    lo, hi = exp_partial(x, lower_n), exp_partial(x, upper_n)
    if lo > hi:
        raise ArithmeticError("reversed exponential enclosure")
    return lo, hi


def add(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def mul(a: Interval, b: Interval) -> Interval:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(products), max(products)


def scale(a: Interval, c: Fraction) -> Interval:
    return (a[0] * c, a[1] * c) if c >= 0 else (a[1] * c, a[0] * c)


def square_nonnegative(a: Interval) -> Interval:
    if a[0] < 0:
        raise ValueError("expected nonnegative state interval")
    return a[0] * a[0], a[1] * a[1]


def overlap(a: Interval, b: Interval) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema"] == "luminous.airroad.dc_fault_contactor_event_state.v0_1"
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_rational_analytical_reference"

    pre = data["pre_event_reference"]
    event = data["event"]
    post = data["post_event_mode"]
    expected = data["expected_enclosures"]

    assert pre["event_id"] == event["event_id"] if "event_id" in event else True
    assert pre["configuration_digest"] == event["configuration_digest"]
    assert pre["topology_digest"] == event["topology_before_digest"]
    assert event["topology_before_digest"] != event["topology_after_digest"]
    assert event["source_isolated_after_event"] is True
    assert event["reset_map"] == "identity_on_inductor_current_and_capacitor_voltage"
    assert event["inductor_current_continuity"] is True
    assert event["capacitor_voltage_continuity"] is True
    assert event["event_time_provenance"] == "synthetic_fixture_only_not_device_clearing_evidence"

    path = event["commutation_path"]
    assert path["path_id"].strip()
    assert path["closes_local_inductor_current_loop"] is True
    assert event["capacitor_discharge_path"]["remains_connected_after_source_contactor_open"] is True

    # Exact pre-event state is taken from the independently qualified coupled
    # source/capacitor reference and is frozen in this fixture.
    i_event = read_pair(pre["inductor_current_a"])
    v_event = read_pair(pre["capacitor_voltage_v"])
    assert i_event == read_pair({
        "lower": {"numerator": "35417676570294412301827991287871", "denominator": "2238854161337196046103064084480"},
        "upper": {"numerator": "133943213211295239456991292377272929", "denominator": "8466939373784305047080678719488000"},
    })
    assert v_event == read_pair({
        "lower": {"numerator": "727353481508308483417434796794934193", "denominator": "1552272218527122591964791098572800"},
        "upper": {"numerator": "3524866871924879577457283825333576573", "denominator": "7522549982092978714906295323852800"},
    })

    delta_t = q(post["elapsed_time_s"])
    L = q(post["inductor_h"])
    C = q(post["capacitance_f"])
    r_comm = q(post["commutation_resistance_ohm"])
    r_fault = q(post["fault_resistance_ohm"])
    assert (delta_t, L, C, r_comm, r_fault) == (
        Fraction(1, 4), Fraction(1), Fraction(1), Fraction(1), Fraction(1)
    )
    k_l = r_comm / L
    k_c = Fraction(1, 1) / (r_fault * C)
    assert k_l == q(post["inductor_decay_rate_per_s"])
    assert k_c == q(post["capacitor_decay_rate_per_s"])

    bounds = data["taylor_bounds"]
    e_l = exp_bounds(k_l * delta_t,
                     bounds["exp_minus_one_quarter"]["lower_order"],
                     bounds["exp_minus_one_quarter"]["upper_order"])
    e_c = exp_bounds(k_c * delta_t,
                     bounds["exp_minus_one_quarter"]["lower_order"],
                     bounds["exp_minus_one_quarter"]["upper_order"])
    e_l2 = exp_bounds(2 * k_l * delta_t,
                      bounds["exp_minus_one_half"]["lower_order"],
                      bounds["exp_minus_one_half"]["upper_order"])
    e_c2 = exp_bounds(2 * k_c * delta_t,
                      bounds["exp_minus_one_half"]["lower_order"],
                      bounds["exp_minus_one_half"]["upper_order"])

    # Identity reset: same exact interval object on both sides of the event.
    i_after_reset = i_event
    v_after_reset = v_event
    assert i_after_reset == i_event and v_after_reset == v_event

    # Post-event decay through the explicitly declared independent paths.
    i_post = mul(i_after_reset, e_l)
    v_post = mul(v_after_reset, e_c)
    assert i_post == read_pair(expected["post_event_inductor_current_a"])
    assert v_post == read_pair(expected["post_event_capacitor_voltage_v"])

    source_current_after_event = (Fraction(0), Fraction(0))
    assert source_current_after_event == (0, 0)

    # Stored and dissipated energies on the two isolated post-event branches.
    e_l_initial = scale(square_nonnegative(i_event), L / 2)
    e_c_initial = scale(square_nonnegative(v_event), C / 2)
    e_l_final = scale(square_nonnegative(i_post), L / 2)
    e_c_final = scale(square_nonnegative(v_post), C / 2)

    one = (Fraction(1), Fraction(1))
    decay_l = (Fraction(1) - e_l2[1], Fraction(1) - e_l2[0])
    decay_c = (Fraction(1) - e_c2[1], Fraction(1) - e_c2[0])
    e_comm = scale(mul(square_nonnegative(i_event), decay_l), r_comm / (2 * k_l))
    e_fault = scale(mul(square_nonnegative(v_event), decay_c), Fraction(1, 1) / (2 * k_c * r_fault))
    e_initial_total = add(e_l_initial, e_c_initial)
    e_final_plus_loss = add(add(e_l_final, e_c_final), add(e_comm, e_fault))
    assert overlap(e_initial_total, e_final_plus_loss)

    # Silent reset controls: zeroing either state erases its stored energy.
    zero_state = (Fraction(0), Fraction(0))
    reset_i_initial = add(scale(square_nonnegative(zero_state), L / 2), e_c_initial)
    reset_i_accounted = add(add(e_c_final, e_comm), e_fault)
    assert not overlap(reset_i_initial, reset_i_accounted)
    reset_v_initial = add(e_l_initial, scale(square_nonnegative(zero_state), C / 2))
    reset_v_accounted = add(add(e_l_final, e_comm), e_fault)
    assert not overlap(reset_v_initial, reset_v_accounted)

    controls = {item["id"]: item["expected"] for item in data["negative_controls"]}
    assert controls["silent-inductor-current-reset"] == "reject_energy_loss"
    assert controls["silent-capacitor-voltage-reset"] == "reject_energy_loss"
    assert controls["missing-commutation-path"] == "not_evaluable"
    assert controls["source-remains-connected-after-open-event"] == "reject_topology"
    assert controls["pre-post-configuration-identity-mismatch"] == "not_evaluable"
    assert controls["nominal-delay-promoted-to-device-clearing-evidence"] == "reject"
    assert controls["event-timebase-mismatch"] == "not_evaluable"
    assert controls["energy-balance-fails"] == "not_established"

    print(f"PASS state continuity: iL event in [{float(i_event[0]):.9f}, {float(i_event[1]):.9f}] A")
    print(f"PASS state continuity: vC event in [{float(v_event[0]):.9f}, {float(v_event[1]):.9f}] V")
    print(f"PASS post-event freewheel current: [{float(i_post[0]):.9f}, {float(i_post[1]):.9f}] A")
    print(f"PASS post-event capacitor voltage: [{float(v_post[0]):.9f}, {float(v_post[1]):.9f}] V")
    print("PASS source isolation and declared commutation paths")
    print("PASS initial/final/resistive energy enclosure overlap")
    print("PASS silent state-reset negative controls")
    print("PASS exact-rational contactor event reference")


if __name__ == "__main__":
    main()
