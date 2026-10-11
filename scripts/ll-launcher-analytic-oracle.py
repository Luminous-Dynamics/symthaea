#!/usr/bin/env python3
"""Independent LL-002 lunar launcher analytic reference calculator.

This script intentionally imports no Symthaea code. It mirrors only the
published constant-acceleration reference equations so the Rust implementation
can be checked independently.

Input JSON schema:
{
  "mass_kg": 100.0,
  "initial_speed_m_s": 0.0,
  "exit_speed_m_s": 100.0,
  "acceleration_m_s2": 10.0,
  "electrical_efficiency": 0.5,
  "launch_interval_s": 20.0
}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _finite(value: float) -> bool:
    return math.isfinite(value)


def validate(raw: dict[str, Any]) -> dict[str, float]:
    data = {key: float(raw[key]) for key in (
        "mass_kg",
        "initial_speed_m_s",
        "exit_speed_m_s",
        "acceleration_m_s2",
        "electrical_efficiency",
        "launch_interval_s",
    )}
    if not all(_finite(value) for value in data.values()):
        raise ValueError("all inputs must be finite")
    if data["mass_kg"] <= 0.0:
        raise ValueError("mass must be positive")
    if data["initial_speed_m_s"] < 0.0:
        raise ValueError("initial speed must be non-negative")
    if data["exit_speed_m_s"] <= data["initial_speed_m_s"]:
        raise ValueError("exit speed must exceed initial speed")
    if data["acceleration_m_s2"] <= 0.0:
        raise ValueError("acceleration must be positive")
    if not 0.0 < data["electrical_efficiency"] <= 1.0:
        raise ValueError("efficiency must be in (0, 1]")
    if data["launch_interval_s"] <= 0.0:
        raise ValueError("launch interval must be positive")
    return data


def solve(raw: dict[str, Any]) -> dict[str, float]:
    data = validate(raw)
    mass = data["mass_kg"]
    v0 = data["initial_speed_m_s"]
    v = data["exit_speed_m_s"]
    acceleration = data["acceleration_m_s2"]
    efficiency = data["electrical_efficiency"]
    interval = data["launch_interval_s"]

    delta_v = v - v0
    accel_time = delta_v / acceleration
    track = (v * v - v0 * v0) / (2.0 * acceleration)
    force = mass * acceleration
    kinetic_gain = 0.5 * mass * (v * v - v0 * v0)
    electrical = kinetic_gain / efficiency

    return {
        "delta_v_m_s": delta_v,
        "acceleration_time_s": accel_time,
        "minimum_track_length_m": track,
        "average_force_n": force,
        "payload_kinetic_energy_gain_j": kinetic_gain,
        "electrical_input_energy_j": electrical,
        "average_power_during_acceleration_w": electrical / accel_time,
        "long_run_average_power_w": electrical / interval,
        "launches_per_hour": 3600.0 / interval,
        "energy_per_kg_j": electrical / mass,
    }


def _close(actual: float, expected: float, tol: float = 1e-9) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(f"{actual} != {expected} within {tol}")


def self_test() -> None:
    result = solve({
        "mass_kg": 100.0,
        "initial_speed_m_s": 0.0,
        "exit_speed_m_s": 100.0,
        "acceleration_m_s2": 10.0,
        "electrical_efficiency": 0.5,
        "launch_interval_s": 20.0,
    })
    _close(result["acceleration_time_s"], 10.0)
    _close(result["minimum_track_length_m"], 500.0)
    _close(result["payload_kinetic_energy_gain_j"], 500_000.0)
    _close(result["electrical_input_energy_j"], 1_000_000.0)
    _close(result["average_power_during_acceleration_w"], 100_000.0)
    _close(result["long_run_average_power_w"], 50_000.0)
    _close(result["launches_per_hour"], 180.0)

    nonzero = solve({
        "mass_kg": 20.0,
        "initial_speed_m_s": 50.0,
        "exit_speed_m_s": 150.0,
        "acceleration_m_s2": 20.0,
        "electrical_efficiency": 0.8,
        "launch_interval_s": 60.0,
    })
    _close(nonzero["acceleration_time_s"], 5.0)
    _close(nonzero["minimum_track_length_m"], 500.0)
    _close(nonzero["payload_kinetic_energy_gain_j"], 200_000.0)
    _close(nonzero["electrical_input_energy_j"], 250_000.0)

    slow = solve({
        "mass_kg": 100.0,
        "initial_speed_m_s": 0.0,
        "exit_speed_m_s": 200.0,
        "acceleration_m_s2": 10.0,
        "electrical_efficiency": 0.75,
        "launch_interval_s": 100.0,
    })
    fast = solve({
        "mass_kg": 100.0,
        "initial_speed_m_s": 0.0,
        "exit_speed_m_s": 200.0,
        "acceleration_m_s2": 40.0,
        "electrical_efficiency": 0.75,
        "launch_interval_s": 100.0,
    })
    _close(fast["minimum_track_length_m"], slow["minimum_track_length_m"] / 4.0)
    _close(fast["electrical_input_energy_j"], slow["electrical_input_energy_j"])
    if fast["average_power_during_acceleration_w"] <= slow["average_power_during_acceleration_w"]:
        raise AssertionError("higher acceleration should increase in-section average power")

    print("LL-002 independent launcher oracle self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.input is None:
        parser.error("provide a JSON input file or --self-test")

    raw = json.loads(args.input.read_text())
    json.dump(solve(raw), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
