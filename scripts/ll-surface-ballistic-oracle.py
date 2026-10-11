#!/usr/bin/env python3
"""Independent LL-003A short-range ballistic oracle.

This script intentionally models only equal-elevation, constant-gravity,
flat-surface projectile motion. It is a validation oracle for short-range
limits, not a lunar operational trajectory or safety model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Result:
    approximation: str
    speed_m_s: float
    elevation_deg: float
    gravity_m_s2: float
    range_m: float
    flight_time_s: float
    max_height_m: float
    arrival_horizontal_speed_m_s: float
    arrival_vertical_speed_m_s: float
    arrival_speed_m_s: float


def solve(speed_m_s: float, elevation_deg: float, gravity_m_s2: float) -> Result:
    vals = (speed_m_s, elevation_deg, gravity_m_s2)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("all inputs must be finite")
    if speed_m_s <= 0.0 or gravity_m_s2 <= 0.0:
        raise ValueError("speed and gravity must be positive")
    if not (0.0 < elevation_deg < 90.0):
        raise ValueError("elevation must be strictly between 0 and 90 degrees")

    theta = math.radians(elevation_deg)
    vx = speed_m_s * math.cos(theta)
    vy = speed_m_s * math.sin(theta)
    range_m = speed_m_s**2 * math.sin(2.0 * theta) / gravity_m_s2
    flight_time_s = 2.0 * vy / gravity_m_s2
    max_height_m = vy**2 / (2.0 * gravity_m_s2)

    # Equal-elevation ideal case: horizontal speed is unchanged and vertical
    # speed reverses sign at impact.
    arrival_vx = vx
    arrival_vy = -vy
    arrival_speed = math.hypot(arrival_vx, arrival_vy)

    return Result(
        approximation="flat-constant-g-equal-elevation",
        speed_m_s=speed_m_s,
        elevation_deg=elevation_deg,
        gravity_m_s2=gravity_m_s2,
        range_m=range_m,
        flight_time_s=flight_time_s,
        max_height_m=max_height_m,
        arrival_horizontal_speed_m_s=arrival_vx,
        arrival_vertical_speed_m_s=arrival_vy,
        arrival_speed_m_s=arrival_speed,
    )


def close(actual: float, expected: float, tol: float = 1.0e-10) -> None:
    if abs(actual - expected) > tol * max(1.0, abs(expected)):
        raise AssertionError(f"{actual} != {expected} within relative tolerance {tol}")


def self_test() -> None:
    # 45-degree range identity: R = v^2/g.
    r = solve(100.0, 45.0, 2.0)
    close(r.range_m, 5000.0)
    close(r.flight_time_s, 100.0 * math.sqrt(2.0) / 2.0)
    close(r.max_height_m, 1250.0)
    close(r.arrival_speed_m_s, 100.0)

    # Complementary launch angles have identical ideal range.
    low = solve(80.0, 30.0, 1.5)
    high = solve(80.0, 60.0, 1.5)
    close(low.range_m, high.range_m)
    if high.flight_time_s <= low.flight_time_s:
        raise AssertionError("higher complementary angle should have longer flight time")
    if high.max_height_m <= low.max_height_m:
        raise AssertionError("higher complementary angle should have larger apogee")

    # Scaling: range is quadratic in speed and inverse in gravity.
    base = solve(20.0, 45.0, 2.0)
    fast = solve(40.0, 45.0, 2.0)
    weak_g = solve(20.0, 45.0, 1.0)
    close(fast.range_m, 4.0 * base.range_m)
    close(weak_g.range_m, 2.0 * base.range_m)

    # Invalid domains fail closed.
    invalid = [
        (0.0, 45.0, 1.0),
        (10.0, 0.0, 1.0),
        (10.0, 90.0, 1.0),
        (10.0, 45.0, 0.0),
        (math.nan, 45.0, 1.0),
    ]
    for args in invalid:
        try:
            solve(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid input unexpectedly accepted: {args}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--speed-m-s", type=float)
    parser.add_argument("--elevation-deg", type=float)
    parser.add_argument("--gravity-m-s2", type=float)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("LL-003A oracle self-tests: PASS")
        return 0

    if args.speed_m_s is None or args.elevation_deg is None or args.gravity_m_s2 is None:
        parser.error("speed, elevation, and gravity are required unless --self-test is used")

    try:
        result = solve(args.speed_m_s, args.elevation_deg, args.gravity_m_s2)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = asdict(result)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
