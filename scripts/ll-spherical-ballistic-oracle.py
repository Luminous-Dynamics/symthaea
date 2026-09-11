#!/usr/bin/env python3
"""Independent LL-003B spherical two-body ballistic oracle.

Propagates an ideal ballistic cargo pod from a spherical body's surface under a
caller-supplied central gravitational parameter. This is a research oracle for
surface-hop dynamics, not an operational launch or safe-corridor model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

Vec3 = tuple[float, float, float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def scale(a: Vec3, s: float) -> Vec3:
    return tuple(x * s for x in a)  # type: ignore[return-value]


def dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def acceleration(position: Vec3, mu: float) -> Vec3:
    r = norm(position)
    factor = -mu / (r * r * r)
    return scale(position, factor)


def rk4(position: Vec3, velocity: Vec3, dt: float, mu: float) -> tuple[Vec3, Vec3]:
    k1r = velocity
    k1v = acceleration(position, mu)

    r2 = add(position, scale(k1r, 0.5 * dt))
    v2 = add(velocity, scale(k1v, 0.5 * dt))
    k2r = v2
    k2v = acceleration(r2, mu)

    r3 = add(position, scale(k2r, 0.5 * dt))
    v3 = add(velocity, scale(k2v, 0.5 * dt))
    k3r = v3
    k3v = acceleration(r3, mu)

    r4 = add(position, scale(k3r, dt))
    v4 = add(velocity, scale(k3v, dt))
    k4r = v4
    k4v = acceleration(r4, mu)

    rn = tuple(
        position[i] + dt * (k1r[i] + 2 * k2r[i] + 2 * k3r[i] + k4r[i]) / 6
        for i in range(3)
    )
    vn = tuple(
        velocity[i] + dt * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) / 6
        for i in range(3)
    )
    return rn, vn  # type: ignore[return-value]


def specific_energy(position: Vec3, velocity: Vec3, mu: float) -> float:
    return 0.5 * dot(velocity, velocity) - mu / norm(position)


def angular_momentum(position: Vec3, velocity: Vec3) -> float:
    return norm(cross(position, velocity))


def surface_frame(latitude_deg: float, longitude_deg: float) -> tuple[Vec3, Vec3, Vec3]:
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    up = (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
    east = (-math.sin(lon), math.cos(lon), 0.0)
    north = (-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat))
    return north, east, up


def initial_state(
    radius_m: float,
    speed_m_s: float,
    elevation_deg: float,
    azimuth_deg: float,
    latitude_deg: float,
    longitude_deg: float,
) -> tuple[Vec3, Vec3]:
    north, east, up = surface_frame(latitude_deg, longitude_deg)
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    horizontal = add(scale(north, math.cos(az)), scale(east, math.sin(az)))
    direction = add(scale(horizontal, math.cos(el)), scale(up, math.sin(el)))
    return scale(up, radius_m), scale(direction, speed_m_s)


def lat_lon(position: Vec3) -> tuple[float, float]:
    r = norm(position)
    lat = math.degrees(math.asin(position[2] / r))
    lon = math.degrees(math.atan2(position[1], position[0]))
    return lat, lon


@dataclass(frozen=True)
class Result:
    approximation: str
    outcome: str
    flight_time_s: float | None
    ground_range_m: float | None
    max_altitude_m: float
    arrival_speed_m_s: float | None
    arrival_flight_path_angle_deg: float | None
    arrival_latitude_deg: float | None
    arrival_longitude_deg: float | None
    max_relative_energy_error: float
    max_relative_angular_momentum_error: float


def validate(mu: float, radius: float, speed: float, elevation: float, azimuth: float, dt: float, max_time: float) -> None:
    values = (mu, radius, speed, elevation, azimuth, dt, max_time)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("all inputs must be finite")
    if mu <= 0 or radius <= 0 or speed <= 0 or dt <= 0 or max_time <= 0:
        raise ValueError("mu, radius, speed, dt, and max_time must be positive")
    if not (0.0 < elevation < 90.0):
        raise ValueError("elevation must be strictly between 0 and 90 degrees")


def propagate(
    mu_m3_s2: float,
    radius_m: float,
    speed_m_s: float,
    elevation_deg: float,
    azimuth_deg: float,
    latitude_deg: float,
    longitude_deg: float,
    dt_s: float,
    max_time_s: float,
) -> Result:
    validate(mu_m3_s2, radius_m, speed_m_s, elevation_deg, azimuth_deg, dt_s, max_time_s)
    if not (-90.0 <= latitude_deg <= 90.0) or not math.isfinite(longitude_deg):
        raise ValueError("invalid launch coordinates")

    initial_position, initial_velocity = initial_state(
        radius_m, speed_m_s, elevation_deg, azimuth_deg, latitude_deg, longitude_deg
    )
    position, velocity = initial_position, initial_velocity
    e0 = specific_energy(position, velocity, mu_m3_s2)
    h0 = angular_momentum(position, velocity)

    if e0 >= 0.0 and dot(position, velocity) > 0.0:
        return Result(
            "spherical-two-body-rk4",
            "escape",
            None,
            None,
            0.0,
            None,
            None,
            None,
            None,
            0.0,
            0.0,
        )

    t = 0.0
    max_altitude = 0.0
    max_energy_error = 0.0
    max_h_error = 0.0
    lifted = False

    while t < max_time_s:
        next_position, next_velocity = rk4(position, velocity, dt_s, mu_m3_s2)
        next_altitude = norm(next_position) - radius_m
        current_altitude = norm(position) - radius_m
        max_altitude = max(max_altitude, next_altitude)
        if next_altitude > 1.0e-10 * radius_m:
            lifted = True

        e = specific_energy(next_position, next_velocity, mu_m3_s2)
        h = angular_momentum(next_position, next_velocity)
        max_energy_error = max(max_energy_error, abs(e - e0) / max(abs(e0), 1.0e-30))
        max_h_error = max(max_h_error, abs(h - h0) / max(abs(h0), 1.0e-30))

        if lifted and current_altitude > 0.0 and next_altitude <= 0.0:
            lo, hi = 0.0, dt_s
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                mid_position, _ = rk4(position, velocity, mid, mu_m3_s2)
                if norm(mid_position) - radius_m > 0.0:
                    lo = mid
                else:
                    hi = mid
            tau = 0.5 * (lo + hi)
            impact_position, impact_velocity = rk4(position, velocity, tau, mu_m3_s2)
            impact_time = t + tau

            u0 = scale(initial_position, 1.0 / radius_m)
            ui = scale(impact_position, 1.0 / norm(impact_position))
            central_angle = math.acos(max(-1.0, min(1.0, dot(u0, ui))))
            ground_range = radius_m * central_angle
            arrival_speed = norm(impact_velocity)
            radial_speed = dot(impact_velocity, ui)
            tangential_speed = math.sqrt(max(0.0, arrival_speed**2 - radial_speed**2))
            flight_path_angle = math.degrees(math.atan2(radial_speed, tangential_speed))
            arrival_lat, arrival_lon = lat_lon(impact_position)

            return Result(
                "spherical-two-body-rk4",
                "reimpact",
                impact_time,
                ground_range,
                max_altitude,
                arrival_speed,
                flight_path_angle,
                arrival_lat,
                arrival_lon,
                max_energy_error,
                max_h_error,
            )

        position, velocity = next_position, next_velocity
        t += dt_s

    return Result(
        "spherical-two-body-rk4",
        "no-return-within-window",
        None,
        None,
        max_altitude,
        None,
        None,
        None,
        None,
        max_energy_error,
        max_h_error,
    )


def flat_range(speed: float, elevation_deg: float, g: float) -> float:
    theta = math.radians(elevation_deg)
    return speed * speed * math.sin(2.0 * theta) / g


def close(actual: float, expected: float, rel: float) -> None:
    if abs(actual - expected) > rel * max(1.0, abs(expected)):
        raise AssertionError(f"{actual} != {expected} within relative tolerance {rel}")


def self_test() -> None:
    # Normalized body: mu=1, radius=1 => local surface gravity g=1.
    short = propagate(1.0, 1.0, 0.05, 45.0, 90.0, 0.0, 0.0, 1.0e-4, 1.0)
    if short.outcome != "reimpact":
        raise AssertionError(short)
    expected_flat = flat_range(0.05, 45.0, 1.0)
    close(short.ground_range_m or 0.0, expected_flat, 2.0e-3)
    close(short.arrival_speed_m_s or 0.0, 0.05, 1.0e-9)
    if short.max_relative_energy_error > 1.0e-9 or short.max_relative_angular_momentum_error > 1.0e-9:
        raise AssertionError("two-body invariants drifted beyond oracle tolerance")

    # East/west symmetry at the equator.
    east = propagate(1.0, 1.0, 0.1, 45.0, 90.0, 0.0, 0.0, 2.0e-4, 1.0)
    west = propagate(1.0, 1.0, 0.1, 45.0, 270.0, 0.0, 0.0, 2.0e-4, 1.0)
    close(east.ground_range_m or 0.0, west.ground_range_m or 0.0, 1.0e-10)
    close(east.flight_time_s or 0.0, west.flight_time_s or 0.0, 1.0e-10)
    if not ((east.arrival_longitude_deg or 0.0) > 0.0 and (west.arrival_longitude_deg or 0.0) < 0.0):
        raise AssertionError("east/west arrival longitudes did not reflect")

    # Outward launch above local escape speed must not be forced to reimpact.
    escape_speed = math.sqrt(2.0)
    escaped = propagate(1.0, 1.0, 1.01 * escape_speed, 45.0, 90.0, 0.0, 0.0, 1.0e-3, 1.0)
    if escaped.outcome != "escape":
        raise AssertionError(escaped)

    # Step refinement should converge for an ordinary suborbital arc.
    coarse = propagate(1.0, 1.0, 0.3, 45.0, 90.0, 0.0, 0.0, 2.0e-3, 2.0)
    fine = propagate(1.0, 1.0, 0.3, 45.0, 90.0, 0.0, 0.0, 1.0e-3, 2.0)
    close(coarse.ground_range_m or 0.0, fine.ground_range_m or 0.0, 1.0e-5)
    close(coarse.flight_time_s or 0.0, fine.flight_time_s or 0.0, 1.0e-5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mu-m3-s2", type=float)
    parser.add_argument("--radius-m", type=float)
    parser.add_argument("--speed-m-s", type=float)
    parser.add_argument("--elevation-deg", type=float)
    parser.add_argument("--azimuth-deg", type=float, default=90.0)
    parser.add_argument("--latitude-deg", type=float, default=0.0)
    parser.add_argument("--longitude-deg", type=float, default=0.0)
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--max-time-s", type=float, default=100000.0)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("LL-003B spherical oracle self-tests: PASS")
        return 0

    required = (args.mu_m3_s2, args.radius_m, args.speed_m_s, args.elevation_deg)
    if any(value is None for value in required):
        parser.error("mu, radius, speed, and elevation are required unless --self-test is used")

    result = propagate(
        args.mu_m3_s2,
        args.radius_m,
        args.speed_m_s,
        args.elevation_deg,
        args.azimuth_deg,
        args.latitude_deg,
        args.longitude_deg,
        args.dt_s,
        args.max_time_s,
    )
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
