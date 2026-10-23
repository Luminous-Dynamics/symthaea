#!/usr/bin/env python3
"""Independent LL-004 moving-target encounter oracle.

Phase-0 research only. Supports:
- explicit inertial release state;
- provenance-bearing target track samples;
- cubic-Hermite target interpolation;
- zero-gravity synthetic analytic mode and positive-mu central gravity mode;
- bounded closest-approach search and timing sensitivity.

It does not perform surface-site -> inertial conversion, fetch ephemerides, guide a pod,
or authorize launch.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def scale(a: Vec3, scalar: float) -> Vec3:
    return tuple(x * scalar for x in a)  # type: ignore[return-value]


def dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


@dataclass(frozen=True)
class State:
    position: Vec3
    velocity: Vec3


@dataclass(frozen=True)
class Sample:
    seconds: float
    state: State


def acceleration(position: Vec3, mu: float) -> Vec3:
    if mu == 0.0:
        return (0.0, 0.0, 0.0)
    radius = norm(position)
    if radius <= 0.0 or not math.isfinite(radius):
        raise ValueError("invalid radius")
    return scale(position, -mu / radius**3)


def rk4(state: State, dt_s: float, mu: float) -> State:
    r, v = state.position, state.velocity
    k1r, k1v = v, acceleration(r, mu)
    r2, v2 = add(r, scale(k1r, dt_s / 2)), add(v, scale(k1v, dt_s / 2))
    k2r, k2v = v2, acceleration(r2, mu)
    r3, v3 = add(r, scale(k2r, dt_s / 2)), add(v, scale(k2v, dt_s / 2))
    k3r, k3v = v3, acceleration(r3, mu)
    r4, v4 = add(r, scale(k3r, dt_s)), add(v, scale(k3v, dt_s))
    k4r, k4v = v4, acceleration(r4, mu)
    rn = tuple(
        r[i] + dt_s * (k1r[i] + 2 * k2r[i] + 2 * k3r[i] + k4r[i]) / 6
        for i in range(3)
    )
    vn = tuple(
        v[i] + dt_s * (k1v[i] + 2 * k2v[i] + 2 * k3v[i] + k4v[i]) / 6
        for i in range(3)
    )
    return State(rn, vn)  # type: ignore[arg-type]


def propagate(state: State, seconds: float, dt_s: float, mu: float) -> State:
    if seconds < 0.0 or dt_s <= 0.0 or mu < 0.0:
        raise ValueError("bad propagation input")
    current = state
    elapsed = 0.0
    while elapsed < seconds:
        step = min(dt_s, seconds - elapsed)
        current = rk4(current, step, mu)
        elapsed += step
    return current


def hermite(a: Sample, b: Sample, seconds: float) -> State:
    if not a.seconds <= seconds <= b.seconds or b.seconds <= a.seconds:
        raise ValueError("bad interpolation interval")
    dt = b.seconds - a.seconds
    u = (seconds - a.seconds) / dt
    u2, u3 = u * u, u * u * u
    h00, h10 = 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u
    h01, h11 = -2 * u3 + 3 * u2, u3 - u2
    dh00, dh10 = 6 * u2 - 6 * u, 3 * u2 - 4 * u + 1
    dh01, dh11 = -6 * u2 + 6 * u, 3 * u2 - 2 * u
    position, velocity = [], []
    for i in range(3):
        position.append(
            h00 * a.state.position[i]
            + h10 * dt * a.state.velocity[i]
            + h01 * b.state.position[i]
            + h11 * dt * b.state.velocity[i]
        )
        velocity.append(
            (dh00 * a.state.position[i] + dh01 * b.state.position[i]) / dt
            + dh10 * a.state.velocity[i]
            + dh11 * b.state.velocity[i]
        )
    return State(tuple(position), tuple(velocity))  # type: ignore[arg-type]


def track_state(samples: list[Sample], seconds: float) -> State:
    if len(samples) < 2:
        raise ValueError("track too short")
    if seconds < samples[0].seconds or seconds > samples[-1].seconds:
        raise ValueError("extrapolation forbidden")
    for sample in samples:
        if abs(seconds - sample.seconds) <= 1.0e-12:
            return sample.state
    for a, b in zip(samples, samples[1:]):
        if a.seconds < seconds < b.seconds:
            return hermite(a, b, seconds)
    raise ValueError("no interpolation bracket")


def separation(
    release: State,
    target: list[Sample],
    seconds: float,
    dt_s: float,
    mu: float,
) -> tuple[float, float, State, State]:
    pod = propagate(release, seconds, dt_s, mu)
    receiver = track_state(target, seconds)
    return (
        norm(sub(pod.position, receiver.position)),
        norm(sub(pod.velocity, receiver.velocity)),
        pod,
        receiver,
    )


def closest_approach(
    release: State,
    target: list[Sample],
    start_s: float,
    end_s: float,
    scan_step_s: float,
    integration_step_s: float,
    mu: float,
    refinement_iterations: int = 50,
) -> tuple[float, float, float, State, State]:
    if not (0.0 <= start_s < end_s and scan_step_s > 0.0 and integration_step_s > 0.0):
        raise ValueError("bad study window")
    t = start_s
    best_distance, best_t = math.inf, start_s
    while t <= end_s + 1.0e-12:
        sample_t = min(t, end_s)
        distance, *_ = separation(release, target, sample_t, integration_step_s, mu)
        if distance < best_distance:
            best_distance, best_t = distance, sample_t
        t += scan_step_s

    lo = max(start_s, best_t - scan_step_s)
    hi = min(end_s, best_t + scan_step_s)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    c, d = hi - (hi - lo) / phi, lo + (hi - lo) / phi
    fc = separation(release, target, c, integration_step_s, mu)[0]
    fd = separation(release, target, d, integration_step_s, mu)[0]
    for _ in range(refinement_iterations):
        if fc <= fd:
            hi, d, fd = d, c, fc
            c = hi - (hi - lo) / phi
            fc = separation(release, target, c, integration_step_s, mu)[0]
        else:
            lo, c, fc = c, d, fd
            d = lo + (hi - lo) / phi
            fd = separation(release, target, d, integration_step_s, mu)[0]
    best_t = 0.5 * (lo + hi)
    distance, relative_speed, pod, receiver = separation(
        release, target, best_t, integration_step_s, mu
    )
    return best_t, distance, relative_speed, pod, receiver


def self_test() -> None:
    # Constant-velocity Hermite interpolation is exact.
    samples = [
        Sample(0.0, State((10.0, 0.0, 0.0), (0.1, 0.0, 0.0))),
        Sample(100.0, State((20.0, 0.0, 0.0), (0.1, 0.0, 0.0))),
    ]
    mid = track_state(samples, 50.0)
    assert abs(mid.position[0] - 15.0) < 1.0e-12
    assert abs(mid.velocity[0] - 0.1) < 1.0e-12

    # Closed-form zero-gravity moving-target intercept at exactly t=10 s.
    release = State((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    target = [
        Sample(0.0, State((10.0, 1.0, 0.0), (0.0, -0.1, 0.0))),
        Sample(20.0, State((10.0, -1.0, 0.0), (0.0, -0.1, 0.0))),
    ]
    encounter_t, miss, relative_speed, _, _ = closest_approach(
        release, target, 0.0, 20.0, 1.0, 0.1, 0.0, 60
    )
    assert abs(encounter_t - 10.0) < 1.0e-8
    assert miss < 1.0e-8
    assert abs(relative_speed - math.sqrt(1.01)) < 1.0e-9

    # A one-second target-phase shift no longer produces a zero-miss encounter.
    shifted = [
        Sample(0.0, State((10.0, 0.9, 0.0), (0.0, -0.1, 0.0))),
        Sample(20.0, State((10.0, -1.1, 0.0), (0.0, -0.1, 0.0))),
    ]
    _, shifted_miss, _, _, _ = closest_approach(
        release, shifted, 0.0, 20.0, 1.0, 0.1, 0.0, 60
    )
    assert shifted_miss > 0.0

    # Positive-mu deterministic replay fixture.
    mu = 4_902.8
    pod = State((2_000.0, 0.0, 0.0), (0.0, 1.55, 0.0))
    receiver_initial = State((2_100.0, 0.0, 0.0), (0.0, 1.50, 0.0))
    receiver_track = [
        Sample(t, propagate(receiver_initial, t, 0.2, mu))
        for t in (0.0, 50.0, 100.0, 150.0, 200.0)
    ]
    first = closest_approach(pod, receiver_track, 0.0, 200.0, 10.0, 0.2, mu, 40)
    second = closest_approach(pod, receiver_track, 0.0, 200.0, 10.0, 0.2, mu, 40)
    assert all(abs(a - b) < 1.0e-12 for a, b in zip(first[:3], second[:3]))

    print("LL-004 moving-target oracle self-test PASS")
    print(
        {
            "analytic_encounter_time_s": encounter_t,
            "analytic_miss": miss,
            "positive_mu_closest_s": first[0],
            "positive_mu_miss_km": first[1],
        }
    )


if __name__ == "__main__":
    self_test()
