#!/usr/bin/env python3
"""Independent LL-004C cislunar timing-sensitivity reference oracle.

Phase-0 only. This deliberately bounded reference covers an exact constant-velocity
moving-target segment, where closest approach and phase sensitivity have closed-form
solutions. It is intended to supply golden cases for the general encounter solver;
it is not a lunar ephemeris, trajectory propagator, guidance law, capture model, or
launch authority.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

Vec3 = tuple[float, float, float]


def _finite(values: Sequence[float]) -> bool:
    return all(math.isfinite(value) for value in values)


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, scalar: float) -> Vec3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


@dataclass(frozen=True)
class State:
    position_m: Vec3
    velocity_mps: Vec3


@dataclass(frozen=True)
class Sample:
    epoch_s: float
    state: State


def validate_constant_velocity_track(samples: Sequence[Sample]) -> tuple[float, float, State]:
    if len(samples) != 2:
        raise ValueError("reference requires exactly two track samples")
    a, b = samples
    values = (a.epoch_s, b.epoch_s, *a.state.position_m, *a.state.velocity_mps,
              *b.state.position_m, *b.state.velocity_mps)
    if not _finite(values):
        raise ValueError("non-finite track input")
    if b.epoch_s <= a.epoch_s:
        raise ValueError("track epochs must be strictly increasing")
    if any(abs(x - y) > 1.0e-12 for x, y in zip(a.state.velocity_mps, b.state.velocity_mps)):
        raise ValueError("reference requires constant target velocity")
    expected_end = add(a.state.position_m, scale(a.state.velocity_mps, b.epoch_s - a.epoch_s))
    if norm(sub(expected_end, b.state.position_m)) > 1.0e-9:
        raise ValueError("track samples are inconsistent with constant velocity")
    return a.epoch_s, b.epoch_s, a.state


def target_state_at(samples: Sequence[Sample], epoch_s: float) -> State:
    start_s, end_s, first = validate_constant_velocity_track(samples)
    if not math.isfinite(epoch_s) or epoch_s < start_s or epoch_s > end_s:
        raise ValueError("target-track extrapolation is forbidden")
    dt = epoch_s - start_s
    return State(add(first.position_m, scale(first.velocity_mps, dt)), first.velocity_mps)


def closest_approach(
    release: State,
    target_track: Sequence[Sample],
    start_s: float,
    end_s: float,
    target_phase_offset_s: float,
) -> dict[str, float]:
    inputs = (start_s, end_s, target_phase_offset_s, *release.position_m, *release.velocity_mps)
    if not _finite(inputs):
        raise ValueError("non-finite encounter input")
    if not (0.0 <= start_s < end_s):
        raise ValueError("invalid encounter window")
    if norm(release.position_m) > 1.0e15 or norm(release.velocity_mps) > 1.0e9:
        raise ValueError("reference input exceeds bounded study envelope")

    # A phase offset advances or retards the receiver on its already-declared
    # trajectory relative to the pod's release epoch. No new ephemeris is inferred.
    receiver_start = target_state_at(target_track, start_s + target_phase_offset_s)
    target_state_at(target_track, end_s + target_phase_offset_s)  # validate full-window coverage
    pod_at_start = State(
        add(release.position_m, scale(release.velocity_mps, start_s)),
        release.velocity_mps,
    )
    relative_position = sub(pod_at_start.position_m, receiver_start.position_m)
    relative_velocity = sub(release.velocity_mps, receiver_start.velocity_mps)
    vv = dot(relative_velocity, relative_velocity)
    if vv <= 0.0:
        tca_s = start_s
    else:
        unconstrained_offset = -dot(relative_position, relative_velocity) / vv
        tca_s = min(max(start_s + unconstrained_offset, start_s), end_s)
    elapsed = tca_s - start_s
    miss_vector = add(relative_position, scale(relative_velocity, elapsed))
    return {
        "target_phase_offset_s": target_phase_offset_s,
        "tca_s": tca_s,
        "miss_distance_m": norm(miss_vector),
        "relative_speed_mps": norm(relative_velocity),
    }


def timing_sweep(
    release: State,
    target_track: Sequence[Sample],
    start_s: float,
    end_s: float,
    phase_offsets_s: Sequence[float],
) -> dict[str, object]:
    if not phase_offsets_s:
        raise ValueError("timing sweep requires at least one phase offset")
    if not _finite(tuple(phase_offsets_s)):
        raise ValueError("phase offsets must be finite")
    if len(set(phase_offsets_s)) != len(phase_offsets_s):
        raise ValueError("phase offsets must be unique")
    if 0.0 not in phase_offsets_s:
        raise ValueError("timing sweep requires a zero-offset nominal case")

    rows = [
        closest_approach(release, target_track, start_s, end_s, offset)
        for offset in phase_offsets_s
    ]
    rows.sort(key=lambda row: row["target_phase_offset_s"])
    by_offset = {row["target_phase_offset_s"]: row for row in rows}
    nominal = by_offset[0.0]
    symmetric = sorted(abs(x) for x in phase_offsets_s if x != 0.0 and -x in by_offset)
    if not symmetric:
        raise ValueError("sweep requires at least one symmetric non-zero offset pair")
    h = symmetric[0]
    plus, minus = by_offset[h], by_offset[-h]
    d_tca_d_phase = (plus["tca_s"] - minus["tca_s"]) / (2.0 * h)
    miss_growth_per_s = max(
        abs(row["miss_distance_m"] - nominal["miss_distance_m"])
        / abs(row["target_phase_offset_s"])
        for row in rows if row["target_phase_offset_s"] != 0.0
    )
    return {
        "scenario_id": "LL004C_LINEAR_MOVING_TARGET_PHASE_SWEEP_V1",
        "model": "exact_constant_velocity_relative_motion",
        "evidence_class": "synthetic_analytic_reference",
        "rows": rows,
        "summary": {
            "nominal_tca_s": nominal["tca_s"],
            "nominal_miss_distance_m": nominal["miss_distance_m"],
            "central_tca_sensitivity_s_per_s": d_tca_d_phase,
            "maximum_miss_distance_m": max(row["miss_distance_m"] for row in rows),
            "maximum_miss_growth_m_per_s": miss_growth_per_s,
            "phase_envelope_s": [min(phase_offsets_s), max(phase_offsets_s)],
        },
        "claim_ceiling": (
            "Closed-form constant-velocity timing sensitivity only; not lunar ephemeris, "
            "gravity propagation, covariance, guidance, capture qualification, or flight evidence."
        ),
    }


def self_test() -> None:
    # Pod: r_p(t) = (t, 0, 0) m; target: r_t(t) = (10, 1 - 0.1 t, 0) m.
    # A target phase shift delta changes the target sample time to t + delta.
    # The exact minimum is t_TCA = 10 - delta/101 and d_min = |delta|/sqrt(101).
    release = State((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    target_track = [
        Sample(-5.0, State((10.0, 1.5, 0.0), (0.0, -0.1, 0.0))),
        Sample(25.0, State((10.0, -1.5, 0.0), (0.0, -0.1, 0.0))),
    ]
    report = timing_sweep(release, target_track, 0.0, 20.0, (-2.0, -1.0, 0.0, 1.0, 2.0))
    rows = {row["target_phase_offset_s"]: row for row in report["rows"]}
    for offset, row in rows.items():
        expected_tca = 10.0 - offset / 101.0
        expected_miss = abs(offset) / math.sqrt(101.0)
        assert abs(row["tca_s"] - expected_tca) < 1.0e-10
        assert abs(row["miss_distance_m"] - expected_miss) < 1.0e-10
    assert abs(report["summary"]["central_tca_sensitivity_s_per_s"] + 1.0 / 101.0) < 1.0e-10
    assert abs(report["summary"]["maximum_miss_distance_m"] - 2.0 / math.sqrt(101.0)) < 1.0e-10

    # Negative controls: no extrapolation, duplicate epochs, and missing nominal offset.
    try:
        closest_approach(release, target_track, 0.0, 20.0, 6.0)
        raise AssertionError("out-of-coverage phase shift unexpectedly accepted")
    except ValueError as exc:
        assert "extrapolation" in str(exc)
    try:
        timing_sweep(release, target_track, 0.0, 20.0, (-1.0, 1.0))
        raise AssertionError("sweep without nominal case unexpectedly accepted")
    except ValueError as exc:
        assert "zero-offset" in str(exc)
    try:
        validate_constant_velocity_track((target_track[0], target_track[0]))
        raise AssertionError("duplicate target epochs unexpectedly accepted")
    except ValueError as exc:
        assert "strictly increasing" in str(exc)

    print("LL-004C timing-sensitivity oracle self-test PASS")
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    self_test()
