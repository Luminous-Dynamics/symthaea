#!/usr/bin/env python3
"""Independent LL-006 bounded smart-pod correction oracle.

Uses the LL-003B spherical two-body oracle as the forward model. Estimates a
local endpoint Jacobian at a declared correction epoch and applies a bounded
minimum-norm impulse. Research/trade-study only; no flight guidance authority.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

Vec3 = tuple[float, float, float]


def _load_ballistics():
    path = Path(__file__).with_name("ll-spherical-ballistic-oracle.py")
    spec = importlib.util.spec_from_file_location("ll_spherical_ballistic_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ballistic oracle: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = _load_ballistics()


def add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def scale(a: Vec3, s: float) -> Vec3:
    return tuple(x * s for x in a)  # type: ignore[return-value]


def dot(a: Vec3, b: Vec3) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vec3) -> Vec3:
    n = norm(a)
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError("cannot normalize vector")
    return scale(a, 1.0 / n)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass(frozen=True)
class Case:
    mu_m3_s2: float
    radius_m: float
    speed_m_s: float
    elevation_deg: float
    azimuth_deg: float
    latitude_deg: float
    longitude_deg: float
    dt_s: float
    max_time_s: float
    model_ref: str
    constants_ref: str
    frame_ref: str


@dataclass(frozen=True)
class ReleasePerturbation:
    speed_delta_m_s: float = 0.0
    elevation_delta_deg: float = 0.0
    azimuth_delta_deg: float = 0.0


@dataclass(frozen=True)
class CorrectionPolicy:
    correction_fraction_of_nominal_flight: float
    total_delta_v_budget_m_s: float
    protected_reserve_fraction: float
    jacobian_probe_delta_v_m_s: float
    max_condition_number: float = 1.0e8

    def validate(self) -> None:
        vals = [
            self.correction_fraction_of_nominal_flight,
            self.total_delta_v_budget_m_s,
            self.protected_reserve_fraction,
            self.jacobian_probe_delta_v_m_s,
            self.max_condition_number,
        ]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("policy values must be finite")
        if not (0.0 < self.correction_fraction_of_nominal_flight < 1.0):
            raise ValueError("correction fraction must be strictly inside flight")
        if self.total_delta_v_budget_m_s < 0.0 or self.jacobian_probe_delta_v_m_s <= 0.0:
            raise ValueError("invalid delta-v values")
        if not (0.0 <= self.protected_reserve_fraction < 1.0):
            raise ValueError("invalid reserve fraction")
        if self.max_condition_number <= 1.0:
            raise ValueError("condition threshold must exceed one")


@dataclass(frozen=True)
class CorrectionResult:
    pre_correction_miss_m: float
    requested_delta_v_m_s: float
    applied_delta_v_m_s: float
    saturated: bool
    protected_reserve_m_s: float
    remaining_nominal_budget_m_s: float
    condition_number: float
    requested_components_m_s: tuple[float, float, float]
    applied_components_m_s: tuple[float, float, float]
    residual_miss_m: float
    improvement_ratio: float
    corrected_outcome: str
    target_latitude_deg: float
    target_longitude_deg: float
    model_ref: str
    constants_ref: str
    frame_ref: str


def validate_case(case: Case) -> None:
    B.validate(
        case.mu_m3_s2,
        case.radius_m,
        case.speed_m_s,
        case.elevation_deg,
        case.azimuth_deg,
        case.dt_s,
        case.max_time_s,
    )
    if not (-90.0 <= case.latitude_deg <= 90.0) or not math.isfinite(case.longitude_deg):
        raise ValueError("invalid coordinates")
    if not case.model_ref.strip() or not case.constants_ref.strip() or not case.frame_ref.strip():
        raise ValueError("missing provenance")


def initial_state(case: Case, perturb: ReleasePerturbation) -> tuple[Vec3, Vec3]:
    speed = case.speed_m_s + perturb.speed_delta_m_s
    elevation = case.elevation_deg + perturb.elevation_delta_deg
    azimuth = case.azimuth_deg + perturb.azimuth_delta_deg
    B.validate(case.mu_m3_s2, case.radius_m, speed, elevation, azimuth, case.dt_s, case.max_time_s)
    return B.initial_state(case.radius_m, speed, elevation, azimuth, case.latitude_deg, case.longitude_deg)


def advance_to_time(
    position: Vec3,
    velocity: Vec3,
    target_time_s: float,
    dt_s: float,
    mu: float,
    radius_m: float,
) -> tuple[Vec3, Vec3]:
    if target_time_s <= 0.0:
        return position, velocity
    t = 0.0
    while t < target_time_s:
        step = min(dt_s, target_time_s - t)
        position, velocity = B.rk4(position, velocity, step, mu)
        t += step
        if B.norm(position) <= radius_m:
            raise ValueError("trajectory reimpacted before correction epoch")
    return position, velocity


def propagate_state_to_surface(
    position: Vec3,
    velocity: Vec3,
    mu: float,
    radius_m: float,
    dt_s: float,
    max_time_s: float,
) -> tuple[str, float | None, float | None]:
    energy = 0.5 * B.dot(velocity, velocity) - mu / B.norm(position)
    if energy >= 0.0 and B.dot(position, velocity) > 0.0:
        return "escape", None, None

    t = 0.0
    previous_position = position
    previous_velocity = velocity
    previous_alt = B.norm(position) - radius_m
    if previous_alt <= 0.0:
        raise ValueError("correction state must be above reference sphere")

    while t < max_time_s:
        step = min(dt_s, max_time_s - t)
        next_position, next_velocity = B.rk4(previous_position, previous_velocity, step, mu)
        next_alt = B.norm(next_position) - radius_m
        if next_alt <= 0.0:
            lo, hi = 0.0, step
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                mid_position, _ = B.rk4(previous_position, previous_velocity, mid, mu)
                if B.norm(mid_position) - radius_m > 0.0:
                    lo = mid
                else:
                    hi = mid
            tau = 0.5 * (lo + hi)
            impact_position, _ = B.rk4(previous_position, previous_velocity, tau, mu)
            lat, lon = B.lat_lon(impact_position)
            return "reimpact", lat, lon
        previous_position, previous_velocity = next_position, next_velocity
        previous_alt = next_alt
        t += step
    return "no-return-within-window", None, None


def target_tangent_basis(lat_deg: float, lon_deg: float) -> tuple[Vec3, Vec3, Vec3]:
    north, east, up = B.surface_frame(lat_deg, lon_deg)
    return north, east, up


def endpoint_error(
    target_lat_deg: float,
    target_lon_deg: float,
    sample_lat_deg: float,
    sample_lon_deg: float,
    radius_m: float,
) -> tuple[float, float]:
    north, east, up = target_tangent_basis(target_lat_deg, target_lon_deg)
    sample_u = (
        math.cos(math.radians(sample_lat_deg)) * math.cos(math.radians(sample_lon_deg)),
        math.cos(math.radians(sample_lat_deg)) * math.sin(math.radians(sample_lon_deg)),
        math.sin(math.radians(sample_lat_deg)),
    )
    denom = B.dot(sample_u, up)
    if denom <= 0.0 or not math.isfinite(denom):
        raise ValueError("endpoint outside target tangent hemisphere")
    return (
        radius_m * B.dot(sample_u, east) / denom,
        radius_m * B.dot(sample_u, north) / denom,
    )


def correction_basis(position: Vec3, velocity: Vec3) -> tuple[Vec3, Vec3, Vec3]:
    radial = normalize(position)
    tangential = sub(velocity, scale(radial, dot(velocity, radial)))
    along = normalize(tangential)
    cross_track = normalize(cross(radial, along))
    return radial, along, cross_track


def apply_components(velocity: Vec3, basis: tuple[Vec3, Vec3, Vec3], components: tuple[float, float, float]) -> Vec3:
    out = velocity
    for axis, component in zip(basis, components):
        out = add(out, scale(axis, component))
    return out


def solve_min_norm_2x3(j: list[list[float]], error: tuple[float, float], max_condition: float) -> tuple[tuple[float, float, float], float]:
    a = sum(j[0][k] * j[0][k] for k in range(3))
    b = sum(j[0][k] * j[1][k] for k in range(3))
    d = sum(j[1][k] * j[1][k] for k in range(3))
    trace = a + d
    disc = math.sqrt(max(0.0, (a - d) ** 2 + 4.0 * b * b))
    lam_max = 0.5 * (trace + disc)
    lam_min = 0.5 * (trace - disc)
    if lam_min <= 0.0 or not math.isfinite(lam_min) or not math.isfinite(lam_max):
        raise ValueError("singular endpoint Jacobian")
    condition = lam_max / lam_min
    if condition > max_condition:
        raise ValueError(f"ill-conditioned endpoint Jacobian: {condition}")

    det = a * d - b * b
    inv00, inv01, inv11 = d / det, -b / det, a / det
    y0 = -(inv00 * error[0] + inv01 * error[1])
    y1 = -(inv01 * error[0] + inv11 * error[1])
    x = tuple(j[0][k] * y0 + j[1][k] * y1 for k in range(3))
    return x, condition  # type: ignore[return-value]


def correct(case: Case, perturb: ReleasePerturbation, policy: CorrectionPolicy) -> CorrectionResult:
    validate_case(case)
    policy.validate()

    nominal = B.propagate(
        case.mu_m3_s2,
        case.radius_m,
        case.speed_m_s,
        case.elevation_deg,
        case.azimuth_deg,
        case.latitude_deg,
        case.longitude_deg,
        case.dt_s,
        case.max_time_s,
    )
    if nominal.outcome != "reimpact":
        raise ValueError("nominal trajectory must reimpact")
    assert nominal.flight_time_s is not None
    assert nominal.arrival_latitude_deg is not None
    assert nominal.arrival_longitude_deg is not None

    correction_time = policy.correction_fraction_of_nominal_flight * nominal.flight_time_s
    p0, v0 = initial_state(case, perturb)
    p, v = advance_to_time(p0, v0, correction_time, case.dt_s, case.mu_m3_s2, case.radius_m)

    uncorrected_outcome, ulat, ulon = propagate_state_to_surface(
        p, v, case.mu_m3_s2, case.radius_m, case.dt_s, case.max_time_s - correction_time
    )
    if uncorrected_outcome != "reimpact":
        raise ValueError(f"perturbed trajectory does not reimpact: {uncorrected_outcome}")
    assert ulat is not None and ulon is not None
    pre_error = endpoint_error(
        nominal.arrival_latitude_deg, nominal.arrival_longitude_deg, ulat, ulon, case.radius_m
    )
    pre_miss = math.hypot(*pre_error)

    basis = correction_basis(p, v)
    probe = policy.jacobian_probe_delta_v_m_s
    jacobian = [[0.0] * 3 for _ in range(2)]
    for k in range(3):
        comps = [0.0, 0.0, 0.0]
        comps[k] = probe
        probe_v = apply_components(v, basis, tuple(comps))  # type: ignore[arg-type]
        outcome, plat, plon = propagate_state_to_surface(
            p, probe_v, case.mu_m3_s2, case.radius_m, case.dt_s, case.max_time_s - correction_time
        )
        if outcome != "reimpact" or plat is None or plon is None:
            raise ValueError("Jacobian probe did not reimpact")
        probe_error = endpoint_error(
            nominal.arrival_latitude_deg, nominal.arrival_longitude_deg, plat, plon, case.radius_m
        )
        jacobian[0][k] = (probe_error[0] - pre_error[0]) / probe
        jacobian[1][k] = (probe_error[1] - pre_error[1]) / probe

    requested, condition = solve_min_norm_2x3(jacobian, pre_error, policy.max_condition_number)
    requested_norm = norm(requested)
    protected_reserve = policy.total_delta_v_budget_m_s * policy.protected_reserve_fraction
    nominal_budget = policy.total_delta_v_budget_m_s - protected_reserve
    if requested_norm > nominal_budget and requested_norm > 0.0:
        factor = nominal_budget / requested_norm
        applied = tuple(x * factor for x in requested)
        saturated = True
    else:
        applied = requested
        saturated = False
    applied_norm = norm(applied)
    remaining_nominal = max(0.0, nominal_budget - applied_norm)

    corrected_v = apply_components(v, basis, applied)
    corrected_outcome, clat, clon = propagate_state_to_surface(
        p, corrected_v, case.mu_m3_s2, case.radius_m, case.dt_s, case.max_time_s - correction_time
    )
    if corrected_outcome != "reimpact" or clat is None or clon is None:
        return CorrectionResult(
            pre_miss, requested_norm, applied_norm, saturated, protected_reserve,
            remaining_nominal, condition, requested, applied, math.inf, 0.0,
            corrected_outcome, nominal.arrival_latitude_deg, nominal.arrival_longitude_deg,
            case.model_ref, case.constants_ref, case.frame_ref,
        )

    residual = endpoint_error(
        nominal.arrival_latitude_deg, nominal.arrival_longitude_deg, clat, clon, case.radius_m
    )
    residual_miss = math.hypot(*residual)
    improvement = 1.0 if pre_miss == 0.0 and residual_miss == 0.0 else (
        pre_miss / residual_miss if residual_miss > 0.0 else math.inf
    )
    return CorrectionResult(
        pre_miss, requested_norm, applied_norm, saturated, protected_reserve,
        remaining_nominal, condition, requested, applied, residual_miss, improvement,
        corrected_outcome, nominal.arrival_latitude_deg, nominal.arrival_longitude_deg,
        case.model_ref, case.constants_ref, case.frame_ref,
    )


def fixture_case() -> Case:
    return Case(
        mu_m3_s2=2.0e12,
        radius_m=1.0e6,
        speed_m_s=100.0,
        elevation_deg=45.0,
        azimuth_deg=90.0,
        latitude_deg=0.0,
        longitude_deg=0.0,
        dt_s=0.05,
        max_time_s=300.0,
        model_ref="ll003b-python-rk4-v1",
        constants_ref="synthetic-g2-r1e6",
        frame_ref="synthetic-body-fixed-proxy",
    )


def self_test() -> None:
    case = fixture_case()
    policy = CorrectionPolicy(0.5, 1.0, 0.2, 0.001)

    nominal = correct(case, ReleasePerturbation(), policy)
    if nominal.pre_correction_miss_m > 1.0e-8 or nominal.requested_delta_v_m_s > 1.0e-6:
        raise AssertionError(f"nominal correction not near zero: {nominal}")

    perturbed = correct(case, ReleasePerturbation(speed_delta_m_s=0.1, azimuth_delta_deg=0.01), policy)
    if perturbed.corrected_outcome != "reimpact":
        raise AssertionError("helpful correction did not reimpact")
    if not (perturbed.residual_miss_m < 0.2 * perturbed.pre_correction_miss_m):
        raise AssertionError(f"correction did not materially reduce miss: {perturbed}")
    if perturbed.applied_delta_v_m_s > 0.8 + 1.0e-12:
        raise AssertionError("protected reserve was consumed")

    tiny = correct(
        case,
        ReleasePerturbation(speed_delta_m_s=0.2, azimuth_delta_deg=0.02),
        CorrectionPolicy(0.5, 0.005, 0.2, 0.001),
    )
    if not tiny.saturated:
        raise AssertionError("tiny budget did not saturate")
    if tiny.applied_delta_v_m_s > 0.004 + 1.0e-12:
        raise AssertionError("tiny budget exceeded nominal allocation")
    if tiny.residual_miss_m <= 0.0 or tiny.residual_miss_m >= tiny.pre_correction_miss_m:
        raise AssertionError("tiny budget residual behavior is not honest/helpful")
    if abs(tiny.protected_reserve_m_s - 0.001) > 1.0e-12:
        raise AssertionError("protected reserve changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("LL-006 smart-pod correction oracle self-test: PASS")
        return

    if not args.input:
        parser.error("--input or --self-test is required")

    payload = json.loads(Path(args.input).read_text())
    result = correct(
        Case(**payload["case"]),
        ReleasePerturbation(**payload["perturbation"]),
        CorrectionPolicy(**payload["policy"]),
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
