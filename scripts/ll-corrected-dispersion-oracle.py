#!/usr/bin/env python3
"""LL-006C corrected-dispersion ensemble oracle.

Uses the LL-005 deterministic release-state sampler and LL-003B forward model,
but independently implements the local minimum-norm correction calculation.
Research/trade-study only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def _load_dispersion():
    path = Path(__file__).with_name("ll-ballistic-dispersion-oracle.py")
    spec = importlib.util.spec_from_file_location("ll_dispersion", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load LL-005 oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D = _load_dispersion()
B = D.BALLISTICS
Vec3 = tuple[float, float, float]


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
    if n <= 0.0 or not math.isfinite(n):
        raise ValueError("cannot normalize")
    return scale(a, 1.0 / n)


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@dataclass(frozen=True)
class Policy:
    correction_fraction: float
    total_delta_v_budget_m_s: float
    protected_reserve_fraction: float
    jacobian_probe_m_s: float
    max_condition_number: float = 1.0e8

    def validate(self) -> None:
        vals = [
            self.correction_fraction,
            self.total_delta_v_budget_m_s,
            self.protected_reserve_fraction,
            self.jacobian_probe_m_s,
            self.max_condition_number,
        ]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("nonfinite policy")
        if not 0.0 < self.correction_fraction < 1.0:
            raise ValueError("bad correction fraction")
        if self.total_delta_v_budget_m_s < 0.0 or self.jacobian_probe_m_s <= 0.0:
            raise ValueError("bad delta-v")
        if not 0.0 <= self.protected_reserve_fraction < 1.0:
            raise ValueError("bad reserve")
        if self.max_condition_number <= 1.0:
            raise ValueError("bad conditioning threshold")


@dataclass(frozen=True)
class Summary:
    seed: int
    sample_count: int
    corrected_count: int
    saturated_count: int
    failed_count: int
    pre_rms_m: float
    post_rms_m: float
    pre_p95_m: float
    post_p95_m: float
    mean_applied_delta_v_m_s: float
    p95_applied_delta_v_m_s: float
    saturation_fraction: float
    protected_reserve_m_s: float
    model_ref: str
    constants_ref: str
    frame_ref: str


def advance(p: Vec3, v: Vec3, target: float, dt: float, mu: float, radius: float) -> tuple[Vec3, Vec3]:
    t = 0.0
    while t < target:
        step = min(dt, target - t)
        p, v = B.rk4(p, v, step, mu)
        t += step
        if B.norm(p) <= radius:
            raise ValueError("reimpact before correction")
    return p, v


def impact_from_state(p: Vec3, v: Vec3, mu: float, radius: float, dt: float, max_time: float):
    energy = 0.5 * B.dot(v, v) - mu / B.norm(p)
    if energy >= 0.0 and B.dot(p, v) > 0.0:
        return None
    t = 0.0
    while t < max_time:
        step = min(dt, max_time - t)
        pn, vn = B.rk4(p, v, step, mu)
        if B.norm(pn) - radius <= 0.0:
            lo, hi = 0.0, step
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                mp, _ = B.rk4(p, v, mid, mu)
                if B.norm(mp) - radius > 0.0:
                    lo = mid
                else:
                    hi = mid
            ip, _ = B.rk4(p, v, 0.5 * (lo + hi), mu)
            return B.lat_lon(ip)
        p, v = pn, vn
        t += step
    return None


def basis(p: Vec3, v: Vec3):
    radial = normalize(p)
    tangential = sub(v, scale(radial, dot(v, radial)))
    along = normalize(tangential)
    cross_track = normalize(cross(radial, along))
    return radial, along, cross_track


def apply(v: Vec3, axes, components):
    out = v
    for axis, component in zip(axes, components):
        out = add(out, scale(axis, component))
    return out


def solve(j, error, max_condition):
    a = sum(j[0][k] ** 2 for k in range(3))
    b = sum(j[0][k] * j[1][k] for k in range(3))
    d = sum(j[1][k] ** 2 for k in range(3))
    trace = a + d
    disc = math.sqrt(max(0.0, (a - d) ** 2 + 4.0 * b * b))
    lmax = 0.5 * (trace + disc)
    lmin = 0.5 * (trace - disc)
    if lmin <= 0.0 or lmax / lmin > max_condition:
        raise ValueError("ill-conditioned")
    det = a * d - b * b
    y0 = -(d * error[0] - b * error[1]) / det
    y1 = -(-b * error[0] + a * error[1]) / det
    return tuple(j[0][k] * y0 + j[1][k] * y1 for k in range(3))


def corrected_sample(case, speed_delta, elevation_delta, azimuth_delta, policy, target):
    speed = case.speed_m_s + speed_delta
    elevation = case.elevation_deg + elevation_delta
    azimuth = case.azimuth_deg + azimuth_delta
    if speed <= 0.0 or not 0.0 < elevation < 90.0:
        raise ValueError("invalid perturbed release")
    p0, v0 = B.initial_state(
        case.radius_m, speed, elevation, azimuth, case.latitude_deg, case.longitude_deg
    )
    nominal = D.propagate(case)
    correction_time = policy.correction_fraction * nominal.flight_time_s
    p, v = advance(p0, v0, correction_time, case.dt_s, case.mu_m3_s2, case.radius_m)
    uncorrected = impact_from_state(
        p, v, case.mu_m3_s2, case.radius_m, case.dt_s, case.max_time_s - correction_time
    )
    if uncorrected is None:
        raise ValueError("uncorrected no reimpact")
    pre = D.tangent_offset_m(target[0], target[1], uncorrected[0], uncorrected[1], case.radius_m)
    pre_miss = math.hypot(*pre)

    axes = basis(p, v)
    probe = policy.jacobian_probe_m_s
    jacobian = [[0.0] * 3 for _ in range(2)]
    for k in range(3):
        components = [0.0, 0.0, 0.0]
        components[k] = probe
        hit = impact_from_state(
            p,
            apply(v, axes, components),
            case.mu_m3_s2,
            case.radius_m,
            case.dt_s,
            case.max_time_s - correction_time,
        )
        if hit is None:
            raise ValueError("probe no reimpact")
        err = D.tangent_offset_m(target[0], target[1], hit[0], hit[1], case.radius_m)
        jacobian[0][k] = (err[0] - pre[0]) / probe
        jacobian[1][k] = (err[1] - pre[1]) / probe

    requested = solve(jacobian, pre, policy.max_condition_number)
    requested_norm = norm(requested)
    nominal_budget = policy.total_delta_v_budget_m_s * (1.0 - policy.protected_reserve_fraction)
    saturated = requested_norm > nominal_budget and requested_norm > 0.0
    applied = scale(requested, nominal_budget / requested_norm) if saturated else requested
    applied_norm = norm(applied)
    hit = impact_from_state(
        p,
        apply(v, axes, applied),
        case.mu_m3_s2,
        case.radius_m,
        case.dt_s,
        case.max_time_s - correction_time,
    )
    if hit is None:
        raise ValueError("corrected no reimpact")
    post = D.tangent_offset_m(target[0], target[1], hit[0], hit[1], case.radius_m)
    return pre_miss, math.hypot(*post), applied_norm, saturated


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lo, hi = int(math.floor(position)), int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    w = position - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def run(case, uncertainty, policy, seed, sample_count):
    case.validate()
    uncertainty.validate()
    policy.validate()
    if uncertainty.site_north_sigma_m != 0.0 or uncertainty.site_east_sigma_m != 0.0:
        raise ValueError("LL-006C v0 does not yet correct displaced launch-site uncertainty")
    nominal = D.propagate(case)
    if nominal.outcome != "reimpact":
        raise ValueError("nominal must reimpact")
    target = (nominal.arrival_latitude_deg, nominal.arrival_longitude_deg)
    rng = D.SplitMix64(seed)
    pre, post, delta_vs = [], [], []
    saturated = failed = 0

    for _ in range(sample_count):
        speed_delta = uncertainty.speed_sigma_m_s * rng.normal()
        elevation_delta = uncertainty.elevation_sigma_deg * rng.normal()
        azimuth_delta = uncertainty.azimuth_sigma_deg * rng.normal()
        rng.normal()
        rng.normal()
        try:
            before, after, delta_v, is_saturated = corrected_sample(
                case, speed_delta, elevation_delta, azimuth_delta, policy, target
            )
        except ValueError:
            failed += 1
            continue
        pre.append(before)
        post.append(after)
        delta_vs.append(delta_v)
        saturated += int(is_saturated)

    if not pre:
        raise ValueError("no corrected samples")

    return Summary(
        seed=seed,
        sample_count=sample_count,
        corrected_count=len(pre),
        saturated_count=saturated,
        failed_count=failed,
        pre_rms_m=math.sqrt(sum(x * x for x in pre) / len(pre)),
        post_rms_m=math.sqrt(sum(x * x for x in post) / len(post)),
        pre_p95_m=quantile(pre, 0.95),
        post_p95_m=quantile(post, 0.95),
        mean_applied_delta_v_m_s=sum(delta_vs) / len(delta_vs),
        p95_applied_delta_v_m_s=quantile(delta_vs, 0.95),
        saturation_fraction=saturated / len(pre),
        protected_reserve_m_s=policy.total_delta_v_budget_m_s * policy.protected_reserve_fraction,
        model_ref=case.model_ref,
        constants_ref=case.constants_ref,
        frame_ref=case.frame_ref,
    )


def fixture():
    return (
        D.fixture_case(),
        D.Uncertainty(speed_sigma_m_s=0.1, elevation_sigma_deg=0.01, azimuth_sigma_deg=0.01),
        Policy(0.5, 0.5, 0.2, 0.001),
    )


def self_test() -> None:
    case, uncertainty, policy = fixture()
    a = run(case, uncertainty, policy, 42, 128)
    b = run(case, uncertainty, policy, 42, 128)
    if a != b:
        raise AssertionError("replay mismatch")
    if not a.post_rms_m < 0.15 * a.pre_rms_m:
        raise AssertionError(a)
    if a.p95_applied_delta_v_m_s > 0.4 + 1.0e-12:
        raise AssertionError("reserve consumed")

    tiny = run(case, uncertainty, Policy(0.5, 0.01, 0.2, 0.001), 42, 128)
    if tiny.saturation_fraction <= 0.0:
        raise AssertionError("tiny budget never saturated")
    if not tiny.post_rms_m < tiny.pre_rms_m:
        raise AssertionError("tiny budget not helpful")

    zero = run(case, D.Uncertainty(), policy, 42, 32)
    if zero.pre_rms_m > 1.0e-8 or zero.post_rms_m > 1.0e-8:
        raise AssertionError("zero noise did not collapse")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("LL-006C corrected-dispersion oracle self-test: PASS")
        return
    if not args.input:
        parser.error("--input or --self-test required")
    data = json.loads(Path(args.input).read_text())
    case = D.Case(**data["case"])
    uncertainty = D.Uncertainty(**data["uncertainty"])
    policy = Policy(**data["policy"])
    print(json.dumps(asdict(run(case, uncertainty, policy, int(data["seed"]), int(data["sample_count"]))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
