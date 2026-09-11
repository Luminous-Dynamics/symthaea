#!/usr/bin/env python3
"""Independent LL-005 seeded ballistic dispersion/sensitivity oracle.

Builds on the independent LL-003B spherical two-body oracle in the same scripts
directory. This is a research uncertainty propagator, not launch authorization
or a claim about real lunar launcher tolerances.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

Vec3 = tuple[float, float, float]
MASK64 = (1 << 64) - 1


def _load_ballistics():
    path = Path(__file__).with_name("ll-spherical-ballistic-oracle.py")
    spec = importlib.util.spec_from_file_location("ll_spherical_ballistic_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load ballistic oracle: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BALLISTICS = _load_ballistics()


class SplitMix64:
    """Small explicitly specified PRNG for cross-language replay."""

    def __init__(self, seed: int):
        self.state = seed & MASK64
        self._spare: float | None = None

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def uniform_open(self) -> float:
        return ((self.next_u64() >> 11) + 0.5) / float(1 << 53)

    def normal(self) -> float:
        if self._spare is not None:
            out = self._spare
            self._spare = None
            return out
        u1 = self.uniform_open()
        u2 = self.uniform_open()
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        self._spare = radius * math.sin(angle)
        return radius * math.cos(angle)


@dataclass(frozen=True)
class Uncertainty:
    speed_sigma_m_s: float = 0.0
    elevation_sigma_deg: float = 0.0
    azimuth_sigma_deg: float = 0.0
    site_north_sigma_m: float = 0.0
    site_east_sigma_m: float = 0.0

    def validate(self) -> None:
        values = asdict(self).values()
        if not all(math.isfinite(v) and v >= 0.0 for v in values):
            raise ValueError("all uncertainty sigmas must be finite and nonnegative")


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

    def validate(self) -> None:
        BALLISTICS.validate(
            self.mu_m3_s2,
            self.radius_m,
            self.speed_m_s,
            self.elevation_deg,
            self.azimuth_deg,
            self.dt_s,
            self.max_time_s,
        )
        if not (-90.0 <= self.latitude_deg <= 90.0):
            raise ValueError("invalid latitude")
        if not math.isfinite(self.longitude_deg):
            raise ValueError("invalid longitude")
        if not self.model_ref.strip() or not self.constants_ref.strip() or not self.frame_ref.strip():
            raise ValueError("model/constants/frame provenance must be nonempty")


@dataclass(frozen=True)
class DispersionSummary:
    seed: int
    sample_count: int
    uncertainty: dict[str, float]
    model_ref: str
    constants_ref: str
    frame_ref: str
    nominal_outcome: str
    reimpact_count: int
    escape_count: int
    no_return_count: int
    mean_east_m: float | None
    mean_north_m: float | None
    covariance_ee_m2: float | None
    covariance_en_m2: float | None
    covariance_nn_m2: float | None
    rms_radial_miss_m: float | None
    radial_p50_m: float | None
    radial_p95_m: float | None
    radial_p99_m: float | None
    mean_flight_time_s: float | None
    mean_ground_range_m: float | None
    mean_max_altitude_m: float | None


def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vec3) -> Vec3:
    n = norm(a)
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError("cannot normalize vector")
    return vec_scale(a, 1.0 / n)


def unit_from_lat_lon(latitude_deg: float, longitude_deg: float) -> Vec3:
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    return (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))


def lat_lon_from_unit(u: Vec3) -> tuple[float, float]:
    u = normalize(u)
    lat = math.degrees(math.asin(max(-1.0, min(1.0, u[2]))))
    lon = math.degrees(math.atan2(u[1], u[0]))
    return lat, lon


def local_basis(latitude_deg: float, longitude_deg: float) -> tuple[Vec3, Vec3, Vec3]:
    north, east, up = BALLISTICS.surface_frame(latitude_deg, longitude_deg)
    return north, east, up


def displaced_site(
    latitude_deg: float,
    longitude_deg: float,
    radius_m: float,
    north_m: float,
    east_m: float,
) -> tuple[float, float]:
    north, east, up = local_basis(latitude_deg, longitude_deg)
    shifted = vec_add(
        up,
        vec_add(vec_scale(north, north_m / radius_m), vec_scale(east, east_m / radius_m)),
    )
    return lat_lon_from_unit(shifted)


def propagate(case: Case, *, speed=None, elevation=None, azimuth=None, latitude=None, longitude=None):
    return BALLISTICS.propagate(
        case.mu_m3_s2,
        case.radius_m,
        case.speed_m_s if speed is None else speed,
        case.elevation_deg if elevation is None else elevation,
        case.azimuth_deg if azimuth is None else azimuth,
        case.latitude_deg if latitude is None else latitude,
        case.longitude_deg if longitude is None else longitude,
        case.dt_s,
        case.max_time_s,
    )


def tangent_offset_m(
    nominal_lat_deg: float,
    nominal_lon_deg: float,
    sample_lat_deg: float,
    sample_lon_deg: float,
    radius_m: float,
) -> tuple[float, float]:
    north, east, up = local_basis(nominal_lat_deg, nominal_lon_deg)
    sample_u = unit_from_lat_lon(sample_lat_deg, sample_lon_deg)
    denom = dot(sample_u, up)
    if not math.isfinite(denom) or denom <= 0.0:
        raise ValueError("sample impact is outside nominal arrival tangent hemisphere")
    east_m = radius_m * dot(sample_u, east) / denom
    north_m = radius_m * dot(sample_u, north) / denom
    return east_m, north_m


def quantile_sorted(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty quantile")
    if not (0.0 <= q <= 1.0):
        raise ValueError("invalid quantile")
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return values[lo]
    w = position - lo
    return values[lo] * (1.0 - w) + values[hi] * w


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def run_dispersion(case: Case, uncertainty: Uncertainty, seed: int, sample_count: int) -> DispersionSummary:
    case.validate()
    uncertainty.validate()
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not (0 <= seed <= MASK64):
        raise ValueError("seed must fit unsigned 64-bit range")

    nominal = propagate(case)
    if nominal.outcome != "reimpact":
        raise ValueError("dispersion reference currently requires a nominal reimpact")
    assert nominal.arrival_latitude_deg is not None
    assert nominal.arrival_longitude_deg is not None

    rng = SplitMix64(seed)
    east_offsets: list[float] = []
    north_offsets: list[float] = []
    radial_offsets: list[float] = []
    times: list[float] = []
    ranges: list[float] = []
    altitudes: list[float] = []
    reimpact = escape = no_return = 0

    for _ in range(sample_count):
        speed = case.speed_m_s + uncertainty.speed_sigma_m_s * rng.normal()
        elevation = case.elevation_deg + uncertainty.elevation_sigma_deg * rng.normal()
        azimuth = case.azimuth_deg + uncertainty.azimuth_sigma_deg * rng.normal()
        north_shift = uncertainty.site_north_sigma_m * rng.normal()
        east_shift = uncertainty.site_east_sigma_m * rng.normal()
        latitude, longitude = displaced_site(
            case.latitude_deg, case.longitude_deg, case.radius_m, north_shift, east_shift
        )

        if speed <= 0.0 or not (0.0 < elevation < 90.0):
            no_return += 1
            continue

        result = propagate(
            case,
            speed=speed,
            elevation=elevation,
            azimuth=azimuth,
            latitude=latitude,
            longitude=longitude,
        )
        if result.outcome == "escape":
            escape += 1
            continue
        if result.outcome != "reimpact":
            no_return += 1
            continue

        reimpact += 1
        assert result.arrival_latitude_deg is not None
        assert result.arrival_longitude_deg is not None
        east_m, north_m = tangent_offset_m(
            nominal.arrival_latitude_deg,
            nominal.arrival_longitude_deg,
            result.arrival_latitude_deg,
            result.arrival_longitude_deg,
            case.radius_m,
        )
        east_offsets.append(east_m)
        north_offsets.append(north_m)
        radial_offsets.append(math.hypot(east_m, north_m))
        times.append(float(result.flight_time_s))
        ranges.append(float(result.ground_range_m))
        altitudes.append(float(result.max_altitude_m))

    if reimpact + escape + no_return != sample_count:
        raise RuntimeError("sample accounting failure")

    if not east_offsets:
        mean_e = mean_n = cov_ee = cov_en = cov_nn = rms = None
        p50 = p95 = p99 = mean_t = mean_r = mean_h = None
    else:
        mean_e = mean(east_offsets)
        mean_n = mean(north_offsets)
        if len(east_offsets) > 1:
            denom = len(east_offsets) - 1
            cov_ee = sum((x - mean_e) ** 2 for x in east_offsets) / denom
            cov_nn = sum((y - mean_n) ** 2 for y in north_offsets) / denom
            cov_en = sum((x - mean_e) * (y - mean_n) for x, y in zip(east_offsets, north_offsets)) / denom
        else:
            cov_ee = cov_nn = cov_en = 0.0
        rms = math.sqrt(mean([r * r for r in radial_offsets]))
        radial_sorted = sorted(radial_offsets)
        p50 = quantile_sorted(radial_sorted, 0.50)
        p95 = quantile_sorted(radial_sorted, 0.95)
        p99 = quantile_sorted(radial_sorted, 0.99)
        mean_t = mean(times)
        mean_r = mean(ranges)
        mean_h = mean(altitudes)

    return DispersionSummary(
        seed=seed,
        sample_count=sample_count,
        uncertainty=asdict(uncertainty),
        model_ref=case.model_ref,
        constants_ref=case.constants_ref,
        frame_ref=case.frame_ref,
        nominal_outcome=nominal.outcome,
        reimpact_count=reimpact,
        escape_count=escape,
        no_return_count=no_return,
        mean_east_m=mean_e,
        mean_north_m=mean_n,
        covariance_ee_m2=cov_ee,
        covariance_en_m2=cov_en,
        covariance_nn_m2=cov_nn,
        rms_radial_miss_m=rms,
        radial_p50_m=p50,
        radial_p95_m=p95,
        radial_p99_m=p99,
        mean_flight_time_s=mean_t,
        mean_ground_range_m=mean_r,
        mean_max_altitude_m=mean_h,
    )


def sensitivity(case: Case, base: Uncertainty, seed: int, sample_count: int) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    fields = asdict(base)
    for field in fields:
        kwargs = {name: 0.0 for name in fields}
        kwargs[field] = getattr(base, field)
        summary = run_dispersion(case, Uncertainty(**kwargs), seed, sample_count)
        out[field] = summary.rms_radial_miss_m
    return out


def fixture_case(azimuth_deg: float = 90.0) -> Case:
    return Case(
        mu_m3_s2=2.0e12,
        radius_m=1.0e6,
        speed_m_s=100.0,
        elevation_deg=45.0,
        azimuth_deg=azimuth_deg,
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
    zero = Uncertainty()
    a = run_dispersion(case, zero, 0x123456789ABCDEF0, 32)
    b = run_dispersion(case, zero, 0x123456789ABCDEF0, 32)
    if a != b:
        raise AssertionError("deterministic replay failed")
    if a.reimpact_count != 32 or a.escape_count or a.no_return_count:
        raise AssertionError("zero-noise outcome accounting failed")
    if a.rms_radial_miss_m is None or a.rms_radial_miss_m > 1.0e-9:
        raise AssertionError(f"zero-noise dispersion did not collapse: {a.rms_radial_miss_m}")

    small = run_dispersion(case, Uncertainty(speed_sigma_m_s=0.05), 1234, 256)
    large = run_dispersion(case, Uncertainty(speed_sigma_m_s=0.20), 1234, 256)
    if not (
        small.rms_radial_miss_m is not None
        and large.rms_radial_miss_m is not None
        and large.rms_radial_miss_m > 3.0 * small.rms_radial_miss_m
    ):
        raise AssertionError("speed uncertainty monotonic synthetic fixture failed")

    az = run_dispersion(case, Uncertainty(azimuth_sigma_deg=0.02), 999, 256)
    if az.rms_radial_miss_m is None or az.rms_radial_miss_m <= 0.0:
        raise AssertionError("azimuth uncertainty produced no dispersion")

    pole_case = Case(**{**asdict(case), "latitude_deg": -89.9})
    pole = run_dispersion(
        pole_case,
        Uncertainty(site_north_sigma_m=1.0, site_east_sigma_m=1.0),
        77,
        64,
    )
    if pole.reimpact_count != 64 or pole.rms_radial_miss_m is None:
        raise AssertionError("near-pole site uncertainty failed")

    nominal1 = propagate(case)
    nominal2 = propagate(case)
    if nominal1 != nominal2:
        raise AssertionError("identical release state changed trajectory")

    sens = sensitivity(
        case,
        Uncertainty(
            speed_sigma_m_s=0.1,
            elevation_sigma_deg=0.01,
            azimuth_sigma_deg=0.01,
            site_north_sigma_m=0.5,
            site_east_sigma_m=0.5,
        ),
        42,
        128,
    )
    if any(v is None or v < 0.0 for v in sens.values()):
        raise AssertionError("invalid sensitivity output")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", help="JSON file with case, uncertainty, seed, sample_count")
    parser.add_argument("--sensitivity", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("LL-005 dispersion oracle self-test: PASS")
        return

    if not args.input:
        parser.error("--input or --self-test is required")

    payload = json.loads(Path(args.input).read_text())
    case = Case(**payload["case"])
    uncertainty = Uncertainty(**payload["uncertainty"])
    seed = int(payload["seed"])
    sample_count = int(payload["sample_count"])

    if args.sensitivity:
        print(json.dumps(sensitivity(case, uncertainty, seed, sample_count), indent=2, sort_keys=True))
    else:
        print(json.dumps(asdict(run_dispersion(case, uncertainty, seed, sample_count)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
