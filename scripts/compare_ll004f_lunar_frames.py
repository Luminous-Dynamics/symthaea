#!/usr/bin/env python3
"""Compare paired LL-004F lunar body-frame snapshots.

This dependency-free evidence tool consumes generated DE421 and DE440 LL-004F
snapshots at matching TDB epochs, computes the rigid source->destination frame
rotation, and reports relative rotation plus representative spherical South-Pole
surface displacements. It is a frame-sensitivity receipt, not terrain accuracy,
navigation, site selection, or launch qualification.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

SCHEMA_VERSION = "ll004f.lunar-frame-sensitivity.v1"
DEFAULT_RADIUS_KM = 1737.4
DEFAULT_LATITUDES_DEG = (-85.0, -88.0, -89.0, -89.9)
DEFAULT_LONGITUDES_DEG = (0.0, 90.0, 180.0, -90.0)


class CompareError(RuntimeError):
    pass


def transpose(m: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*m)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def relative_rotation_angle_rad(rotation: list[list[float]]) -> float:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    return math.acos(clamp((trace - 1.0) / 2.0, -1.0, 1.0))


def spherical_vector(radius_km: float, latitude_deg: float, longitude_deg: float) -> list[float]:
    lat = math.radians(latitude_deg)
    lon = math.radians(longitude_deg)
    c = math.cos(lat)
    return [
        radius_km * c * math.cos(lon),
        radius_km * c * math.sin(lon),
        radius_km * math.sin(lat),
    ]


def load_snapshot(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CompareError(f"expected JSON object: {path}")
    frames = value.get("frames")
    samples = value.get("samples")
    if not isinstance(frames, dict) or not isinstance(samples, list) or not samples:
        raise CompareError(f"invalid LL-004F snapshot: {path}")
    for field in ("body_fixed", "inertial"):
        if not isinstance(frames.get(field), str) or not frames[field].strip():
            raise CompareError(f"snapshot {path}: missing frames.{field}")
    return value


def matrix3(value: Any) -> list[list[float]]:
    try:
        result = [[float(value[i][j]) for j in range(3)] for i in range(3)]
    except (TypeError, ValueError, IndexError) as exc:
        raise CompareError("rotation must be 3x3") from exc
    if any(not math.isfinite(x) for row in result for x in row):
        raise CompareError("rotation contains non-finite value")
    return result


def match_samples(source: dict[str, Any], destination: dict[str, Any], tolerance_days: float) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if source["frames"]["inertial"] != destination["frames"]["inertial"]:
        raise CompareError("source and destination snapshots use different inertial frames")
    destination_samples = list(destination["samples"])
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used: set[int] = set()
    for src in source["samples"]:
        epoch = float(src.get("epoch_tdb_jd", float("nan")))
        if not math.isfinite(epoch):
            raise CompareError("source sample missing finite epoch_tdb_jd")
        candidates = [
            (abs(float(dst.get("epoch_tdb_jd", float("nan"))) - epoch), index, dst)
            for index, dst in enumerate(destination_samples)
            if index not in used and math.isfinite(float(dst.get("epoch_tdb_jd", float("nan"))))
        ]
        if not candidates:
            raise CompareError(f"no destination sample available for epoch {epoch}")
        delta, index, dst = min(candidates, key=lambda item: item[0])
        if delta > tolerance_days:
            raise CompareError(f"epoch mismatch {delta} days exceeds tolerance {tolerance_days}")
        used.add(index)
        pairs.append((src, dst))
    if len(pairs) != len(destination_samples):
        raise CompareError("source/destination sample counts do not close one-to-one")
    return pairs


def compare(
    source_path: Path,
    destination_path: Path,
    radius_km: float,
    latitudes_deg: tuple[float, ...],
    longitudes_deg: tuple[float, ...],
    epoch_tolerance_days: float,
) -> dict[str, Any]:
    if not math.isfinite(radius_km) or radius_km <= 0.0:
        raise CompareError("radius_km must be positive")
    source = load_snapshot(source_path)
    destination = load_snapshot(destination_path)
    pairs = match_samples(source, destination, epoch_tolerance_days)

    epoch_results: list[dict[str, Any]] = []
    all_displacements_m: list[float] = []
    for src, dst in pairs:
        r_src = matrix3(src.get("body_fixed_to_inertial_rotation"))
        r_dst = matrix3(dst.get("body_fixed_to_inertial_rotation"))
        bridge = matmul(transpose(r_dst), r_src)
        angle = relative_rotation_angle_rad(bridge)
        points: list[dict[str, Any]] = []
        for lat in latitudes_deg:
            for lon in longitudes_deg:
                source_vector = spherical_vector(radius_km, lat, lon)
                destination_vector = matvec(bridge, source_vector)
                displacement_m = norm(sub(destination_vector, source_vector)) * 1000.0
                all_displacements_m.append(displacement_m)
                points.append({
                    "latitude_deg": lat,
                    "longitude_deg": lon,
                    "source_radius_km": norm(source_vector),
                    "destination_radius_km": norm(destination_vector),
                    "coordinate_displacement_m": displacement_m,
                })
        epoch_results.append({
            "epoch_tdb_jd": float(src["epoch_tdb_jd"]),
            "source_epoch_input": src.get("epoch_input"),
            "destination_epoch_input": dst.get("epoch_input"),
            "relative_rotation_angle_rad": angle,
            "relative_rotation_angle_arcsec": math.degrees(angle) * 3600.0,
            "bridge_rotation_source_to_destination": bridge,
            "representative_surface_points": points,
            "max_representative_displacement_m": max(p["coordinate_displacement_m"] for p in points),
            "min_representative_displacement_m": min(p["coordinate_displacement_m"] for p in points),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source_snapshot": source_path.name,
        "destination_snapshot": destination_path.name,
        "source_lineage_id": source.get("lineage_id"),
        "destination_lineage_id": destination.get("lineage_id"),
        "source_body_fixed_frame": source["frames"]["body_fixed"],
        "destination_body_fixed_frame": destination["frames"]["body_fixed"],
        "common_inertial_frame": source["frames"]["inertial"],
        "representative_radius_km": radius_km,
        "representative_latitudes_deg": list(latitudes_deg),
        "representative_longitudes_deg": list(longitudes_deg),
        "epochs": epoch_results,
        "overall_max_representative_displacement_m": max(all_displacements_m),
        "overall_min_representative_displacement_m": min(all_displacements_m),
        "non_claims": [
            "Representative spherical points are sensitivity probes, not selected launcher sites.",
            "Coordinate displacement between frame realizations is not LOLA geolocation error or launcher pointing error.",
            "This receipt does not establish terrain, navigation, corridor, or launch qualification.",
        ],
    }


def self_test() -> None:
    import tempfile

    def snapshot(frame: str, rotation: list[list[float]]) -> dict[str, Any]:
        return {
            "lineage_id": frame,
            "frames": {"body_fixed": frame, "inertial": "J2000"},
            "samples": [{
                "epoch_input": "synthetic",
                "epoch_tdb_jd": 2460000.5,
                "body_fixed_to_inertial_rotation": rotation,
            }],
        }

    def rz(theta: float) -> list[list[float]]:
        c, s = math.cos(theta), math.sin(theta)
        return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "src.json"
        dst = root / "dst.json"
        src.write_text(json.dumps(snapshot("SRC", rz(0.0))), encoding="utf-8")
        dst.write_text(json.dumps(snapshot("DST", rz(0.0))), encoding="utf-8")
        identity = compare(src, dst, 1.0, (-90.0, -45.0), (0.0, 90.0), 1e-12)
        assert identity["overall_max_representative_displacement_m"] < 1e-9

        theta = 1e-6
        dst.write_text(json.dumps(snapshot("DST", rz(theta))), encoding="utf-8")
        shifted = compare(src, dst, 1.0, (0.0,), (0.0,), 1e-12)
        expected_m = 2.0 * 1.0 * math.sin(theta / 2.0) * 1000.0
        actual_m = shifted["epochs"][0]["max_representative_displacement_m"]
        assert abs(actual_m - expected_m) < 1e-9
        assert abs(shifted["epochs"][0]["relative_rotation_angle_rad"] - theta) < 5e-11


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--radius-km", type=float, default=DEFAULT_RADIUS_KM)
    parser.add_argument("--epoch-tolerance-days", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-004F frame comparison self-test: PASS")
        return 0
    if args.source is None or args.destination is None:
        raise CompareError("--source and --destination are required unless --self-test is used")
    result = compare(
        args.source,
        args.destination,
        args.radius_km,
        DEFAULT_LATITUDES_DEG,
        DEFAULT_LONGITUDES_DEG,
        args.epoch_tolerance_days,
    )
    payload = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CompareError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
