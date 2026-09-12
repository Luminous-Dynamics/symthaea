#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Iterable

INPUT_SCHEMA = "ll009j.horizon-visibility-input.v1"
OUTPUT_SCHEMA = "ll009j.horizon-visibility-receipt.v1"
KINDS = {"sun", "earth", "relay"}


class HorizonError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def vec3(v: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(v, list) or len(v) != 3 or not all(finite(x) for x in v):
        raise HorizonError(f"{label} must be a finite 3-vector")
    return float(v[0]), float(v[1]), float(v[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a, b):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def scale(a, s):
    return a[0] * s, a[1] * s, a[2] * s


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a, label: str):
    n = norm(a)
    if not math.isfinite(n) or n <= 1e-12:
        raise HorizonError(f"{label} has degenerate norm")
    return scale(a, 1.0 / n)


def local_basis(site, pole):
    up = unit(site, "site_position")
    p = unit(pole, "pole_vector")
    north_raw = sub(p, scale(up, dot(p, up)))
    fallback = False
    if norm(north_raw) <= 1e-10:
        fallback = True
        # Deterministic azimuth zero at/near the exact pole: project the global
        # axis least aligned with local up. The receipt records that fallback.
        axes = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        ref = min(axes, key=lambda x: abs(dot(x, up)))
        north_raw = sub(ref, scale(up, dot(ref, up)))
    north = unit(north_raw, "local_north")
    east = unit(cross(north, up), "local_east")
    north = unit(cross(up, east), "local_north_reorthogonalized")
    return north, east, up, fallback


def az_el(direction, basis):
    north, east, up, _ = basis
    d = unit(direction, "line_of_sight")
    n, e, u = dot(d, north), dot(d, east), dot(d, up)
    az = math.degrees(math.atan2(e, n)) % 360.0
    el = math.degrees(math.atan2(u, math.hypot(n, e)))
    return az, el


def circular_gap(a: float, b: float) -> float:
    return (b - a) % 360.0


def build_horizon(site, terrain_samples, basis, max_gap_deg: float):
    by_az: list[tuple[float, float, str]] = []
    for i, sample in enumerate(terrain_samples):
        if not isinstance(sample, dict):
            raise HorizonError("terrain sample must be object")
        pos = vec3(sample.get("position_m"), f"terrain[{i}].position_m")
        d = sub(pos, site)
        if norm(d) <= 1e-9:
            raise HorizonError(f"terrain[{i}] coincides with site")
        az, el = az_el(d, basis)
        margin = sample.get("angular_uncertainty_deg")
        if not finite(margin) or margin < 0:
            raise HorizonError(
                f"terrain[{i}].angular_uncertainty_deg must be nonnegative"
            )
        source_ref = sample.get("source_ref")
        if not isinstance(source_ref, str) or not source_ref:
            raise HorizonError(f"terrain[{i}] missing source_ref")
        by_az.append((az, el + float(margin), source_ref))
    if len(by_az) < 3:
        raise HorizonError("at least three terrain samples are required")
    by_az.sort()
    gaps = [
        circular_gap(by_az[i][0], by_az[(i + 1) % len(by_az)][0])
        for i in range(len(by_az))
    ]
    largest = max(gaps)
    if largest > max_gap_deg + 1e-12:
        raise HorizonError(
            f"horizon azimuth gap {largest:.6f} exceeds policy {max_gap_deg}"
        )
    return by_az, largest


def horizon_at(profile, az: float) -> float:
    a = az % 360.0
    pts = profile
    for i in range(len(pts)):
        az0, el0, _ = pts[i]
        az1, el1, _ = pts[(i + 1) % len(pts)]
        gap = circular_gap(az0, az1)
        rel = circular_gap(az0, a)
        if rel <= gap + 1e-12:
            if gap <= 1e-15:
                return max(el0, el1)
            f = min(1.0, max(0.0, rel / gap))
            return el0 + f * (el1 - el0)
    raise HorizonError("internal horizon interpolation failure")


def safe_rel(path: str) -> PurePosixPath:
    p = PurePosixPath(path)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise HorizonError(f"unsafe source path: {path!r}")
    return p


def validate_sources(data, artifact_root: Path) -> set[str]:
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise HorizonError("sources must be a non-empty list")
    ids = set()
    for src in sources:
        if not isinstance(src, dict):
            raise HorizonError("source entry must be object")
        sid, path, digest = src.get("source_id"), src.get("path"), src.get("sha256")
        if not all(isinstance(x, str) and x for x in (sid, path, digest)):
            raise HorizonError("malformed source entry")
        if sid in ids:
            raise HorizonError(f"duplicate source_id {sid}")
        ids.add(sid)
        file = artifact_root / Path(*safe_rel(path).parts)
        if not file.is_file():
            raise HorizonError(f"missing source artifact {path}")
        if sha256_file(file) != digest:
            raise HorizonError(f"source digest mismatch {path}")
    return ids


def time_visibility_interval(samples):
    if len(samples) < 2:
        raise HorizonError("at least two target samples are required for time statistics")
    times = [s["t_s"] for s in samples]
    visible = [s["visible"] for s in samples]
    total = times[-1] - times[0]
    if total <= 0:
        raise HorizonError("time span must be positive")
    low = central = high = 0.0
    for i in range(len(samples) - 1):
        dt = times[i + 1] - times[i]
        if visible[i] and visible[i + 1]:
            low += dt
            central += dt
            high += dt
        elif visible[i] != visible[i + 1]:
            # Transition time is unknown inside the sampled interval.
            central += dt / 2.0
            high += dt
    return {"low": low / total, "central": central / total, "high": high / total}


def longest_blocked_bracket(samples):
    times = [s["t_s"] for s in samples]
    vis = [s["visible"] for s in samples]
    best = {
        "lower_s": 0.0,
        "central_s": 0.0,
        "upper_s": 0.0,
        "start_index": None,
        "end_index": None,
    }
    i = 0
    while i < len(vis):
        if vis[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(vis) and not vis[j + 1]:
            j += 1
        lower = max(0.0, times[j] - times[i])
        left = (times[i] - times[i - 1]) if i > 0 else 0.0
        right = (times[j + 1] - times[j]) if j + 1 < len(times) else 0.0
        upper = lower + left + right
        central = lower + 0.5 * (left + right)
        if upper > best["upper_s"] + 1e-12:
            best = {
                "lower_s": lower,
                "central_s": central,
                "upper_s": upper,
                "start_index": i,
                "end_index": j,
            }
        i = j + 1
    return best


def validate_target_series(
    target,
    frame: str,
    max_time_gap_s: float,
    basis,
    profile,
    los_margin_deg: float,
):
    if not isinstance(target, dict):
        raise HorizonError("target must be object")
    tid, kind = target.get("target_id"), target.get("kind")
    if not isinstance(tid, str) or not tid or kind not in KINDS:
        raise HorizonError("target id/kind invalid")
    if target.get("frame") != frame:
        raise HorizonError(f"{tid}: frame mismatch")
    angular_radius = target.get("apparent_angular_radius_deg", 0.0)
    if not finite(angular_radius) or angular_radius < 0:
        raise HorizonError(f"{tid}: invalid angular radius")
    if kind == "sun" and "apparent_angular_radius_deg" not in target:
        raise HorizonError("sun target requires apparent_angular_radius_deg")
    samples = target.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise HorizonError(f"{tid}: samples must contain at least two entries")
    processed = []
    prev = None
    for idx, s in enumerate(samples):
        if not isinstance(s, dict) or not finite(s.get("t_s")):
            raise HorizonError(f"{tid}[{idx}]: invalid t_s")
        t = float(s["t_s"])
        if prev is not None:
            dt = t - prev
            if dt <= 0:
                raise HorizonError(f"{tid}: epochs must be strictly increasing")
            if dt > max_time_gap_s + 1e-12:
                raise HorizonError(
                    f"{tid}: time gap {dt} exceeds policy {max_time_gap_s}"
                )
        prev = t
        d = vec3(s.get("direction"), f"{tid}[{idx}].direction")
        az, el = az_el(d, basis)
        hz = horizon_at(profile, az) + los_margin_deg
        center = el > hz
        full = (el - angular_radius) > hz
        processed.append(
            {
                "t_s": t,
                "azimuth_deg": az,
                "elevation_deg": el,
                "conservative_horizon_deg": hz,
                "center_visible": center,
                "full_disc_visible": full,
            }
        )
    return tid, kind, angular_radius, processed


def metric_record(
    category: str,
    interval,
    unit: str,
    source_refs: list[str],
    evidence_class="derived_verified",
):
    return {
        "category": category,
        "status": "available",
        "low": interval["low"],
        "central": interval["central"],
        "high": interval["high"],
        "unit": unit,
        "evidence_class": evidence_class,
        "source_refs": source_refs,
    }


def generate(input_path: Path, artifact_root: Path) -> dict[str, Any]:
    data = json.loads(input_path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != INPUT_SCHEMA:
        raise HorizonError(f"schema_version must be {INPUT_SCHEMA}")
    for key in ("study_id", "frame_contract_id", "epoch_contract_id", "site_ref", "frame"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise HorizonError(f"missing {key}")
    source_ids = validate_sources(data, artifact_root)
    for ref in data.get("lineage_source_refs", []):
        if ref not in source_ids:
            raise HorizonError(f"unknown lineage source ref {ref}")
    site = vec3(data.get("site_position_m"), "site_position_m")
    pole = vec3(data.get("pole_vector"), "pole_vector")
    basis = local_basis(site, pole)
    policies = data.get("policies")
    if not isinstance(policies, dict):
        raise HorizonError("missing policies")
    max_az = policies.get("max_horizon_gap_deg")
    max_t = policies.get("max_time_gap_s")
    margin = policies.get("los_margin_deg")
    if not finite(max_az) or not 0 < max_az <= 180:
        raise HorizonError("invalid max_horizon_gap_deg")
    if not finite(max_t) or max_t <= 0:
        raise HorizonError("invalid max_time_gap_s")
    if not finite(margin) or margin < 0:
        raise HorizonError("invalid los_margin_deg")
    terrain = data.get("terrain_samples")
    if not isinstance(terrain, list):
        raise HorizonError("terrain_samples must be list")
    for sample in terrain:
        if isinstance(sample, dict) and sample.get("source_ref") not in source_ids:
            raise HorizonError("terrain sample source_ref unknown")
    profile, largest_gap = build_horizon(site, terrain, basis, float(max_az))
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        raise HorizonError("targets must be non-empty list")
    metrics = {}
    target_receipts = []
    for target in targets:
        refs = target.get("source_refs", []) if isinstance(target, dict) else []
        if not isinstance(refs, list) or not refs or any(r not in source_ids for r in refs):
            raise HorizonError("target source_refs invalid")
        tid, kind, radius, rows = validate_target_series(
            target, data["frame"], float(max_t), basis, profile, float(margin)
        )
        center_rows = [{"t_s": r["t_s"], "visible": r["center_visible"]} for r in rows]
        full_rows = [{"t_s": r["t_s"], "visible": r["full_disc_visible"]} for r in rows]
        center_int = time_visibility_interval(center_rows)
        center_block = longest_blocked_bracket(center_rows)
        if kind == "sun":
            full_int = time_visibility_interval(full_rows)
            full_block = longest_blocked_bracket(full_rows)
            metrics["solar_center_visibility_fraction"] = metric_record(
                "illumination", center_int, "fraction", refs
            )
            metrics["solar_full_disc_visibility_fraction"] = metric_record(
                "illumination", full_int, "fraction", refs
            )
            metrics["longest_full_solar_occlusion_h"] = metric_record(
                "illumination",
                {
                    "low": full_block["lower_s"] / 3600,
                    "central": full_block["central_s"] / 3600,
                    "high": full_block["upper_s"] / 3600,
                },
                "h",
                refs,
            )
        elif kind == "earth":
            metrics["dte_los_availability_fraction"] = metric_record(
                "communications", center_int, "fraction", refs
            )
            metrics["max_contiguous_dte_outage_h"] = metric_record(
                "communications",
                {
                    "low": center_block["lower_s"] / 3600,
                    "central": center_block["central_s"] / 3600,
                    "high": center_block["upper_s"] / 3600,
                },
                "h",
                refs,
            )
        else:
            key = tid.lower().replace("-", "_")
            metrics[f"{key}_los_availability_fraction"] = metric_record(
                "communications", center_int, "fraction", refs
            )
            metrics[f"max_contiguous_{key}_outage_h"] = metric_record(
                "communications",
                {
                    "low": center_block["lower_s"] / 3600,
                    "central": center_block["central_s"] / 3600,
                    "high": center_block["upper_s"] / 3600,
                },
                "h",
                refs,
            )
        target_receipts.append(
            {
                "target_id": tid,
                "kind": kind,
                "apparent_angular_radius_deg": radius,
                "samples": rows,
                "center_visibility_interval": center_int,
                "longest_center_blocked_s": center_block,
                "full_disc_visibility_interval": time_visibility_interval(full_rows)
                if kind == "sun"
                else None,
                "longest_full_disc_blocked_s": longest_blocked_bracket(full_rows)
                if kind == "sun"
                else None,
            }
        )
    hzvals = [x[1] for x in profile]
    output = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "pass",
        "study_id": data["study_id"],
        "frame_contract_id": data["frame_contract_id"],
        "epoch_contract_id": data["epoch_contract_id"],
        "site_ref": data["site_ref"],
        "frame": data["frame"],
        "input_sha256": sha256_file(input_path),
        "source_artifact_hashes": {s["source_id"]: s["sha256"] for s in data["sources"]},
        "basis": {
            "north": basis[0],
            "east": basis[1],
            "up": basis[2],
            "fallback_used": basis[3],
        },
        "horizon": {
            "sample_count": len(profile),
            "largest_azimuth_gap_deg": largest_gap,
            "min_conservative_elevation_deg": min(hzvals),
            "max_conservative_elevation_deg": max(hzvals),
            "los_margin_deg": float(margin),
            "max_allowed_gap_deg": float(max_az),
        },
        "metrics": metrics,
        "targets": target_receipts,
        "sampling_semantics": (
            "Visibility fractions and blocked-duration intervals bracket unknown "
            "transition times between samples; no sub-sample transition timing is inferred."
        ),
        "non_claims": [
            "LOS geometry is not an RF link budget or communications qualification.",
            "Solar-disc visibility is not delivered electrical power or a thermal model.",
            "This receipt does not select/qualify a site, corridor, vehicle, or launch operation.",
        ],
    }
    output["receipt_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def terrain_point(site, basis, az_deg, el_deg, dist):
    n, e, u, _ = basis
    az, el = math.radians(az_deg), math.radians(el_deg)
    horiz = add(scale(n, math.cos(az)), scale(e, math.sin(az)))
    los = add(scale(horiz, math.cos(el)), scale(u, math.sin(el)))
    return list(add(site, scale(los, dist)))


def direction_from_azel(basis, az_deg, el_deg):
    n, e, u, _ = basis
    az, el = math.radians(az_deg), math.radians(el_deg)
    horiz = add(scale(n, math.cos(az)), scale(e, math.sin(az)))
    return list(add(scale(horiz, math.cos(el)), scale(u, math.sin(el))))


def self_test():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source.json"
        source.write_text("{}")
        sd = sha256_file(source)
        site = (1_000_000.0, 0, 0)
        basis = local_basis(site, (0, 0, 1))
        # Near/exact-pole fallback is deterministic, finite, and orthonormal.
        pb = local_basis((0, 0, 1_000_000.0), (0, 0, 1))
        for axis in pb[:3]:
            assert abs(norm(axis) - 1) < 1e-12
        assert abs(dot(pb[0], pb[1])) < 1e-12 and pb[3]
        terrain = []
        for az in range(0, 360, 30):
            el = 15.0 if az == 90 else (0.75 if az == 180 else 0.0)
            terrain.append(
                {
                    "position_m": terrain_point(site, basis, az, el, 1000),
                    "angular_uncertainty_deg": 0.0,
                    "source_ref": "terrain",
                }
            )
        targets = [
            {
                "target_id": "SUN",
                "kind": "sun",
                "frame": "FRAME",
                "apparent_angular_radius_deg": 0.5,
                "source_refs": ["ephem"],
                "samples": [
                    {"t_s": 0.0, "direction": direction_from_azel(basis, 0, 10)},
                    {"t_s": 3600.0, "direction": direction_from_azel(basis, 90, 10)},
                    {"t_s": 7200.0, "direction": direction_from_azel(basis, 180, 1.0)},
                    {"t_s": 10800.0, "direction": direction_from_azel(basis, 270, 0.4)},
                ],
            },
            {
                "target_id": "EARTH",
                "kind": "earth",
                "frame": "FRAME",
                "source_refs": ["ephem"],
                "samples": [
                    {"t_s": 0.0, "direction": direction_from_azel(basis, 0, 10)},
                    {"t_s": 3600.0, "direction": direction_from_azel(basis, 90, 10)},
                    {"t_s": 7200.0, "direction": direction_from_azel(basis, 180, 10)},
                    {"t_s": 10800.0, "direction": direction_from_azel(basis, 270, 10)},
                ],
            },
        ]
        data = {
            "schema_version": INPUT_SCHEMA,
            "study_id": "study",
            "frame_contract_id": "frame",
            "epoch_contract_id": "epoch",
            "site_ref": "site",
            "frame": "FRAME",
            "site_position_m": list(site),
            "pole_vector": [0, 0, 1],
            "sources": [
                {"source_id": "terrain", "path": "source.json", "sha256": sd},
                {"source_id": "ephem", "path": "source.json", "sha256": sd},
            ],
            "lineage_source_refs": ["terrain", "ephem"],
            "policies": {
                "max_horizon_gap_deg": 30.0,
                "max_time_gap_s": 3600.0,
                "los_margin_deg": 0.0,
            },
            "terrain_samples": terrain,
            "targets": targets,
        }
        inp = root / "in.json"
        inp.write_bytes(canonical_bytes(data))
        out = generate(inp, root)
        sun = next(t for t in out["targets"] if t["kind"] == "sun")
        assert sun["samples"][0]["center_visible"]
        assert not sun["samples"][1]["center_visible"]
        # Center 1°, apparent radius .5°, horizon .75°: center visible, full disc hidden.
        assert sun["samples"][2]["center_visible"] and not sun["samples"][2]["full_disc_visible"]
        # Earth visibility T,F,T,T yields conservative temporal availability [1/3, 2/3, 1].
        earth = next(t for t in out["targets"] if t["kind"] == "earth")
        iv = earth["center_visibility_interval"]
        assert abs(iv["low"] - 1 / 3) < 1e-12
        assert abs(iv["central"] - 2 / 3) < 1e-12
        assert abs(iv["high"] - 1.0) < 1e-12
        # Missing horizon sector/gap must fail.
        bad = json.loads(json.dumps(data))
        bad["terrain_samples"] = bad["terrain_samples"][:-1]
        bp = root / "bad.json"
        bp.write_bytes(canonical_bytes(bad))
        try:
            generate(bp, root)
            raise AssertionError("gap not rejected")
        except HorizonError:
            pass
        # Excessive time gap must fail.
        bad2 = json.loads(json.dumps(data))
        bad2["targets"][0]["samples"][1]["t_s"] = 4000.0
        b2 = root / "bad2.json"
        b2.write_bytes(canonical_bytes(bad2))
        try:
            generate(b2, root)
            raise AssertionError("time gap not rejected")
        except HorizonError:
            pass


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.self_test:
            self_test()
            print("LL-009J self-test: PASS")
            return 0
        if not all((args.input, args.artifact_root, args.output)):
            raise HorizonError("--input, --artifact-root, --output required")
        out = generate(args.input, args.artifact_root)
        payload = canonical_bytes(out)
        if args.output.exists() and args.output.read_bytes() != payload:
            raise HorizonError(f"refusing to overwrite differing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        print(json.dumps(out, sort_keys=True, indent=2))
        return 0
    except (HorizonError, OSError, json.JSONDecodeError) as exc:
        print(f"LL-009J ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
