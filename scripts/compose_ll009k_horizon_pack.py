#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import tempfile
from typing import Any, Iterable

SCHEMA = "ll009k.horizon-materialization-input.v1"
OUT = "ll009k.horizon-pack.v1"


class KError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vec3(value: Any, label: str):
    if not isinstance(value, list) or len(value) != 3 or not all(finite(x) for x in value):
        raise KError(f"{label} must be finite vec3")
    return tuple(float(x) for x in value)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def scale(a, scalar):
    return tuple(x * scalar for x in a)


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a, label: str):
    n = norm(a)
    if n <= 1e-12 or not math.isfinite(n):
        raise KError(f"{label} degenerate")
    return scale(a, 1 / n)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def local_basis(site, pole):
    up = unit(site, "site")
    pole_u = unit(pole, "pole")
    north_raw = sub(pole_u, scale(up, dot(pole_u, up)))
    fallback = False
    if norm(north_raw) <= 1e-10:
        fallback = True
        axes = ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
        ref = min(axes, key=lambda axis: abs(dot(axis, up)))
        north_raw = sub(ref, scale(up, dot(ref, up)))
    north = unit(north_raw, "north")
    east = unit(cross(north, up), "east")
    north = unit(cross(up, east), "north")
    return north, east, up, fallback


def az_el(direction, basis):
    north, east, up, _ = basis
    direction_u = unit(direction, "line_of_sight")
    n = dot(direction_u, north)
    e = dot(direction_u, east)
    u = dot(direction_u, up)
    return (
        math.degrees(math.atan2(e, n)) % 360,
        math.degrees(math.atan2(u, math.hypot(n, e))),
    )


def safe_path(value: str):
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise KError(f"unsafe path {value}")
    return path


def validate_sources(data, artifact_root: pathlib.Path):
    sources = {}
    for source in data.get("sources", []):
        if not isinstance(source, dict):
            raise KError("source must be object")
        source_id = source.get("source_id")
        path = source.get("path")
        digest = source.get("sha256")
        if not all(isinstance(item, str) and item for item in (source_id, path, digest)):
            raise KError("bad source")
        if source_id in sources:
            raise KError(f"duplicate source {source_id}")
        full_path = artifact_root / pathlib.Path(*safe_path(path).parts)
        if not full_path.is_file() or sha256_file(full_path) != digest:
            raise KError(f"source missing/hash mismatch: {source_id}")
        sources[source_id] = {"path": path, "sha256": digest}
    if not sources:
        raise KError("no sources")
    return sources


def materialize(input_path: pathlib.Path, artifact_root: pathlib.Path):
    data = json.loads(input_path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA:
        raise KError(f"schema must be {SCHEMA}")
    for key in ("study_id", "frame_contract_id", "epoch_contract_id", "site_ref", "frame"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise KError(f"missing {key}")
    sources = validate_sources(data, artifact_root)
    site = vec3(data.get("site_position_m"), "site_position_m")
    pole = vec3(data.get("pole_vector"), "pole_vector")
    basis = local_basis(site, pole)
    up = basis[2]
    site_uncertainty = data.get("site_vertical_uncertainty_m")
    if not finite(site_uncertainty) or site_uncertainty < 0:
        raise KError("site_vertical_uncertainty_m invalid")
    bin_width = data.get("azimuth_bin_width_deg")
    if not finite(bin_width) or bin_width <= 0 or bin_width > 90:
        raise KError("azimuth_bin_width_deg invalid")
    bin_count = int(round(360 / bin_width))
    if bin_count < 4 or abs(bin_count * bin_width - 360) > 1e-9:
        raise KError("azimuth_bin_width_deg must divide 360 exactly")

    layers = data.get("layers")
    if not isinstance(layers, list) or not layers:
        raise KError("layers required")
    seen_layers = set()
    candidates = [[] for _ in range(bin_count)]
    layer_records = []

    for layer in layers:
        if not isinstance(layer, dict):
            raise KError("layer must be object")
        layer_id = layer.get("layer_id")
        source_ref = layer.get("source_ref")
        if not isinstance(layer_id, str) or not layer_id or layer_id in seen_layers:
            raise KError("layer_id invalid/duplicate")
        seen_layers.add(layer_id)
        if source_ref not in sources:
            raise KError(f"{layer_id}: unknown source_ref")
        if layer.get("frame") != data["frame"]:
            raise KError(f"{layer_id}: frame mismatch")
        minimum = layer.get("min_range_m")
        maximum = layer.get("max_range_m")
        if not finite(minimum) or not finite(maximum) or minimum < 0 or maximum <= minimum:
            raise KError(f"{layer_id}: invalid range")
        resolution = layer.get("nominal_resolution_m")
        if not finite(resolution) or resolution <= 0:
            raise KError(f"{layer_id}: invalid resolution")
        angular_margin = layer.get("additional_angular_margin_deg")
        if not finite(angular_margin) or angular_margin < 0:
            raise KError(f"{layer_id}: invalid angular margin")
        samples = layer.get("samples")
        if not isinstance(samples, list) or not samples:
            raise KError(f"{layer_id}: samples required")

        admitted = 0
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise KError(f"{layer_id}[{sample_index}]: sample object required")
            position = vec3(sample.get("position_m"), f"{layer_id}[{sample_index}].position_m")
            vertical_uncertainty = sample.get("vertical_uncertainty_m")
            if not finite(vertical_uncertainty) or vertical_uncertainty < 0:
                raise KError(f"{layer_id}[{sample_index}]: vertical uncertainty invalid")
            nominal_los = sub(position, site)
            distance = norm(nominal_los)
            if distance < minimum - 1e-9 or distance > maximum + 1e-9:
                raise KError(
                    f"{layer_id}[{sample_index}]: range {distance} outside declared layer band"
                )

            # Conservative upper skyline angle: terrain high and site low along local up.
            high_terrain = add(position, scale(up, float(vertical_uncertainty)))
            low_site = sub(site, scale(up, float(site_uncertainty)))
            conservative_los = sub(high_terrain, low_site)
            azimuth, elevation = az_el(conservative_los, basis)
            conservative_elevation = elevation + float(angular_margin)
            bin_index = int(math.floor((azimuth + 1e-9) / bin_width)) % bin_count
            candidate = {
                "layer_id": layer_id,
                "source_ref": source_ref,
                "sample_index": sample_index,
                "range_m": distance,
                "nominal_resolution_m": float(resolution),
                "nominal_elevation_deg": az_el(nominal_los, basis)[1],
                "terrain_vertical_uncertainty_m": float(vertical_uncertainty),
                "site_vertical_uncertainty_m": float(site_uncertainty),
                "additional_angular_margin_deg": float(angular_margin),
                "conservative_elevation_deg": conservative_elevation,
                "azimuth_deg": azimuth,
                "position_m": list(position),
            }
            candidates[bin_index].append(candidate)
            admitted += 1

        layer_records.append(
            {
                "layer_id": layer_id,
                "source_ref": source_ref,
                "min_range_m": float(minimum),
                "max_range_m": float(maximum),
                "nominal_resolution_m": float(resolution),
                "additional_angular_margin_deg": float(angular_margin),
                "admitted_sample_count": admitted,
            }
        )

    bins = []
    missing = []
    for index, values in enumerate(candidates):
        if not values:
            missing.append(index)
            continue
        ordered = sorted(
            values,
            key=lambda item: (
                -item["conservative_elevation_deg"],
                item["layer_id"],
                item["sample_index"],
            ),
        )
        winner = ordered[0]
        bins.append(
            {
                "bin_index": index,
                "azimuth_start_deg": index * bin_width,
                "azimuth_end_deg": (index + 1) * bin_width,
                "conservative_elevation_deg": winner["conservative_elevation_deg"],
                "winner": winner,
                "runner_up": ordered[1] if len(ordered) > 1 else None,
                "candidate_count": len(ordered),
            }
        )
    if missing:
        raise KError("missing azimuth bins: " + ",".join(map(str, missing)))

    output = {
        "schema_version": OUT,
        "status": "pass",
        "study_id": data["study_id"],
        "frame_contract_id": data["frame_contract_id"],
        "epoch_contract_id": data["epoch_contract_id"],
        "site_ref": data["site_ref"],
        "frame": data["frame"],
        "input_sha256": sha256_file(input_path),
        "source_hashes": {key: value["sha256"] for key, value in sources.items()},
        "site_position_m": list(site),
        "site_vertical_uncertainty_m": float(site_uncertainty),
        "basis": {
            "north": basis[0],
            "east": basis[1],
            "up": basis[2],
            "fallback_used": basis[3],
        },
        "azimuth_bin_width_deg": float(bin_width),
        "bin_count": bin_count,
        "layers": layer_records,
        "bins": bins,
        "semantics": (
            "Each azimuth bin retains the maximum conservative terrain elevation "
            "across all admitted layers/samples; input ordering cannot lower the envelope."
        ),
        "non_claims": [
            "This pack is terrain-horizon geometry evidence only.",
            "Raster/COG extraction and frame conversion must have their own evidence lineage.",
            "This does not qualify a site, communications link, power system, corridor, or launch operation.",
        ],
    }
    output["receipt_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def synthetic_point(site, basis, azimuth_deg, elevation_deg, distance):
    north, east, up, _ = basis
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = add(
        scale(north, math.cos(azimuth)),
        scale(east, math.sin(azimuth)),
    )
    direction = add(
        scale(horizontal, math.cos(elevation)),
        scale(up, math.sin(elevation)),
    )
    return list(add(site, scale(direction, distance)))


def self_test():
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        source = root / "source"
        source.write_text("x")
        digest = sha256_file(source)
        site = (1e6, 0, 0)
        local = local_basis(site, (0, 0, 1))

        local_samples = [
            {
                "position_m": synthetic_point(site, local, azimuth, 1, 1000),
                "vertical_uncertainty_m": 0.0,
            }
            for azimuth in (45, 135, 225, 315)
        ]
        far_samples = [
            {
                "position_m": synthetic_point(
                    site,
                    local,
                    azimuth,
                    12 if azimuth == 45 else 2,
                    10000,
                ),
                "vertical_uncertainty_m": 0.0,
            }
            for azimuth in (45, 135, 225, 315)
        ]
        data = {
            "schema_version": SCHEMA,
            "study_id": "study",
            "frame_contract_id": "frame",
            "epoch_contract_id": "epoch",
            "site_ref": "site",
            "frame": "FRAME",
            "site_position_m": list(site),
            "pole_vector": [0, 0, 1],
            "site_vertical_uncertainty_m": 0.0,
            "azimuth_bin_width_deg": 90.0,
            "sources": [
                {"source_id": "local", "path": "source", "sha256": digest},
                {"source_id": "far", "path": "source", "sha256": digest},
            ],
            "layers": [
                {
                    "layer_id": "local5",
                    "source_ref": "local",
                    "frame": "FRAME",
                    "min_range_m": 0,
                    "max_range_m": 2000,
                    "nominal_resolution_m": 5,
                    "additional_angular_margin_deg": 0,
                    "samples": local_samples,
                },
                {
                    "layer_id": "far60",
                    "source_ref": "far",
                    "frame": "FRAME",
                    "min_range_m": 2000,
                    "max_range_m": 20000,
                    "nominal_resolution_m": 60,
                    "additional_angular_margin_deg": 0,
                    "samples": far_samples,
                },
            ],
        }
        input_path = root / "input.json"
        input_path.write_bytes(canonical_bytes(data))
        result = materialize(input_path, root)
        assert len(result["bins"]) == 4
        assert result["bins"][0]["winner"]["layer_id"] == "far60"
        assert abs(result["bins"][0]["conservative_elevation_deg"] - 12) < 1e-8

        # Input order cannot change the envelope.
        reversed_data = json.loads(json.dumps(data))
        reversed_data["layers"].reverse()
        for layer in reversed_data["layers"]:
            layer["samples"].reverse()
        reversed_path = root / "reversed.json"
        reversed_path.write_bytes(canonical_bytes(reversed_data))
        reversed_result = materialize(reversed_path, root)
        assert [x["conservative_elevation_deg"] for x in result["bins"]] == [
            x["conservative_elevation_deg"] for x in reversed_result["bins"]
        ]

        # More uncertainty cannot lower the conservative skyline.
        uncertain = json.loads(json.dumps(data))
        uncertain["site_vertical_uncertainty_m"] = 10.0
        uncertain["layers"][1]["samples"][0]["vertical_uncertainty_m"] = 10.0
        uncertain_path = root / "uncertain.json"
        uncertain_path.write_bytes(canonical_bytes(uncertain))
        uncertain_result = materialize(uncertain_path, root)
        assert (
            uncertain_result["bins"][0]["conservative_elevation_deg"]
            >= result["bins"][0]["conservative_elevation_deg"]
        )

        # Missing bin fails.
        missing = json.loads(json.dumps(data))
        missing["layers"][0]["samples"] = missing["layers"][0]["samples"][:-1]
        missing["layers"][1]["samples"] = missing["layers"][1]["samples"][:-1]
        missing_path = root / "missing.json"
        missing_path.write_bytes(canonical_bytes(missing))
        try:
            materialize(missing_path, root)
            raise AssertionError("missing bin not rejected")
        except KError:
            pass

        # Sample outside declared radial layer band fails.
        bad_range = json.loads(json.dumps(data))
        bad_range["layers"][0]["max_range_m"] = 500
        bad_range_path = root / "bad-range.json"
        bad_range_path.write_bytes(canonical_bytes(bad_range))
        try:
            materialize(bad_range_path, root)
            raise AssertionError("range mismatch not rejected")
        except KError:
            pass


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--artifact-root", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.self_test:
            self_test()
            print("LL-009K core self-test: PASS")
            return 0
        if not all((args.input, args.artifact_root, args.output)):
            raise KError("input/artifact-root/output required")
        output = materialize(args.input, args.artifact_root)
        payload = canonical_bytes(output)
        if args.output.exists() and args.output.read_bytes() != payload:
            raise KError(f"refusing differing output {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
        print(json.dumps(output, sort_keys=True, indent=2))
        return 0
    except (KError, OSError, json.JSONDecodeError) as exc:
        print(f"LL-009K ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
