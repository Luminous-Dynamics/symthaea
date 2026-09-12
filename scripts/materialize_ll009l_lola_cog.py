#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
from typing import Any

SCHEMA = "ll009l.cog-materialization-config.v1"
OUT = "ll009l.terrain-sample-pack.v1"
K_SCHEMA = "ll009k.horizon-materialization-input.v1"


class LError(RuntimeError):
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
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(finite(x) for x in value):
        raise LError(f"{label} must be finite vec3")
    return tuple(float(x) for x in value)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, scalar):
    return tuple(x * scalar for x in a)


def norm(a):
    return math.sqrt(dot(a, a))


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def unit(a, label: str):
    magnitude = norm(a)
    if magnitude <= 1e-12 or not math.isfinite(magnitude):
        raise LError(f"{label} degenerate")
    return scale(a, 1 / magnitude)


def local_basis(site, pole):
    up = unit(site, "site")
    pole_u = unit(pole, "pole")
    north_raw = sub(pole_u, scale(up, dot(pole_u, up)))
    fallback = False
    if norm(north_raw) <= 1e-10:
        fallback = True
        axes = ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))
        reference = min(axes, key=lambda axis: abs(dot(axis, up)))
        north_raw = sub(reference, scale(up, dot(reference, up)))
    north = unit(north_raw, "north")
    east = unit(cross(north, up), "east")
    north = unit(cross(up, east), "north")
    return north, east, up, fallback


def az_el(direction, basis):
    north, east, up, _ = basis
    direction_u = unit(direction, "line_of_sight")
    north_component = dot(direction_u, north)
    east_component = dot(direction_u, east)
    up_component = dot(direction_u, up)
    return (
        math.degrees(math.atan2(east_component, north_component)) % 360,
        math.degrees(math.atan2(up_component, math.hypot(north_component, east_component))),
    )


def safe_relpath(value: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise LError(f"unsafe path {value}")
    return path


def resolve_hashed(root: pathlib.Path, entry: dict, label: str) -> pathlib.Path:
    path = entry.get("path")
    digest = entry.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise LError(f"{label}: path/sha256 required")
    full = root / pathlib.Path(*safe_relpath(path).parts)
    if not full.is_file():
        raise LError(f"{label}: missing file")
    actual = sha256_file(full)
    if actual != digest:
        raise LError(f"{label}: source hash mismatch")
    return full


def lonlat_to_xyz(lon_deg: float, lat_deg: float, radius_m: float):
    longitude = math.radians(lon_deg)
    latitude = math.radians(lat_deg)
    cos_latitude = math.cos(latitude)
    return (
        radius_m * cos_latitude * math.cos(longitude),
        radius_m * cos_latitude * math.sin(longitude),
        radius_m * math.sin(latitude),
    )


def require_rasterio():
    try:
        import rasterio
        from rasterio.warp import transform as warp_transform
    except Exception as exc:
        raise LError("Rasterio is required only for LL-009L offline raster materialization") from exc
    return rasterio, warp_transform


def same_affine(left, right, tolerance=1e-12):
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def observed_meta(dataset):
    wkt = dataset.crs.to_wkt() if dataset.crs else None
    return {
        "width": dataset.width,
        "height": dataset.height,
        "count": dataset.count,
        "dtype": dataset.dtypes[0] if dataset.count else None,
        "nodata": dataset.nodata,
        "transform": list(dataset.transform)[:6],
        "crs_wkt_sha256": sha256_bytes((wkt or "").encode()),
        "crs_wkt": wkt,
        "block_shapes": [list(value) for value in dataset.block_shapes],
    }


def validate_config(data):
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA:
        raise LError(f"schema must be {SCHEMA}")
    for key in ("study_id", "frame_contract_id", "epoch_contract_id", "site_ref", "native_frame"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise LError(f"missing {key}")
    projection = data.get("projection_contract")
    if not isinstance(projection, dict):
        raise LError("projection_contract required")
    radius = projection.get("reference_radius_m")
    if not finite(radius) or radius <= 0:
        raise LError("reference_radius_m invalid")
    if projection.get("projection") != "south_polar_stereographic":
        raise LError("V1 requires south_polar_stereographic")
    if projection.get("true_scale_at_pole") is not True:
        raise LError("V1 requires true_scale_at_pole=true")
    if not finite(projection.get("central_meridian_deg")):
        raise LError("central_meridian_deg required")
    site = data.get("site")
    if not isinstance(site, dict) or not all(
        finite(site.get(key)) for key in ("x_m", "y_m", "elevation_m", "vertical_uncertainty_m")
    ):
        raise LError("site x/y/elevation/vertical_uncertainty required")
    pole = vec3(data.get("pole_vector"), "pole_vector")
    bin_width = data.get("azimuth_bin_width_deg")
    if not finite(bin_width) or bin_width <= 0 or bin_width > 90:
        raise LError("azimuth_bin_width_deg invalid")
    bin_count = int(round(360 / float(bin_width)))
    if bin_count < 4 or abs(bin_count * float(bin_width) - 360) > 1e-9:
        raise LError("azimuth_bin_width_deg must divide 360")
    top_k = data.get("top_k_per_bin", 3)
    if not isinstance(top_k, int) or top_k < 1 or top_k > 16:
        raise LError("top_k_per_bin invalid")
    layers = data.get("layers")
    if not isinstance(layers, list) or not layers:
        raise LError("layers required")
    return float(radius), pole, float(bin_width), bin_count, top_k


def dataset_contract(dataset, layer, role):
    if dataset.count != 1:
        raise LError(f"{layer['layer_id']}:{role}: exactly one band required")
    expected_crs = layer.get("expected_crs_wkt_sha256")
    metadata = observed_meta(dataset)
    if not isinstance(expected_crs, str) or metadata["crs_wkt_sha256"] != expected_crs:
        raise LError(f"{layer['layer_id']}:{role}: CRS WKT hash mismatch")
    expected_pixel_size = layer.get("expected_pixel_size_m")
    if finite(expected_pixel_size):
        a, b, _c, d, e, _f = list(dataset.transform)[:6]
        x_scale = math.hypot(a, d)
        y_scale = math.hypot(b, e)
        if abs(x_scale - float(expected_pixel_size)) > 1e-8 or abs(y_scale - float(expected_pixel_size)) > 1e-8:
            raise LError(f"{layer['layer_id']}:{role}: pixel size mismatch")
    return metadata


def aligned(left, right, label):
    if left.width != right.width or left.height != right.height:
        raise LError(f"{label}: shape mismatch")
    if left.crs != right.crs:
        raise LError(f"{label}: CRS mismatch")
    if not same_affine(left.transform, right.transform):
        raise LError(f"{label}: transform mismatch")


def update_top(best_list, candidate, top_k):
    best_list.append(candidate)
    best_list.sort(
        key=lambda item: (
            -item["selection_conservative_elevation_deg"],
            item["row"],
            item["col"],
        )
    )
    if len(best_list) > top_k:
        del best_list[top_k:]


def materialize(config_path: pathlib.Path, artifact_root: pathlib.Path):
    rasterio, warp_transform = require_rasterio()
    data = json.loads(config_path.read_text())
    radius, pole, bin_width, bin_count, top_k = validate_config(data)
    geographic_crs = rasterio.crs.CRS.from_proj4(
        f"+proj=longlat +R={radius:.12f} +no_defs +type=crs"
    )
    site_config = data["site"]
    site_xy = (float(site_config["x_m"]), float(site_config["y_m"]))
    site_elevation = float(site_config["elevation_m"])
    site_uncertainty = float(site_config["vertical_uncertainty_m"])
    output_layers = []
    source_hashes = {}
    native_crs_wkt = None
    site_position = None
    basis = None

    for layer in data["layers"]:
        if not isinstance(layer, dict) or not isinstance(layer.get("layer_id"), str):
            raise LError("layer object/layer_id required")
        layer_id = layer["layer_id"]
        minimum = layer.get("min_range_m")
        maximum = layer.get("max_range_m")
        if not finite(minimum) or not finite(maximum) or minimum < 0 or maximum <= minimum:
            raise LError(f"{layer_id}: invalid radial range")
        max_effective_resolution = layer.get("max_effective_resolution_m")
        if max_effective_resolution is not None and (
            not finite(max_effective_resolution) or max_effective_resolution <= 0
        ):
            raise LError(f"{layer_id}: max_effective_resolution_m invalid")

        elevation_path = resolve_hashed(
            artifact_root, layer.get("elevation_source", {}), f"{layer_id}:elevation"
        )
        uncertainty_path = resolve_hashed(
            artifact_root, layer.get("uncertainty_source", {}), f"{layer_id}:uncertainty"
        )
        effective_entry = layer.get("effective_resolution_source")
        effective_path = (
            resolve_hashed(artifact_root, effective_entry, f"{layer_id}:effective_resolution")
            if isinstance(effective_entry, dict)
            else None
        )
        source_hashes[f"{layer_id}:elevation"] = sha256_file(elevation_path)
        source_hashes[f"{layer_id}:uncertainty"] = sha256_file(uncertainty_path)
        if effective_path:
            source_hashes[f"{layer_id}:effective_resolution"] = sha256_file(effective_path)

        with rasterio.open(elevation_path) as elevation, rasterio.open(uncertainty_path) as uncertainty:
            effective = rasterio.open(effective_path) if effective_path else None
            try:
                aligned(elevation, uncertainty, f"{layer_id}:elevation/uncertainty")
                if effective:
                    aligned(elevation, effective, f"{layer_id}:elevation/effective_resolution")
                elevation_metadata = dataset_contract(elevation, layer, "elevation")
                dataset_contract(uncertainty, layer, "uncertainty")
                if effective:
                    dataset_contract(effective, layer, "effective_resolution")
                wkt = elevation.crs.to_wkt()
                if native_crs_wkt is None:
                    native_crs_wkt = wkt
                    longitudes, latitudes = warp_transform(
                        elevation.crs,
                        geographic_crs,
                        [site_xy[0]],
                        [site_xy[1]],
                    )
                    site_position = lonlat_to_xyz(
                        longitudes[0], latitudes[0], radius + site_elevation
                    )
                    basis = local_basis(site_position, pole)
                elif wkt != native_crs_wkt:
                    raise LError(f"{layer_id}: CRS differs from other layers")

                best = [[] for _ in range(bin_count)]
                scanned_pixels = 0
                admitted_pixels = 0
                nodata_pixels = 0
                bounding_box = (
                    site_xy[0] - maximum,
                    site_xy[1] - maximum,
                    site_xy[0] + maximum,
                    site_xy[1] + maximum,
                )
                for _, window in elevation.block_windows(1):
                    left, bottom, right, top = rasterio.windows.bounds(window, elevation.transform)
                    if (
                        right < bounding_box[0]
                        or left > bounding_box[2]
                        or top < bounding_box[1]
                        or bottom > bounding_box[3]
                    ):
                        continue
                    elevation_array = elevation.read(1, window=window, masked=True)
                    uncertainty_array = uncertainty.read(1, window=window, masked=True)
                    effective_array = effective.read(1, window=window, masked=True) if effective else None
                    row0, col0 = int(window.row_off), int(window.col_off)
                    selected_xy = []
                    selected_records = []
                    for local_row in range(elevation_array.shape[0]):
                        for local_col in range(elevation_array.shape[1]):
                            row = row0 + local_row
                            col = col0 + local_col
                            x, y = elevation.transform * (col + 0.5, row + 0.5)
                            projected_range = math.hypot(x - site_xy[0], y - site_xy[1])
                            if projected_range < minimum - 1e-9 or projected_range > maximum + 1e-9:
                                continue
                            scanned_pixels += 1
                            if (
                                bool(elevation_array.mask[local_row, local_col])
                                or bool(uncertainty_array.mask[local_row, local_col])
                                or (
                                    effective_array is not None
                                    and bool(effective_array.mask[local_row, local_col])
                                )
                            ):
                                nodata_pixels += 1
                                continue
                            elevation_value = float(elevation_array[local_row, local_col])
                            uncertainty_value = float(uncertainty_array[local_row, local_col])
                            if (
                                not math.isfinite(elevation_value)
                                or not math.isfinite(uncertainty_value)
                                or uncertainty_value < 0
                            ):
                                raise LError(
                                    f"{layer_id}: non-finite/negative raster value at {row},{col}"
                                )
                            effective_value = (
                                float(effective_array[local_row, local_col])
                                if effective_array is not None
                                else None
                            )
                            if effective_value is not None and (
                                not math.isfinite(effective_value) or effective_value <= 0
                            ):
                                raise LError(
                                    f"{layer_id}: invalid effective resolution at {row},{col}"
                                )
                            if max_effective_resolution is not None:
                                if effective_value is None:
                                    raise LError(
                                        f"{layer_id}: effective-resolution evidence required"
                                    )
                                if effective_value > float(max_effective_resolution):
                                    continue
                            selected_xy.append((x, y))
                            selected_records.append(
                                (
                                    row,
                                    col,
                                    x,
                                    y,
                                    elevation_value,
                                    uncertainty_value,
                                    effective_value,
                                    projected_range,
                                )
                            )
                    if not selected_records:
                        continue
                    longitudes, latitudes = warp_transform(
                        elevation.crs,
                        geographic_crs,
                        [item[0] for item in selected_xy],
                        [item[1] for item in selected_xy],
                    )
                    for record, longitude, latitude in zip(
                        selected_records, longitudes, latitudes
                    ):
                        (
                            row,
                            col,
                            x,
                            y,
                            elevation_value,
                            uncertainty_value,
                            effective_value,
                            projected_range,
                        ) = record
                        position = lonlat_to_xyz(
                            longitude, latitude, radius + elevation_value
                        )
                        high_terrain = add(position, scale(basis[2], uncertainty_value))
                        low_site = sub(
                            site_position, scale(basis[2], site_uncertainty)
                        )
                        azimuth, conservative_elevation = az_el(
                            sub(high_terrain, low_site), basis
                        )
                        bin_index = int(
                            math.floor((azimuth + 1e-9) / bin_width)
                        ) % bin_count
                        candidate = {
                            "row": row,
                            "col": col,
                            "x_m": x,
                            "y_m": y,
                            "longitude_deg": longitude,
                            "latitude_deg": latitude,
                            "elevation_m": elevation_value,
                            "vertical_uncertainty_m": uncertainty_value,
                            "effective_resolution_m": effective_value,
                            "range_xy_m": projected_range,
                            "position_m": list(position),
                            "selection_conservative_elevation_deg": conservative_elevation,
                            "selection_azimuth_deg": azimuth,
                        }
                        update_top(best[bin_index], candidate, top_k)
                        admitted_pixels += 1

                if (
                    layer.get("nodata_policy", "fail_required_coverage")
                    == "fail_required_coverage"
                    and nodata_pixels
                ):
                    raise LError(
                        f"{layer_id}: {nodata_pixels} nodata pixels inside required annulus"
                    )
                samples = []
                bins_with_candidates = 0
                for bin_index, values in enumerate(best):
                    if values:
                        bins_with_candidates += 1
                    for rank, candidate in enumerate(values):
                        output_candidate = dict(candidate)
                        output_candidate["azimuth_bin_index"] = bin_index
                        output_candidate["selection_rank"] = rank
                        samples.append(output_candidate)
                output_layers.append(
                    {
                        "layer_id": layer_id,
                        "min_range_m": float(minimum),
                        "max_range_m": float(maximum),
                        "nominal_resolution_m": float(layer["expected_pixel_size_m"]),
                        "additional_angular_margin_deg": float(
                            layer.get("additional_angular_margin_deg", 0.0)
                        ),
                        "elevation_source_sha256": source_hashes[
                            f"{layer_id}:elevation"
                        ],
                        "uncertainty_source_sha256": source_hashes[
                            f"{layer_id}:uncertainty"
                        ],
                        "effective_resolution_source_sha256": source_hashes.get(
                            f"{layer_id}:effective_resolution"
                        ),
                        "observed_raster_metadata": elevation_metadata,
                        "scanned_annulus_pixels": scanned_pixels,
                        "admitted_pixels_before_topk": admitted_pixels,
                        "nodata_pixels_in_annulus": nodata_pixels,
                        "bins_with_candidates": bins_with_candidates,
                        "samples": samples,
                    }
                )
            finally:
                if effective:
                    effective.close()

    if not output_layers or site_position is None:
        raise LError("no materialized layers")
    output = {
        "schema_version": OUT,
        "status": "pass",
        "study_id": data["study_id"],
        "frame_contract_id": data["frame_contract_id"],
        "epoch_contract_id": data["epoch_contract_id"],
        "site_ref": data["site_ref"],
        "native_frame": data["native_frame"],
        "projection_contract": data["projection_contract"],
        "config_sha256": sha256_file(config_path),
        "rasterio_version": rasterio.__version__,
        "source_hashes": source_hashes,
        "site_position_m": list(site_position),
        "site_vertical_uncertainty_m": site_uncertainty,
        "pole_vector": list(pole),
        "azimuth_bin_width_deg": bin_width,
        "top_k_per_bin": top_k,
        "layers": output_layers,
        "non_claims": [
            "This is an offline raster materialization artifact, not a site qualification.",
            "Rasterio/GDAL is evidence tooling only and is not a Symthaea runtime dependency.",
            "No resampling or reprojection of source raster cells is performed; only pixel-center coordinate transformation to an explicit lunar geographic sphere is used.",
        ],
    }
    output["receipt_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def k_input_from_pack(pack: dict, pack_path: str, pack_sha256: str):
    layers = []
    for layer in pack["layers"]:
        samples = []
        for sample in layer["samples"]:
            samples.append(
                {
                    "position_m": sample["position_m"],
                    "vertical_uncertainty_m": sample["vertical_uncertainty_m"],
                    "ll009l_row": sample["row"],
                    "ll009l_col": sample["col"],
                    "ll009l_bin_index": sample["azimuth_bin_index"],
                    "ll009l_rank": sample["selection_rank"],
                }
            )
        if not samples:
            continue
        layers.append(
            {
                "layer_id": layer["layer_id"],
                "source_ref": "ll009l-pack",
                "frame": pack["native_frame"],
                "min_range_m": layer["min_range_m"],
                "max_range_m": layer["max_range_m"],
                "nominal_resolution_m": layer["nominal_resolution_m"],
                "additional_angular_margin_deg": layer[
                    "additional_angular_margin_deg"
                ],
                "samples": samples,
            }
        )
    return {
        "schema_version": K_SCHEMA,
        "study_id": pack["study_id"],
        "frame_contract_id": pack["frame_contract_id"],
        "epoch_contract_id": pack["epoch_contract_id"],
        "site_ref": pack["site_ref"],
        "frame": pack["native_frame"],
        "sources": [
            {
                "source_id": "ll009l-pack",
                "path": pack_path,
                "sha256": pack_sha256,
            }
        ],
        "site_position_m": pack["site_position_m"],
        "site_vertical_uncertainty_m": pack["site_vertical_uncertainty_m"],
        "pole_vector": pack["pole_vector"],
        "azimuth_bin_width_deg": pack["azimuth_bin_width_deg"],
        "layers": layers,
    }


def write_immutable(path: pathlib.Path, value: dict):
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise LError(f"refusing to overwrite differing immutable output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test():
    rasterio, _warp_transform = require_rasterio()
    import numpy as np
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as temporary_directory:
        root = pathlib.Path(temporary_directory)
        crs = CRS.from_proj4(
            "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs"
        )
        transform = from_origin(-25, 25, 10, 10)
        elevation = np.zeros((5, 5), dtype="float32")
        elevation[2, 4] = 20.0
        uncertainty = np.full((5, 5), 0.5, dtype="float32")
        effective_resolution = np.full((5, 5), 10.0, dtype="float32")

        def write(name, array, raster_transform=transform, nodata=-9999.0):
            path = root / name
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=5,
                height=5,
                count=1,
                dtype="float32",
                crs=crs,
                transform=raster_transform,
                nodata=nodata,
                tiled=False,
            ) as dataset:
                dataset.write(array, 1)
            return path

        elevation_path = write("elevation.tif", elevation)
        uncertainty_path = write("uncertainty.tif", uncertainty)
        effective_path = write("effective.tif", effective_resolution)
        with rasterio.open(elevation_path) as dataset:
            crs_hash = sha256_bytes(dataset.crs.to_wkt().encode())
        config = {
            "schema_version": SCHEMA,
            "study_id": "study-x",
            "frame_contract_id": "frame-x",
            "epoch_contract_id": "epoch-x",
            "site_ref": "site-x",
            "native_frame": "MOON_ME_DE421",
            "projection_contract": {
                "projection": "south_polar_stereographic",
                "reference_radius_m": 1737400.0,
                "central_meridian_deg": 0.0,
                "true_scale_at_pole": True,
                "pixel_registration": "center",
            },
            "site": {
                "x_m": 0.0,
                "y_m": 0.0,
                "elevation_m": 0.0,
                "vertical_uncertainty_m": 0.25,
            },
            "pole_vector": [0, 0, 1],
            "azimuth_bin_width_deg": 90.0,
            "top_k_per_bin": 2,
            "layers": [
                {
                    "layer_id": "local",
                    "min_range_m": 5.0,
                    "max_range_m": 40.0,
                    "expected_pixel_size_m": 10.0,
                    "expected_crs_wkt_sha256": crs_hash,
                    "max_effective_resolution_m": 12.0,
                    "additional_angular_margin_deg": 0.0,
                    "nodata_policy": "fail_required_coverage",
                    "elevation_source": {
                        "path": "elevation.tif",
                        "sha256": sha256_file(elevation_path),
                    },
                    "uncertainty_source": {
                        "path": "uncertainty.tif",
                        "sha256": sha256_file(uncertainty_path),
                    },
                    "effective_resolution_source": {
                        "path": "effective.tif",
                        "sha256": sha256_file(effective_path),
                    },
                }
            ],
        }
        config_path = root / "config.json"
        config_path.write_bytes(canonical_bytes(config))
        output = materialize(config_path, root)
        assert output["schema_version"] == OUT and output["layers"][0]["samples"]
        ridge = [
            sample
            for sample in output["layers"][0]["samples"]
            if sample["row"] == 2 and sample["col"] == 4
        ]
        assert ridge, "raised ridge should survive top-k selection"
        assert canonical_bytes(output) == canonical_bytes(materialize(config_path, root))

        shifted_transform = from_origin(-24, 25, 10, 10)
        write("uncertainty-shifted.tif", uncertainty, raster_transform=shifted_transform)
        bad = json.loads(json.dumps(config))
        bad["layers"][0]["uncertainty_source"] = {
            "path": "uncertainty-shifted.tif",
            "sha256": sha256_file(root / "uncertainty-shifted.tif"),
        }
        bad_path = root / "bad.json"
        bad_path.write_bytes(canonical_bytes(bad))
        try:
            materialize(bad_path, root)
        except LError as exc:
            assert "transform mismatch" in str(exc)
        else:
            raise AssertionError("shifted uncertainty grid must fail")

        nodata_uncertainty = uncertainty.copy()
        nodata_uncertainty[0, 0] = -9999.0
        nodata_path = write("uncertainty-nodata.tif", nodata_uncertainty)
        bad2 = json.loads(json.dumps(config))
        bad2["layers"][0]["uncertainty_source"] = {
            "path": "uncertainty-nodata.tif",
            "sha256": sha256_file(nodata_path),
        }
        bad2_path = root / "bad2.json"
        bad2_path.write_bytes(canonical_bytes(bad2))
        try:
            materialize(bad2_path, root)
        except LError as exc:
            assert "nodata pixels" in str(exc)
        else:
            raise AssertionError("required nodata must fail")
        print("LL-009L self-test PASS", rasterio.__version__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--k-input-output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.config or not args.output:
        raise SystemExit("--config and --output required")
    config_path = pathlib.Path(args.config)
    artifact_root = pathlib.Path(args.artifact_root)
    output_path = pathlib.Path(args.output)
    pack = materialize(config_path, artifact_root)
    write_immutable(output_path, pack)
    if args.k_input_output:
        try:
            relative_pack_path = output_path.relative_to(artifact_root).as_posix()
        except ValueError as exc:
            raise LError(
                "output must live under artifact-root when emitting K input"
            ) from exc
        k_input = k_input_from_pack(
            pack, relative_pack_path, sha256_file(output_path)
        )
        write_immutable(pathlib.Path(args.k_input_output), k_input)
    print(json.dumps(pack, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
