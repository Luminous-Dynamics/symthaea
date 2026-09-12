#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
from typing import Any

import acquire_ll009n_nasa_rasters as N

CONFIG_SCHEMA = "ll009p.site-anchor-config.v1"
RECEIPT_SCHEMA = "ll009p.site-anchor-receipt.v1"


class PError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return N.canonical_bytes(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_json(path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PError(f"{label} must contain object")
    return value


def require_rasterio():
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_transform
    except Exception as exc:
        raise PError(
            "Rasterio/Numpy required only for offline LL-009P materialization"
        ) from exc
    return rasterio, warp_transform, np


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def same_affine(left, right, tolerance=1e-12) -> bool:
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def validate_config(config: dict) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PError(f"schema must be {CONFIG_SCHEMA}")
    for key in ("study_id", "anchor_id", "site_ref", "native_frame"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise PError(f"missing {key}")

    reference = config.get("published_reference")
    if (
        not isinstance(reference, dict)
        or not finite(reference.get("latitude_deg"))
        or not finite(reference.get("longitude_deg"))
        or not isinstance(reference.get("source_reference"), str)
    ):
        raise PError(
            "published_reference latitude/longitude/source_reference required"
        )
    if not -90 <= float(reference["latitude_deg"]) <= 90:
        raise PError("latitude invalid")

    projection = config.get("projection_contract")
    if (
        not isinstance(projection, dict)
        or projection.get("projection") != "south_polar_stereographic"
    ):
        raise PError("south_polar_stereographic projection_contract required")
    for key in (
        "reference_radius_m",
        "latitude_of_origin_deg",
        "latitude_true_scale_deg",
        "central_meridian_deg",
    ):
        if not finite(projection.get(key)):
            raise PError(f"projection {key} required")
    if float(projection["reference_radius_m"]) <= 0:
        raise PError("reference radius invalid")

    sources = config.get("sources")
    if not isinstance(sources, dict):
        raise PError("sources required")
    for key in ("elevation_source_id", "uncertainty_source_id"):
        if not isinstance(sources.get(key), str) or not sources[key]:
            raise PError(f"sources.{key} required")

    pixel_size = config.get("expected_pixel_size_m")
    max_offset = config.get("max_reference_to_pixel_center_m")
    if not finite(pixel_size) or float(pixel_size) <= 0:
        raise PError("expected_pixel_size_m invalid")
    if not finite(max_offset) or float(max_offset) < 0:
        raise PError("max_reference_to_pixel_center_m invalid")
    if config.get("sampling_policy") != "containing_pixel_center_no_interpolation":
        raise PError("V1 requires containing_pixel_center_no_interpolation")


def source_by_id(lock: dict, source_id: str) -> dict:
    if lock.get("schema_version") != N.LOCK_SCHEMA:
        raise PError(f"source lock schema must be {N.LOCK_SCHEMA}")
    files = lock.get("files")
    if not isinstance(files, list):
        raise PError("source lock files missing")
    matches = [
        item
        for item in files
        if isinstance(item, dict) and item.get("source_id") == source_id
    ]
    if len(matches) != 1:
        raise PError(f"expected exactly one locked source {source_id}")
    return matches[0]


def resolve_locked(
    root: pathlib.Path, entry: dict, label: str
) -> pathlib.Path:
    relative = entry.get("artifact_path")
    expected_hash = entry.get("sha256")
    if (
        not isinstance(relative, str)
        or not isinstance(expected_hash, str)
        or len(expected_hash) != 64
    ):
        raise PError(f"{label}: artifact_path/sha256 required")
    path = root / pathlib.Path(*N.safe_relpath(relative).parts)
    if not path.is_file():
        raise PError(f"{label}: locked artifact missing")
    actual_hash, actual_size = N.sha256_file(path)
    if actual_hash != expected_hash:
        raise PError(f"{label}: locked source hash mismatch")
    if isinstance(entry.get("byte_size"), int) and actual_size != entry["byte_size"]:
        raise PError(f"{label}: locked source size mismatch")
    return path


def aligned(left, right) -> None:
    if left.width != right.width or left.height != right.height:
        raise PError("elevation/uncertainty shape mismatch")
    if left.crs != right.crs:
        raise PError("elevation/uncertainty CRS mismatch")
    if not same_affine(left.transform, right.transform):
        raise PError("elevation/uncertainty transform mismatch")


def dataset_checks(dataset, expected_pixel_size: float, label: str) -> dict:
    if dataset.count != 1:
        raise PError(f"{label}: exactly one band required")
    if dataset.crs is None:
        raise PError(f"{label}: CRS required")
    a, b, _c, d, e, _f = list(dataset.transform)[:6]
    x_scale = math.hypot(a, d)
    y_scale = math.hypot(b, e)
    if (
        abs(x_scale - expected_pixel_size) > 1e-8
        or abs(y_scale - expected_pixel_size) > 1e-8
    ):
        raise PError(f"{label}: pixel size mismatch")
    wkt = dataset.crs.to_wkt()
    return {
        "width": dataset.width,
        "height": dataset.height,
        "dtype": dataset.dtypes[0],
        "nodata": dataset.nodata,
        "transform": list(dataset.transform)[:6],
        "crs_wkt": wkt,
        "crs_wkt_sha256": sha256_bytes(wkt.encode()),
        "pixel_scale_x_m": x_scale,
        "pixel_scale_y_m": y_scale,
    }


def materialize(
    config_path: pathlib.Path,
    source_lock_path: pathlib.Path,
    artifact_root: pathlib.Path,
) -> dict:
    rasterio, warp_transform, np = require_rasterio()
    config = safe_json(config_path, "config")
    source_lock = safe_json(source_lock_path, "source lock")
    validate_config(config)
    if source_lock.get("study_id") != config["study_id"]:
        raise PError("config/source-lock study mismatch")

    elevation_entry = source_by_id(
        source_lock, config["sources"]["elevation_source_id"]
    )
    uncertainty_entry = source_by_id(
        source_lock, config["sources"]["uncertainty_source_id"]
    )
    if elevation_entry.get("role") != "surface_elevation":
        raise PError("elevation source role mismatch")
    if uncertainty_entry.get("role") not in {
        "elevation_rms_uncertainty_m",
        "surface_height_error_m",
        "vertical_uncertainty_m",
    }:
        raise PError("uncertainty source role mismatch")

    elevation_path = resolve_locked(
        artifact_root, elevation_entry, "elevation"
    )
    uncertainty_path = resolve_locked(
        artifact_root, uncertainty_entry, "uncertainty"
    )
    expected_pixel_size = float(config["expected_pixel_size_m"])

    with rasterio.open(elevation_path) as elevation, rasterio.open(
        uncertainty_path
    ) as uncertainty:
        aligned(elevation, uncertainty)
        elevation_metadata = dataset_checks(
            elevation, expected_pixel_size, "elevation"
        )
        dataset_checks(uncertainty, expected_pixel_size, "uncertainty")

        projection = config["projection_contract"]
        radius = float(projection["reference_radius_m"])
        geographic_crs = rasterio.crs.CRS.from_proj4(
            f"+proj=longlat +R={radius:.12f} +no_defs +type=crs"
        )
        reference = config["published_reference"]
        longitude = float(reference["longitude_deg"])
        latitude = float(reference["latitude_deg"])
        xs, ys = warp_transform(
            geographic_crs,
            elevation.crs,
            [longitude],
            [latitude],
        )
        reference_x = float(xs[0])
        reference_y = float(ys[0])
        try:
            row, col = elevation.index(reference_x, reference_y)
        except Exception as exc:
            raise PError(f"cannot index published anchor: {exc}") from exc
        if (
            row < 0
            or col < 0
            or row >= elevation.height
            or col >= elevation.width
        ):
            raise PError("published anchor falls outside raster")

        window = rasterio.windows.Window(col, row, 1, 1)
        elevation_array = elevation.read(1, window=window, masked=True)
        uncertainty_array = uncertainty.read(1, window=window, masked=True)
        elevation_mask = np.ma.getmaskarray(elevation_array)
        uncertainty_mask = np.ma.getmaskarray(uncertainty_array)
        if bool(elevation_mask[0, 0]) or bool(uncertainty_mask[0, 0]):
            raise PError("published anchor cell is nodata")

        elevation_value = float(elevation_array[0, 0])
        uncertainty_value = float(uncertainty_array[0, 0])
        if (
            not math.isfinite(elevation_value)
            or not math.isfinite(uncertainty_value)
            or uncertainty_value < 0
        ):
            raise PError("anchor elevation/uncertainty invalid")

        center_x, center_y = elevation.transform * (col + 0.5, row + 0.5)
        center_x = float(center_x)
        center_y = float(center_y)
        center_offset = math.hypot(
            center_x - reference_x, center_y - reference_y
        )
        if (
            center_offset
            > float(config["max_reference_to_pixel_center_m"]) + 1e-9
        ):
            raise PError("reference-to-pixel-center offset exceeds contract")
        center_longitudes, center_latitudes = warp_transform(
            elevation.crs,
            geographic_crs,
            [center_x],
            [center_y],
        )

    output = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "pass",
        "study_id": config["study_id"],
        "anchor_id": config["anchor_id"],
        "site_ref": config["site_ref"],
        "native_frame": config["native_frame"],
        "config_sha256": N.sha256_file(config_path)[0],
        "source_lock_sha256": N.sha256_file(source_lock_path)[0],
        "rasterio_version": rasterio.__version__,
        "sampling_policy": config["sampling_policy"],
        "published_reference": {
            **reference,
            "projected_x_m": reference_x,
            "projected_y_m": reference_y,
        },
        "selected_pixel": {
            "row": int(row),
            "col": int(col),
            "center_x_m": center_x,
            "center_y_m": center_y,
            "center_longitude_deg": float(center_longitudes[0]),
            "center_latitude_deg": float(center_latitudes[0]),
            "reference_to_center_offset_m": center_offset,
            "elevation_m": elevation_value,
            "vertical_uncertainty_m": uncertainty_value,
        },
        "ll009l_site_block": {
            "x_m": center_x,
            "y_m": center_y,
            "elevation_m": elevation_value,
            "vertical_uncertainty_m": uncertainty_value,
        },
        "elevation_source_id": elevation_entry["source_id"],
        "elevation_source_sha256": elevation_entry["sha256"],
        "uncertainty_source_id": uncertainty_entry["source_id"],
        "uncertainty_source_sha256": uncertainty_entry["sha256"],
        "observed_raster_metadata": elevation_metadata,
        "ll009l_expected_crs_wkt_sha256": elevation_metadata[
            "crs_wkt_sha256"
        ],
        "uncertainty_semantics": "preserve_source_semantics_ll009o_required",
        "non_claims": [
            "The published reference coordinate is a reproducible Site01 anchor, not an optimized infrastructure location.",
            "The selected pixel uncertainty is not relabeled as a hard bound; LL-009O remains authoritative for semantics.",
            "No interpolation or resampling is performed by LL-009P.",
        ],
    }
    output["receipt_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def write_immutable(path: pathlib.Path, value: dict) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise PError(f"refusing to overwrite differing immutable output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> None:
    rasterio, warp_transform, np = require_rasterio()
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        data_root = root / "data"
        data_root.mkdir()
        crs = CRS.from_proj4(
            "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 "
            "+R=1737400 +units=m +no_defs"
        )
        geographic_crs = CRS.from_proj4(
            "+proj=longlat +R=1737400 +no_defs +type=crs"
        )
        transform = from_origin(-25, 25, 5, 5)
        elevation = np.zeros((10, 10), dtype="float32")
        uncertainty = np.full((10, 10), 0.5, dtype="float32")
        target_x, target_y = -9.0, -11.0
        row, col = rasterio.transform.rowcol(transform, target_x, target_y)
        elevation[row, col] = 123.25
        uncertainty[row, col] = 0.75

        def write(name, array, raster_transform=transform):
            path = data_root / name
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=1,
                dtype="float32",
                crs=crs,
                transform=raster_transform,
                nodata=-9999.0,
            ) as dataset:
                dataset.write(array, 1)
            return path

        elevation_path = write("e.tif", elevation)
        uncertainty_path = write("u.tif", uncertainty)
        elevation_hash, elevation_size = N.sha256_file(elevation_path)
        uncertainty_hash, uncertainty_size = N.sha256_file(uncertainty_path)
        longitudes, latitudes = warp_transform(
            crs, geographic_crs, [target_x], [target_y]
        )
        source_lock = {
            "schema_version": N.LOCK_SCHEMA,
            "study_id": "study-p",
            "files": [
                {
                    "source_id": "e",
                    "role": "surface_elevation",
                    "artifact_path": "e.tif",
                    "sha256": elevation_hash,
                    "byte_size": elevation_size,
                },
                {
                    "source_id": "u",
                    "role": "elevation_rms_uncertainty_m",
                    "artifact_path": "u.tif",
                    "sha256": uncertainty_hash,
                    "byte_size": uncertainty_size,
                },
            ],
        }
        lock_path = root / "source-lock.json"
        lock_path.write_bytes(canonical_bytes(source_lock))
        config = {
            "schema_version": CONFIG_SCHEMA,
            "study_id": "study-p",
            "anchor_id": "anchor-p",
            "site_ref": "site-p",
            "native_frame": "MOON_ME_DE421",
            "published_reference": {
                "latitude_deg": float(latitudes[0]),
                "longitude_deg": float(longitudes[0]),
                "source_reference": "synthetic",
            },
            "projection_contract": {
                "projection": "south_polar_stereographic",
                "reference_radius_m": 1737400.0,
                "latitude_of_origin_deg": -90.0,
                "latitude_true_scale_deg": -90.0,
                "central_meridian_deg": 0.0,
            },
            "sources": {
                "elevation_source_id": "e",
                "uncertainty_source_id": "u",
            },
            "expected_pixel_size_m": 5.0,
            "max_reference_to_pixel_center_m": 4.0,
            "sampling_policy": "containing_pixel_center_no_interpolation",
        }
        config_path = root / "config.json"
        config_path.write_bytes(canonical_bytes(config))

        first = materialize(config_path, lock_path, data_root)
        second = materialize(config_path, lock_path, data_root)
        assert canonical_bytes(first) == canonical_bytes(second)
        assert abs(first["selected_pixel"]["elevation_m"] - 123.25) < 1e-6
        assert (
            abs(first["selected_pixel"]["vertical_uncertainty_m"] - 0.75)
            < 1e-6
        )
        assert first["selected_pixel"]["reference_to_center_offset_m"] <= 4.0

        shifted_transform = from_origin(-24, 25, 5, 5)
        shifted_path = write("u2.tif", uncertainty, shifted_transform)
        shifted_hash, shifted_size = N.sha256_file(shifted_path)
        source_lock["files"][1].update(
            {
                "artifact_path": "u2.tif",
                "sha256": shifted_hash,
                "byte_size": shifted_size,
            }
        )
        lock_path.write_bytes(canonical_bytes(source_lock))
        try:
            materialize(config_path, lock_path, data_root)
        except PError as exc:
            assert "transform mismatch" in str(exc)
        else:
            raise PError("self-test expected shifted-grid rejection")
        print("LL-009P self-test: PASS", rasterio.__version__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LL-009P exact Site01 reference-anchor materialization"
    )
    parser.add_argument("--config")
    parser.add_argument("--source-lock")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not all((args.config, args.source_lock, args.output)):
            parser.error("--config --source-lock --output required")
        output = materialize(
            pathlib.Path(args.config),
            pathlib.Path(args.source_lock),
            pathlib.Path(args.artifact_root),
        )
        write_immutable(pathlib.Path(args.output), output)
        print(json.dumps(output, sort_keys=True, indent=2))
        return 0
    except (PError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
