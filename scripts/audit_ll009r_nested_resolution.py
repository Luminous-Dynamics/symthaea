#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
from typing import Any

CONFIG_SCHEMA = "ll009r.nested-resolution-audit-config.v1"
RECEIPT_SCHEMA = "ll009r.nested-resolution-audit-receipt.v1"


class RAuditError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_relpath(value: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RAuditError("source path must be non-empty string")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RAuditError(f"unsafe source path {value!r}")
    return path


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require_rasterio():
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_transform
    except Exception as exc:
        raise RAuditError("Rasterio/Numpy required only for offline LL-009R audit") from exc
    return np, rasterio, warp_transform


def safe_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RAuditError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RAuditError(f"{label} must contain object")
    return value


def validate_source(source: Any, label: str) -> dict[str, Any]:
    if not isinstance(source, dict):
        raise RAuditError(f"{label} source required")
    for key in ("path", "sha256"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise RAuditError(f"{label}.{key} required")
    digest = source["sha256"]
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RAuditError(f"{label}.sha256 must be lowercase SHA-256")
    pixel_size = source.get("expected_pixel_size_m")
    if not finite(pixel_size) or float(pixel_size) <= 0:
        raise RAuditError(f"{label}.expected_pixel_size_m invalid")
    safe_relpath(source["path"])
    return dict(source)


def validate_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != CONFIG_SCHEMA:
        raise RAuditError(f"schema_version must be {CONFIG_SCHEMA}")
    cfg = dict(value)
    for key in ("study_id", "coarse_layer_id", "fine_layer_id", "native_frame"):
        if not isinstance(cfg.get(key), str) or not cfg[key]:
            raise RAuditError(f"missing {key}")
    if cfg["coarse_layer_id"] == cfg["fine_layer_id"]:
        raise RAuditError("coarse_layer_id and fine_layer_id must differ")
    site = cfg.get("site")
    if not isinstance(site, dict) or not all(finite(site.get(k)) for k in ("x_m", "y_m", "elevation_m")):
        raise RAuditError("site x_m/y_m/elevation_m required")
    projection = cfg.get("projection_contract")
    if not isinstance(projection, dict) or projection.get("projection") != "south_polar_stereographic":
        raise RAuditError("south_polar_stereographic projection_contract required")
    radius = projection.get("reference_radius_m")
    if not finite(radius) or float(radius) <= 0:
        raise RAuditError("reference_radius_m invalid")
    pole = cfg.get("pole_vector")
    if not isinstance(pole, list) or len(pole) != 3 or not all(finite(v) for v in pole):
        raise RAuditError("pole_vector must be finite vec3")
    bin_width = cfg.get("azimuth_bin_width_deg")
    if not finite(bin_width) or float(bin_width) <= 0 or float(bin_width) > 90:
        raise RAuditError("azimuth_bin_width_deg invalid")
    bin_count = int(round(360.0 / float(bin_width)))
    if bin_count < 4 or abs(bin_count * float(bin_width) - 360.0) > 1e-9:
        raise RAuditError("azimuth_bin_width_deg must divide 360")
    minimum = cfg.get("min_range_m")
    maximum = cfg.get("max_range_m")
    if not finite(minimum) or not finite(maximum) or float(minimum) < 0 or float(maximum) <= float(minimum):
        raise RAuditError("min_range_m/max_range_m invalid")
    if not isinstance(cfg.get("require_all_bins"), bool):
        raise RAuditError("require_all_bins must be boolean")
    coarse = validate_source(cfg.get("coarse_source"), "coarse_source")
    fine = validate_source(cfg.get("fine_source"), "fine_source")
    if float(fine["expected_pixel_size_m"]) >= float(coarse["expected_pixel_size_m"]):
        raise RAuditError("fine source must have smaller nominal pixel size than coarse source")
    cfg["coarse_source"] = coarse
    cfg["fine_source"] = fine
    return cfg


def resolve_source(root: pathlib.Path, source: dict[str, Any], label: str) -> pathlib.Path:
    path = root / pathlib.Path(*safe_relpath(source["path"]).parts)
    if not path.is_file():
        raise RAuditError(f"{label}: missing source file")
    if sha256_file(path) != source["sha256"]:
        raise RAuditError(f"{label}: source hash mismatch")
    return path


def unit(np, vector, label: str):
    value = np.asarray(vector, dtype="float64")
    magnitude = float(np.linalg.norm(value))
    if not math.isfinite(magnitude) or magnitude <= 1e-15:
        raise RAuditError(f"{label} degenerate")
    return value / magnitude


def local_basis(np, site_up, pole):
    up = unit(np, site_up, "site up")
    pole_u = unit(np, pole, "pole")
    north_raw = pole_u - up * float(np.dot(pole_u, up))
    if float(np.linalg.norm(north_raw)) <= 1e-10:
        axes = np.eye(3, dtype="float64")
        reference = axes[int(np.argmin(np.abs(axes @ up)))]
        north_raw = reference - up * float(np.dot(reference, up))
    north = unit(np, north_raw, "north")
    east = unit(np, np.cross(north, up), "east")
    north = unit(np, np.cross(up, east), "north")
    return north, east, up


def lonlat_units(np, longitudes, latitudes):
    lon = np.radians(np.asarray(longitudes, dtype="float64"))
    lat = np.radians(np.asarray(latitudes, dtype="float64"))
    cos_lat = np.cos(lat)
    return np.stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)), axis=1)


def dataset_checks(dataset, expected_pixel_size: float, label: str) -> dict[str, Any]:
    if dataset.count != 1:
        raise RAuditError(f"{label}: exactly one band required")
    if dataset.crs is None:
        raise RAuditError(f"{label}: CRS required")
    a, b, _c, d, e, _f = list(dataset.transform)[:6]
    x_scale = math.hypot(a, d)
    y_scale = math.hypot(b, e)
    if abs(x_scale - expected_pixel_size) > 1e-8 or abs(y_scale - expected_pixel_size) > 1e-8:
        raise RAuditError(f"{label}: pixel size mismatch")
    wkt = dataset.crs.to_wkt()
    return {
        "width": dataset.width,
        "height": dataset.height,
        "dtype": dataset.dtypes[0],
        "nodata": dataset.nodata,
        "transform": list(dataset.transform)[:6],
        "crs_wkt_sha256": sha256_bytes(wkt.encode()),
        "pixel_scale_x_m": x_scale,
        "pixel_scale_y_m": y_scale,
    }


def horizon_for_dataset(dataset, cfg, np, rasterio, warp_transform):
    site_x = float(cfg["site"]["x_m"])
    site_y = float(cfg["site"]["y_m"])
    site_elevation = float(cfg["site"]["elevation_m"])
    radius = float(cfg["projection_contract"]["reference_radius_m"])
    bin_width = float(cfg["azimuth_bin_width_deg"])
    bin_count = int(round(360.0 / bin_width))
    minimum = float(cfg["min_range_m"])
    maximum = float(cfg["max_range_m"])
    geographic_crs = rasterio.crs.CRS.from_proj4(f"+proj=longlat +R={radius:.12f} +no_defs +type=crs")
    site_lon, site_lat = warp_transform(dataset.crs, geographic_crs, [site_x], [site_y])
    site_up = lonlat_units(np, site_lon, site_lat)[0]
    north, east, up = local_basis(np, site_up, cfg["pole_vector"])
    site_position = (radius + site_elevation) * site_up
    horizon = np.full(bin_count, -np.inf, dtype="float64")
    winning = [None for _ in range(bin_count)]
    scanned = admitted = nodata = 0
    bbox = (site_x - maximum, site_y - maximum, site_x + maximum, site_y + maximum)
    a, b, c, d, e, f = list(dataset.transform)[:6]

    for _, window in dataset.block_windows(1):
        left, bottom, right, top = rasterio.windows.bounds(window, dataset.transform)
        if right < bbox[0] or left > bbox[2] or top < bbox[1] or bottom > bbox[3]:
            continue
        array = dataset.read(1, window=window, masked=True)
        rows_local, cols_local = np.indices(array.shape)
        rows_global = rows_local.astype("float64") + float(window.row_off)
        cols_global = cols_local.astype("float64") + float(window.col_off)
        centers_col = cols_global + 0.5
        centers_row = rows_global + 0.5
        xs = a * centers_col + b * centers_row + c
        ys = d * centers_col + e * centers_row + f
        ranges = np.hypot(xs - site_x, ys - site_y)
        in_annulus = (ranges >= minimum - 1e-9) & (ranges <= maximum + 1e-9) & (ranges > 1e-9)
        scanned += int(np.count_nonzero(in_annulus))
        mask = np.ma.getmaskarray(array)
        nodata += int(np.count_nonzero(in_annulus & mask))
        if cfg.get("nodata_policy", "fail_required_coverage") == "fail_required_coverage" and np.any(in_annulus & mask):
            raise RAuditError("nodata inside required audit annulus")
        valid = in_annulus & ~mask
        if not np.any(valid):
            continue
        elevations = np.asarray(array.data[valid], dtype="float64")
        if not np.all(np.isfinite(elevations)):
            raise RAuditError("non-finite elevation in audit annulus")
        selected_x = xs[valid].astype("float64")
        selected_y = ys[valid].astype("float64")
        longitudes, latitudes = warp_transform(dataset.crs, geographic_crs, selected_x.tolist(), selected_y.tolist())
        units = lonlat_units(np, longitudes, latitudes)
        positions = units * (radius + elevations)[:, None]
        los = positions - site_position
        los_norm = np.linalg.norm(los, axis=1)
        if np.any(~np.isfinite(los_norm)) or np.any(los_norm <= 1e-12):
            raise RAuditError("degenerate audit line of sight")
        directions = los / los_norm[:, None]
        north_c = directions @ north
        east_c = directions @ east
        up_c = directions @ up
        azimuth = np.degrees(np.arctan2(east_c, north_c)) % 360.0
        elevation_deg = np.degrees(np.arctan2(up_c, np.hypot(north_c, east_c)))
        bins = np.floor((azimuth + 1e-9) / bin_width).astype("int64") % bin_count
        rows = rows_global[valid].astype("int64")
        cols = cols_global[valid].astype("int64")
        ranges_v = ranges[valid]
        for idx in range(len(elevation_deg)):
            bin_index = int(bins[idx])
            value = float(elevation_deg[idx])
            candidate = {
                "row": int(rows[idx]),
                "col": int(cols[idx]),
                "x_m": float(selected_x[idx]),
                "y_m": float(selected_y[idx]),
                "range_xy_m": float(ranges_v[idx]),
                "terrain_elevation_m": float(elevations[idx]),
                "horizon_elevation_deg": value,
            }
            current = horizon[bin_index]
            current_winner = winning[bin_index]
            if value > current + 1e-15 or (
                abs(value - current) <= 1e-15
                and (current_winner is None or (candidate["row"], candidate["col"]) < (current_winner["row"], current_winner["col"]))
            ):
                horizon[bin_index] = value
                winning[bin_index] = candidate
        admitted += len(elevation_deg)

    return horizon, winning, {
        "scanned_annulus_pixels": scanned,
        "admitted_pixels": admitted,
        "nodata_pixels_in_annulus": nodata,
    }


def audit(config_path: pathlib.Path, artifact_root: pathlib.Path) -> dict[str, Any]:
    np, rasterio, warp_transform = require_rasterio()
    cfg = validate_config(safe_json(config_path, "audit config"))
    coarse_path = resolve_source(artifact_root, cfg["coarse_source"], "coarse")
    fine_path = resolve_source(artifact_root, cfg["fine_source"], "fine")

    with rasterio.open(coarse_path) as coarse, rasterio.open(fine_path) as fine:
        coarse_meta = dataset_checks(coarse, float(cfg["coarse_source"]["expected_pixel_size_m"]), "coarse")
        fine_meta = dataset_checks(fine, float(cfg["fine_source"]["expected_pixel_size_m"]), "fine")
        if coarse.crs != fine.crs:
            raise RAuditError("coarse/fine CRS mismatch")
        coarse_horizon, coarse_winning, coarse_stats = horizon_for_dataset(coarse, cfg, np, rasterio, warp_transform)
        fine_horizon, fine_winning, fine_stats = horizon_for_dataset(fine, cfg, np, rasterio, warp_transform)

    common = np.isfinite(coarse_horizon) & np.isfinite(fine_horizon)
    if cfg["require_all_bins"] and not bool(np.all(common)):
        raise RAuditError(f"coarse/fine audit lacks common support for bins {np.nonzero(~common)[0].tolist()}")
    if not bool(np.any(common)):
        raise RAuditError("coarse/fine audit has no common azimuth bins")

    per_bin = []
    max_positive = 0.0
    worst_bin = None
    for bin_index in range(len(coarse_horizon)):
        if not common[bin_index]:
            per_bin.append({
                "azimuth_bin_index": bin_index,
                "coarse_horizon_deg": None,
                "fine_horizon_deg": None,
                "fine_minus_coarse_deg": None,
                "positive_excursion_deg": None,
                "coarse_winner": coarse_winning[bin_index],
                "fine_winner": fine_winning[bin_index],
            })
            continue
        delta = float(fine_horizon[bin_index] - coarse_horizon[bin_index])
        positive = max(0.0, delta)
        if positive > max_positive:
            max_positive = positive
            worst_bin = bin_index
        per_bin.append({
            "azimuth_bin_index": bin_index,
            "coarse_horizon_deg": float(coarse_horizon[bin_index]),
            "fine_horizon_deg": float(fine_horizon[bin_index]),
            "fine_minus_coarse_deg": delta,
            "positive_excursion_deg": positive,
            "coarse_winner": coarse_winning[bin_index],
            "fine_winner": fine_winning[bin_index],
        })

    output = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "pass",
        "study_id": cfg["study_id"],
        "coarse_layer_id": cfg["coarse_layer_id"],
        "fine_layer_id": cfg["fine_layer_id"],
        "native_frame": cfg["native_frame"],
        "support_semantics": "empirical_multiscale_bound",
        "config_sha256": sha256_file(config_path),
        "coarse_source_sha256": cfg["coarse_source"]["sha256"],
        "fine_source_sha256": cfg["fine_source"]["sha256"],
        "coarse_nominal_resolution_m": float(cfg["coarse_source"]["expected_pixel_size_m"]),
        "fine_nominal_resolution_m": float(cfg["fine_source"]["expected_pixel_size_m"]),
        "coarse_metadata": coarse_meta,
        "fine_metadata": fine_meta,
        "site": cfg["site"],
        "min_range_m": float(cfg["min_range_m"]),
        "max_range_m": float(cfg["max_range_m"]),
        "azimuth_bin_width_deg": float(cfg["azimuth_bin_width_deg"]),
        "common_bin_count": int(np.count_nonzero(common)),
        "azimuth_bin_count": len(coarse_horizon),
        "max_positive_fine_minus_coarse_deg": max_positive,
        "worst_positive_excursion_bin": worst_bin,
        "coarse_stats": coarse_stats,
        "fine_stats": fine_stats,
        "per_bin": per_bin,
        "interpretation": (
            "Within the exact configured overlap, max_positive_fine_minus_coarse_deg is the largest observed positive skyline excursion present in the finer raster relative to the coarser raster at pixel-center support. It is an empirical multiscale discrepancy, not a theorem over unresolved continuous terrain."
        ),
        "non_claims": [
            "The audit does not prove that the fine raster itself resolves every physical terrain feature.",
            "A zero observed coarse-vs-fine excursion is not a deterministic continuous-terrain bound.",
            "This audit does not change vertical-error/RMS/ensemble uncertainty semantics.",
            "Effective resolution or nominal pixel size is not treated as an interpolation theorem.",
        ],
    }
    output["receipt_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def write_immutable(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RAuditError(f"refusing to overwrite differing output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> None:
    np, rasterio, _warp_transform = require_rasterio()
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        crs = CRS.from_proj4("+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs")
        coarse_transform = from_origin(-25, 25, 10, 10)
        fine_transform = from_origin(-22.5, 22.5, 5, 5)
        coarse_array = np.zeros((5, 5), dtype="float32")
        fine_array = np.zeros((9, 9), dtype="float32")
        fine_array[4, 7] = 12.0

        def write(path: pathlib.Path, array, transform):
            with rasterio.open(
                path, "w", driver="GTiff", width=array.shape[1], height=array.shape[0], count=1,
                dtype="float32", crs=crs, transform=transform, nodata=-9999.0
            ) as ds:
                ds.write(array, 1)

        coarse_path = root / "coarse.tif"
        fine_path = root / "fine.tif"
        write(coarse_path, coarse_array, coarse_transform)
        write(fine_path, fine_array, fine_transform)
        cfg = {
            "schema_version": CONFIG_SCHEMA,
            "study_id": "study-r-audit",
            "coarse_layer_id": "coarse",
            "fine_layer_id": "fine",
            "native_frame": "MOON_ME_DE421",
            "projection_contract": {"projection": "south_polar_stereographic", "reference_radius_m": 1737400.0},
            "site": {"x_m": 0.0, "y_m": 0.0, "elevation_m": 0.0},
            "pole_vector": [0.0, 0.0, 1.0],
            "azimuth_bin_width_deg": 90.0,
            "min_range_m": 4.0,
            "max_range_m": 24.0,
            "require_all_bins": True,
            "nodata_policy": "fail_required_coverage",
            "coarse_source": {"path": "coarse.tif", "sha256": sha256_file(coarse_path), "expected_pixel_size_m": 10.0},
            "fine_source": {"path": "fine.tif", "sha256": sha256_file(fine_path), "expected_pixel_size_m": 5.0},
        }
        cfg_path = root / "config.json"
        cfg_path.write_bytes(canonical_bytes(cfg))
        first = audit(cfg_path, root)
        second = audit(cfg_path, root)
        assert canonical_bytes(first) == canonical_bytes(second)
        assert first["support_semantics"] == "empirical_multiscale_bound"
        assert first["max_positive_fine_minus_coarse_deg"] > 1.0
        assert first["worst_positive_excursion_bin"] is not None
        assert any(
            item["positive_excursion_deg"] is not None and item["positive_excursion_deg"] > 1.0
            for item in first["per_bin"]
        ), "fine-only narrow peak must be detected"

        broken = json.loads(json.dumps(cfg))
        broken["fine_source"]["sha256"] = "0" * 64
        broken_path = root / "broken.json"
        broken_path.write_bytes(canonical_bytes(broken))
        try:
            audit(broken_path, root)
        except RAuditError as exc:
            assert "hash mismatch" in str(exc)
        else:
            raise RAuditError("self-test expected hash mismatch rejection")

        fine_bad = fine_array.copy()
        fine_bad[4, 7] = -9999.0
        write(fine_path, fine_bad, fine_transform)
        cfg["fine_source"]["sha256"] = sha256_file(fine_path)
        cfg_path.write_bytes(canonical_bytes(cfg))
        try:
            audit(cfg_path, root)
        except RAuditError as exc:
            assert "nodata" in str(exc)
        else:
            raise RAuditError("self-test expected nodata rejection")

        print("LL-009R nested-resolution audit self-test: PASS", rasterio.__version__)


def main() -> int:
    parser = argparse.ArgumentParser(description="LL-009R exact nested-resolution skyline audit")
    parser.add_argument("--config")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not args.config or not args.output:
            raise RAuditError("--config and --output are required")
        receipt = audit(pathlib.Path(args.config), pathlib.Path(args.artifact_root))
        write_immutable(pathlib.Path(args.output), receipt)
        print(json.dumps(receipt, sort_keys=True, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, RAuditError) as exc:
        raise SystemExit(f"LL-009R audit failure: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
