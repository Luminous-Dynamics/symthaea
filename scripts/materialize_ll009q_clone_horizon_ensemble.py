#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import pathlib
import tempfile
from typing import Any

import generate_ll009q_clone_plan as QP

POLICY_SCHEMA = "ll009q.ensemble-policy.v1"
LOCK_SCHEMA = "ll009n.nasa-source-lock.v1"
P_RECEIPT_SCHEMA = "ll009p.site-anchor-receipt.v1"
L_CONFIG_SCHEMA = "ll009l.cog-materialization-config.v1"
RECEIPT_SCHEMA = "ll009q.clone-horizon-ensemble-receipt.v1"


class QError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
    return digest.hexdigest(), count


def safe_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise QError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise QError(f"{label} must contain object")
    return value


def safe_relpath(value: str) -> pathlib.PurePosixPath:
    try:
        return QP.safe_relpath(value)
    except Exception as exc:
        raise QError(str(exc)) from exc


def require_rasterio():
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_transform
    except Exception as exc:
        raise QError(
            "Rasterio/Numpy required only for offline LL-009Q materialization"
        ) from exc
    return np, rasterio, warp_transform


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def same_affine(left, right, tolerance: float = 1e-12) -> bool:
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def validate_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != POLICY_SCHEMA:
        raise QError(f"policy schema must be {POLICY_SCHEMA}")
    policy = dict(value)
    for key in (
        "study_id",
        "ensemble_layer_id",
        "nominal_elevation_source_id",
        "clone_interpretation",
        "observer_policy",
        "quantile_estimator",
        "spatial_support_class",
    ):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise QError(f"policy missing {key}")
    if policy["clone_interpretation"] not in {
        "additive_z_error_to_nominal_ldem",
        "full_clone_surface",
    }:
        raise QError("policy clone_interpretation must be explicit")
    if policy["observer_policy"] != "same_realization_site_pixel":
        raise QError("V1 requires observer_policy=same_realization_site_pixel")
    if policy["quantile_estimator"] != "empirical_cdf_nearest_rank":
        raise QError("V1 requires empirical_cdf_nearest_rank")
    if policy["spatial_support_class"] not in {
        "sample_points_only",
        "resolution_qualified",
        "empirical_multiscale_bound",
        "continuous_hard_bound",
    }:
        raise QError("unsupported spatial_support_class")
    count = policy.get("required_member_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        raise QError("required_member_count must be integer >=2")
    quantiles = policy.get("quantiles")
    if (
        not isinstance(quantiles, list)
        or not quantiles
        or not all(finite(q) and 0 < float(q) <= 1 for q in quantiles)
    ):
        raise QError("quantiles must be finite values in (0,1]")
    q_values = [float(q) for q in quantiles]
    if q_values != sorted(set(q_values)):
        raise QError("quantiles must be unique and ascending")
    require_all = policy.get("require_all_azimuth_bins")
    if not isinstance(require_all, bool):
        raise QError("require_all_azimuth_bins must be boolean")
    return policy


def lock_files(lock: dict[str, Any], label: str) -> list[dict[str, Any]]:
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise QError(f"{label} schema must be {LOCK_SCHEMA}")
    files = lock.get("files")
    if not isinstance(files, list):
        raise QError(f"{label} files missing")
    if not all(isinstance(item, dict) for item in files):
        raise QError(f"{label} files must be objects")
    return files


def source_by_id(lock: dict[str, Any], source_id: str, label: str) -> dict[str, Any]:
    matches = [
        item for item in lock_files(lock, label) if item.get("source_id") == source_id
    ]
    if len(matches) != 1:
        raise QError(f"{label}: expected exactly one source {source_id}")
    return matches[0]


def resolve_locked(root: pathlib.Path, entry: dict[str, Any], label: str) -> pathlib.Path:
    rel = entry.get("artifact_path")
    digest = entry.get("sha256")
    if (
        not isinstance(rel, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(ch not in "0123456789abcdef" for ch in digest)
    ):
        raise QError(f"{label}: artifact_path and lowercase sha256 required")
    path = root / pathlib.Path(*safe_relpath(rel).parts)
    if not path.is_file():
        raise QError(f"{label}: locked artifact missing: {rel}")
    actual, size = sha256_file(path)
    if actual != digest:
        raise QError(f"{label}: source hash mismatch")
    expected_size = entry.get("byte_size")
    if (
        isinstance(expected_size, int)
        and not isinstance(expected_size, bool)
        and size != expected_size
    ):
        raise QError(f"{label}: source byte-size mismatch")
    return path


def validate_clone_plan_and_lock(
    family: dict[str, Any],
    clone_plan: dict[str, Any],
    clone_plan_path: pathlib.Path,
    clone_lock: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_plan = QP.expand_family(family)
    if canonical_bytes(clone_plan) != canonical_bytes(expected_plan):
        raise QError("clone acquisition plan does not exactly match family expansion")
    plan_hash = sha256_file(clone_plan_path)[0]
    if clone_lock.get("plan_sha256") != plan_hash:
        raise QError("clone source lock does not bind exact clone acquisition plan")
    if clone_lock.get("study_id") != expected_plan["study_id"]:
        raise QError("clone lock study mismatch")
    expected_files = expected_plan["files"]
    locked_files = lock_files(clone_lock, "clone lock")
    expected_ids = [item["source_id"] for item in expected_files]
    locked_ids = [item.get("source_id") for item in locked_files]
    if locked_ids != expected_ids:
        raise QError("clone lock member order/content does not match exact family expansion")
    for planned, locked in zip(expected_files, locked_files):
        for key in ("source_id", "dataset_id", "role", "artifact_path"):
            if locked.get(key) != planned.get(key):
                raise QError(f"clone lock drift for {planned['source_id']}: {key}")
        if locked.get("role") != "terrain_error_realization_m":
            raise QError("V1 expects terrain_error_realization_m clone role")
        digest = locked.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise QError(f"{planned['source_id']}: locked SHA-256 required")
    return locked_files


def validate_site_receipt(
    receipt: dict[str, Any],
    nominal_lock_path: pathlib.Path,
    nominal_entry: dict[str, Any],
    policy: dict[str, Any],
) -> None:
    if receipt.get("schema_version") != P_RECEIPT_SCHEMA:
        raise QError(f"site receipt schema must be {P_RECEIPT_SCHEMA}")
    if receipt.get("status") != "pass":
        raise QError("site receipt must pass")
    if receipt.get("study_id") != policy["study_id"]:
        raise QError("site receipt study mismatch")
    if receipt.get("source_lock_sha256") != sha256_file(nominal_lock_path)[0]:
        raise QError("site receipt does not bind exact nominal source lock")
    if receipt.get("elevation_source_id") != nominal_entry.get("source_id"):
        raise QError("site receipt nominal elevation source mismatch")
    if receipt.get("elevation_source_sha256") != nominal_entry.get("sha256"):
        raise QError("site receipt nominal elevation hash mismatch")
    selected = receipt.get("selected_pixel")
    site_block = receipt.get("ll009l_site_block")
    if not isinstance(selected, dict) or not isinstance(site_block, dict):
        raise QError("site receipt selected_pixel/ll009l_site_block required")
    for key in ("row", "col"):
        if not isinstance(selected.get(key), int) or isinstance(selected[key], bool):
            raise QError(f"site receipt selected_pixel.{key} required")
    for key in ("center_x_m", "center_y_m", "elevation_m"):
        if not finite(selected.get(key)):
            raise QError(f"site receipt selected_pixel.{key} invalid")
    for key in ("x_m", "y_m", "elevation_m", "vertical_uncertainty_m"):
        if not finite(site_block.get(key)):
            raise QError(f"site receipt ll009l_site_block.{key} invalid")


def validate_l_config(
    config: dict[str, Any],
    site_receipt: dict[str, Any],
    nominal_entry: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    if config.get("schema_version") != L_CONFIG_SCHEMA:
        raise QError(f"L config schema must be {L_CONFIG_SCHEMA}")
    if config.get("study_id") != policy["study_id"]:
        raise QError("L config study mismatch")
    if config.get("site_ref") != site_receipt.get("site_ref"):
        raise QError("L config/site receipt site_ref mismatch")
    if config.get("native_frame") != site_receipt.get("native_frame"):
        raise QError("L config/site receipt frame mismatch")

    expected_site = site_receipt["ll009l_site_block"]
    observed_site = config.get("site")
    if not isinstance(observed_site, dict):
        raise QError("L config site block missing")
    for key in ("x_m", "y_m", "elevation_m", "vertical_uncertainty_m"):
        if not finite(observed_site.get(key)):
            raise QError(f"L config site.{key} invalid")
        if abs(float(observed_site[key]) - float(expected_site[key])) > 1e-9:
            raise QError(f"L config site.{key} does not match LL-009P receipt")

    bin_width = config.get("azimuth_bin_width_deg")
    if not finite(bin_width) or float(bin_width) <= 0 or float(bin_width) > 90:
        raise QError("L config azimuth_bin_width_deg invalid")
    bin_count = int(round(360.0 / float(bin_width)))
    if bin_count < 4 or abs(bin_count * float(bin_width) - 360.0) > 1e-9:
        raise QError("azimuth_bin_width_deg must divide 360")

    projection = config.get("projection_contract")
    if not isinstance(projection, dict):
        raise QError("L config projection_contract missing")
    radius = projection.get("reference_radius_m")
    if not finite(radius) or float(radius) <= 0:
        raise QError("L config reference_radius_m invalid")
    if projection.get("projection") != "south_polar_stereographic":
        raise QError("V1 requires south_polar_stereographic")

    pole = config.get("pole_vector")
    if (
        not isinstance(pole, list)
        or len(pole) != 3
        or not all(finite(value) for value in pole)
    ):
        raise QError("L config pole_vector invalid")

    layers = config.get("layers")
    if not isinstance(layers, list):
        raise QError("L config layers missing")
    matches = [
        layer
        for layer in layers
        if isinstance(layer, dict)
        and layer.get("layer_id") == policy["ensemble_layer_id"]
    ]
    if len(matches) != 1:
        raise QError("expected exactly one L ensemble layer")
    layer = matches[0]
    minimum = layer.get("min_range_m")
    maximum = layer.get("max_range_m")
    pixel_size = layer.get("expected_pixel_size_m")
    if (
        not finite(minimum)
        or not finite(maximum)
        or float(minimum) < 0
        or float(maximum) <= float(minimum)
    ):
        raise QError("ensemble L layer range invalid")
    if not finite(pixel_size) or float(pixel_size) <= 0:
        raise QError("ensemble L layer expected_pixel_size_m invalid")
    elevation_source = layer.get("elevation_source")
    if not isinstance(elevation_source, dict):
        raise QError("ensemble L layer elevation_source required")
    if elevation_source.get("sha256") != nominal_entry.get("sha256"):
        raise QError("L layer does not bind nominal elevation hash from source lock")
    if elevation_source.get("path") != nominal_entry.get("artifact_path"):
        raise QError("L layer elevation path does not match source lock")
    return {
        "layer": layer,
        "bin_width": float(bin_width),
        "bin_count": bin_count,
        "radius": float(radius),
        "pole": [float(value) for value in pole],
    }


def dataset_metadata(dataset, expected_pixel_size: float) -> dict[str, Any]:
    if dataset.count != 1:
        raise QError("raster must have exactly one band")
    if dataset.crs is None:
        raise QError("raster CRS required")
    a, b, _c, d, e, _f = list(dataset.transform)[:6]
    x_scale = math.hypot(a, d)
    y_scale = math.hypot(b, e)
    if (
        abs(x_scale - expected_pixel_size) > 1e-8
        or abs(y_scale - expected_pixel_size) > 1e-8
    ):
        raise QError("raster pixel size mismatch")
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


def aligned(reference, candidate, label: str) -> None:
    if reference.width != candidate.width or reference.height != candidate.height:
        raise QError(f"{label}: shape mismatch")
    if reference.crs != candidate.crs:
        raise QError(f"{label}: CRS mismatch")
    if not same_affine(reference.transform, candidate.transform):
        raise QError(f"{label}: transform mismatch")


def unit_vector(value, label: str):
    np, _rasterio, _warp_transform = require_rasterio()
    array = np.asarray(value, dtype="float64")
    magnitude = float(np.linalg.norm(array))
    if not math.isfinite(magnitude) or magnitude <= 1e-15:
        raise QError(f"{label} is degenerate")
    return array / magnitude


def local_basis(site_up, pole):
    np, _rasterio, _warp_transform = require_rasterio()
    up = unit_vector(site_up, "site up")
    pole_u = unit_vector(pole, "pole")
    north_raw = pole_u - up * float(np.dot(pole_u, up))
    if float(np.linalg.norm(north_raw)) <= 1e-10:
        axes = np.eye(3, dtype="float64")
        reference = axes[int(np.argmin(np.abs(axes @ up)))]
        north_raw = reference - up * float(np.dot(reference, up))
    north = unit_vector(north_raw, "north")
    east = unit_vector(np.cross(north, up), "east")
    north = unit_vector(np.cross(up, east), "north")
    return north, east, up


def lonlat_units(longitudes, latitudes):
    np, _rasterio, _warp_transform = require_rasterio()
    lon = np.radians(np.asarray(longitudes, dtype="float64"))
    lat = np.radians(np.asarray(latitudes, dtype="float64"))
    cos_lat = np.cos(lat)
    return np.stack(
        (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)), axis=1
    )


def nearest_rank(sorted_values, q: float) -> float:
    count = len(sorted_values)
    if count < 1:
        raise QError("cannot quantile empty ensemble")
    rank = max(1, int(math.ceil(float(q) * count)))
    return float(sorted_values[rank - 1])


def summarize_horizons(horizons, quantiles: list[float]):
    np, _rasterio, _warp_transform = require_rasterio()
    _member_count, bin_count = horizons.shape
    summaries: list[dict[str, Any]] = []
    for bin_index in range(bin_count):
        values = horizons[:, bin_index]
        finite_mask = np.isfinite(values)
        finite_values = values[finite_mask]
        members = np.nonzero(finite_mask)[0]
        if len(finite_values) == 0:
            summaries.append(
                {
                    "azimuth_bin_index": bin_index,
                    "member_count_with_candidate": 0,
                    "min_deg": None,
                    "max_deg": None,
                    "max_member_ordinal": None,
                    "quantiles_deg": {f"{q:.6f}": None for q in quantiles},
                }
            )
            continue
        order = np.argsort(finite_values, kind="mergesort")
        sorted_values = finite_values[order]
        local_max_index = int(np.argmax(finite_values))
        summaries.append(
            {
                "azimuth_bin_index": bin_index,
                "member_count_with_candidate": int(len(finite_values)),
                "min_deg": float(sorted_values[0]),
                "max_deg": float(sorted_values[-1]),
                "max_member_ordinal": int(members[local_max_index]),
                "quantiles_deg": {
                    f"{q:.6f}": nearest_rank(sorted_values, q) for q in quantiles
                },
            }
        )
    return summaries


def materialize(
    family_path: pathlib.Path,
    policy_path: pathlib.Path,
    clone_plan_path: pathlib.Path,
    clone_lock_path: pathlib.Path,
    nominal_lock_path: pathlib.Path,
    site_receipt_path: pathlib.Path,
    l_config_path: pathlib.Path,
    artifact_root: pathlib.Path,
) -> dict[str, Any]:
    np, rasterio, warp_transform = require_rasterio()
    family = safe_json(family_path, "clone family")
    policy = validate_policy(safe_json(policy_path, "ensemble policy"))
    clone_plan = safe_json(clone_plan_path, "clone acquisition plan")
    clone_lock = safe_json(clone_lock_path, "clone source lock")
    nominal_lock = safe_json(nominal_lock_path, "nominal source lock")
    site_receipt = safe_json(site_receipt_path, "LL-009P site receipt")
    l_config = safe_json(l_config_path, "LL-009L config")

    if family.get("study_id") != policy["study_id"]:
        raise QError("clone family/policy study mismatch")
    if family.get("member_count") != policy["required_member_count"]:
        raise QError("clone family member count does not match policy")
    if family.get("interpretation") != policy["clone_interpretation"]:
        raise QError("clone family interpretation does not match policy")
    clone_entries = validate_clone_plan_and_lock(
        family, clone_plan, clone_plan_path, clone_lock
    )
    if len(clone_entries) != policy["required_member_count"]:
        raise QError("clone lock does not contain required member count")
    if nominal_lock.get("study_id") != policy["study_id"]:
        raise QError("nominal source lock study mismatch")
    nominal_entry = source_by_id(
        nominal_lock, policy["nominal_elevation_source_id"], "nominal lock"
    )
    if nominal_entry.get("role") != "surface_elevation":
        raise QError("nominal source role must be surface_elevation")
    validate_site_receipt(site_receipt, nominal_lock_path, nominal_entry, policy)
    l_info = validate_l_config(l_config, site_receipt, nominal_entry, policy)

    nominal_path = resolve_locked(artifact_root, nominal_entry, "nominal elevation")
    clone_paths = [
        resolve_locked(artifact_root, entry, entry["source_id"])
        for entry in clone_entries
    ]

    layer = l_info["layer"]
    min_range = float(layer["min_range_m"])
    max_range = float(layer["max_range_m"])
    bin_width = l_info["bin_width"]
    bin_count = l_info["bin_count"]
    radius = l_info["radius"]
    pole = np.asarray(l_info["pole"], dtype="float64")
    expected_pixel_size = float(layer["expected_pixel_size_m"])
    site = site_receipt["selected_pixel"]
    site_row = int(site["row"])
    site_col = int(site["col"])
    site_x = float(site["center_x_m"])
    site_y = float(site["center_y_m"])
    nominal_site_elevation = float(site["elevation_m"])
    quantiles = [float(q) for q in policy["quantiles"]]

    with contextlib.ExitStack() as stack:
        nominal = stack.enter_context(rasterio.open(nominal_path))
        metadata = dataset_metadata(nominal, expected_pixel_size)
        if site_receipt.get("ll009l_expected_crs_wkt_sha256") != metadata[
            "crs_wkt_sha256"
        ]:
            raise QError("nominal raster CRS hash does not match LL-009P receipt")
        if (
            site_row < 0
            or site_col < 0
            or site_row >= nominal.height
            or site_col >= nominal.width
        ):
            raise QError("LL-009P site pixel outside nominal raster")
        nominal_site = nominal.read(
            1,
            window=rasterio.windows.Window(site_col, site_row, 1, 1),
            masked=True,
        )
        if bool(np.ma.getmaskarray(nominal_site)[0, 0]):
            raise QError("nominal site pixel is nodata")
        if abs(float(nominal_site[0, 0]) - nominal_site_elevation) > 1e-6:
            raise QError("nominal raster site elevation does not match LL-009P receipt")

        clone_datasets = []
        clone_site_values: list[float] = []
        for entry, path in zip(clone_entries, clone_paths):
            dataset = stack.enter_context(rasterio.open(path))
            aligned(nominal, dataset, entry["source_id"])
            dataset_metadata(dataset, expected_pixel_size)
            cell = dataset.read(
                1,
                window=rasterio.windows.Window(site_col, site_row, 1, 1),
                masked=True,
            )
            if bool(np.ma.getmaskarray(cell)[0, 0]):
                raise QError(f"{entry['source_id']}: site pixel is nodata")
            value = float(cell[0, 0])
            if not math.isfinite(value):
                raise QError(f"{entry['source_id']}: site realization value invalid")
            clone_datasets.append(dataset)
            clone_site_values.append(value)

        geographic_crs = rasterio.crs.CRS.from_proj4(
            f"+proj=longlat +R={radius:.12f} +no_defs +type=crs"
        )
        site_longitudes, site_latitudes = warp_transform(
            nominal.crs, geographic_crs, [site_x], [site_y]
        )
        site_up = lonlat_units(site_longitudes, site_latitudes)[0]
        north, east, up = local_basis(site_up, pole)

        site_positions = []
        for site_realization in clone_site_values:
            if policy["clone_interpretation"] == "additive_z_error_to_nominal_ldem":
                clone_site_elevation = nominal_site_elevation + site_realization
            else:
                clone_site_elevation = site_realization
            if not math.isfinite(clone_site_elevation):
                raise QError("clone site elevation invalid")
            site_positions.append((radius + clone_site_elevation) * site_up)
        site_positions_array = np.asarray(site_positions, dtype="float64")

        horizons = np.full(
            (len(clone_datasets), bin_count), -np.inf, dtype="float64"
        )
        admitted_per_member = np.zeros(len(clone_datasets), dtype="int64")
        scanned_annulus_pixels = 0

        bounding_box = (
            site_x - max_range,
            site_y - max_range,
            site_x + max_range,
            site_y + max_range,
        )
        transform = nominal.transform
        a, b, c, d, e, f = list(transform)[:6]

        for _, window in nominal.block_windows(1):
            left, bottom, right, top = rasterio.windows.bounds(window, transform)
            if (
                right < bounding_box[0]
                or left > bounding_box[2]
                or top < bounding_box[1]
                or bottom > bounding_box[3]
            ):
                continue
            nominal_array = nominal.read(1, window=window, masked=True)
            rows_local, cols_local = np.indices(nominal_array.shape)
            rows_global = rows_local.astype("float64") + float(window.row_off)
            cols_global = cols_local.astype("float64") + float(window.col_off)
            centers_col = cols_global + 0.5
            centers_row = rows_global + 0.5
            xs = a * centers_col + b * centers_row + c
            ys = d * centers_col + e * centers_row + f
            ranges = np.hypot(xs - site_x, ys - site_y)
            in_annulus = (
                (ranges >= min_range - 1e-9)
                & (ranges <= max_range + 1e-9)
                & (ranges > 1e-9)
            )
            scanned_annulus_pixels += int(np.count_nonzero(in_annulus))
            nominal_mask = np.ma.getmaskarray(nominal_array)
            if (
                layer.get("nodata_policy", "fail_required_coverage")
                == "fail_required_coverage"
                and np.any(in_annulus & nominal_mask)
            ):
                raise QError("nominal raster nodata inside required ensemble annulus")
            valid_nominal = in_annulus & ~nominal_mask
            if not np.any(valid_nominal):
                continue

            selected_x = xs[valid_nominal].astype("float64")
            selected_y = ys[valid_nominal].astype("float64")
            selected_nominal = np.asarray(
                nominal_array.data[valid_nominal], dtype="float64"
            )
            longitudes, latitudes = warp_transform(
                nominal.crs,
                geographic_crs,
                selected_x.tolist(),
                selected_y.tolist(),
            )
            terrain_units = lonlat_units(longitudes, latitudes)
            if terrain_units.shape[0] != selected_nominal.shape[0]:
                raise QError("coordinate transform cardinality drift")

            for member_index, dataset in enumerate(clone_datasets):
                clone_array = dataset.read(1, window=window, masked=True)
                clone_mask = np.ma.getmaskarray(clone_array)
                if (
                    layer.get("nodata_policy", "fail_required_coverage")
                    == "fail_required_coverage"
                    and np.any(valid_nominal & clone_mask)
                ):
                    raise QError(
                        f"{clone_entries[member_index]['source_id']}: nodata inside required annulus"
                    )
                compact_valid = ~clone_mask[valid_nominal]
                if not np.any(compact_valid):
                    continue
                clone_values = np.asarray(
                    clone_array.data[valid_nominal], dtype="float64"
                )[compact_valid]
                if not np.all(np.isfinite(clone_values)):
                    raise QError(
                        f"{clone_entries[member_index]['source_id']}: non-finite realization"
                    )
                nominal_values = selected_nominal[compact_valid]
                units = terrain_units[compact_valid]
                if policy["clone_interpretation"] == "additive_z_error_to_nominal_ldem":
                    elevations = nominal_values + clone_values
                else:
                    elevations = clone_values
                if not np.all(np.isfinite(elevations)):
                    raise QError("reconstructed clone elevation invalid")

                terrain_positions = units * (radius + elevations)[:, None]
                los = terrain_positions - site_positions_array[member_index]
                los_norm = np.linalg.norm(los, axis=1)
                if np.any(~np.isfinite(los_norm)) or np.any(los_norm <= 1e-12):
                    raise QError("degenerate/non-finite clone line of sight")
                direction = los / los_norm[:, None]
                north_component = direction @ north
                east_component = direction @ east
                up_component = direction @ up
                azimuth = (
                    np.degrees(np.arctan2(east_component, north_component)) % 360.0
                )
                elevation_deg = np.degrees(
                    np.arctan2(
                        up_component,
                        np.hypot(north_component, east_component),
                    )
                )
                bin_indices = (
                    np.floor((azimuth + 1e-9) / bin_width).astype("int64") % bin_count
                )
                np.maximum.at(horizons[member_index], bin_indices, elevation_deg)
                admitted_per_member[member_index] += len(elevation_deg)

        if scanned_annulus_pixels == 0:
            raise QError("ensemble annulus contains no nominal raster pixels")
        if np.any(admitted_per_member == 0):
            raise QError("one or more clone members admitted no pixels")
        if policy["require_all_azimuth_bins"]:
            missing = np.argwhere(~np.isfinite(horizons))
            if len(missing):
                first = missing[0]
                raise QError(
                    f"missing horizon candidate for member {int(first[0])} bin {int(first[1])}"
                )

        summaries = summarize_horizons(horizons, quantiles)
        members = []
        for ordinal, (entry, site_value) in enumerate(
            zip(clone_entries, clone_site_values)
        ):
            if policy["clone_interpretation"] == "additive_z_error_to_nominal_ldem":
                site_elevation = nominal_site_elevation + site_value
                site_error = site_value
            else:
                site_elevation = site_value
                site_error = site_value - nominal_site_elevation
            members.append(
                {
                    "ordinal": ordinal,
                    "source_id": entry["source_id"],
                    "sha256": entry["sha256"],
                    "artifact_path": entry["artifact_path"],
                    "site_realization_value_m": site_value,
                    "site_error_relative_to_nominal_m": site_error,
                    "site_elevation_m": site_elevation,
                    "admitted_pixels": int(admitted_per_member[ordinal]),
                    "horizon_deg": [
                        None if not math.isfinite(float(value)) else float(value)
                        for value in horizons[ordinal]
                    ],
                }
            )

    output = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "pass",
        "semantics_class": "empirical_ensemble",
        "study_id": policy["study_id"],
        "site_ref": site_receipt["site_ref"],
        "native_frame": site_receipt["native_frame"],
        "family_sha256": sha256_file(family_path)[0],
        "policy_sha256": sha256_file(policy_path)[0],
        "clone_plan_sha256": sha256_file(clone_plan_path)[0],
        "clone_lock_sha256": sha256_file(clone_lock_path)[0],
        "nominal_lock_sha256": sha256_file(nominal_lock_path)[0],
        "site_receipt_sha256": sha256_file(site_receipt_path)[0],
        "l_config_sha256": sha256_file(l_config_path)[0],
        "nominal_elevation_source_id": nominal_entry["source_id"],
        "nominal_elevation_source_sha256": nominal_entry["sha256"],
        "clone_interpretation": policy["clone_interpretation"],
        "observer_policy": policy["observer_policy"],
        "quantile_estimator": policy["quantile_estimator"],
        "quantiles": quantiles,
        "spatial_support_class": policy["spatial_support_class"],
        "azimuth_bin_width_deg": bin_width,
        "azimuth_bin_count": bin_count,
        "min_range_m": min_range,
        "max_range_m": max_range,
        "scanned_annulus_pixels": scanned_annulus_pixels,
        "member_count": len(members),
        "members": members,
        "per_bin_empirical_summary": summaries,
        "finite_ensemble_max_semantics": (
            "Per-bin max is the largest horizon elevation observed among the exact "
            "published ensemble members. It is not a deterministic upper bound on "
            "all possible physical terrain or on the ensemble-generating distribution."
        ),
        "theorem_boundary": (
            "Every emitted member horizon is computed from every admitted raster cell "
            "for that exact realization; LL-009L top-K reduction is not used in this "
            "ensemble scan. Statistical coverage beyond the finite ensemble is not inferred."
        ),
        "non_claims": [
            "This receipt is empirical ensemble evidence, not a hard terrain upper bound.",
            "The Site01 clone ensemble represents the error model constructed by Barker et al.; it does not prove all real terrain errors are contained by the 100 realizations.",
            "This receipt does not close unresolved terrain between raster support points.",
            "This receipt covers only the configured ensemble layer; far-field terrain uncertainty remains a separate gate.",
            "Visibility, illumination, communications, site viability and architecture viability remain downstream claims."
        ],
    }
    output["receipt_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def write_immutable(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise QError(f"refusing to overwrite differing output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> None:
    np, rasterio, _warp_transform = require_rasterio()
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        data_root = root / "data"
        data_root.mkdir()
        family = {
            "schema_version": QP.FAMILY_SCHEMA,
            "study_id": "study-q",
            "provider": "NASA Goddard PGDA / LRO-LOLA",
            "dataset_id": "pgda-site01-clones",
            "source_page": "https://pgda.gsfc.nasa.gov/products/78",
            "allowed_hosts": ["pgda.gsfc.nasa.gov"],
            "source_url_directory": "https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/Site01/Clones/",
            "artifact_directory": "clones",
            "source_id_prefix": "site01-clone",
            "filename_prefix": "Site01_final_adj_5mpp_",
            "filename_suffix": "_err.tif",
            "first_index": 1,
            "last_index": 4,
            "index_width": 4,
            "member_count": 4,
            "role": "terrain_error_realization_m",
            "interpretation": "additive_z_error_to_nominal_ldem",
            "interpretation_basis": "synthetic",
            "expected_sha256_by_index": {},
        }
        family_path = root / "family.json"
        family_path.write_bytes(canonical_bytes(family))
        clone_plan = QP.expand_family(family)
        clone_plan_path = root / "clone-plan.json"
        clone_plan_path.write_bytes(canonical_bytes(clone_plan))

        crs = CRS.from_proj4(
            "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 "
            "+R=1737400 +units=m +no_defs"
        )
        transform = from_origin(-45, 45, 10, 10)
        nominal_array = np.zeros((9, 9), dtype="float32")
        site_row = site_col = 4

        def write_raster(path: pathlib.Path, array, raster_transform=transform):
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=array.shape[1],
                height=array.shape[0],
                count=1,
                dtype="float32",
                crs=crs,
                transform=raster_transform,
                nodata=-9999.0,
            ) as dataset:
                dataset.write(array.astype("float32"), 1)

        nominal_rel = "nominal.tif"
        nominal_path = data_root / nominal_rel
        write_raster(nominal_path, nominal_array)
        nominal_hash, nominal_size = sha256_file(nominal_path)
        nominal_lock = {
            "schema_version": LOCK_SCHEMA,
            "study_id": "study-q",
            "files": [
                {
                    "source_id": "site01-elevation-5m",
                    "dataset_id": "synthetic",
                    "role": "surface_elevation",
                    "artifact_path": nominal_rel,
                    "sha256": nominal_hash,
                    "byte_size": nominal_size,
                }
            ],
        }
        nominal_lock_path = root / "nominal-lock.json"
        nominal_lock_path.write_bytes(canonical_bytes(nominal_lock))

        clone_arrays = []
        base = np.zeros((9, 9), dtype="float32")
        clone_arrays.append(base.copy())
        c2 = base.copy()
        c2[4, 6] = 20.0
        clone_arrays.append(c2)
        c3 = base.copy()
        c3[4, 4] = 10.0
        c3[4, 6] = 10.0
        clone_arrays.append(c3)
        c4 = base.copy()
        c4[2, 4] = 12.0
        clone_arrays.append(c4)

        clone_lock_files = []
        for planned, array in zip(clone_plan["files"], clone_arrays):
            rel = planned["artifact_path"]
            path = data_root / pathlib.Path(*safe_relpath(rel).parts)
            write_raster(path, array)
            digest, size = sha256_file(path)
            clone_lock_files.append(
                {
                    "source_id": planned["source_id"],
                    "dataset_id": planned["dataset_id"],
                    "role": planned["role"],
                    "source_url": planned["source_url"],
                    "artifact_path": rel,
                    "sha256": digest,
                    "byte_size": size,
                    "required_for": planned["required_for"],
                }
            )
        clone_lock = {
            "schema_version": LOCK_SCHEMA,
            "status": "exact_source_bytes_locked",
            "study_id": "study-q",
            "plan_sha256": sha256_file(clone_plan_path)[0],
            "files": clone_lock_files,
        }
        clone_lock_path = root / "clone-lock.json"
        clone_lock_path.write_bytes(canonical_bytes(clone_lock))

        site_x, site_y = transform * (site_col + 0.5, site_row + 0.5)
        with rasterio.open(nominal_path) as nominal_for_hash:
            opened_crs_hash = sha256_bytes(nominal_for_hash.crs.to_wkt().encode())
        site_receipt = {
            "schema_version": P_RECEIPT_SCHEMA,
            "status": "pass",
            "study_id": "study-q",
            "site_ref": "site-q",
            "native_frame": "MOON_ME_DE421",
            "source_lock_sha256": sha256_file(nominal_lock_path)[0],
            "elevation_source_id": "site01-elevation-5m",
            "elevation_source_sha256": nominal_hash,
            "selected_pixel": {
                "row": site_row,
                "col": site_col,
                "center_x_m": float(site_x),
                "center_y_m": float(site_y),
                "center_longitude_deg": 0.0,
                "center_latitude_deg": -90.0,
                "reference_to_center_offset_m": 0.0,
                "elevation_m": 0.0,
                "vertical_uncertainty_m": 0.5,
            },
            "ll009l_site_block": {
                "x_m": float(site_x),
                "y_m": float(site_y),
                "elevation_m": 0.0,
                "vertical_uncertainty_m": 0.5,
            },
            "ll009l_expected_crs_wkt_sha256": opened_crs_hash,
        }
        site_receipt_path = root / "site-receipt.json"
        site_receipt_path.write_bytes(canonical_bytes(site_receipt))

        with rasterio.open(nominal_path) as ds:
            crs_hash = sha256_bytes(ds.crs.to_wkt().encode())
        l_config = {
            "schema_version": L_CONFIG_SCHEMA,
            "study_id": "study-q",
            "frame_contract_id": "frame-q",
            "epoch_contract_id": "epoch-q",
            "site_ref": "site-q",
            "native_frame": "MOON_ME_DE421",
            "projection_contract": {
                "projection": "south_polar_stereographic",
                "reference_radius_m": 1737400.0,
                "central_meridian_deg": 0.0,
                "true_scale_at_pole": True,
                "pixel_registration": "center",
            },
            "site": site_receipt["ll009l_site_block"],
            "pole_vector": [0.0, 0.0, 1.0],
            "azimuth_bin_width_deg": 90.0,
            "top_k_per_bin": 1,
            "layers": [
                {
                    "layer_id": "near",
                    "min_range_m": 5.0,
                    "max_range_m": 40.0,
                    "expected_pixel_size_m": 10.0,
                    "expected_crs_wkt_sha256": crs_hash,
                    "additional_angular_margin_deg": 0.0,
                    "nodata_policy": "fail_required_coverage",
                    "elevation_source": {
                        "path": nominal_rel,
                        "sha256": nominal_hash,
                    },
                    "uncertainty_source": {
                        "path": nominal_rel,
                        "sha256": nominal_hash,
                    },
                }
            ],
        }
        l_config_path = root / "l-config.json"
        l_config_path.write_bytes(canonical_bytes(l_config))

        policy = {
            "schema_version": POLICY_SCHEMA,
            "study_id": "study-q",
            "ensemble_layer_id": "near",
            "nominal_elevation_source_id": "site01-elevation-5m",
            "required_member_count": 4,
            "clone_interpretation": "additive_z_error_to_nominal_ldem",
            "observer_policy": "same_realization_site_pixel",
            "quantile_estimator": "empirical_cdf_nearest_rank",
            "quantiles": [0.5, 0.75, 1.0],
            "require_all_azimuth_bins": True,
            "spatial_support_class": "sample_points_only",
        }
        policy_path = root / "policy.json"
        policy_path.write_bytes(canonical_bytes(policy))

        first = materialize(
            family_path,
            policy_path,
            clone_plan_path,
            clone_lock_path,
            nominal_lock_path,
            site_receipt_path,
            l_config_path,
            data_root,
        )
        second = materialize(
            family_path,
            policy_path,
            clone_plan_path,
            clone_lock_path,
            nominal_lock_path,
            site_receipt_path,
            l_config_path,
            data_root,
        )
        assert canonical_bytes(first) == canonical_bytes(second)
        assert first["member_count"] == 4
        assert first["members"][2]["site_error_relative_to_nominal_m"] == 10.0
        assert any(
            summary["max_deg"] is not None
            and summary["min_deg"] is not None
            and summary["max_deg"] > summary["min_deg"]
            for summary in first["per_bin_empirical_summary"]
        )
        assert first["spatial_support_class"] == "sample_points_only"

        shifted = from_origin(-44, 45, 10, 10)
        broken_path = data_root / pathlib.Path(
            *safe_relpath(clone_plan["files"][1]["artifact_path"]).parts
        )
        write_raster(broken_path, clone_arrays[1], shifted)
        digest, size = sha256_file(broken_path)
        clone_lock["files"][1]["sha256"] = digest
        clone_lock["files"][1]["byte_size"] = size
        clone_lock_path.write_bytes(canonical_bytes(clone_lock))
        try:
            materialize(
                family_path,
                policy_path,
                clone_plan_path,
                clone_lock_path,
                nominal_lock_path,
                site_receipt_path,
                l_config_path,
                data_root,
            )
        except QError as exc:
            assert "transform mismatch" in str(exc)
        else:
            raise QError("self-test expected shifted-clone alignment rejection")

        assert nearest_rank(np.asarray([1.0, 2.0, 3.0, 4.0]), 0.5) == 2.0
        assert nearest_rank(np.asarray([1.0, 2.0, 3.0, 4.0]), 0.75) == 3.0
        assert nearest_rank(np.asarray([1.0, 2.0, 3.0, 4.0]), 1.0) == 4.0
        print("LL-009Q ensemble self-test: PASS", rasterio.__version__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LL-009Q exact clone-ensemble near-field horizon materializer"
    )
    parser.add_argument("--family")
    parser.add_argument("--policy")
    parser.add_argument("--clone-plan")
    parser.add_argument("--clone-lock")
    parser.add_argument("--nominal-lock")
    parser.add_argument("--site-receipt")
    parser.add_argument("--l-config")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        required = [
            args.family,
            args.policy,
            args.clone_plan,
            args.clone_lock,
            args.nominal_lock,
            args.site_receipt,
            args.l_config,
            args.output,
        ]
        if not all(required):
            raise QError(
                "--family --policy --clone-plan --clone-lock --nominal-lock "
                "--site-receipt --l-config --output are required"
            )
        output = materialize(
            pathlib.Path(args.family),
            pathlib.Path(args.policy),
            pathlib.Path(args.clone_plan),
            pathlib.Path(args.clone_lock),
            pathlib.Path(args.nominal_lock),
            pathlib.Path(args.site_receipt),
            pathlib.Path(args.l_config),
            pathlib.Path(args.artifact_root),
        )
        write_immutable(pathlib.Path(args.output), output)
        print(json.dumps(output, sort_keys=True, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, QError) as exc:
        raise SystemExit(f"LL-009Q ensemble failure: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
