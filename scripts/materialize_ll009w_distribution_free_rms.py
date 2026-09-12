#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
from typing import Any

POLICY_SCHEMA = "ll009w.distribution-free-rms-policy.v1"
V_SCHEMA = "ll009v.product90-rms-semantics-receipt.v1"
L_SCHEMA = "ll009l.cog-materialization-config.v1"
OUT_SCHEMA = "ll009w.distribution-free-familywise-horizon-receipt.v1"


class WError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_obj(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise WError(f"{label} must contain an object")
    return value


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def require_hex(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise WError(f"{label} must be lowercase SHA-256")
    return value


def verify_self_hash(value: dict[str, Any], label: str) -> None:
    expected = require_hex(value.get("receipt_sha256"), f"{label} receipt_sha256")
    body = json.loads(json.dumps(value))
    body.pop("receipt_sha256", None)
    actual = sha256_bytes(canonical_bytes(body))
    if actual != expected:
        raise WError(f"{label}: receipt self-hash mismatch")


def safe_relpath(value: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value:
        raise WError("source path must be non-empty string")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise WError(f"unsafe source path {value!r}")
    return path


def require_rasterio():
    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as warp_transform
    except Exception as exc:
        raise WError("Rasterio/Numpy required only for offline LL-009W execution") from exc
    return np, rasterio, warp_transform


def same_affine(left, right, tolerance: float = 1e-12) -> bool:
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def vec3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(finite(x) for x in value):
        raise WError(f"{label} must be finite vec3")
    return tuple(float(x) for x in value)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, s):
    return tuple(x * s for x in a)


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
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        raise WError(f"{label} degenerate")
    return scale(a, 1.0 / magnitude)


def local_basis(site, pole):
    up = unit(site, "site")
    pole_u = unit(pole, "pole")
    north_raw = sub(pole_u, scale(up, dot(pole_u, up)))
    fallback = False
    if norm(north_raw) <= 1e-10:
        fallback = True
        axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        reference = min(axes, key=lambda axis: abs(dot(axis, up)))
        north_raw = sub(reference, scale(up, dot(reference, up)))
    north = unit(north_raw, "north")
    east = unit(cross(north, up), "east")
    north = unit(cross(up, east), "north")
    return north, east, up, fallback


def az_el(direction, basis):
    north, east, up, _ = basis
    d = unit(direction, "line of sight")
    n = dot(d, north)
    e = dot(d, east)
    u = dot(d, up)
    return (
        math.degrees(math.atan2(e, n)) % 360.0,
        math.degrees(math.atan2(u, math.hypot(n, e))),
    )


def lonlat_to_unit(lon_deg: float, lat_deg: float):
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    cos_lat = math.cos(lat)
    return (
        cos_lat * math.cos(lon),
        cos_lat * math.sin(lon),
        math.sin(lat),
    )


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise WError(f"policy schema must be {POLICY_SCHEMA}")
    for key in (
        "study_id",
        "layer_id",
        "theorem",
        "distribution_assumption",
        "independence_assumption",
        "height_to_geometry_mapping",
        "risk_scope",
    ):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise WError(f"policy missing {key}")
    if policy["theorem"] != "markov_squared_error_plus_union_bound":
        raise WError("unsupported theorem")
    if policy["distribution_assumption"] != "none_beyond_rms_second_moment_model":
        raise WError("W forbids undeclared distribution assumptions")
    if policy["independence_assumption"] != "none":
        raise WError("W does not require or permit an independence assumption")
    if policy["height_to_geometry_mapping"] != "positive_scalar_height_error_along_local_lunar_radial":
        raise WError("unsupported height_to_geometry_mapping")
    if policy["risk_scope"] != "represented_admitted_raster_support_points":
        raise WError("unsupported risk_scope")
    alpha = policy.get("familywise_exceedance_budget_alpha")
    if not finite(alpha) or not (0.0 < float(alpha) < 1.0):
        raise WError("familywise_exceedance_budget_alpha must be in (0,1)")
    if policy.get("multiplier_rule") != "derive_exactly_sqrt_admitted_pixel_count_over_alpha":
        raise WError("multiplier_rule must derive exact sqrt(N/alpha)")
    if policy.get("multiplier_override") is not None:
        raise WError("multiplier_override is forbidden; multiplier must be data-derived")
    if policy.get("require_all_azimuth_bins") is not True:
        raise WError("V1 requires all azimuth bins")
    return policy


def validate_v_receipt(v: dict[str, Any], policy: dict[str, Any], l_config_path: pathlib.Path) -> None:
    if v.get("schema_version") != V_SCHEMA or v.get("status") != "pass":
        raise WError("V receipt must be passing LL-009V")
    verify_self_hash(v, "V receipt")
    if v.get("study_id") != policy["study_id"] or v.get("layer_id") != policy["layer_id"]:
        raise WError("V policy lineage mismatch")
    if v.get("semantics_class") != "rms_error":
        raise WError("W requires V semantics_class=rms_error")
    if v.get("deterministic_upper_bound_eligible") is not False:
        raise WError("V receipt unexpectedly marks RMS as deterministic bound")
    if v.get("statistical_closure_required") is not True:
        raise WError("V receipt must require statistical closure")
    if v.get("l_config_sha256") != sha256_file(l_config_path):
        raise WError("V receipt does not bind exact L config")


def resolve_source(root: pathlib.Path, entry: dict[str, Any], label: str) -> pathlib.Path:
    path = entry.get("path")
    digest = entry.get("sha256")
    if not isinstance(path, str):
        raise WError(f"{label}: path missing")
    require_hex(digest, f"{label} sha256")
    full = root / pathlib.Path(*safe_relpath(path).parts)
    if not full.is_file():
        raise WError(f"{label}: source missing")
    if sha256_file(full) != digest:
        raise WError(f"{label}: source hash mismatch")
    return full


def validate_l_config(l_config: dict[str, Any], policy: dict[str, Any], v: dict[str, Any]) -> dict[str, Any]:
    if l_config.get("schema_version") != L_SCHEMA or l_config.get("study_id") != policy["study_id"]:
        raise WError("L config schema/study mismatch")
    projection = l_config.get("projection_contract")
    if not isinstance(projection, dict) or projection.get("projection") != "south_polar_stereographic":
        raise WError("south_polar_stereographic projection required")
    radius = projection.get("reference_radius_m")
    if not finite(radius) or float(radius) <= 0:
        raise WError("invalid lunar reference radius")
    site = l_config.get("site")
    if not isinstance(site, dict) or not all(finite(site.get(k)) for k in ("x_m", "y_m", "elevation_m")):
        raise WError("L site x/y/elevation required")
    pole = vec3(l_config.get("pole_vector"), "pole_vector")
    width = l_config.get("azimuth_bin_width_deg")
    if not finite(width) or float(width) <= 0 or float(width) > 90:
        raise WError("invalid azimuth bin width")
    count = int(round(360.0 / float(width)))
    if count < 4 or abs(count * float(width) - 360.0) > 1e-9:
        raise WError("azimuth bin width must divide 360")
    matches = [
        x
        for x in l_config.get("layers", [])
        if isinstance(x, dict) and x.get("layer_id") == policy["layer_id"]
    ]
    if len(matches) != 1:
        raise WError("expected exactly one W layer in L config")
    layer = matches[0]
    minimum, maximum = layer.get("min_range_m"), layer.get("max_range_m")
    if not finite(minimum) or not finite(maximum) or float(minimum) < 0 or float(maximum) <= float(minimum):
        raise WError("invalid layer radial range")
    expected_px = layer.get("expected_pixel_size_m")
    if not finite(expected_px) or float(expected_px) <= 0:
        raise WError("expected_pixel_size_m required")
    max_eff = layer.get("max_effective_resolution_m")
    if max_eff is not None and (not finite(max_eff) or float(max_eff) <= 0):
        raise WError("invalid max_effective_resolution_m")
    for key in ("elevation_source", "uncertainty_source", "effective_resolution_source"):
        if not isinstance(layer.get(key), dict):
            raise WError(f"{key} required")
    u = layer["uncertainty_source"]
    if u.get("sha256") != v.get("uncertainty_source_sha256"):
        raise WError("L uncertainty hash differs from V receipt")
    if u.get("path") != v.get("uncertainty_artifact_path"):
        raise WError("L uncertainty path differs from V receipt")
    return {
        "radius": float(radius),
        "site": {k: float(site[k]) for k in ("x_m", "y_m", "elevation_m")},
        "pole": pole,
        "bin_width": float(width),
        "bin_count": count,
        "layer": layer,
        "min_range": float(minimum),
        "max_range": float(maximum),
        "expected_pixel_size": float(expected_px),
        "max_effective_resolution": None if max_eff is None else float(max_eff),
    }


def dataset_metadata(dataset, layer: dict[str, Any], label: str) -> dict[str, Any]:
    if dataset.count != 1:
        raise WError(f"{label}: exactly one band required")
    if dataset.crs is None:
        raise WError(f"{label}: CRS required")
    wkt = dataset.crs.to_wkt()
    observed_hash = sha256_bytes(wkt.encode())
    expected_hash = layer.get("expected_crs_wkt_sha256")
    if not isinstance(expected_hash, str) or observed_hash != expected_hash:
        raise WError(f"{label}: CRS WKT hash mismatch")
    a, b, _c, d, e, _f = list(dataset.transform)[:6]
    sx = math.hypot(a, d)
    sy = math.hypot(b, e)
    expected = float(layer["expected_pixel_size_m"])
    if abs(sx - expected) > 1e-8 or abs(sy - expected) > 1e-8:
        raise WError(f"{label}: pixel size mismatch")
    return {
        "width": dataset.width,
        "height": dataset.height,
        "nodata": dataset.nodata,
        "dtype": dataset.dtypes[0],
        "transform": list(dataset.transform)[:6],
        "crs_wkt_sha256": observed_hash,
        "pixel_scale_x_m": sx,
        "pixel_scale_y_m": sy,
    }


def aligned(reference, other, label: str) -> None:
    if reference.width != other.width or reference.height != other.height:
        raise WError(f"{label}: shape mismatch")
    if reference.crs != other.crs:
        raise WError(f"{label}: CRS mismatch")
    if not same_affine(reference.transform, other.transform):
        raise WError(f"{label}: affine transform mismatch")


def derive_multiplier(admitted_count: int, alpha: float) -> float:
    if not isinstance(admitted_count, int) or admitted_count <= 0:
        raise WError("admitted pixel count must be positive")
    if not (0.0 < alpha < 1.0):
        raise WError("alpha invalid")
    return math.sqrt(admitted_count / alpha)


def scan_admission(elevation, uncertainty, effective, info, np, rasterio) -> dict[str, Any]:
    layer = info["layer"]
    site_x, site_y = info["site"]["x_m"], info["site"]["y_m"]
    minimum, maximum = info["min_range"], info["max_range"]
    max_eff = info["max_effective_resolution"]
    scanned = admitted = nodata = eff_rejected = 0
    max_rms = 0.0
    bbox = (site_x - maximum, site_y - maximum, site_x + maximum, site_y + maximum)
    for _, window in elevation.block_windows(1):
        left, bottom, right, top = rasterio.windows.bounds(window, elevation.transform)
        if right < bbox[0] or left > bbox[2] or top < bbox[1] or bottom > bbox[3]:
            continue
        ea = elevation.read(1, window=window, masked=True)
        ua = uncertainty.read(1, window=window, masked=True)
        ra = effective.read(1, window=window, masked=True)
        rows, cols = np.indices(ea.shape)
        rows = rows + int(window.row_off)
        cols = cols + int(window.col_off)
        xs = elevation.transform.c + (cols + 0.5) * elevation.transform.a + (rows + 0.5) * elevation.transform.b
        ys = elevation.transform.f + (cols + 0.5) * elevation.transform.d + (rows + 0.5) * elevation.transform.e
        ranges = np.hypot(xs - site_x, ys - site_y)
        annulus = (ranges >= minimum - 1e-9) & (ranges <= maximum + 1e-9)
        scanned += int(np.count_nonzero(annulus))
        masks = np.ma.getmaskarray(ea) | np.ma.getmaskarray(ua) | np.ma.getmaskarray(ra)
        nodata_here = annulus & masks
        nodata += int(np.count_nonzero(nodata_here))
        valid = annulus & ~masks
        if np.any(valid):
            ev = np.asarray(ea.data[valid], dtype="float64")
            uv = np.asarray(ua.data[valid], dtype="float64")
            rv = np.asarray(ra.data[valid], dtype="float64")
            if np.any(~np.isfinite(ev)) or np.any(~np.isfinite(uv)) or np.any(uv < 0):
                raise WError("non-finite/negative elevation or RMS within admitted annulus")
            if np.any(~np.isfinite(rv)) or np.any(rv <= 0):
                raise WError("invalid effective resolution within annulus")
            if max_eff is not None:
                keep = rv <= max_eff
                eff_rejected += int(np.count_nonzero(~keep))
                uv = uv[keep]
                admitted += int(np.count_nonzero(keep))
            else:
                admitted += int(len(uv))
            if len(uv):
                max_rms = max(max_rms, float(np.max(uv)))
    if layer.get("nodata_policy", "fail_required_coverage") == "fail_required_coverage" and nodata:
        raise WError(f"{nodata} nodata pixels inside required annulus")
    if admitted <= 0:
        raise WError("no admitted pixels")
    return {
        "scanned_annulus_pixels": scanned,
        "admitted_pixel_count": admitted,
        "nodata_pixels_in_annulus": nodata,
        "effective_resolution_rejected_pixels": eff_rejected,
        "max_rms_m": max_rms,
    }


def materialize(
    policy_path: pathlib.Path,
    v_receipt_path: pathlib.Path,
    l_config_path: pathlib.Path,
    artifact_root: pathlib.Path,
) -> dict[str, Any]:
    np, rasterio, warp_transform = require_rasterio()
    policy = validate_policy(read_obj(policy_path, "W policy"))
    v = read_obj(v_receipt_path, "V receipt")
    l_config = read_obj(l_config_path, "L config")
    validate_v_receipt(v, policy, l_config_path)
    info = validate_l_config(l_config, policy, v)
    layer = info["layer"]
    elevation_path = resolve_source(artifact_root, layer["elevation_source"], "elevation")
    uncertainty_path = resolve_source(artifact_root, layer["uncertainty_source"], "RMS uncertainty")
    effective_path = resolve_source(artifact_root, layer["effective_resolution_source"], "effective resolution")

    with rasterio.open(elevation_path) as elevation, rasterio.open(uncertainty_path) as uncertainty, rasterio.open(effective_path) as effective:
        aligned(elevation, uncertainty, "elevation/RMS")
        aligned(elevation, effective, "elevation/effective-resolution")
        metadata = dataset_metadata(elevation, layer, "elevation")
        dataset_metadata(uncertainty, layer, "RMS uncertainty")
        dataset_metadata(effective, layer, "effective resolution")

        admission = scan_admission(elevation, uncertainty, effective, info, np, rasterio)
        alpha = float(policy["familywise_exceedance_budget_alpha"])
        multiplier = derive_multiplier(admission["admitted_pixel_count"], alpha)
        per_pixel_tail_bound = 1.0 / (multiplier * multiplier)
        union_bound = admission["admitted_pixel_count"] * per_pixel_tail_bound

        radius = info["radius"]
        geographic_crs = rasterio.crs.CRS.from_proj4(
            f"+proj=longlat +R={radius:.12f} +no_defs +type=crs"
        )
        site_x, site_y = info["site"]["x_m"], info["site"]["y_m"]
        site_lon, site_lat = warp_transform(elevation.crs, geographic_crs, [site_x], [site_y])
        site_unit = lonlat_to_unit(float(site_lon[0]), float(site_lat[0]))
        site_position = scale(site_unit, radius + info["site"]["elevation_m"])
        basis = local_basis(site_position, info["pole"])

        bins: list[dict[str, Any] | None] = [None] * info["bin_count"]
        maximum_applied_vertical_margin_m = 0.0
        admitted_second_pass = 0
        minimum, maximum = info["min_range"], info["max_range"]
        max_eff = info["max_effective_resolution"]
        bbox = (site_x - maximum, site_y - maximum, site_x + maximum, site_y + maximum)

        for _, window in elevation.block_windows(1):
            left, bottom, right, top = rasterio.windows.bounds(window, elevation.transform)
            if right < bbox[0] or left > bbox[2] or top < bbox[1] or bottom > bbox[3]:
                continue
            ea = elevation.read(1, window=window, masked=True)
            ua = uncertainty.read(1, window=window, masked=True)
            ra = effective.read(1, window=window, masked=True)
            rows_local, cols_local = np.indices(ea.shape)
            rows = rows_local + int(window.row_off)
            cols = cols_local + int(window.col_off)
            xs = elevation.transform.c + (cols + 0.5) * elevation.transform.a + (rows + 0.5) * elevation.transform.b
            ys = elevation.transform.f + (cols + 0.5) * elevation.transform.d + (rows + 0.5) * elevation.transform.e
            ranges = np.hypot(xs - site_x, ys - site_y)
            annulus = (ranges >= minimum - 1e-9) & (ranges <= maximum + 1e-9)
            masks = np.ma.getmaskarray(ea) | np.ma.getmaskarray(ua) | np.ma.getmaskarray(ra)
            valid = annulus & ~masks
            if max_eff is not None:
                valid &= np.asarray(ra.data, dtype="float64") <= max_eff
            if not np.any(valid):
                continue

            selected_rows = rows[valid].astype("int64")
            selected_cols = cols[valid].astype("int64")
            selected_xs = xs[valid].astype("float64")
            selected_ys = ys[valid].astype("float64")
            selected_range = ranges[valid].astype("float64")
            elevations = np.asarray(ea.data[valid], dtype="float64")
            rms = np.asarray(ua.data[valid], dtype="float64")
            eff = np.asarray(ra.data[valid], dtype="float64")
            margins = multiplier * rms
            if len(margins):
                maximum_applied_vertical_margin_m = max(
                    maximum_applied_vertical_margin_m, float(np.max(margins))
                )
            longitudes, latitudes = warp_transform(
                elevation.crs,
                geographic_crs,
                selected_xs.tolist(),
                selected_ys.tolist(),
            )
            for row, col, x, y, rng, z, s, er, margin, lon, lat in zip(
                selected_rows,
                selected_cols,
                selected_xs,
                selected_ys,
                selected_range,
                elevations,
                rms,
                eff,
                margins,
                longitudes,
                latitudes,
            ):
                if not all(math.isfinite(float(q)) for q in (z, s, er, margin, lon, lat)):
                    raise WError("non-finite second-pass value")
                terrain_unit = lonlat_to_unit(float(lon), float(lat))
                upper_position = scale(terrain_unit, radius + float(z) + float(margin))
                azimuth, elevation_deg = az_el(sub(upper_position, site_position), basis)
                bin_index = int(math.floor((azimuth + 1e-9) / info["bin_width"])) % info["bin_count"]
                candidate = {
                    "bin_index": bin_index,
                    "azimuth_deg": azimuth,
                    "upper_horizon_elevation_deg": elevation_deg,
                    "row": int(row),
                    "col": int(col),
                    "x_m": float(x),
                    "y_m": float(y),
                    "range_xy_m": float(rng),
                    "nominal_elevation_m": float(z),
                    "rms_uncertainty_m": float(s),
                    "applied_vertical_margin_m": float(margin),
                    "effective_resolution_m": float(er),
                    "longitude_deg": float(lon),
                    "latitude_deg": float(lat),
                }
                current = bins[bin_index]
                if current is None or (
                    candidate["upper_horizon_elevation_deg"],
                    -candidate["row"],
                    -candidate["col"],
                ) > (
                    current["upper_horizon_elevation_deg"],
                    -current["row"],
                    -current["col"],
                ):
                    bins[bin_index] = candidate
                admitted_second_pass += 1

        if admitted_second_pass != admission["admitted_pixel_count"]:
            raise WError("admitted pixel count drift between theorem and geometry passes")
        missing = [i for i, item in enumerate(bins) if item is None]
        if policy["require_all_azimuth_bins"] and missing:
            raise WError("missing horizon bins: " + ",".join(map(str, missing)))

    theorem = {
        "name": "markov_squared_error_plus_union_bound",
        "model_condition": (
            "For every admitted pixel i, the Product 90 RMS scale s_i is treated as "
            "satisfying E[e_i^2] <= s_i^2 for the scalar surface-height error e_i."
        ),
        "familywise_exceedance_budget_alpha": alpha,
        "admitted_pixel_count": admission["admitted_pixel_count"],
        "derived_multiplier_k": multiplier,
        "per_pixel_two_sided_markov_bound": per_pixel_tail_bound,
        "familywise_union_bound": union_bound,
        "distribution_assumption": policy["distribution_assumption"],
        "independence_assumption": policy["independence_assumption"],
        "risk_scope": policy["risk_scope"],
        "height_to_geometry_mapping": policy["height_to_geometry_mapping"],
    }
    output = {
        "schema_version": OUT_SCHEMA,
        "status": "pass",
        "semantics_class": "distribution_free_familywise_rms_upper_envelope",
        "study_id": policy["study_id"],
        "layer_id": policy["layer_id"],
        "policy_sha256": sha256_file(policy_path),
        "v_receipt_sha256": sha256_file(v_receipt_path),
        "l_config_sha256": sha256_file(l_config_path),
        "source_hashes": {
            "elevation": layer["elevation_source"]["sha256"],
            "rms_uncertainty": layer["uncertainty_source"]["sha256"],
            "effective_resolution": layer["effective_resolution_source"]["sha256"],
        },
        "observer": {
            "policy": "nominal_l_config_site_only_no_site_uncertainty",
            "x_m": info["site"]["x_m"],
            "y_m": info["site"]["y_m"],
            "elevation_m": info["site"]["elevation_m"],
            "position_m": list(site_position),
        },
        "basis": {
            "north": list(basis[0]),
            "east": list(basis[1]),
            "up": list(basis[2]),
            "fallback_used": basis[3],
        },
        "raster_metadata": metadata,
        "admission": admission,
        "theorem": theorem,
        "maximum_applied_vertical_margin_m": maximum_applied_vertical_margin_m,
        "azimuth_bin_width_deg": info["bin_width"],
        "bin_count": info["bin_count"],
        "bins": bins,
        "claim_rule": (
            "Conditional on the per-pixel RMS second-moment model, the union bound makes "
            "the probability that any represented admitted raster support point exceeds "
            "its z + k*RMS positive-height envelope no greater than alpha. No independence "
            "or Gaussian tail model is used."
        ),
        "non_claims": [
            "This is not a deterministic terrain upper bound.",
            "The familywise probability statement is conditional on the RMS second-moment model being valid for every admitted pixel.",
            "The theorem covers represented admitted raster support points only; unresolved terrain between support points remains an LL-009R obligation.",
            "The nominal observer used for this receipt does not close Site01 observer RMS/clone uncertainty; LL-009U handles same-realization observer propagation.",
            "This receipt is not a site, visibility, operations or architecture qualification.",
        ],
    }
    output["receipt_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def write_immutable(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise WError(f"refusing to overwrite differing output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> None:
    np, rasterio, _warp_transform = require_rasterio()
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        crs = CRS.from_proj4(
            "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs"
        )
        transform = from_origin(-25, 25, 10, 10)
        elevation = np.zeros((5, 5), dtype="float32")
        elevation[2, 4] = 10.0
        rms = np.full((5, 5), 0.5, dtype="float32")
        effective = np.full((5, 5), 10.0, dtype="float32")

        def write_raster(name, array, raster_transform=transform):
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
                nodata=-9999.0,
            ) as ds:
                ds.write(array, 1)
            return path

        ep = write_raster("e.tif", elevation)
        up = write_raster("u.tif", rms)
        rp = write_raster("r.tif", effective)
        with rasterio.open(ep) as ds:
            crs_hash = sha256_bytes(ds.crs.to_wkt().encode())

        l = {
            "schema_version": L_SCHEMA,
            "study_id": "study-w",
            "frame_contract_id": "frame-w",
            "epoch_contract_id": "epoch-w",
            "site_ref": "site-w",
            "native_frame": "MOON_ME_DE421",
            "projection_contract": {
                "projection": "south_polar_stereographic",
                "reference_radius_m": 1737400.0,
                "central_meridian_deg": 0.0,
                "true_scale_at_pole": True,
            },
            "site": {
                "x_m": 0.0,
                "y_m": 0.0,
                "elevation_m": 0.0,
                "vertical_uncertainty_m": 0.5,
            },
            "pole_vector": [0.0, 0.0, 1.0],
            "azimuth_bin_width_deg": 90.0,
            "top_k_per_bin": 2,
            "layers": [
                {
                    "layer_id": "far",
                    "min_range_m": 5.0,
                    "max_range_m": 40.0,
                    "expected_pixel_size_m": 10.0,
                    "expected_crs_wkt_sha256": crs_hash,
                    "max_effective_resolution_m": 12.0,
                    "nodata_policy": "fail_required_coverage",
                    "elevation_source": {"path": "e.tif", "sha256": sha256_file(ep)},
                    "uncertainty_source": {"path": "u.tif", "sha256": sha256_file(up)},
                    "effective_resolution_source": {"path": "r.tif", "sha256": sha256_file(rp)},
                }
            ],
        }
        lp = root / "l.json"
        lp.write_bytes(canonical_bytes(l))
        v = {
            "schema_version": V_SCHEMA,
            "status": "pass",
            "study_id": "study-w",
            "layer_id": "far",
            "uncertainty_source_sha256": sha256_file(up),
            "uncertainty_artifact_path": "u.tif",
            "semantics_class": "rms_error",
            "deterministic_upper_bound_eligible": False,
            "statistical_closure_required": True,
            "l_config_sha256": sha256_file(lp),
        }
        v["receipt_sha256"] = sha256_bytes(canonical_bytes(v))
        vp = root / "v.json"
        vp.write_bytes(canonical_bytes(v))
        policy = {
            "schema_version": POLICY_SCHEMA,
            "study_id": "study-w",
            "layer_id": "far",
            "theorem": "markov_squared_error_plus_union_bound",
            "familywise_exceedance_budget_alpha": 0.25,
            "multiplier_rule": "derive_exactly_sqrt_admitted_pixel_count_over_alpha",
            "multiplier_override": None,
            "distribution_assumption": "none_beyond_rms_second_moment_model",
            "independence_assumption": "none",
            "height_to_geometry_mapping": "positive_scalar_height_error_along_local_lunar_radial",
            "risk_scope": "represented_admitted_raster_support_points",
            "require_all_azimuth_bins": True,
        }
        pp = root / "policy.json"
        pp.write_bytes(canonical_bytes(policy))
        first = materialize(pp, vp, lp, root)
        second = materialize(pp, vp, lp, root)
        assert canonical_bytes(first) == canonical_bytes(second)
        n = first["admission"]["admitted_pixel_count"]
        expected_k = math.sqrt(n / 0.25)
        assert abs(first["theorem"]["derived_multiplier_k"] - expected_k) < 1e-12
        assert abs(first["theorem"]["familywise_union_bound"] - 0.25) < 1e-12
        assert first["semantics_class"] == "distribution_free_familywise_rms_upper_envelope"
        assert len(first["bins"]) == 4 and all(item is not None for item in first["bins"])

        l_small = json.loads(json.dumps(l))
        l_small["layers"][0]["max_range_m"] = 25.0
        lsp = root / "l-small.json"
        lsp.write_bytes(canonical_bytes(l_small))
        v_small = json.loads(json.dumps(v))
        v_small.pop("receipt_sha256", None)
        v_small["l_config_sha256"] = sha256_file(lsp)
        v_small["receipt_sha256"] = sha256_bytes(canonical_bytes(v_small))
        vsp = root / "v-small.json"
        vsp.write_bytes(canonical_bytes(v_small))
        small = materialize(pp, vsp, lsp, root)
        assert small["admission"]["admitted_pixel_count"] < n
        ratio = first["theorem"]["derived_multiplier_k"] / small["theorem"]["derived_multiplier_k"]
        expected_ratio = math.sqrt(n / small["admission"]["admitted_pixel_count"])
        assert abs(ratio - expected_ratio) < 1e-12

        bad_policy = json.loads(json.dumps(policy))
        bad_policy["multiplier_override"] = 1.0
        bpp = root / "bad-policy.json"
        bpp.write_bytes(canonical_bytes(bad_policy))
        try:
            materialize(bpp, vp, lp, root)
            raise AssertionError("unauthorized multiplier accepted")
        except WError as exc:
            assert "forbidden" in str(exc)

        bad_l = json.loads(json.dumps(l))
        bad_l["layers"][0]["uncertainty_source"]["sha256"] = "1" * 64
        blp = root / "bad-l.json"
        blp.write_bytes(canonical_bytes(bad_l))
        bad_v = json.loads(json.dumps(v))
        bad_v.pop("receipt_sha256", None)
        bad_v["l_config_sha256"] = sha256_file(blp)
        bad_v["uncertainty_source_sha256"] = "1" * 64
        bad_v["receipt_sha256"] = sha256_bytes(canonical_bytes(bad_v))
        bvp = root / "bad-v.json"
        bvp.write_bytes(canonical_bytes(bad_v))
        try:
            materialize(pp, bvp, blp, root)
            raise AssertionError("source hash drift accepted")
        except WError as exc:
            assert "hash" in str(exc).lower()

        gaussian = json.loads(json.dumps(policy))
        gaussian["distribution_assumption"] = "gaussian"
        gp = root / "gaussian.json"
        gp.write_bytes(canonical_bytes(gaussian))
        try:
            materialize(gp, vp, lp, root)
            raise AssertionError("Gaussian assumption accepted")
        except WError:
            pass

        print(
            "LL-009W distribution-free RMS envelope self-test: PASS",
            rasterio.__version__,
            "N=",
            n,
            "k=",
            expected_k,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LL-009W distribution-free Product 90 RMS familywise horizon"
    )
    parser.add_argument("--policy")
    parser.add_argument("--v-receipt")
    parser.add_argument("--l-config")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not all((args.policy, args.v_receipt, args.l_config, args.output)):
            raise WError("--policy --v-receipt --l-config --output required")
        out = materialize(
            pathlib.Path(args.policy),
            pathlib.Path(args.v_receipt),
            pathlib.Path(args.l_config),
            pathlib.Path(args.artifact_root),
        )
        write_immutable(pathlib.Path(args.output), out)
        print(json.dumps(out, sort_keys=True, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, WError) as exc:
        raise SystemExit(f"LL-009W failure: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
