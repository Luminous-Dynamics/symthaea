#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
from typing import Any

import materialize_ll009w_distribution_free_rms as W

POLICY_SCHEMA = "ll009x.geometry-aware-rms-policy.v1"
OUT_SCHEMA = "ll009x.geometry-aware-familywise-rms-horizon-receipt.v1"
RECORD_DTYPE_FIELDS = [
    ("c", "<f8"),
    ("radius_m", "<f8"),
    ("rms_m", "<f8"),
    ("row", "<i8"),
    ("col", "<i8"),
    ("nominal_elevation_deg", "<f8"),
]


class XError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return W.canonical_bytes(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return W.sha256_file(path)


def read_obj(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise XError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise XError(f"{label} must contain object")
    return value


def finite(value: Any) -> bool:
    return W.finite(value)


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise XError(f"policy schema must be {POLICY_SCHEMA}")
    for key in ("study_id", "layer_id", "risk_allocation", "solver"):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise XError(f"policy missing {key}")
    if policy["risk_allocation"] != "proportional_to_admitted_pixel_count_by_azimuth_bin":
        raise XError("V1 requires count-proportional bin risk allocation")
    if policy["solver"] != "deterministic_bisection":
        raise XError("V1 requires deterministic_bisection")
    iterations = policy.get("bisection_iterations")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or not 32 <= iterations <= 160:
        raise XError("bisection_iterations must be integer in [32,160]")
    for key in ("risk_tolerance_abs", "domination_tolerance_deg"):
        if not finite(policy.get(key)) or float(policy[key]) <= 0:
            raise XError(f"{key} must be positive")
    if policy.get("distribution_assumption") != "inherit_ll009w_none_beyond_rms_second_moment_model":
        raise XError("X may not add a distribution assumption")
    if policy.get("independence_assumption") != "inherit_ll009w_none":
        raise XError("X may not add an independence assumption")
    return policy


def verify_receipt_self_hash(receipt: dict[str, Any], label: str) -> None:
    expected = receipt.get("receipt_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise XError(f"{label} receipt_sha256 missing")
    body = json.loads(json.dumps(receipt))
    body.pop("receipt_sha256", None)
    if sha256_bytes(canonical_bytes(body)) != expected:
        raise XError(f"{label} receipt self-hash mismatch")


def validate_w(w: dict[str, Any], policy: dict[str, Any], l_config_path: pathlib.Path) -> dict[str, Any]:
    if w.get("schema_version") != W.OUT_SCHEMA or w.get("status") != "pass":
        raise XError("W receipt must be passing LL-009W")
    verify_receipt_self_hash(w, "W")
    if w.get("semantics_class") != "distribution_free_familywise_rms_upper_envelope":
        raise XError("unexpected W semantics")
    if w.get("study_id") != policy["study_id"] or w.get("layer_id") != policy["layer_id"]:
        raise XError("W/policy lineage mismatch")
    if w.get("l_config_sha256") != sha256_file(l_config_path):
        raise XError("W does not bind exact L config")
    theorem = w.get("theorem")
    if not isinstance(theorem, dict):
        raise XError("W theorem missing")
    if theorem.get("name") != "markov_squared_error_plus_union_bound":
        raise XError("unexpected W theorem")
    if theorem.get("distribution_assumption") != "none_beyond_rms_second_moment_model":
        raise XError("W distribution semantics drift")
    if theorem.get("independence_assumption") != "none":
        raise XError("W independence semantics drift")
    alpha = theorem.get("familywise_exceedance_budget_alpha")
    count = theorem.get("admitted_pixel_count")
    if not finite(alpha) or not (0 < float(alpha) < 1):
        raise XError("W alpha invalid")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise XError("W admitted count invalid")
    bins = w.get("bins")
    bin_count = w.get("bin_count")
    width = w.get("azimuth_bin_width_deg")
    if not isinstance(bin_count, int) or not finite(width) or not isinstance(bins, list) or len(bins) != bin_count:
        raise XError("W bin metadata invalid")
    mapped = {}
    for item in bins:
        if not isinstance(item, dict):
            raise XError("W requires complete bins")
        index = item.get("bin_index")
        horizon = item.get("upper_horizon_elevation_deg")
        if not isinstance(index, int) or index in mapped or not finite(horizon):
            raise XError("W bin invalid/duplicate")
        mapped[index] = item
    if set(mapped) != set(range(bin_count)):
        raise XError("W bin coverage incomplete")
    return {
        "alpha": float(alpha),
        "count": count,
        "bin_count": bin_count,
        "bin_width": float(width),
        "bins": mapped,
    }


def source_paths(l_config: dict[str, Any], policy: dict[str, Any], w: dict[str, Any], root: pathlib.Path):
    if l_config.get("schema_version") != W.L_SCHEMA or l_config.get("study_id") != policy["study_id"]:
        raise XError("L config schema/study mismatch")
    matches = [
        x for x in l_config.get("layers", [])
        if isinstance(x, dict) and x.get("layer_id") == policy["layer_id"]
    ]
    if len(matches) != 1:
        raise XError("expected exactly one X layer")
    layer = matches[0]
    sources = w.get("source_hashes")
    if not isinstance(sources, dict):
        raise XError("W source hashes missing")
    expected = {
        "elevation": layer.get("elevation_source", {}).get("sha256"),
        "rms_uncertainty": layer.get("uncertainty_source", {}).get("sha256"),
        "effective_resolution": layer.get("effective_resolution_source", {}).get("sha256"),
    }
    if sources != expected:
        raise XError("W/L source hash set mismatch")
    try:
        elevation = W.resolve_source(root, layer["elevation_source"], "elevation")
        rms = W.resolve_source(root, layer["uncertainty_source"], "RMS")
        effective = W.resolve_source(root, layer["effective_resolution_source"], "effective resolution")
    except W.WError as exc:
        raise XError(str(exc)) from exc
    return layer, elevation, rms, effective


def risk_contributions(records, horizon_deg: float, observer_radius: float, np):
    c = records["c"]
    q = np.sqrt(np.maximum(0.0, 1.0 - c * c))
    tangent = math.tan(math.radians(float(horizon_deg)))
    denominator = c - q * tangent
    radius = records["radius_m"]
    rms = records["rms_m"]
    delta = np.full(len(records), np.inf, dtype="float64")
    possible = denominator > 0.0
    delta[possible] = observer_radius / denominator[possible] - radius[possible]
    contribution = np.zeros(len(records), dtype="float64")
    nonzero_rms = rms > 0.0
    at_or_below = possible & (delta <= 0.0) & nonzero_rms
    contribution[at_or_below] = 1.0
    positive = possible & (delta > 0.0) & nonzero_rms
    if np.any(positive):
        ratio = rms[positive] / delta[positive]
        contribution[positive] = np.minimum(1.0, ratio * ratio)
    return delta, contribution


def solve_bin(
    records,
    alpha_bin: float,
    w_horizon: float,
    observer_radius: float,
    iterations: int,
    tolerance: float,
    np,
):
    if len(records) == 0:
        raise XError("cannot solve empty bin")
    nominal_max = float(np.max(records["nominal_elevation_deg"]))
    if w_horizon + 1e-12 < nominal_max:
        raise XError("W horizon below nominal skyline")
    _, hi_contrib = risk_contributions(records, w_horizon, observer_radius, np)
    hi_risk = float(np.sum(hi_contrib, dtype="float64"))
    if hi_risk > alpha_bin + tolerance:
        raise XError(
            f"W horizon is not feasible under reconstructed bin budget: {hi_risk} > {alpha_bin}"
        )
    _, lo_contrib = risk_contributions(records, nominal_max, observer_radius, np)
    lo_risk = float(np.sum(lo_contrib, dtype="float64"))
    if lo_risk <= alpha_bin + tolerance:
        return nominal_max, lo_risk, nominal_max, 0
    lo, hi = nominal_max, w_horizon
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        _, contrib = risk_contributions(records, mid, observer_radius, np)
        risk = float(np.sum(contrib, dtype="float64"))
        if risk <= alpha_bin:
            hi = mid
        else:
            lo = mid
    _, final_contrib = risk_contributions(records, hi, observer_radius, np)
    final_risk = float(np.sum(final_contrib, dtype="float64"))
    if final_risk > alpha_bin + tolerance:
        raise XError("bisection emitted risk above budget")
    return hi, final_risk, nominal_max, iterations


def contribution_digest(records, horizon_deg: float, observer_radius: float, np) -> str:
    h = hashlib.sha256()
    chunk = 250_000
    for start in range(0, len(records), chunk):
        part = records[start:start + chunk]
        delta, bound = risk_contributions(part, horizon_deg, observer_radius, np)
        out_dtype = np.dtype([
            ("row", "<i8"),
            ("col", "<i8"),
            ("delta_m", "<f8"),
            ("markov_bound", "<f8"),
        ])
        out = np.empty(len(part), dtype=out_dtype)
        out["row"] = part["row"]
        out["col"] = part["col"]
        out["delta_m"] = delta
        out["markov_bound"] = bound
        h.update(out.tobytes(order="C"))
    return h.hexdigest()


def materialize(
    policy_path: pathlib.Path,
    w_receipt_path: pathlib.Path,
    l_config_path: pathlib.Path,
    artifact_root: pathlib.Path,
) -> dict[str, Any]:
    np, rasterio, warp_transform = W.require_rasterio()
    policy = validate_policy(read_obj(policy_path, "X policy"))
    w = read_obj(w_receipt_path, "W receipt")
    l_config = read_obj(l_config_path, "L config")
    winfo = validate_w(w, policy, l_config_path)
    layer, elevation_path, rms_path, effective_path = source_paths(
        l_config, policy, w, artifact_root
    )

    projection = l_config.get("projection_contract", {})
    radius = projection.get("reference_radius_m")
    if not finite(radius) or float(radius) <= 0:
        raise XError("reference radius invalid")
    radius = float(radius)
    site = l_config.get("site")
    if not isinstance(site, dict) or not all(
        finite(site.get(k)) for k in ("x_m", "y_m", "elevation_m")
    ):
        raise XError("site state invalid")
    site_x, site_y, site_elevation = (
        float(site["x_m"]),
        float(site["y_m"]),
        float(site["elevation_m"]),
    )
    pole = W.vec3(l_config.get("pole_vector"), "pole")
    minimum, maximum = layer.get("min_range_m"), layer.get("max_range_m")
    if not finite(minimum) or not finite(maximum):
        raise XError("layer range invalid")
    minimum, maximum = float(minimum), float(maximum)
    max_eff = layer.get("max_effective_resolution_m")
    max_eff = None if max_eff is None else float(max_eff)

    with rasterio.open(elevation_path) as elevation, rasterio.open(rms_path) as rms_ds, rasterio.open(effective_path) as effective:
        try:
            W.aligned(elevation, rms_ds, "elevation/RMS")
            W.aligned(elevation, effective, "elevation/effective")
            W.dataset_metadata(elevation, layer, "elevation")
            W.dataset_metadata(rms_ds, layer, "RMS")
            W.dataset_metadata(effective, layer, "effective")
        except W.WError as exc:
            raise XError(str(exc)) from exc
        geographic_crs = rasterio.crs.CRS.from_proj4(
            f"+proj=longlat +R={radius:.12f} +no_defs +type=crs"
        )
        site_lon, site_lat = warp_transform(
            elevation.crs, geographic_crs, [site_x], [site_y]
        )
        site_unit = W.lonlat_to_unit(float(site_lon[0]), float(site_lat[0]))
        observer_radius = radius + site_elevation
        site_position = W.scale(site_unit, observer_radius)
        basis = W.local_basis(site_position, pole)
        if w.get("observer", {}).get("position_m") is None:
            raise XError("W observer position missing")
        w_site = W.vec3(w["observer"]["position_m"], "W observer")
        if max(abs(a - b) for a, b in zip(w_site, site_position)) > 1e-6:
            raise XError("W/L observer position mismatch")

        record_dtype = np.dtype(RECORD_DTYPE_FIELDS)
        with tempfile.TemporaryDirectory(prefix="ll009x-") as scratch:
            scratch_root = pathlib.Path(scratch)
            paths = [
                scratch_root / f"bin-{i:04d}.bin"
                for i in range(winfo["bin_count"])
            ]
            handles = [path.open("wb") for path in paths]
            counts = [0] * winfo["bin_count"]
            nominal_max = [-math.inf] * winfo["bin_count"]
            try:
                bbox = (
                    site_x - maximum,
                    site_y - maximum,
                    site_x + maximum,
                    site_y + maximum,
                )
                for _, window in elevation.block_windows(1):
                    left, bottom, right, top = rasterio.windows.bounds(
                        window, elevation.transform
                    )
                    if (
                        right < bbox[0]
                        or left > bbox[2]
                        or top < bbox[1]
                        or bottom > bbox[3]
                    ):
                        continue
                    ea = elevation.read(1, window=window, masked=True)
                    ua = rms_ds.read(1, window=window, masked=True)
                    ra = effective.read(1, window=window, masked=True)
                    rows_local, cols_local = np.indices(ea.shape)
                    rows = rows_local + int(window.row_off)
                    cols = cols_local + int(window.col_off)
                    xs = (
                        elevation.transform.c
                        + (cols + 0.5) * elevation.transform.a
                        + (rows + 0.5) * elevation.transform.b
                    )
                    ys = (
                        elevation.transform.f
                        + (cols + 0.5) * elevation.transform.d
                        + (rows + 0.5) * elevation.transform.e
                    )
                    ranges = np.hypot(xs - site_x, ys - site_y)
                    annulus = (
                        (ranges >= minimum - 1e-9)
                        & (ranges <= maximum + 1e-9)
                    )
                    masks = (
                        np.ma.getmaskarray(ea)
                        | np.ma.getmaskarray(ua)
                        | np.ma.getmaskarray(ra)
                    )
                    if (
                        layer.get("nodata_policy", "fail_required_coverage")
                        == "fail_required_coverage"
                        and np.any(annulus & masks)
                    ):
                        raise XError("nodata inside required annulus")
                    valid = annulus & ~masks
                    if max_eff is not None:
                        rdata = np.asarray(ra.data, dtype="float64")
                        if np.any(valid & (~np.isfinite(rdata) | (rdata <= 0))):
                            raise XError("invalid effective resolution")
                        valid &= rdata <= max_eff
                    if not np.any(valid):
                        continue
                    z = np.asarray(ea.data[valid], dtype="float64")
                    s = np.asarray(ua.data[valid], dtype="float64")
                    if (
                        np.any(~np.isfinite(z))
                        or np.any(~np.isfinite(s))
                        or np.any(s < 0)
                    ):
                        raise XError("invalid elevation/RMS")
                    sx = xs[valid].astype("float64")
                    sy = ys[valid].astype("float64")
                    slon, slat = warp_transform(
                        elevation.crs,
                        geographic_crs,
                        sx.tolist(),
                        sy.tolist(),
                    )
                    lon = np.radians(np.asarray(slon, dtype="float64"))
                    lat = np.radians(np.asarray(slat, dtype="float64"))
                    clat = np.cos(lat)
                    units = np.stack(
                        (clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)),
                        axis=1,
                    )
                    c = units @ np.asarray(site_unit, dtype="float64")
                    c = np.clip(c, -1.0, 1.0)
                    q = np.sqrt(np.maximum(0.0, 1.0 - c * c))
                    if np.any(q <= 1e-15):
                        raise XError(
                            "admitted support point is radially coincident with observer"
                        )
                    terrain_radius = radius + z
                    nominal_el = np.degrees(
                        np.arctan2(
                            terrain_radius * c - observer_radius,
                            terrain_radius * q,
                        )
                    )
                    north = units @ np.asarray(basis[0], dtype="float64")
                    east = units @ np.asarray(basis[1], dtype="float64")
                    azimuth = np.degrees(np.arctan2(east, north)) % 360.0
                    bin_index = (
                        np.floor((azimuth + 1e-9) / winfo["bin_width"])
                        .astype("int64")
                        % winfo["bin_count"]
                    )
                    vr = rows[valid].astype("int64")
                    vc = cols[valid].astype("int64")
                    for b in np.unique(bin_index):
                        mask = bin_index == b
                        rec = np.empty(
                            int(np.count_nonzero(mask)), dtype=record_dtype
                        )
                        rec["c"] = c[mask]
                        rec["radius_m"] = terrain_radius[mask]
                        rec["rms_m"] = s[mask]
                        rec["row"] = vr[mask]
                        rec["col"] = vc[mask]
                        rec["nominal_elevation_deg"] = nominal_el[mask]
                        handles[int(b)].write(rec.tobytes(order="C"))
                        counts[int(b)] += len(rec)
                        nominal_max[int(b)] = max(
                            nominal_max[int(b)],
                            float(np.max(rec["nominal_elevation_deg"])),
                        )
            finally:
                for handle in handles:
                    handle.close()

            total = sum(counts)
            if total != winfo["count"]:
                raise XError(
                    f"X/W admitted population mismatch: {total} != {winfo['count']}"
                )
            if any(count <= 0 for count in counts):
                raise XError("X requires all bins to contain admitted pixels")

            alpha = winfo["alpha"]
            budgets = [alpha * count / total for count in counts]
            budget_sum = math.fsum(budgets)
            if (
                abs(budget_sum - alpha)
                > float(policy["risk_tolerance_abs"])
            ):
                raise XError("bin budgets do not sum to whole-sky alpha")

            output_bins = []
            for b, (path, count, budget) in enumerate(
                zip(paths, counts, budgets)
            ):
                records = np.memmap(
                    path, dtype=record_dtype, mode="r", shape=(count,)
                )
                w_horizon = float(
                    winfo["bins"][b]["upper_horizon_elevation_deg"]
                )
                horizon, risk, nominal, used_iterations = solve_bin(
                    records,
                    budget,
                    w_horizon,
                    observer_radius,
                    int(policy["bisection_iterations"]),
                    float(policy["risk_tolerance_abs"]),
                    np,
                )
                if (
                    horizon
                    > w_horizon + float(policy["domination_tolerance_deg"])
                ):
                    raise XError("X horizon exceeds W baseline")
                digest = contribution_digest(
                    records, horizon, observer_radius, np
                )
                input_digest = sha256_file(path)
                output_bins.append(
                    {
                        "bin_index": b,
                        "admitted_pixel_count": count,
                        "alpha_bin": budget,
                        "nominal_max_horizon_deg": nominal,
                        "w_uniform_baseline_horizon_deg": w_horizon,
                        "optimized_horizon_deg": horizon,
                        "improvement_vs_w_deg": w_horizon - horizon,
                        "summed_markov_bound_at_horizon": risk,
                        "solver_iterations": used_iterations,
                        "input_population_digest_sha256": input_digest,
                        "threshold_contribution_digest_sha256": digest,
                    }
                )
                del records

    out = {
        "schema_version": OUT_SCHEMA,
        "status": "pass",
        "semantics_class": (
            "distribution_free_geometry_aware_familywise_rms_upper_envelope"
        ),
        "study_id": policy["study_id"],
        "layer_id": policy["layer_id"],
        "policy_sha256": sha256_file(policy_path),
        "w_receipt_sha256": sha256_file(w_receipt_path),
        "l_config_sha256": sha256_file(l_config_path),
        "source_hashes": w["source_hashes"],
        "whole_sky_alpha": winfo["alpha"],
        "admitted_pixel_count": total,
        "risk_allocation": policy["risk_allocation"],
        "sum_alpha_bins": budget_sum,
        "solver": {
            "name": policy["solver"],
            "bisection_iterations": int(policy["bisection_iterations"]),
            "risk_tolerance_abs": float(policy["risk_tolerance_abs"]),
            "domination_tolerance_deg": float(
                policy["domination_tolerance_deg"]
            ),
        },
        "geometry_theorem": {
            "radial_height_monotonicity": (
                "for q>0, d/dR [(R*c-Rs)/(R*q)] = Rs/(R^2*q) > 0"
            ),
            "radial_azimuth_invariance": (
                "observer has no tangent component in its own local frame, so tangent "
                "LOS is R*projection_tangent(u_i) and radial scaling cannot change azimuth"
            ),
            "threshold_radius": (
                "R_target = R_s / (c - q*tan(H)) when denominator>0"
            ),
            "tail_bound": (
                "P(e_i > delta_i(H)) <= min(1, s_i^2/delta_i(H)^2)"
            ),
        },
        "bins": output_bins,
        "claim_rule": (
            "Each bin receives alpha_b = alpha*N_b/N, exactly matching the aggregate budget "
            "that W's uniform alpha/N allocation gave the same pixels. X then solves the "
            "smallest geometry-aware skyline whose summed per-pixel Markov bounds fit alpha_b."
        ),
        "non_claims": [
            "X introduces no Gaussian, covariance or independence assumption beyond LL-009W.",
            "The envelope remains conditional on the LL-009V/W RMS second-moment model.",
            "The theorem covers represented raster support points, not unresolved terrain between them.",
            "The observer is the same nominal observer as W; LL-009U or a successor must propagate Q member-specific observer states.",
            "This is not site, visibility or operations authority.",
        ],
    }
    out["receipt_sha256"] = sha256_bytes(canonical_bytes(out))
    return out


def write_immutable(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise XError(f"refusing to overwrite differing output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> None:
    np, rasterio, _warp_transform = W.require_rasterio()
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        crs = CRS.from_proj4(
            "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs"
        )
        transform = from_origin(-25, 25, 10, 10)
        elevation = np.full((5, 5), -100.0, dtype="float32")
        elevation[2, 4] = 20.0
        rms = np.full((5, 5), 0.5, dtype="float32")
        rms[2, 4] = 2.0
        effective = np.full((5, 5), 10.0, dtype="float32")

        def write_raster(name, array):
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
                transform=transform,
                nodata=-9999.0,
            ) as ds:
                ds.write(array, 1)
            return path

        ep = write_raster("e.tif", elevation)
        up = write_raster("u.tif", rms)
        rp = write_raster("r.tif", effective)
        with rasterio.open(ep) as ds:
            crs_hash = W.sha256_bytes(ds.crs.to_wkt().encode())

        l = {
            "schema_version": W.L_SCHEMA,
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
                    "elevation_source": {
                        "path": "e.tif",
                        "sha256": W.sha256_file(ep),
                    },
                    "uncertainty_source": {
                        "path": "u.tif",
                        "sha256": W.sha256_file(up),
                    },
                    "effective_resolution_source": {
                        "path": "r.tif",
                        "sha256": W.sha256_file(rp),
                    },
                }
            ],
        }
        lp = root / "l.json"
        lp.write_bytes(W.canonical_bytes(l))
        v = {
            "schema_version": W.V_SCHEMA,
            "status": "pass",
            "study_id": "study-x",
            "layer_id": "far",
            "uncertainty_source_sha256": W.sha256_file(up),
            "uncertainty_artifact_path": "u.tif",
            "semantics_class": "rms_error",
            "deterministic_upper_bound_eligible": False,
            "statistical_closure_required": True,
            "l_config_sha256": W.sha256_file(lp),
        }
        v["receipt_sha256"] = W.sha256_bytes(W.canonical_bytes(v))
        vp = root / "v.json"
        vp.write_bytes(W.canonical_bytes(v))
        wp = {
            "schema_version": W.POLICY_SCHEMA,
            "study_id": "study-x",
            "layer_id": "far",
            "theorem": "markov_squared_error_plus_union_bound",
            "familywise_exceedance_budget_alpha": 0.25,
            "multiplier_rule": "derive_exactly_sqrt_admitted_pixel_count_over_alpha",
            "multiplier_override": None,
            "distribution_assumption": "none_beyond_rms_second_moment_model",
            "independence_assumption": "none",
            "height_to_geometry_mapping": (
                "positive_scalar_height_error_along_local_lunar_radial"
            ),
            "risk_scope": "represented_admitted_raster_support_points",
            "require_all_azimuth_bins": True,
        }
        wpp = root / "w-policy.json"
        wpp.write_bytes(W.canonical_bytes(wp))
        wrec = W.materialize(wpp, vp, lp, root)
        wrp = root / "w.json"
        wrp.write_bytes(W.canonical_bytes(wrec))

        xp = {
            "schema_version": POLICY_SCHEMA,
            "study_id": "study-x",
            "layer_id": "far",
            "risk_allocation": (
                "proportional_to_admitted_pixel_count_by_azimuth_bin"
            ),
            "solver": "deterministic_bisection",
            "bisection_iterations": 80,
            "risk_tolerance_abs": 1e-12,
            "domination_tolerance_deg": 1e-9,
            "distribution_assumption": (
                "inherit_ll009w_none_beyond_rms_second_moment_model"
            ),
            "independence_assumption": "inherit_ll009w_none",
        }
        xpp = root / "x-policy.json"
        xpp.write_bytes(canonical_bytes(xp))
        first = materialize(xpp, wrp, lp, root)
        second = materialize(xpp, wrp, lp, root)
        assert canonical_bytes(first) == canonical_bytes(second)
        assert (
            first["admitted_pixel_count"]
            == wrec["admission"]["admitted_pixel_count"]
        )
        assert abs(first["sum_alpha_bins"] - first["whole_sky_alpha"]) < 1e-12
        assert all(
            b["summed_markov_bound_at_horizon"] <= b["alpha_bin"] + 1e-12
            for b in first["bins"]
        )
        assert all(
            b["optimized_horizon_deg"]
            <= b["w_uniform_baseline_horizon_deg"] + 1e-9
            for b in first["bins"]
        )
        assert any(b["improvement_vs_w_deg"] > 1e-6 for b in first["bins"])

        observer = (1737400.0, 0.0, 0.0)
        basis = W.local_basis(observer, (0.0, 0.0, 1.0))
        terrain_unit = W.unit((0.9999, 0.01, 0.01), "terrain")
        p1 = W.scale(terrain_unit, 1737400.0)
        p2 = W.scale(terrain_unit, 1738400.0)
        az1 = W.az_el(W.sub(p1, observer), basis)[0]
        az2 = W.az_el(W.sub(p2, observer), basis)[0]
        diff = abs(((az1 - az2 + 180.0) % 360.0) - 180.0)
        assert diff < 1e-12

        c = W.dot(terrain_unit, W.unit(observer, "observer"))
        q = math.sqrt(max(0.0, 1.0 - c * c))
        target_h = 5.0
        denom = c - q * math.tan(math.radians(target_h))
        target_radius = W.norm(observer) / denom
        direct_h = W.az_el(
            W.sub(W.scale(terrain_unit, target_radius), observer), basis
        )[1]
        assert abs(direct_h - target_h) < 1e-9

        tampered = json.loads(json.dumps(wrec))
        tampered["theorem"]["admitted_pixel_count"] += 1
        trp = root / "tampered-w.json"
        trp.write_bytes(W.canonical_bytes(tampered))
        try:
            materialize(xpp, trp, lp, root)
            raise AssertionError("tampered W accepted")
        except XError as exc:
            assert "self-hash" in str(exc)

        print(
            "LL-009X geometry-aware RMS optimizer self-test: PASS",
            rasterio.__version__,
            "max_improvement_deg=",
            max(b["improvement_vs_w_deg"] for b in first["bins"]),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LL-009X geometry-aware distribution-free RMS horizon optimizer"
    )
    parser.add_argument("--policy")
    parser.add_argument("--w-receipt")
    parser.add_argument("--l-config")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not all((args.policy, args.w_receipt, args.l_config, args.output)):
            raise XError("--policy --w-receipt --l-config --output required")
        out = materialize(
            pathlib.Path(args.policy),
            pathlib.Path(args.w_receipt),
            pathlib.Path(args.l_config),
            pathlib.Path(args.artifact_root),
        )
        write_immutable(pathlib.Path(args.output), out)
        print(json.dumps(out, sort_keys=True, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, XError, W.WError) as exc:
        raise SystemExit(f"LL-009X failure: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
