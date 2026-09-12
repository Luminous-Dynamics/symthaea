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

POLICY_SCHEMA = "ll009y.memberwise-hybrid-horizon-policy.v1"
Q_SCHEMA = "ll009q.clone-horizon-ensemble-receipt.v1"
V_SCHEMA = "ll009v.product90-rms-semantics-receipt.v1"
X_SCHEMA = "ll009x.geometry-aware-familywise-rms-horizon-receipt.v1"
R_SCHEMA = "ll009r.spatial-support-classification-receipt.v1"
K_SCHEMA = "ll009k.horizon-pack.v1"
PRODUCER_SCHEMA = "ll009y.memberwise-hybrid-horizon-receipt.v1"
RECORD_DTYPE_FIELDS = [
    ("c", "<f8"),
    ("radius_m", "<f8"),
    ("rms_m", "<f8"),
    ("row", "<i8"),
    ("col", "<i8"),
    ("nominal_elevation_deg", "<f8"),
]


class YError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return W.canonical_bytes(value)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return W.sha256_file(path)


def finite(value: Any) -> bool:
    return W.finite(value)


def read_obj(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise YError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise YError(f"{label} must contain object")
    return value


def verify_self_hash(value: dict[str, Any], label: str) -> None:
    expected = value.get("receipt_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise YError(f"{label}: receipt_sha256 missing/invalid")
    body = json.loads(json.dumps(value))
    body.pop("receipt_sha256", None)
    if sha256_bytes(canonical_bytes(body)) != expected:
        raise YError(f"{label}: receipt self-hash mismatch")


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise YError(f"policy schema must be {POLICY_SCHEMA}")
    required = (
        "study_id",
        "near_layer_id",
        "far_layer_id",
        "quantile_estimator",
        "full_horizon_composition",
        "far_envelope_mode",
        "spatial_support_margin_policy",
    )
    for key in required:
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise YError(f"policy missing {key}")
    if policy["near_layer_id"] == policy["far_layer_id"]:
        raise YError("near/far layer IDs must differ")
    if policy["quantile_estimator"] != "empirical_cdf_nearest_rank":
        raise YError("V1 requires empirical_cdf_nearest_rank")
    if policy["full_horizon_composition"] != "memberwise_max_then_empirical_quantile":
        raise YError("V1 requires memberwise max before ensemble statistic")
    if policy["far_envelope_mode"] != "geometry_aware_distribution_free_rms_for_each_q_observer_scenario":
        raise YError("unsupported far_envelope_mode")
    if policy["spatial_support_margin_policy"] != "apply_exact_r_observed_positive_excursion_if_present":
        raise YError("unsupported spatial-support margin policy")
    q = policy.get("quantile")
    if not finite(q) or not 0.0 < float(q) <= 1.0:
        raise YError("quantile must be in (0,1]")
    member_batch = policy.get("member_batch_size")
    pixel_chunk = policy.get("pixel_chunk_size")
    if not isinstance(member_batch, int) or isinstance(member_batch, bool) or not 1 <= member_batch <= 64:
        raise YError("member_batch_size must be integer in [1,64]")
    if not isinstance(pixel_chunk, int) or isinstance(pixel_chunk, bool) or not 1024 <= pixel_chunk <= 1_000_000:
        raise YError("pixel_chunk_size must be integer in [1024,1000000]")
    if policy.get("require_all_members_all_bins") is not True:
        raise YError("V1 requires complete member/bin coverage")
    return policy


def validate_q(q: dict[str, Any], policy: dict[str, Any], l_path: pathlib.Path) -> dict[str, Any]:
    if q.get("schema_version") != Q_SCHEMA or q.get("status") != "pass":
        raise YError("Q receipt must be passing clone evidence")
    verify_self_hash(q, "Q receipt")
    if q.get("semantics_class") != "empirical_ensemble":
        raise YError("Q semantics must remain empirical_ensemble")
    if q.get("study_id") != policy["study_id"]:
        raise YError("Q study mismatch")
    if q.get("observer_policy") != "same_realization_site_pixel":
        raise YError("Q observer policy mismatch")
    if q.get("quantile_estimator") != policy["quantile_estimator"]:
        raise YError("Q/Y estimator mismatch")
    if q.get("l_config_sha256") != sha256_file(l_path):
        raise YError("Q does not bind exact L config")
    declared = q.get("quantiles")
    if not isinstance(declared, list) or not any(
        finite(x) and abs(float(x) - float(policy["quantile"])) <= 1e-15
        for x in declared
    ):
        raise YError("requested Y quantile is not declared by Q")
    width, count = q.get("azimuth_bin_width_deg"), q.get("azimuth_bin_count")
    members, member_count = q.get("members"), q.get("member_count")
    if not finite(width) or not isinstance(count, int) or count < 4:
        raise YError("Q bin metadata invalid")
    if not isinstance(member_count, int) or member_count < 2 or not isinstance(members, list) or len(members) != member_count:
        raise YError("Q member cardinality invalid")
    ids, hashes, parsed = set(), set(), []
    for ordinal, member in enumerate(members):
        if not isinstance(member, dict) or member.get("ordinal") != ordinal:
            raise YError("Q member ordering drift")
        source_id, digest = member.get("source_id"), member.get("sha256")
        site_elevation, horizon = member.get("site_elevation_m"), member.get("horizon_deg")
        if not isinstance(source_id, str) or not source_id or source_id in ids:
            raise YError("Q source ID invalid/duplicate")
        if not isinstance(digest, str) or len(digest) != 64 or digest in hashes:
            raise YError("Q source hash invalid/duplicate")
        if not finite(site_elevation) or not isinstance(horizon, list) or len(horizon) != count or not all(finite(x) for x in horizon):
            raise YError("Q member geometry invalid")
        ids.add(source_id)
        hashes.add(digest)
        parsed.append(
            {
                "ordinal": ordinal,
                "source_id": source_id,
                "sha256": digest,
                "site_elevation_m": float(site_elevation),
                "near_horizon_deg": [float(x) for x in horizon],
            }
        )
    return {
        "bin_width": float(width),
        "bin_count": count,
        "member_count": member_count,
        "members": parsed,
        "site_ref": q.get("site_ref"),
        "native_frame": q.get("native_frame"),
    }


def validate_v_w_x(
    v: dict[str, Any],
    w: dict[str, Any],
    x: dict[str, Any],
    v_path: pathlib.Path,
    w_path: pathlib.Path,
    x_policy_path: pathlib.Path,
    l_path: pathlib.Path,
    policy: dict[str, Any],
) -> dict[str, Any]:
    l_hash = sha256_file(l_path)
    if v.get("schema_version") != V_SCHEMA or v.get("status") != "pass":
        raise YError("V receipt must pass")
    verify_self_hash(v, "V receipt")
    if (
        v.get("study_id") != policy["study_id"]
        or v.get("layer_id") != policy["far_layer_id"]
        or v.get("semantics_class") != "rms_error"
        or v.get("l_config_sha256") != l_hash
    ):
        raise YError("V lineage/semantics mismatch")

    if w.get("schema_version") != W.OUT_SCHEMA or w.get("status") != "pass":
        raise YError("W receipt must pass")
    verify_self_hash(w, "W receipt")
    if (
        w.get("study_id") != policy["study_id"]
        or w.get("layer_id") != policy["far_layer_id"]
        or w.get("v_receipt_sha256") != sha256_file(v_path)
        or w.get("l_config_sha256") != l_hash
    ):
        raise YError("W lineage mismatch")
    theorem = w.get("theorem")
    if not isinstance(theorem, dict):
        raise YError("W theorem missing")
    alpha = theorem.get("familywise_exceedance_budget_alpha")
    multiplier = theorem.get("derived_multiplier_k")
    admitted = theorem.get("admitted_pixel_count")
    if (
        not finite(alpha)
        or not 0 < float(alpha) < 1
        or not finite(multiplier)
        or float(multiplier) <= 0
        or not isinstance(admitted, int)
        or admitted <= 0
    ):
        raise YError("W theorem parameters invalid")
    if theorem.get("distribution_assumption") != "none_beyond_rms_second_moment_model" or theorem.get("independence_assumption") != "none":
        raise YError("W stochastic semantics drift")

    if x.get("schema_version") != X_SCHEMA or x.get("status") != "pass":
        raise YError("X receipt must pass")
    verify_self_hash(x, "X receipt")
    if (
        x.get("study_id") != policy["study_id"]
        or x.get("layer_id") != policy["far_layer_id"]
        or x.get("w_receipt_sha256") != sha256_file(w_path)
        or x.get("l_config_sha256") != l_hash
        or x.get("policy_sha256") != sha256_file(x_policy_path)
        or x.get("source_hashes") != w.get("source_hashes")
        or x.get("admitted_pixel_count") != admitted
        or abs(float(x.get("whole_sky_alpha", math.nan)) - float(alpha)) > 1e-15
    ):
        raise YError("X/W/L lineage mismatch")
    if x.get("risk_allocation") != "proportional_to_admitted_pixel_count_by_azimuth_bin":
        raise YError("unexpected X risk allocation")
    solver = x.get("solver")
    if not isinstance(solver, dict):
        raise YError("X solver metadata missing")
    iterations, tolerance = solver.get("bisection_iterations"), solver.get("risk_tolerance_abs")
    if not isinstance(iterations, int) or iterations < 32 or not finite(tolerance) or float(tolerance) <= 0:
        raise YError("X solver metadata invalid")
    bins = x.get("bins")
    if not isinstance(bins, list) or not bins:
        raise YError("X bins missing")
    mapped: dict[int, dict[str, Any]] = {}
    for item in bins:
        if not isinstance(item, dict):
            raise YError("X bin invalid")
        index, count, budget = item.get("bin_index"), item.get("admitted_pixel_count"), item.get("alpha_bin")
        digest = item.get("input_population_digest_sha256")
        if (
            not isinstance(index, int)
            or index in mapped
            or not isinstance(count, int)
            or count <= 0
            or not finite(budget)
            or float(budget) <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise YError("X bin metadata invalid")
        mapped[index] = item
    if sum(item["admitted_pixel_count"] for item in mapped.values()) != admitted:
        raise YError("X bin counts do not reproduce W population")
    if abs(math.fsum(float(item["alpha_bin"]) for item in mapped.values()) - float(alpha)) > float(tolerance):
        raise YError("X bin budgets do not reproduce W alpha")
    return {
        "alpha": float(alpha),
        "uniform_k": float(multiplier),
        "admitted_count": admitted,
        "iterations": iterations,
        "risk_tolerance": float(tolerance),
        "x_bins": mapped,
        "source_hashes": w["source_hashes"],
    }


def validate_r(r: dict[str, Any], policy: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if r.get("schema_version") != R_SCHEMA:
        raise YError("R receipt schema mismatch")
    verify_self_hash(r, "R receipt")
    if r.get("study_id") != policy["study_id"]:
        raise YError("R study mismatch")
    index: dict[str, dict[str, Any]] = {}
    for layer in r.get("layers", []):
        if not isinstance(layer, dict) or not isinstance(layer.get("layer_id"), str):
            raise YError("R layer invalid")
        layer_id, support = layer["layer_id"], layer.get("support_class")
        if layer_id in index or support not in {
            "continuous_hard_bound",
            "empirical_multiscale_bound",
            "resolution_qualified",
            "sample_points_only",
            "unknown",
        }:
            raise YError("R layer duplicate/support invalid")
        margin = layer.get("observed_positive_excursion_margin_deg", 0.0)
        if not finite(margin) or float(margin) < 0:
            raise YError("R spatial margin invalid")
        index[layer_id] = {
            "support_class": support,
            "margin_deg": float(margin),
            "nested_audit_sha256": layer.get("nested_audit_sha256"),
        }
    if not {policy["near_layer_id"], policy["far_layer_id"]} <= set(index):
        raise YError("R does not classify both Y layers")
    return index


def validate_l(
    l: dict[str, Any], policy: dict[str, Any], q_info: dict[str, Any], far_info: dict[str, Any]
) -> dict[str, Any]:
    if l.get("schema_version") != W.L_SCHEMA or l.get("study_id") != policy["study_id"]:
        raise YError("L schema/study mismatch")
    if l.get("site_ref") != q_info["site_ref"] or l.get("native_frame") != q_info["native_frame"]:
        raise YError("Q/L site/frame mismatch")
    projection = l.get("projection_contract")
    if not isinstance(projection, dict) or projection.get("projection") != "south_polar_stereographic":
        raise YError("L projection invalid")
    radius = projection.get("reference_radius_m")
    site, width = l.get("site"), l.get("azimuth_bin_width_deg")
    if not finite(radius) or float(radius) <= 0 or not isinstance(site, dict):
        raise YError("L radius/site invalid")
    if not all(finite(site.get(k)) for k in ("x_m", "y_m", "elevation_m")):
        raise YError("L site state invalid")
    if not finite(width) or abs(float(width) - q_info["bin_width"]) > 1e-12:
        raise YError("Q/L bin width mismatch")
    count = int(round(360.0 / float(width)))
    if count != q_info["bin_count"] or set(far_info["x_bins"]) != set(range(count)):
        raise YError("Q/L/X bin count mismatch")
    matches = [
        layer for layer in l.get("layers", [])
        if isinstance(layer, dict) and layer.get("layer_id") == policy["far_layer_id"]
    ]
    if len(matches) != 1:
        raise YError("expected exactly one far L layer")
    layer = matches[0]
    expected_sources = {
        "elevation": layer.get("elevation_source", {}).get("sha256"),
        "rms_uncertainty": layer.get("uncertainty_source", {}).get("sha256"),
        "effective_resolution": layer.get("effective_resolution_source", {}).get("sha256"),
    }
    if expected_sources != far_info["source_hashes"]:
        raise YError("L/W/X source mismatch")
    return {
        "radius": float(radius),
        "site_x": float(site["x_m"]),
        "site_y": float(site["y_m"]),
        "nominal_site_elevation": float(site["elevation_m"]),
        "pole": W.vec3(l.get("pole_vector"), "pole"),
        "bin_width": float(width),
        "bin_count": count,
        "layer": layer,
        "frame_contract_id": l.get("frame_contract_id"),
        "epoch_contract_id": l.get("epoch_contract_id"),
        "site_ref": l.get("site_ref"),
        "frame": l.get("native_frame"),
    }


def nearest_rank(values: list[float], q: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        raise YError("cannot summarize empty values")
    return ordered[max(1, int(math.ceil(q * len(ordered)))) - 1]


def solve_members_for_bin(
    records,
    observer_radii,
    alpha_bin: float,
    uniform_k: float,
    iterations: int,
    risk_tolerance: float,
    member_batch_size: int,
    pixel_chunk_size: int,
    np,
):
    if len(records) <= 0:
        raise YError("cannot solve empty far bin")
    c_all = records["c"]
    q_all = np.sqrt(np.maximum(0.0, 1.0 - c_all * c_all))
    if np.any(q_all <= 1e-15):
        raise YError("far population has radial observer coincidence")
    r_all, s_all = records["radius_m"], records["rms_m"]
    member_count = len(observer_radii)
    horizons = np.empty(member_count, dtype="float64")
    risks = np.empty(member_count, dtype="float64")
    nominal_maxes = np.empty(member_count, dtype="float64")

    def risk(rs_batch, h_batch):
        totals = np.zeros(len(rs_batch), dtype="float64")
        tan_h = np.tan(np.radians(h_batch))
        for start in range(0, len(records), pixel_chunk_size):
            stop = min(len(records), start + pixel_chunk_size)
            c, qv = c_all[start:stop][None, :], q_all[start:stop][None, :]
            radius, rms = r_all[start:stop][None, :], s_all[start:stop][None, :]
            denominator = c - qv * tan_h[:, None]
            possible = denominator > 0.0
            delta = np.full(denominator.shape, np.inf, dtype="float64")
            np.divide(rs_batch[:, None], denominator, out=delta, where=possible)
            delta -= radius
            contribution = np.zeros_like(delta)
            nonzero = rms > 0.0
            at_or_below = possible & (delta <= 0.0) & nonzero
            contribution[at_or_below] = 1.0
            positive = possible & (delta > 0.0) & nonzero
            ratio = np.zeros_like(delta)
            np.divide(rms, delta, out=ratio, where=positive)
            contribution[positive] = np.minimum(1.0, ratio[positive] ** 2)
            totals += np.sum(contribution, axis=1, dtype="float64")
        return totals

    for start_member in range(0, member_count, member_batch_size):
        stop_member = min(member_count, start_member + member_batch_size)
        rs = np.asarray(observer_radii[start_member:stop_member], dtype="float64")
        nominal = np.full(len(rs), -np.inf, dtype="float64")
        feasible = np.full(len(rs), -np.inf, dtype="float64")
        for start in range(0, len(records), pixel_chunk_size):
            stop = min(len(records), start + pixel_chunk_size)
            c, qv = c_all[start:stop][None, :], q_all[start:stop][None, :]
            radius, rms = r_all[start:stop][None, :], s_all[start:stop][None, :]
            nominal_h = np.degrees(np.arctan2(radius * c - rs[:, None], radius * qv))
            upper_radius = radius + uniform_k * rms
            upper_h = np.degrees(np.arctan2(upper_radius * c - rs[:, None], upper_radius * qv))
            nominal = np.maximum(nominal, np.max(nominal_h, axis=1))
            feasible = np.maximum(feasible, np.max(upper_h, axis=1))
        if np.any(risk(rs, feasible) > alpha_bin + risk_tolerance):
            raise YError("uniform W-like member envelope is not feasible")
        lo_risk = risk(rs, nominal)
        done = lo_risk <= alpha_bin + risk_tolerance
        lo, hi = nominal.copy(), feasible.copy()
        hi[done] = nominal[done]
        for _ in range(iterations):
            if np.all(done):
                break
            mid = (lo + hi) / 2.0
            mid_risk = risk(rs, mid)
            feasible_mask = mid_risk <= alpha_bin
            active = ~done
            hi = np.where(active & feasible_mask, mid, hi)
            lo = np.where(active & ~feasible_mask, mid, lo)
        final_risk = risk(rs, hi)
        if np.any(final_risk > alpha_bin + risk_tolerance):
            raise YError("member far solver exceeded bin risk budget")
        horizons[start_member:stop_member] = hi
        risks[start_member:stop_member] = final_risk
        nominal_maxes[start_member:stop_member] = nominal
    return horizons, risks, nominal_maxes


def materialize(
    policy_path: pathlib.Path,
    q_path: pathlib.Path,
    v_path: pathlib.Path,
    w_path: pathlib.Path,
    x_policy_path: pathlib.Path,
    x_path: pathlib.Path,
    r_path: pathlib.Path,
    l_path: pathlib.Path,
    artifact_root: pathlib.Path,
) -> dict[str, Any]:
    np, rasterio, warp_transform = W.require_rasterio()
    policy = validate_policy(read_obj(policy_path, "Y policy"))
    q, v = read_obj(q_path, "Q receipt"), read_obj(v_path, "V receipt")
    w, x = read_obj(w_path, "W receipt"), read_obj(x_path, "X receipt")
    r, l = read_obj(r_path, "R receipt"), read_obj(l_path, "L config")
    q_info = validate_q(q, policy, l_path)
    far_info = validate_v_w_x(v, w, x, v_path, w_path, x_policy_path, l_path, policy)
    r_layers = validate_r(r, policy)
    l_info = validate_l(l, policy, q_info, far_info)
    layer = l_info["layer"]
    try:
        elevation_path = W.resolve_source(artifact_root, layer["elevation_source"], "far elevation")
        rms_path = W.resolve_source(artifact_root, layer["uncertainty_source"], "far RMS")
        effective_path = W.resolve_source(artifact_root, layer["effective_resolution_source"], "far effective resolution")
    except W.WError as exc:
        raise YError(str(exc)) from exc

    radius, site_x, site_y = l_info["radius"], l_info["site_x"], l_info["site_y"]
    minimum, maximum = float(layer["min_range_m"]), float(layer["max_range_m"])
    max_eff = layer.get("max_effective_resolution_m")
    max_eff = None if max_eff is None else float(max_eff)
    record_dtype = np.dtype(RECORD_DTYPE_FIELDS)

    with rasterio.open(elevation_path) as elevation, rasterio.open(rms_path) as rms_ds, rasterio.open(effective_path) as effective:
        try:
            W.aligned(elevation, rms_ds, "far elevation/RMS")
            W.aligned(elevation, effective, "far elevation/effective")
            W.dataset_metadata(elevation, layer, "far elevation")
            W.dataset_metadata(rms_ds, layer, "far RMS")
            W.dataset_metadata(effective, layer, "far effective")
        except W.WError as exc:
            raise YError(str(exc)) from exc
        geographic = rasterio.crs.CRS.from_proj4(f"+proj=longlat +R={radius:.12f} +no_defs +type=crs")
        site_lon, site_lat = warp_transform(elevation.crs, geographic, [site_x], [site_y])
        site_unit = W.lonlat_to_unit(float(site_lon[0]), float(site_lat[0]))
        nominal_site_position = W.scale(site_unit, radius + l_info["nominal_site_elevation"])
        basis = W.local_basis(nominal_site_position, l_info["pole"])
        observer_radii = np.asarray(
            [radius + member["site_elevation_m"] for member in q_info["members"]], dtype="float64"
        )
        if np.any(~np.isfinite(observer_radii)) or np.any(observer_radii <= 0):
            raise YError("Q member observer radius invalid")

        with tempfile.TemporaryDirectory(prefix="ll009y-") as scratch:
            root = pathlib.Path(scratch)
            paths = [root / f"bin-{i:04d}.bin" for i in range(l_info["bin_count"])]
            handles = [path.open("wb") for path in paths]
            counts = [0] * l_info["bin_count"]
            try:
                bbox = (site_x - maximum, site_y - maximum, site_x + maximum, site_y + maximum)
                for _, window in elevation.block_windows(1):
                    left, bottom, right, top = rasterio.windows.bounds(window, elevation.transform)
                    if right < bbox[0] or left > bbox[2] or top < bbox[1] or bottom > bbox[3]:
                        continue
                    ea = elevation.read(1, window=window, masked=True)
                    ua = rms_ds.read(1, window=window, masked=True)
                    ra = effective.read(1, window=window, masked=True)
                    rr, cc = np.indices(ea.shape)
                    rows, cols = rr + int(window.row_off), cc + int(window.col_off)
                    xs = elevation.transform.c + (cols + 0.5) * elevation.transform.a + (rows + 0.5) * elevation.transform.b
                    ys = elevation.transform.f + (cols + 0.5) * elevation.transform.d + (rows + 0.5) * elevation.transform.e
                    ranges = np.hypot(xs - site_x, ys - site_y)
                    annulus = (ranges >= minimum - 1e-9) & (ranges <= maximum + 1e-9)
                    masks = np.ma.getmaskarray(ea) | np.ma.getmaskarray(ua) | np.ma.getmaskarray(ra)
                    if layer.get("nodata_policy", "fail_required_coverage") == "fail_required_coverage" and np.any(annulus & masks):
                        raise YError("nodata inside required far annulus")
                    valid = annulus & ~masks
                    if max_eff is not None:
                        eff_data = np.asarray(ra.data, dtype="float64")
                        if np.any(valid & (~np.isfinite(eff_data) | (eff_data <= 0))):
                            raise YError("invalid effective resolution")
                        valid &= eff_data <= max_eff
                    if not np.any(valid):
                        continue
                    z, s = np.asarray(ea.data[valid], dtype="float64"), np.asarray(ua.data[valid], dtype="float64")
                    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(s)) or np.any(s < 0):
                        raise YError("invalid far elevation/RMS")
                    sx, sy = xs[valid].astype("float64"), ys[valid].astype("float64")
                    lon, lat = warp_transform(elevation.crs, geographic, sx.tolist(), sy.tolist())
                    lonr, latr = np.radians(np.asarray(lon)), np.radians(np.asarray(lat))
                    clat = np.cos(latr)
                    units = np.stack((clat * np.cos(lonr), clat * np.sin(lonr), np.sin(latr)), axis=1)
                    c = np.clip(units @ np.asarray(site_unit, dtype="float64"), -1.0, 1.0)
                    qv = np.sqrt(np.maximum(0.0, 1.0 - c * c))
                    if np.any(qv <= 1e-15):
                        raise YError("far support point radially coincides with observer")
                    terrain_radius = radius + z
                    nominal_el = np.degrees(
                        np.arctan2(
                            terrain_radius * c - (radius + l_info["nominal_site_elevation"]),
                            terrain_radius * qv,
                        )
                    )
                    north = units @ np.asarray(basis[0], dtype="float64")
                    east = units @ np.asarray(basis[1], dtype="float64")
                    azimuth = np.degrees(np.arctan2(east, north)) % 360.0
                    bin_index = np.floor((azimuth + 1e-9) / l_info["bin_width"]).astype("int64") % l_info["bin_count"]
                    vr, vc = rows[valid].astype("int64"), cols[valid].astype("int64")
                    for bin_id in np.unique(bin_index):
                        mask = bin_index == bin_id
                        rec = np.empty(int(np.count_nonzero(mask)), dtype=record_dtype)
                        rec["c"], rec["radius_m"], rec["rms_m"] = c[mask], terrain_radius[mask], s[mask]
                        rec["row"], rec["col"], rec["nominal_elevation_deg"] = vr[mask], vc[mask], nominal_el[mask]
                        handles[int(bin_id)].write(rec.tobytes(order="C"))
                        counts[int(bin_id)] += len(rec)
            finally:
                for handle in handles:
                    handle.close()

            if sum(counts) != far_info["admitted_count"]:
                raise YError("Y/W/X admitted population mismatch")
            for bin_id, (path, count) in enumerate(zip(paths, counts)):
                x_bin = far_info["x_bins"].get(bin_id)
                if x_bin is None or count != x_bin["admitted_pixel_count"]:
                    raise YError("Y/X per-bin count mismatch")
                if sha256_file(path) != x_bin["input_population_digest_sha256"]:
                    raise YError("Y did not reconstruct exact X bin population")

            member_count = q_info["member_count"]
            far_horizons = np.empty((member_count, l_info["bin_count"]), dtype="float64")
            far_risks, far_nominals = np.empty_like(far_horizons), np.empty_like(far_horizons)
            for bin_id, path in enumerate(paths):
                records = np.memmap(path, dtype=record_dtype, mode="r", shape=(counts[bin_id],))
                solved, risks, nominals = solve_members_for_bin(
                    records,
                    observer_radii,
                    float(far_info["x_bins"][bin_id]["alpha_bin"]),
                    far_info["uniform_k"],
                    far_info["iterations"],
                    far_info["risk_tolerance"],
                    int(policy["member_batch_size"]),
                    int(policy["pixel_chunk_size"]),
                    np,
                )
                far_horizons[:, bin_id], far_risks[:, bin_id], far_nominals[:, bin_id] = solved, risks, nominals
                del records

            near_margin = r_layers[policy["near_layer_id"]]["margin_deg"]
            far_margin = r_layers[policy["far_layer_id"]]["margin_deg"]
            near_horizons = np.asarray([member["near_horizon_deg"] for member in q_info["members"]], dtype="float64")
            full_horizons = np.maximum(near_horizons + near_margin, far_horizons + far_margin)
            near_winner = (near_horizons + near_margin) >= (far_horizons + far_margin)
            if policy["require_all_members_all_bins"] and (
                np.any(~np.isfinite(full_horizons)) or np.any(~np.isfinite(far_horizons))
            ):
                raise YError("non-finite member/bin horizon")

            quantile = float(policy["quantile"])
            summaries, k_bins = [], []
            for bin_id in range(l_info["bin_count"]):
                values = full_horizons[:, bin_id]
                ordered = np.sort(values, kind="mergesort")
                rank = max(1, int(math.ceil(quantile * member_count)))
                selected = float(ordered[rank - 1])
                near_wins = int(np.count_nonzero(near_winner[:, bin_id]))
                summaries.append(
                    {
                        "bin_index": bin_id,
                        "member_count": member_count,
                        "min_deg": float(ordered[0]),
                        "median_deg": nearest_rank(values.tolist(), 0.5),
                        "selected_quantile": quantile,
                        "selected_quantile_deg": selected,
                        "finite_observed_max_deg": float(ordered[-1]),
                        "max_member_ordinal": int(np.argmax(values)),
                        "near_winner_member_count": near_wins,
                        "far_winner_member_count": member_count - near_wins,
                        "maximum_far_markov_bound": float(np.max(far_risks[:, bin_id])),
                        "alpha_bin": float(far_info["x_bins"][bin_id]["alpha_bin"]),
                    }
                )
                k_bins.append(
                    {
                        "bin_index": bin_id,
                        "azimuth_start_deg": bin_id * l_info["bin_width"],
                        "azimuth_end_deg": (bin_id + 1) * l_info["bin_width"],
                        "conservative_elevation_deg": selected,
                        "numeric_semantics": "finite Q empirical scenario quantile of memberwise near maximum with scenario-parameterized distribution-free far RMS envelope",
                    }
                )

            members_out, solver_digest = [], hashlib.sha256()
            for member_index, member in enumerate(q_info["members"]):
                record = {
                    "ordinal": member["ordinal"],
                    "source_id": member["source_id"],
                    "source_sha256": member["sha256"],
                    "site_elevation_m": member["site_elevation_m"],
                    "observer_radius_m": float(observer_radii[member_index]),
                    "near_horizon_deg": near_horizons[member_index].tolist(),
                    "far_scenario_horizon_deg": far_horizons[member_index].tolist(),
                    "far_nominal_horizon_deg": far_nominals[member_index].tolist(),
                    "far_markov_bound": far_risks[member_index].tolist(),
                    "full_horizon_deg": full_horizons[member_index].tolist(),
                    "winner_by_bin": ["near" if bool(v) else "far" for v in near_winner[member_index]],
                }
                solver_digest.update(canonical_bytes(record))
                members_out.append(record)

    binding = {
        "status": "bound",
        "mode": "empirical_q_of_scenario_parameterized_distribution_free_far_envelopes",
        "quantile_estimator": policy["quantile_estimator"],
        "quantile": float(policy["quantile"]),
        "q_receipt_sha256": sha256_file(q_path),
        "v_receipt_sha256": sha256_file(v_path),
        "w_receipt_sha256": sha256_file(w_path),
        "x_policy_sha256": sha256_file(x_policy_path),
        "x_receipt_sha256": sha256_file(x_path),
        "r_receipt_sha256": sha256_file(r_path),
        "l_config_sha256": sha256_file(l_path),
        "whole_sky_far_exceedance_budget_alpha": far_info["alpha"],
        "q_member_count": q_info["member_count"],
        "memberwise_composition": policy["full_horizon_composition"],
        "far_solver_semantics": "For each exact Q observer scenario, the same marginal Product 90 RMS model is geometrically re-evaluated under exact X/W risk budgets; no Q×far joint probability coupling is asserted.",
        "memberwise_solver_digest_sha256": solver_digest.hexdigest(),
    }
    output = {
        "schema_version": K_SCHEMA,
        "producer_schema_version": PRODUCER_SCHEMA,
        "status": "pass",
        "study_id": policy["study_id"],
        "frame_contract_id": l_info["frame_contract_id"],
        "epoch_contract_id": l_info["epoch_contract_id"],
        "site_ref": l_info["site_ref"],
        "frame": l_info["frame"],
        "input_sha256": sha256_file(policy_path),
        "source_hashes": far_info["source_hashes"],
        "site_position_m": list(nominal_site_position),
        "site_vertical_uncertainty_m": 0.0,
        "basis": {
            "north": list(basis[0]),
            "east": list(basis[1]),
            "up": list(basis[2]),
            "fallback_used": basis[3],
        },
        "azimuth_bin_width_deg": l_info["bin_width"],
        "bin_count": l_info["bin_count"],
        "layers": [
            {
                "layer_id": policy["near_layer_id"],
                "uncertainty_semantics": "empirical_ensemble",
                "spatial_support_class": r_layers[policy["near_layer_id"]]["support_class"],
                "applied_spatial_margin_deg": near_margin,
            },
            {
                "layer_id": policy["far_layer_id"],
                "uncertainty_semantics": "distribution_free_familywise_rms_scenario_parameterized",
                "spatial_support_class": r_layers[policy["far_layer_id"]]["support_class"],
                "applied_spatial_margin_deg": far_margin,
            },
        ],
        "bins": k_bins,
        "statistical_horizon_binding": binding,
        "memberwise_hybrid_evidence": {
            "semantics_class": "finite_empirical_q_of_scenario_parameterized_distribution_free_far_envelopes",
            "q_semantics": "finite_empirical_published_clone_ensemble",
            "far_semantics": "marginal_rms_model_re_evaluated_at_fixed_q_observer_scenarios_no_joint_probability_claim",
            "near_spatial_support": r_layers[policy["near_layer_id"]],
            "far_spatial_support": r_layers[policy["far_layer_id"]],
            "per_bin_summary": summaries,
            "members": members_out,
        },
        "semantics": "Every Q member retains its own near terrain and observer elevation; the marginal far RMS envelope is re-solved geometrically for that same observer scenario; near/far are max-composed memberwise before the declared finite-ensemble quantile.",
        "non_claims": [
            "The selected Q quantile is a finite empirical ensemble statistic, not a population confidence guarantee.",
            "The far alpha theorem is a marginal Product 90 RMS statement evaluated at each fixed Q observer scenario; Y does not establish a conditional far-error distribution given a Q clone state.",
            "No independence, correlation model, or other joint Q×Product90 probability coupling is assumed, and no Q empirical quantile is multiplied with far 1-alpha into a confidence probability.",
            "Spatial-support semantics remain exactly those of the bound LL-009R receipt; unresolved physical terrain may still block stronger claims.",
            "This pack is terrain-horizon evidence, not site, visibility, operations, or architecture authority.",
        ],
    }
    output["receipt_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def write_immutable(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise YError(f"refusing to overwrite differing output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test_math() -> None:
    a = [10.0, 0.0, 0.0, 0.0]
    b = [0.0, 10.0, 0.0, 0.0]
    combined = [max(x, y) for x, y in zip(a, b)]
    assert nearest_rank(a, 0.75) == 0.0
    assert nearest_rank(b, 0.75) == 0.0
    assert nearest_rank(combined, 0.75) == 10.0
    print("LL-009Y ordering theorem self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="LL-009Y memberwise Q + geometry-aware far RMS horizon")
    parser.add_argument("--policy")
    parser.add_argument("--q-receipt")
    parser.add_argument("--v-receipt")
    parser.add_argument("--w-receipt")
    parser.add_argument("--x-policy")
    parser.add_argument("--x-receipt")
    parser.add_argument("--r-receipt")
    parser.add_argument("--l-config")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test_math()
            return 0
        required = (
            args.policy,
            args.q_receipt,
            args.v_receipt,
            args.w_receipt,
            args.x_policy,
            args.x_receipt,
            args.r_receipt,
            args.l_config,
            args.output,
        )
        if not all(required):
            raise YError("--policy --q-receipt --v-receipt --w-receipt --x-policy --x-receipt --r-receipt --l-config --output required")
        output = materialize(
            pathlib.Path(args.policy),
            pathlib.Path(args.q_receipt),
            pathlib.Path(args.v_receipt),
            pathlib.Path(args.w_receipt),
            pathlib.Path(args.x_policy),
            pathlib.Path(args.x_receipt),
            pathlib.Path(args.r_receipt),
            pathlib.Path(args.l_config),
            pathlib.Path(args.artifact_root),
        )
        write_immutable(pathlib.Path(args.output), output)
        print(json.dumps(output, sort_keys=True, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, YError, W.WError) as exc:
        raise SystemExit(f"LL-009Y failure: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
