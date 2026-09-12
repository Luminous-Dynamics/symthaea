#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import math
import pathlib
import tempfile
from typing import Any

L = importlib.import_module("materialize_ll009l_lola_cog")

RECEIPT_SCHEMA = "ll009m.radial-uncertainty-receipt.v1"
POLICY = "ll009m.uniform-layer-radial-bound.v1"


class MError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return L.canonical_bytes(value)


def write_immutable(path: pathlib.Path, value: dict):
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise MError(f"refusing to overwrite differing immutable output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def safe_json(path: pathlib.Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MError(f"{path} must contain object")
    return value


def verify_l_pack_and_k_input(l_pack_path: pathlib.Path, k_input_path: pathlib.Path):
    l_pack = safe_json(l_pack_path)
    k_input = safe_json(k_input_path)
    if l_pack.get("schema_version") != L.OUT:
        raise MError("unexpected LL-009L pack schema")
    if k_input.get("schema_version") != L.K_SCHEMA:
        raise MError("unexpected LL-009K input schema")
    for key in ("study_id", "frame_contract_id", "epoch_contract_id", "site_ref"):
        if l_pack.get(key) != k_input.get(key):
            raise MError(f"L/K lineage mismatch: {key}")
    if k_input.get("frame") != l_pack.get("native_frame"):
        raise MError("L/K frame mismatch")
    sources = k_input.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise MError("M expects K input to bind exactly one normalized L pack source")
    source = sources[0]
    if source.get("sha256") != L.sha256_file(l_pack_path):
        raise MError("K input does not bind exact L pack bytes")
    if source.get("source_id") != "ll009l-pack":
        raise MError("unexpected K source_id")
    return l_pack, k_input


def bound_for_pixel(site_position, terrain_position, terrain_uncertainty_m, site_uncertainty_m, basis):
    site_up = basis[2]
    terrain_up = L.unit(terrain_position, "terrain_radial")
    low_site = L.sub(site_position, L.scale(site_up, float(site_uncertainty_m)))
    approximate_high = L.add(
        terrain_position, L.scale(site_up, float(terrain_uncertainty_m))
    )
    radial_high = L.add(
        terrain_position, L.scale(terrain_up, float(terrain_uncertainty_m))
    )
    approximate_los = L.sub(approximate_high, low_site)
    radial_los = L.sub(radial_high, low_site)
    approximate_range = L.norm(approximate_los)
    direction_difference_m = float(terrain_uncertainty_m) * L.norm(
        L.sub(terrain_up, site_up)
    )
    if direction_difference_m >= approximate_range:
        raise MError(
            "terrain-radial uncertainty perturbation is too large for finite conservative bound"
        )
    bound_deg = math.degrees(
        math.atan2(direction_difference_m, approximate_range - direction_difference_m)
    )
    approximate_elevation_deg = L.az_el(approximate_los, basis)[1]
    radial_elevation_deg = L.az_el(radial_los, basis)[1]
    exact_delta_deg = radial_elevation_deg - approximate_elevation_deg
    if exact_delta_deg > bound_deg + 1e-10:
        raise MError("exact terrain-radial elevation exceeds analytic bound")
    central_dot = max(-1.0, min(1.0, L.dot(terrain_up, site_up)))
    return {
        "bound_deg": bound_deg,
        "exact_delta_deg": exact_delta_deg,
        "direction_difference_m": direction_difference_m,
        "central_angle_deg": math.degrees(math.acos(central_dot)),
        "approximate_elevation_deg": approximate_elevation_deg,
        "radial_elevation_deg": radial_elevation_deg,
    }


def scan_bounds(config_path: pathlib.Path, artifact_root: pathlib.Path, l_pack: dict):
    rasterio, warp_transform = L.require_rasterio()
    import numpy as np

    config = safe_json(config_path)
    radius, pole, bin_width, bin_count, _top_k = L.validate_config(config)
    geographic_crs = rasterio.crs.CRS.from_proj4(
        f"+proj=longlat +R={radius:.12f} +no_defs +type=crs"
    )
    site_cfg = config["site"]
    site_xy = (float(site_cfg["x_m"]), float(site_cfg["y_m"]))
    site_elevation = float(site_cfg["elevation_m"])
    site_uncertainty = float(site_cfg["vertical_uncertainty_m"])
    expected_site = tuple(float(x) for x in l_pack["site_position_m"])

    results = []
    common_wkt = None
    computed_site = None
    basis = None

    for layer in config["layers"]:
        layer_id = layer["layer_id"]
        minimum = float(layer["min_range_m"])
        maximum = float(layer["max_range_m"])
        max_effective_resolution = layer.get("max_effective_resolution_m")
        declared_margin = float(layer.get("additional_angular_margin_deg", 0.0))

        elevation_path = L.resolve_hashed(
            artifact_root, layer.get("elevation_source", {}), f"{layer_id}:elevation"
        )
        uncertainty_path = L.resolve_hashed(
            artifact_root,
            layer.get("uncertainty_source", {}),
            f"{layer_id}:uncertainty",
        )
        effective_entry = layer.get("effective_resolution_source")
        effective_path = (
            L.resolve_hashed(
                artifact_root, effective_entry, f"{layer_id}:effective_resolution"
            )
            if isinstance(effective_entry, dict)
            else None
        )

        with rasterio.open(elevation_path) as elevation, rasterio.open(
            uncertainty_path
        ) as uncertainty:
            effective = rasterio.open(effective_path) if effective_path else None
            try:
                L.aligned(
                    elevation, uncertainty, f"{layer_id}:elevation/uncertainty"
                )
                if effective:
                    L.aligned(
                        elevation,
                        effective,
                        f"{layer_id}:elevation/effective_resolution",
                    )
                L.dataset_contract(elevation, layer, "elevation")
                L.dataset_contract(uncertainty, layer, "uncertainty")
                if effective:
                    L.dataset_contract(effective, layer, "effective_resolution")

                wkt = elevation.crs.to_wkt()
                if common_wkt is None:
                    common_wkt = wkt
                    longitudes, latitudes = warp_transform(
                        elevation.crs,
                        geographic_crs,
                        [site_xy[0]],
                        [site_xy[1]],
                    )
                    computed_site = L.lonlat_to_xyz(
                        longitudes[0], latitudes[0], radius + site_elevation
                    )
                    if L.norm(L.sub(computed_site, expected_site)) > 1e-6:
                        raise MError("L pack site position does not match exact config/CRS")
                    basis = L.local_basis(computed_site, pole)
                elif wkt != common_wkt:
                    raise MError(f"{layer_id}: CRS differs from other layers")

                max_bound = -1.0
                max_exact_delta = -math.inf
                worst = None
                bin_bounds = [0.0 for _ in range(bin_count)]
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
                    left, bottom, right, top = rasterio.windows.bounds(
                        window, elevation.transform
                    )
                    if (
                        right < bounding_box[0]
                        or left > bounding_box[2]
                        or top < bounding_box[1]
                        or bottom > bounding_box[3]
                    ):
                        continue
                    elevation_array = elevation.read(1, window=window, masked=True)
                    uncertainty_array = uncertainty.read(
                        1, window=window, masked=True
                    )
                    effective_array = (
                        effective.read(1, window=window, masked=True)
                        if effective
                        else None
                    )
                    elevation_mask = np.ma.getmaskarray(elevation_array)
                    uncertainty_mask = np.ma.getmaskarray(uncertainty_array)
                    effective_mask = (
                        np.ma.getmaskarray(effective_array)
                        if effective_array is not None
                        else None
                    )
                    row0, col0 = int(window.row_off), int(window.col_off)
                    selected_xy = []
                    selected_records = []

                    for local_row in range(elevation_array.shape[0]):
                        for local_col in range(elevation_array.shape[1]):
                            row = row0 + local_row
                            col = col0 + local_col
                            x, y = elevation.transform * (col + 0.5, row + 0.5)
                            projected_range = math.hypot(
                                x - site_xy[0], y - site_xy[1]
                            )
                            if (
                                projected_range < minimum - 1e-9
                                or projected_range > maximum + 1e-9
                            ):
                                continue
                            scanned_pixels += 1
                            if (
                                bool(elevation_mask[local_row, local_col])
                                or bool(uncertainty_mask[local_row, local_col])
                                or (
                                    effective_mask is not None
                                    and bool(effective_mask[local_row, local_col])
                                )
                            ):
                                nodata_pixels += 1
                                continue
                            elevation_value = float(
                                elevation_array[local_row, local_col]
                            )
                            uncertainty_value = float(
                                uncertainty_array[local_row, local_col]
                            )
                            if (
                                not math.isfinite(elevation_value)
                                or not math.isfinite(uncertainty_value)
                                or uncertainty_value < 0
                            ):
                                raise MError(
                                    f"{layer_id}: invalid raster value at {row},{col}"
                                )
                            effective_value = (
                                float(effective_array[local_row, local_col])
                                if effective_array is not None
                                else None
                            )
                            if effective_value is not None and (
                                not math.isfinite(effective_value)
                                or effective_value <= 0
                            ):
                                raise MError(
                                    f"{layer_id}: invalid effective resolution at {row},{col}"
                                )
                            if max_effective_resolution is not None:
                                if effective_value is None:
                                    raise MError(
                                        f"{layer_id}: effective-resolution evidence required"
                                    )
                                if effective_value > float(max_effective_resolution):
                                    continue
                            selected_xy.append((x, y))
                            selected_records.append(
                                (
                                    row,
                                    col,
                                    elevation_value,
                                    uncertainty_value,
                                    effective_value,
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
                        row, col, elevation_value, uncertainty_value, _ = record
                        position = L.lonlat_to_xyz(
                            longitude, latitude, radius + elevation_value
                        )
                        proof = bound_for_pixel(
                            computed_site,
                            position,
                            uncertainty_value,
                            site_uncertainty,
                            basis,
                        )
                        nominal_azimuth = L.az_el(
                            L.sub(position, computed_site), basis
                        )[0]
                        bin_index = int(
                            math.floor((nominal_azimuth + 1e-9) / bin_width)
                        ) % bin_count
                        bin_bounds[bin_index] = max(
                            bin_bounds[bin_index], proof["bound_deg"]
                        )
                        if proof["bound_deg"] > max_bound:
                            max_bound = proof["bound_deg"]
                            worst = {
                                "row": row,
                                "col": col,
                                "longitude_deg": longitude,
                                "latitude_deg": latitude,
                                **proof,
                            }
                        max_exact_delta = max(
                            max_exact_delta, proof["exact_delta_deg"]
                        )
                        admitted_pixels += 1

                if (
                    layer.get("nodata_policy", "fail_required_coverage")
                    == "fail_required_coverage"
                    and nodata_pixels
                ):
                    raise MError(
                        f"{layer_id}: {nodata_pixels} nodata pixels inside required annulus"
                    )
                if admitted_pixels == 0 or worst is None:
                    raise MError(f"{layer_id}: no admitted pixels in M scan")
                results.append(
                    {
                        "layer_id": layer_id,
                        "declared_angular_margin_deg": declared_margin,
                        "radial_uncertainty_margin_deg": max_bound,
                        "max_exact_radial_minus_site_up_elevation_deg": max_exact_delta,
                        "worst_case": worst,
                        "per_bin_max_bound_deg": bin_bounds,
                        "scanned_annulus_pixels": scanned_pixels,
                        "admitted_pixels": admitted_pixels,
                        "nodata_pixels_in_annulus": nodata_pixels,
                        "elevation_source_sha256": L.sha256_file(elevation_path),
                        "uncertainty_source_sha256": L.sha256_file(uncertainty_path),
                        "effective_resolution_source_sha256": (
                            L.sha256_file(effective_path) if effective_path else None
                        ),
                    }
                )
            finally:
                if effective:
                    effective.close()

    return results


def augment_k_input(k_input: dict, layer_results: list[dict]):
    by_id = {item["layer_id"]: item for item in layer_results}
    augmented = json.loads(json.dumps(k_input))
    seen = set()
    for layer in augmented.get("layers", []):
        layer_id = layer.get("layer_id")
        result = by_id.get(layer_id)
        if result is None:
            raise MError(f"K layer {layer_id} missing M scan")
        original = layer.get("additional_angular_margin_deg")
        if not L.finite(original) or original < 0:
            raise MError(f"K layer {layer_id} invalid original margin")
        if abs(float(original) - result["declared_angular_margin_deg"]) > 1e-12:
            raise MError(f"K/L declared margin drift for {layer_id}")
        layer["ll009l_declared_angular_margin_deg"] = float(original)
        layer["ll009m_radial_uncertainty_margin_deg"] = result[
            "radial_uncertainty_margin_deg"
        ]
        layer["ll009m_policy"] = POLICY
        layer["additional_angular_margin_deg"] = (
            float(original) + result["radial_uncertainty_margin_deg"]
        )
        seen.add(layer_id)
    if seen != set(by_id):
        raise MError("M scan contains layer absent from K input")
    return augmented


def run(
    config_path: pathlib.Path,
    artifact_root: pathlib.Path,
    l_pack_path: pathlib.Path,
    k_input_path: pathlib.Path,
):
    l_pack, k_input = verify_l_pack_and_k_input(l_pack_path, k_input_path)
    config = safe_json(config_path)
    for key in ("study_id", "frame_contract_id", "epoch_contract_id", "site_ref"):
        if config.get(key) != l_pack.get(key):
            raise MError(f"config/L pack lineage mismatch: {key}")
    if config.get("native_frame") != l_pack.get("native_frame"):
        raise MError("config/L pack frame mismatch")
    layer_results = scan_bounds(config_path, artifact_root, l_pack)
    augmented = augment_k_input(k_input, layer_results)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "pass",
        "policy": POLICY,
        "study_id": l_pack["study_id"],
        "frame_contract_id": l_pack["frame_contract_id"],
        "epoch_contract_id": l_pack["epoch_contract_id"],
        "site_ref": l_pack["site_ref"],
        "native_frame": l_pack["native_frame"],
        "l_config_sha256": L.sha256_file(config_path),
        "l_pack_sha256": L.sha256_file(l_pack_path),
        "input_k_sha256": L.sha256_file(k_input_path),
        "augmented_k_sha256": L.sha256_bytes(canonical_bytes(augmented)),
        "layers": layer_results,
        "theorem": (
            "For every admitted terrain pixel i, exact terrain-radial elevation <= "
            "site-up approximate elevation + b_i. With B_layer=max_i(b_i), the "
            "retained site-up layer/bin maximum plus B_layer bounds omitted pixels too."
        ),
        "non_claims": [
            "This receipt closes terrain uncertainty direction geometry only.",
            "It does not establish terrain completeness, source resolution sufficiency, site suitability, or operations qualification.",
        ],
    }
    receipt["receipt_sha256"] = L.sha256_bytes(canonical_bytes(receipt))
    return augmented, receipt


def self_test():
    rasterio, _ = L.require_rasterio()
    import numpy as np
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        crs = CRS.from_proj4(
            "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400 +units=m +no_defs"
        )
        transform = from_origin(-250000, 250000, 100000, 100000)
        elevation = np.zeros((5, 5), dtype="float32")
        elevation[2, 4] = 1000.0
        uncertainty = np.full((5, 5), 100.0, dtype="float32")
        effective = np.full((5, 5), 100000.0, dtype="float32")

        def write(name, array):
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
            ) as dataset:
                dataset.write(array, 1)
            return path

        ep = write("elev.tif", elevation)
        up = write("unc.tif", uncertainty)
        rp = write("eff.tif", effective)
        with rasterio.open(ep) as dataset:
            crs_hash = L.sha256_bytes(dataset.crs.to_wkt().encode())
        config = {
            "schema_version": L.SCHEMA,
            "study_id": "study-m",
            "frame_contract_id": "frame-m",
            "epoch_contract_id": "epoch-m",
            "site_ref": "site-m",
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
                "vertical_uncertainty_m": 10.0,
            },
            "pole_vector": [0, 0, 1],
            "azimuth_bin_width_deg": 90.0,
            "top_k_per_bin": 1,
            "layers": [
                {
                    "layer_id": "wide",
                    "min_range_m": 50000.0,
                    "max_range_m": 400000.0,
                    "expected_pixel_size_m": 100000.0,
                    "expected_crs_wkt_sha256": crs_hash,
                    "max_effective_resolution_m": 100001.0,
                    "additional_angular_margin_deg": 0.02,
                    "nodata_policy": "fail_required_coverage",
                    "elevation_source": {
                        "path": "elev.tif",
                        "sha256": L.sha256_file(ep),
                    },
                    "uncertainty_source": {
                        "path": "unc.tif",
                        "sha256": L.sha256_file(up),
                    },
                    "effective_resolution_source": {
                        "path": "eff.tif",
                        "sha256": L.sha256_file(rp),
                    },
                }
            ],
        }
        config_path = root / "config.json"
        config_path.write_bytes(canonical_bytes(config))
        pack = L.materialize(config_path, root)
        pack_path = root / "pack.json"
        pack_path.write_bytes(canonical_bytes(pack))

        # Construct the baseline LL-009L K wrapper directly: M's self-test must
        # not depend on scratch-only ranking metadata.
        k_layers = []
        for pack_layer in pack["layers"]:
            samples = [
                {
                    "position_m": sample["position_m"],
                    "vertical_uncertainty_m": sample["vertical_uncertainty_m"],
                }
                for sample in pack_layer["samples"]
            ]
            if not samples:
                continue
            k_layers.append(
                {
                    "layer_id": pack_layer["layer_id"],
                    "source_ref": "ll009l-pack",
                    "frame": pack["native_frame"],
                    "min_range_m": pack_layer["min_range_m"],
                    "max_range_m": pack_layer["max_range_m"],
                    "nominal_resolution_m": pack_layer["nominal_resolution_m"],
                    "additional_angular_margin_deg": pack_layer[
                        "additional_angular_margin_deg"
                    ],
                    "samples": samples,
                }
            )
        k_input = {
            "schema_version": L.K_SCHEMA,
            "study_id": pack["study_id"],
            "frame_contract_id": pack["frame_contract_id"],
            "epoch_contract_id": pack["epoch_contract_id"],
            "site_ref": pack["site_ref"],
            "frame": pack["native_frame"],
            "sources": [
                {
                    "source_id": "ll009l-pack",
                    "path": "pack.json",
                    "sha256": L.sha256_file(pack_path),
                }
            ],
            "site_position_m": pack["site_position_m"],
            "site_vertical_uncertainty_m": pack["site_vertical_uncertainty_m"],
            "pole_vector": pack["pole_vector"],
            "azimuth_bin_width_deg": pack["azimuth_bin_width_deg"],
            "layers": k_layers,
        }
        k_path = root / "k-input.json"
        k_path.write_bytes(canonical_bytes(k_input))

        augmented, receipt = run(config_path, root, pack_path, k_path)
        layer = receipt["layers"][0]
        assert layer["radial_uncertainty_margin_deg"] > 0.0
        assert (
            layer["max_exact_radial_minus_site_up_elevation_deg"]
            <= layer["radial_uncertainty_margin_deg"] + 1e-10
        )
        assert (
            augmented["layers"][0]["additional_angular_margin_deg"]
            > k_input["layers"][0]["additional_angular_margin_deg"]
        )
        emitted = len(pack["layers"][0]["samples"])
        assert layer["admitted_pixels"] > emitted, (
            "the theorem scan must cover omitted raster candidates, not only emitted top-K"
        )
        augmented2, receipt2 = run(config_path, root, pack_path, k_path)
        assert canonical_bytes(augmented) == canonical_bytes(augmented2)
        assert canonical_bytes(receipt) == canonical_bytes(receipt2)
        print("LL-009M self-test PASS", rasterio.__version__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l-config")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--l-pack")
    parser.add_argument("--k-input")
    parser.add_argument("--output-k-input")
    parser.add_argument("--receipt")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    required = (
        args.l_config,
        args.l_pack,
        args.k_input,
        args.output_k_input,
        args.receipt,
    )
    if not all(required):
        raise SystemExit(
            "--l-config --l-pack --k-input --output-k-input --receipt required"
        )
    augmented, receipt = run(
        pathlib.Path(args.l_config),
        pathlib.Path(args.artifact_root),
        pathlib.Path(args.l_pack),
        pathlib.Path(args.k_input),
    )
    write_immutable(pathlib.Path(args.output_k_input), augmented)
    write_immutable(pathlib.Path(args.receipt), receipt)
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
