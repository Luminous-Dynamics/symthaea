#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
from typing import Any

import compose_ll009k_horizon_pack as K

POLICY_SCHEMA = "ll009t.statistical-horizon-composition-policy.v1"
Q_POLICY_SCHEMA = "ll009q.ensemble-policy.v1"
Q_RECEIPT_SCHEMA = "ll009q.clone-horizon-ensemble-receipt.v1"
O_SCHEMA = "ll009o.uncertainty-semantics-receipt.v1"
R_SCHEMA = "ll009r.spatial-support-classification-receipt.v1"
M_SCHEMA = "ll009m.radial-uncertainty-receipt.v1"
K_INPUT_SCHEMA = "ll009k.horizon-materialization-input.v1"
K_OUTPUT_SCHEMA = "ll009k.horizon-pack.v1"
STAT_MODES = {"empirical_ensemble_quantile", "finite_ensemble_observed_max"}
SUPPORT_CLASSES = {
    "continuous_hard_bound",
    "empirical_multiscale_bound",
    "resolution_qualified",
    "sample_points_only",
    "unknown",
}


class TError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def read_obj(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise TError(f"{label} must contain object")
    return value


def verify_self_hash(value: dict[str, Any], label: str) -> str:
    observed = value.get("receipt_sha256")
    if not isinstance(observed, str) or len(observed) != 64:
        raise TError(f"{label}: receipt_sha256 required")
    body = json.loads(json.dumps(value))
    body.pop("receipt_sha256", None)
    actual = sha256_bytes(canonical_bytes(body))
    if observed != actual:
        raise TError(f"{label}: receipt self-hash mismatch")
    return observed


def validate_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != POLICY_SCHEMA:
        raise TError(f"policy schema must be {POLICY_SCHEMA}")
    policy = dict(value)
    for key in ("study_id", "ensemble_layer_id", "statistic_mode"):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise TError(f"policy missing {key}")
    if policy["statistic_mode"] not in STAT_MODES:
        raise TError("unsupported statistic_mode")

    companions = policy.get("companion_layer_ids")
    expected = policy.get("expected_layer_ids")
    if (
        not isinstance(companions, list)
        or not companions
        or not all(isinstance(item, str) and item for item in companions)
    ):
        raise TError("companion_layer_ids must be a non-empty string list")
    if len(companions) != len(set(companions)):
        raise TError("companion_layer_ids must be unique")
    if (
        not isinstance(expected, list)
        or not expected
        or not all(isinstance(item, str) and item for item in expected)
    ):
        raise TError("expected_layer_ids must be a non-empty string list")
    if len(expected) != len(set(expected)):
        raise TError("expected_layer_ids must be unique")
    if policy["ensemble_layer_id"] in companions:
        raise TError("ensemble layer cannot also be a companion layer")
    if set(expected) != {policy["ensemble_layer_id"], *companions}:
        raise TError("expected_layer_ids must equal ensemble + companion layers")

    if policy["statistic_mode"] == "empirical_ensemble_quantile":
        quantile = policy.get("quantile")
        if not finite(quantile) or not 0 < float(quantile) <= 1:
            raise TError("quantile must be in (0,1]")
    elif policy.get("quantile") is not None:
        raise TError("quantile must be null/absent for finite ensemble max mode")

    if (
        policy.get("site_observer_compatibility")
        != "require_hard_upper_bound_site_v1"
    ):
        raise TError(
            "V1 requires site_observer_compatibility=require_hard_upper_bound_site_v1"
        )
    if policy.get("companion_vertical_semantics") != "hard_upper_bound":
        raise TError("V1 requires hard_upper_bound companion vertical semantics")
    if (
        policy.get("spatial_support_margin_policy")
        != "apply_observed_positive_excursion_if_present"
    ):
        raise TError("unsupported spatial_support_margin_policy")
    if policy.get("require_all_q_members_per_bin") is not True:
        raise TError("V1 requires require_all_q_members_per_bin=true")
    return policy


def validate_q_policy(
    value: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    if value.get("schema_version") != Q_POLICY_SCHEMA:
        raise TError(f"Q policy schema must be {Q_POLICY_SCHEMA}")
    if value.get("study_id") != policy["study_id"]:
        raise TError("Q/T policy study mismatch")
    if value.get("ensemble_layer_id") != policy["ensemble_layer_id"]:
        raise TError("Q/T ensemble layer mismatch")
    if value.get("quantile_estimator") != "empirical_cdf_nearest_rank":
        raise TError("T V1 requires Q empirical_cdf_nearest_rank estimator")
    return value


def validate_q_receipt(
    receipt: dict[str, Any],
    q_policy_path: pathlib.Path,
    q_policy: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], float, int, float | None]:
    if (
        receipt.get("schema_version") != Q_RECEIPT_SCHEMA
        or receipt.get("status") != "pass"
    ):
        raise TError("Q receipt must be a passing clone-horizon ensemble receipt")
    verify_self_hash(receipt, "Q receipt")
    if receipt.get("semantics_class") != "empirical_ensemble":
        raise TError("Q receipt must retain empirical_ensemble semantics")
    if receipt.get("study_id") != policy["study_id"]:
        raise TError("Q/T study mismatch")
    if receipt.get("policy_sha256") != sha256_file(q_policy_path):
        raise TError("Q receipt does not bind exact Q policy")
    if receipt.get("quantile_estimator") != q_policy.get("quantile_estimator"):
        raise TError("Q receipt/policy quantile estimator mismatch")

    width = receipt.get("azimuth_bin_width_deg")
    count = receipt.get("azimuth_bin_count")
    member_count = receipt.get("member_count")
    if (
        not finite(width)
        or float(width) <= 0
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 4
        or abs(float(width) * count - 360.0) > 1e-9
    ):
        raise TError("Q receipt bin geometry invalid")
    if (
        not isinstance(member_count, int)
        or isinstance(member_count, bool)
        or member_count < 2
    ):
        raise TError("Q receipt member_count invalid")

    summaries = receipt.get("per_bin_empirical_summary")
    if not isinstance(summaries, list) or len(summaries) != count:
        raise TError("Q receipt per-bin summary count mismatch")
    indexed: dict[int, dict[str, Any]] = {}
    for row in summaries:
        if not isinstance(row, dict):
            raise TError("Q per-bin summary must be object")
        index = row.get("azimuth_bin_index")
        if (
            not isinstance(index, int)
            or index in indexed
            or not 0 <= index < count
        ):
            raise TError("Q per-bin summary index invalid/duplicate")
        if policy["require_all_q_members_per_bin"]:
            if row.get("member_count_with_candidate") != member_count:
                raise TError(f"Q bin {index} does not contain all ensemble members")
        indexed[index] = row
    if set(indexed) != set(range(count)):
        raise TError("Q receipt azimuth coverage incomplete")

    selected_quantile = (
        float(policy["quantile"])
        if policy["statistic_mode"] == "empirical_ensemble_quantile"
        else None
    )
    if selected_quantile is not None:
        declared = receipt.get("quantiles")
        if (
            not isinstance(declared, list)
            or not any(
                abs(float(value) - selected_quantile) <= 1e-12
                for value in declared
            )
        ):
            raise TError("requested T quantile is not explicitly declared in Q receipt")
    return indexed, float(width), count, selected_quantile


def index_layers(
    receipt: dict[str, Any], label: str, validate_support: bool = False
) -> dict[str, dict[str, Any]]:
    layers = receipt.get("layers")
    if not isinstance(layers, list) or not layers:
        raise TError(f"{label} layers missing")
    indexed: dict[str, dict[str, Any]] = {}
    for layer in layers:
        if not isinstance(layer, dict) or not isinstance(layer.get("layer_id"), str):
            raise TError(f"{label} layer invalid")
        layer_id = layer["layer_id"]
        if layer_id in indexed:
            raise TError(f"{label} duplicate layer")
        if validate_support and layer.get("support_class") not in SUPPORT_CLASSES:
            raise TError(f"R layer {layer_id}: unsupported support class")
        indexed[layer_id] = layer
    return indexed


def validate_o_r_m(
    policy: dict[str, Any],
    o: dict[str, Any],
    r: dict[str, Any],
    m_path: pathlib.Path,
    m: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if o.get("schema_version") != O_SCHEMA:
        raise TError("O receipt schema mismatch")
    if r.get("schema_version") != R_SCHEMA:
        raise TError("R receipt schema mismatch")
    if m.get("schema_version") != M_SCHEMA or m.get("status") != "pass":
        raise TError("M receipt must be a passing radial-uncertainty receipt")
    verify_self_hash(o, "O receipt")
    verify_self_hash(r, "R receipt")
    verify_self_hash(m, "M receipt")

    study = policy["study_id"]
    if (
        o.get("study_id") != study
        or r.get("study_id") != study
        or m.get("study_id") != study
    ):
        raise TError("O/R/M/T study mismatch")
    if o.get("m_receipt_sha256") != sha256_file(m_path):
        raise TError("O receipt does not bind exact M receipt")
    if o.get("l_pack_sha256") != m.get("l_pack_sha256"):
        raise TError("O/M L-pack binding mismatch")

    o_layers = index_layers(o, "O receipt")
    r_layers = index_layers(r, "R receipt", validate_support=True)
    expected = set(policy["expected_layer_ids"])
    if set(o_layers) != expected or set(r_layers) != expected:
        raise TError("T expected layer set differs from O/R layer sets")

    ensemble_semantics = o_layers[policy["ensemble_layer_id"]].get(
        "semantics_class"
    )
    if ensemble_semantics not in {"rms_error", "empirical_ensemble"}:
        raise TError("Q ensemble layer must close an RMS/empirical O layer")
    for layer_id in policy["companion_layer_ids"]:
        if o_layers[layer_id].get("semantics_class") != "hard_upper_bound":
            raise TError(
                f"{layer_id}: T V1 companion requires O hard_upper_bound semantics"
            )

    site = o.get("site_vertical_uncertainty")
    if not isinstance(site, dict) or site.get("semantics_class") != "hard_upper_bound":
        raise TError(
            "T V1 blocks composition unless site vertical uncertainty is hard_upper_bound; "
            "Q same-realization site statistics have not yet been propagated into companion far-field geometry"
        )
    return o_layers, r_layers


def support_margin(layer: dict[str, Any]) -> float:
    value = layer.get("observed_positive_excursion_margin_deg", 0.0)
    if not finite(value) or float(value) < 0:
        raise TError(f"{layer.get('layer_id')}: invalid spatial-support margin")
    return float(value)


def select_q_value(
    row: dict[str, Any], mode: str, quantile: float | None
) -> tuple[float, dict[str, Any]]:
    if mode == "finite_ensemble_observed_max":
        value = row.get("max_deg")
        if not finite(value):
            raise TError("Q finite ensemble max missing/non-finite")
        return float(value), {
            "mode": mode,
            "estimator": "finite_observed_max",
            "quantile": None,
        }
    assert quantile is not None
    key = f"{quantile:.6f}"
    quantiles = row.get("quantiles_deg")
    if not isinstance(quantiles, dict) or not finite(quantiles.get(key)):
        raise TError(f"Q requested quantile {key} missing/non-finite")
    return float(quantiles[key]), {
        "mode": mode,
        "estimator": "empirical_cdf_nearest_rank",
        "quantile": quantile,
        "quantile_key": key,
    }


def filtered_companion_input(
    m_k_input: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    if m_k_input.get("schema_version") != K_INPUT_SCHEMA:
        raise TError("M augmented K input schema mismatch")
    layers = m_k_input.get("layers")
    if not isinstance(layers, list) or not layers:
        raise TError("M augmented K input layers missing")
    indexed: dict[str, dict[str, Any]] = {}
    for layer in layers:
        if not isinstance(layer, dict) or not isinstance(layer.get("layer_id"), str):
            raise TError("M augmented K input layer invalid")
        layer_id = layer["layer_id"]
        if layer_id in indexed:
            raise TError("M augmented K input duplicate layer")
        indexed[layer_id] = layer
    if set(indexed) != set(policy["expected_layer_ids"]):
        raise TError("M augmented K input layer set differs from T expected layers")
    output = json.loads(json.dumps(m_k_input))
    output["layers"] = [
        indexed[layer_id] for layer_id in policy["companion_layer_ids"]
    ]
    return output


def compose(
    policy_path: pathlib.Path,
    q_policy_path: pathlib.Path,
    q_receipt_path: pathlib.Path,
    o_receipt_path: pathlib.Path,
    r_receipt_path: pathlib.Path,
    m_receipt_path: pathlib.Path,
    m_k_input_path: pathlib.Path,
    artifact_root: pathlib.Path,
) -> dict[str, Any]:
    policy = validate_policy(read_obj(policy_path, "T policy"))
    q_policy = validate_q_policy(read_obj(q_policy_path, "Q policy"), policy)
    q_receipt = read_obj(q_receipt_path, "Q receipt")
    o_receipt = read_obj(o_receipt_path, "O receipt")
    r_receipt = read_obj(r_receipt_path, "R receipt")
    m_receipt = read_obj(m_receipt_path, "M receipt")
    m_k_input = read_obj(m_k_input_path, "M augmented K input")

    q_bins, bin_width, bin_count, quantile = validate_q_receipt(
        q_receipt, q_policy_path, q_policy, policy
    )
    _, r_layers = validate_o_r_m(
        policy, o_receipt, r_receipt, m_receipt_path, m_receipt
    )

    if m_receipt.get("augmented_k_sha256") != sha256_file(m_k_input_path):
        raise TError("M receipt does not bind exact augmented K input")
    if m_k_input.get("study_id") != policy["study_id"]:
        raise TError("M augmented K input/T study mismatch")
    if m_k_input.get("site_ref") != q_receipt.get("site_ref"):
        raise TError("Q/M-K site_ref mismatch")
    if m_k_input.get("frame") != q_receipt.get("native_frame"):
        raise TError("Q/M-K frame mismatch")
    if (
        not finite(m_k_input.get("azimuth_bin_width_deg"))
        or abs(float(m_k_input["azimuth_bin_width_deg"]) - bin_width) > 1e-12
    ):
        raise TError("Q/M-K azimuth bin width mismatch")

    companion_input = filtered_companion_input(m_k_input, policy)
    with tempfile.TemporaryDirectory() as directory:
        companion_input_path = pathlib.Path(directory) / "companion-k-input.json"
        companion_input_path.write_bytes(canonical_bytes(companion_input))
        try:
            companion_k = K.materialize(companion_input_path, artifact_root)
        except Exception as exc:
            raise TError(f"companion K materialization failed: {exc}") from exc

    if (
        companion_k.get("schema_version") != K_OUTPUT_SCHEMA
        or companion_k.get("status") != "pass"
    ):
        raise TError("companion K materialization did not pass")
    if (
        companion_k.get("study_id") != policy["study_id"]
        or companion_k.get("site_ref") != q_receipt.get("site_ref")
        or companion_k.get("frame") != q_receipt.get("native_frame")
    ):
        raise TError("Q/companion K lineage mismatch")
    if (
        abs(float(companion_k.get("azimuth_bin_width_deg")) - bin_width) > 1e-12
        or companion_k.get("bin_count") != bin_count
    ):
        raise TError("Q/companion K bin geometry mismatch")

    companion_bins = {
        row["bin_index"]: row for row in companion_k.get("bins", [])
    }
    if set(companion_bins) != set(range(bin_count)):
        raise TError("companion K bin coverage incomplete")

    ensemble_layer = policy["ensemble_layer_id"]
    ensemble_margin = support_margin(r_layers[ensemble_layer])
    output_bins = []
    ensemble_wins = 0
    companion_wins = 0

    for index in range(bin_count):
        q_value, statistic = select_q_value(
            q_bins[index], policy["statistic_mode"], quantile
        )
        q_final = q_value + ensemble_margin

        companion_bin = companion_bins[index]
        companion_base = companion_bin.get("conservative_elevation_deg")
        companion_winner = companion_bin.get("winner")
        if not finite(companion_base) or not isinstance(companion_winner, dict):
            raise TError(f"companion K bin {index} invalid")
        companion_layer = companion_winner.get("layer_id")
        if companion_layer not in policy["companion_layer_ids"]:
            raise TError(
                f"companion K bin {index} winner outside companion layer set"
            )
        companion_margin = support_margin(r_layers[companion_layer])
        companion_final = float(companion_base) + companion_margin

        if q_final >= companion_final:
            final = q_final
            ensemble_wins += 1
            winner = {
                "source_kind": "ll009q_empirical_ensemble",
                "layer_id": ensemble_layer,
                "vertical_horizon_deg": q_value,
                "spatial_support_margin_deg": ensemble_margin,
                "final_elevation_deg": final,
                "statistic": statistic,
                "q_member_count": q_receipt["member_count"],
            }
            runner_up = {
                "source_kind": "ll009k_companion",
                "layer_id": companion_layer,
                "vertical_horizon_deg": float(companion_base),
                "spatial_support_margin_deg": companion_margin,
                "final_elevation_deg": companion_final,
            }
        else:
            final = companion_final
            companion_wins += 1
            winner = {
                "source_kind": "ll009k_companion",
                "layer_id": companion_layer,
                "vertical_horizon_deg": float(companion_base),
                "spatial_support_margin_deg": companion_margin,
                "final_elevation_deg": final,
                "k_winner": companion_winner,
            }
            runner_up = {
                "source_kind": "ll009q_empirical_ensemble",
                "layer_id": ensemble_layer,
                "vertical_horizon_deg": q_value,
                "spatial_support_margin_deg": ensemble_margin,
                "final_elevation_deg": q_final,
                "statistic": statistic,
            }

        output_bins.append(
            {
                "bin_index": index,
                "azimuth_start_deg": index * bin_width,
                "azimuth_end_deg": (index + 1) * bin_width,
                "conservative_elevation_deg": final,
                "winner": winner,
                "runner_up": runner_up,
                "candidate_count": 2,
                "numeric_semantics": "statistical_or_hard_bound_max_envelope",
            }
        )

    layers = [
        {
            "layer_id": ensemble_layer,
            "composition_role": "empirical_ensemble",
            "statistic_mode": policy["statistic_mode"],
            "quantile": quantile,
            "spatial_support_class": r_layers[ensemble_layer]["support_class"],
            "spatial_support_margin_deg": ensemble_margin,
        }
    ]
    for layer in companion_k["layers"]:
        record = json.loads(json.dumps(layer))
        layer_id = record["layer_id"]
        record["composition_role"] = "hard_upper_bound_companion"
        record["spatial_support_class"] = r_layers[layer_id]["support_class"]
        record["spatial_support_margin_deg"] = support_margin(r_layers[layer_id])
        layers.append(record)

    policy_hash = sha256_file(policy_path)
    q_receipt_hash = sha256_file(q_receipt_path)
    o_receipt_hash = sha256_file(o_receipt_path)
    r_receipt_hash = sha256_file(r_receipt_path)
    m_receipt_hash = sha256_file(m_receipt_path)
    m_k_hash = sha256_file(m_k_input_path)
    companion_k_hash = sha256_bytes(canonical_bytes(companion_k))

    binding = {
        "status": "bound",
        "schema_version": "ll009t.statistical-horizon-binding.v1",
        "mode": policy["statistic_mode"],
        "estimator": (
            "empirical_cdf_nearest_rank"
            if policy["statistic_mode"] == "empirical_ensemble_quantile"
            else "finite_observed_max"
        ),
        "quantile": quantile,
        "ensemble_layer_id": ensemble_layer,
        "companion_layer_ids": list(policy["companion_layer_ids"]),
        "covered_layer_ids": list(policy["expected_layer_ids"]),
        "q_policy_sha256": sha256_file(q_policy_path),
        "q_receipt_sha256": q_receipt_hash,
        "o_uncertainty_receipt_sha256": o_receipt_hash,
        "r_spatial_support_receipt_sha256": r_receipt_hash,
        "m_radial_uncertainty_receipt_sha256": m_receipt_hash,
        "m_augmented_k_input_sha256": m_k_hash,
        "derived_companion_k_sha256": companion_k_hash,
        "numeric_composition_policy_sha256": policy_hash,
        "site_observer_compatibility": policy["site_observer_compatibility"],
        "site_observer_closure": (
            "companion_site_vertical_uncertainty_is_hard_upper_bound"
        ),
        "spatial_support_margin_policy": policy["spatial_support_margin_policy"],
        "bin_composition_rule": (
            "max(q_selected_statistic_plus_r_margin,companion_k_plus_r_margin)"
        ),
    }

    output = {
        "schema_version": K_OUTPUT_SCHEMA,
        "status": "pass",
        "study_id": companion_k["study_id"],
        "frame_contract_id": companion_k["frame_contract_id"],
        "epoch_contract_id": companion_k["epoch_contract_id"],
        "site_ref": companion_k["site_ref"],
        "frame": companion_k["frame"],
        "input_sha256": policy_hash,
        "source_hashes": {
            "ll009q-receipt": q_receipt_hash,
            "ll009o-receipt": o_receipt_hash,
            "ll009r-receipt": r_receipt_hash,
            "ll009m-receipt": m_receipt_hash,
            "ll009m-augmented-k-input": m_k_hash,
            **{
                f"companion:{key}": value
                for key, value in companion_k.get("source_hashes", {}).items()
            },
        },
        "site_position_m": companion_k["site_position_m"],
        "site_vertical_uncertainty_m": companion_k["site_vertical_uncertainty_m"],
        "basis": companion_k["basis"],
        "azimuth_bin_width_deg": bin_width,
        "bin_count": bin_count,
        "layers": layers,
        "bins": output_bins,
        "statistical_horizon_binding": binding,
        "composition_summary": {
            "ensemble_winning_bins": ensemble_wins,
            "companion_winning_bins": companion_wins,
            "total_bins": bin_count,
        },
        "semantics": (
            "Each output bin is the maximum of an exact LL-009Q empirical ensemble "
            "statistic and an independently rematerialized hard-bound companion K "
            "horizon, after any LL-009R observed positive-excursion margins explicitly "
            "present in the exact spatial-support receipt."
        ),
        "non_claims": [
            "The selected LL-009Q empirical statistic is not a deterministic upper bound on all possible terrain.",
            "A finite ensemble quantile is not promoted to a population confidence guarantee.",
            "Spatial-support classes remain exactly those declared by LL-009R; sample/empirical support is not continuous-hard terrain closure.",
            "T V1 refuses RMS/statistical site uncertainty for companion far-field geometry because the Q same-realization observer has not yet been propagated into that geometry.",
            "This horizon pack does not establish illumination, communications, site viability, architecture superiority, or operations authority."
        ],
    }
    output["receipt_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def write_immutable(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise TError(f"refusing to overwrite differing output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        artifact_root = root / "artifacts"
        artifact_root.mkdir()
        source = artifact_root / "l-pack.json"
        source.write_text('{"synthetic":"l-pack"}\n')
        l_hash = sha256_file(source)

        site = (1_000_000.0, 0.0, 0.0)
        basis = K.local_basis(site, (0.0, 0.0, 1.0))
        near_samples = [
            {
                "position_m": K.synthetic_point(site, basis, az, 1.0, 1000.0),
                "vertical_uncertainty_m": 0.0,
            }
            for az in (45.0, 135.0, 225.0, 315.0)
        ]
        far_samples = [
            {
                "position_m": K.synthetic_point(
                    site, basis, az, 8.0 if az == 135.0 else 2.0, 10000.0
                ),
                "vertical_uncertainty_m": 0.0,
            }
            for az in (45.0, 135.0, 225.0, 315.0)
        ]
        m_k_input = {
            "schema_version": K_INPUT_SCHEMA,
            "study_id": "study-t",
            "frame_contract_id": "frame-t",
            "epoch_contract_id": "epoch-t",
            "site_ref": "site-t",
            "frame": "FRAME-T",
            "sources": [
                {
                    "source_id": "ll009l-pack",
                    "path": "l-pack.json",
                    "sha256": l_hash,
                }
            ],
            "site_position_m": list(site),
            "site_vertical_uncertainty_m": 1.0,
            "pole_vector": [0.0, 0.0, 1.0],
            "azimuth_bin_width_deg": 90.0,
            "layers": [
                {
                    "layer_id": "near",
                    "source_ref": "ll009l-pack",
                    "frame": "FRAME-T",
                    "min_range_m": 500.0,
                    "max_range_m": 2000.0,
                    "nominal_resolution_m": 5.0,
                    "additional_angular_margin_deg": 0.0,
                    "samples": near_samples,
                },
                {
                    "layer_id": "far",
                    "source_ref": "ll009l-pack",
                    "frame": "FRAME-T",
                    "min_range_m": 5000.0,
                    "max_range_m": 20000.0,
                    "nominal_resolution_m": 80.0,
                    "additional_angular_margin_deg": 0.1,
                    "samples": far_samples,
                },
            ],
        }
        m_k_path = root / "m-k.json"
        m_k_path.write_bytes(canonical_bytes(m_k_input))

        m_receipt = {
            "schema_version": M_SCHEMA,
            "status": "pass",
            "study_id": "study-t",
            "l_pack_sha256": l_hash,
            "augmented_k_sha256": sha256_file(m_k_path),
        }
        m_receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(m_receipt))
        m_path = root / "m.json"
        m_path.write_bytes(canonical_bytes(m_receipt))

        q_policy = {
            "schema_version": Q_POLICY_SCHEMA,
            "study_id": "study-t",
            "ensemble_layer_id": "near",
            "quantile_estimator": "empirical_cdf_nearest_rank",
        }
        q_policy_path = root / "q-policy.json"
        q_policy_path.write_bytes(canonical_bytes(q_policy))

        summaries = []
        for index, (q95, qmax) in enumerate(
            zip((6.0, 3.0, 4.0, 5.0), (7.0, 4.0, 5.0, 6.0))
        ):
            summaries.append(
                {
                    "azimuth_bin_index": index,
                    "member_count_with_candidate": 100,
                    "min_deg": 0.0,
                    "max_deg": qmax,
                    "max_member_ordinal": 99,
                    "quantiles_deg": {"0.950000": q95},
                }
            )
        q_receipt = {
            "schema_version": Q_RECEIPT_SCHEMA,
            "status": "pass",
            "semantics_class": "empirical_ensemble",
            "study_id": "study-t",
            "site_ref": "site-t",
            "native_frame": "FRAME-T",
            "policy_sha256": sha256_file(q_policy_path),
            "quantile_estimator": "empirical_cdf_nearest_rank",
            "quantiles": [0.95],
            "azimuth_bin_width_deg": 90.0,
            "azimuth_bin_count": 4,
            "member_count": 100,
            "per_bin_empirical_summary": summaries,
        }
        q_receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(q_receipt))
        q_path = root / "q.json"
        q_path.write_bytes(canonical_bytes(q_receipt))

        o_receipt = {
            "schema_version": O_SCHEMA,
            "status": "pass",
            "study_id": "study-t",
            "l_pack_sha256": l_hash,
            "m_receipt_sha256": sha256_file(m_path),
            "site_vertical_uncertainty": {
                "semantics_class": "hard_upper_bound",
                "source_id": "site-hard",
            },
            "layers": [
                {"layer_id": "near", "semantics_class": "rms_error"},
                {"layer_id": "far", "semantics_class": "hard_upper_bound"},
            ],
        }
        o_receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(o_receipt))
        o_path = root / "o.json"
        o_path.write_bytes(canonical_bytes(o_receipt))

        r_receipt = {
            "schema_version": R_SCHEMA,
            "status": "pass",
            "study_id": "study-t",
            "layers": [
                {"layer_id": "near", "support_class": "sample_points_only"},
                {
                    "layer_id": "far",
                    "support_class": "empirical_multiscale_bound",
                    "observed_positive_excursion_margin_deg": 0.25,
                },
            ],
        }
        r_receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(r_receipt))
        r_path = root / "r.json"
        r_path.write_bytes(canonical_bytes(r_receipt))

        policy = {
            "schema_version": POLICY_SCHEMA,
            "study_id": "study-t",
            "ensemble_layer_id": "near",
            "companion_layer_ids": ["far"],
            "expected_layer_ids": ["near", "far"],
            "statistic_mode": "empirical_ensemble_quantile",
            "quantile": 0.95,
            "site_observer_compatibility": "require_hard_upper_bound_site_v1",
            "companion_vertical_semantics": "hard_upper_bound",
            "spatial_support_margin_policy": "apply_observed_positive_excursion_if_present",
            "require_all_q_members_per_bin": True,
        }
        policy_path = root / "t-policy.json"
        policy_path.write_bytes(canonical_bytes(policy))

        first = compose(
            policy_path,
            q_policy_path,
            q_path,
            o_path,
            r_path,
            m_path,
            m_k_path,
            artifact_root,
        )
        second = compose(
            policy_path,
            q_policy_path,
            q_path,
            o_path,
            r_path,
            m_path,
            m_k_path,
            artifact_root,
        )
        assert canonical_bytes(first) == canonical_bytes(second)
        assert first["statistical_horizon_binding"]["status"] == "bound"
        assert first["composition_summary"]["ensemble_winning_bins"] > 0
        assert first["composition_summary"]["companion_winning_bins"] > 0
        assert first["bins"][1]["winner"]["source_kind"] == "ll009k_companion"

        bad_policy = json.loads(json.dumps(policy))
        bad_policy["quantile"] = 0.99
        bad_policy_path = root / "bad-policy.json"
        bad_policy_path.write_bytes(canonical_bytes(bad_policy))
        try:
            compose(
                bad_policy_path,
                q_policy_path,
                q_path,
                o_path,
                r_path,
                m_path,
                m_k_path,
                artifact_root,
            )
        except TError as exc:
            assert "not explicitly declared" in str(exc)
        else:
            raise TError("self-test expected undeclared quantile rejection")

        rms_o = json.loads(json.dumps(o_receipt))
        rms_o["site_vertical_uncertainty"]["semantics_class"] = "rms_error"
        rms_o.pop("receipt_sha256")
        rms_o["receipt_sha256"] = sha256_bytes(canonical_bytes(rms_o))
        rms_o_path = root / "o-rms.json"
        rms_o_path.write_bytes(canonical_bytes(rms_o))
        try:
            compose(
                policy_path,
                q_policy_path,
                q_path,
                rms_o_path,
                r_path,
                m_path,
                m_k_path,
                artifact_root,
            )
        except TError as exc:
            assert "site vertical uncertainty is hard_upper_bound" in str(exc)
        else:
            raise TError("self-test expected observer incompatibility rejection")

        print("LL-009T statistical horizon composition self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LL-009T exact statistical horizon composition and K binding"
    )
    parser.add_argument("--policy")
    parser.add_argument("--q-policy")
    parser.add_argument("--q-receipt")
    parser.add_argument("--o-receipt")
    parser.add_argument("--r-receipt")
    parser.add_argument("--m-receipt")
    parser.add_argument("--m-k-input")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        required = (
            args.policy,
            args.q_policy,
            args.q_receipt,
            args.o_receipt,
            args.r_receipt,
            args.m_receipt,
            args.m_k_input,
            args.output,
        )
        if not all(required):
            raise TError(
                "--policy --q-policy --q-receipt --o-receipt --r-receipt "
                "--m-receipt --m-k-input --output required"
            )
        output = compose(
            pathlib.Path(args.policy),
            pathlib.Path(args.q_policy),
            pathlib.Path(args.q_receipt),
            pathlib.Path(args.o_receipt),
            pathlib.Path(args.r_receipt),
            pathlib.Path(args.m_receipt),
            pathlib.Path(args.m_k_input),
            pathlib.Path(args.artifact_root),
        )
        write_immutable(pathlib.Path(args.output), output)
        print(json.dumps(output, sort_keys=True, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, TError) as exc:
        raise SystemExit(f"LL-009T failure: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
