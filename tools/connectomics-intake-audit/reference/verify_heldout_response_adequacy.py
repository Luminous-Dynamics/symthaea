#!/usr/bin/env python3
"""Independent scoring and disposition audit for held-out response trajectories.

This V1 implementation scores discrete full-trajectory distributions. It reports
finite-sample descriptive diagnostics only and deliberately does not claim
population calibration or a conformal guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def valid_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def unique(values: list[str], label: str) -> None:
    require(all(isinstance(v, str) and v for v in values), f"{label} contains an empty ID")
    require(len(values) == len(set(values)), f"{label} contains duplicate IDs")


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["model_id"], row["intervention_id"], row["unit_id"]


def highest_probability_set(
    probabilities: dict[str, float], nominal_coverage: float = 0.90
) -> set[str]:
    ordered = sorted(probabilities.items(), key=lambda row: (-row[1], row[0]))
    selected: set[str] = set()
    cumulative = 0.0
    for trajectory_id, probability in ordered:
        if probability <= 0.0:
            continue
        selected.add(trajectory_id)
        cumulative += probability
        if cumulative + 1e-12 >= nominal_coverage:
            break
    return selected


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(data["schema_version"] == 1, "unsupported schema version")
    require(data["claim_scope"] == "synthetic_known_truth", "V1 oracle only accepts synthetic known-truth fixtures")
    require(data["dataset_binding"]["evidence_tier"] == "SYNTHETIC", "synthetic fixture must use SYNTHETIC evidence tier")
    require(valid_sha(data["dataset_binding"]["dataset_revision_sha256"]), "dataset revision digest malformed")
    require(valid_sha(data["dataset_binding"]["source_access_receipt_sha256"]), "source-access receipt malformed")
    require(valid_sha(data["split_manifest"]["split_sha256"]), "split digest malformed")

    time_grid = data["time_grid_ms"]
    require(len(time_grid) >= 2 and all(math.isfinite(x) for x in time_grid), "time grid must contain finite values")
    require(all(time_grid[i] < time_grid[i + 1] for i in range(len(time_grid) - 1)), "time grid must be strictly increasing")

    models = sorted(data["models"], key=lambda row: row["model_id"])
    interventions = sorted(data["interventions"], key=lambda row: row["intervention_id"])
    units = sorted(data["units"], key=lambda row: row["unit_id"])
    predictions = data["predictions"]
    model_ids = [row["model_id"] for row in models]
    intervention_ids = [row["intervention_id"] for row in interventions]
    unit_ids = [row["unit_id"] for row in units]
    unique(model_ids, "model IDs")
    unique(intervention_ids, "intervention IDs")
    unique(unit_ids, "independent unit IDs")

    split = data["split_manifest"]
    train = split["training_units"]
    calibration = split["calibration_units"]
    lockbox = split["lockbox_units"]
    unique(train, "training split")
    unique(calibration, "calibration split")
    unique(lockbox, "lockbox split")
    require(set(train).isdisjoint(calibration), "training/calibration unit leakage")
    require(set(train).isdisjoint(lockbox), "training/lockbox unit leakage")
    require(set(calibration).isdisjoint(lockbox), "calibration/lockbox unit leakage")
    require(set(unit_ids) == set(calibration) | set(lockbox), "unit ledger must exactly match calibration and lockbox split manifests")
    require({row["unit_id"] for row in units if row["split"] == "CALIBRATION"} == set(calibration), "calibration unit dispositions do not match split manifest")
    require({row["unit_id"] for row in units if row["split"] == "LOCKBOX"} == set(lockbox), "lockbox unit dispositions do not match split manifest")
    require(train, "training unit set must not be empty")

    model_map = {row["model_id"]: row for row in models}
    for model in models:
        require(valid_sha(model["model_sha256"]), "model digest malformed")
        require(valid_sha(model["observation_kernel_sha256"]), "observation-kernel digest malformed")
        require(valid_sha(model["training_data_sha256"]), "training-data digest malformed")
    intervention_map = {row["intervention_id"]: row for row in interventions}
    for intervention in interventions:
        require(valid_sha(intervention["intervention_receipt_sha256"]), "intervention receipt malformed")
    unit_map = {row["unit_id"]: row for row in units}
    for unit in units:
        require(valid_sha(unit["source_receipt_sha256"]), "unit source receipt malformed")
        status = unit["observation_status"]
        if status == "OBSERVED":
            require("observed_trajectory_id" in unit, "observed unit lacks trajectory identity")
        else:
            require(status in {"FAILED", "INACCESSIBLE", "NOT_ATTEMPTED"}, "unknown observation disposition")
            require("observed_trajectory_id" not in unit, "missing unit has an observed trajectory")

    expected_keys = {
        (model_id, intervention_id, unit_id)
        for model_id in model_ids
        for intervention_id in intervention_ids
        for unit_id in unit_ids
    }
    pred_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    trajectory_values: dict[str, tuple[float, ...]] = {}
    for row in predictions:
        key = prediction_key(row)
        require(key in expected_keys, f"prediction has unknown model/intervention/unit key: {key}")
        require(key not in pred_index, f"duplicate prediction key: {key}")
        distribution: dict[str, Any] = {}
        total = 0.0
        for trajectory in row["predicted_trajectories"]:
            trajectory_id = trajectory["trajectory_id"]
            require(trajectory_id not in distribution, f"duplicate trajectory ID in prediction {key}")
            values = trajectory["values"]
            require(len(values) == len(time_grid), f"trajectory dimension differs from time grid for {key}")
            require(all(math.isfinite(value) for value in values), f"non-finite trajectory value for {key}")
            probability = trajectory["probability"]
            require(math.isfinite(probability) and 0.0 <= probability <= 1.0, f"invalid trajectory probability for {key}")
            values_tuple = tuple(values)
            if trajectory_id in trajectory_values:
                require(trajectory_values[trajectory_id] == values_tuple, f"trajectory ID {trajectory_id} maps to inconsistent values")
            trajectory_values[trajectory_id] = values_tuple
            distribution[trajectory_id] = trajectory
            total += probability
        require(abs(total - 1.0) <= 1e-9, f"trajectory probabilities do not sum to one for {key}")
        pred_index[key] = distribution
    require(set(pred_index) == expected_keys, "prediction matrix must cover every model × intervention × evaluation unit combination")

    rows_by_model_split: dict[tuple[str, str], list[dict[str, Any]]] = {}
    unit_score_rows = []
    zero_support_events = []
    for model_id in model_ids:
        for intervention_id in intervention_ids:
            for unit_id in unit_ids:
                unit = unit_map[unit_id]
                if unit["observation_status"] != "OBSERVED":
                    continue
                distribution_rows = pred_index[(model_id, intervention_id, unit_id)]
                probabilities = {
                    trajectory_id: row["probability"]
                    for trajectory_id, row in distribution_rows.items()
                }
                outcome_universe = sorted(
                    {
                        trajectory_id
                        for other_model in model_ids
                        for trajectory_id in pred_index[(other_model, intervention_id, unit_id)]
                    }
                )
                observed_id = unit["observed_trajectory_id"]
                observed_probability = probabilities.get(observed_id, 0.0)
                zero_support = observed_probability <= 0.0
                negative_log_score = None if zero_support else -math.log(observed_probability)
                brier = sum(
                    (
                        probabilities.get(trajectory_id, 0.0)
                        - (1.0 if trajectory_id == observed_id else 0.0)
                    ) ** 2
                    for trajectory_id in outcome_universe
                )
                predictive_set = highest_probability_set(probabilities, 0.90)
                covered_90 = observed_id in predictive_set
                score = {
                    "model_id": model_id,
                    "intervention_id": intervention_id,
                    "unit_id": unit_id,
                    "split": unit["split"],
                    "observed_trajectory_id": observed_id,
                    "observed_probability": observed_probability,
                    "zero_support": zero_support,
                    "negative_log_score": negative_log_score,
                    "multiclass_brier_score": brier,
                    "predictive_set_90": sorted(predictive_set),
                    "covered_90": covered_90,
                }
                unit_score_rows.append(score)
                rows_by_model_split.setdefault((model_id, unit["split"]), []).append(score)
                if zero_support and unit["split"] == "LOCKBOX":
                    zero_support_events.append(
                        {"model_id": model_id, "unit_id": unit_id, "trajectory_id": observed_id}
                    )

    summaries = []
    for model_id in model_ids:
        for split_name in ("CALIBRATION", "LOCKBOX"):
            scored = rows_by_model_split.get((model_id, split_name), [])
            units_for_split = [row for row in units if row["split"] == split_name]
            missing_count = sum(row["observation_status"] != "OBSERVED" for row in units_for_split)
            supported_nll = [
                row["negative_log_score"] for row in scored
                if row["negative_log_score"] is not None
            ]
            summaries.append(
                {
                    "model_id": model_id,
                    "split": split_name,
                    "observed_unit_count": len(scored),
                    "missing_unit_count": missing_count,
                    "zero_support_count": sum(row["zero_support"] for row in scored),
                    "mean_negative_log_score_on_supported_units": (
                        sum(supported_nll) / len(supported_nll) if supported_nll else None
                    ),
                    "mean_multiclass_brier_score": (
                        sum(row["multiclass_brier_score"] for row in scored) / len(scored)
                        if scored else None
                    ),
                    "predictive_set_coverage_90_count": sum(row["covered_90"] for row in scored),
                    "predictive_set_coverage_90_fraction": (
                        sum(row["covered_90"] for row in scored) / len(scored) if scored else None
                    ),
                    "calibration_claim_status": "DESCRIPTIVE_ONLY_INDEPENDENT_UNIT_SAMPLE_TOO_SMALL",
                }
            )

    expected = data["expected_outcomes"]
    require(len(calibration) == expected["calibration_unit_count"], "calibration unit count differs from known answer")
    lock_observed = sum(unit_map[unit_id]["observation_status"] == "OBSERVED" for unit_id in lockbox)
    lock_failed = sum(unit_map[unit_id]["observation_status"] in {"FAILED", "INACCESSIBLE", "NOT_ATTEMPTED"} for unit_id in lockbox)
    require(lock_observed == expected["lockbox_observed_unit_count"], "lockbox observed unit count differs from known answer")
    require(lock_failed == expected["lockbox_failed_unit_count"], "lockbox missing unit count differs from known answer")

    summary_map = {(row["model_id"], row["split"]): row for row in summaries}
    baseline_model = model_ids[0]
    challenger_model = model_ids[1] if len(model_ids) > 1 else None
    require(challenger_model is not None, "synthetic known-answer fixture requires baseline and challenger models")
    baseline_lockbox = summary_map[(baseline_model, "LOCKBOX")]
    challenger_lockbox = summary_map[(challenger_model, "LOCKBOX")]
    require(baseline_lockbox["zero_support_count"] == expected["baseline_lockbox_zero_support_count"], "baseline lockbox zero-support count mismatch")
    require(challenger_lockbox["zero_support_count"] == expected["challenger_lockbox_zero_support_count"], "challenger lockbox zero-support count mismatch")
    require(baseline_lockbox["predictive_set_coverage_90_count"] == expected["baseline_lockbox_coverage_90_count"], "baseline 90% set coverage count mismatch")
    require(challenger_lockbox["predictive_set_coverage_90_count"] == expected["challenger_lockbox_coverage_90_count"], "challenger 90% set coverage count mismatch")

    return {
        "schema_version": 1,
        "tool": "verify-heldout-response-adequacy",
        "status": "SYNTHETIC_HELDOUT_TRAJECTORY_AUDIT_VALIDATED",
        "claim_scope": "synthetic_known_truth",
        "dataset_revision_sha256": data["dataset_binding"]["dataset_revision_sha256"],
        "split_sha256": split["split_sha256"],
        "training_unit_count": len(train),
        "calibration_unit_count": len(calibration),
        "lockbox_unit_count": len(lockbox),
        "lockbox_observed_unit_count": lock_observed,
        "lockbox_missing_unit_count": lock_failed,
        "model_count": len(models),
        "intervention_count": len(interventions),
        "prediction_record_count": len(predictions),
        "per_model_split_summaries": summaries,
        "unit_level_scores": unit_score_rows,
        "lockbox_zero_support_events": zero_support_events,
        "posterior_update_disposition": "WITHHOLD_FOR_ZERO_SUPPORT_OBSERVATIONS" if zero_support_events else "NO_ZERO_SUPPORT_EVENT_OBSERVED",
        "calibration_claim_status": expected["calibration_claim_status"],
        "population_calibration": "NOT_ESTIMATED",
        "conformal_coverage_guarantee": "NOT_CLAIMED",
        "scope_note": "Scores and coverage counts are descriptive for the declared synthetic independent-unit fixture only. They do not establish population calibration, temporal exchangeability, biological response-model adequacy, or causal synaptic mechanisms.",
    }


def self_test() -> None:
    # The deterministic set rule includes outcomes in descending probability
    # order until the nominal mass is reached, with stable ID tie-breaking.
    selected = highest_probability_set({"A": 0.60, "B": 0.30, "C": 0.10}, 0.90)
    require(selected == {"A", "B"}, "90% highest-probability set negative control failed")
    require(highest_probability_set({"A": 0.8, "B": 0.2}, 0.9) == {"A", "B"}, "baseline predictive set negative control failed")
    print("held-out trajectory set-scoring negative controls: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        try:
            self_test()
            return 0
        except Exception as exc:
            print(f"held-out response self-test: FAIL: {exc}", file=sys.stderr)
            return 1
    try:
        require(args.input is not None and args.report is not None, "input and report paths are required")
        raw = args.input.read_bytes()
        data = json.loads(raw)
        report = verify(data)
        report["input_sha256"] = sha256(raw)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "held-out trajectory audit: PASS; "
            f"lockbox_units={report['lockbox_unit_count']}; "
            f"zero_support_events={len(report['lockbox_zero_support_events'])}; "
            f"missing_units={report['lockbox_missing_unit_count']}; "
            f"calibration={report['calibration_claim_status']}"
        )
        return 0
    except Exception as exc:
        print(f"held-out response adequacy audit: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
