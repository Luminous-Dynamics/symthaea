#!/usr/bin/env python3
"""Synthetic historical high-fidelity transfer benchmark for casting-like tasks.

This benchmark measures out-of-task prediction transfer, not an optimizer win.
Target high-fidelity labels are used only by the audit scorer after all policy
predictions have been generated. Source-task high/low pairs form the historical
archive; target high-fidelity evaluation cost is never hidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Sequence

SCHEMA_VERSION = "1.0.0"
BENCHMARK_ID = "CASTING-HISTORY-TRANSFER-001"
LOW_COST_UNITS = 1
HIGH_COST_UNITS = 25
DEFAULT_HISTORY_POINTS = 48
DEFAULT_TEST_POINTS = 128
DEFAULT_NEAREST_K = 16
DEFAULT_REUSE_COUNT = 50
ABSTENTION_POLICY_ID = "descriptor_x_knn_distance_abstention"
ABSTENTION_DISTANCE_THRESHOLD = 0.9
FAMILY_BOOTSTRAP_REPLICATES = 2_000
FAMILY_BOOTSTRAP_SEED = 17

# Descriptors are normalized proxies for composition, thermal properties and
# geometry. Task family is a split/audit label and is never an input feature.
SOURCE_TASKS = (
    {"task_id": "source-alpha-1", "family": "alloy_alpha", "descriptor": [0.05, 0.20, 0.30]},
    {"task_id": "source-alpha-2", "family": "alloy_alpha", "descriptor": [0.15, 0.25, 0.35]},
    {"task_id": "source-beta-1", "family": "alloy_beta", "descriptor": [0.80, 0.20, 0.70]},
    {"task_id": "source-beta-2", "family": "alloy_beta", "descriptor": [0.90, 0.25, 0.75]},
    {"task_id": "source-sand-1", "family": "mold_sand", "descriptor": [0.35, 0.70, 0.40]},
    {"task_id": "source-sand-2", "family": "mold_sand", "descriptor": [0.45, 0.65, 0.45]},
    {"task_id": "source-ceramic-1", "family": "mold_ceramic", "descriptor": [0.60, 0.85, 0.70]},
    {"task_id": "source-ceramic-2", "family": "mold_ceramic", "descriptor": [0.70, 0.75, 0.65]},
)
# Four related-but-distinct task instances per held-out family. Families are
# categorically absent from the historical source archive; within-family variants
# are reported as nested cases rather than pretending they are independent families.
TARGET_TASKS = (
    {"task_id": "target-gamma-1", "family": "unseen_alloy_gamma", "descriptor": [0.42, 0.35, 0.45]},
    {"task_id": "target-gamma-2", "family": "unseen_alloy_gamma", "descriptor": [0.38, 0.30, 0.40]},
    {"task_id": "target-gamma-3", "family": "unseen_alloy_gamma", "descriptor": [0.46, 0.40, 0.48]},
    {"task_id": "target-gamma-4", "family": "unseen_alloy_gamma", "descriptor": [0.40, 0.45, 0.42]},
    {"task_id": "target-investment-1", "family": "unseen_investment_geometry", "descriptor": [0.55, 0.78, 0.90]},
    {"task_id": "target-investment-2", "family": "unseen_investment_geometry", "descriptor": [0.52, 0.72, 0.84]},
    {"task_id": "target-investment-3", "family": "unseen_investment_geometry", "descriptor": [0.58, 0.82, 0.94]},
    {"task_id": "target-investment-4", "family": "unseen_investment_geometry", "descriptor": [0.50, 0.75, 0.98]},
    {"task_id": "target-thermal-ood-1", "family": "out_of_domain_thermal", "descriptor": [0.98, 0.95, 0.25]},
    {"task_id": "target-thermal-ood-2", "family": "out_of_domain_thermal", "descriptor": [0.95, 0.90, 0.20]},
    {"task_id": "target-thermal-ood-3", "family": "out_of_domain_thermal", "descriptor": [0.99, 0.99, 0.32]},
    {"task_id": "target-thermal-ood-4", "family": "out_of_domain_thermal", "descriptor": [0.92, 0.96, 0.15]},
    {"task_id": "target-new-mold-1", "family": "unseen_mold_system", "descriptor": [0.30, 0.90, 0.55]},
    {"task_id": "target-new-mold-2", "family": "unseen_mold_system", "descriptor": [0.25, 0.85, 0.48]},
    {"task_id": "target-new-mold-3", "family": "unseen_mold_system", "descriptor": [0.32, 0.95, 0.60]},
    {"task_id": "target-new-mold-4", "family": "unseen_mold_system", "descriptor": [0.28, 0.88, 0.64]},
    {"task_id": "target-titanium-1", "family": "unseen_alloy_titanium", "descriptor": [0.72, 0.42, 0.56]},
    {"task_id": "target-titanium-2", "family": "unseen_alloy_titanium", "descriptor": [0.68, 0.38, 0.62]},
    {"task_id": "target-titanium-3", "family": "unseen_alloy_titanium", "descriptor": [0.78, 0.48, 0.50]},
    {"task_id": "target-titanium-4", "family": "unseen_alloy_titanium", "descriptor": [0.75, 0.45, 0.66]},
    {"task_id": "target-pressure-die-1", "family": "unseen_pressure_die_casting", "descriptor": [0.80, 0.30, 0.92]},
    {"task_id": "target-pressure-die-2", "family": "unseen_pressure_die_casting", "descriptor": [0.76, 0.35, 0.86]},
    {"task_id": "target-pressure-die-3", "family": "unseen_pressure_die_casting", "descriptor": [0.86, 0.25, 0.96]},
    {"task_id": "target-pressure-die-4", "family": "unseen_pressure_die_casting", "descriptor": [0.82, 0.40, 0.90]},
    {"task_id": "target-sand-core-1", "family": "unseen_sand_core_complexity", "descriptor": [0.32, 0.68, 0.82]},
    {"task_id": "target-sand-core-2", "family": "unseen_sand_core_complexity", "descriptor": [0.28, 0.62, 0.75]},
    {"task_id": "target-sand-core-3", "family": "unseen_sand_core_complexity", "descriptor": [0.38, 0.72, 0.90]},
    {"task_id": "target-sand-core-4", "family": "unseen_sand_core_complexity", "descriptor": [0.35, 0.66, 0.86]},
    {"task_id": "target-thin-wall-1", "family": "unseen_thin_wall_geometry", "descriptor": [0.62, 0.58, 0.08]},
    {"task_id": "target-thin-wall-2", "family": "unseen_thin_wall_geometry", "descriptor": [0.66, 0.52, 0.12]},
    {"task_id": "target-thin-wall-3", "family": "unseen_thin_wall_geometry", "descriptor": [0.58, 0.64, 0.05]},
    {"task_id": "target-thin-wall-4", "family": "unseen_thin_wall_geometry", "descriptor": [0.70, 0.60, 0.15]},
)
POLICY_IDS = (
    "low_fidelity_only",
    "pooled_residual_transfer",
    "descriptor_x_knn_transfer",
    "descriptor_x_knn_distance_abstention",
    "shuffled_descriptor_control",
)
TASK_SUITE = {
    "schema_version": SCHEMA_VERSION,
    "benchmark_id": BENCHMARK_ID,
    "source_tasks": SOURCE_TASKS,
    "target_tasks": TARGET_TASKS,
    "low_cost_units": LOW_COST_UNITS,
    "high_cost_units": HIGH_COST_UNITS,
    "target_high_fidelity_not_in_predictor": True,
}
TASK_SUITE_SHA256 = hashlib.sha256(
    json.dumps(TASK_SUITE, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
).hexdigest()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive_int(value: object, label: str, maximum: int = 100_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1, {maximum}]")
    return value


def _radical_inverse(index: int, base: int) -> float:
    value, factor = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def design_point(index: int) -> tuple[float, float]:
    """Deterministic two-dimensional Halton point; no dependencies or RNG."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError("design index must be a positive integer")
    return _radical_inverse(index, 2), _radical_inverse(index, 3)


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if isinstance(quantile, bool) or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(_finite(value, "percentile value") for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    replicates: int = FAMILY_BOOTSTRAP_REPLICATES,
    seed: int = FAMILY_BOOTSTRAP_SEED,
) -> list[float] | None:
    """Deterministic percentile interval resampling family-level units only."""
    if not values:
        return None
    replicates = _positive_int(replicates, "bootstrap replicates", maximum=100_000)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap seed must be an integer")
    clean = [_finite(value, "bootstrap value") for value in values]
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choice(clean) for _ in range(len(clean)))
        for _ in range(replicates)
    ]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _descriptor(task: dict[str, Any]) -> tuple[float, float, float]:
    raw = task.get("descriptor")
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("task descriptor must be a three-element list")
    values = tuple(_finite(value, "descriptor component") for value in raw)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("normalized task descriptors must be in [0, 1]")
    return values  # type: ignore[return-value]


def high_fidelity(task: dict[str, Any], x: tuple[float, float]) -> tuple[float, float, float]:
    """Deterministic task-conditional truth: two minimized objectives and margin."""
    alloy, thermal, geometry = _descriptor(task)
    x0, x1 = x
    center0 = 0.22 + 0.12 * alloy - 0.04 * thermal
    center1 = 0.72 - 0.10 * thermal + 0.05 * geometry
    f0 = (
        0.08 + (x0 - center0) ** 2
        + 0.22 * (x1 - (0.25 + 0.25 * geometry)) ** 2
        + 0.03 * thermal * math.sin(2.0 * math.pi * x0 * x1)
    )
    f1 = (
        0.08 + (x0 - (0.75 - 0.10 * geometry)) ** 2
        + 0.24 * (x1 - center1) ** 2
        + 0.025 * alloy * math.cos(2.0 * math.pi * x0) * math.sin(math.pi * x1)
    )
    margin_center0 = 0.48 + 0.05 * alloy - 0.03 * geometry
    margin_center1 = 0.50 + 0.05 * thermal
    margin = (
        0.11 - 1.2 * (x0 - margin_center0) ** 2
        - 0.9 * (x1 - margin_center1) ** 2
        + 0.04 * (geometry - 0.5) * math.sin(2.0 * math.pi * x0)
    )
    result = (f0, f1, margin)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("high-fidelity equation returned a non-finite result")
    return result


def low_fidelity(task: dict[str, Any], x: tuple[float, float]) -> tuple[float, float, float]:
    """Biased inexpensive proxy whose discrepancy depends on task descriptors and x."""
    alloy, thermal, geometry = _descriptor(task)
    x0, x1 = x
    high = high_fidelity(task, x)
    local = math.exp(-((x0 - 0.72) ** 2 + (x1 - 0.30) ** 2) / 0.025)
    delta0 = (
        0.06 * (thermal - 0.5)
        + 0.07 * (alloy - 0.5) * math.sin(2.0 * math.pi * x0)
        + 0.03 * (geometry - 0.5) * math.cos(2.0 * math.pi * x1)
        + 0.045 * (alloy - 0.5) * local
    )
    delta1 = (
        -0.05 * (geometry - 0.5)
        + 0.06 * (thermal - 0.5) * math.cos(2.0 * math.pi * x1)
        + 0.04 * (alloy - 0.5) * math.sin(2.0 * math.pi * x0)
        - 0.025 * (thermal - 0.5) * local
    )
    delta_margin = (
        0.12 * (thermal - 0.5) * math.sin(math.pi * x0)
        + 0.10 * (geometry - 0.5) * math.cos(math.pi * x1)
        + 0.07 * (alloy - 0.5) * local
    )
    result = (high[0] + delta0, high[1] + delta1, high[2] + delta_margin)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("low-fidelity equation returned a non-finite result")
    return result


def _mean_vector(rows: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
    if not rows:
        raise ValueError("cannot average an empty residual archive")
    return tuple(statistics.fmean(row[i] for row in rows) for i in range(3))  # type: ignore[return-value]


def _distance(
    query_descriptor: tuple[float, float, float],
    query_x: tuple[float, float],
    history_row: dict[str, Any],
) -> float:
    # Balanced normalized distance: half task descriptor, half candidate geometry.
    hdesc = tuple(history_row["descriptor"])
    hx = tuple(history_row["x"])
    descriptor_part = statistics.fmean(
        ((query_descriptor[i] - hdesc[i]) / 0.35) ** 2 for i in range(3)
    )
    design_part = statistics.fmean(
        ((query_x[i] - hx[i]) / 0.25) ** 2 for i in range(2)
    )
    return math.sqrt(0.5 * descriptor_part + 0.5 * design_part)


def _knn_residual_with_distance(
    task: dict[str, Any],
    x: tuple[float, float],
    history: Sequence[dict[str, Any]],
    k: int,
) -> tuple[tuple[float, float, float], float]:
    query_descriptor = _descriptor(task)
    nearest = sorted(
        ((_distance(query_descriptor, x, row), index, row) for index, row in enumerate(history)),
        key=lambda item: (item[0], item[1]),
    )[: min(k, len(history))]
    if not nearest:
        raise ValueError("historical transfer requires a non-empty archive")
    raw_weights = [1.0 / ((distance + 0.025) ** 2) for distance, _, _ in nearest]
    total = sum(raw_weights)
    weights = [weight / total for weight in raw_weights]
    residual = tuple(
        sum(weight * row["residual"][j] for weight, (_, _, row) in zip(weights, nearest))
        for j in range(3)
    )
    return residual, nearest[0][0]  # type: ignore[return-value]


def _knn_residual(
    task: dict[str, Any],
    x: tuple[float, float],
    history: Sequence[dict[str, Any]],
    k: int,
) -> tuple[float, float, float]:
    residual, _ = _knn_residual_with_distance(task, x, history, k)
    return residual


def _metrics(
    truth: Sequence[tuple[float, float, float]],
    predicted: Sequence[tuple[float, float, float]],
) -> dict[str, Any]:
    if not truth or len(truth) != len(predicted):
        raise ValueError("truth and prediction arrays must be non-empty and aligned")
    objective_abs = [
        abs(predicted[i][objective] - truth[i][objective])
        for i in range(len(truth)) for objective in (0, 1)
    ]
    margin_abs = [abs(predicted[i][2] - truth[i][2]) for i in range(len(truth))]
    pred_feasible = [row[2] >= 0.0 for row in predicted]
    true_feasible = [row[2] >= 0.0 for row in truth]
    false_feasible = sum(pred and not actual for pred, actual in zip(pred_feasible, true_feasible))
    true_infeasible = sum(not value for value in true_feasible)
    false_infeasible = sum(not pred and actual for pred, actual in zip(pred_feasible, true_feasible))
    correct = sum(pred == actual for pred, actual in zip(pred_feasible, true_feasible))
    tp = sum(pred and actual for pred, actual in zip(pred_feasible, true_feasible))
    fp = false_feasible
    fn = false_infeasible
    f1_denominator = 2 * tp + fp + fn
    return {
        "objective_mae": statistics.fmean(objective_abs),
        "objective_f0_mae": statistics.fmean(
            abs(predicted[i][0] - truth[i][0]) for i in range(len(truth))
        ),
        "objective_f1_mae": statistics.fmean(
            abs(predicted[i][1] - truth[i][1]) for i in range(len(truth))
        ),
        "constraint_margin_mae": statistics.fmean(margin_abs),
        "feasibility_accuracy": correct / len(truth),
        "predicted_feasible_count": sum(pred_feasible),
        "true_feasible_count": sum(true_feasible),
        "false_feasible_count": false_feasible,
        "false_feasible_rate_among_predicted_feasible": (
            false_feasible / sum(pred_feasible) if sum(pred_feasible) else None
        ),
        "false_infeasible_count": false_infeasible,
        "false_infeasible_rate_among_true_feasible": (
            false_infeasible / sum(true_feasible) if sum(true_feasible) else None
        ),
        "feasibility_f1": 2 * tp / f1_denominator if f1_denominator else None,
        "audit_sample_count": len(truth),
        "independent_task_unit": "target_task_id; points within task are nested, not independent replicates",
    }


def _pareto_ids(rows: Sequence[tuple[float, float, float]]) -> set[int]:
    feasible = [(index, row) for index, row in enumerate(rows) if row[2] >= 0.0]
    front: set[int] = set()
    for index, row in feasible:
        dominated = any(
            other_index != index
            and other[0] <= row[0] and other[1] <= row[1]
            and (other[0] < row[0] or other[1] < row[1])
            for other_index, other in feasible
        )
        if not dominated:
            front.add(index)
    return front


def _f1(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    precision = len(left & right) / len(left)
    recall = len(left & right) / len(right)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _validate_task_registry() -> None:
    current_suite_sha256 = hashlib.sha256(
        json.dumps(TASK_SUITE, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if current_suite_sha256 != TASK_SUITE_SHA256:
        raise ValueError("task suite changed after its frozen digest was computed")
    all_tasks = (*SOURCE_TASKS, *TARGET_TASKS)
    ids = [task["task_id"] for task in all_tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task IDs must be globally unique")
    source_families = {task["family"] for task in SOURCE_TASKS}
    for task in TARGET_TASKS:
        if task["family"] in source_families:
            raise ValueError("target family overlaps source family; leave-family-out split invalid")
    for task in all_tasks:
        _nonblank(task["task_id"], "task_id")
        _nonblank(task["family"], "family")
        _descriptor(task)


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty trimmed text")
    return value


def run_benchmark(
    *,
    history_points: int = DEFAULT_HISTORY_POINTS,
    test_points: int = DEFAULT_TEST_POINTS,
    nearest_k: int = DEFAULT_NEAREST_K,
    reuse_count: int = DEFAULT_REUSE_COUNT,
) -> dict[str, Any]:
    history_points = _positive_int(history_points, "history_points", maximum=10_000)
    test_points = _positive_int(test_points, "test_points", maximum=10_000)
    nearest_k = _positive_int(nearest_k, "nearest_k", maximum=10_000)
    reuse_count = _positive_int(reuse_count, "reuse_count", maximum=1_000_000)
    # Each target point scans both the labeled archive and shuffled-descriptor control.
    estimated_distance_evaluations = (
        2 * history_points * len(SOURCE_TASKS) * test_points * len(TARGET_TASKS)
    )
    if estimated_distance_evaluations > 25_000_000:
        raise ValueError(
            "2 × history_points × source_tasks × test_points × target_tasks must be <= 25000000"
        )
    _validate_task_registry()

    history: list[dict[str, Any]] = []
    source_pairs = 0
    for task_index, task in enumerate(SOURCE_TASKS):
        desc = _descriptor(task)
        for point_index in range(history_points):
            # History and test design-index ranges are disjoint.
            x = design_point(1 + task_index * history_points + point_index)
            high = high_fidelity(task, x)
            low = low_fidelity(task, x)
            history.append({
                "source_task_id": task["task_id"],
                "source_task_family": task["family"],
                "descriptor": list(desc),
                "x": list(x),
                "residual": [high[j] - low[j] for j in range(3)],
            })
            source_pairs += 1

    shuffled_task_descriptors: dict[str, list[float]] = {}
    for i, task in enumerate(SOURCE_TASKS):
        decoy = SOURCE_TASKS[(i + 1) % len(SOURCE_TASKS)]
        shuffled_task_descriptors[task["task_id"]] = list(decoy["descriptor"])
    shuffled_history = [
        {**row, "descriptor": shuffled_task_descriptors[row["source_task_id"]]}
        for row in history
    ]
    pooled_residual = _mean_vector([tuple(row["residual"]) for row in history])

    archive_cost = source_pairs * (LOW_COST_UNITS + HIGH_COST_UNITS)
    target_low_cost = test_points * LOW_COST_UNITS
    target_high_only_cost = test_points * HIGH_COST_UNITS
    amortized_archive_cost = archive_cost / reuse_count
    transfer_full_cost = target_low_cost + amortized_archive_cost

    target_results = []
    for task_index, task in enumerate(TARGET_TASKS):
        target_descriptor = _descriptor(task)
        points = [design_point(10_001 + task_index * test_points + i) for i in range(test_points)]
        target_low = [low_fidelity(task, x) for x in points]
        # Audit truth is generated only after each prediction below is determined.
        knn_residuals_and_distances = [
            _knn_residual_with_distance(task, x, history, nearest_k) for x in points
        ]
        knn_residuals = [row[0] for row in knn_residuals_and_distances]
        nearest_distances = [row[1] for row in knn_residuals_and_distances]
        knn_predictions = [
            tuple(row[j] + residual[j] for j in range(3))
            for row, residual in zip(target_low, knn_residuals)
        ]
        distance_abstained = [
            distance > ABSTENTION_DISTANCE_THRESHOLD for distance in nearest_distances
        ]
        predictions = {
            "low_fidelity_only": target_low,
            "pooled_residual_transfer": [
                tuple(row[j] + pooled_residual[j] for j in range(3)) for row in target_low
            ],
            "descriptor_x_knn_transfer": knn_predictions,
            "descriptor_x_knn_distance_abstention": knn_predictions,
            "shuffled_descriptor_control": [
                tuple(row[j] + _knn_residual(task, x, shuffled_history, nearest_k)[j] for j in range(3))
                for x, row in zip(points, target_low)
            ],
        }
        truth = [high_fidelity(task, x) for x in points]
        by_policy = {}
        for policy in POLICY_IDS:
            policy_metrics = _metrics(truth, predictions[policy])
            predicted_front = _pareto_ids(predictions[policy])
            true_front = _pareto_ids(truth)
            policy_metrics["pareto_membership_f1"] = _f1(predicted_front, true_front)
            policy_metrics["mean_pareto_front_size_predicted"] = len(predicted_front)
            policy_metrics["mean_pareto_front_size_truth"] = len(true_front)
            if policy == ABSTENTION_POLICY_ID:
                policy_metrics["prediction_coverage_semantics"] = (
                    "underlying KNN scores are diagnostic-only on all points; use selective_prediction for accepted-set metrics"
                )
                covered_indices = [i for i, abstain in enumerate(distance_abstained) if not abstain]
                covered_truth = [truth[i] for i in covered_indices]
                covered_predictions = [predictions[policy][i] for i in covered_indices]
                covered = _metrics(covered_truth, covered_predictions) if covered_indices else None
                policy_metrics["selective_prediction"] = {
                    "method": "fixed nearest archive distance threshold; heuristic OOD guard, not calibrated uncertainty",
                    "nearest_distance_threshold": ABSTENTION_DISTANCE_THRESHOLD,
                    "abstained_count": sum(distance_abstained),
                    "covered_point_count": len(covered_indices),
                    "total_point_count": len(points),
                    "coverage": len(covered_indices) / len(points),
                    "abstention_rate": sum(distance_abstained) / len(points),
                    "covered_objective_mae": covered["objective_mae"] if covered else None,
                    "covered_constraint_margin_mae": covered["constraint_margin_mae"] if covered else None,
                    "covered_feasibility_f1": covered["feasibility_f1"] if covered else None,
                    "covered_false_feasible_count": covered["false_feasible_count"] if covered else 0,
                    "covered_predicted_feasible_count": covered["predicted_feasible_count"] if covered else 0,
                    "covered_false_feasible_rate_among_predicted_feasible": (
                        covered["false_feasible_rate_among_predicted_feasible"] if covered else None
                    ),
                }
            by_policy[policy] = policy_metrics
        target_results.append({
            "target_task_id": task["task_id"],
            "held_out_task_family": task["family"],
            "target_descriptor": list(target_descriptor),
            "design_points_audited": test_points,
            "policy_metrics": by_policy,
        })

    aggregation: dict[str, Any] = {}
    for policy in POLICY_IDS:
        metric_names = (
            "objective_mae", "objective_f0_mae", "objective_f1_mae",
            "constraint_margin_mae", "feasibility_accuracy", "feasibility_f1",
            "pareto_membership_f1",
        )
        aggregate = {
            metric: statistics.fmean(row["policy_metrics"][policy][metric] for row in target_results)
            for metric in metric_names
        }
        aggregate["worst_heldout_task_objective_mae"] = max(
            row["policy_metrics"][policy]["objective_mae"] for row in target_results
        )
        aggregate["worst_heldout_task_margin_mae"] = max(
            row["policy_metrics"][policy]["constraint_margin_mae"] for row in target_results
        )
        aggregate["false_feasible_count_all_target_points"] = sum(
            row["policy_metrics"][policy]["false_feasible_count"] for row in target_results
        )
        aggregate["predicted_feasible_count_all_target_points"] = sum(
            row["policy_metrics"][policy]["predicted_feasible_count"] for row in target_results
        )
        n_pred = aggregate["predicted_feasible_count_all_target_points"]
        aggregate["false_feasible_rate_among_predicted_feasible"] = (
            aggregate["false_feasible_count_all_target_points"] / n_pred if n_pred else None
        )
        if policy == ABSTENTION_POLICY_ID:
            selective_rows = [row["policy_metrics"][policy]["selective_prediction"] for row in target_results]
            covered_predictions = sum(row["covered_predicted_feasible_count"] for row in selective_rows)
            covered_false = sum(row["covered_false_feasible_count"] for row in selective_rows)
            covered_mae = [row["covered_objective_mae"] for row in selective_rows if row["covered_objective_mae"] is not None]
            aggregate["selective_prediction"] = {
                "nearest_distance_threshold": ABSTENTION_DISTANCE_THRESHOLD,
                "macro_coverage_over_target_tasks": statistics.fmean(row["coverage"] for row in selective_rows),
                "macro_covered_objective_mae_over_tasks_with_coverage": statistics.fmean(covered_mae) if covered_mae else None,
                "covered_predicted_feasible_count": covered_predictions,
                "covered_false_feasible_count": covered_false,
                "covered_false_feasible_rate_among_predicted_feasible": covered_false / covered_predictions if covered_predictions else None,
                "claim_ceiling": "Selective synthetic diagnostic only; distance threshold is not a calibrated risk guarantee.",
            }
        aggregate["target_task_units"] = len(target_results)
        aggregate["target_points_nested_within_tasks"] = True
        aggregation[policy] = aggregate

    baseline = aggregation["low_fidelity_only"]
    comparison: dict[str, Any] = {}
    for policy in POLICY_IDS:
        if policy in ("low_fidelity_only", ABSTENTION_POLICY_ID):
            continue
        paired_task_rows = []
        for row in target_results:
            baseline_mae = row["policy_metrics"]["low_fidelity_only"]["objective_mae"]
            candidate_mae = row["policy_metrics"][policy]["objective_mae"]
            paired_task_rows.append({
                "target_task_id": row["target_task_id"],
                "objective_mae_reduction": baseline_mae - candidate_mae,
                "relative_objective_mae_reduction": (
                    (baseline_mae - candidate_mae) / baseline_mae if baseline_mae > 0 else None
                ),
                "constraint_margin_mae_reduction": (
                    row["policy_metrics"]["low_fidelity_only"]["constraint_margin_mae"]
                    - row["policy_metrics"][policy]["constraint_margin_mae"]
                ),
                "pareto_membership_f1_delta": (
                    row["policy_metrics"][policy]["pareto_membership_f1"]
                    - row["policy_metrics"]["low_fidelity_only"]["pareto_membership_f1"]
                ),
            })
        defined_relative = [
            row["relative_objective_mae_reduction"] for row in paired_task_rows
            if row["relative_objective_mae_reduction"] is not None
        ]
        comparison[policy] = {
            "paired_target_task_units": len(paired_task_rows),
            "mean_objective_mae_reduction": statistics.fmean(row["objective_mae_reduction"] for row in paired_task_rows),
            "mean_relative_objective_mae_reduction": (
                statistics.fmean(defined_relative) if defined_relative else None
            ),
            "relative_effect_defined_task_units": len(defined_relative),
            "target_tasks_with_lower_objective_mae": sum(row["objective_mae_reduction"] > 0 for row in paired_task_rows),
            "target_tasks_with_tied_objective_mae": sum(abs(row["objective_mae_reduction"]) <= 1e-12 for row in paired_task_rows),
            "target_tasks_with_higher_objective_mae": sum(row["objective_mae_reduction"] < 0 for row in paired_task_rows),
            "paired_task_results": paired_task_rows,
        }

    # Report family-macro metrics separately from task-macro metrics. Four
    # parameter variants inside one held-out family are not four independent
    # demonstrations of cross-family generalization.
    target_families = sorted({row["held_out_task_family"] for row in target_results})
    rows_by_family = {
        family: [row for row in target_results if row["held_out_task_family"] == family]
        for family in target_families
    }
    family_aggregation: dict[str, Any] = {}
    family_comparisons: dict[str, Any] = {}
    for policy in POLICY_IDS:
        family_metrics: dict[str, Any] = {}
        for family, family_rows in rows_by_family.items():
            family_metrics[family] = {
                metric: statistics.fmean(
                    row["policy_metrics"][policy][metric] for row in family_rows
                )
                for metric in (
                    "objective_mae", "objective_f0_mae", "objective_f1_mae",
                    "constraint_margin_mae", "feasibility_accuracy", "feasibility_f1",
                    "pareto_membership_f1",
                )
            }
            family_metrics[family]["task_count_nested_within_family"] = len(family_rows)
            if policy == ABSTENTION_POLICY_ID:
                selective_rows = [row["policy_metrics"][policy]["selective_prediction"] for row in family_rows]
                covered_n = sum(row["covered_predicted_feasible_count"] for row in selective_rows)
                covered_false = sum(row["covered_false_feasible_count"] for row in selective_rows)
                covered_mae = [row["covered_objective_mae"] for row in selective_rows if row["covered_objective_mae"] is not None]
                family_metrics[family]["selective_prediction"] = {
                    "macro_coverage_over_task_variants": statistics.fmean(row["coverage"] for row in selective_rows),
                    "macro_covered_objective_mae_over_variants_with_coverage": statistics.fmean(covered_mae) if covered_mae else None,
                    "covered_predicted_feasible_count": covered_n,
                    "covered_false_feasible_count": covered_false,
                    "covered_false_feasible_rate_among_predicted_feasible": covered_false / covered_n if covered_n else None,
                    "nearest_distance_threshold": ABSTENTION_DISTANCE_THRESHOLD,
                }
        family_aggregation[policy] = {
            "family_count": len(family_metrics),
            "family_metrics": family_metrics,
            "macro_mean_over_families": {
                metric: statistics.fmean(row[metric] for row in family_metrics.values())
                for metric in (
                    "objective_mae", "objective_f0_mae", "objective_f1_mae",
                    "constraint_margin_mae", "feasibility_accuracy", "feasibility_f1",
                    "pareto_membership_f1",
                )
            },
            "worst_family_objective_mae": max(
                row["objective_mae"] for row in family_metrics.values()
            ),
            "independent_generalization_unit": "held_out_task_family; task variants are nested",
        }

        if policy in ("low_fidelity_only", ABSTENTION_POLICY_ID):
            continue
        by_family = {}
        for family, family_rows in rows_by_family.items():
            baseline_mae = statistics.fmean(
                row["policy_metrics"]["low_fidelity_only"]["objective_mae"]
                for row in family_rows
            )
            candidate_mae = statistics.fmean(
                row["policy_metrics"][policy]["objective_mae"] for row in family_rows
            )
            reduction = baseline_mae - candidate_mae
            by_family[family] = {
                "task_count_nested_within_family": len(family_rows),
                "baseline_mean_objective_mae": baseline_mae,
                "candidate_mean_objective_mae": candidate_mae,
                "objective_mae_reduction": reduction,
                "relative_objective_mae_reduction": (
                    reduction / baseline_mae if baseline_mae > 0 else None
                ),
            }
        defined_family_effects = [
            row["relative_objective_mae_reduction"] for row in by_family.values()
            if row["relative_objective_mae_reduction"] is not None
        ]
        family_comparisons[policy] = {
            "held_out_family_units": len(by_family),
            "independent_generalization_unit": "held_out_task_family",
            "mean_relative_objective_mae_reduction_macro_over_families": (
                statistics.fmean(defined_family_effects) if defined_family_effects else None
            ),
            "bootstrap_method": "percentile bootstrap over held-out family effects; task variants remain nested",
            "bootstrap_replicates": FAMILY_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": FAMILY_BOOTSTRAP_SEED,
            "bootstrap_95pct_ci_macro_mean_relative_objective_mae_reduction": _bootstrap_mean_interval(
                defined_family_effects
            ),
            "families_with_lower_objective_mae": sum(
                row["objective_mae_reduction"] > 0 for row in by_family.values()
            ),
            "families_with_tied_objective_mae": sum(
                abs(row["objective_mae_reduction"]) <= 1e-12 for row in by_family.values()
            ),
            "families_with_higher_objective_mae": sum(
                row["objective_mae_reduction"] < 0 for row in by_family.values()
            ),
            "by_family": by_family,
            "independent_generalization_unit": "held_out_task_family",
        }

    policy_configuration = {
        "policy_ids": list(POLICY_IDS),
        "nearest_k": nearest_k,
        "distance_abstention": {
            "policy_id": ABSTENTION_POLICY_ID,
            "nearest_distance_threshold": ABSTENTION_DISTANCE_THRESHOLD,
            "decision_rule": "abstain when nearest normalized task-descriptor/design distance to source archive exceeds threshold",
            "calibration_status": "fixed heuristic threshold; not calibrated probability or safety bound",
        },
    }
    policy_config_sha256 = hashlib.sha256(
        json.dumps(policy_configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "evidence_class": "synthetic",
        "receipt_label": "HISTORICAL_TRANSFER_SYNTHETIC_PREDICTION_AUDIT_NOT_OPTIMIZER_SOTA_EVIDENCE",
        "target_high_fidelity_used_by_predictor": False,
        "split": "leave_task_family_out",
        "source_task_count": len(SOURCE_TASKS),
        "target_task_count": len(TARGET_TASKS),
        "target_family_count": len({task["family"] for task in TARGET_TASKS}),
        "task_variants_per_target_family": {
            family: sum(task["family"] == family for task in TARGET_TASKS)
            for family in sorted({task["family"] for task in TARGET_TASKS})
        },
        "history_points_per_source_task": history_points,
        "history_pair_count": source_pairs,
        "target_points_per_task": test_points,
        "nearest_k": nearest_k,
        "historical_archive_sha256": hashlib.sha256(
            json.dumps(history, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "task_suite_sha256": TASK_SUITE_SHA256,
        "policy_ids": list(POLICY_IDS),
        "policy_configuration": policy_configuration,
        "policy_config_sha256": policy_config_sha256,
        "cost_accounting": {
            "low_fidelity_cost_units": LOW_COST_UNITS,
            "high_fidelity_cost_units": HIGH_COST_UNITS,
            "source_archive_pair_count": source_pairs,
            "archive_build_cost_units": archive_cost,
            "reuse_count_assumption": reuse_count,
            "amortized_archive_cost_per_target_task_units": amortized_archive_cost,
            "target_low_evaluation_cost_per_task_units": target_low_cost,
            "target_high_only_evaluation_cost_per_task_units": target_high_only_cost,
            "transfer_full_cost_per_target_task_units": transfer_full_cost,
            "target_truth_audit_cost_units_excluded_from_policy_cost": target_high_only_cost,
            "cost_assumptions": "Illustrative normalized units. Includes one low+high source pair per archive point and the target low scan; excludes target audit truth from policy costs. Replace with measured solver/runtime/hardware/engineering costs for real comparisons.",
        },
        "aggregate_metrics_macro_averaged_over_target_tasks": aggregation,
        "aggregate_metrics_macro_averaged_over_target_families": family_aggregation,
        "comparisons_vs_low_fidelity_only": comparison,
        "comparisons_vs_low_fidelity_only_macro_by_target_family": family_comparisons,
        "target_task_results": target_results,
        "claim_ceiling": (
            "Deterministic synthetic prediction-transfer audit over 32 task variants in eight held-out families. "
            "Family-macro summaries avoid treating variants as independent evidence of cross-family generalization. "
            "Eight synthetic families remain insufficient for broad SOTA or physical claims. The distance-abstention "
            "policy is an uncalibrated defer heuristic. Does not benchmark an optimizer, prove descriptor availability, "
            "show transfer superiority on physical casting, calibrate uncertainty, or certify feasibility."
        ),
        "source_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant forbidden: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-points", type=int, default=DEFAULT_HISTORY_POINTS)
    parser.add_argument("--test-points", type=int, default=DEFAULT_TEST_POINTS)
    parser.add_argument("--nearest-k", type=int, default=DEFAULT_NEAREST_K)
    parser.add_argument("--reuse-count", type=int, default=DEFAULT_REUSE_COUNT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        out = args.output.resolve()
        if out == Path(__file__).resolve():
            raise ValueError("output must not overwrite the source module")
        receipt = run_benchmark(
            history_points=args.history_points,
            test_points=args.test_points,
            nearest_k=args.nearest_k,
            reuse_count=args.reuse_count,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"historical transfer benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(f"historical transfer receipt written: {out}")
    print(f"source_tasks={receipt['source_task_count']} target_tasks={receipt['target_task_count']}")
    print(f"history_pairs={receipt['history_pair_count']} suite_sha256={receipt['task_suite_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
