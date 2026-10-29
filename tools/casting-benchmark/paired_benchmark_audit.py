#!/usr/bin/env python3
"""Paired, matched-budget analysis for casting optimization runs.

Uses only the Python standard library. Sampling uncertainty is computed over the
manifest's declared independent unit_id values, never over candidate or shift rows.
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
BENCHMARK_ID = "CASTING-PAIRED-COST-001"
TOP_FIELDS = frozenset({
    "schema_version", "benchmark_id", "protocol_sha256", "normalization_id",
    "baseline_policy", "quality_target", "sampling_unit", "independence_note",
    "bootstrap_replicates", "bootstrap_seed", "runs",
})
RUN_FIELDS = frozenset({
    "unit_id", "task_family", "constraint_family", "dimension", "severity",
    "seed", "policy_id", "budget_units", "cost_spent_units", "completed",
    "decision", "curve", "shift_results", "run_receipt_sha256",
})
CURVE_FIELDS = frozenset({"cost_units", "quality"})
SHIFT_FIELDS = frozenset({
    "shift_id", "quality", "false_feasible_count", "decision_count",
})
PAIR_META_FIELDS = (
    "task_family", "constraint_family", "dimension", "severity", "seed", "budget_units"
)
TOL = 1e-12


def _strict_object(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _quality(value: object, label: str) -> float:
    number = _number(value, label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1] under the frozen normalization")
    return number


def _curve_auc(curve: list[dict[str, Any]], budget: float) -> tuple[float, float, float | None]:
    """Return budget-normalized trapezoidal AUC, terminal score, and target cost."""
    area = 0.0
    for left, right in zip(curve, curve[1:]):
        area += (right["cost_units"] - left["cost_units"]) * (
            left["quality"] + right["quality"]
        ) / 2.0
    last = curve[-1]
    area += max(0.0, budget - last["cost_units"]) * last["quality"]
    auc = area / budget
    target_cost = None
    # The target is attached by caller for the threshold scan; compute separately below.
    return auc, last["quality"], target_cost


def _cost_to_target(curve: list[dict[str, Any]], target: float) -> float | None:
    if curve[0]["quality"] >= target:
        return 0.0
    for left, right in zip(curve, curve[1:]):
        if right["quality"] >= target:
            if right["quality"] == left["quality"]:
                return float(right["cost_units"])
            fraction = (target - left["quality"]) / (right["quality"] - left["quality"])
            return left["cost_units"] + fraction * (right["cost_units"] - left["cost_units"])
    return None


def validate_manifest(document: Any) -> dict[str, Any]:
    doc = _strict_object(document, TOP_FIELDS, "manifest")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if doc["benchmark_id"] != BENCHMARK_ID:
        raise ValueError(f"benchmark_id must be {BENCHMARK_ID}")
    _sha256(doc["protocol_sha256"], "protocol_sha256")
    _nonblank(doc["normalization_id"], "normalization_id")
    _nonblank(doc["baseline_policy"], "baseline_policy")
    _nonblank(doc["sampling_unit"], "sampling_unit")
    _nonblank(doc["independence_note"], "independence_note")
    target = _quality(doc["quality_target"], "quality_target")
    if target <= 0.0:
        raise ValueError("quality_target must be greater than zero")
    replicates = _integer(doc["bootstrap_replicates"], "bootstrap_replicates", minimum=100)
    if replicates > 100_000:
        raise ValueError("bootstrap_replicates must be <= 100000")
    seed = _integer(doc["bootstrap_seed"], "bootstrap_seed")
    runs = doc["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a non-empty array")

    by_unit: dict[str, dict[str, dict[str, Any]]] = {}
    policies: set[str] | None = None
    shared_meta: dict[str, tuple[Any, ...]] = {}
    for index, raw in enumerate(runs):
        run = _strict_object(raw, RUN_FIELDS, f"runs[{index}]")
        unit_id = _nonblank(run["unit_id"], f"runs[{index}].unit_id")
        task_family = _nonblank(run["task_family"], f"runs[{index}].task_family")
        constraint_family = _nonblank(run["constraint_family"], f"runs[{index}].constraint_family")
        policy_id = _nonblank(run["policy_id"], f"runs[{index}].policy_id")
        _sha256(run["run_receipt_sha256"], f"runs[{index}].run_receipt_sha256")
        dimension = _integer(run["dimension"], f"runs[{index}].dimension", minimum=1)
        _integer(run["seed"], f"runs[{index}].seed")
        severity = _number(run["severity"], f"runs[{index}].severity", minimum=0.0)
        budget = _number(run["budget_units"], f"runs[{index}].budget_units", minimum=1.0)
        spent = _number(run["cost_spent_units"], f"runs[{index}].cost_spent_units", minimum=0.0)
        if not isinstance(run["completed"], bool):
            raise ValueError(f"runs[{index}].completed must be boolean")
        if run["decision"] not in {"recommend", "abstain"}:
            raise ValueError(f"runs[{index}].decision must be recommend or abstain")
        if not math.isfinite(float(budget)) or budget <= 0.0:
            raise ValueError(f"runs[{index}].budget_units must be positive and finite")

        curve = run["curve"]
        if not isinstance(curve, list) or not curve:
            raise ValueError(f"runs[{index}].curve must be a non-empty array")
        normalized_curve: list[dict[str, float]] = []
        previous_cost = -1.0
        for c_index, raw_point in enumerate(curve):
            point = _strict_object(raw_point, CURVE_FIELDS, f"runs[{index}].curve[{c_index}]")
            cost = _number(point["cost_units"], f"runs[{index}].curve[{c_index}].cost_units", minimum=0.0)
            quality = _quality(point["quality"], f"runs[{index}].curve[{c_index}].quality")
            if cost <= previous_cost:
                raise ValueError(f"runs[{index}].curve costs must be strictly increasing")
            if cost > budget + TOL:
                raise ValueError(f"runs[{index}].curve cost exceeds the common budget")
            if cost > spent + TOL:
                raise ValueError(f"runs[{index}].curve point exceeds realized cost")
            normalized_curve.append({"cost_units": cost, "quality": quality})
            previous_cost = cost
        if normalized_curve[0]["cost_units"] != 0.0:
            raise ValueError(f"runs[{index}].curve must start at cost zero")

        shifts = run["shift_results"]
        if not isinstance(shifts, list) or not shifts:
            raise ValueError(f"runs[{index}].shift_results must be a non-empty array")
        shift_map: dict[str, dict[str, Any]] = {}
        for s_index, raw_shift in enumerate(shifts):
            shift = _strict_object(raw_shift, SHIFT_FIELDS, f"runs[{index}].shift_results[{s_index}]")
            shift_id = _nonblank(shift["shift_id"], f"runs[{index}].shift_results[{s_index}].shift_id")
            if shift_id in shift_map:
                raise ValueError(f"runs[{index}] has duplicate shift_id: {shift_id}")
            quality = _quality(shift["quality"], f"shift {shift_id}.quality")
            false_count = _integer(shift["false_feasible_count"], f"shift {shift_id}.false_feasible_count")
            decision_count = _integer(shift["decision_count"], f"shift {shift_id}.decision_count")
            if false_count > decision_count:
                raise ValueError(f"shift {shift_id}: false_feasible_count exceeds decision_count")
            if run["decision"] == "abstain":
                if decision_count != 0 or false_count != 0 or quality != 0.0:
                    raise ValueError(f"abstaining run must have zero shift quality and zero decision counts ({shift_id})")
            elif decision_count == 0:
                raise ValueError(f"recommended run must have at least one audited decision per shift ({shift_id})")
            shift_map[shift_id] = {
                "quality": quality,
                "false_feasible_count": false_count,
                "decision_count": decision_count,
            }

        meta = (task_family, constraint_family, dimension, severity, int(run["seed"]), budget)
        if unit_id in by_unit and policy_id in by_unit[unit_id]:
            raise ValueError(f"duplicate run for unit_id={unit_id}, policy_id={policy_id}")
        if unit_id in shared_meta and shared_meta[unit_id] != meta:
            raise ValueError(f"paired runs do not match scenario/budget metadata for unit {unit_id}")
        shared_meta[unit_id] = meta
        if unit_id not in by_unit:
            by_unit[unit_id] = {}
        by_unit[unit_id][policy_id] = {
            **run,
            "unit_id": unit_id,
            "policy_id": policy_id,
            "budget_units": budget,
            "cost_spent_units": spent,
            "curve": normalized_curve,
            "shift_map": shift_map,
        }
        this_policies = set(by_unit[unit_id])
        if policies is None:
            policies = this_policies
        # Verify global policy identity consistency after the loop.

    assert policies is not None
    if doc["baseline_policy"] not in policies:
        raise ValueError("baseline_policy is absent from run manifest")
    expected_shift_set: set[str] | None = None
    for unit_id in sorted(by_unit):
        unit_runs = by_unit[unit_id]
        if set(unit_runs) != policies:
            raise ValueError(
                f"policy coverage mismatch for unit {unit_id}; "
                f"missing={sorted(policies - set(unit_runs))}, extra={sorted(set(unit_runs) - policies)}"
            )
        for policy_id, run in unit_runs.items():
            shift_ids = set(run["shift_map"])
            if expected_shift_set is None:
                expected_shift_set = shift_ids
            elif shift_ids != expected_shift_set:
                raise ValueError(f"shift coverage mismatch for unit {unit_id}, policy {policy_id}")
    if len(by_unit) < 2:
        raise ValueError("at least two declared independent units are required for uncertainty reporting")
    if len(policies) < 2:
        raise ValueError("at least two policies are required for a comparison")

    return {
        **doc,
        "quality_target": target,
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "policy_ids": sorted(policies),
        "units": by_unit,
        "shift_ids": sorted(expected_shift_set or set()),
    }


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile for an empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return sorted_values[left]
    fraction = position - left
    return sorted_values[left] * (1.0 - fraction) + sorted_values[right] * fraction


def _bootstrap_interval(
    values: list[float],
    *,
    replicates: int,
    rng: random.Random,
    statistic: str,
) -> list[float] | None:
    if not values:
        return None
    estimates: list[float] = []
    count = len(values)
    for _ in range(replicates):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        estimate = statistics.mean(sample) if statistic == "mean" else statistics.median(sample)
        estimates.append(float(estimate))
    estimates.sort()
    return [_quantile(estimates, 0.025), _quantile(estimates, 0.975)]


def _paired_effect(
    differences: list[float],
    *,
    replicates: int,
    rng: random.Random,
    tie_tolerance: float = TOL,
) -> dict[str, Any]:
    if not differences:
        return {
            "n_paired_units": 0, "mean_delta": None, "median_delta": None,
            "wins": 0, "ties": 0, "losses": 0, "win_rate_excluding_ties": None,
            "bootstrap_95pct_ci_mean_delta": None, "bootstrap_95pct_ci_median_delta": None,
        }
    wins = sum(value > tie_tolerance for value in differences)
    losses = sum(value < -tie_tolerance for value in differences)
    ties = len(differences) - wins - losses
    n_not_tied = wins + losses
    ordered = sorted(differences)
    return {
        "n_paired_units": len(differences),
        "mean_delta": statistics.mean(differences),
        "median_delta": statistics.median(differences),
        "wins": wins, "ties": ties, "losses": losses,
        "win_rate_excluding_ties": wins / n_not_tied if n_not_tied else None,
        "bootstrap_95pct_ci_mean_delta": _bootstrap_interval(
            differences, replicates=replicates, rng=rng, statistic="mean"
        ),
        "bootstrap_95pct_ci_median_delta": _bootstrap_interval(
            differences, replicates=replicates, rng=rng, statistic="median"
        ),
    }


def _run_metrics(run: dict[str, Any], target: float) -> dict[str, Any]:
    auc, final_quality, _ = _curve_auc(run["curve"], run["budget_units"])
    cost_to_target = _cost_to_target(run["curve"], target)
    shift_qualities = [run["shift_map"][shift]["quality"] for shift in sorted(run["shift_map"])]
    false_count = sum(run["shift_map"][s]["false_feasible_count"] for s in run["shift_map"])
    decision_count = sum(run["shift_map"][s]["decision_count"] for s in run["shift_map"])
    return {
        "cost_normalized_quality_auc": auc,
        "observed_final_quality": final_quality,
        "reached_quality_target": cost_to_target is not None,
        "cost_to_target_units": cost_to_target,
        "mean_shift_quality": statistics.mean(shift_qualities),
        "worst_shift_quality": min(shift_qualities),
        "false_feasible_count": false_count,
        "audited_decision_count": decision_count,
        "false_feasible_rate": false_count / decision_count if decision_count else None,
        "abstained": run["decision"] == "abstain",
        "completed": run["completed"],
        "over_budget": run["cost_spent_units"] > run["budget_units"] + TOL,
        "realized_cost_units": run["cost_spent_units"],
    }


def analyze_manifest(document: Any) -> dict[str, Any]:
    manifest = validate_manifest(document)
    units: dict[str, dict[str, dict[str, Any]]] = manifest["units"]
    policies = manifest["policy_ids"]
    baseline = manifest["baseline_policy"]
    target = manifest["quality_target"]
    reps = manifest["bootstrap_replicates"]
    rng = random.Random(manifest["bootstrap_seed"])
    metrics: dict[str, dict[str, dict[str, Any]]] = {
        unit_id: {
            policy_id: _run_metrics(run, target)
            for policy_id, run in unit_runs.items()
        }
        for unit_id, unit_runs in units.items()
    }

    policy_summaries: dict[str, Any] = {}
    for policy_id in policies:
        rows = [metrics[unit_id][policy_id] for unit_id in sorted(metrics)]
        hits = [row["cost_to_target_units"] for row in rows if row["reached_quality_target"]]
        policy_summaries[policy_id] = {
            "n_independent_units": len(rows),
            "median_cost_normalized_quality_auc": statistics.median(
                row["cost_normalized_quality_auc"] for row in rows
            ),
            "mean_cost_normalized_quality_auc": statistics.mean(
                row["cost_normalized_quality_auc"] for row in rows
            ),
            "target_hit_count": len(hits),
            "target_hit_rate": len(hits) / len(rows),
            "median_cost_to_target_among_hits": statistics.median(hits) if hits else None,
            "mean_shift_quality": statistics.mean(row["mean_shift_quality"] for row in rows),
            "median_worst_shift_quality": statistics.median(row["worst_shift_quality"] for row in rows),
            "abstention_count": sum(row["abstained"] for row in rows),
            "abstention_rate": sum(row["abstained"] for row in rows) / len(rows),
            "incomplete_count": sum(not row["completed"] for row in rows),
            "over_budget_count": sum(row["over_budget"] for row in rows),
            "median_realized_cost_units": statistics.median(row["realized_cost_units"] for row in rows),
            "false_feasible_events": sum(row["false_feasible_count"] for row in rows),
            "audited_decisions": sum(row["audited_decision_count"] for row in rows),
            "false_feasible_rate": (
                sum(row["false_feasible_count"] for row in rows)
                / sum(row["audited_decision_count"] for row in rows)
                if sum(row["audited_decision_count"] for row in rows) else None
            ),
        }

    comparisons: dict[str, Any] = {}
    for policy_id in policies:
        if policy_id == baseline:
            continue
        per_unit: dict[str, dict[str, float | None]] = {}
        auc_deltas: list[float] = []
        relative_auc: list[float] = []
        mean_shift_deltas: list[float] = []
        worst_shift_deltas: list[float] = []
        target_hit_deltas: list[float] = []
        cost_reductions: list[float] = []
        false_rate_deltas: list[float] = []
        false_rate_excluded = 0

        for unit_id in sorted(units):
            candidate = metrics[unit_id][policy_id]
            reference = metrics[unit_id][baseline]
            delta_auc = candidate["cost_normalized_quality_auc"] - reference["cost_normalized_quality_auc"]
            auc_deltas.append(delta_auc)
            if reference["cost_normalized_quality_auc"] > 0.0:
                relative_auc.append(delta_auc / reference["cost_normalized_quality_auc"])
            mean_shift_deltas.append(candidate["mean_shift_quality"] - reference["mean_shift_quality"])
            worst_shift_deltas.append(candidate["worst_shift_quality"] - reference["worst_shift_quality"])
            target_hit_deltas.append(
                float(candidate["reached_quality_target"]) - float(reference["reached_quality_target"])
            )
            if candidate["reached_quality_target"] and reference["reached_quality_target"]:
                base_cost = reference["cost_to_target_units"]
                candidate_cost = candidate["cost_to_target_units"]
                if base_cost > 0.0:
                    cost_reductions.append((base_cost - candidate_cost) / base_cost)
            if candidate["false_feasible_rate"] is None or reference["false_feasible_rate"] is None:
                false_rate_excluded += 1
            else:
                false_rate_deltas.append(reference["false_feasible_rate"] - candidate["false_feasible_rate"])
            per_unit[unit_id] = {
                "auc_delta_candidate_minus_baseline": delta_auc,
                "relative_auc_improvement": (
                    delta_auc / reference["cost_normalized_quality_auc"]
                    if reference["cost_normalized_quality_auc"] > 0.0 else None
                ),
                "mean_shift_quality_delta_candidate_minus_baseline": (
                    candidate["mean_shift_quality"] - reference["mean_shift_quality"]
                ),
                "worst_shift_quality_delta_candidate_minus_baseline": (
                    candidate["worst_shift_quality"] - reference["worst_shift_quality"]
                ),
            }

        # Stable per-policy/metric RNG order is part of deterministic receipt reproduction.
        effects = {
            "cost_normalized_quality_auc_delta": _paired_effect(auc_deltas, replicates=reps, rng=rng),
            "relative_auc_improvement": _paired_effect(relative_auc, replicates=reps, rng=rng),
            "target_hit_rate_delta": _paired_effect(target_hit_deltas, replicates=reps, rng=rng),
            "mean_shift_quality_delta": _paired_effect(mean_shift_deltas, replicates=reps, rng=rng),
            "worst_shift_quality_delta": _paired_effect(worst_shift_deltas, replicates=reps, rng=rng),
            "false_feasible_rate_reduction": _paired_effect(false_rate_deltas, replicates=reps, rng=rng),
            "cost_to_target_reduction_when_both_reach": _paired_effect(cost_reductions, replicates=reps, rng=rng),
        }
        per_shift: dict[str, Any] = {}
        for shift_id in manifest["shift_ids"]:
            deltas = [
                units[unit_id][policy_id]["shift_map"][shift_id]["quality"]
                - units[unit_id][baseline]["shift_map"][shift_id]["quality"]
                for unit_id in sorted(units)
            ]
            per_shift[shift_id] = _paired_effect(deltas, replicates=reps, rng=rng)
        per_family: dict[str, Any] = {}
        families = sorted({units[unit_id][baseline]["task_family"] for unit_id in units})
        for family in families:
            family_ids = [
                unit_id for unit_id in sorted(units)
                if units[unit_id][baseline]["task_family"] == family
            ]
            family_deltas = [
                metrics[unit_id][policy_id]["cost_normalized_quality_auc"]
                - metrics[unit_id][baseline]["cost_normalized_quality_auc"]
                for unit_id in family_ids
            ]
            family_worst_deltas = [
                metrics[unit_id][policy_id]["worst_shift_quality"]
                - metrics[unit_id][baseline]["worst_shift_quality"]
                for unit_id in family_ids
            ]
            per_family[family] = {
                "unit_count": len(family_ids),
                "auc_delta": _paired_effect(family_deltas, replicates=reps, rng=rng),
                "worst_shift_quality_delta": _paired_effect(family_worst_deltas, replicates=reps, rng=rng),
            }
        comparisons[policy_id] = {
            "baseline_policy": baseline,
            "candidate_policy": policy_id,
            "independent_unit_count": len(units),
            "relative_auc_denominator_units": len(relative_auc),
            "auc_pairs_with_zero_baseline_excluded_from_relative_improvement": len(units) - len(relative_auc),
            "false_feasible_pairs_excluded_due_to_zero_decisions": false_rate_excluded,
            "effects": effects,
            "by_shift": per_shift,
            "by_task_family": per_family,
            "per_unit": per_unit,
            "interpretation": (
                "Positive deltas favor candidate for quality metrics. "
                "False-feasible-rate reduction is baseline rate minus candidate rate; positive favors candidate. "
                "Cost-to-target reduction is (baseline cost - candidate cost) / baseline cost; positive favors candidate. "
                "Bootstrap intervals resample declared independent unit_id records and do not prove independence."
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "receipt_label": "PAIRED_MATCHED_COST_COMPARISON_NOT_PHYSICAL_CERTIFICATION",
        "baseline_policy": baseline,
        "policy_ids": policies,
        "unit_count": len(units),
        "shift_ids": manifest["shift_ids"],
        "quality_target": target,
        "normalization_id": manifest["normalization_id"],
        "sampling_unit": manifest["sampling_unit"],
        "independence_note": manifest["independence_note"],
        "bootstrap": {
            "method": "percentile bootstrap over paired unit_id values",
            "replicates": reps,
            "seed": manifest["bootstrap_seed"],
            "confidence_level": 0.95,
            "independent_unit_assumption": "declared_by_manifest_not_verified_by_evaluator",
        },
        "per_policy": policy_summaries,
        "comparisons_vs_baseline": comparisons,
        "source_artifacts": {
            "manifest_sha256": hashlib.sha256(canonical_json(document)).hexdigest(),
            "raw_manifest_sha256": None,
        },
        "claim_ceiling": (
            "Paired descriptive benchmark analysis on supplied run summaries. "
            "Does not establish that runs are independent, source receipts truthful, "
            "the quality normalization appropriate, baseline implementations reproduced fairly, "
            "or any synthetic result valid for physical casting."
        ),
    }


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="frozen paired run manifest JSON")
    parser.add_argument("--output", type=Path, required=True, help="receipt output path")
    args = parser.parse_args(argv)
    try:
        source_path = args.manifest.resolve()
        output_path = args.output.resolve()
        if source_path == output_path:
            raise ValueError("receipt output must not overwrite the input manifest")
        raw = source_path.read_bytes()
        try:
            document = json.loads(
                raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("manifest is not valid strict UTF-8 JSON") from exc
        receipt = analyze_manifest(document)
        receipt["source_artifacts"]["raw_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        receipt["evaluator_sha256"] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"paired benchmark analysis failed: {exc}", file=sys.stderr)
        return 2
    print(f"paired benchmark receipt written: {output_path}")
    print(f"manifest_sha256={receipt['source_artifacts']['raw_manifest_sha256']}")
    print(f"unit_count={receipt['unit_count']}")
    print(f"comparison_count={len(receipt['comparisons_vs_baseline'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
