#!/usr/bin/env python3
"""Post-selection robustness audit for multi-objective casting decisions.

This evaluator scores a frozen nominal candidate set against independent,
post-selection high-fidelity observations from named shift scenarios. It is a
research audit, not a proof of holdout independence or physical validity.
Uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

BENCHMARK_ID = "CASTING-DECISION-ROBUSTNESS-001"
SCHEMA_VERSION = "1.1.0"
TOL = 1e-12


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, not {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys mismatch; missing={missing}, extra={extra}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def hypervolume_2d(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """Exact 2-D minimization hypervolume for points inside the reference box."""
    ref_x, ref_y = reference
    unique = sorted(set(points))
    front: list[tuple[float, float]] = []
    best_y = math.inf
    for x, y in unique:
        if x < 0.0 or y < 0.0 or x >= ref_x or y >= ref_y:
            raise ValueError("objectives must lie inside the fixed [0, reference) box")
        if y < best_y:
            front.append((x, y))
            best_y = y
    area = 0.0
    for i, (x, y) in enumerate(front):
        next_x = front[i + 1][0] if i + 1 < len(front) else ref_x
        area += max(0.0, next_x - x) * max(0.0, ref_y - y)
    if not math.isfinite(area):
        raise ValueError("hypervolume is non-finite")
    return area


def _objectives(value: Any, label: str, reference: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two objective values")
    values = tuple(_finite_number(v, f"{label}[{i}]") for i, v in enumerate(value))
    for i, v in enumerate(values):
        if v < 0.0 or v >= reference[i]:
            raise ValueError(f"{label}[{i}] must be in [0, reference_point[{i}])")
    return values  # type: ignore[return-value]


def validate_document(document: Any) -> dict[str, Any]:
    required = {
        "schema_version", "benchmark_id", "task_id", "run_id", "split",
        "reference_point", "candidate_set", "holdout_shifts", "source_run",
    }
    doc = _require_exact_keys(document, required, "document")
    if doc["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if doc["benchmark_id"] != BENCHMARK_ID:
        raise ValueError(f"benchmark_id must be {BENCHMARK_ID}")
    source_run = _require_exact_keys(
        doc["source_run"],
        {
            "sweep_sha256", "adapter_sha256", "candidate_pool_sha256", "objective_family",
            "constraint_family", "dimension", "severity", "seed", "policy",
            "fidelity_costs", "cost_budget", "cost_spent",
        },
        "source_run",
    )
    _sha256(source_run["sweep_sha256"], "source_run.sweep_sha256")
    _sha256(source_run["adapter_sha256"], "source_run.adapter_sha256")
    _sha256(source_run["candidate_pool_sha256"], "source_run.candidate_pool_sha256")
    for field in ("objective_family", "constraint_family", "policy"):
        if not isinstance(source_run[field], str) or not source_run[field].strip():
            raise ValueError(f"source_run.{field} must be a non-empty string")
    for field in ("dimension", "seed"):
        if isinstance(source_run[field], bool) or not isinstance(source_run[field], int) or source_run[field] < 0:
            raise ValueError(f"source_run.{field} must be a non-negative integer")
    if source_run["dimension"] < 2:
        raise ValueError("source_run.dimension must be at least 2")
    _finite_number(source_run["severity"], "source_run.severity")
    if source_run["severity"] < 0:
        raise ValueError("source_run.severity must be non-negative")
    fidelity_costs = _require_exact_keys(
        source_run["fidelity_costs"], {"low", "medium", "high"}, "source_run.fidelity_costs"
    )
    for level, value in fidelity_costs.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"source_run.fidelity_costs.{level} must be a positive integer")
    cost_budget = _finite_number(source_run["cost_budget"], "source_run.cost_budget")
    cost_spent = _finite_number(source_run["cost_spent"], "source_run.cost_spent")
    if cost_budget <= 0 or cost_spent < 0 or cost_spent > cost_budget + TOL:
        raise ValueError("source_run cost accounting is invalid")
    for field in ("task_id", "run_id"):
        if not isinstance(doc[field], str) or not doc[field].strip():
            raise ValueError(f"{field} must be a non-empty string")

    split = _require_exact_keys(
        doc["split"],
        {"selection_frozen_before_holdout", "holdout_role", "selection_freeze_sha256", "holdout_manifest_sha256"},
        "split",
    )
    if split["selection_frozen_before_holdout"] is not True:
        raise ValueError("selection_frozen_before_holdout must be true")
    if split["holdout_role"] != "independent_post_selection":
        raise ValueError("holdout_role must be independent_post_selection")
    _sha256(split["selection_freeze_sha256"], "split.selection_freeze_sha256")
    _sha256(split["holdout_manifest_sha256"], "split.holdout_manifest_sha256")

    ref_raw = doc["reference_point"]
    if not isinstance(ref_raw, list) or len(ref_raw) != 2:
        raise ValueError("reference_point must contain exactly two values")
    reference = tuple(_finite_number(v, f"reference_point[{i}]") for i, v in enumerate(ref_raw))
    if any(v <= 0.0 for v in reference):
        raise ValueError("reference_point values must be positive")

    candidates_raw = doc["candidate_set"]
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise ValueError("candidate_set must be a non-empty array")
    candidates: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(candidates_raw):
        item = _require_exact_keys(
            item,
            {"candidate_id", "predicted_objectives", "predicted_constraint_margin", "selected"},
            f"candidate_set[{index}]",
        )
        candidate_id = item["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"candidate_set[{index}].candidate_id must be a non-empty string")
        if candidate_id in candidates:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        if not isinstance(item["selected"], bool):
            raise ValueError(f"candidate {candidate_id}.selected must be boolean")
        candidates[candidate_id] = {
            "predicted_objectives": _objectives(item["predicted_objectives"], f"candidate {candidate_id}.predicted_objectives", reference),
            "predicted_margin": _finite_number(item["predicted_constraint_margin"], f"candidate {candidate_id}.predicted_constraint_margin"),
            "selected": item["selected"],
        }
    selected_ids = {candidate_id for candidate_id, item in candidates.items() if item["selected"]}
    if not selected_ids:
        raise ValueError("at least one candidate must be selected")

    shifts_raw = doc["holdout_shifts"]
    if not isinstance(shifts_raw, list) or not shifts_raw:
        raise ValueError("holdout_shifts must be a non-empty array")
    shifts: list[dict[str, Any]] = []
    seen_shifts: set[str] = set()
    expected_ids = set(candidates)
    for index, shift in enumerate(shifts_raw):
        shift = _require_exact_keys(shift, {"shift_id", "shift_family", "evaluations"}, f"holdout_shifts[{index}]")
        shift_id, shift_family = shift["shift_id"], shift["shift_family"]
        for value, label in ((shift_id, "shift_id"), (shift_family, "shift_family")):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"holdout_shifts[{index}].{label} must be a non-empty string")
        if shift_id in seen_shifts:
            raise ValueError(f"duplicate shift_id: {shift_id}")
        seen_shifts.add(shift_id)
        evaluations_raw = shift["evaluations"]
        if not isinstance(evaluations_raw, list):
            raise ValueError(f"shift {shift_id}.evaluations must be an array")
        evaluations: dict[str, dict[str, Any]] = {}
        for j, evaluation in enumerate(evaluations_raw):
            evaluation = _require_exact_keys(
                evaluation, {"candidate_id", "objectives", "constraint_margin"},
                f"shift {shift_id}.evaluations[{j}]",
            )
            candidate_id = evaluation["candidate_id"]
            if candidate_id not in candidates:
                raise ValueError(f"shift {shift_id} contains unknown candidate_id: {candidate_id!r}")
            if candidate_id in evaluations:
                raise ValueError(f"shift {shift_id} contains duplicate evaluation for {candidate_id}")
            evaluations[candidate_id] = {
                "objectives": _objectives(evaluation["objectives"], f"shift {shift_id} candidate {candidate_id}.objectives", reference),
                "margin": _finite_number(evaluation["constraint_margin"], f"shift {shift_id} candidate {candidate_id}.constraint_margin"),
            }
        if set(evaluations) != expected_ids:
            missing = sorted(expected_ids - set(evaluations))
            extra = sorted(set(evaluations) - expected_ids)
            raise ValueError(f"shift {shift_id} must evaluate every candidate exactly once; missing={missing}, extra={extra}")
        shifts.append({"shift_id": shift_id, "shift_family": shift_family, "evaluations": evaluations})

    return {
        "reference_point": reference, "candidates": candidates,
        "selected_ids": selected_ids, "shifts": shifts,
        "task_id": doc["task_id"], "run_id": doc["run_id"], "split": split,
        "source_run": source_run,
    }


def score_document(document: Any, raw_bytes: bytes) -> dict[str, Any]:
    parsed = validate_document(document)
    reference: tuple[float, float] = parsed["reference_point"]
    candidates: dict[str, dict[str, Any]] = parsed["candidates"]
    selected_ids: set[str] = parsed["selected_ids"]
    shifts: list[dict[str, Any]] = parsed["shifts"]

    pair_count = len(selected_ids) * len(shifts)
    observed_feasible = 0
    predicted_feasible_pairs = 0
    false_feasible = 0
    predicted_infeasible_pairs = 0
    false_infeasible = 0
    normalized_errors: list[float] = []
    per_shift: list[dict[str, Any]] = []
    retainments: list[float] = []
    regrets: list[float] = []

    for shift in shifts:
        evaluations = shift["evaluations"]
        selected_truth_points: list[tuple[float, float]] = []
        all_truth_points: list[tuple[float, float]] = []
        shift_observed_feasible = 0
        shift_predicted_feasible = 0
        shift_false_feasible = 0
        shift_predicted_infeasible = 0
        shift_false_infeasible = 0
        shift_error = 0.0

        for candidate_id, candidate in candidates.items():
            evaluation = evaluations[candidate_id]
            truth_margin = evaluation["margin"]
            truth_objectives = evaluation["objectives"]
            predicted_margin = candidate["predicted_margin"]
            predicted_objectives = candidate["predicted_objectives"]
            predicted_is_feasible = predicted_margin >= 0.0
            truth_is_feasible = truth_margin >= 0.0
            if candidate_id in selected_ids:
                observed_feasible += int(truth_is_feasible)
                shift_observed_feasible += int(truth_is_feasible)
                if predicted_is_feasible:
                    predicted_feasible_pairs += 1
                    shift_predicted_feasible += 1
                    if not truth_is_feasible:
                        false_feasible += 1
                        shift_false_feasible += 1
                else:
                    predicted_infeasible_pairs += 1
                    shift_predicted_infeasible += 1
                    if truth_is_feasible:
                        false_infeasible += 1
                        shift_false_infeasible += 1
                for j in range(2):
                    error = abs(predicted_objectives[j] - truth_objectives[j]) / reference[j]
                    normalized_errors.append(error)
                    shift_error += error / 2.0
            if truth_is_feasible:
                all_truth_points.append(truth_objectives)
                if candidate_id in selected_ids:
                    selected_truth_points.append(truth_objectives)

        pool_hv = hypervolume_2d(all_truth_points, reference)
        selected_hv = hypervolume_2d(selected_truth_points, reference)
        if pool_hv > TOL:
            retention = min(1.0, max(0.0, selected_hv / pool_hv))
            regret = min(1.0, max(0.0, (pool_hv - selected_hv) / pool_hv))
            retainments.append(retention)
            regrets.append(regret)
        else:
            retention = None
            regret = None
        denom_feasible = shift_predicted_feasible
        denom_infeasible = shift_predicted_infeasible
        per_shift.append({
            "shift_id": shift["shift_id"],
            "shift_family": shift["shift_family"],
            "selected_feasibility_rate": shift_observed_feasible / len(selected_ids),
            "selected_predicted_feasible_count": shift_predicted_feasible,
            "selected_false_feasible_count": shift_false_feasible,
            "selected_false_feasible_rate": (shift_false_feasible / denom_feasible) if denom_feasible else None,
            "selected_predicted_infeasible_count": shift_predicted_infeasible,
            "selected_false_infeasible_count": shift_false_infeasible,
            "selected_false_infeasible_rate": (shift_false_infeasible / denom_infeasible) if denom_infeasible else None,
            "selected_feasible_hypervolume": selected_hv,
            "candidate_pool_feasible_hypervolume": pool_hv,
            "finite_pool_hypervolume_retention": retention,
            "finite_pool_normalized_regret": regret,
            "normalized_objective_absolute_error_mean_selected": shift_error / (2.0 * len(selected_ids)),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "receipt_label": "POST_SELECTION_SHIFT_AUDIT_NOT_PHYSICAL_CERTIFICATION",
        "input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "evaluator_source_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "task_id": parsed["task_id"],
        "run_id": parsed["run_id"],
        "source_run": parsed["source_run"],
        "split_declarations": {
            "holdout_role": parsed["split"]["holdout_role"],
            "selection_frozen_before_holdout": parsed["split"]["selection_frozen_before_holdout"],
            "selection_freeze_sha256": parsed["split"]["selection_freeze_sha256"],
            "holdout_manifest_sha256": parsed["split"]["holdout_manifest_sha256"],
            "independence_verified_by_evaluator": False,
        },
        "metrics": {
            "candidate_count": len(candidates),
            "selected_candidate_count": len(selected_ids),
            "holdout_shift_count": len(shifts),
            "selected_candidate_shift_pairs": pair_count,
            "selected_feasibility_rate": observed_feasible / pair_count,
            "selected_predicted_feasible_count": predicted_feasible_pairs,
            "selected_false_feasible_count": false_feasible,
            "selected_false_feasible_rate": (false_feasible / predicted_feasible_pairs) if predicted_feasible_pairs else None,
            "selected_predicted_infeasible_count": predicted_infeasible_pairs,
            "selected_false_infeasible_count": false_infeasible,
            "selected_false_infeasible_rate": (false_infeasible / predicted_infeasible_pairs) if predicted_infeasible_pairs else None,
            "normalized_objective_absolute_error_mean_selected": (sum(normalized_errors) / len(normalized_errors)) if normalized_errors else None,
            "finite_pool_hypervolume_retention_mean": (sum(retainments) / len(retainments)) if retainments else None,
            "finite_pool_hypervolume_retention_worst_shift": min(retainments) if retainments else None,
            "finite_pool_normalized_regret_mean": (sum(regrets) / len(regrets)) if regrets else None,
            "finite_pool_normalized_regret_worst_shift": max(regrets) if regrets else None,
            "shifts_without_feasible_reference_hypervolume": sum(1 for row in per_shift if row["candidate_pool_feasible_hypervolume"] <= TOL),
        },
        "per_shift": per_shift,
        "limitations": [
            "Finite-candidate-pool regret is not global optimizer regret.",
            "The evaluator does not authenticate the declared holdout partition or source measurements.",
            "Synthetic observations support mechanism tests only; simulator observations support simulation claims only.",
            "Physical casting claims require traceable held-out foundry observations and process-coverage evidence.",
            "Objective values must be normalized into [0, reference_point) before this two-objective evaluator is used.",
            "No risk certification or superiority claim is issued by this receipt.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="frozen JSON input document")
    parser.add_argument("--output", type=Path, help="write receipt JSON to this path (default stdout)")
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        document = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON constant: {value}")),
        )
        receipt = score_document(document, raw)
        rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"decision-robustness-eval: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
