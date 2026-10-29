#!/usr/bin/env python3
"""Task-level split-conformal screen for feasibility recommendations.

Uses a frozen calibration pool and never reads target outcomes. Its finite-sample
interpretation requires exchangeable calibration and target *tasks* within each
predeclared stratum. Descriptor support is an additional abstention guard, not a
proof of exchangeability or a physical-safety certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

SCHEMA_VERSION = "1.0.0"
RECEIPT_LABEL = "TASK_LEVEL_SPLIT_CONFORMAL_FEASIBILITY_SCREEN_NOT_PHYSICAL_SAFETY_CERTIFICATE"
EVIDENCE_CLASSES = {"synthetic", "public_analytic", "simulator_backed", "physical"}
CONFIG_FIELDS = frozenset({
    "schema_version", "config_id", "evidence_class", "policy_source_sha256",
    "policy_config_sha256", "candidate_pool_sha256", "alpha", "minimum_calibration_tasks_per_stratum",
    "maximum_descriptor_distance", "frozen_before_outcome_unseal",
})
CALIBRATION_FIELDS = frozenset({
    "schema_version", "calibration_id", "evidence_class", "policy_source_sha256",
    "policy_config_sha256", "split", "policy_frozen_before_outcomes",
    "candidate_pool_frozen_before_outcomes", "outcomes_unsealed_after_freeze",
    "independence_unit", "independence_note", "tasks",
})
CALIBRATION_TASK_FIELDS = frozenset({"task_id", "task_family", "stratum", "descriptor", "rows"})
CALIBRATION_ROW_FIELDS = frozenset({"candidate_id", "predicted_margin", "observed_margin"})
TARGET_FIELDS = frozenset({
    "schema_version", "target_manifest_id", "evidence_class", "policy_source_sha256",
    "policy_config_sha256", "split", "candidate_pool_frozen_before_outcomes", "tasks",
})
TARGET_TASK_FIELDS = frozenset({"task_id", "task_family", "stratum", "descriptor", "predictions"})
TARGET_PREDICTION_FIELDS = frozenset({"candidate_id", "predicted_margin"})


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
        raise ValueError(f"{label} must be non-empty trimmed text")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be an integer >= 1")
    return value


def _descriptor(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty array")
    clean = tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))
    if any(item < 0.0 or item > 1.0 for item in clean):
        raise ValueError(f"{label} components must be normalized to [0, 1]")
    return clean


def _unit_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("task descriptors must all have the same dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _candidate_pool_digest(candidate_ids: Sequence[str]) -> str:
    encoded = json.dumps(list(candidate_ids), separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_config(raw: Any) -> dict[str, Any]:
    config = _strict_object(raw, CONFIG_FIELDS, "config")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"config schema_version must be {SCHEMA_VERSION}")
    _nonblank(config["config_id"], "config_id")
    if config["evidence_class"] not in EVIDENCE_CLASSES:
        raise ValueError("config evidence_class is unsupported")
    _sha256(config["policy_source_sha256"], "policy_source_sha256")
    _sha256(config["policy_config_sha256"], "policy_config_sha256")
    _sha256(config["candidate_pool_sha256"], "candidate_pool_sha256")
    alpha = _finite(config["alpha"], "alpha")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    minimum = _positive_integer(
        config["minimum_calibration_tasks_per_stratum"],
        "minimum_calibration_tasks_per_stratum",
    )
    # A finite split-conformal quantile at level 1-alpha needs this many
    # independent calibration tasks: ceil((n+1)(1-alpha)) must be <= n.
    minimum_for_finite_quantile = math.ceil((1.0 - alpha) / alpha)
    if minimum < minimum_for_finite_quantile:
        raise ValueError(
            "minimum_calibration_tasks_per_stratum is too small for a finite "
            f"1-alpha conformal quantile; require >= {minimum_for_finite_quantile}"
        )
    max_distance = _finite(config["maximum_descriptor_distance"], "maximum_descriptor_distance")
    if max_distance < 0.0:
        raise ValueError("maximum_descriptor_distance must be non-negative")
    if config["frozen_before_outcome_unseal"] is not True:
        raise ValueError("frozen_before_outcome_unseal must be true")
    return {**config, "alpha": alpha, "minimum_calibration_tasks_per_stratum": minimum,
            "maximum_descriptor_distance": max_distance}


def validate_calibration(raw: Any, config: dict[str, Any]) -> dict[str, Any]:
    manifest = _strict_object(raw, CALIBRATION_FIELDS, "calibration manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"calibration schema_version must be {SCHEMA_VERSION}")
    _nonblank(manifest["calibration_id"], "calibration_id")
    if manifest["evidence_class"] != config["evidence_class"]:
        raise ValueError("calibration evidence class does not match config")
    for field in ("policy_source_sha256", "policy_config_sha256"):
        if _sha256(manifest[field], f"calibration.{field}") != config[field].lower():
            raise ValueError(f"calibration {field} does not match frozen config")
    if manifest["split"] != "independent_calibration":
        raise ValueError("calibration split must be independent_calibration")
    if manifest["policy_frozen_before_outcomes"] is not True:
        raise ValueError("calibration policy must be frozen before outcomes")
    if manifest["candidate_pool_frozen_before_outcomes"] is not True:
        raise ValueError("calibration candidate pool must be frozen before outcomes")
    if manifest["outcomes_unsealed_after_freeze"] is not True:
        raise ValueError("calibration outcomes must be unsealed after policy and pool freeze")
    if manifest["independence_unit"] != "task_id":
        raise ValueError("independence_unit must be task_id; candidate rows are nested, not independent")
    _nonblank(manifest["independence_note"], "calibration.independence_note")
    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("calibration tasks must be a non-empty array")
    task_ids: set[str] = set()
    descriptor_size: int | None = None
    candidate_pool_roster: list[str] | None = None
    for ti, item in enumerate(tasks):
        task = _strict_object(item, CALIBRATION_TASK_FIELDS, f"calibration.tasks[{ti}]")
        task_id = _nonblank(task["task_id"], f"calibration.tasks[{ti}].task_id")
        if task_id in task_ids:
            raise ValueError(f"duplicate calibration task_id: {task_id}")
        task_ids.add(task_id)
        _nonblank(task["task_family"], f"calibration.tasks[{ti}].task_family")
        _nonblank(task["stratum"], f"calibration.tasks[{ti}].stratum")
        desc = _descriptor(task["descriptor"], f"calibration.tasks[{ti}].descriptor")
        if descriptor_size is None:
            descriptor_size = len(desc)
        elif len(desc) != descriptor_size:
            raise ValueError("task descriptors must all have the same dimension")
        rows = task["rows"]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"calibration.tasks[{ti}].rows must be non-empty")
        row_ids: set[str] = set()
        task_roster: list[str] = []
        for ri, raw_row in enumerate(rows):
            row = _strict_object(raw_row, CALIBRATION_ROW_FIELDS, f"calibration.tasks[{ti}].rows[{ri}]")
            cid = _nonblank(row["candidate_id"], f"calibration.tasks[{ti}].rows[{ri}].candidate_id")
            if cid in row_ids:
                raise ValueError(f"duplicate candidate_id within calibration task {task_id}: {cid}")
            row_ids.add(cid)
            task_roster.append(cid)
            _finite(row["predicted_margin"], f"calibration.tasks[{ti}].rows[{ri}].predicted_margin")
            _finite(row["observed_margin"], f"calibration.tasks[{ti}].rows[{ri}].observed_margin")
        if candidate_pool_roster is None:
            candidate_pool_roster = task_roster
        elif task_roster != candidate_pool_roster:
            raise ValueError("calibration tasks must share the same ordered frozen candidate pool")
    if candidate_pool_roster is None or _candidate_pool_digest(candidate_pool_roster) != config["candidate_pool_sha256"].lower():
        raise ValueError("candidate pool roster digest does not match frozen config")
    return manifest


def validate_targets(raw: Any, config: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    manifest = _strict_object(raw, TARGET_FIELDS, "target prediction manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"target prediction schema_version must be {SCHEMA_VERSION}")
    _nonblank(manifest["target_manifest_id"], "target_manifest_id")
    if manifest["evidence_class"] != config["evidence_class"]:
        raise ValueError("target evidence class does not match config")
    for field in ("policy_source_sha256", "policy_config_sha256"):
        if _sha256(manifest[field], f"target.{field}") != config[field].lower():
            raise ValueError(f"target {field} does not match frozen config")
    if manifest["split"] != "independent_held_out_predictions":
        raise ValueError("target split must be independent_held_out_predictions")
    if manifest["candidate_pool_frozen_before_outcomes"] is not True:
        raise ValueError("target candidate pool must be frozen before outcomes")
    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("target prediction tasks must be a non-empty array")
    calibration_ids = {task["task_id"] for task in calibration["tasks"]}
    target_ids: set[str] = set()
    calibration_roster = [row["candidate_id"] for row in calibration["tasks"][0]["rows"]]
    descriptor_size = len(calibration["tasks"][0]["descriptor"])
    for ti, item in enumerate(tasks):
        task = _strict_object(item, TARGET_TASK_FIELDS, f"target.tasks[{ti}]")
        task_id = _nonblank(task["task_id"], f"target.tasks[{ti}].task_id")
        if task_id in calibration_ids:
            raise ValueError(f"target task overlaps calibration task: {task_id}")
        if task_id in target_ids:
            raise ValueError(f"duplicate target task_id: {task_id}")
        target_ids.add(task_id)
        _nonblank(task["task_family"], f"target.tasks[{ti}].task_family")
        _nonblank(task["stratum"], f"target.tasks[{ti}].stratum")
        desc = _descriptor(task["descriptor"], f"target.tasks[{ti}].descriptor")
        if len(desc) != descriptor_size:
            raise ValueError("target descriptor dimension does not match calibration")
        predictions = task["predictions"]
        if not isinstance(predictions, list) or not predictions:
            raise ValueError(f"target.tasks[{ti}].predictions must be a non-empty array")
        candidate_ids: set[str] = set()
        target_roster: list[str] = []
        for pi, prediction in enumerate(predictions):
            row = _strict_object(prediction, TARGET_PREDICTION_FIELDS, f"target.tasks[{ti}].predictions[{pi}]")
            cid = _nonblank(row["candidate_id"], f"target.tasks[{ti}].predictions[{pi}].candidate_id")
            if cid in candidate_ids:
                raise ValueError(f"duplicate target candidate_id in task {task_id}: {cid}")
            candidate_ids.add(cid)
            target_roster.append(cid)
            _finite(row["predicted_margin"], f"target.tasks[{ti}].predictions[{pi}].predicted_margin")
        if target_roster != calibration_roster:
            raise ValueError("target task must use the exact frozen calibration candidate-pool roster")
    return manifest


def _conformal_quantile(scores: Sequence[float], alpha: float) -> tuple[float | None, int]:
    """Finite-sample one-sided split-conformal order statistic, task-level scores."""
    n = len(scores)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or rank > n:
        return None, rank
    return sorted(scores)[rank - 1], rank


def evaluate(config_raw: Any, calibration_raw: Any, target_raw: Any) -> dict[str, Any]:
    config = validate_config(config_raw)
    calibration = validate_calibration(calibration_raw, config)
    targets = validate_targets(target_raw, config, calibration)

    calibration_by_stratum: dict[str, list[dict[str, Any]]] = {}
    for task in calibration["tasks"]:
        calibration_by_stratum.setdefault(task["stratum"], []).append(task)

    strata: dict[str, dict[str, Any]] = {}
    for stratum, tasks in calibration_by_stratum.items():
        task_scores: list[float] = []
        descriptors: list[tuple[float, ...]] = []
        for task in tasks:
            # The maximum overprediction error across the frozen finite candidate
            # pool makes the unit one independent task, not one candidate row.
            score = max(
                max(0.0, float(row["predicted_margin"]) - float(row["observed_margin"]))
                for row in task["rows"]
            )
            task_scores.append(score)
            descriptors.append(tuple(float(v) for v in task["descriptor"]))
        quantile, rank = _conformal_quantile(task_scores, config["alpha"])
        enough = len(tasks) >= config["minimum_calibration_tasks_per_stratum"] and quantile is not None
        strata[stratum] = {
            "calibration_task_count": len(tasks),
            "task_level_overprediction_scores": task_scores,
            "conformal_rank_1based": rank,
            "overprediction_quantile": quantile if enough else None,
            "minimum_calibration_tasks_per_stratum": config["minimum_calibration_tasks_per_stratum"],
            "finite_quantile": quantile is not None,
            "calibration_descriptors": [list(values) for values in descriptors],
            "decision_ready": bool(enough),
        }

    results = []
    total_candidates = 0
    recommended_candidates = 0
    tasks_with_any_recommendation = 0
    for task in targets["tasks"]:
        stratum = task["stratum"]
        calibration_summary = strata.get(stratum)
        q = calibration_summary["overprediction_quantile"] if calibration_summary else None
        descriptor = tuple(float(v) for v in task["descriptor"])
        nearest_distance = None
        if calibration_summary is not None:
            nearest_distance = min(
                _unit_distance(descriptor, tuple(values))
                for values in calibration_summary["calibration_descriptors"]
            )
        support_ok = (
            nearest_distance is not None
            and nearest_distance <= config["maximum_descriptor_distance"]
        )
        calibration_ok = q is not None
        candidate_results = []
        task_recommended = 0
        for pred in task["predictions"]:
            predicted_margin = float(pred["predicted_margin"])
            lower_margin = predicted_margin - q if calibration_ok else None
            recommend = bool(calibration_ok and support_ok and lower_margin is not None and lower_margin >= 0.0)
            action = "recommend_feasible" if recommend else "abstain"
            total_candidates += 1
            if recommend:
                recommended_candidates += 1
                task_recommended += 1
            candidate_results.append({
                "candidate_id": pred["candidate_id"],
                "predicted_margin": predicted_margin,
                "calibrated_lower_margin": lower_margin,
                "action": action,
            })
        if task_recommended:
            tasks_with_any_recommendation += 1
        results.append({
            "task_id": task["task_id"],
            "task_family": task["task_family"],
            "stratum": stratum,
            "nearest_calibration_descriptor_distance": nearest_distance,
            "maximum_descriptor_distance": config["maximum_descriptor_distance"],
            "descriptor_support_ok": bool(support_ok),
            "calibration_available": bool(calibration_ok),
            "candidate_count": len(candidate_results),
            "recommended_feasible_count": task_recommended,
            "abstained_count": len(candidate_results) - task_recommended,
            "candidates": candidate_results,
        })

    underpowered_target_strata = sorted({
        task["stratum"] for task in targets["tasks"]
        if task["stratum"] not in strata or not strata[task["stratum"]]["decision_ready"]
    })
    unsupported_target_strata = sorted({
        task["stratum"] for task in targets["tasks"]
        if task["stratum"] not in strata
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_label": RECEIPT_LABEL,
        "decision": "inconclusive" if underpowered_target_strata else "evaluated_with_abstention",
        "evidence_class": config["evidence_class"],
        "config_id": config["config_id"],
        "calibration_id": calibration["calibration_id"],
        "target_manifest_id": targets["target_manifest_id"],
        "policy_source_sha256": config["policy_source_sha256"].lower(),
        "policy_config_sha256": config["policy_config_sha256"].lower(),
        "alpha": config["alpha"],
        "minimum_calibration_tasks_per_stratum": config["minimum_calibration_tasks_per_stratum"],
        "maximum_descriptor_distance": config["maximum_descriptor_distance"],
        "calibration_task_count": len(calibration["tasks"]),
        "target_task_count": len(targets["tasks"]),
        "candidate_count": total_candidates,
        "recommended_feasible_count": recommended_candidates,
        "abstained_candidate_count": total_candidates - recommended_candidates,
        "candidate_acceptance_coverage": recommended_candidates / total_candidates if total_candidates else 0.0,
        "target_tasks_with_any_recommendation": tasks_with_any_recommendation,
        "task_recommendation_coverage": tasks_with_any_recommendation / len(targets["tasks"]),
        "underpowered_target_strata": underpowered_target_strata,
        "unsupported_target_strata": unsupported_target_strata,
        "strata": {
            name: {key: value for key, value in summary.items() if key != "calibration_descriptors"}
            for name, summary in sorted(strata.items())
        },
        "target_task_results": results,
        "claim_ceiling": (
            "Task-level split-conformal screening receipt only. Its finite-sample interpretation requires "
            "exchangeability of independent calibration and target tasks within each predeclared stratum, "
            "a frozen candidate pool, and honestly specified evidence. Descriptor distance is an abstention "
            "guard, not proof of exchangeability. It does not establish physical feasibility, physical safety, "
            "calibration under unseen distribution shift, or optimizer/SOTA superiority."
        ),
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant forbidden: {value}")


def _load_json(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        return raw, json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("target_predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source_paths = [args.config.resolve(), args.calibration.resolve(), args.target_predictions.resolve()]
        if len(set(source_paths)) != len(source_paths):
            raise ValueError("config, calibration, and target prediction inputs must be different files")
        config_raw, config = _load_json(args.config)
        calibration_raw, calibration = _load_json(args.calibration)
        target_raw, targets = _load_json(args.target_predictions)
        result = evaluate(config, calibration, targets)
        result["source_artifacts"] = {
            "config_sha256": hashlib.sha256(config_raw).hexdigest(),
            "calibration_manifest_sha256": hashlib.sha256(calibration_raw).hexdigest(),
            "target_prediction_manifest_sha256": hashlib.sha256(target_raw).hexdigest(),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        output_bytes = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        output_path = args.output.resolve()
        if output_path in source_paths or output_path.exists():
            raise ValueError("output must be a new path and must not overwrite any input")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            handle.write(output_bytes)
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"task-conformal feasibility gate failed: {exc}", file=sys.stderr)
        return 2
    print(f"task-conformal receipt written: {args.output}")
    print(f"calibration_tasks={result['calibration_task_count']} target_tasks={result['target_task_count']}")
    print(f"recommended={result['recommended_feasible_count']} abstained={result['abstained_candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
