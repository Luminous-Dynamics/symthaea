#!/usr/bin/env python3
"""Fail-closed preflight contract for a future MF-SCBO comparison.

This module validates compatibility and accounting before running the upstream
optimizer. A valid preflight receipt is not an optimizer run or a performance claim.
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
PREFLIGHT_LABEL = "MF_SCBO_COMPARISON_CONTRACT_PREFLIGHT_NOT_EXECUTION_OR_SOTA_EVIDENCE"
ALLOWED_EVIDENCE = {"synthetic", "public_analytic", "simulator_backed", "physical"}
ALLOWED_NORMALIZATION_SOURCES = {"predeclared_bounds", "training_archive_only"}
FIDELITIES = ("low", "medium", "high")
CONTRACT_FIELDS = frozenset({
    "schema_version", "contract_id", "upstream_repository", "upstream_commit_sha",
    "task_suite_path", "task_suite_sha256", "evidence_class", "objective_directions",
    "normalization_offset", "normalization_scale", "normalization_source",
    "normalization_frozen_before_unseal", "scalarization_weights",
    "scalarization_frozen_before_unseal", "constraint_fidelity_invariant",
    "constraint_definition_sha256", "constraint_margin_sign",
    "upstream_constraint_sign", "target_hf_queries_allowed_in_search",
    "full_pool_audit_truth_used_in_search", "policy_frozen_before_outcome_unseal",
    "holdout_split", "fidelity_cost_units", "initial_points_per_fidelity",
    "cost_budget_units", "seed", "stop_rule",
})


def _strict_object(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive_int(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
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


def _is_sha1(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 40:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _safe_repo_file(relative: object, label: str) -> Path:
    name = _nonblank(relative, label)
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be a repository-relative path without parent traversal")
    root = Path(__file__).resolve().parents[2]
    resolved = (root / candidate).resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"{label} must resolve to an existing file inside the repository")
    return resolved


def _load_lock() -> tuple[dict[str, Any], bytes]:
    path = Path(__file__).with_name("baselines") / "mf_scbo_lock_v1.json"
    raw = path.read_bytes()
    try:
        lock = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys, parse_constant=_reject_json_constant)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load source lock: {exc}") from exc
    if not isinstance(lock, dict):
        raise ValueError("source lock must be a JSON object")
    if lock.get("baseline_id") != "MF-SCBO" or lock.get("status") != "source_pinned_not_reproduced":
        raise ValueError("MF-SCBO source lock is missing or does not have the expected status")
    if not _is_sha1(lock.get("upstream_commit_sha")):
        raise ValueError("MF-SCBO source lock upstream commit SHA is invalid")
    return lock, raw


def validate_contract(raw: Any) -> dict[str, Any]:
    contract = _strict_object(raw, CONTRACT_FIELDS, "MF-SCBO comparison contract")
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    _nonblank(contract["contract_id"], "contract_id")

    lock, _ = _load_lock()
    if contract["upstream_repository"] != lock.get("upstream_repository"):
        raise ValueError("upstream_repository does not match the pinned MF-SCBO source lock")
    if contract["upstream_commit_sha"] != lock.get("upstream_commit_sha"):
        raise ValueError("upstream_commit_sha does not match the pinned MF-SCBO source lock")

    declared_suite_digest = _sha256(contract["task_suite_sha256"], "task_suite_sha256")
    suite_path = _safe_repo_file(contract["task_suite_path"], "task_suite_path")
    actual_suite_digest = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    if declared_suite_digest != actual_suite_digest:
        raise ValueError("task_suite_sha256 does not match the referenced suite file bytes")
    _sha256(contract["constraint_definition_sha256"], "constraint_definition_sha256")
    if contract["evidence_class"] not in ALLOWED_EVIDENCE:
        raise ValueError("evidence_class is unsupported")

    directions = contract["objective_directions"]
    if directions != ["minimize", "minimize"]:
        raise ValueError("this comparator contract expects two minimization objectives in fixed order")
    offsets = contract["normalization_offset"]
    scales = contract["normalization_scale"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError("normalization_offset must contain two numbers")
    if not isinstance(scales, list) or len(scales) != 2:
        raise ValueError("normalization_scale must contain two numbers")
    for i, value in enumerate(offsets):
        _finite(value, f"normalization_offset[{i}]")
    for i, value in enumerate(scales):
        if _finite(value, f"normalization_scale[{i}]") <= 0.0:
            raise ValueError(f"normalization_scale[{i}] must be positive")
    if contract["normalization_source"] not in ALLOWED_NORMALIZATION_SOURCES:
        raise ValueError("normalization_source must use predeclared bounds or training archive only")
    for field in (
        "normalization_frozen_before_unseal", "scalarization_frozen_before_unseal",
        "policy_frozen_before_outcome_unseal",
    ):
        if contract[field] is not True:
            raise ValueError(f"{field} must be true")

    weights = contract["scalarization_weights"]
    if not isinstance(weights, list) or len(weights) < 5:
        raise ValueError("scalarization_weights must contain at least five preregistered weight pairs")
    normalized_weights: list[tuple[float, float]] = []
    for i, pair in enumerate(weights):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"scalarization_weights[{i}] must be a two-value array")
        w0 = _finite(pair[0], f"scalarization_weights[{i}][0]")
        w1 = _finite(pair[1], f"scalarization_weights[{i}][1]")
        if w0 < 0.0 or w1 < 0.0 or abs((w0 + w1) - 1.0) > 1e-12:
            raise ValueError(f"scalarization_weights[{i}] must be non-negative and sum to one")
        normalized_weights.append((w0, w1))
    if len(set(normalized_weights)) != len(normalized_weights):
        raise ValueError("scalarization_weights must be unique")
    if (1.0, 0.0) not in normalized_weights or (0.0, 1.0) not in normalized_weights:
        raise ValueError("scalarization_weights must include both single-objective endpoints")

    if contract["constraint_fidelity_invariant"] is not True:
        raise ValueError(
            "MF-SCBO exposes one shared constraint callable; fidelity-dependent constraint margins are incompatible"
        )
    if contract["constraint_margin_sign"] != "positive_means_feasible":
        raise ValueError("constraint_margin_sign must be positive_means_feasible")
    if contract["upstream_constraint_sign"] != "c(x) <= 0":
        raise ValueError("upstream_constraint_sign must match MF-SCBO's c(x) <= 0 convention")

    if contract["target_hf_queries_allowed_in_search"] is not True:
        raise ValueError("MF-SCBO comparison must declare that budgeted target-HF queries are allowed in the search")
    if contract["full_pool_audit_truth_used_in_search"] is not False:
        raise ValueError("full-pool audit truth must never be passed to the search loop")
    if contract["holdout_split"] != "independent_held_out":
        raise ValueError("holdout_split must be independent_held_out")

    costs = contract["fidelity_cost_units"]
    points = contract["initial_points_per_fidelity"]
    if not isinstance(costs, dict) or set(costs) != set(FIDELITIES):
        raise ValueError("fidelity_cost_units must define exactly low, medium, and high")
    if not isinstance(points, dict) or set(points) != set(FIDELITIES):
        raise ValueError("initial_points_per_fidelity must define exactly low, medium, and high")
    clean_costs = {level: _positive_int(costs[level], f"fidelity_cost_units.{level}") for level in FIDELITIES}
    clean_points = {level: _positive_int(points[level], f"initial_points_per_fidelity.{level}") for level in FIDELITIES}
    cost_budget = _positive_int(contract["cost_budget_units"], "cost_budget_units")
    initial_cost = sum(clean_costs[level] * clean_points[level] for level in FIDELITIES)
    if initial_cost > cost_budget:
        raise ValueError(f"initialization cost {initial_cost} exceeds total cost budget {cost_budget}")
    seed = contract["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    _nonblank(contract["stop_rule"], "stop_rule")

    return {
        **contract,
        "fidelity_cost_units": clean_costs,
        "initial_points_per_fidelity": clean_points,
        "initialization_cost_units": initial_cost,
        "cost_budget_units": cost_budget,
        "scalarization_weights": [[w0, w1] for w0, w1 in normalized_weights],
    }


def scalarize_minimization_objectives(
    values: Sequence[float],
    offsets: Sequence[float],
    scales: Sequence[float],
    weights: Sequence[float],
) -> float:
    """Return a scalar to maximize from two normalized minimization objectives."""
    if len(values) != 2 or len(offsets) != 2 or len(scales) != 2 or len(weights) != 2:
        raise ValueError("values, offsets, scales, and weights must each have length two")
    total = 0.0
    weight_sum = 0.0
    for i in range(2):
        value = _finite(values[i], f"values[{i}]")
        offset = _finite(offsets[i], f"offsets[{i}]")
        scale = _finite(scales[i], f"scales[{i}]")
        weight = _finite(weights[i], f"weights[{i}]")
        if scale <= 0.0 or weight < 0.0:
            raise ValueError("scales must be positive and weights non-negative")
        total += weight * ((value - offset) / scale)
        weight_sum += weight
    if abs(weight_sum - 1.0) > 1e-12:
        raise ValueError("weights must sum to one")
    return -total


def margin_to_mfscbo_constraint(margin: float) -> float:
    """MF-SCBO wants c(x) <= 0; the CASTING convention is margin >= 0."""
    value = _finite(margin, "constraint margin")
    return -value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _load_json(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    return raw, value


def evaluate_contract_receipt(contract_bytes: bytes, validated_contract: dict[str, Any]) -> dict[str, Any]:
    lock, lock_bytes = _load_lock()
    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_label": PREFLIGHT_LABEL,
        "decision": "compatible_contract_validated_not_run",
        "contract_id": validated_contract["contract_id"],
        "upstream_repository": lock["upstream_repository"],
        "upstream_commit_sha": lock["upstream_commit_sha"],
        "evidence_class": validated_contract["evidence_class"],
        "task_suite_path": validated_contract["task_suite_path"],
        "task_suite_sha256": validated_contract["task_suite_sha256"],
        "constraint_definition_sha256": validated_contract["constraint_definition_sha256"],
        "scalarization_weight_count": len(validated_contract["scalarization_weights"]),
        "constraint_fidelity_invariant": validated_contract["constraint_fidelity_invariant"],
        "fidelity_cost_units": validated_contract["fidelity_cost_units"],
        "initial_points_per_fidelity": validated_contract["initial_points_per_fidelity"],
        "initialization_cost_units": validated_contract["initialization_cost_units"],
        "cost_budget_units": validated_contract["cost_budget_units"],
        "source_artifacts": {
            "raw_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "mf_scbo_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "preflight_evaluator_sha256": source_digest,
        },
        "holdout_truth_limit": (
            "The preflight validates a declared independent_held_out split and freeze-before-unseal flags, "
            "but cannot prove independence, source/target family disjointness, or holdout secrecy from the manifest alone."
        ),
        "claim_ceiling": (
            "The preflight verified the referenced task-suite file digest and passed schema, source-pin, "
            "scalarization, constraint-compatibility and initialization-cost checks. Holdout timing and independence "
            "remain assertions. This is not an MF-SCBO execution, benchmark result, reproduced baseline, "
            "SOTA comparison, or physical-safety certification."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path, help="frozen comparison contract JSON")
    parser.add_argument("--output", type=Path, help="write receipt to a new path (must not exist)")
    args = parser.parse_args(argv)
    try:
        raw, parsed = _load_json(args.contract)
        validated = validate_contract(parsed)
        receipt = evaluate_contract_receipt(raw, validated)
        encoded = json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            if args.output.resolve() == args.contract.resolve() or args.output.exists():
                raise ValueError("output path must be new and must not overwrite an input")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(encoded)
            print(f"MF-SCBO comparison preflight receipt written: {args.output}")
        return 0
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"MF-SCBO comparison preflight failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
