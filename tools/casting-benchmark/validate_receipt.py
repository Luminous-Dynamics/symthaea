#!/usr/bin/env python3
"""Strict validator for CASTING-MO-BO-001 receipts.

Benchmark-only tooling: standard library only, no production/runtime dependency.
This validator checks semantic invariants that JSON Schema alone cannot express.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
GIT_REV_RE = re.compile(r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
STATUSES = {"success", "failed", "refused", "incomplete"}
TERMINATIONS = {
    "budget_exhausted",
    "predeclared_target_reached",
    "no_valid_candidates",
    "error",
    "cancelled",
    "missing_or_skipped",
}
ROOT_KEYS = {
    "schema_version", "protocol_id", "run_id", "status", "synthetic_fixture_only",
    "algorithm", "suite", "evaluation_mode", "source_revision", "protocol_sha256", "environment_lock_sha256",
    "analysis_lock_sha256", "seed", "budget", "accounting", "termination_reason",
    "metrics", "artifacts", "notes",
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _closed_object(value: Any, required: set[str], allowed: set[str], path: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected object")
        return False
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing:
        errors.append(f"{path}: missing required keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"{path}: unknown keys: {', '.join(unknown)}")
    return not missing and not unknown


def _check_hash(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        errors.append(f"{path}: expected 64 lowercase hexadecimal SHA-256 characters")


def _check_revision(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not GIT_REV_RE.fullmatch(value):
        errors.append(f"{path}: expected immutable 40- or 64-character lowercase Git revision")


def _check_fraction(value: Any, path: str, errors: list[str]) -> None:
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        errors.append(f"{path}: expected finite number in [0, 1]")


def validate_receipt(receipt: Any, for_publication: bool = False) -> list[str]:
    errors: list[str] = []
    if not _closed_object(receipt, ROOT_KEYS - {"metrics", "notes"}, ROOT_KEYS, "$", errors):
        return errors

    if receipt["schema_version"] != "1.0.0":
        errors.append("$.schema_version: unsupported version")
    if receipt["protocol_id"] != "CASTING-MO-BO-001":
        errors.append("$.protocol_id: wrong protocol")
    if not isinstance(receipt["evaluation_mode"], str) or receipt["evaluation_mode"] not in {"single_fidelity", "multi_fidelity"}:
        errors.append("$.evaluation_mode: expected single_fidelity or multi_fidelity")
    if not isinstance(receipt["run_id"], str) or not re.fullmatch(
        r"^[A-Za-z0-9][A-Za-z0-9._-]+$", receipt["run_id"]
    ):
        errors.append("$.run_id: malformed run identifier")
    if not isinstance(receipt["status"], str) or receipt["status"] not in STATUSES:
        errors.append("$.status: queued/skipped/pending is not a terminal receipt status")
    if not isinstance(receipt["synthetic_fixture_only"], bool):
        errors.append("$.synthetic_fixture_only: expected boolean")
    elif for_publication and receipt["synthetic_fixture_only"]:
        errors.append("$.synthetic_fixture_only: fixture receipts cannot be published as empirical results")

    algorithm_allowed = {"id", "version", "configuration_sha256"}
    algorithm_required = algorithm_allowed
    if _closed_object(receipt["algorithm"], algorithm_required, algorithm_allowed, "$.algorithm", errors):
        for key in ("id", "version"):
            if not isinstance(receipt["algorithm"][key], str) or not receipt["algorithm"][key].strip():
                errors.append(f"$.algorithm.{key}: expected non-empty immutable identity")
        if str(receipt["algorithm"].get("version", "")).lower() == "latest":
            errors.append("$.algorithm.version: 'latest' is mutable and prohibited")
        _check_hash(receipt["algorithm"]["configuration_sha256"], "$.algorithm.configuration_sha256", errors)

    suite_allowed = {
        "id", "version", "instance_id", "dimension", "objective_count",
        "objective_direction", "objective_evaluator_sha256", "ideal_point", "reference_point",
    }
    if _closed_object(receipt["suite"], suite_allowed, suite_allowed, "$.suite", errors):
        for key in ("id", "version", "instance_id"):
            if not isinstance(receipt["suite"][key], str) or not receipt["suite"][key].strip():
                errors.append(f"$.suite.{key}: expected non-empty immutable identity")
        for key in ("dimension", "objective_count"):
            if not _is_int(receipt["suite"][key]) or receipt["suite"][key] < (1 if key == "dimension" else 2):
                errors.append(f"$.suite.{key}: invalid positive integer")
        if not isinstance(receipt["suite"]["objective_direction"], str) or receipt["suite"]["objective_direction"] not in {
            "minimize_all", "maximize_all", "mixed_explicit"
        }:
            errors.append("$.suite.objective_direction: invalid direction convention")
        _check_hash(receipt["suite"]["objective_evaluator_sha256"], "$.suite.objective_evaluator_sha256", errors)
        dimension = receipt["suite"].get("objective_count")
        ideal = receipt["suite"].get("ideal_point")
        reference = receipt["suite"].get("reference_point")
        for key, point in (("ideal_point", ideal), ("reference_point", reference)):
            if not isinstance(point, list) or not _is_int(dimension) or len(point) != dimension:
                errors.append(f"$.suite.{key}: expected exactly objective_count finite coordinates")
            elif not all(_is_finite_number(v) for v in point):
                errors.append(f"$.suite.{key}: coordinates must be finite numbers")
        if isinstance(ideal, list) and isinstance(reference, list) and len(ideal) == len(reference) and all(_is_finite_number(v) for v in ideal + reference):
            if any(a >= b for a, b in zip(ideal, reference)):
                errors.append("$.suite: each ideal coordinate must be strictly below its matching reference coordinate")

    _check_revision(receipt["source_revision"], "$.source_revision", errors)
    for key in ("protocol_sha256", "environment_lock_sha256", "analysis_lock_sha256"):
        _check_hash(receipt[key], f"$.{key}", errors)

    if not _is_int(receipt["seed"]) or not 0 <= receipt["seed"] <= 9007199254740991:
        errors.append("$.seed: expected non-negative exactly representable integer")
    budget_allowed = {"objective_evaluations", "initial_design_evaluations"}
    if _closed_object(receipt["budget"], budget_allowed, budget_allowed, "$.budget", errors):
        budget = receipt["budget"]
        if not _is_int(budget["objective_evaluations"]) or budget["objective_evaluations"] < 1:
            errors.append("$.budget.objective_evaluations: must be positive")
        if not _is_int(budget["initial_design_evaluations"]) or budget["initial_design_evaluations"] < 0:
            errors.append("$.budget.initial_design_evaluations: must be non-negative")
        elif _is_int(budget.get("objective_evaluations")) and budget["initial_design_evaluations"] > budget["objective_evaluations"]:
            errors.append("$.budget.initial_design_evaluations: cannot exceed budget")

    accounting_allowed = {
        "objective_evaluations", "feasible_evaluations", "infeasible_evaluations",
        "invalid_evaluations", "failed_evaluations", "wall_clock_ms",
        "peak_memory_bytes", "objective_oracle_cost_units",
        "fidelity_evaluations", "fidelity_cost_units_by_level",
    }
    accounting_required = {
        "objective_evaluations", "feasible_evaluations", "infeasible_evaluations",
        "invalid_evaluations", "failed_evaluations", "wall_clock_ms",
    }
    if _closed_object(receipt["accounting"], accounting_required, accounting_allowed, "$.accounting", errors):
        accounting = receipt["accounting"]
        integer_counts = [
            "objective_evaluations", "feasible_evaluations", "infeasible_evaluations",
            "invalid_evaluations", "failed_evaluations", "wall_clock_ms",
        ]
        for key in integer_counts:
            if not _is_int(accounting[key]) or accounting[key] < (1 if key == "objective_evaluations" else 0):
                errors.append(f"$.accounting.{key}: invalid non-negative integer")
        if all(_is_int(accounting[k]) and accounting[k] >= 0 for k in integer_counts):
            partition = sum(accounting[k] for k in (
                "feasible_evaluations", "infeasible_evaluations",
                "invalid_evaluations", "failed_evaluations",
            ))
            if partition != accounting["objective_evaluations"]:
                errors.append("$.accounting: feasibility/invalid/failure buckets must sum exactly to objective_evaluations")
        if "peak_memory_bytes" in accounting and accounting["peak_memory_bytes"] is not None:
            if not _is_int(accounting["peak_memory_bytes"]) or accounting["peak_memory_bytes"] < 0:
                errors.append("$.accounting.peak_memory_bytes: expected null or non-negative integer")
        if "objective_oracle_cost_units" in accounting and accounting["objective_oracle_cost_units"] is not None:
            v = accounting["objective_oracle_cost_units"]
            if not _is_finite_number(v) or v < 0:
                errors.append("$.accounting.objective_oracle_cost_units: expected null or non-negative finite number")
        has_fidelity_counts = "fidelity_evaluations" in accounting
        has_fidelity_costs = "fidelity_cost_units_by_level" in accounting
        if receipt["evaluation_mode"] == "multi_fidelity" and not (has_fidelity_counts and has_fidelity_costs):
            errors.append("$.accounting: multi-fidelity runs require counts and cost units for every fidelity level")
        if has_fidelity_counts != has_fidelity_costs:
            errors.append("$.accounting: fidelity_evaluations and fidelity_cost_units_by_level must be provided together")
        if has_fidelity_counts and has_fidelity_costs:
            fidelity_counts = accounting["fidelity_evaluations"]
            fidelity_costs = accounting["fidelity_cost_units_by_level"]
            if not isinstance(fidelity_counts, dict) or not fidelity_counts:
                errors.append("$.accounting.fidelity_evaluations: expected non-empty object")
            if not isinstance(fidelity_costs, dict) or not fidelity_costs:
                errors.append("$.accounting.fidelity_cost_units_by_level: expected non-empty object")
            if isinstance(fidelity_counts, dict) and isinstance(fidelity_costs, dict):
                key_pattern = re.compile(r"^[a-z][a-z0-9_-]*$")
                if any(not isinstance(k, str) or not key_pattern.fullmatch(k) for k in fidelity_counts):
                    errors.append("$.accounting.fidelity_evaluations: invalid fidelity level identifier")
                if any(not isinstance(k, str) or not key_pattern.fullmatch(k) for k in fidelity_costs):
                    errors.append("$.accounting.fidelity_cost_units_by_level: invalid fidelity level identifier")
                if set(fidelity_counts) != set(fidelity_costs):
                    errors.append("$.accounting: fidelity count and cost level keys must match exactly")
                valid_counts = all(_is_int(v) and v >= 0 for v in fidelity_counts.values())
                if not valid_counts:
                    errors.append("$.accounting.fidelity_evaluations: counts must be non-negative integers")
                elif _is_int(accounting["objective_evaluations"]) and sum(fidelity_counts.values()) != accounting["objective_evaluations"]:
                    errors.append("$.accounting.fidelity_evaluations: counts must sum exactly to objective_evaluations")
                if not all(_is_finite_number(v) and v >= 0 for v in fidelity_costs.values()):
                    errors.append("$.accounting.fidelity_cost_units_by_level: costs must be finite non-negative numbers")
        if isinstance(receipt.get("budget"), dict) and _is_int(accounting["objective_evaluations"]):
            if _is_int(receipt["budget"].get("objective_evaluations")) and accounting["objective_evaluations"] > receipt["budget"]["objective_evaluations"]:
                errors.append("$.accounting.objective_evaluations: exceeds frozen budget")
            if _is_int(receipt["budget"].get("initial_design_evaluations")) and receipt["budget"]["initial_design_evaluations"] > accounting["objective_evaluations"]:
                errors.append("$.budget.initial_design_evaluations: exceeds actual objective evaluations")

    if not isinstance(receipt["termination_reason"], str) or receipt["termination_reason"] not in TERMINATIONS:
        errors.append("$.termination_reason: unknown or non-terminal reason")
    if receipt["status"] == "success":
        if receipt["termination_reason"] not in {"budget_exhausted", "predeclared_target_reached"}:
            errors.append("$.status: success requires predeclared terminal condition; skipped/missing/error cannot pass")
        if "metrics" not in receipt or not isinstance(receipt.get("metrics"), dict):
            errors.append("$.metrics: successful receipt requires metric object")
    elif receipt.get("metrics") is not None and not isinstance(receipt.get("metrics"), dict):
        errors.append("$.metrics: expected object or null")

    metrics = receipt.get("metrics")
    if isinstance(metrics, dict):
        metric_allowed = {
            "final_hypervolume", "normalized_hypervolume_auc", "igd_plus", "spacing",
            "feasible_fraction", "invalid_or_failed_fraction", "evaluations_to_target",
            "post_selection_coverage", "regional_min_coverage", "weighted_ess",
            "mean_interval_width_proxy", "abstention_fraction",
            "cost_adjusted_normalized_hypervolume_auc", "false_promotion_rate",
            "high_fidelity_constraint_confirmation_rate", "high_fidelity_pareto_classification_error",
            "quality_at_fixed_high_fidelity_budget",
        }
        metric_required = {
            "final_hypervolume", "normalized_hypervolume_auc",
            "feasible_fraction", "invalid_or_failed_fraction",
        }
        if _closed_object(metrics, metric_required, metric_allowed, "$.metrics", errors):
            for key in ("final_hypervolume", "normalized_hypervolume_auc", "feasible_fraction", "invalid_or_failed_fraction"):
                value = metrics[key]
                if not _is_finite_number(value) or value < 0:
                    errors.append(f"$.metrics.{key}: expected finite non-negative number")
            if _is_finite_number(metrics["normalized_hypervolume_auc"]) and metrics["normalized_hypervolume_auc"] > 1:
                errors.append("$.metrics.normalized_hypervolume_auc: exceeds normalized range [0, 1]")
            for key in (
                "feasible_fraction", "invalid_or_failed_fraction",
                "post_selection_coverage", "regional_min_coverage", "abstention_fraction",
                "cost_adjusted_normalized_hypervolume_auc", "false_promotion_rate",
                "high_fidelity_constraint_confirmation_rate", "high_fidelity_pareto_classification_error",
                "quality_at_fixed_high_fidelity_budget",
            ):
                if key in metrics and metrics[key] is not None:
                    _check_fraction(metrics[key], f"$.metrics.{key}", errors)
            for key in ("igd_plus", "spacing", "weighted_ess", "mean_interval_width_proxy"):
                if key in metrics and metrics[key] is not None and (
                    not _is_finite_number(metrics[key]) or metrics[key] < 0
                ):
                    errors.append(f"$.metrics.{key}: expected null or finite non-negative number")
            if "evaluations_to_target" in metrics and metrics["evaluations_to_target"] is not None:
                v = metrics["evaluations_to_target"]
                if not _is_int(v) or v < 1:
                    errors.append("$.metrics.evaluations_to_target: expected null or positive integer")

    artifact_allowed = {"raw_trace_sha256", "result_sha256", "trace_schema_sha256", "scorer_sha256", "log_sha256"}
    artifact_required = {"raw_trace_sha256", "result_sha256", "trace_schema_sha256", "scorer_sha256"}
    if _closed_object(receipt["artifacts"], artifact_required, artifact_allowed, "$.artifacts", errors):
        for key in artifact_required:
            _check_hash(receipt["artifacts"][key], f"$.artifacts.{key}", errors)
        if receipt["artifacts"].get("log_sha256") is not None:
            _check_hash(receipt["artifacts"]["log_sha256"], "$.artifacts.log_sha256", errors)

    if "notes" in receipt and (not isinstance(receipt["notes"], str) or len(receipt["notes"]) > 4000):
        errors.append("$.notes: expected string up to 4000 characters")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="JSON receipt path")
    parser.add_argument("--for-publication", action="store_true",
                        help="reject synthetic fixture receipts as empirical evidence")
    args = parser.parse_args()
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID: unable to read valid JSON: {exc}", file=sys.stderr)
        return 2
    errors = validate_receipt(receipt, for_publication=args.for_publication)
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"VALID STRUCTURE: {receipt['run_id']} status={receipt['status']} synthetic_fixture_only={receipt['synthetic_fixture_only']}")
    if receipt["status"] != "success":
        print("NOT AN ALGORITHM PASS: receipt is not status=success")
        return 0
    if receipt["synthetic_fixture_only"]:
        print("FIXTURE ONLY: structurally valid, not empirical evidence")
        if args.for_publication:
            return 1
    else:
        print("TERMINAL SUCCESS RECEIPT: metric acceptance still requires independent recomputation and cross-run analysis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
