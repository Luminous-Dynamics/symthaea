#!/usr/bin/env python3
"""Independently replay and score compositional multi-fidelity traces.

This module intentionally does not import the scenario implementation or the
policy runner. Response equations and Halton design generation are re-derived
here so a mutation to the runner/scenario code is detectable during replay.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

BENCHMARK_ID = "CASTING-MF-COMPOSITION-001"
RECEIPT_LABEL = "COMPOSITIONAL_SYNTHETIC_TRACE_REPLAYED_NOT_SOTA_EVIDENCE"
SUPPORTED_FAMILIES = (
    "smooth_stationary",
    "local_bias_island",
    "warped_coordinates",
    "constraint_boundary",
    "discontinuous_step",
    "heteroscedastic_noise",
)
SUPPORTED_DIMENSIONS = (2, 5, 10)
POLICY_ORDER = ("high_only", "low_screen", "fixed_top_k", "discrepancy_aware", "adaptive_budget")
REFERENCE_POINT = (1.2, 1.2)
TOL = 1e-9
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
SOURCE_PATHS = (
    "tools/casting-benchmark/multifidelity_compositional_bakeoff.py",
    "tools/casting-benchmark/score_compositional_bakeoff.py",
    "tools/casting-benchmark/multifidelity_scenario_suite.py",
    "tools/casting-benchmark/multifidelity_stress.py",
    "docs/research/CASTING_MULTIFIDELITY_SCENARIO_SUITE_V1.json",
    "docs/research/CASTING_COMPOSITIONAL_BAKEOFF_V1.json",
)
TOP_KEYS = {
    "benchmark_id", "synthetic_fixture_only", "objective_families", "constraint_families",
    "dimensions", "severities", "seeds", "policies", "design_count", "cost_budget",
    "fidelity_costs", "run_count", "source_sha256", "paired_runs", "warning",
}
RUN_KEYS = {
    "benchmark_id", "synthetic_fixture_only", "objective_family", "constraint_family",
    "dimension", "severity", "seed", "policy", "design_count", "candidate_pool_sha256",
    "fidelity_costs", "cost_budget", "cost_spent", "unused_cost", "evaluation_count",
    "fidelity_evaluations", "fidelity_cost_units_by_level", "high_fidelity_call_count",
    "true_finite_pool_hypervolume", "observed_high_fidelity_hypervolume",
    "normalized_hypervolume_fraction", "promotion_records",
    "predicted_feasible_promotion_count", "false_promotion_count", "false_promotion_rate",
    "evaluations",
}
EVAL_KEYS = {"candidate_id", "fidelity", "objectives", "constraint_margin", "cost_units"}
PROMOTION_KEYS = {"candidate_id", "predicted_margin", "method", "observed_high_margin"}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def loads_no_duplicate_keys(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_lock(repo_root: Path) -> dict[str, str]:
    return {rel: _sha((repo_root / rel).read_bytes()) for rel in SOURCE_PATHS}


def _radical_inverse(index: int, base: int) -> float:
    value, factor = 0.0, 1.0 / base
    while index:
        index, digit = divmod(index, base)
        value += digit * factor
        factor /= base
    return value


def design_point(candidate_id: int, dimension: int, seed: int) -> tuple[float, ...]:
    offset = 1 + (seed % 100_003) * 13
    return tuple(_radical_inverse(offset + candidate_id, PRIMES[j]) for j in range(dimension))


def truth_at(x: tuple[float, ...]) -> tuple[tuple[float, float], float]:
    d = len(x)
    target0 = tuple(0.22 + 0.035 * math.sin(i + 1.0) for i in range(d))
    target1 = tuple(0.78 - 0.035 * math.cos(i + 1.0) for i in range(d))
    f0 = 0.08 + sum((x[i] - target0[i]) ** 2 for i in range(d)) / d
    f1 = 0.08 + sum((x[i] - target1[i]) ** 2 for i in range(d)) / d
    if d > 1:
        f0 += 0.012 * math.sin(4.0 * math.pi * x[0]) * math.cos(2.0 * math.pi * x[-1])
        f1 += 0.012 * math.cos(3.0 * math.pi * x[0]) * math.sin(2.0 * math.pi * x[-1])
    center = tuple(0.48 + 0.035 * math.sin(0.7 * (i + 1)) for i in range(d))
    radius = sum((x[i] - center[i]) ** 2 for i in range(d)) / d
    return (f0, f1), 0.075 - radius


def family_response(
    family: str, x: tuple[float, ...], fidelity: str, severity: float, seed: int
) -> tuple[tuple[float, float], float]:
    """Independent equation replay; must not import the generator implementation."""
    truth, truth_margin = truth_at(x)
    if fidelity == "high":
        return truth, truth_margin
    scale = severity * (0.38 if fidelity == "medium" else 1.0)
    x0, xlast = x[0], x[-1]
    f0, f1 = truth
    margin = truth_margin

    if family == "smooth_stationary":
        objectives = (
            f0 + scale * (0.12 * x0 + 0.045 * math.sin(2.0 * math.pi * (x0 + xlast))),
            f1 + scale * (-0.09 * xlast + 0.035 * math.cos(2.0 * math.pi * x0)),
        )
        margin += scale * (0.11 * math.sin(3.0 * math.pi * x0 + 2.0 * math.pi * xlast) - 0.018)
    elif family == "local_bias_island":
        center = tuple(0.72 if i % 2 == 0 else 0.28 for i in range(len(x)))
        distance = sum((x[i] - center[i]) ** 2 for i in range(len(x))) / len(x)
        bump = math.exp(-distance / (2.0 * 0.16 ** 2))
        objectives = (f0 + scale * 0.32 * bump, f1 - scale * 0.24 * bump)
        margin += scale * (-0.34 * bump + 0.015 * math.sin(5.0 * math.pi * x0))
    elif family == "warped_coordinates":
        warp = (0.045 if fidelity == "medium" else 0.15) * severity
        warped = tuple(
            min(1.0, max(0.0, value + warp * math.sin(
                2.0 * math.pi * (x[(i + 1) % len(x)] + i / 7.0)
            )))
            for i, value in enumerate(x)
        )
        objectives, margin = truth_at(warped)
        objectives = (
            objectives[0] + scale * 0.025 * math.sin(2.0 * math.pi * x0),
            objectives[1] - scale * 0.02 * math.cos(2.0 * math.pi * xlast),
        )
    elif family == "constraint_boundary":
        boundary_weight = math.exp(-abs(truth_margin) / 0.055)
        objectives = (
            f0 + scale * 0.035 * math.sin(5.0 * math.pi * x0),
            f1 + scale * 0.035 * math.cos(4.0 * math.pi * xlast),
        )
        margin += scale * 0.28 * math.sin(
            5.0 * math.pi * x0 + 3.0 * math.pi * xlast
        ) * boundary_weight
    elif family == "discontinuous_step":
        threshold = 0.61 if fidelity == "medium" else 0.48
        step = 1.0 if x0 >= threshold else -0.25
        objective_step = 0.16 if fidelity == "medium" else 0.30
        objectives = (f0 + scale * objective_step * step, f1 - scale * objective_step * 0.7 * step)
        margin += scale * (0.20 if x0 >= threshold else -0.11)
    elif family == "heteroscedastic_noise":
        phase = (seed % 997) * 0.013
        boundary_weight = 1.0 + 2.0 * math.exp(-abs(truth_margin) / 0.05)
        edge_weight = 1.0 + 0.8 * (abs(x0 - 0.5) + abs(xlast - 0.5))
        amplitude = scale * 0.035 * boundary_weight * edge_weight
        noise0 = amplitude * math.sin(13.0 * x0 + 17.0 * xlast + phase)
        noise1 = amplitude * math.cos(19.0 * x0 - 11.0 * xlast + phase)
        objectives = (f0 + noise0, f1 + noise1)
        margin += scale * 0.18 * math.sin(11.0 * x0 + 7.0 * xlast + phase) / boundary_weight
    else:
        raise ValueError(f"unsupported family: {family}")
    if not all(math.isfinite(value) for value in (*objectives, margin)):
        raise ValueError("replay equation produced non-finite value")
    return objectives, margin


def hypervolume_2d(points: list[tuple[float, float]]) -> float:
    unique = sorted(set(points))
    front = []
    best_y = math.inf
    for x, y in unique:
        if x >= REFERENCE_POINT[0] or y >= REFERENCE_POINT[1]:
            continue
        if y < best_y:
            front.append((x, y))
            best_y = y
    area = 0.0
    for i, (x, y) in enumerate(front):
        next_x = front[i + 1][0] if i + 1 < len(front) else REFERENCE_POINT[0]
        area += max(0.0, next_x - x) * max(0.0, REFERENCE_POINT[1] - y)
    if not math.isfinite(area):
        raise ValueError("hypervolume is non-finite")
    return area


def expected_call_counts(policy: str, costs: dict[str, int], budget: int, design_count: int) -> dict[str, int]:
    fixed = {
        "high_only": {"low": 0, "medium": 0, "high": 8},
        "low_screen": {"low": 75, "medium": 0, "high": 5},
        "fixed_top_k": {"low": 50, "medium": 10, "high": 4},
        "discrepancy_aware": {"low": 55, "medium": 4, "high": 5},
    }
    if policy in fixed:
        return fixed[policy]
    if policy != "adaptive_budget":
        raise ValueError(f"unrecognized frozen policy {policy!r}")
    low_cost, medium_cost, high_cost = costs["low"], costs["medium"], costs["high"]
    reserved_high = min(2, budget // high_cost)
    reserved_high_cost = reserved_high * high_cost
    low_target = min(75, max(5, int((budget * 0.30) // low_cost)))
    affordable_low = max(0, (budget - reserved_high_cost - 2 * medium_cost) // low_cost)
    low_target = min(low_target, max(5, affordable_low), design_count)
    medium_target = min(10, max(2, int((budget * 0.20) // medium_cost)))
    medium_target = min(medium_target, low_target)
    affordable_medium = max(0, (budget - low_target * low_cost - reserved_high_cost) // medium_cost)
    medium_target = min(medium_target, affordable_medium)
    if medium_target < 1:
        raise ValueError("budget cannot afford medium-fidelity probe")
    remaining = budget - low_target * low_cost - medium_target * medium_cost
    high_slots = min(low_target, remaining // high_cost)
    anchors = min(2, high_slots, medium_target)
    promotions = min(max(0, high_slots - anchors), low_target - anchors)
    return {"low": low_target, "medium": medium_target, "high": anchors + promotions}


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _near(actual: Any, expected: float, label: str, errors: list[str]) -> None:
    if not _finite(actual) or abs(float(actual) - expected) > TOL:
        errors.append(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _require_keys(value: Any, expected: set[str], label: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return False
    if set(value) != expected:
        errors.append(f"{label} keys mismatch: missing={sorted(expected-set(value))}, extra={sorted(set(value)-expected)}")
        return False
    return True


def score_document(document: dict, repo_root: Path | None = None) -> dict:
    root = repo_root or Path(__file__).resolve().parents[2]
    errors: list[str] = []
    if not _require_keys(document, TOP_KEYS, "document", errors):
        raise ValueError("; ".join(errors))
    if document["benchmark_id"] != BENCHMARK_ID or document["synthetic_fixture_only"] is not True:
        errors.append("benchmark identity or synthetic-only declaration is invalid")
    if isinstance(document["design_count"], bool) or not isinstance(document["design_count"], int) or document["design_count"] < 75:
        errors.append("design_count must be an integer at least 75")
    if not isinstance(document["cost_budget"], int) or isinstance(document["cost_budget"], bool) or document["cost_budget"] < 1:
        errors.append("cost_budget must be a positive integer")
    costs = document["fidelity_costs"]
    if not isinstance(costs, dict) or set(costs) != {"low", "medium", "high"} or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (costs.values() if isinstance(costs, dict) else [])
    ):
        errors.append("invalid fidelity cost map")
        costs = {"low": 1, "medium": 5, "high": 25}
    if document["source_sha256"] != source_lock(root):
        errors.append("source SHA-256 lock differs from checked-out source/manifest bytes")

    def valid_sequence(key: str, allowed: tuple | None = None, kind: str = "str") -> list:
        seq = document[key]
        if not isinstance(seq, list) or not seq:
            errors.append(f"{key} must be a non-empty list")
            return []
        if kind == "str":
            well_typed = all(isinstance(v, str) for v in seq)
        elif kind == "int":
            well_typed = all(isinstance(v, int) and not isinstance(v, bool) for v in seq)
        elif kind == "number":
            well_typed = all(_finite(v) for v in seq)
        else:
            well_typed = False
        if not well_typed:
            errors.append(f"{key} contains values of an invalid type")
            return []
        if len(set(seq)) != len(seq):
            errors.append(f"{key} must not contain duplicates")
            return []
        if allowed is not None and any(v not in allowed for v in seq):
            errors.append(f"{key} contains unsupported values")
        return seq

    objective_families = valid_sequence("objective_families", SUPPORTED_FAMILIES)
    constraint_families = valid_sequence("constraint_families", SUPPORTED_FAMILIES)
    dimensions = valid_sequence("dimensions", SUPPORTED_DIMENSIONS, "int")
    severities = valid_sequence("severities", kind="number")
    seeds = valid_sequence("seeds", kind="int")
    policies = valid_sequence("policies", POLICY_ORDER)
    if errors:
        raise ValueError("compositional replay rejected: " + "; ".join(errors[:20]))
    if any(float(s) < 0.0 for s in severities if _finite(s)):
        errors.append("severities must be non-negative")
    if any(float(s) not in (0.0, 0.5, 1.0, 1.5, 2.0) for s in severities if _finite(s)):
        errors.append("severities are outside the preregistered diagnostic set")
    if document["policies"] != [p for p in POLICY_ORDER if p in policies]:
        errors.append("policy ordering differs from frozen policy order")

    expected_count = len(objective_families) * len(constraint_families) * len(dimensions) * len(severities) * len(seeds) * len(policies)
    _near(document["run_count"], float(expected_count), "run_count", errors)
    if not isinstance(document["paired_runs"], list) or len(document["paired_runs"]) != expected_count:
        errors.append("paired_runs length does not match the declared Cartesian product")
    runs = document["paired_runs"] if isinstance(document["paired_runs"], list) else []
    indexed = {}

    for ri, run in enumerate(runs):
        label = f"run[{ri}]"
        if not _require_keys(run, RUN_KEYS, label, errors):
            continue
        if run["benchmark_id"] != BENCHMARK_ID or run["synthetic_fixture_only"] is not True:
            errors.append(f"{label} benchmark identity is invalid")
            continue
        obj_family, con_family, dim, sev, seed, policy = (
            run["objective_family"], run["constraint_family"], run["dimension"],
            run["severity"], run["seed"], run["policy"]
        )
        if (
            not isinstance(obj_family, str) or not isinstance(con_family, str)
            or isinstance(dim, bool) or not isinstance(dim, int)
            or not _finite(sev)
            or isinstance(seed, bool) or not isinstance(seed, int)
            or not isinstance(policy, str)
        ):
            errors.append(f"{label} has malformed run identity fields")
            continue
        key = (obj_family, con_family, dim, float(sev), seed, policy)
        if key in indexed:
            errors.append(f"duplicate run key: {key}")
        indexed[key] = run
        if obj_family not in objective_families or con_family not in constraint_families:
            errors.append(f"{label} family not registered in top-level contract")
            continue
        if dim not in dimensions or float(sev) not in severities or seed not in seeds or policy not in policies:
            errors.append(f"{label} key not registered in top-level contract")
            continue
        if run["design_count"] != document["design_count"] or run["cost_budget"] != document["cost_budget"] or run["fidelity_costs"] != costs:
            errors.append(f"{label} run configuration differs from top-level settings")
        count_map = {"low": 0, "medium": 0, "high": 0}
        cost_map = {"low": 0, "medium": 0, "high": 0}
        seen_eval_ids = []
        observed_high = {}
        ledger = run["evaluations"]
        if not isinstance(ledger, list):
            errors.append(f"{label}.evaluations must be a list")
            ledger = []
        for ei, evaluation in enumerate(ledger):
            elabel = f"{label}.evaluations[{ei}]"
            if not _require_keys(evaluation, EVAL_KEYS, elabel, errors):
                continue
            cid, fidelity = evaluation["candidate_id"], evaluation["fidelity"]
            if isinstance(cid, bool) or not isinstance(cid, int) or not 0 <= cid < document["design_count"]:
                errors.append(f"{elabel} invalid candidate_id")
                continue
            if fidelity not in count_map:
                errors.append(f"{elabel} invalid fidelity")
                continue
            seen_eval_ids.append((cid, fidelity))
            count_map[fidelity] += 1
            expected_cost = costs[fidelity]
            cost_map[fidelity] += expected_cost
            _near(evaluation["cost_units"], expected_cost, f"{elabel}.cost_units", errors)
            x = design_point(cid, dim, seed)
            expected_obj, _ = family_response(obj_family, x, fidelity, float(sev), seed)
            _, expected_margin = family_response(con_family, x, fidelity, float(sev), seed)
            actual_objectives = evaluation["objectives"]
            if not isinstance(actual_objectives, list) or len(actual_objectives) != 2:
                errors.append(f"{elabel}.objectives must contain two values")
            else:
                _near(actual_objectives[0], expected_obj[0], f"{elabel}.objective[0]", errors)
                _near(actual_objectives[1], expected_obj[1], f"{elabel}.objective[1]", errors)
            _near(evaluation["constraint_margin"], expected_margin, f"{elabel}.constraint_margin", errors)
            if fidelity == "high":
                observed_high[cid] = (tuple(expected_obj), expected_margin)

        _near(run["evaluation_count"], len(seen_eval_ids), f"{label}.evaluation_count", errors)
        _near(run["high_fidelity_call_count"], count_map["high"], f"{label}.high_fidelity_call_count", errors)
        if run["fidelity_evaluations"] != count_map:
            errors.append(f"{label} fidelity call counts disagree with ledger")
        if run["fidelity_cost_units_by_level"] != cost_map:
            errors.append(f"{label} per-fidelity costs disagree with ledger")
        actual_spend = sum(cost_map.values())
        _near(run["cost_spent"], actual_spend, f"{label}.cost_spent", errors)
        _near(run["unused_cost"], document["cost_budget"] - actual_spend, f"{label}.unused_cost", errors)
        if actual_spend > document["cost_budget"]:
            errors.append(f"{label} exceeds budget")
        try:
            expected_counts = expected_call_counts(policy, costs, document["cost_budget"], document["design_count"])
            if count_map != expected_counts:
                errors.append(f"{label} call template mismatch: expected {expected_counts}, got {count_map}")
        except ValueError as exc:
            errors.append(f"{label} invalid call template: {exc}")

        if len(ledger) != run["evaluation_count"]:
            errors.append(f"{label} ledger length disagrees with evaluation_count")
        payload = [
            [cid, list(design_point(cid, dim, seed))]
            for cid in range(document["design_count"])
        ]
        pool_hash = _sha(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        if run["candidate_pool_sha256"] != pool_hash:
            errors.append(f"{label} candidate pool hash mismatch")

        truth_points = []
        for cid in range(document["design_count"]):
            x = design_point(cid, dim, seed)
            obj, margin = truth_at(x)
            if margin >= 0.0:
                truth_points.append(obj)
        true_hv = hypervolume_2d(truth_points)
        observed_points = [obj for obj, margin in observed_high.values() if margin >= 0.0]
        observed_hv = hypervolume_2d(observed_points)
        _near(run["true_finite_pool_hypervolume"], true_hv, f"{label}.true_hv", errors)
        _near(run["observed_high_fidelity_hypervolume"], observed_hv, f"{label}.observed_hv", errors)
        _near(run["normalized_hypervolume_fraction"], observed_hv / true_hv if true_hv > 0.0 else 0.0, f"{label}.normalized_hv", errors)

        if not isinstance(run["promotion_records"], list):
            errors.append(f"{label}.promotion_records must be a list")
            continue
        false_promotions = 0
        predicted_feasible = 0
        for pi, promotion in enumerate(run["promotion_records"]):
            plabel = f"{label}.promotion_records[{pi}]"
            if not _require_keys(promotion, PROMOTION_KEYS, plabel, errors):
                continue
            cid = promotion["candidate_id"]
            if isinstance(cid, bool) or not isinstance(cid, int) or cid not in observed_high:
                errors.append(f"{plabel} lacks corresponding high-fidelity observation")
                continue
            _near(promotion["observed_high_margin"], observed_high[cid][1], f"{plabel}.observed_high_margin", errors)
            if not _finite(promotion["predicted_margin"]) or not isinstance(promotion["method"], str) or not promotion["method"]:
                errors.append(f"{plabel} has invalid promotion fields")
                continue
            predicted_feasible += int(promotion["predicted_margin"] >= 0.0)
            false_promotions += int(promotion["predicted_margin"] >= 0.0 and observed_high[cid][1] < 0.0)
        _near(run["predicted_feasible_promotion_count"], predicted_feasible, f"{label}.predicted_feasible_promotion_count", errors)
        _near(run["false_promotion_count"], false_promotions, f"{label}.false_promotion_count", errors)
        rate = false_promotions / predicted_feasible if predicted_feasible else None
        if rate is None:
            if run["false_promotion_rate"] is not None:
                errors.append(f"{label}.false_promotion_rate should be null")
        else:
            _near(run["false_promotion_rate"], rate, f"{label}.false_promotion_rate", errors)

    expected_keys = {
        (obj, con, dim, float(sev), seed, policy)
        for obj, con, dim, sev, seed, policy in itertools.product(
            objective_families, constraint_families, dimensions, severities, seeds, policies
        )
    }
    if set(indexed) != expected_keys:
        errors.append(
            f"run-key coverage mismatch: missing={len(expected_keys-set(indexed))}, extra={len(set(indexed)-expected_keys)}"
        )
    if errors:
        raise ValueError("compositional replay rejected: " + "; ".join(errors[:20]))

    metrics = []
    for key, run in sorted(indexed.items(), key=lambda item: item[0]):
        metrics.append({
            "objective_family": run["objective_family"],
            "constraint_family": run["constraint_family"],
            "dimension": run["dimension"],
            "severity": run["severity"],
            "seed": run["seed"],
            "policy": run["policy"],
            "normalized_hypervolume_fraction": run["normalized_hypervolume_fraction"],
            "cost_spent": run["cost_spent"],
            "high_fidelity_call_count": run["high_fidelity_call_count"],
            "false_promotion_count": run["false_promotion_count"],
            "false_promotion_rate": run["false_promotion_rate"],
        })

    summaries = []
    comparisons = []
    for obj, con, dim, sev in itertools.product(objective_families, constraint_families, dimensions, severities):
        by_policy = {}
        for policy in policies:
            group = [indexed[(obj, con, dim, float(sev), seed, policy)] for seed in seeds]
            hv_values = [float(r["normalized_hypervolume_fraction"]) for r in group]
            false_rates = [float(r["false_promotion_rate"]) for r in group if r["false_promotion_rate"] is not None]
            summary = {
                "objective_family": obj,
                "constraint_family": con,
                "dimension": dim,
                "severity": sev,
                "policy": policy,
                "paired_seed_count": len(group),
                "median_normalized_hypervolume_fraction": statistics.median(hv_values),
                "mean_normalized_hypervolume_fraction": statistics.fmean(hv_values),
                "mean_false_promotion_rate": statistics.fmean(false_rates) if false_rates else None,
                "mean_cost_spent": statistics.fmean(float(r["cost_spent"]) for r in group),
                "mean_high_fidelity_calls": statistics.fmean(float(r["high_fidelity_call_count"]) for r in group),
                "inference_status": "diagnostic_only_requires_at_least_20_paired_seeds",
            }
            summaries.append(summary)
            by_policy[policy] = group
        if "high_only" in policies:
            for policy in policies:
                if policy == "high_only":
                    continue
                diffs = [
                    float(indexed[(obj, con, dim, float(sev), seed, policy)]["normalized_hypervolume_fraction"])
                    - float(indexed[(obj, con, dim, float(sev), seed, "high_only")]["normalized_hypervolume_fraction"])
                    for seed in seeds
                ]
                comparisons.append({
                    "objective_family": obj,
                    "constraint_family": con,
                    "dimension": dim,
                    "severity": sev,
                    "reference_policy": "high_only",
                    "policy": policy,
                    "paired_seed_count": len(diffs),
                    "mean_paired_delta": statistics.fmean(diffs),
                    "median_paired_delta": statistics.median(diffs),
                    "wins": sum(v > 1e-12 for v in diffs),
                    "ties": sum(abs(v) <= 1e-12 for v in diffs),
                    "losses": sum(v < -1e-12 for v in diffs),
                    "raw_paired_deltas_by_seed": {str(seed): diff for seed, diff in zip(seeds, diffs)},
                    "inference_status": "diagnostic_only_unadjusted_not_sota_evidence",
                })

    raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "benchmark_id": BENCHMARK_ID,
        "receipt_label": RECEIPT_LABEL,
        "valid": True,
        "result_sha256": _sha(raw),
        "source_sha256": source_lock(root),
        "source_lock_semantics": "Compared against checked-out bytes at replay time; unsigned and not proof of who executed the run.",
        "run_count": len(metrics),
        "run_metrics": metrics,
        "cell_summary_count": len(summaries),
        "cell_summaries": summaries,
        "paired_comparison_count": len(comparisons),
        "paired_comparisons_vs_high_only": comparisons,
        "limitations": [
            "Known-truth synthetic composition only; no external SOTA or casting validity is established.",
            "The default three seeds are diagnostic, not sufficient for inferential superiority claims.",
            "The scorer independently reimplements equations, Halton generation, ledger checks and hypervolume, but a receipt remains unsigned.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        document = loads_no_duplicate_keys(args.result_json.read_text(encoding="utf-8"))
        receipt = score_document(document, root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"REJECTED: {exc}\n")
    raw = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.receipt:
        args.receipt.write_text(raw, encoding="utf-8")
    else:
        print(raw, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
