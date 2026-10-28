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
POLICY_ORDER = (
    "high_only", "low_screen", "fixed_top_k", "discrepancy_aware",
    "adaptive_budget", "decision_value_of_information", "joint_action_cost_aware",
)
ALLOWED_PROMOTION_METHODS = {
    "high_only": set(),
    "low_screen": {"low_predicted_margin"},
    "fixed_top_k": {"medium_top_k"},
    "discrepancy_aware": {"discrepancy_anchor", "discrepancy_corrected"},
    "adaptive_budget": {"adaptive_discrepancy_anchor", "adaptive_cost_aware_corrected"},
    "decision_value_of_information": {"decision_impact_anchor", "decision_impact_confirmation"},
    "joint_action_cost_aware": {"joint_action_cost_aware"},
}
REFERENCE_POINT = (1.2, 1.2)
TOL = 1e-9
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
SOURCE_PATHS = (
    "tools/casting-benchmark/multifidelity_compositional_bakeoff.py",
    "tools/casting-benchmark/multifidelity_decision_value_policy.py",
    "tools/casting-benchmark/multifidelity_joint_action_policy.py",
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
    "normalized_hypervolume_fraction", "promotion_records", "decision_value_trace",
    "predicted_feasible_promotion_count", "false_promotion_count", "false_promotion_rate",
    "evaluations", "joint_action_trace",
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
        "decision_value_of_information": {"low": 60, "medium": 8, "high": 4},
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



TRACE_KEYS = {
    "step", "high_fidelity_candidates_before_decision",
    "candidate_scores", "selected_candidate_id",
}
CANDIDATE_SCORE_KEYS = {
    "candidate_id", "predicted_objectives", "predicted_margin", "sigma",
    "probability_feasible_proxy", "normalized_expected_hypervolume_gain",
    "boundary_information", "objective_information", "score",
}
DIWA_WEIGHTS = {
    "expected_hypervolume_gain": 0.70,
    "constraint_boundary_information": 0.20,
    "frontier_objective_information": 0.10,
}
DIWA_FIXED_PARAMETERS = {
    "neighborhood_size": 4,
    "inverse_distance_offset": 0.02,
    "uncertainty_boundary_scale": 0.08,
    "objective_uncertainty_scale": 0.20,
    "frontier_distance_scale": 0.10,
    "robust_sigma_mad_multiplier": 1.4826,
    "uncertainty_extrapolation_multiplier": 0.35,
    "minimum_sigma": 0.000001,
    "medium_probe_rule": "two closest absolute constraint margins, best observed value for each objective, then diverse scalarization picks to fill eight unique candidates",
    "anchor_rule": "first medium candidate closest to the feasibility boundary; second maximizes input-space RMS distance from the first",
    "confirmation_rule": "recompute all remaining low-sample candidate predictions after each high call and choose the maximum independently replayable score; ties break on lowest candidate ID",
    "audit": "For each of the two confirmation decisions, store all eligible candidate scores and prediction components. The scorer independently recomputes predictions, score components, score ordering and selected candidate from the raw trace.",
}
DIWA_POLICY = "decision_value_of_information"


def _diwa_distance(candidate_a: int, candidate_b: int, dimension: int, seed: int) -> float:
    xa, xb = design_point(candidate_a, dimension, seed), design_point(candidate_b, dimension, seed)
    return math.sqrt(statistics.fmean((a - b) ** 2 for a, b in zip(xa, xb)))


def _diwa_robust_sigma(values: list[float]) -> float:
    if not values:
        return 1e-6
    center = statistics.median(values)
    mad = statistics.median([abs(value - center) for value in values])
    mean = statistics.fmean(values)
    variance = statistics.fmean((value - mean) ** 2 for value in values)
    return max(1e-6, 1.4826 * mad, math.sqrt(max(0.0, variance)) * 0.5)


def _diwa_weighted_local(
    candidate_id: int,
    references: list[dict],
    deltas: dict[int, tuple[float, ...]],
    dimension: int,
    seed: int,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    if not references:
        width = len(next(iter(deltas.values()))) if deltas else 3
        return tuple(0.0 for _ in range(width)), tuple(0.1 for _ in range(width)), 1.0
    target = design_point(candidate_id, dimension, seed)
    ranked = sorted(
        ((_diwa_distance(candidate_id, int(obs["candidate_id"]), dimension, seed), int(obs["candidate_id"]))
         for obs in references),
        key=lambda item: (item[0], item[1]),
    )
    nearest = ranked[:min(4, len(ranked))]
    raw_weights = [1.0 / ((distance + 0.02) ** 2) for distance, _ in nearest]
    total_weight = sum(raw_weights)
    weights = [weight / total_weight for weight in raw_weights]
    width = len(next(iter(deltas.values())))
    means = tuple(
        sum(weight * deltas[candidate][j] for weight, (_, candidate) in zip(weights, nearest))
        for j in range(width)
    )
    global_scales = [
        _diwa_robust_sigma([delta[j] for delta in deltas.values()])
        for j in range(width)
    ]
    local_scales = []
    nearest_distance = nearest[0][0]
    for j in range(width):
        local_var = sum(
            weight * (deltas[candidate][j] - means[j]) ** 2
            for weight, (_, candidate) in zip(weights, nearest)
        )
        local_scales.append(max(
            1e-6,
            math.sqrt(local_var + (0.35 * global_scales[j]) ** 2
                      + (nearest_distance * global_scales[j]) ** 2),
        ))
    return means, tuple(local_scales), nearest_distance


def _diwa_predict(
    candidate_id: int,
    low_by_id: dict[int, dict],
    medium: list[dict],
    anchor_high: list[dict],
    dimension: int,
    seed: int,
) -> tuple[tuple[float, float], float, tuple[float, float, float]]:
    medium_by_id = {int(obs["candidate_id"]): obs for obs in medium}
    anchor_ids = {int(obs["candidate_id"]) for obs in anchor_high}
    paired_low = [obs for obs in medium if int(obs["candidate_id"]) in low_by_id]
    low_to_medium = {
        int(obs["candidate_id"]): (
            obs["objectives"][0] - low_by_id[int(obs["candidate_id"])]["objectives"][0],
            obs["objectives"][1] - low_by_id[int(obs["candidate_id"])]["objectives"][1],
            obs["constraint_margin"] - low_by_id[int(obs["candidate_id"])]["constraint_margin"],
        )
        for obs in paired_low
    }
    high_by_id = {int(obs["candidate_id"]): obs for obs in anchor_high}
    high_to_medium = {
        int(obs["candidate_id"]): (
            high_by_id[int(obs["candidate_id"])]["objectives"][0] - obs["objectives"][0],
            high_by_id[int(obs["candidate_id"])]["objectives"][1] - obs["objectives"][1],
            high_by_id[int(obs["candidate_id"])]["constraint_margin"] - obs["constraint_margin"],
        )
        for obs in medium if int(obs["candidate_id"]) in anchor_ids
    }
    first_mean, first_sigma, _ = _diwa_weighted_local(
        candidate_id, paired_low, low_to_medium, dimension, seed
    )
    anchor_refs = [medium_by_id[cid] for cid in [int(obs["candidate_id"]) for obs in anchor_high]]
    second_mean, second_sigma, _ = _diwa_weighted_local(
        candidate_id, anchor_refs, high_to_medium, dimension, seed
    )
    low = low_by_id[candidate_id]
    objectives = (
        low["objectives"][0] + first_mean[0] + second_mean[0],
        low["objectives"][1] + first_mean[1] + second_mean[1],
    )
    margin = low["constraint_margin"] + first_mean[2] + second_mean[2]
    sigma = tuple(
        math.sqrt(first_sigma[j] ** 2 + second_sigma[j] ** 2) for j in range(3)
    )
    return objectives, margin, sigma


def _diwa_pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    front = []
    best_y = math.inf
    for x, y in unique:
        if x >= REFERENCE_POINT[0] or y >= REFERENCE_POINT[1]:
            continue
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


def _diwa_components(
    objectives: tuple[float, float],
    margin: float,
    sigma: tuple[float, float, float],
    feasible_archive: list[dict],
) -> dict[str, float]:
    sigma_f0, sigma_f1, sigma_margin = sigma
    probability_feasible = 0.5 * (
        1.0 + math.erf(margin / (max(sigma_margin, 1e-6) * math.sqrt(2.0)))
    )
    archive_points = [
        tuple(obs["objectives"]) for obs in feasible_archive
        if obs["constraint_margin"] >= 0.0
    ]
    current_hv = hypervolume_2d(archive_points)
    proposed_hv = hypervolume_2d(archive_points + [objectives])
    normalized_gain = max(0.0, proposed_hv - current_hv) / (1.2 * 1.2)
    boundary_information = min(1.0, sigma_margin / 0.08) * math.exp(
        -abs(margin) / max(sigma_margin, 1e-6)
    )
    objective_uncertainty = min(1.0, (sigma_f0 + sigma_f1) / 0.20)
    front = _diwa_pareto_front(archive_points)
    frontier_distance = (
        min(math.sqrt((objectives[0] - x) ** 2 + (objectives[1] - y) ** 2) for x, y in front)
        if front else 0.0
    )
    frontier_proximity = math.exp(-frontier_distance / 0.10)
    objective_information = objective_uncertainty * frontier_proximity
    score = (
        DIWA_WEIGHTS["expected_hypervolume_gain"] * probability_feasible * normalized_gain
        + DIWA_WEIGHTS["constraint_boundary_information"] * boundary_information
        + DIWA_WEIGHTS["frontier_objective_information"] * objective_information
    )
    return {
        "probability_feasible_proxy": probability_feasible,
        "normalized_expected_hypervolume_gain": normalized_gain,
        "boundary_information": boundary_information,
        "objective_information": objective_information,
        "score": score,
    }


def _replay_diwa_run(run: dict, errors: list[str]) -> None:
    """Re-derive the new policy's prediction/score trace from its raw oracle ledger."""
    label = f"run[{run.get('policy')}] DIWA replay"
    trace = run.get("decision_value_trace")
    if run.get("policy") != DIWA_POLICY:
        if trace != []:
            errors.append(f"{label}: non-DIWA policy must have an empty decision_value_trace")
        return
    if not isinstance(trace, list) or len(trace) != 2:
        errors.append(f"{label}: expected two sequential acquisition decisions")
        return
    ledger = run.get("evaluations")
    if not isinstance(ledger, list) or len(ledger) != 72 or any(not isinstance(row, dict) for row in ledger):
        errors.append(f"{label}: expected 72 ordered evaluation objects")
        return
    for index, row in enumerate(ledger):
        cid, fidelity = row.get("candidate_id"), row.get("fidelity")
        objectives, margin, cost_units = row.get("objectives"), row.get("constraint_margin"), row.get("cost_units")
        if (
            isinstance(cid, bool) or not isinstance(cid, int)
            or not isinstance(fidelity, str) or fidelity not in {"low", "medium", "high"}
            or not isinstance(objectives, list) or len(objectives) != 2
            or any(not _finite(value) for value in objectives)
            or not _finite(margin) or not _finite(cost_units)
        ):
            errors.append(f"{label}: malformed evaluation record at index {index}")
            return
    raw_promotions = run.get("promotion_records")
    if not isinstance(raw_promotions, list) or any(not isinstance(promotion, dict) for promotion in raw_promotions):
        errors.append(f"{label}: expected a list of promotion objects")
        return
    for index, promotion in enumerate(raw_promotions):
        if (
            isinstance(promotion.get("candidate_id"), bool)
            or not isinstance(promotion.get("candidate_id"), int)
            or not _finite(promotion.get("predicted_margin"))
            or not _finite(promotion.get("observed_high_margin"))
            or not isinstance(promotion.get("method"), str)
        ):
            errors.append(f"{label}: malformed promotion record at index {index}")
            return
    if any(row.get("fidelity") != "low" for row in ledger[:60]):
        errors.append(f"{label}: first 60 evaluations must be low fidelity")
        return
    if any(row.get("fidelity") != "medium" for row in ledger[60:68]):
        errors.append(f"{label}: evaluations 61-68 must be medium fidelity")
        return
    if any(row.get("fidelity") != "high" for row in ledger[68:72]):
        errors.append(f"{label}: final four evaluations must be high fidelity")
        return
    dimension, seed = run["dimension"], run["seed"]
    low = ledger[:60]
    medium = ledger[60:68]
    high = ledger[68:72]
    low_by_id = {int(obs["candidate_id"]): obs for obs in low}
    medium_by_id = {int(obs["candidate_id"]): obs for obs in medium}
    if len(low_by_id) != 60 or len(medium_by_id) != 8:
        errors.append(f"{label}: duplicate candidate within low or medium sample")
        return
    anchor_promotions = [
        promotion for promotion in raw_promotions
        if promotion.get("method") == "decision_impact_anchor"
    ]
    confirmation_promotions = [
        promotion for promotion in raw_promotions
        if promotion.get("method") == "decision_impact_confirmation"
    ]
    if len(anchor_promotions) != 2 or len(confirmation_promotions) != 2:
        errors.append(f"{label}: expected two anchor and two confirmation promotion records")
        return
    anchor_ids = [int(p["candidate_id"]) for p in anchor_promotions]
    actual_anchor_ids = [int(row["candidate_id"]) for row in high[:2]]
    if anchor_ids != actual_anchor_ids:
        errors.append(f"{label}: anchor promotion order disagrees with high-fidelity ledger")
        return
    expected_anchor_first = min(
        medium,
        key=lambda obs: (abs(float(obs["constraint_margin"])), int(obs["candidate_id"])),
    )
    second_candidates = [
        obs for obs in medium if int(obs["candidate_id"]) != int(expected_anchor_first["candidate_id"])
    ]
    expected_anchor_second = max(
        second_candidates,
        key=lambda obs: (
            _diwa_distance(int(obs["candidate_id"]), int(expected_anchor_first["candidate_id"]), dimension, seed),
            int(obs["candidate_id"]),
        ),
    )
    if anchor_ids != [int(expected_anchor_first["candidate_id"]), int(expected_anchor_second["candidate_id"])]:
        errors.append(f"{label}: anchor selection does not match the frozen anchor rule")
    anchor_high = high[:2]
    promotions_by_id = {
        int(p["candidate_id"]): p for p in raw_promotions
        if isinstance(p.get("candidate_id"), int) and not isinstance(p.get("candidate_id"), bool)
    }
    for high_obs in anchor_high:
        cid = int(high_obs["candidate_id"])
        _near(
            promotions_by_id[cid].get("predicted_margin"),
            float(medium_by_id[cid]["constraint_margin"]),
            f"{label} anchor predicted margin for {cid}",
            errors,
        )

    for step, entry in enumerate(trace, start=1):
        if not _require_keys(entry, TRACE_KEYS, f"{label}.trace[{step-1}]", errors):
            continue
        if entry["step"] != step:
            errors.append(f"{label}: trace step ordering mismatch")
        prior_high = high[:2 + step - 1]
        prior_high_ids = [int(obs["candidate_id"]) for obs in prior_high]
        if entry["high_fidelity_candidates_before_decision"] != prior_high_ids:
            errors.append(f"{label}: prior high-fidelity state does not match step {step}")
        archive = [obs for obs in prior_high if float(obs["constraint_margin"]) >= 0.0]
        expected_candidates = set(low_by_id) - set(prior_high_ids)
        rows = entry["candidate_scores"]
        if not isinstance(rows, list) or len(rows) != len(expected_candidates):
            errors.append(f"{label}: candidate score coverage mismatch at step {step}")
            continue
        scored = {}
        for index, row in enumerate(rows):
            row_label = f"{label}.trace[{step-1}].candidate_scores[{index}]"
            if not _require_keys(row, CANDIDATE_SCORE_KEYS, row_label, errors):
                continue
            cid = row["candidate_id"]
            if isinstance(cid, bool) or not isinstance(cid, int) or cid not in expected_candidates:
                errors.append(f"{row_label}: candidate is not an available low-fidelity candidate")
                continue
            if cid in scored:
                errors.append(f"{row_label}: duplicate candidate score")
                continue
            predicted, margin, sigma = _diwa_predict(cid, low_by_id, medium, anchor_high, dimension, seed)
            _near(row["predicted_objectives"][0] if isinstance(row["predicted_objectives"], list) and len(row["predicted_objectives"]) == 2 else None, predicted[0], f"{row_label} predicted objective 0", errors)
            _near(row["predicted_objectives"][1] if isinstance(row["predicted_objectives"], list) and len(row["predicted_objectives"]) == 2 else None, predicted[1], f"{row_label} predicted objective 1", errors)
            _near(row["predicted_margin"], margin, f"{row_label} predicted margin", errors)
            for j in range(3):
                _near(row["sigma"][j] if isinstance(row["sigma"], list) and len(row["sigma"]) == 3 else None, sigma[j], f"{row_label} sigma {j}", errors)
            components = _diwa_components(predicted, margin, sigma, archive)
            for key, value in components.items():
                _near(row.get(key), value, f"{row_label} {key}", errors)
            scored[cid] = components["score"]
        if set(scored) != expected_candidates:
            errors.append(f"{label}: missing or extra score candidates at step {step}")
            continue
        ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
        expected_winner = ranked[0][0]
        if entry["selected_candidate_id"] != expected_winner:
            errors.append(f"{label}: selected candidate is not the independently recomputed maximum at step {step}")
        if [row.get("candidate_id") for row in rows] != [cid for cid, _ in ranked]:
            errors.append(f"{label}: candidate score rows are not correctly ordered at step {step}")
        actual_confirmation = int(high[1 + step]["candidate_id"])
        if actual_confirmation != expected_winner:
            errors.append(f"{label}: high-fidelity confirmation does not match independently recomputed winner at step {step}")
        promotion = promotions_by_id.get(expected_winner)
        if promotion is None or promotion.get("method") != "decision_impact_confirmation":
            errors.append(f"{label}: missing confirmation promotion record for selected candidate")
        else:
            predicted, margin, _ = _diwa_predict(
                expected_winner, low_by_id, medium, anchor_high, dimension, seed
            )
            _near(promotion.get("predicted_margin"), margin, f"{label} selected promotion margin", errors)

def score_document(
    document: dict,
    repo_root: Path | None = None,
    allow_diagnostic_subset: bool = False,
) -> dict:
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

    # The sealed manifest defines the formal default matrix. Arbitrary subsets are
    # useful for rapid diagnosis, but must be explicitly labelled as such rather
    # than receiving an indistinguishable valid receipt.
    manifest_path = root / "docs/research/CASTING_COMPOSITIONAL_BAKEOFF_V1.json"
    manifest: dict[str, Any] = {}
    try:
        manifest = loads_no_duplicate_keys(manifest_path.read_text(encoding="utf-8"))
        manifest_defaults = {
            "objective_families": manifest["scenarios"]["objective_families"],
            "constraint_families": manifest["scenarios"]["constraint_families"],
            "dimensions": manifest["design_space"]["dimensions"],
            "severities": manifest["default_sweep"]["severities"],
            "seeds": manifest["default_sweep"]["seeds"],
            "policies": [item["id"] for item in manifest["policies"]],
            "design_count": manifest["candidate_pool"]["design_count"],
            "cost_budget": manifest["budget"]["budget_units"],
            "fidelity_costs": manifest["budget"]["costs"],
        }
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"cannot load frozen composition manifest: {exc}")
        manifest_defaults = {}
    declared_defaults = {
        key: document.get(key) for key in (
            "objective_families", "constraint_families", "dimensions", "severities",
            "seeds", "policies", "design_count", "cost_budget", "fidelity_costs"
        )
    }
    diwa_contract = manifest.get("decision_value_policy", {})
    if diwa_contract.get("id") != DIWA_POLICY:
        errors.append("decision-impact policy identifier differs from scorer contract")
    if diwa_contract.get("fidelity_call_template") != {"low": 60, "medium": 8, "high": 4}:
        errors.append("decision-impact policy call template differs from scorer contract")
    if diwa_contract.get("cost_units") != {"low": 1, "medium": 5, "high": 25}:
        errors.append("decision-impact fidelity costs differ from scorer contract")
    if diwa_contract.get("frozen_score_weights") != DIWA_WEIGHTS:
        errors.append("decision-impact policy weights differ from scorer contract")
    if diwa_contract.get("fixed_parameters") != DIWA_FIXED_PARAMETERS:
        errors.append("decision-impact policy fixed parameters differ from scorer contract")
    is_preregistered_default = declared_defaults == manifest_defaults
    if not is_preregistered_default and not allow_diagnostic_subset:
        errors.append(
            "result differs from the full preregistered default matrix; "
            "use --allow-diagnostic-subset for explicitly diagnostic-only runs"
        )
    if allow_diagnostic_subset and is_preregistered_default:
        # Permissible, but use the stronger scope label below for this result.
        pass

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
            if not isinstance(fidelity, str) or fidelity not in count_map:
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
        seen_promotion_ids = set()
        for pi, promotion in enumerate(run["promotion_records"]):
            plabel = f"{label}.promotion_records[{pi}]"
            if not _require_keys(promotion, PROMOTION_KEYS, plabel, errors):
                continue
            cid = promotion["candidate_id"]
            if isinstance(cid, bool) or not isinstance(cid, int) or cid not in observed_high:
                errors.append(f"{plabel} lacks corresponding high-fidelity observation")
                continue
            if cid in seen_promotion_ids:
                errors.append(f"{plabel} duplicates a candidate promotion")
            seen_promotion_ids.add(cid)
            _near(promotion["observed_high_margin"], observed_high[cid][1], f"{plabel}.observed_high_margin", errors)
            method = promotion["method"]
            if not _finite(promotion["predicted_margin"]) or not isinstance(method, str) or method not in ALLOWED_PROMOTION_METHODS.get(policy, set()):
                errors.append(f"{plabel} has invalid or unregistered promotion fields for policy {policy!r}")
                continue
            predicted_feasible += int(promotion["predicted_margin"] >= 0.0)
            false_promotions += int(promotion["predicted_margin"] >= 0.0 and observed_high[cid][1] < 0.0)
        if policy == "high_only" and run["promotion_records"]:
            errors.append(f"{label} high_only must not emit promotion records")
        _near(run["predicted_feasible_promotion_count"], predicted_feasible, f"{label}.predicted_feasible_promotion_count", errors)
        _near(run["false_promotion_count"], false_promotions, f"{label}.false_promotion_count", errors)
        rate = false_promotions / predicted_feasible if predicted_feasible else None
        if rate is None:
            if run["false_promotion_rate"] is not None:
                errors.append(f"{label}.false_promotion_rate should be null")
        else:
            _near(run["false_promotion_rate"], rate, f"{label}.false_promotion_rate", errors)

        _replay_diwa_run(run, errors)

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
        "execution_scope": "preregistered_default_matrix" if is_preregistered_default else "diagnostic_subset",
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
    parser.add_argument(
        "--allow-diagnostic-subset",
        action="store_true",
        help="allow a non-default matrix, marked diagnostic_subset in the receipt",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        document = loads_no_duplicate_keys(args.result_json.read_text(encoding="utf-8"))
        receipt = score_document(document, root, allow_diagnostic_subset=args.allow_diagnostic_subset)
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
