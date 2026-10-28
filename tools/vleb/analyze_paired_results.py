#!/usr/bin/env python3
"""Analyze paired VLEB results without shrinking the locked denominator.

This module computes descriptive metrics, Wilson intervals and an exact two-sided
McNemar test. It does not authenticate evidence, admit tasks, or establish that
an upstream "assured_pass" label is true. Production use requires an evaluator-
pinned input digest and an independently trusted upstream evidence gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist, median
from typing import Any

OUTCOMES = ("assured_pass", "raw_pass_unverified", "fail", "unresolved", "abstained")
HEX = set("0123456789abcdef")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in HEX for char in value)


def _wilson(successes: int, trials: int, confidence: float) -> list[float] | None:
    if trials == 0:
        return None
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) / trials) + (z2 / (4.0 * trials * trials))) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _exact_mcnemar_p(a_only: int, b_only: int) -> float:
    """Exact two-sided binomial test conditional on discordant pairs."""
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = min(a_only, b_only)
    numerator = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, (2.0 * numerator) / (2 ** discordant))


def _nearest_rank(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _check_keys(value: Any, allowed: set[str], required: set[str], label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label}:expected_object"]
    errors = [f"{label}:unknown_field:{key}" for key in sorted(set(value) - allowed)]
    errors.extend(f"{label}:missing_field:{key}" for key in sorted(required - set(value)))
    return errors


def _validate_outcome(record: Any, label: str) -> list[str]:
    allowed = {"outcome", "cost_usd", "duration_ms"}
    errors = _check_keys(record, allowed, allowed, label)
    if not isinstance(record, dict):
        return errors
    if record.get("outcome") not in OUTCOMES:
        errors.append(f"{label}:invalid_outcome")
    cost = record.get("cost_usd")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        errors.append(f"{label}:invalid_cost")
    duration = record.get("duration_ms")
    if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
        errors.append(f"{label}:invalid_duration")
    return errors


def _validate_input(data: Any) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    root_keys = {"schema_version", "cohort_id", "protocol_sha256", "analysis_plan", "candidates", "tasks"}
    errors.extend(_check_keys(data, root_keys, root_keys, "root"))
    if not isinstance(data, dict):
        return errors, []
    if data.get("schema_version") != "vleb-paired-results-v1":
        errors.append("schema_version:unsupported")
    if not isinstance(data.get("cohort_id"), str) or len(data["cohort_id"]) < 3:
        errors.append("cohort_id:missing_or_invalid")
    if not _is_sha256(data.get("protocol_sha256")):
        errors.append("protocol_sha256:invalid")
    plan = data.get("analysis_plan")
    plan_keys = {"primary_metric", "pairwise_test", "confidence_level", "attempt_aggregation", "cost_currency", "cluster_bootstrap_replicates", "cluster_bootstrap_seed"}
    errors.extend(_check_keys(plan, plan_keys, plan_keys, "analysis_plan"))
    if isinstance(plan, dict):
        if plan.get("primary_metric") != "assurance_adjusted_semantic_success_rate":
            errors.append("analysis_plan:primary_metric_must_be_aassr")
        if plan.get("pairwise_test") != "exact_mcnemar_two_sided":
            errors.append("analysis_plan:pairwise_test_must_be_exact_mcnemar")
        confidence = plan.get("confidence_level")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0.8 <= confidence <= 0.9999:
            errors.append("analysis_plan:invalid_confidence_level")
        if not isinstance(plan.get("attempt_aggregation"), str) or not plan["attempt_aggregation"].strip():
            errors.append("analysis_plan:attempt_aggregation_must_be_preregistered")
        if plan.get("cost_currency") != "USD":
            errors.append("analysis_plan:cost_currency_must_be_usd")
        replicates = plan.get("cluster_bootstrap_replicates")
        if isinstance(replicates, bool) or not isinstance(replicates, int) or not 1000 <= replicates <= 100000:
            errors.append("analysis_plan:invalid_cluster_bootstrap_replicates")
        seed = plan.get("cluster_bootstrap_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            errors.append("analysis_plan:invalid_cluster_bootstrap_seed")
    candidates = data.get("candidates")
    errors.extend(_check_keys(candidates, {"A", "B"}, {"A", "B"}, "candidates"))
    if isinstance(candidates, dict):
        candidate_keys = {"name", "configuration_sha256"}
        for label in ("A", "B"):
            candidate = candidates.get(label)
            errors.extend(_check_keys(candidate, candidate_keys, candidate_keys, f"candidate_{label}"))
            if isinstance(candidate, dict):
                if not isinstance(candidate.get("name"), str) or not candidate["name"].strip() or not _is_sha256(candidate.get("configuration_sha256")):
                    errors.append(f"candidate_{label}:invalid_identity_or_configuration_pin")
    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks:must_be_nonempty_array")
        return errors, []
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    task_keys = {"task_id", "cluster_id", "lane", "candidate_A", "candidate_B", "exclusion"}
    exclusion_keys = {"reason", "adjudicated_before_unblinding", "adjudication_sha256"}
    for index, task in enumerate(tasks):
        label = f"tasks[{index}]"
        errors.extend(_check_keys(task, task_keys, task_keys, label))
        if not isinstance(task, dict):
            continue
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            errors.append(f"{label}:invalid_task_id")
        elif task_id in seen:
            errors.append(f"{label}:duplicate_task_id")
        else:
            seen.add(task_id)
        if not isinstance(task.get("cluster_id"), str) or not task["cluster_id"].strip():
            errors.append(f"{label}:invalid_cluster_id")
        if not isinstance(task.get("lane"), str) or not task["lane"].strip():
            errors.append(f"{label}:invalid_lane")
        for candidate_label, key in (("A", "candidate_A"), ("B", "candidate_B")):
            errors.extend(_validate_outcome(task.get(key), f"{label}.{candidate_label}"))
        exclusion = task.get("exclusion")
        if exclusion is not None:
            errors.extend(_check_keys(exclusion, exclusion_keys, exclusion_keys, f"{label}.exclusion"))
            if isinstance(exclusion, dict):
                if not isinstance(exclusion.get("reason"), str) or not exclusion["reason"].strip():
                    errors.append(f"{label}:exclusion_requires_reason")
                if exclusion.get("adjudicated_before_unblinding") is not True:
                    errors.append(f"{label}:exclusion_not_adjudicated_blind")
                if not _is_sha256(exclusion.get("adjudication_sha256")):
                    errors.append(f"{label}:exclusion_requires_pinned_adjudication")
        normalized.append(task)
    return errors, normalized

def _percentile(sorted_values: list[float], probability: float) -> float:
    """Linearly interpolated percentile, defined for a sorted non-empty sample."""
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _cluster_bootstrap_interval(
    records: list[dict[str, Any]],
    candidate_key: str,
    success_fn,
    confidence: float,
    replicates: int,
    seed: int,
) -> list[float] | None:
    clusters = sorted({record["cluster_id"] for record in records})
    if len(clusters) < 2 or not records:
        return None
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cluster[record["cluster_id"]].append(record)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_clusters = [rng.choice(clusters) for _ in clusters]
        sample = [record for cluster in sampled_clusters for record in by_cluster[cluster]]
        estimates.append(sum(success_fn(record[candidate_key]["outcome"]) for record in sample) / len(sample))
    estimates.sort()
    alpha = 1.0 - confidence
    return [
        _percentile(estimates, alpha / 2.0),
        _percentile(estimates, 1.0 - alpha / 2.0),
    ]


def _cluster_bootstrap_difference(
    records: list[dict[str, Any]],
    confidence: float,
    replicates: int,
    seed: int,
) -> list[float] | None:
    clusters = sorted({record["cluster_id"] for record in records})
    if len(clusters) < 2 or not records:
        return None
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cluster[record["cluster_id"]].append(record)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_clusters = [rng.choice(clusters) for _ in clusters]
        sample = [record for cluster in sampled_clusters for record in by_cluster[cluster]]
        a_rate = sum(record["candidate_A"]["outcome"] == "assured_pass" for record in sample) / len(sample)
        b_rate = sum(record["candidate_B"]["outcome"] == "assured_pass" for record in sample) / len(sample)
        estimates.append(a_rate - b_rate)
    estimates.sort()
    alpha = 1.0 - confidence
    return [
        _percentile(estimates, alpha / 2.0),
        _percentile(estimates, 1.0 - alpha / 2.0),
    ]


def _metrics(records: list[dict[str, Any]], candidate_key: str, confidence: float, cost_records: list[dict[str, Any]] | None = None, bootstrap_replicates: int = 1000, bootstrap_seed: int = 0) -> dict[str, Any]:
    outcomes = Counter(record[candidate_key]["outcome"] for record in records)
    n = len(records)
    assured = outcomes["assured_pass"]
    raw = assured + outcomes["raw_pass_unverified"]
    economic_records = cost_records if cost_records is not None else records
    # Correctness denominator can exclude only pre-unblinding-adjudicated tasks,
    # but all cost/time already incurred on assigned tasks remains in economics.
    cost = sum(float(record[candidate_key]["cost_usd"]) for record in economic_records)
    durations = [record[candidate_key]["duration_ms"] for record in economic_records]
    assured_rate = lambda value: value == "assured_pass"
    raw_rate = lambda value: value in ("assured_pass", "raw_pass_unverified")
    cluster_count = len({record["cluster_id"] for record in records})
    return {
        "n": n,
        "outcome_counts": {outcome: outcomes[outcome] for outcome in OUTCOMES},
        "aassr": assured / n if n else None,
        "aassr_wilson_interval": _wilson(assured, n, confidence),
        "aassr_cluster_bootstrap_interval": _cluster_bootstrap_interval(records, candidate_key, assured_rate, confidence, bootstrap_replicates, bootstrap_seed),
        "raw_resolution_rate": raw / n if n else None,
        "raw_resolution_wilson_interval": _wilson(raw, n, confidence),
        "raw_resolution_cluster_bootstrap_interval": _cluster_bootstrap_interval(records, candidate_key, raw_rate, confidence, bootstrap_replicates, bootstrap_seed),
        "cluster_count": cluster_count,
        "uncertainty_note": "Wilson intervals assume task independence; cluster-bootstrap intervals resample the supplied repository/codebase clusters and are null when fewer than two clusters are present.",
        "unresolved_rate": outcomes["unresolved"] / n if n else None,
        "abstention_rate": outcomes["abstained"] / n if n else None,
        "total_cost_usd": round(cost, 8),
        "cost_per_independently_verified_pass_usd": round(cost / assured, 8) if assured else None,
        "duration_ms": {
            "median": median(durations) if durations else None,
            "p90": _nearest_rank(durations, 0.9),
            "total": sum(durations),
        },
    }


def analyze(data: Any) -> dict[str, Any]:
    errors, tasks = _validate_input(data)
    if errors:
        raise ValueError("invalid paired results: " + "; ".join(errors))
    confidence = float(data["analysis_plan"]["confidence_level"])
    excluded = [task for task in tasks if task.get("exclusion") is not None]
    scored = [task for task in tasks if task.get("exclusion") is None]
    if not scored:
        raise ValueError("invalid paired results: no tasks remain after pre-unblinding exclusions")

    bootstrap_replicates = data["analysis_plan"]["cluster_bootstrap_replicates"]
    bootstrap_seed = data["analysis_plan"]["cluster_bootstrap_seed"]
    scored_a = _metrics(scored, "candidate_A", confidence, tasks, bootstrap_replicates, bootstrap_seed)
    scored_b = _metrics(scored, "candidate_B", confidence, tasks, bootstrap_replicates, bootstrap_seed)
    a_only = both = b_only = neither = 0
    for task in scored:
        a_pass = task["candidate_A"]["outcome"] == "assured_pass"
        b_pass = task["candidate_B"]["outcome"] == "assured_pass"
        if a_pass and b_pass:
            both += 1
        elif a_pass:
            a_only += 1
        elif b_pass:
            b_only += 1
        else:
            neither += 1

    lane_results: dict[str, Any] = {}
    for lane in sorted({task["lane"] for task in tasks}):
        lane_assigned = [task for task in tasks if task["lane"] == lane]
        lane_scored = [task for task in scored if task["lane"] == lane]
        lane_results[lane] = {
            "candidate_A": _metrics(lane_scored, "candidate_A", confidence, lane_assigned, bootstrap_replicates, bootstrap_seed),
            "candidate_B": _metrics(lane_scored, "candidate_B", confidence, lane_assigned, bootstrap_replicates, bootstrap_seed),
            "assigned_tasks": len(lane_assigned),
            "scored_tasks": len(lane_scored),
            "pre_unblinding_exclusions": len(lane_assigned) - len(lane_scored),
        }

    return {
        "schema_version": "vleb-paired-analysis-v1",
        "analysis_only": True,
        "score_eligible": False,
        "score_eligibility_note": "Statistics cannot grant score eligibility; the protected evaluator must admit the task and authenticate provenance, receipt, and semantic evidence first.",
        "cohort_id": data["cohort_id"],
        "protocol_sha256": data["protocol_sha256"],
        "analysis_plan": data["analysis_plan"],
        "candidates": data["candidates"],
        "assignment_accounting": {
            "assigned_tasks": len(tasks),
            "scored_tasks": len(scored),
            "pre_unblinding_exclusions": len(excluded),
            "excluded_task_ids": [task["task_id"] for task in excluded],
            "cluster_ids": sorted({task["cluster_id"] for task in tasks}),
            "task_ids": [task["task_id"] for task in tasks],
        },
        "candidate_A": scored_a,
        "candidate_B": scored_b,
        "paired_aassr_comparison": {
            "both_assured_pass": both,
            "candidate_A_only_assured_pass": a_only,
            "candidate_B_only_assured_pass": b_only,
            "neither_assured_pass": neither,
            "aassr_risk_difference_A_minus_B": (a_only - b_only) / len(scored),
            "exact_mcnemar_p_value_two_sided": _exact_mcnemar_p(a_only, b_only),
            "cluster_bootstrap_risk_difference_interval_A_minus_B": _cluster_bootstrap_difference(scored, confidence, bootstrap_replicates, bootstrap_seed),
            "cluster_count": len({task["cluster_id"] for task in scored}),
            "interpretation": "The exact McNemar p-value is a task-pair test and is not cluster-adjusted. The paired cluster-bootstrap interval resamples repository/codebase clusters. Neither corrects for invalid task admission, multiple unregistered comparisons, or mismatched protocols.",
        },
        "by_lane": lane_results,
        "limitations": [
            "Assured-pass labels must come from a separately authenticated evaluator pipeline; this script does not validate evidence or attestations.",
            "Unresolved, abstained, failed and raw-but-unassured results remain non-passes in the AASSR denominator.",
            "Exclusions are accepted only when explicitly marked adjudicated before unblinding and carrying a pinned adjudication digest; the script cannot independently prove that the adjudication was blind.",
            "Model/scaffold comparisons are meaningful only when task cohort, budgets, tools, retries, runtime and protocol are matched.",
            "Confidence intervals and p-values describe the supplied locked cohort; they do not cure selection bias or benchmark contamination.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not _is_sha256(args.expected_input_sha256):
        parser.error("--expected-input-sha256 must be a lowercase SHA-256 digest")
    try:
        raw = args.input.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != args.expected_input_sha256:
            raise ValueError(f"input digest mismatch: expected {args.expected_input_sha256}, observed {observed}")
        data = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON number {value!r} is not allowed")))
        result = analyze(data)
        result["input_sha256"] = observed
        result["analysis_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"eligible": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
