"""Paired, group-bootstrap analysis for memory retrieval run artifacts.

Both JSONL inputs must use the schema described in README.md. The analyzer
fails closed on missing/extra questions, duplicate keys, failed rows, mismatched
group IDs, and invalid cost/latency fields. It never silently drops a run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

SCHEMA = "broca-paired-run-analysis-v1"
MAX_LINE_BYTES = 32 * 1024 * 1024
DEFAULT_REPLICATES = 20_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                raise ValueError(f"{path}:{line_number}: line exceeds size bound")
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(row)
    if not records:
        raise ValueError(f"{path}: run file must not be empty")
    return records


def _finite_nonnegative(value: Any, field: str, key: tuple[str, str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key}: {field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{key}: {field} must be finite and non-negative")
    return converted


def validate_run(records: list[dict[str, Any]], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    required = {
        "domain", "question_id", "group_id", "status", "correct",
        "query_latency_ms", "context_tokens", "lifecycle_cost_units",
    }
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(records, 1):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"{label} row {row_number}: missing fields {missing}")
        domain, question_id, group_id = row["domain"], row["question_id"], row["group_id"]
        for field, value in (("domain", domain), ("question_id", question_id), ("group_id", group_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} row {row_number}: {field} must be a non-empty string")
        key = (domain, question_id)
        if key in indexed:
            raise ValueError(f"{label}: duplicate domain/question key {key}")
        if row["status"] != "success":
            raise ValueError(f"{label} {key}: status must be success; failed rows cannot be silently dropped")
        if not isinstance(row["correct"], bool):
            raise ValueError(f"{label} {key}: correct must be a JSON boolean")
        latency = _finite_nonnegative(row["query_latency_ms"], "query_latency_ms", key)
        context = row["context_tokens"]
        if isinstance(context, bool) or not isinstance(context, int) or context < 0:
            raise ValueError(f"{label} {key}: context_tokens must be a non-negative integer")
        lifecycle_cost = _finite_nonnegative(row["lifecycle_cost_units"], "lifecycle_cost_units", key)
        indexed[key] = {
            "domain": domain,
            "question_id": question_id,
            "group_id": group_id,
            "correct": row["correct"],
            "query_latency_ms": latency,
            "context_tokens": context,
            "lifecycle_cost_units": lifecycle_cost,
        }
    return indexed


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty values")
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(values)
    # Nearest-rank percentile, with rank clamped to [1, n].
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _interval(samples: list[float]) -> dict[str, float]:
    return {
        "lower_95": round(_percentile(samples, 0.025), 6),
        "upper_95": round(_percentile(samples, 0.975), 6),
    }


def analyze_paired_runs(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    *,
    bootstrap_replicates: int = DEFAULT_REPLICATES,
    seed: int = 20261028,
    quality_gain_pp: float = 3.0,
    quality_noninferiority_pp: float = 1.0,
    cost_reduction_percent: float = 25.0,
) -> dict[str, Any]:
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 100:
        raise ValueError("bootstrap_replicates must be an integer >= 100")
    for name, value in (
        ("quality_gain_pp", quality_gain_pp),
        ("quality_noninferiority_pp", quality_noninferiority_pp),
        ("cost_reduction_percent", cost_reduction_percent),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    baseline = validate_run(baseline_records, "baseline")
    candidate = validate_run(candidate_records, "candidate")
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            f"paired question keys differ; missing candidate={missing_candidate[:10]}, "
            f"missing baseline={missing_baseline[:10]}"
        )

    paired: dict[tuple[str, str], dict[str, Any]] = {}
    for key, left in baseline.items():
        right = candidate[key]
        if left["group_id"] != right["group_id"]:
            raise ValueError(f"{key}: baseline/candidate group_id mismatch")
        paired[key] = {
            "domain": key[0],
            "question_id": key[1],
            "group_id": left["group_id"],
            "baseline_correct": left["correct"],
            "candidate_correct": right["correct"],
            "baseline_latency_ms": left["query_latency_ms"],
            "candidate_latency_ms": right["query_latency_ms"],
            "baseline_context_tokens": left["context_tokens"],
            "candidate_context_tokens": right["context_tokens"],
            "baseline_lifecycle_cost": left["lifecycle_cost_units"],
            "candidate_lifecycle_cost": right["lifecycle_cost_units"],
        }

    rows = list(paired.values())
    n = len(rows)
    baseline_accuracy = sum(row["baseline_correct"] for row in rows) / n
    candidate_accuracy = sum(row["candidate_correct"] for row in rows) / n
    delta_pp = 100.0 * (candidate_accuracy - baseline_accuracy)
    wins = sum(row["candidate_correct"] and not row["baseline_correct"] for row in rows)
    losses = sum(row["baseline_correct"] and not row["candidate_correct"] for row in rows)
    ties = n - wins - losses

    baseline_latency = [row["baseline_latency_ms"] for row in rows]
    candidate_latency = [row["candidate_latency_ms"] for row in rows]
    baseline_context = [float(row["baseline_context_tokens"]) for row in rows]
    candidate_context = [float(row["candidate_context_tokens"]) for row in rows]
    baseline_cost = sum(row["baseline_lifecycle_cost"] for row in rows)
    candidate_cost = sum(row["candidate_lifecycle_cost"] for row in rows)
    if baseline_cost <= 0:
        raise ValueError("baseline total lifecycle cost must be positive to calculate reduction")
    cost_reduction = 100.0 * (1.0 - candidate_cost / baseline_cost)
    if not math.isfinite(cost_reduction):
        raise ValueError("lifecycle cost reduction is not finite")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    group_stats = []
    for group_id in sorted(grouped):
        group_rows = grouped[group_id]
        group_stats.append({
            "n": len(group_rows),
            "base_correct": sum(row["baseline_correct"] for row in group_rows),
            "candidate_correct": sum(row["candidate_correct"] for row in group_rows),
            "base_cost": sum(row["baseline_lifecycle_cost"] for row in group_rows),
            "candidate_cost": sum(row["candidate_lifecycle_cost"] for row in group_rows),
        })

    rng = random.Random(seed)
    accuracy_delta_samples: list[float] = []
    cost_reduction_samples: list[float] = []
    group_count = len(group_stats)
    for _ in range(bootstrap_replicates):
        sample_n = base_correct = cand_correct = 0
        sample_base_cost = sample_cand_cost = 0.0
        for _draw in range(group_count):
            sampled = group_stats[rng.randrange(group_count)]
            sample_n += sampled["n"]
            base_correct += sampled["base_correct"]
            cand_correct += sampled["candidate_correct"]
            sample_base_cost += sampled["base_cost"]
            sample_cand_cost += sampled["candidate_cost"]
        accuracy_delta_samples.append(100.0 * (cand_correct - base_correct) / sample_n)
        if sample_base_cost > 0:
            cost_reduction_samples.append(100.0 * (1.0 - sample_cand_cost / sample_base_cost))

    accuracy_ci = _interval(accuracy_delta_samples)
    cost_ci = _interval(cost_reduction_samples)
    quality_gate = delta_pp >= quality_gain_pp and accuracy_ci["lower_95"] > 0.0
    efficiency_gate = (
        accuracy_ci["lower_95"] > -quality_noninferiority_pp
        and cost_reduction >= cost_reduction_percent
    )

    domains: dict[str, dict[str, Any]] = {}
    for domain in sorted({row["domain"] for row in rows}):
        subset = [row for row in rows if row["domain"] == domain]
        base_acc = sum(row["baseline_correct"] for row in subset) / len(subset)
        cand_acc = sum(row["candidate_correct"] for row in subset) / len(subset)
        domains[domain] = {
            "question_count": len(subset),
            "baseline_accuracy_percent": round(100 * base_acc, 4),
            "candidate_accuracy_percent": round(100 * cand_acc, 4),
            "delta_accuracy_percentage_points": round(100 * (cand_acc - base_acc), 4),
        }

    return {
        "schema": SCHEMA,
        "seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "grouping": "cluster bootstrap resamples group_id values with replacement and retains every paired row in each sampled group",
        "question_count": n,
        "group_count": group_count,
        "domains": domains,
        "accuracy": {
            "baseline_percent": round(100 * baseline_accuracy, 4),
            "candidate_percent": round(100 * candidate_accuracy, 4),
            "delta_percentage_points": round(delta_pp, 4),
            "delta_ci95_percentage_points": accuracy_ci,
            "paired_wins": wins,
            "paired_losses": losses,
            "paired_ties": ties,
        },
        "query_latency_ms": {
            "baseline": {f"p{q}": round(_percentile(baseline_latency, q / 100), 4) for q in (50, 95, 99)},
            "candidate": {f"p{q}": round(_percentile(candidate_latency, q / 100), 4) for q in (50, 95, 99)},
            "total_reduction_percent": round(100 * (1 - sum(candidate_latency) / sum(baseline_latency)), 4) if sum(baseline_latency) > 0 else None,
        },
        "context_tokens_per_query": {
            "baseline_mean": round(sum(baseline_context) / n, 4),
            "candidate_mean": round(sum(candidate_context) / n, 4),
            "baseline_p95": round(_percentile(baseline_context, 0.95), 4),
            "candidate_p95": round(_percentile(candidate_context, 0.95), 4),
        },
        "lifecycle_cost": {
            "unit": "caller-defined; keep unit definition identical across both inputs",
            "baseline_total": round(baseline_cost, 6),
            "candidate_total": round(candidate_cost, 6),
            "reduction_percent": round(cost_reduction, 4),
            "reduction_ci95_percent": cost_ci,
        },
        "decision_gates": {
            "quality": {
                "required_delta_pp": quality_gain_pp,
                "required_ci_lower_gt": 0.0,
                "pass": quality_gate,
            },
            "efficiency": {
                "required_noninferiority_margin_pp": quality_noninferiority_pp,
                "required_cost_reduction_percent": cost_reduction_percent,
                "pass": efficiency_gate,
            },
            "overall_pass": quality_gate or efficiency_gate,
            "warning": "A numeric gate does not establish benchmark SOTA unless the official harness, pinned reader/evaluator, exact split, and all receipts are verified.",
        },
        "claim_ceiling": "paired outcome analysis; not an official score unless inputs come from a valid official run",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="baseline per-question JSONL")
    parser.add_argument("--candidate", required=True, type=Path, help="candidate per-question JSONL")
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--quality-gain-pp", type=float, default=3.0)
    parser.add_argument("--quality-noninferiority-pp", type=float, default=1.0)
    parser.add_argument("--cost-reduction-percent", type=float, default=25.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline)
    candidate = load_jsonl(args.candidate)
    report = analyze_paired_runs(
        baseline,
        candidate,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
        quality_gain_pp=args.quality_gain_pp,
        quality_noninferiority_pp=args.quality_noninferiority_pp,
        cost_reduction_percent=args.cost_reduction_percent,
    )
    report["input_receipts"] = {
        "baseline_path": str(args.baseline.resolve()),
        "baseline_sha256": sha256_file(args.baseline),
        "candidate_path": str(args.candidate.resolve()),
        "candidate_sha256": sha256_file(args.candidate),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
