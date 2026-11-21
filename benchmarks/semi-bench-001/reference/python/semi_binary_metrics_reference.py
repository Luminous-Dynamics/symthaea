#!/usr/bin/env python3
"""Independent standard-library reference scorer for SEMI-BENCH-001.

This is a test/evaluation oracle, not production Symthaea code and not a
semiconductor process model. It scores frozen binary probability predictions
(e.g. SECOM pass/fail risk) and rejects malformed inputs rather than silently
repairing them.

Input contract: luminous.semi-bench.binary-scores.v1
Output contract: luminous.semi-bench.binary-metrics.v1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "luminous.semi-bench.binary-scores.v1"
OUTPUT_SCHEMA = "luminous.semi-bench.binary-metrics.v1"


class ContractError(ValueError):
    """The supplied benchmark subject violates the frozen scoring contract."""


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{field}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{field}: expected finite number")
    return result


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 12)


def validate_subject(subject: dict[str, Any]) -> tuple[list[dict[str, Any]], float, int, float]:
    if not isinstance(subject, dict):
        raise ContractError("subject: expected object")
    if subject.get("schema") != INPUT_SCHEMA:
        raise ContractError("schema: unsupported or missing input schema")

    threshold = _finite_number(subject.get("decision_threshold"), "decision_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ContractError("decision_threshold: must be in [0, 1]")

    bins_raw = subject.get("calibration_bins")
    if type(bins_raw) is not int or not 2 <= bins_raw <= 100:
        raise ContractError("calibration_bins: expected integer in [2, 100]")

    recall_target = _finite_number(subject.get("recall_target"), "recall_target")
    if not 0.0 < recall_target <= 1.0:
        raise ContractError("recall_target: must be in (0, 1]")

    samples = subject.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ContractError("samples: expected non-empty array")

    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ContractError(f"samples[{index}]: expected object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ContractError(f"samples[{index}].sample_id: expected non-empty string")
        if sample_id in seen:
            raise ContractError(f"samples[{index}].sample_id: duplicate {sample_id!r}")
        seen.add(sample_id)
        label = sample.get("positive")
        if type(label) is not bool:
            raise ContractError(f"samples[{index}].positive: expected boolean")
        score = _finite_number(sample.get("score"), f"samples[{index}].score")
        if not 0.0 <= score <= 1.0:
            raise ContractError(f"samples[{index}].score: must be in [0, 1]")
        checked.append({"sample_id": sample_id, "positive": label, "score": score})

    positives = sum(1 for row in checked if row["positive"])
    negatives = len(checked) - positives
    if positives == 0 or negatives == 0:
        raise ContractError("samples: both positive and negative classes are required")
    return checked, threshold, bins_raw, recall_target


def _grouped_desc(rows: list[dict[str, Any]], key: str) -> list[tuple[float, list[dict[str, Any]]]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(float(row[key]), []).append(row)
    return [(value, groups[value]) for value in sorted(groups, reverse=True)]


def score_subject(subject: dict[str, Any]) -> dict[str, Any]:
    rows, threshold, bins, recall_target = validate_subject(subject)
    n = len(rows)
    total_pos = sum(1 for row in rows if row["positive"])
    total_neg = n - total_pos

    tp = fp = tn = fn = 0
    for row in rows:
        predicted_positive = row["score"] >= threshold
        if row["positive"] and predicted_positive:
            tp += 1
        elif not row["positive"] and predicted_positive:
            fp += 1
        elif not row["positive"] and not predicted_positive:
            tn += 1
        else:
            fn += 1

    sensitivity = tp / total_pos
    specificity = tn / total_neg
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / n
    balanced_accuracy = (sensitivity + specificity) / 2.0

    # Non-interpolated average precision. Tied scores are admitted as one
    # threshold group so input order cannot change the result.
    cumulative_tp = cumulative_fp = 0
    average_precision = 0.0
    previous_recall = 0.0
    for _, group in _grouped_desc(rows, "score"):
        group_tp = sum(1 for row in group if row["positive"])
        group_fp = len(group) - group_tp
        cumulative_tp += group_tp
        cumulative_fp += group_fp
        recall = cumulative_tp / total_pos
        group_precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        average_precision += (recall - previous_recall) * group_precision
        previous_recall = recall

    # Pairwise AUROC with half-credit for ties; exact and dependency-free.
    positive_scores = [row["score"] for row in rows if row["positive"]]
    negative_scores = [row["score"] for row in rows if not row["positive"]]
    wins = 0.0
    for pos_score in positive_scores:
        for neg_score in negative_scores:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    auroc = wins / (total_pos * total_neg)

    brier = sum((row["score"] - (1.0 if row["positive"] else 0.0)) ** 2 for row in rows) / n

    # Fixed-width calibration bins; intervals are [lo, hi), except the final
    # bin, which includes score=1.0. Empty bins contribute zero to ECE.
    calibration = []
    ece = 0.0
    for bin_index in range(bins):
        low = bin_index / bins
        high = (bin_index + 1) / bins
        members = [row for row in rows if (low <= row["score"] < high) or (bin_index == bins - 1 and row["score"] == 1.0)]
        if members:
            mean_score = sum(row["score"] for row in members) / len(members)
            observed_rate = sum(1 for row in members if row["positive"]) / len(members)
            contribution = abs(mean_score - observed_rate) * len(members) / n
            ece += contribution
        else:
            mean_score = observed_rate = contribution = None
        calibration.append({
            "bin": bin_index,
            "lower_inclusive": _round(low),
            "upper_exclusive": _round(high) if bin_index < bins - 1 else None,
            "count": len(members),
            "mean_score": _round(mean_score),
            "observed_positive_rate": _round(observed_rate),
            "ece_contribution": _round(contribution),
        })

    # Risk-coverage groups by max(p, 1-p). Do not split equal-confidence ties.
    confidence_rows = [
        {**row, "confidence": max(row["score"], 1.0 - row["score"]),
         "error": (row["score"] >= 0.5) != row["positive"]}
        for row in rows
    ]
    risk_coverage = []
    accepted = errors = 0
    for confidence, group in _grouped_desc(confidence_rows, "confidence"):
        accepted += len(group)
        errors += sum(1 for row in group if row["error"])
        risk_coverage.append({
            "minimum_confidence_inclusive": _round(confidence),
            "accepted": accepted,
            "coverage": _round(accepted / n),
            "errors": errors,
            "risk": _round(errors / accepted),
        })

    # Lowest false-positive rate threshold which reaches the requested recall.
    # Descending score groups visit thresholds from strict to permissive.
    cumulative_tp = cumulative_fp = 0
    fpr_at_recall = None
    for score, group in _grouped_desc(rows, "score"):
        cumulative_tp += sum(1 for row in group if row["positive"])
        cumulative_fp += sum(1 for row in group if not row["positive"])
        if cumulative_tp / total_pos >= recall_target:
            fpr_at_recall = {
                "target_recall": _round(recall_target),
                "threshold": _round(score),
                "achieved_recall": _round(cumulative_tp / total_pos),
                "false_positive_rate": _round(cumulative_fp / total_neg),
            }
            break

    return {
        "schema": OUTPUT_SCHEMA,
        "case_id": subject.get("case_id"),
        "sample_count": n,
        "class_counts": {"positive": total_pos, "negative": total_neg},
        "threshold_metrics": {
            "threshold": _round(threshold),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": _round(sensitivity),
            "specificity": _round(specificity),
            "precision": _round(precision),
            "accuracy": _round(accuracy),
            "balanced_accuracy": _round(balanced_accuracy),
        },
        "ranking_metrics": {
            "average_precision": _round(average_precision),
            "auroc": _round(auroc),
            "fpr_at_recall": fpr_at_recall,
        },
        "probability_metrics": {
            "brier_score": _round(brier),
            "expected_calibration_error": _round(ece),
            "calibration_bins": calibration,
        },
        "risk_coverage": risk_coverage,
    }


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError("input JSON root must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="score subject JSON")
    parser.add_argument("--output", type=Path, help="write metrics JSON to this path")
    args = parser.parse_args(argv)

    default_input = Path(__file__).resolve().parents[1] / "fixtures" / "binary-metrics-known-answer-v1.json"
    source = args.input or default_input
    try:
        result = score_subject(load_json(source))
    except ContractError as exc:
        print(f"SEMI-BENCH-001 reference scorer: REJECT: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
