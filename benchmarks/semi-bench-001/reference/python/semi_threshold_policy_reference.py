#!/usr/bin/env python3
"""Independent validation-threshold policy oracle for SEMI-BENCH-001.

This test-only helper chooses a decision threshold on validation scores under
an explicit precision floor, then applies that frozen threshold to a disjoint
test set. It imports the separate binary metric reference scorer but no
Symthaea production code.
"""
from __future__ import annotations

import math
from typing import Any

import semi_binary_metrics_reference as scorer


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise scorer.ContractError(f"{field}: expected finite number")
    value = float(value)
    if not math.isfinite(value):
        raise scorer.ContractError(f"{field}: expected finite number")
    return value


def _groups_desc(rows: list[dict[str, Any]]) -> list[tuple[float, list[dict[str, Any]]]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(float(row["score"]), []).append(row)
    return [(score, groups[score]) for score in sorted(groups, reverse=True)]


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 12)


def evaluate_with_validation_threshold(
    validation_samples: list[dict[str, Any]],
    test_subject: dict[str, Any],
    minimum_precision: float,
    fallback_threshold: float = 0.5,
) -> dict[str, Any]:
    """Select max validation recall subject to precision floor; score disjoint test.

    Equal validation scores are admitted as a group. If no observed threshold
    reaches the precision floor, the caller-supplied fallback is applied and
    reported explicitly. Validation identities must be disjoint from test.
    """
    minimum_precision = _finite(minimum_precision, "minimum_precision")
    fallback_threshold = _finite(fallback_threshold, "fallback_threshold")
    if not 0.0 < minimum_precision <= 1.0:
        raise scorer.ContractError("minimum_precision: must be in (0, 1]")
    if not 0.0 <= fallback_threshold <= 1.0:
        raise scorer.ContractError("fallback_threshold: must be in [0, 1]")
    if not isinstance(validation_samples, list) or not validation_samples:
        raise scorer.ContractError("validation_samples: expected non-empty array")

    rows: list[dict[str, Any]] = []
    validation_ids: set[str] = set()
    for index, sample in enumerate(validation_samples):
        if not isinstance(sample, dict):
            raise scorer.ContractError(f"validation_samples[{index}]: expected object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise scorer.ContractError(f"validation_samples[{index}].sample_id: expected non-empty string")
        if sample_id in validation_ids:
            raise scorer.ContractError(f"validation_samples[{index}].sample_id: duplicate {sample_id!r}")
        validation_ids.add(sample_id)
        label = sample.get("positive")
        if type(label) is not bool:
            raise scorer.ContractError(f"validation_samples[{index}].positive: expected boolean")
        score = _finite(sample.get("score"), f"validation_samples[{index}].score")
        if not 0.0 <= score <= 1.0:
            raise scorer.ContractError(f"validation_samples[{index}].score: must be in [0, 1]")
        rows.append({"sample_id": sample_id, "positive": label, "score": score})

    positive_count = sum(1 for row in rows if row["positive"])
    negative_count = len(rows) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise scorer.ContractError("validation_samples: both positive and negative classes are required")

    test_rows, _, _, _ = scorer.validate_subject(test_subject)
    test_ids = {row["sample_id"] for row in test_rows}
    overlap = validation_ids & test_ids
    if overlap:
        raise scorer.ContractError(f"validation/test identity overlap: {sorted(overlap)[0]!r}")

    selected: dict[str, float | int] | None = None
    cumulative_tp = cumulative_fp = 0
    best_recall = -1.0
    for threshold, group in _groups_desc(rows):
        cumulative_tp += sum(1 for row in group if row["positive"])
        cumulative_fp += len(group) - sum(1 for row in group if row["positive"])
        precision = cumulative_tp / (cumulative_tp + cumulative_fp)
        recall = cumulative_tp / positive_count
        if precision >= minimum_precision and recall > best_recall:
            best_recall = recall
            selected = {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "tp": cumulative_tp,
                "fp": cumulative_fp,
            }

    fallback_used = selected is None
    threshold = fallback_threshold if fallback_used else float(selected["threshold"])
    scoring_subject = dict(test_subject)
    scoring_subject["decision_threshold"] = threshold
    test_metrics = scorer.score_subject(scoring_subject)

    return {
        "schema": "luminous.semi-bench.validation-threshold-report.v1",
        "case_id": test_subject.get("case_id"),
        "minimum_validation_precision": _round(minimum_precision),
        "fallback_threshold": _round(fallback_threshold),
        "fallback_used": fallback_used,
        "selection": None if selected is None else {
            "threshold": _round(float(selected["threshold"])),
            "validation_precision": _round(float(selected["precision"])),
            "validation_recall": _round(float(selected["recall"])),
            "validation_tp": int(selected["tp"]),
            "validation_fp": int(selected["fp"]),
        },
        "test_metrics": test_metrics,
    }
