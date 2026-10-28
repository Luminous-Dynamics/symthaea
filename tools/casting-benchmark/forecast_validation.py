#!/usr/bin/env python3
"""Prequential validation of predictive distributions used by decision VOI.

Forecasts and outcomes are separate artifacts. Every outcome must bind to the
exact canonical forecast bytes and a valid outcome digest. Scores are
descriptive; they do not establish independence, calibration under physical
shift, or safety.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence


SCHEMA_VERSION = "1.0.0"
TOL = 1e-10
FORECAST_FIELDS = frozenset({
    "forecast_id", "task_id", "regime", "model_sha256", "state_sha256", "probabilities"
})
OUTCOME_FIELDS = frozenset({
    "forecast_id", "forecast_sha256", "outcome_index", "outcome_sha256"
})


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string without edge whitespace")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def forecast_sha256(forecast: dict[str, Any]) -> str:
    """Digest exact forecast fields; reject extras to prevent ambiguous binding."""
    if not isinstance(forecast, dict) or set(forecast) != FORECAST_FIELDS:
        raise ValueError("forecast keys do not match the exact schema")
    return hashlib.sha256(canonical_json(forecast)).hexdigest()


def make_outcome_record(forecast: dict[str, Any], outcome_index: int) -> dict[str, Any]:
    """Bind an observed class to the exact forecast digest."""
    digest = forecast_sha256(forecast)
    core = {
        "forecast_id": forecast["forecast_id"],
        "forecast_sha256": digest,
        "outcome_index": outcome_index,
    }
    return {
        **core,
        "outcome_sha256": hashlib.sha256(canonical_json(core)).hexdigest(),
    }


def _probabilities(value: object, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{label} must be a non-empty probability sequence")
    probabilities = tuple(
        _finite_number(probability, f"{label}[{index}]")
        for index, probability in enumerate(value)
    )
    if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError(f"{label} probabilities must be in [0, 1]")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=TOL):
        raise ValueError(f"{label} probabilities must sum to one")
    return probabilities


def _validate_forecast(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != FORECAST_FIELDS:
        raise ValueError("forecast shape or keys are invalid")
    _nonblank(value["forecast_id"], "forecast_id")
    _nonblank(value["task_id"], "task_id")
    _nonblank(value["regime"], "regime")
    _sha256(value["model_sha256"], "model_sha256")
    _sha256(value["state_sha256"], "state_sha256")
    _probabilities(value["probabilities"], "probabilities")
    return value


def _validate_outcome(value: Any, class_count: int) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != OUTCOME_FIELDS:
        raise ValueError("outcome shape or keys are invalid")
    _nonblank(value["forecast_id"], "outcome.forecast_id")
    _sha256(value["forecast_sha256"], "forecast_sha256")
    index = value["outcome_index"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < class_count:
        raise ValueError("outcome_index is outside the forecast class range")
    expected = hashlib.sha256(canonical_json({
        "forecast_id": value["forecast_id"],
        "forecast_sha256": value["forecast_sha256"],
        "outcome_index": index,
    })).hexdigest()
    if _sha256(value["outcome_sha256"], "outcome_sha256") != expected:
        raise ValueError("outcome_sha256 does not bind the outcome record fields")
    return value


def _score_group(rows: list[dict[str, Any]], bin_count: int) -> dict[str, Any]:
    class_count = len(rows[0]["probabilities"])
    brier_values: list[float] = []
    log_values: list[float] = []
    zero_probability_count = 0
    correct_count = 0
    bin_sums = [
        {"count": 0, "confidence_sum": 0.0, "correct_sum": 0}
        for _ in range(bin_count)
    ]

    for row in rows:
        probabilities = row["probabilities"]
        observed = row["outcome_index"]
        brier_values.append(sum(
            (probabilities[class_index] - (1.0 if class_index == observed else 0.0)) ** 2
            for class_index in range(class_count)
        ))
        observed_probability = probabilities[observed]
        if observed_probability == 0.0:
            zero_probability_count += 1
        else:
            log_values.append(-math.log(observed_probability))

        predicted = max(range(class_count), key=lambda index: (probabilities[index], -index))
        correct = int(predicted == observed)
        correct_count += correct
        confidence = probabilities[predicted]
        bin_index = min(int(confidence * bin_count), bin_count - 1)
        bin_sums[bin_index]["count"] += 1
        bin_sums[bin_index]["confidence_sum"] += confidence
        bin_sums[bin_index]["correct_sum"] += correct

    reliability = []
    ece = 0.0
    count = len(rows)
    for index, bucket in enumerate(bin_sums):
        bucket_count = bucket["count"]
        if bucket_count == 0:
            reliability.append({
                "bin_index": index, "lower_inclusive": index / bin_count,
                "upper_exclusive": (index + 1) / bin_count,
                "count": 0, "mean_confidence": None, "accuracy": None, "absolute_gap": None,
            })
            continue
        mean_confidence = bucket["confidence_sum"] / bucket_count
        accuracy = bucket["correct_sum"] / bucket_count
        gap = abs(mean_confidence - accuracy)
        ece += (bucket_count / count) * gap
        reliability.append({
            "bin_index": index,
            "lower_inclusive": index / bin_count,
            "upper_exclusive": (index + 1) / bin_count,
            "count": bucket_count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "absolute_gap": gap,
        })

    return {
        "sample_count": count,
        "multiclass_brier_score_mean": sum(brier_values) / count,
        "log_loss_mean": (sum(log_values) / len(log_values)) if zero_probability_count == 0 else None,
        "log_loss_zero_probability_count": zero_probability_count,
        "top_label_accuracy": correct_count / count,
        "top_label_expected_calibration_error": ece,
        "top_label_reliability_bins": reliability,
    }


def score_forecasts(
    forecasts: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    """Score pre-outcome forecasts, requiring a complete one-to-one digest-bound outcome set."""
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or not 2 <= bin_count <= 100:
        raise ValueError("bin_count must be an integer in [2, 100]")
    if isinstance(forecasts, (str, bytes)) or not isinstance(forecasts, Sequence) or not forecasts:
        raise ValueError("forecasts must be a non-empty sequence")
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence) or not outcomes:
        raise ValueError("outcomes must be a non-empty sequence")

    validated_forecasts = [_validate_forecast(forecast) for forecast in forecasts]
    forecast_by_id: dict[str, dict[str, Any]] = {}
    digest_by_id: dict[str, str] = {}
    class_count: int | None = None
    for forecast in validated_forecasts:
        forecast_id = forecast["forecast_id"]
        if forecast_id in forecast_by_id:
            raise ValueError(f"duplicate forecast_id: {forecast_id}")
        probs = _probabilities(forecast["probabilities"], f"forecast[{forecast_id}].probabilities")
        if class_count is None:
            class_count = len(probs)
        elif len(probs) != class_count:
            raise ValueError("all forecasts must use the same outcome class count")
        forecast_by_id[forecast_id] = forecast
        digest_by_id[forecast_id] = forecast_sha256(forecast)

    assert class_count is not None
    outcomes_by_id: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        validated = _validate_outcome(outcome, class_count)
        forecast_id = validated["forecast_id"]
        if forecast_id in outcomes_by_id:
            raise ValueError(f"duplicate outcome for forecast_id: {forecast_id}")
        if forecast_id not in forecast_by_id:
            raise ValueError(f"outcome references unknown forecast_id: {forecast_id}")
        if validated["forecast_sha256"] != digest_by_id[forecast_id]:
            raise ValueError(f"forecast digest mismatch for {forecast_id}")
        outcomes_by_id[forecast_id] = validated

    if set(outcomes_by_id) != set(forecast_by_id):
        missing = sorted(set(forecast_by_id) - set(outcomes_by_id))
        extra = sorted(set(outcomes_by_id) - set(forecast_by_id))
        raise ValueError(f"forecast/outcome coverage mismatch; missing={missing}, extra={extra}")

    joined = []
    for forecast_id in sorted(forecast_by_id):
        row = dict(forecast_by_id[forecast_id])
        row["outcome_index"] = outcomes_by_id[forecast_id]["outcome_index"]
        joined.append(row)

    overall = _score_group(joined, bin_count)
    regimes: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        regimes.setdefault(row["regime"], []).append(row)
    by_regime = {
        regime: _score_group(regimes[regime], bin_count)
        for regime in sorted(regimes)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "forecast_count": len(joined),
        "class_count": class_count,
        "bin_count": bin_count,
        "overall": overall,
        "by_regime": by_regime,
        "forecast_outcome_digest_pairs": [
            {
                "forecast_id": forecast_id,
                "forecast_sha256": digest_by_id[forecast_id],
                "outcome_sha256": outcomes_by_id[forecast_id]["outcome_sha256"],
            }
            for forecast_id in sorted(forecast_by_id)
        ],
        "claim_ceiling": (
            "Descriptive predictive scoring on the supplied forecast/outcome pairs; "
            "does not prove independence, calibration under unobserved shift, or physical safety."
        ),
    }
