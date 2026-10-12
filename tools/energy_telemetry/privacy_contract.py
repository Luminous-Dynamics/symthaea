"""Offline contract checks for privacy-preserving aggregate energy telemetry.

This validator checks envelope shape and declared policy constraints. It does not
prove a gateway's cohort count, cryptographic authenticity, DP accounting, or
meter truth.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import re
from typing import Any

SCHEMA_VERSION = "privacy-preserving-energy-telemetry/v1"
MINIMUM_RESIDENTIAL_COHORT = 20
MIN_WINDOW_SECONDS = 300
MAX_WINDOW_SECONDS = 86400
MAX_RELEASE_EPSILON = 1.0
MAX_RELEASE_DELTA = 1e-6
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
BATCH_ID_RE = re.compile(r"^[a-f0-9]{32}$")
REGION_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
POLICY_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

TOP_KEYS = {
    "schema_version", "batch_id", "created_at_utc", "window",
    "aggregation", "recipient_class", "dp_release",
}
WINDOW_KEYS = {"start_utc", "end_utc", "duration_seconds"}
AGGREGATION_KEYS = {
    "scope", "region_token", "cohort_size", "minimum_cohort_size",
    "energy_wh_total", "method", "coverage_fraction", "measurement_quality",
}
DP_KEYS = {
    "mechanism", "epsilon", "delta", "budget_policy_id", "budget_receipt_sha256",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {token}")


def strict_json_loads(raw: str | bytes | bytearray) -> Any:
    """Parse JSON while rejecting duplicate keys and NaN/Infinity constants."""
    return json.loads(raw, object_pairs_hook=_strict_object, parse_constant=_reject_constant)


def _exact_keys(value: Any, expected: set[str], prefix: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{prefix}: must be an object")
        return False
    actual = set(value)
    missing, extra = expected - actual, actual - expected
    if missing:
        errors.append(f"{prefix}: missing required keys {sorted(missing)}")
    if extra:
        errors.append(f"{prefix}: forbidden or unknown keys {sorted(str(k) for k in extra)}")
    return not missing and not extra


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _parse_utc(value: Any, prefix: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        errors.append(f"{prefix}: must be an ISO-8601 UTC timestamp ending in Z")
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        errors.append(f"{prefix}: invalid ISO-8601 timestamp")
        return None
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        errors.append(f"{prefix}: timestamp must be UTC")
        return None
    return parsed


def validate_envelope(document: Any) -> list[str]:
    errors: list[str] = []
    if not _exact_keys(document, TOP_KEYS, "envelope", errors):
        if not isinstance(document, dict):
            return errors
    if not isinstance(document, dict):
        return errors

    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version: unsupported version")
    batch_id = document.get("batch_id")
    if not isinstance(batch_id, str) or not BATCH_ID_RE.fullmatch(batch_id):
        errors.append("batch_id: must be 32 lowercase hexadecimal characters; do not use stable device IDs")

    created = _parse_utc(document.get("created_at_utc"), "created_at_utc", errors)

    window = document.get("window")
    if _exact_keys(window, WINDOW_KEYS, "window", errors):
        start = _parse_utc(window.get("start_utc"), "window.start_utc", errors)
        end = _parse_utc(window.get("end_utc"), "window.end_utc", errors)
        duration = window.get("duration_seconds")
        if not _is_int(duration) or not MIN_WINDOW_SECONDS <= duration <= MAX_WINDOW_SECONDS:
            errors.append(f"window.duration_seconds: must be integer from {MIN_WINDOW_SECONDS} to {MAX_WINDOW_SECONDS}")
        if start is not None and end is not None:
            elapsed = (end - start).total_seconds()
            if elapsed <= 0:
                errors.append("window: end_utc must be later than start_utc")
            elif _is_int(duration) and elapsed != duration:
                errors.append("window.duration_seconds: does not match timestamps")
            if created is not None and end > created:
                errors.append("created_at_utc: cannot precede the measurement window end")

    aggregation = document.get("aggregation")
    if _exact_keys(aggregation, AGGREGATION_KEYS, "aggregation", errors):
        if aggregation.get("scope") not in ("distribution_area", "microgrid"):
            errors.append("aggregation.scope: unsupported scope")
        token = aggregation.get("region_token")
        if not isinstance(token, str) or not REGION_TOKEN_RE.fullmatch(token):
            errors.append("aggregation.region_token: must be opaque group-level token, not a meter ID or coordinate")
        cohort = aggregation.get("cohort_size")
        minimum = aggregation.get("minimum_cohort_size")
        if not _is_int(minimum) or minimum < MINIMUM_RESIDENTIAL_COHORT:
            errors.append(f"aggregation.minimum_cohort_size: policy floor is {MINIMUM_RESIDENTIAL_COHORT}")
        if not _is_int(cohort) or cohort < MINIMUM_RESIDENTIAL_COHORT:
            errors.append(f"aggregation.cohort_size: must be integer of at least {MINIMUM_RESIDENTIAL_COHORT}")
        elif _is_int(minimum) and cohort < minimum:
            errors.append("aggregation.cohort_size: below the declared minimum cohort size")

        energy = aggregation.get("energy_wh_total")
        if not _is_int(energy) or not 0 <= energy <= 1_000_000_000_000_000:
            errors.append("aggregation.energy_wh_total: must be a bounded non-negative integer Wh total")
        method = aggregation.get("method")
        if method not in (
            "trusted_edge_aggregation", "secure_aggregation", "secure_aggregation_with_dp"
        ):
            errors.append("aggregation.method: unsupported aggregation method")
        coverage = aggregation.get("coverage_fraction")
        if not _is_finite_number(coverage) or not 0 <= coverage <= 1:
            errors.append("aggregation.coverage_fraction: must be finite and within [0,1]")
        if aggregation.get("measurement_quality") not in ("validated", "partial", "estimated"):
            errors.append("aggregation.measurement_quality: unsupported quality state")
    else:
        method = None

    recipient = document.get("recipient_class")
    if recipient not in ("grid_operations", "approved_research", "public_reporting"):
        errors.append("recipient_class: unsupported recipient class")

    dp_release = document.get("dp_release")
    if dp_release is not None:
        if _exact_keys(dp_release, DP_KEYS, "dp_release", errors):
            mechanism = dp_release.get("mechanism")
            if mechanism not in ("discrete_laplace", "gaussian"):
                errors.append("dp_release.mechanism: unsupported mechanism")
            epsilon, delta = dp_release.get("epsilon"), dp_release.get("delta")
            if not _is_finite_number(epsilon) or not 0 < epsilon <= MAX_RELEASE_EPSILON:
                errors.append(f"dp_release.epsilon: must be finite and in (0,{MAX_RELEASE_EPSILON}] per release")
            if not _is_finite_number(delta) or not 0 <= delta <= MAX_RELEASE_DELTA:
                errors.append(f"dp_release.delta: must be finite and in [0,{MAX_RELEASE_DELTA}]")
            policy_id = dp_release.get("budget_policy_id")
            if not isinstance(policy_id, str) or not POLICY_ID_RE.fullmatch(policy_id):
                errors.append("dp_release.budget_policy_id: invalid identifier")
            receipt = dp_release.get("budget_receipt_sha256")
            if not isinstance(receipt, str) or not SHA256_RE.fullmatch(receipt):
                errors.append("dp_release.budget_receipt_sha256: invalid SHA-256 receipt reference")

    if recipient in ("approved_research", "public_reporting"):
        if method != "secure_aggregation_with_dp":
            errors.append("recipient_class: research/public release requires secure aggregation with DP")
        if dp_release is None:
            errors.append("dp_release: required for research/public release")
    if method == "secure_aggregation_with_dp" and dp_release is None:
        errors.append("dp_release: required when aggregation.method uses DP")
    if method in ("trusted_edge_aggregation", "secure_aggregation") and dp_release is not None:
        errors.append("dp_release: must be null when the declared method does not apply DP")

    return errors


def load_and_validate(raw: str | bytes | bytearray) -> tuple[Any | None, list[str]]:
    try:
        document = strict_json_loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        return None, [f"JSON: invalid or ambiguous input ({exc})"]
    errors = validate_envelope(document)
    return document, errors
