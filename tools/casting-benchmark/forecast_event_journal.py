#!/usr/bin/env python3
"""Audit claimed forecast-before-outcome chronology in a hash-chained event journal.

The journal proves internal consistency and tamper evidence relative to its current
bytes. It does not prove that event times are truthful or independently timestamped.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

from forecast_validation import canonical_json, forecast_sha256, score_forecasts

SCHEMA_VERSION = "1.0.0"
GENESIS_SHA256 = "0" * 64
COMMON_FIELDS = frozenset({
    "sequence", "event_id", "kind", "forecast_id", "occurred_at",
    "artifact_sha256", "previous_event_sha256", "event_sha256",
})
FORECAST_EVENT_FIELDS = COMMON_FIELDS
OUTCOME_EVENT_FIELDS = COMMON_FIELDS | {"forecast_event_sha256"}
UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty string without edge whitespace")
    return value


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not UTC_TIMESTAMP.fullmatch(value):
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp ending in Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid UTC timestamp") from exc


def event_sha256(event: dict[str, Any]) -> str:
    """Hash a journal event excluding only its self-referential event_sha256 field."""
    if not isinstance(event, dict) or event.get("kind") not in {
        "forecast_published", "outcome_observed"
    }:
        raise ValueError("event kind or shape is invalid")
    fields = FORECAST_EVENT_FIELDS if event["kind"] == "forecast_published" else OUTCOME_EVENT_FIELDS
    allowed = fields - {"event_sha256"}
    if set(event) not in (allowed, fields):
        raise ValueError("event keys do not match the exact schema for its kind")
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def seal_event(event: dict[str, Any]) -> dict[str, Any]:
    """Test/producer helper: return a copy with its canonical event digest."""
    candidate = dict(event)
    candidate.pop("event_sha256", None)
    candidate["event_sha256"] = event_sha256(candidate)
    return candidate


def audit_event_journal(
    forecasts: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Verify artifact bindings, contiguous sequence, hash chain and claimed chronology."""
    # Reuse the scoring module as the source of truth for the forecast/outcome schema,
    # identity uniqueness, forecast digest binding and outcome digest validation.
    score_forecasts(forecasts, outcomes)
    if isinstance(events, (str, bytes)) or not isinstance(events, Sequence) or not events:
        raise ValueError("events must be a non-empty sequence")

    forecast_by_id = {item["forecast_id"]: item for item in forecasts}
    outcome_by_id = {item["forecast_id"]: item for item in outcomes}
    forecast_digests = {key: forecast_sha256(value) for key, value in forecast_by_id.items()}
    forecast_events: dict[str, dict[str, Any]] = {}
    outcome_events: dict[str, dict[str, Any]] = {}
    event_ids: set[str] = set()
    prior_hash = GENESIS_SHA256
    prior_time: datetime | None = None

    for expected_sequence, raw_event in enumerate(events):
        if not isinstance(raw_event, dict):
            raise ValueError(f"event at index {expected_sequence} must be an object")
        kind = raw_event.get("kind")
        fields = FORECAST_EVENT_FIELDS if kind == "forecast_published" else (
            OUTCOME_EVENT_FIELDS if kind == "outcome_observed" else None
        )
        if fields is None or set(raw_event) != fields:
            raise ValueError(f"event at index {expected_sequence} has invalid kind or keys")

        sequence = raw_event["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != expected_sequence:
            raise ValueError(f"event sequence must be contiguous from zero; expected {expected_sequence}")
        event_id = _nonblank(raw_event["event_id"], "event_id")
        if event_id in event_ids:
            raise ValueError(f"duplicate event_id: {event_id}")
        event_ids.add(event_id)
        forecast_id = _nonblank(raw_event["forecast_id"], "forecast_id")
        timestamp = _parse_utc(raw_event["occurred_at"], "occurred_at")
        if prior_time is not None and timestamp < prior_time:
            raise ValueError("event timestamps decrease in journal sequence")
        prior_time = timestamp

        previous_hash = _sha256(raw_event["previous_event_sha256"], "previous_event_sha256")
        if previous_hash != prior_hash:
            raise ValueError(f"hash-chain predecessor mismatch at sequence {expected_sequence}")
        calculated_hash = event_sha256(raw_event)
        recorded_hash = _sha256(raw_event["event_sha256"], "event_sha256")
        if recorded_hash != calculated_hash:
            raise ValueError(f"event digest mismatch at sequence {expected_sequence}")
        artifact_digest = _sha256(raw_event["artifact_sha256"], "artifact_sha256")

        if kind == "forecast_published":
            if forecast_id not in forecast_by_id:
                raise ValueError(f"publication event references unknown forecast_id: {forecast_id}")
            if forecast_id in forecast_events:
                raise ValueError(f"duplicate publication event for forecast_id: {forecast_id}")
            if artifact_digest != forecast_digests[forecast_id]:
                raise ValueError(f"publication artifact digest mismatch for {forecast_id}")
            forecast_events[forecast_id] = {
                "event_sha256": recorded_hash,
                "timestamp": timestamp,
                "sequence": sequence,
            }
        else:
            if forecast_id not in outcome_by_id:
                raise ValueError(f"observation event references unknown outcome forecast_id: {forecast_id}")
            if forecast_id in outcome_events:
                raise ValueError(f"duplicate outcome observation event for {forecast_id}")
            if forecast_id not in forecast_events:
                raise ValueError(f"outcome observed before publication event for {forecast_id}")
            publication = forecast_events[forecast_id]
            if timestamp <= publication["timestamp"]:
                raise ValueError(f"outcome timestamp must be strictly later than forecast publication for {forecast_id}")
            if sequence <= publication["sequence"]:
                raise ValueError(f"outcome sequence must follow publication for {forecast_id}")
            if _sha256(raw_event["forecast_event_sha256"], "forecast_event_sha256") != publication["event_sha256"]:
                raise ValueError(f"outcome does not link to exact publication event for {forecast_id}")
            if artifact_digest != outcome_by_id[forecast_id]["outcome_sha256"]:
                raise ValueError(f"observation artifact digest mismatch for {forecast_id}")
            outcome_events[forecast_id] = {"event_sha256": recorded_hash, "timestamp": timestamp}

        prior_hash = recorded_hash

    expected_ids = set(forecast_by_id)
    if set(forecast_events) != expected_ids:
        raise ValueError(f"publication event coverage mismatch; missing={sorted(expected_ids - set(forecast_events))}")
    if set(outcome_events) != set(outcome_by_id):
        raise ValueError(f"outcome event coverage mismatch; missing={sorted(set(outcome_by_id) - set(outcome_events))}")

    return {
        "schema_version": SCHEMA_VERSION,
        "forecast_count": len(forecast_by_id),
        "outcome_count": len(outcome_by_id),
        "event_count": len(events),
        "journal_head_sha256": prior_hash,
        "chronology_check": "DECLARED_UTC_TIMES_AND_SEQUENCE_ARE_CONSISTENT",
        "claim_ceiling": (
            "Validates declared timestamp ordering, event integrity and artifact bindings. "
            "A self-maintained hash chain does not establish truthful event times, "
            "independent timestamping, signature authenticity, or holdout independence."
        ),
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _load_json(path: Path, label: str) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid strict UTF-8 JSON") from exc
    return raw, value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecasts", type=Path, help="forecast manifest used for scoring")
    parser.add_argument("outcomes", type=Path, help="digest-bound outcome manifest")
    parser.add_argument("events", type=Path, help="hash-chained event journal manifest")
    parser.add_argument("--output", type=Path, required=True, help="receipt path (must differ from inputs)")
    args = parser.parse_args(argv)
    try:
        input_paths = {args.forecasts.resolve(), args.outcomes.resolve(), args.events.resolve()}
        output_path = args.output.resolve()
        if len(input_paths) != 3:
            raise ValueError("forecast, outcome, and event inputs must be distinct paths")
        if output_path in input_paths:
            raise ValueError("receipt output must not overwrite an input manifest")
        forecast_raw, forecast_doc = _load_json(args.forecasts, "forecast manifest")
        outcome_raw, outcome_doc = _load_json(args.outcomes, "outcome manifest")
        event_raw, event_doc = _load_json(args.events, "event journal")
        if not all(isinstance(doc, list) for doc in (forecast_doc, outcome_doc, event_doc)):
            raise ValueError("forecast, outcome, and event manifest roots must be JSON arrays")
        receipt = audit_event_journal(forecast_doc, outcome_doc, event_doc)
        receipt["receipt_label"] = "FORECAST_EVENT_JOURNAL_VALIDATION_NOT_TRUSTED_TIMESTAMPING"
        receipt["source_artifacts"] = {
            "forecast_manifest_sha256": hashlib.sha256(forecast_raw).hexdigest(),
            "outcome_manifest_sha256": hashlib.sha256(outcome_raw).hexdigest(),
            "event_journal_sha256": hashlib.sha256(event_raw).hexdigest(),
        }
        receipt["evaluator_sha256"] = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"forecast event journal validation failed: {exc}", file=sys.stderr)
        return 2
    print(f"forecast event journal receipt written: {output_path}")
    print(f"journal_head_sha256={receipt['journal_head_sha256']}")
    print(f"event_count={receipt['event_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
