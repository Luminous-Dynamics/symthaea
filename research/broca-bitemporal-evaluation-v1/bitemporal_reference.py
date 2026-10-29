#!/usr/bin/env python3
"""Bitemporal evidence resolver: what was true when, given what was known when.

Research reference semantics only; not production memory integration.
Valid-time intervals are half-open [valid_from, valid_to). Transaction time
(recorded_at) controls whether a claim was available at a knowledge cutoff.
Within a valid-time point, the most recent effective start wins; authority_rank
breaks ties at that same start. Equally ranked conflicting values return
"conflict", never an arbitrary winner.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_EVENTS = HERE / "fixtures" / "events.jsonl"
DEFAULT_QUERIES = HERE / "fixtures" / "queries.jsonl"
REPORT_SCHEMA = "broca-bitemporal-report-v1"
MAX_LINE_BYTES = 4 * 1024 * 1024
EVENT_FIELDS = {
    "event_id", "subject", "attribute", "value", "valid_from", "valid_to",
    "recorded_at", "source", "authority_rank",
}
QUERY_FIELDS = {
    "query_id", "subject", "attribute", "valid_at", "known_at",
    "expected_state", "expected_value",
}
STATES = {"resolved", "conflict", "unknown"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    payload = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if len(line) > MAX_LINE_BYTES:
            raise ValueError(f"{path}:{line_number}: line exceeds byte bound")
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid UTF-8 JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        records.append(value)
    if not records:
        raise ValueError(f"{path}: input must not be empty")
    return records, payload


def _nonempty_string(record: dict[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {key} must be a non-empty string")
    return value


def parse_day(value: Any, context: str) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError(f"{context}: expected date in YYYY-MM-DD form")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid calendar date {value!r}") from exc


def parse_timestamp(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: timestamp must be a non-empty RFC3339 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{context}: timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def validate_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        context = f"event row {index}"
        if set(raw) != EVENT_FIELDS:
            raise ValueError(
                f"{context}: fields must match schema exactly; "
                f"missing={sorted(EVENT_FIELDS - set(raw))}, "
                f"extra={sorted(set(raw) - EVENT_FIELDS)}"
            )
        event = dict(raw)
        event_id = _nonempty_string(event, "event_id", context)
        if event_id in seen:
            raise ValueError(f"{context}: duplicate event_id {event_id!r}")
        seen.add(event_id)
        for key in ("subject", "attribute", "value", "source"):
            _nonempty_string(event, key, context)
        valid_from = parse_day(event["valid_from"], f"{context} valid_from")
        valid_to = None
        if event["valid_to"] is not None:
            valid_to = parse_day(event["valid_to"], f"{context} valid_to")
            if valid_to <= valid_from:
                raise ValueError(f"{context}: valid_to must be later than valid_from")
        recorded_at = parse_timestamp(event["recorded_at"], f"{context} recorded_at")
        rank = event["authority_rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError(f"{context}: authority_rank must be a non-negative integer")
        event["_valid_from"] = valid_from
        event["_valid_to"] = valid_to
        event["_recorded_at"] = recorded_at
        events.append(event)
    return events


def validate_queries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    queries: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        context = f"query row {index}"
        if set(raw) != QUERY_FIELDS:
            raise ValueError(
                f"{context}: fields must match schema exactly; "
                f"missing={sorted(QUERY_FIELDS - set(raw))}, "
                f"extra={sorted(set(raw) - QUERY_FIELDS)}"
            )
        query = dict(raw)
        query_id = _nonempty_string(query, "query_id", context)
        if query_id in seen:
            raise ValueError(f"{context}: duplicate query_id {query_id!r}")
        seen.add(query_id)
        for key in ("subject", "attribute"):
            _nonempty_string(query, key, context)
        query["_valid_at"] = parse_day(query["valid_at"], f"{context} valid_at")
        query["_known_at"] = parse_timestamp(query["known_at"], f"{context} known_at")
        expected_state = query["expected_state"]
        if not isinstance(expected_state, str) or expected_state not in STATES:
            raise ValueError(f"{context}: expected_state must be one of {sorted(STATES)}")
        expected_value = query["expected_value"]
        if expected_state == "resolved":
            if not isinstance(expected_value, str) or not expected_value:
                raise ValueError(f"{context}: resolved gold requires a non-empty expected_value")
        elif expected_value is not None:
            raise ValueError(f"{context}: conflict/unknown gold must have expected_value=null")
        queries.append(query)
    return queries


def resolve_query(events: list[dict[str, Any]], query: dict[str, Any]) -> dict[str, Any]:
    """Resolve one (subject, attribute, valid_at, known_at) point with an audit trail."""
    matching: list[dict[str, Any]] = []
    for event in events:
        if event["subject"] != query["subject"] or event["attribute"] != query["attribute"]:
            continue
        if event["_recorded_at"] > query["_known_at"]:
            continue
        if event["_valid_from"] > query["_valid_at"]:
            continue
        if event["_valid_to"] is not None and query["_valid_at"] >= event["_valid_to"]:
            continue
        matching.append(event)

    base = {
        "query_id": query["query_id"],
        "subject": query["subject"],
        "attribute": query["attribute"],
        "valid_at": query["valid_at"],
        "known_at": query["known_at"],
        "eligible_event_ids": sorted(event["event_id"] for event in matching),
    }
    if not matching:
        return {
            **base,
            "state": "unknown",
            "value": None,
            "selected_event_ids": [],
            "selected_evidence": [],
            "shadowed_event_ids": [],
            "conflicting_values": [],
            "resolution_rule": "no claim both known by cutoff and valid at requested time",
        }

    latest_effective_start = max(event["_valid_from"] for event in matching)
    temporal_candidates = [
        event for event in matching if event["_valid_from"] == latest_effective_start
    ]
    highest_authority = max(event["authority_rank"] for event in temporal_candidates)
    selected = [
        event for event in temporal_candidates
        if event["authority_rank"] == highest_authority
    ]
    values = sorted({event["value"] for event in selected})
    shadowed = sorted(
        event["event_id"] for event in matching
        if event["event_id"] not in {item["event_id"] for item in selected}
    )
    selected_ids = sorted(event["event_id"] for event in selected)
    selected_evidence = [
        {
            "event_id": event["event_id"],
            "source": event["source"],
            "value": event["value"],
            "valid_from": event["valid_from"],
            "valid_to": event["valid_to"],
            "recorded_at": event["recorded_at"],
            "authority_rank": event["authority_rank"],
        }
        for event in sorted(selected, key=lambda item: item["event_id"])
    ]
    if len(values) > 1:
        return {
            **base,
            "state": "conflict",
            "value": None,
            "selected_event_ids": selected_ids,
            "selected_evidence": selected_evidence,
            "shadowed_event_ids": shadowed,
            "conflicting_values": values,
            "selected_valid_from": latest_effective_start.isoformat(),
            "selected_authority_rank": highest_authority,
            "resolution_rule": "equally ranked claims disagree at the latest effective start",
        }
    return {
        **base,
        "state": "resolved",
        "value": values[0],
        "selected_event_ids": selected_ids,
        "selected_evidence": selected_evidence,
        "shadowed_event_ids": shadowed,
        "conflicting_values": [],
        "selected_valid_from": latest_effective_start.isoformat(),
        "selected_authority_rank": highest_authority,
        "resolution_rule": "latest effective start, then highest authority rank",
    }


def evaluate(events: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, Any]:
    results = []
    for query in queries:
        result = resolve_query(events, query)
        is_correct = result["state"] == query["expected_state"]
        if is_correct and result["state"] == "resolved":
            is_correct = result["value"] == query["expected_value"]
        result["expected_state"] = query["expected_state"]
        result["expected_value"] = query["expected_value"]
        result["correct"] = is_correct
        results.append(result)
    correct = sum(bool(result["correct"]) for result in results)
    state_correct = sum(
        result["state"] == result["expected_state"] for result in results
    )
    resolved_count = sum(result["state"] == "resolved" for result in results)
    return {
        "schema": REPORT_SCHEMA,
        "query_count": len(results),
        "correct_count": correct,
        "exact_accuracy": correct / len(results),
        "state_accuracy": state_correct / len(results),
        "resolved_count": resolved_count,
        "conflict_count": sum(result["state"] == "conflict" for result in results),
        "unknown_count": sum(result["state"] == "unknown" for result in results),
        "results": results,
        "claim_ceiling": (
            "synthetic bitemporal reference semantics only; not production integration, "
            "not real-world truth, and not an end-to-end memory benchmark score"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    events_path = args.events.expanduser().resolve()
    queries_path = args.queries.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else None
    source_path = Path(__file__).resolve()
    protected_paths = {events_path, queries_path, source_path}
    if events_path == queries_path:
        raise SystemExit("--events and --queries must be different files")
    if output_path is not None and output_path in protected_paths:
        raise SystemExit("output path must differ from input fixtures and evaluator source")

    event_rows, event_bytes = load_jsonl(events_path)
    query_rows, query_bytes = load_jsonl(queries_path)
    events = validate_events(event_rows)
    queries = validate_queries(query_rows)
    report = evaluate(events, queries)
    report["receipt"] = {
        "events_sha256": sha256_bytes(event_bytes),
        "queries_sha256": sha256_bytes(query_bytes),
        "evaluator_sha256": sha256_bytes(source_path.read_bytes()),
        "events_path": str(events_path),
        "queries_path": str(queries_path),
    }
    rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if output_path is None:
        print(rendered, end="")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(json.dumps({
            "output": str(output_path),
            "schema": REPORT_SCHEMA,
            "query_count": report["query_count"],
            "exact_accuracy": report["exact_accuracy"],
            "receipt": report["receipt"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
