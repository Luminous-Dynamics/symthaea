#!/usr/bin/env python3
"""Policy-pinned bitemporal ledger with explicit, auditable retractions.

Research reference semantics only. Events do not self-assert authority: a
versioned source registry is selected by knowledge cutoff and supplies rank,
scope, and retraction rights. A claim is admitted under the policy visible at
its recorded time, then considered under the policy visible at query time.
Retractions are append-only records authorized under the policy visible when
the retraction was recorded.
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
FIXTURES = HERE / "fixtures" / "pinned-ledger"
DEFAULT_POLICIES = FIXTURES / "policies.jsonl"
DEFAULT_EVENTS = FIXTURES / "events.jsonl"
DEFAULT_QUERIES = FIXTURES / "queries.jsonl"
REPORT_SCHEMA = "broca-policy-pinned-ledger-report-v1"
MAX_LINE_BYTES = 4 * 1024 * 1024
POLICY_FIELDS = {"policy_version", "recorded_at", "sources"}
SOURCE_FIELDS = {"source", "authority_rank", "attributes", "can_retract"}
ASSERT_FIELDS = {
    "event_id", "kind", "subject", "attribute", "value", "valid_from",
    "valid_to", "recorded_at", "source",
}
RETRACT_FIELDS = {"event_id", "kind", "target_event_id", "recorded_at", "source"}
QUERY_FIELDS = {
    "query_id", "subject", "attribute", "valid_at", "known_at",
    "expected_state", "expected_value", "expected_policy_version",
}
STATES = {"resolved", "conflict", "unknown"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if len(line) > MAX_LINE_BYTES:
            raise ValueError(f"{path}:{line_number}: line exceeds byte bound")
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid UTF-8 JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: input must not be empty")
    return rows, payload


def required_string(record: dict[str, Any], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: {key} must be a non-empty string")
    return value


def parse_day(value: Any, where: str) -> date:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError(f"{where}: date must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{where}: invalid calendar date {value!r}") from exc


def parse_timestamp(value: Any, where: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: timestamp must be a non-empty RFC3339 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{where}: invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{where}: timestamp must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def validate_policies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    previous_version = 0
    previous_time: datetime | None = None
    for index, raw in enumerate(rows):
        where = f"policy row {index}"
        if set(raw) != POLICY_FIELDS:
            raise ValueError(f"{where}: fields must match policy schema exactly")
        version = raw["policy_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(f"{where}: policy_version must be a positive integer")
        if version <= previous_version:
            raise ValueError(f"{where}: policy_version must increase strictly")
        recorded_at = parse_timestamp(raw["recorded_at"], f"{where} recorded_at")
        if previous_time is not None and recorded_at <= previous_time:
            raise ValueError(f"{where}: policy recorded_at must increase strictly")
        sources = raw["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"{where}: sources must be a non-empty array")
        source_map: dict[str, dict[str, Any]] = {}
        for source_index, source_raw in enumerate(sources):
            sw = f"{where} source {source_index}"
            if not isinstance(source_raw, dict) or set(source_raw) != SOURCE_FIELDS:
                raise ValueError(f"{sw}: fields must match source schema exactly")
            source_name = required_string(source_raw, "source", sw)
            rank = source_raw["authority_rank"]
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise ValueError(f"{sw}: authority_rank must be a non-negative integer")
            attributes = source_raw["attributes"]
            if (
                not isinstance(attributes, list)
                or not attributes
                or any(not isinstance(item, str) or not item.strip() for item in attributes)
                or len(set(attributes)) != len(attributes)
            ):
                raise ValueError(f"{sw}: attributes must be unique non-empty strings")
            can_retract = source_raw["can_retract"]
            if not isinstance(can_retract, bool):
                raise ValueError(f"{sw}: can_retract must be boolean")
            if source_name in source_map:
                raise ValueError(f"{where}: duplicate source {source_name!r}")
            source_map[source_name] = {
                "source": source_name,
                "authority_rank": rank,
                "attributes": list(attributes),
                "can_retract": can_retract,
            }
        policy = {
            "policy_version": version,
            "recorded_at": recorded_at,
            "sources": source_map,
            "_raw": raw,
        }
        policies.append(policy)
        previous_version = version
        previous_time = recorded_at
    return policies


def policy_at(policies: list[dict[str, Any]], cutoff: datetime) -> dict[str, Any] | None:
    eligible = [policy for policy in policies if policy["recorded_at"] <= cutoff]
    return eligible[-1] if eligible else None


def validate_events(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        where = f"event row {index}"
        if "kind" not in raw:
            raise ValueError(f"{where}: missing kind")
        kind = raw["kind"]
        required = ASSERT_FIELDS if kind == "assert" else RETRACT_FIELDS if kind == "retract" else None
        if required is None:
            raise ValueError(f"{where}: kind must be assert or retract")
        if set(raw) != required:
            extra = sorted(set(raw) - required)
            missing = sorted(required - set(raw))
            raise ValueError(f"{where}: exact schema mismatch; missing={missing}, extra={extra}")
        event = dict(raw)
        event_id = required_string(event, "event_id", where)
        if event_id in seen:
            raise ValueError(f"{where}: duplicate event_id {event_id!r}")
        seen.add(event_id)
        required_string(event, "source", where)
        event["_recorded_at"] = parse_timestamp(event["recorded_at"], f"{where} recorded_at")
        if kind == "assert":
            for key in ("subject", "attribute", "value"):
                required_string(event, key, where)
            event["_valid_from"] = parse_day(event["valid_from"], f"{where} valid_from")
            if event["valid_to"] is None:
                event["_valid_to"] = None
            else:
                event["_valid_to"] = parse_day(event["valid_to"], f"{where} valid_to")
                if event["_valid_to"] <= event["_valid_from"]:
                    raise ValueError(f"{where}: valid_to must be later than valid_from")
        else:
            required_string(event, "target_event_id", where)
            event["_valid_from"] = None
            event["_valid_to"] = None
        events.append(event)

    by_id = {event["event_id"]: event for event in events}
    retracted_targets: set[str] = set()
    for event in events:
        if event["kind"] != "retract":
            continue
        target_id = event["target_event_id"]
        target = by_id.get(target_id)
        if target is None or target["kind"] != "assert":
            raise ValueError(f"retraction {event['event_id']}: target must name an assertion")
        if target_id in retracted_targets:
            raise ValueError(f"multiple retractions target {target_id!r}; use a new explicit correction event")
        if event["_recorded_at"] < target["_recorded_at"]:
            raise ValueError(f"retraction {event['event_id']}: cannot precede target recorded_at")
        retracted_targets.add(target_id)
    return events


def validate_queries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    queries: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        where = f"query row {index}"
        if set(raw) != QUERY_FIELDS:
            raise ValueError(f"{where}: fields must match query schema exactly")
        query = dict(raw)
        query_id = required_string(query, "query_id", where)
        if query_id in seen:
            raise ValueError(f"{where}: duplicate query_id {query_id!r}")
        seen.add(query_id)
        for key in ("subject", "attribute"):
            required_string(query, key, where)
        query["_valid_at"] = parse_day(query["valid_at"], f"{where} valid_at")
        query["_known_at"] = parse_timestamp(query["known_at"], f"{where} known_at")
        if not isinstance(query["expected_state"], str) or query["expected_state"] not in STATES:
            raise ValueError(f"{where}: expected_state must be one of {sorted(STATES)}")
        if query["expected_state"] == "resolved":
            if not isinstance(query["expected_value"], str) or not query["expected_value"]:
                raise ValueError(f"{where}: resolved gold needs expected_value")
        elif query["expected_value"] is not None:
            raise ValueError(f"{where}: conflict/unknown gold needs expected_value=null")
        version = query["expected_policy_version"]
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise ValueError(f"{where}: expected_policy_version must be positive")
        queries.append(query)
    return queries


def source_policy_for_event(
    policies: list[dict[str, Any]], event: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    policy = policy_at(policies, event["_recorded_at"])
    if policy is None:
        return None, None, "no authority policy existed when event was recorded"
    source = policy["sources"].get(event["source"])
    if source is None:
        return policy, None, "source was not registered when event was recorded"
    if event["kind"] == "assert" and event["attribute"] not in source["attributes"]:
        return policy, source, "source was outside its authorized attribute scope when recorded"
    if event["kind"] == "retract":
        if not source["can_retract"]:
            return policy, source, "source lacked retraction authority when recorded"
    return policy, source, None


def validate_admission(
    events: list[dict[str, Any]], policies: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_id = {event["event_id"]: event for event in events}
    admission: dict[str, dict[str, Any]] = {}
    for event in events:
        policy, source, reason = source_policy_for_event(policies, event)
        if reason is None and event["kind"] == "retract":
            target = by_id[event["target_event_id"]]
            if target["attribute"] not in source["attributes"]:
                reason = "source lacked authority over target attribute when retraction was recorded"
        admission[event["event_id"]] = {
            "admitted": reason is None,
            "record_policy_version": policy["policy_version"] if policy else None,
            "record_authority_rank": source["authority_rank"] if source else None,
            "reason": reason,
        }
    return admission


def resolve_query(
    events: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    admission: dict[str, dict[str, Any]],
    query: dict[str, Any],
) -> dict[str, Any]:
    known_at = query["_known_at"]
    policy = policy_at(policies, known_at)
    base = {
        "query_id": query["query_id"],
        "subject": query["subject"],
        "attribute": query["attribute"],
        "valid_at": query["valid_at"],
        "known_at": query["known_at"],
        "policy_version": policy["policy_version"] if policy else None,
        "eligible_event_ids": [],
        "selected_event_ids": [],
        "selected_evidence": [],
        "excluded_events": [],
        "applied_retractions": [],
    }
    if policy is None:
        return {
            **base,
            "state": "unknown",
            "value": None,
            "conflicting_values": [],
            "resolution_rule": "no registered authority policy was known by cutoff",
        }

    by_id = {event["event_id"]: event for event in events}
    applied_retractions: set[str] = set()
    applied_retraction_ids: list[str] = []
    excluded: list[dict[str, str]] = []
    for event in events:
        if event["kind"] != "retract" or event["_recorded_at"] > known_at:
            continue
        target = by_id[event["target_event_id"]]
        if target["subject"] != query["subject"] or target["attribute"] != query["attribute"]:
            continue
        record = admission[event["event_id"]]
        if not record["admitted"]:
            excluded.append({"event_id": event["event_id"], "reason": record["reason"] or "retraction not admitted"})
            continue
        if not admission[target["event_id"]]["admitted"]:
            excluded.append({
                "event_id": event["event_id"],
                "reason": "retraction target was not admitted as evidence",
            })
            continue
        applied_retractions.add(target["event_id"])
        applied_retraction_ids.append(event["event_id"])

    eligible: list[dict[str, Any]] = []
    for event in events:
        if event["kind"] != "assert":
            continue
        if event["_recorded_at"] > known_at:
            continue
        if event["subject"] != query["subject"] or event["attribute"] != query["attribute"]:
            continue
        if not admission[event["event_id"]]["admitted"]:
            excluded.append({
                "event_id": event["event_id"],
                "reason": admission[event["event_id"]]["reason"] or "event not admitted",
            })
            continue
        current_source = policy["sources"].get(event["source"])
        if current_source is None:
            excluded.append({"event_id": event["event_id"], "reason": "source absent from policy at query cutoff"})
            continue
        if event["attribute"] not in current_source["attributes"]:
            excluded.append({"event_id": event["event_id"], "reason": "source outside current policy scope"})
            continue
        if event["event_id"] in applied_retractions:
            continue
        if event["_valid_from"] > query["_valid_at"]:
            continue
        if event["_valid_to"] is not None and query["_valid_at"] >= event["_valid_to"]:
            continue
        eligible.append(event)

    base["eligible_event_ids"] = sorted(event["event_id"] for event in eligible)
    base["applied_retractions"] = sorted(applied_retraction_ids)
    base["excluded_events"] = sorted(excluded, key=lambda item: (item["event_id"], item["reason"]))
    if not eligible:
        return {
            **base,
            "state": "unknown",
            "value": None,
            "conflicting_values": [],
            "resolution_rule": "no admitted evidence matched subject, attribute, valid time, and knowledge cutoff",
        }

    latest_start = max(event["_valid_from"] for event in eligible)
    temporal_candidates = [event for event in eligible if event["_valid_from"] == latest_start]
    rank = max(policy["sources"][event["source"]]["authority_rank"] for event in temporal_candidates)
    selected = [
        event for event in temporal_candidates
        if policy["sources"][event["source"]]["authority_rank"] == rank
    ]
    selected.sort(key=lambda event: event["event_id"])
    values = sorted({event["value"] for event in selected})
    selected_evidence = [
        {
            "event_id": event["event_id"],
            "source": event["source"],
            "value": event["value"],
            "valid_from": event["valid_from"],
            "valid_to": event["valid_to"],
            "recorded_at": event["recorded_at"],
            "authority_rank": policy["sources"][event["source"]]["authority_rank"],
            "record_policy_version": admission[event["event_id"]]["record_policy_version"],
        }
        for event in selected
    ]
    base["selected_event_ids"] = [event["event_id"] for event in selected]
    base["selected_evidence"] = selected_evidence
    if len(values) > 1:
        return {
            **base,
            "state": "conflict",
            "value": None,
            "conflicting_values": values,
            "selected_valid_from": latest_start.isoformat(),
            "selected_authority_rank": rank,
            "resolution_rule": "latest valid start, then policy-pinned authority; equal-rank disagreement abstains",
        }
    return {
        **base,
        "state": "resolved",
        "value": values[0],
        "conflicting_values": [],
        "selected_valid_from": latest_start.isoformat(),
        "selected_authority_rank": rank,
        "resolution_rule": "latest valid start, then policy-pinned authority",
    }


def evaluate(
    events: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    admission: dict[str, dict[str, Any]],
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    results = []
    for query in queries:
        result = resolve_query(events, policies, admission, query)
        result["expected_policy_version"] = query["expected_policy_version"]
        result["expected_state"] = query["expected_state"]
        result["expected_value"] = query["expected_value"]
        result["correct"] = (
            result["state"] == query["expected_state"]
            and result["policy_version"] == query["expected_policy_version"]
            and (
                result["state"] != "resolved"
                or result["value"] == query["expected_value"]
            )
        )
        results.append(result)
    return {
        "schema": REPORT_SCHEMA,
        "query_count": len(results),
        "correct_count": sum(result["correct"] for result in results),
        "exact_fixture_accuracy": sum(result["correct"] for result in results) / len(results),
        "admitted_event_count": sum(item["admitted"] for item in admission.values()),
        "excluded_event_count": sum(not item["admitted"] for item in admission.values()),
        "event_admission": [
            {"event_id": event_id, **record}
            for event_id, record in sorted(admission.items())
        ],
        "results": results,
        "claim_ceiling": (
            "policy-pinned synthetic ledger semantics only; no production integration, "
            "no held-out generalization result, and no official benchmark score"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, default=DEFAULT_POLICIES)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = {
        "policies": args.policies.expanduser().resolve(),
        "events": args.events.expanduser().resolve(),
        "queries": args.queries.expanduser().resolve(),
    }
    if len(set(paths.values())) != len(paths):
        raise SystemExit("policy, event, and query files must be different")
    source_path = Path(__file__).resolve()
    output_path = args.output.expanduser().resolve() if args.output else None
    if output_path is not None and output_path in set(paths.values()) | {source_path}:
        raise SystemExit("output path must differ from inputs and evaluator source")
    policy_rows, policy_bytes = load_jsonl(paths["policies"])
    event_rows, event_bytes = load_jsonl(paths["events"])
    query_rows, query_bytes = load_jsonl(paths["queries"])
    policies = validate_policies(policy_rows)
    events = validate_events(event_rows)
    queries = validate_queries(query_rows)
    admission = validate_admission(events, policies)
    report = evaluate(events, policies, admission, queries)
    report["receipt"] = {
        "policy_file_sha256": sha256_bytes(policy_bytes),
        "event_file_sha256": sha256_bytes(event_bytes),
        "query_file_sha256": sha256_bytes(query_bytes),
        "evaluator_sha256": sha256_bytes(source_path.read_bytes()),
        "policy_versions": [policy["policy_version"] for policy in policies],
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
            "correct_count": report["correct_count"],
            "exact_fixture_accuracy": report["exact_fixture_accuracy"],
            "receipt": report["receipt"],
        }, sort_keys=True))
    return 0 if report["correct_count"] == report["query_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
