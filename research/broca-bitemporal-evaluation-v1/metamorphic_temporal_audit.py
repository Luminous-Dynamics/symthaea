#!/usr/bin/env python3
"""Deterministic metamorphic audit for temporal evidence semantics.

This research harness checks invariants under controlled input mutations. It is
not an official benchmark and does not test a production memory worker.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import bitemporal_reference as br

HERE = Path(__file__).resolve().parent
DEFAULT_EVENTS = HERE / "fixtures" / "events.jsonl"
DEFAULT_QUERIES = HERE / "fixtures" / "queries.jsonl"
REPORT_SCHEMA = "broca-bitemporal-metamorphic-report-v1"
PERMUTATION_TRIALS = 20
QUERY_ORDER_TRIALS = 5
SEED = 20261029


def raw_event(event: dict[str, Any]) -> dict[str, Any]:
    """Remove parsed helper keys; fixture/CLI event schema remains exact."""
    return {key: event[key] for key in br.EVENT_FIELDS}


def semantic_signature(result: dict[str, Any]) -> tuple[Any, ...]:
    """Fields whose semantic result should survive irrelevant mutations."""
    return (
        result["state"],
        result["value"],
        tuple(result["selected_event_ids"]),
        tuple(result["conflicting_values"]),
        result.get("selected_valid_from"),
        result.get("selected_authority_rank"),
    )


def utc_string(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_same_start_event(
    query: dict[str, Any],
    source_evidence: dict[str, Any],
    event_id: str,
    *,
    value: str,
    rank: int | None = None,
) -> dict[str, Any]:
    """Create an event applicable at this query while respecting the cutoff."""
    recorded_at = query["_known_at"] - timedelta(seconds=1)
    selected_rank = source_evidence["authority_rank"] if rank is None else rank
    return {
        "event_id": event_id,
        "subject": query["subject"],
        "attribute": query["attribute"],
        "value": value,
        "valid_from": source_evidence["valid_from"],
        "valid_to": source_evidence["valid_to"],
        "recorded_at": utc_string(recorded_at),
        "source": "metamorphic-test",
        "authority_rank": selected_rank,
    }


def check_record(
    invariant: str,
    query_id: str,
    expected: Any,
    actual: Any,
    *,
    passed: bool,
    detail: str,
) -> dict[str, Any]:
    return {
        "invariant": invariant,
        "query_id": query_id,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
        "detail": detail,
    }


def run_audit(
    event_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    *,
    permutation_trials: int = PERMUTATION_TRIALS,
    query_order_trials: int = QUERY_ORDER_TRIALS,
    seed: int = SEED,
) -> dict[str, Any]:
    if permutation_trials < 1 or permutation_trials > 1000:
        raise ValueError("permutation_trials must be between 1 and 1000")
    if query_order_trials < 1 or query_order_trials > 1000:
        raise ValueError("query_order_trials must be between 1 and 1000")

    base_events = br.validate_events(deepcopy(event_rows))
    queries = br.validate_queries(deepcopy(query_rows))
    source_results = {
        query["query_id"]: br.resolve_query(base_events, query)
        for query in queries
    }
    checks: list[dict[str, Any]] = []
    rng = random.Random(seed)

    # Metamorphic property 1: event input order is not a hidden tie-break.
    for trial in range(permutation_trials):
        permuted_events = deepcopy(base_events)
        rng.shuffle(permuted_events)
        for query in queries:
            expected = semantic_signature(source_results[query["query_id"]])
            actual = semantic_signature(br.resolve_query(permuted_events, query))
            checks.append(check_record(
                "event_permutation_invariance",
                query["query_id"],
                repr(expected),
                repr(actual),
                passed=expected == actual,
                detail=f"deterministic permutation trial {trial + 1}/{permutation_trials}",
            ))

    # Metamorphic property 2: future-recorded evidence cannot leak through cutoff.
    for query in queries:
        candidate = {
            "event_id": f"META_FUTURE_{query['query_id']}",
            "subject": query["subject"],
            "attribute": query["attribute"],
            "value": "__FUTURE_EVENT_MUST_NOT_LEAK__",
            "valid_from": "1900-01-01",
            "valid_to": None,
            "recorded_at": "2099-12-31T23:59:59Z",
            "source": "metamorphic-future",
            "authority_rank": 2_000_000,
        }
        mutated = br.validate_events(
            [raw_event(event) for event in base_events] + [candidate]
        )
        expected = semantic_signature(source_results[query["query_id"]])
        actual = semantic_signature(br.resolve_query(mutated, query))
        checks.append(check_record(
            "future_record_cutoff_noninterference",
            query["query_id"],
            repr(expected),
            repr(actual),
            passed=expected == actual,
            detail="adds highly authoritative, valid-time-applicable evidence recorded after known_at",
        ))

    # Metamorphic properties 3-5: lower-ranked conflict, equal-rank conflict,
    # and redundant same-value evidence must all have distinct expected effects.
    for query in queries:
        base_result = source_results[query["query_id"]]
        selected = base_result["selected_evidence"]
        if not selected:
            checks.append(check_record(
                "authority_and_witness_mutations",
                query["query_id"],
                "not_applicable_without_selected_evidence",
                base_result["state"],
                passed=True,
                detail="explicitly skipped: base query has no selected evidence",
            ))
            continue

        chosen = selected[0]
        top_rank = max(item["authority_rank"] for item in selected)
        low_rank = top_rank - 1
        if low_rank >= 0:
            lower = make_same_start_event(
                query, chosen, f"META_LOW_{query['query_id']}",
                value="__LOWER_AUTHORITY_CONTRADICTION__",
                rank=low_rank,
            )
            low_events = br.validate_events(
                [raw_event(event) for event in base_events] + [lower]
            )
            low_result = br.resolve_query(low_events, query)
            expected = semantic_signature(base_result)
            actual = semantic_signature(low_result)
            checks.append(check_record(
                "lower_authority_noninterference",
                query["query_id"],
                repr(expected),
                repr(actual),
                passed=expected == actual,
                detail="adds a contradictory claim at the same effective start with strictly lower authority",
            ))

        if base_result["state"] == "resolved":
            competing = make_same_start_event(
                query, chosen, f"META_EQUAL_CONFLICT_{query['query_id']}",
                value="__EQUAL_AUTHORITY_CONTRADICTION__",
                rank=top_rank,
            )
            conflict_events = br.validate_events(
                [raw_event(event) for event in base_events] + [competing]
            )
            conflict_result = br.resolve_query(conflict_events, query)
            checks.append(check_record(
                "equal_authority_disagreement_abstains",
                query["query_id"],
                "conflict with null value",
                f"{conflict_result['state']} with value {conflict_result['value']!r}",
                passed=(
                    conflict_result["state"] == "conflict"
                    and conflict_result["value"] is None
                    and len(conflict_result["conflicting_values"]) >= 2
                ),
                detail="introduces a conflicting claim at the winning effective start and authority rank",
            ))

            corroborating = make_same_start_event(
                query, chosen, f"META_CORROBORATE_{query['query_id']}",
                value=base_result["value"],
                rank=top_rank,
            )
            corroboration_events = br.validate_events(
                [raw_event(event) for event in base_events] + [corroborating]
            )
            corroboration_result = br.resolve_query(corroboration_events, query)
            expected_value = (base_result["state"], base_result["value"])
            actual_value = (corroboration_result["state"], corroboration_result["value"])
            checks.append(check_record(
                "same_value_witness_preserves_answer",
                query["query_id"],
                repr(expected_value),
                repr(actual_value),
                passed=(
                    corroboration_result["state"] == "resolved"
                    and corroboration_result["value"] == base_result["value"]
                    and corroborating["event_id"] in corroboration_result["selected_event_ids"]
                ),
                detail="adds a same-rank, same-value witness; answer stays stable and provenance expands",
            ))

    # Metamorphic property 6: query input order does not change per-ID answers.
    for trial in range(query_order_trials):
        shuffled_queries = deepcopy(queries)
        rng.shuffle(shuffled_queries)
        output = {
            query["query_id"]: semantic_signature(br.resolve_query(base_events, query))
            for query in shuffled_queries
        }
        expected = {
            query_id: semantic_signature(result)
            for query_id, result in source_results.items()
        }
        checks.append(check_record(
            "query_order_invariance",
            f"batch-{trial + 1}",
            hashlib.sha256(json.dumps(
                {key: repr(value) for key, value in sorted(expected.items())},
                sort_keys=True,
            ).encode()).hexdigest(),
            hashlib.sha256(json.dumps(
                {key: repr(value) for key, value in sorted(output.items())},
                sort_keys=True,
            ).encode()).hexdigest(),
            passed=output == expected,
            detail="shuffles query execution order and compares results by stable query ID",
        ))

    passed_count = sum(item["passed"] for item in checks)
    failed_count = len(checks) - passed_count
    invariant_summary: dict[str, dict[str, int]] = {}
    for check in checks:
        counts = invariant_summary.setdefault(check["invariant"], {"passed": 0, "failed": 0})
        counts["passed" if check["passed"] else "failed"] += 1
    return {
        "schema": REPORT_SCHEMA,
        "seed": seed,
        "base_event_count": len(base_events),
        "query_count": len(queries),
        "permutation_trials": permutation_trials,
        "query_order_trials": query_order_trials,
        "check_count": len(checks),
        "passed_count": passed_count,
        "failed_count": failed_count,
        "all_passed": failed_count == 0,
        "invariant_summary": invariant_summary,
        "checks": checks,
        "claim_ceiling": (
            "deterministic metamorphic checks over the supplied synthetic fixtures; "
            "not an official benchmark score, not held-out generalization, and not production integration"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--permutation-trials", type=int, default=PERMUTATION_TRIALS)
    parser.add_argument("--query-order-trials", type=int, default=QUERY_ORDER_TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    events_path = args.events.expanduser().resolve()
    queries_path = args.queries.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else None
    source_path = Path(__file__).resolve()
    if events_path == queries_path:
        raise SystemExit("--events and --queries must be different files")
    if output_path in {events_path, queries_path, source_path}:
        raise SystemExit("output path must differ from input fixtures and audit source")

    event_rows, event_bytes = br.load_jsonl(events_path)
    query_rows, query_bytes = br.load_jsonl(queries_path)
    report = run_audit(
        event_rows,
        query_rows,
        permutation_trials=args.permutation_trials,
        query_order_trials=args.query_order_trials,
        seed=args.seed,
    )
    report["receipt"] = {
        "events_sha256": br.sha256_bytes(event_bytes),
        "queries_sha256": br.sha256_bytes(query_bytes),
        "resolver_sha256": br.sha256_bytes(Path(br.__file__).resolve().read_bytes()),
        "audit_sha256": br.sha256_bytes(source_path.read_bytes()),
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
            "check_count": report["check_count"],
            "passed_count": report["passed_count"],
            "failed_count": report["failed_count"],
            "all_passed": report["all_passed"],
            "receipt": report["receipt"],
        }, sort_keys=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
