#!/usr/bin/env python3
"""Test-only vectors for validating evaluator-side search exposure match reports."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from search_receipt_test_vectors import reseal

PREFIX = b"symthaea.science_bench_refresh.search_match.v1\0"
MATCH_SCHEMA = "symthaea.science_bench_refresh_search_exposure_match_report.v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def framed_hash(domain: bytes, payload: bytes) -> str:
    return sha256(
        PREFIX
        + len(domain).to_bytes(8, "big")
        + domain
        + len(payload).to_bytes(8, "big")
        + payload
    )


def make_match_report(
    receipt_path: Path,
    classification: str,
    method: str,
    score: int,
    threshold: int,
    protected_ref: str | None,
    evidence: str | None,
) -> dict[str, Any]:
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    record = {
        "event_sha256": receipt["events"][0]["event_sha256"],
        "classification": classification,
        "match_method": method,
        "match_score_micros": score,
        "decision_threshold_micros": threshold,
        "protected_reference_sha256": protected_ref,
        "evidence_artifact_sha256": evidence,
    }
    report: dict[str, Any] = {
        "schema": MATCH_SCHEMA,
        "run_id": receipt["run_id"],
        "generation_id": receipt["generation_id"],
        "receipt_sha256": sha256(receipt_bytes),
        "receipt_commitment_sha256": receipt["receipt_commitment_sha256"],
        "matcher_revision_sha256": sha256(b"frozen-matcher-revision"),
        "decision_rules_sha256": sha256(b"frozen-search-match-thresholds"),
        "expected_event_count": len(receipt["events"]),
        "records": [record],
        "report_commitment_sha256": "0" * 64,
    }
    payload = {key: value for key, value in report.items() if key != "report_commitment_sha256"}
    report["report_commitment_sha256"] = framed_hash(b"match-report.v1", encode(payload))
    return report


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: search_match_test_vectors.py OUTPUT_DIRECTORY", file=sys.stderr)
        return 2
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)

    receipt = reseal(
        "audited_open_web",
        [],
        [{
            "event_kind": "fetch",
            "decision": "allow",
            "request": "https://example.org/research",
            "host": "example.org",
            "response": "retrieved response body is not stored in the report",
        }],
    )
    receipt_path = output / "match-receipt.json"
    write_json(receipt_path, receipt)

    no_match = make_match_report(
        receipt_path, "no_match", "none", 0, 500_000, None, None
    )
    write_json(output / "no-match-report.json", no_match)

    explicit_answer = make_match_report(
        receipt_path,
        "explicit_answer",
        "exact_digest",
        1_000_000,
        1_000_000,
        sha256(b"protected-answer-item"),
        sha256(b"evidence-snippet-in-evaluator-store"),
    )
    write_json(output / "explicit-answer-report.json", explicit_answer)

    review = make_match_report(
        receipt_path,
        "review_required",
        "semantic_model",
        420_000,
        700_000,
        sha256(b"protected-context-item"),
        sha256(b"candidate-evidence-snippet"),
    )
    write_json(output / "review-required-report.json", review)

    missing = json.loads(json.dumps(no_match))
    missing["records"] = []
    payload = {key: value for key, value in missing.items() if key != "report_commitment_sha256"}
    missing["report_commitment_sha256"] = framed_hash(b"match-report.v1", encode(payload))
    write_json(output / "missing-match-record.json", missing)

    mutated = json.loads(json.dumps(no_match))
    mutated["records"][0]["classification"] = "explicit_answer"
    write_json(output / "mutated-match-report.json", mutated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
