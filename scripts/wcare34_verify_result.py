#!/usr/bin/env python3
"""Verify WCARE-34 final result census and aggregate integrity."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "docs/release/evidence/WCARE34_CANDIDATE_FREEZE_V1.json"
ALLOWED_CLASSIFICATIONS = {
    "PASS_HOLDOUT",
    "FAIL_HOLDOUT",
    "INVALID_HOLDOUT",
    "INFRASTRUCTURE_INDETERMINATE",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    return raw, json.loads(raw)


def emit(payload: dict) -> int:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if payload.get("report_integrity_verified") else 1


def invalid(detail: str, **extra: object) -> int:
    payload = {
        "authority": "MeasurementOnly",
        "classification": "INVALID_HOLDOUT",
        "detail": detail,
        "report_integrity_verified": False,
    }
    payload.update(extra)
    return emit(payload)


def parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("commitment", type=Path)
    parser.add_argument("reveal_receipt", type=Path)
    parser.add_argument("case_ids", type=Path)
    args = parser.parse_args()

    try:
        _, freeze = load_json(FREEZE_PATH)
        result_bytes, result = load_json(args.result)
        commitment_bytes, commitment = load_json(args.commitment)
        _, reveal = load_json(args.reveal_receipt)
        census_bytes = args.case_ids.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return invalid(f"artifact_read_or_parse_failed:{type(exc).__name__}")

    required = {
        "protocol_version","candidate_epoch","candidate_sha","commitment_sha256",
        "revealed_bundle_sha256","case_count_committed","case_count_scored",
        "pass_count","fail_count","excluded_count","failed_case_ids","excluded_cases",
        "classification","evaluator_lineage_class","evaluator_lineage_note",
        "evaluation_started_utc","evaluation_finished_utc","case_results",
        "hard_invariant_failed_case_ids","internal_wcare33_status",
    }
    missing = sorted(required.difference(result))
    if missing:
        return invalid("result_missing_required_fields", missing=missing)

    if result["classification"] not in ALLOWED_CLASSIFICATIONS:
        return invalid("invalid_result_classification")

    for field in ("protocol_version","candidate_epoch","candidate_sha"):
        if result[field] != freeze[field] or result[field] != commitment.get(field):
            return invalid(f"candidate_binding_mismatch:{field}")
    if reveal.get("classification") != "REVEAL_VERIFIED":
        return invalid("reveal_receipt_not_verified")
    if reveal.get("candidate_epoch") != freeze["candidate_epoch"] or reveal.get("candidate_sha") != freeze["candidate_sha"]:
        return invalid("reveal_receipt_candidate_mismatch")

    commitment_sha = sha256_bytes(commitment_bytes)
    if result["commitment_sha256"] != commitment_sha or reveal.get("commitment_sha256") != commitment_sha:
        return invalid("commitment_digest_mismatch")
    if result["revealed_bundle_sha256"] != commitment.get("bundle_sha256") or result["revealed_bundle_sha256"] != reveal.get("bundle_sha256"):
        return invalid("revealed_bundle_digest_mismatch")
    if result["evaluator_lineage_class"] != commitment.get("evaluator_lineage_class") or reveal.get("evaluator_lineage_class") != commitment.get("evaluator_lineage_class"):
        return invalid("evaluator_lineage_class_mismatch")
    if result["evaluator_lineage_note"] != commitment.get("evaluator_lineage_note") or reveal.get("evaluator_lineage_note") != commitment.get("evaluator_lineage_note"):
        return invalid("evaluator_lineage_note_mismatch")
    if not isinstance(result["evaluator_lineage_note"], str) or not result["evaluator_lineage_note"].strip():
        return invalid("evaluator_lineage_note_missing")

    if not census_bytes.endswith(b"\n") or b"\r" in census_bytes:
        return invalid("case_census_not_canonical")
    try:
        census = census_bytes[:-1].decode("utf-8").split("\n")
    except UnicodeDecodeError:
        return invalid("case_census_not_utf8")
    if any(not case_id for case_id in census) or len(set(census)) != len(census):
        return invalid("case_census_blank_or_duplicate")
    if sha256_bytes(census_bytes) != commitment.get("case_id_commitment_sha256"):
        return invalid("case_census_digest_mismatch")
    if len(census) != commitment.get("case_count") or len(census) != result["case_count_committed"]:
        return invalid("committed_case_count_mismatch")

    case_results = result["case_results"]
    if not isinstance(case_results, list) or len(case_results) != len(census):
        return invalid("case_result_census_length_mismatch")
    if any(not isinstance(entry, dict) for entry in case_results):
        return invalid("case_result_not_object")
    result_ids = [entry.get("case_id") for entry in case_results]
    if result_ids != census:
        return invalid("case_result_census_order_or_membership_mismatch")

    allowed_dispositions = {"PASS","FAIL","EXCLUDED"}
    if any(entry.get("disposition") not in allowed_dispositions for entry in case_results):
        return invalid("invalid_case_disposition")
    if any(not isinstance(entry.get("hard_invariant"), bool) for entry in case_results):
        return invalid("hard_invariant_flag_missing")

    pass_ids = [entry["case_id"] for entry in case_results if entry["disposition"] == "PASS"]
    fail_ids = [entry["case_id"] for entry in case_results if entry["disposition"] == "FAIL"]
    excluded_ids = [entry["case_id"] for entry in case_results if entry["disposition"] == "EXCLUDED"]
    hard_fail_ids = [entry["case_id"] for entry in case_results if entry["disposition"] == "FAIL" and entry["hard_invariant"]]

    if result["pass_count"] != len(pass_ids) or result["fail_count"] != len(fail_ids) or result["excluded_count"] != len(excluded_ids):
        return invalid("aggregate_count_mismatch")
    if result["case_count_scored"] != len(pass_ids) + len(fail_ids):
        return invalid("scored_count_mismatch")
    if result["case_count_scored"] + result["excluded_count"] != result["case_count_committed"]:
        return invalid("complete_census_accounting_mismatch")
    if result["failed_case_ids"] != fail_ids:
        return invalid("failed_case_id_census_mismatch")
    if result["hard_invariant_failed_case_ids"] != hard_fail_ids:
        return invalid("hard_invariant_failure_census_mismatch")

    excluded_cases = result["excluded_cases"]
    if not isinstance(excluded_cases, list) or any(not isinstance(entry, dict) for entry in excluded_cases):
        return invalid("excluded_case_metadata_invalid")
    if [entry.get("case_id") for entry in excluded_cases] != excluded_ids:
        return invalid("excluded_case_census_mismatch")
    for entry in excluded_cases:
        if not isinstance(entry.get("reason"), str) or not entry["reason"].strip() or not isinstance(entry.get("preregistered_exclusion"), bool):
            return invalid("excluded_case_metadata_invalid")

    try:
        started = parse_time(result["evaluation_started_utc"])
        finished = parse_time(result["evaluation_finished_utc"])
        if finished < started:
            return invalid("evaluation_time_regression")
    except (TypeError, ValueError):
        return invalid("evaluation_timestamp_invalid")

    internal_status = result["internal_wcare33_status"]
    if internal_status != "NOT_EXECUTED" and not result.get("internal_wcare33_receipt_sha256"):
        return invalid("internal_status_missing_receipt_digest")

    classification = result["classification"]
    if classification == "PASS_HOLDOUT":
        if hard_fail_ids:
            return invalid("pass_claim_contains_hard_invariant_failure")
        if any(not entry["preregistered_exclusion"] for entry in excluded_cases):
            return invalid("pass_claim_contains_nonpreregistered_exclusion")
    if classification == "FAIL_HOLDOUT" and not fail_ids:
        return invalid("fail_claim_without_failed_case")

    return emit({
        "authority": "MeasurementOnly",
        "classification": classification,
        "candidate_epoch": result["candidate_epoch"],
        "candidate_sha": result["candidate_sha"],
        "commitment_sha256": commitment_sha,
        "result_sha256": sha256_bytes(result_bytes),
        "case_count_committed": len(census),
        "pass_count": len(pass_ids),
        "fail_count": len(fail_ids),
        "excluded_count": len(excluded_ids),
        "hard_invariant_failed_case_ids": hard_fail_ids,
        "report_integrity_verified": True,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
    })


if __name__ == "__main__":
    raise SystemExit(main())
