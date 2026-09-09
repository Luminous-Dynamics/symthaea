#!/usr/bin/env python3
"""HAK-016 cardinality-aware optional-selector coverage records.

This is audit/evidence tooling only. It derives a new adjacent coverage record
from a HAK-014 normalization-execution receipt without mutating or reinterpreting
that historical receipt in place.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import hak_canonical_json as canonical
import hak_normalization_execute as hak14

SCHEMA_VERSION = "hak.selector-coverage.v1"
SOURCE_SCHEMA_VERSION = "hak.normalization-execution-receipt.v1"
CANONICAL_PROFILE = canonical.PROFILE_ID
DIGEST_DOMAIN = SCHEMA_VERSION
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SOURCE_OPTIONAL_STATUSES = {"Present", "Absent", "Failed"}
COVERAGE_STATES = {"Present", "PartiallyPresent", "Absent", "NotApplicable", "Failed"}


class SelectorCoverageError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectorCoverageError(message)


def _nonnegative_int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             f"{field} must be a non-negative integer")
    _require(value <= canonical.MAX_SAFE_INTEGER,
             f"{field} exceeds hak.canonical-json.v1 safe integer range")
    return value


def derive_optional_coverage(result: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(result, dict), "source optional selector result must be object")
    path = result.get("path")
    _require(isinstance(path, str) and path, "source optional selector path must be non-empty")
    source_status = result.get("status")
    _require(source_status in SOURCE_OPTIONAL_STATUSES,
             f"unsupported HAK-014 optional selector status: {source_status}")
    matches = _nonnegative_int(result.get("matches"), "matches")
    missing = _nonnegative_int(result.get("missing"), "missing")

    if source_status == "Failed":
        state = "Failed"
        applicable: int | None = None
    else:
        applicable = matches + missing
        _require(applicable <= canonical.MAX_SAFE_INTEGER, "applicable count exceeds canonical safe range")
        if matches > 0 and missing == 0:
            state = "Present"
        elif matches > 0 and missing > 0:
            state = "PartiallyPresent"
        elif matches == 0 and missing > 0:
            state = "Absent"
        else:
            state = "NotApplicable"

    return {
        "path": path,
        "source_status": source_status,
        "coverage_state": state,
        "applicable": applicable,
        "matches": matches,
        "missing": missing,
    }


def compute_coverage_digest(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "coverage_digest"}
    return canonical.hak_sha256(DIGEST_DOMAIN, payload)


def derive_coverage_record(receipt: dict[str, Any]) -> dict[str, Any]:
    try:
        hak14.validate_receipt(receipt)
    except Exception as exc:
        raise SelectorCoverageError(f"source HAK-014 receipt invalid: {exc}") from exc

    _require(receipt.get("schema_version") == SOURCE_SCHEMA_VERSION,
             "source receipt schema is not HAK-014 v1")
    optional = receipt.get("optional_selector_results")
    _require(isinstance(optional, list), "source optional_selector_results must be array")
    policy = receipt.get("policy")
    _require(isinstance(policy, dict), "source receipt policy must be object")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "canonicalization_profile": CANONICAL_PROFILE,
        "digest_domain": DIGEST_DOMAIN,
        "source_receipt": {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "receipt_digest": receipt.get("receipt_digest"),
            "execution_status": receipt.get("execution_status"),
        },
        "resource_kind": receipt.get("resource_kind"),
        "policy": {
            "policy_id": policy.get("policy_id"),
            "policy_digest": policy.get("policy_digest"),
        },
        "coverage_semantics": {
            "model": "HAKSelectorPathTerminalContextV1",
            "nonfailed_conservation": "applicable == matches + missing",
            "empty_wildcard_semantics": "zero applicable contexts -> NotApplicable",
            "historical_receipt_status_preserved": True,
        },
        "optional_results": [derive_optional_coverage(item) for item in optional],
    }
    record["coverage_digest"] = compute_coverage_digest(record)
    return record


def _validate_coverage_result(result: Any) -> None:
    _require(isinstance(result, dict), "coverage result must be object")
    _require(set(result) == {"path", "source_status", "coverage_state", "applicable", "matches", "missing"},
             "coverage result fields must match HAK-016 v1 exactly")
    path = result.get("path")
    _require(isinstance(path, str) and path, "coverage result path invalid")
    source_status = result.get("source_status")
    state = result.get("coverage_state")
    _require(source_status in SOURCE_OPTIONAL_STATUSES, "coverage source_status invalid")
    _require(state in COVERAGE_STATES, "coverage_state invalid")
    matches = _nonnegative_int(result.get("matches"), "coverage.matches")
    missing = _nonnegative_int(result.get("missing"), "coverage.missing")
    applicable = result.get("applicable")

    if state == "Failed":
        _require(source_status == "Failed", "Failed coverage requires Failed HAK-014 source status")
        _require(applicable is None, "Failed coverage must not claim numeric applicability")
        return

    _require(source_status != "Failed", "non-Failed coverage cannot derive from Failed source status")
    applicable_i = _nonnegative_int(applicable, "coverage.applicable")
    _require(applicable_i == matches + missing,
             "coverage conservation violated: applicable must equal matches + missing")
    if state == "Present":
        _require(applicable_i > 0 and matches == applicable_i and missing == 0,
                 "Present requires full positive coverage")
    elif state == "PartiallyPresent":
        _require(matches > 0 and missing > 0 and applicable_i > matches,
                 "PartiallyPresent requires both matches and missing contexts")
    elif state == "Absent":
        _require(applicable_i > 0 and matches == 0 and missing == applicable_i,
                 "Absent requires applicable contexts with zero matches")
    elif state == "NotApplicable":
        _require(applicable_i == 0 and matches == 0 and missing == 0,
                 "NotApplicable requires zero applicable contexts")


def validate_coverage_record(record: dict[str, Any], receipt: dict[str, Any]) -> None:
    _require(isinstance(record, dict), "coverage record must be object")
    _require(record.get("schema_version") == SCHEMA_VERSION, "coverage schema_version invalid")
    _require(record.get("canonicalization_profile") == CANONICAL_PROFILE,
             "coverage canonicalization profile mismatch")
    _require(record.get("digest_domain") == DIGEST_DOMAIN, "coverage digest domain mismatch")
    source = record.get("source_receipt")
    _require(isinstance(source, dict), "source_receipt must be object")
    _require(source.get("schema_version") == SOURCE_SCHEMA_VERSION, "source receipt schema binding invalid")
    _require(DIGEST_RE.fullmatch(str(source.get("receipt_digest", ""))) is not None,
             "source receipt digest invalid")
    _require(isinstance(record.get("resource_kind"), str) and record["resource_kind"],
             "resource_kind must be non-empty")
    policy = record.get("policy")
    _require(isinstance(policy, dict), "coverage policy must be object")
    _require(isinstance(policy.get("policy_id"), str) and policy["policy_id"], "coverage policy_id invalid")
    _require(DIGEST_RE.fullmatch(str(policy.get("policy_digest", ""))) is not None,
             "coverage policy_digest invalid")
    semantics = record.get("coverage_semantics")
    _require(semantics == {
        "model": "HAKSelectorPathTerminalContextV1",
        "nonfailed_conservation": "applicable == matches + missing",
        "empty_wildcard_semantics": "zero applicable contexts -> NotApplicable",
        "historical_receipt_status_preserved": True,
    }, "coverage semantics object drifted")
    results = record.get("optional_results")
    _require(isinstance(results, list), "optional_results must be array")
    for result in results:
        _validate_coverage_result(result)
    digest = record.get("coverage_digest")
    _require(isinstance(digest, str) and DIGEST_RE.fullmatch(digest) is not None,
             "coverage_digest invalid")
    _require(digest == compute_coverage_digest(record), "coverage_digest mismatch")

    expected = derive_coverage_record(receipt)
    _require(record == expected,
             "coverage record is not the deterministic HAK-016 derivation of the supplied HAK-014 receipt")


def validate_coverage_record_against_inputs(
    record: dict[str, Any],
    receipt: dict[str, Any],
    raw_bytes: bytes,
    policy: dict[str, Any],
    interpreter_bytes: bytes,
    *,
    policy_artifact_ref: str,
    interpreter_ref: str,
    raw_source_ref: str,
    resource_kind: str,
) -> None:
    try:
        hak14.validate_receipt_against_inputs(
            receipt,
            raw_bytes,
            policy,
            interpreter_bytes,
            policy_artifact_ref=policy_artifact_ref,
            interpreter_ref=interpreter_ref,
            raw_source_ref=raw_source_ref,
            resource_kind=resource_kind,
        )
    except Exception as exc:
        raise SelectorCoverageError(f"source HAK-014 input/replay validation failed: {exc}") from exc
    validate_coverage_record(record, receipt)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} root must be object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive/validate HAK-016 selector coverage from a HAK-014 receipt")
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--coverage", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = _load_json(args.receipt)
        derived = derive_coverage_record(receipt)
        if args.coverage:
            record = _load_json(args.coverage)
            validate_coverage_record(record, receipt)
            print(f"OK   {args.coverage} (HAK-016 selector coverage)")
        else:
            print(json.dumps(derived, ensure_ascii=False, indent=2))
    except (OSError, json.JSONDecodeError, SelectorCoverageError) as exc:
        print(f"FAIL {args.coverage or args.receipt}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
