#!/usr/bin/env python3
"""Validate immutable terminal qualification-attempt records and per-subject sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import integration_train_manifest as train

RECORD_SCHEMA = "symthaea.qualification-attempt-record.v1"
SET_SCHEMA = "symthaea.qualification-attempt-set.v1"
RECORD_DOMAIN = b"symthaea.qualification-attempt-record.v1\0"
SET_DOMAIN = b"symthaea.qualification-attempt-set.v1\0"

_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CHECKOUT_KINDS = {"exact_subject", "synthetic_merge_candidate", "other"}
_THEOREM_OUTCOMES = {"Passed", "Failed", "NotExecuted"}
_TERMINAL_DISPOSITIONS = {
    "Passed", "LockStale", "NamespaceInvalid", "FormattingFailed", "CompileFailed",
    "ClippyFailed", "TestsFailed", "DocTestsFailed", "InfrastructureUnavailable",
    "Cancelled", "OutcomeUnknown",
}
_SOURCE_FAILURES = {
    "LockStale", "NamespaceInvalid", "FormattingFailed", "CompileFailed",
    "ClippyFailed", "TestsFailed", "DocTestsFailed",
}


def _require_exact_keys(value: dict[str, Any], required: set[str], optional: set[str], *, where: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise train.TrainManifestError(f"{where}: missing fields: {', '.join(missing)}")
    if unknown:
        raise train.TrainManifestError(f"{where}: unknown fields: {', '.join(unknown)}")


def _text(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise train.TrainManifestError(f"{where}: expected canonical non-empty string")
    if unicodedata.normalize("NFC", value) != value:
        raise train.TrainManifestError(f"{where}: must use Unicode NFC normalization")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise train.TrainManifestError(f"{where}: control characters are not canonical")
    if len(value.encode("utf-8")) > train.MAX_TEXT_BYTES:
        raise train.TrainManifestError(f"{where}: exceeds {train.MAX_TEXT_BYTES} UTF-8 bytes")
    return value


def _id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _sha(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected 40 lowercase hex Git SHA")
    return value


def _canonical_payload(value: dict[str, Any], id_field: str) -> bytes:
    payload = {key: item for key, item in value.items() if key != id_field}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _record_id(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(RECORD_DOMAIN + _canonical_payload(value, "attempt_id")).hexdigest()


def _set_id(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(SET_DOMAIN + _canonical_payload(value, "set_id")).hexdigest()


def normalize_record(raw: Any, *, require_id: bool = False) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise train.TrainManifestError("attempt: expected object")
    _require_exact_keys(
        raw,
        {
            "schema", "subject_id", "qualification_profile", "provider", "provider_attempt_ref",
            "requested_subject_sha", "actual_checkout", "theorem_results", "terminal_disposition",
            "evidence_refs", "non_claims",
        },
        {"attempt_id"},
        where="attempt",
    )
    if raw["schema"] != RECORD_SCHEMA:
        raise train.TrainManifestError(f"attempt.schema: expected {RECORD_SCHEMA!r}")
    if require_id and "attempt_id" not in raw:
        raise train.TrainManifestError("attempt.attempt_id: required but absent")

    checkout = raw["actual_checkout"]
    if not isinstance(checkout, dict):
        raise train.TrainManifestError("attempt.actual_checkout: expected object")
    _require_exact_keys(checkout, {"sha", "kind"}, set(), where="attempt.actual_checkout")
    if checkout["kind"] not in _CHECKOUT_KINDS:
        raise train.TrainManifestError("attempt.actual_checkout.kind: unsupported kind")

    theorem_results = raw["theorem_results"]
    if not isinstance(theorem_results, list) or not theorem_results:
        raise train.TrainManifestError("attempt.theorem_results: expected non-empty array")
    normalized_results: list[dict[str, str]] = []
    names: list[str] = []
    for index, result in enumerate(theorem_results):
        if not isinstance(result, dict):
            raise train.TrainManifestError(f"attempt.theorem_results[{index}]: expected object")
        _require_exact_keys(result, {"name", "outcome"}, set(), where=f"attempt.theorem_results[{index}]")
        name = _text(result["name"], where=f"attempt.theorem_results[{index}].name")
        outcome = result["outcome"]
        if outcome not in _THEOREM_OUTCOMES:
            raise train.TrainManifestError(f"attempt.theorem_results[{index}].outcome: unsupported outcome")
        names.append(name)
        normalized_results.append({"name": name, "outcome": outcome})
    if names != sorted(set(names)):
        raise train.TrainManifestError("attempt.theorem_results: theorem names must be sorted and unique")

    disposition = raw["terminal_disposition"]
    if disposition not in _TERMINAL_DISPOSITIONS:
        raise train.TrainManifestError("attempt.terminal_disposition: unsupported disposition")

    evidence_refs = train._require_sorted_unique_strings(raw["evidence_refs"], where="attempt.evidence_refs")
    non_claims = train._require_sorted_unique_strings(raw["non_claims"], where="attempt.non_claims")
    if not non_claims:
        raise train.TrainManifestError("attempt.non_claims: must contain at least one explicit non-claim")

    normalized: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "subject_id": _id(raw["subject_id"], where="attempt.subject_id"),
        "qualification_profile": _text(raw["qualification_profile"], where="attempt.qualification_profile"),
        "provider": _text(raw["provider"], where="attempt.provider"),
        "provider_attempt_ref": _text(raw["provider_attempt_ref"], where="attempt.provider_attempt_ref"),
        "requested_subject_sha": _sha(raw["requested_subject_sha"], where="attempt.requested_subject_sha"),
        "actual_checkout": {
            "sha": _sha(checkout["sha"], where="attempt.actual_checkout.sha"),
            "kind": checkout["kind"],
        },
        "theorem_results": normalized_results,
        "terminal_disposition": disposition,
        "evidence_refs": evidence_refs,
        "non_claims": non_claims,
    }

    failures = [result["name"] for result in normalized_results if result["outcome"] == "Failed"]
    if disposition == "Passed":
        if any(result["outcome"] != "Passed" for result in normalized_results):
            raise train.TrainManifestError("attempt: Passed requires every theorem outcome to be Passed")
    elif disposition in _SOURCE_FAILURES:
        if len(failures) != 1:
            raise train.TrainManifestError("attempt: source-failure disposition requires exactly one Failed theorem")
    elif failures:
        raise train.TrainManifestError("attempt: infrastructure/cancelled/unknown disposition cannot carry Failed theorem")

    attempt_id = _record_id(normalized)
    normalized["attempt_id"] = attempt_id
    if "attempt_id" in raw:
        declared = _id(raw["attempt_id"], where="attempt.attempt_id")
        if declared != attempt_id:
            raise train.TrainManifestError(f"attempt.attempt_id: expected {attempt_id}, got {declared}")
    return normalized


def build_attempt_set(records: list[Any]) -> dict[str, Any]:
    if not records:
        raise train.TrainManifestError("attempt set: expected at least one terminal record")
    normalized = [normalize_record(record, require_id=True) for record in records]
    subject_id = normalized[0]["subject_id"]
    profile = normalized[0]["qualification_profile"]
    if any(record["subject_id"] != subject_id for record in normalized):
        raise train.TrainManifestError("attempt set: mixed qualification subjects")
    if any(record["qualification_profile"] != profile for record in normalized):
        raise train.TrainManifestError("attempt set: mixed qualification profiles")

    attempt_ids = [record["attempt_id"] for record in normalized]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise train.TrainManifestError("attempt set: duplicate attempt identity")
    normalized.sort(key=lambda record: record["attempt_id"])

    result: dict[str, Any] = {
        "schema": SET_SCHEMA,
        "subject_id": subject_id,
        "qualification_profile": profile,
        "attempts": normalized,
        "non_claims": [
            "does not establish chronological ordering",
            "does not establish hosted qualification beyond contained terminal records",
            "does not link source-changing repairs into the same qualification subject",
        ],
    }
    result["set_id"] = _set_id(result)
    return result


def _load(path: Path) -> Any:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(f"{path}: exceeds {train.MAX_MANIFEST_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=train._object_without_duplicate_keys)
    except train.TrainManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+", type=Path, help="terminal qualification-attempt JSON records")
    parser.add_argument("--set", action="store_true", help="build a per-subject immutable attempt set")
    parser.add_argument("--print-normalized", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = [_load(path) for path in args.records]
        if args.set:
            value = build_attempt_set(raw)
            identity = value["set_id"]
        elif len(raw) == 1:
            value = normalize_record(raw[0], require_id=True)
            identity = value["attempt_id"]
        else:
            raise train.TrainManifestError("multiple records require --set")
    except train.TrainManifestError as error:
        print(f"qualification attempt evidence invalid: {error}", file=sys.stderr)
        return 2
    if args.print_normalized:
        print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        print(identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
