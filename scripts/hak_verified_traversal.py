#!/usr/bin/env python3
"""HAK-018 deterministic verification of retained collection traversal transcripts.

This verifies structure and deterministic transforms over retained evidence only.
It does not authenticate a provider, resolve external evidence refs, prove temporal
snapshot stability, establish population-qualified selector coverage, or grant authority.
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hak_canonical_json as canonical
import hak_collection_completeness as completeness

SCHEMA_VERSION = "hak.collection-traversal-verification.v1"
DIGEST_DOMAIN = SCHEMA_VERSION
DEDUP_DOMAIN = "hak.dedup-execution.v1"
PROFILE_ID = completeness.PROFILE_ID
PROFILE_ARTIFACT_REF = completeness.PROFILE_ARTIFACT_REF
PROFILE_RAW_SHA256 = completeness.PROFILE_RAW_SHA256
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

VERIFICATION_SCOPE = "RetainedTranscriptOnly"
VERIFICATION_STATUS = "VerifiedAgainstRetainedTranscript"
PROVIDER_AUTHENTICATION = "NotEstablished"
TEMPORAL_VERIFICATION = "NotEstablished"
CONTINUATION_SOURCE_VERIFICATION = "NotEstablished"
RAW_RESPONSE_CONTENT_VERIFICATION = "NotEstablished"
ENTITY_PROJECTION_VERIFICATION = "NotEstablished"

PAGINATION_MODELS = {"LinkHeader", "PageNumber", "Cursor", "NonPaginated"}
TEMPORAL_DECLARATIONS = {
    "BestEffortLiveTraversal",
    "DeclaredProviderSnapshotBound",
    "DeclaredQueryVersionBound",
    "DeclaredStableByProviderContract",
    "Unknown",
}
DEDUP_IDENTITY_REF = "hak.entity-id.literal-string.v1"
DEDUP_RULE_REF = "hak.dedup.stable-first-occurrence.v1"


class TraversalVerificationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraversalVerificationError(message)


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def _digest(value: Any, field: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
             f"{field} must be sha256:<64 lowercase hex>")
    return value


def _positive_int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{field} must be a positive integer")
    _require(value <= canonical.MAX_SAFE_INTEGER,
             f"{field} exceeds hak.canonical-json.v1 safe integer range")
    return value


def _parse_time(value: Any, field: str) -> datetime:
    text = _text(value, field)
    _require(text.endswith("Z"), f"{field} must use UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise TraversalVerificationError(f"{field} must be RFC3339-like UTC timestamp") from exc
    _require(parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
             f"{field} must be UTC")
    return parsed


def _profile() -> dict[str, str]:
    return {
        "profile_id": PROFILE_ID,
        "artifact_ref": PROFILE_ARTIFACT_REF,
        "raw_sha256": PROFILE_RAW_SHA256,
    }


def _dedup_entity_ids(entity_ids: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    unique: list[str] = []
    duplicates = 0
    for entity_id in entity_ids:
        if entity_id in seen:
            duplicates += 1
        else:
            seen.add(entity_id)
            unique.append(entity_id)
    return unique, duplicates


def _compute_dedup_execution(flat_entities: list[str]) -> dict[str, Any]:
    unique, duplicates = _dedup_entity_ids(flat_entities)
    receipt: dict[str, Any] = {
        "schema_version": "hak.dedup-execution.v1",
        "identity_ref": DEDUP_IDENTITY_REF,
        "rule_ref": DEDUP_RULE_REF,
        "input_count": len(flat_entities),
        "unique_count": len(unique),
        "duplicate_count": duplicates,
        "retained_unique_ids": unique,
    }
    receipt["receipt_digest"] = canonical.hak_sha256(DEDUP_DOMAIN, receipt)
    return receipt


def compute_transcript_digest(transcript: dict[str, Any]) -> str:
    return canonical.hak_sha256("hak.collection-traversal-transcript.v1", transcript)


def compute_traversal_digest(record: dict[str, Any]) -> str:
    payload = {k: v for k, v in record.items() if k != "traversal_digest"}
    return canonical.hak_sha256(DIGEST_DOMAIN, payload)


def _validate_transcript(transcript: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    _require(isinstance(transcript, dict), "transcript must be object")
    provider = _text(transcript.get("provider"), "transcript.provider")
    resource_kind = _text(transcript.get("resource_kind"), "transcript.resource_kind")
    collection_ref = _text(transcript.get("collection_ref"), "transcript.collection_ref")
    query_scope_ref = _text(transcript.get("query_scope_ref"), "transcript.query_scope_ref")
    scope_ref = _text(transcript.get("scope_ref"), "transcript.scope_ref")

    pagination_model = transcript.get("pagination_model")
    _require(pagination_model in PAGINATION_MODELS,
             f"transcript.pagination_model must be one of {sorted(PAGINATION_MODELS)}")
    pagination_contract_ref = _text(
        transcript.get("pagination_contract_ref"), "transcript.pagination_contract_ref"
    )

    temporal = transcript.get("temporal")
    _require(isinstance(temporal, dict), "transcript.temporal must be object")
    temporal_decl = temporal.get("declared_semantics")
    _require(temporal_decl in TEMPORAL_DECLARATIONS,
             f"transcript.temporal.declared_semantics must be one of {sorted(TEMPORAL_DECLARATIONS)}")
    temporal_refs = temporal.get("evidence_refs")
    _require(isinstance(temporal_refs, list)
             and all(isinstance(v, str) and v.strip() for v in temporal_refs),
             "transcript.temporal.evidence_refs must be array of non-empty strings")
    _require(len(temporal_refs) == len(set(temporal_refs)),
             "transcript.temporal.evidence_refs must be unique")
    if temporal_decl in {"BestEffortLiveTraversal", "Unknown"}:
        _require(not temporal_refs,
                 "BestEffortLiveTraversal/Unknown temporal declarations must not claim positive evidence refs")
    else:
        _require(bool(temporal_refs),
                 "stronger temporal declarations require evidence refs but remain unverified in HAK-018 v1")

    pages = transcript.get("pages")
    _require(isinstance(pages, list) and pages, "transcript.pages must be non-empty array")
    if pagination_model == "NonPaginated":
        _require(len(pages) == 1, "NonPaginated transcript must contain exactly one page")

    page_ids: set[str] = set()
    request_refs: set[str] = set()
    response_refs: set[str] = set()
    continuation_outputs: set[str] = set()
    flat_entities: list[str] = []
    previous_out: str | None = None
    previous_end: datetime | None = None

    for idx, page in enumerate(pages, start=1):
        _require(isinstance(page, dict), f"transcript.pages[{idx-1}] must be object")
        ordinal = _positive_int(page.get("ordinal"), f"transcript.pages[{idx-1}].ordinal")
        _require(ordinal == idx, "page ordinals must be contiguous starting at 1")

        page_id = _text(page.get("page_id"), f"transcript.pages[{idx-1}].page_id")
        _require(page_id not in page_ids, f"duplicate page_id: {page_id}")
        page_ids.add(page_id)

        request_ref = _text(page.get("request_ref"), f"transcript.pages[{idx-1}].request_ref")
        _require(request_ref not in request_refs, f"duplicate request_ref: {request_ref}")
        request_refs.add(request_ref)

        response_ref = _text(page.get("response_ref"), f"transcript.pages[{idx-1}].response_ref")
        _require(response_ref not in response_refs, f"duplicate response_ref: {response_ref}")
        response_refs.add(response_ref)

        _digest(page.get("raw_response_digest"), f"transcript.pages[{idx-1}].raw_response_digest")

        continuation_in = page.get("continuation_in")
        continuation_out = page.get("continuation_out")
        if continuation_in is not None:
            _text(continuation_in, f"transcript.pages[{idx-1}].continuation_in")
        if continuation_out is not None:
            _text(continuation_out, f"transcript.pages[{idx-1}].continuation_out")

        if idx == 1:
            _require(continuation_in is None, "first page continuation_in must be null")
        else:
            _require(continuation_in == previous_out,
                     "page continuation_in must exactly equal prior page continuation_out")
            _require(continuation_in is not None,
                     "non-first page requires prior continuation token")

        if idx < len(pages):
            _require(continuation_out is not None,
                     "non-final page must expose continuation_out")
        else:
            _require(continuation_out is None,
                     "final page must not retain continuation_out in a terminal transcript")

        if continuation_out is not None:
            _require(continuation_out not in continuation_outputs,
                     "continuation_out token reuse/cycle detected")
            continuation_outputs.add(continuation_out)
        previous_out = continuation_out

        observed_start = _parse_time(
            page.get("observed_after_or_at"),
            f"transcript.pages[{idx-1}].observed_after_or_at",
        )
        observed_end = _parse_time(
            page.get("observed_before_or_at"),
            f"transcript.pages[{idx-1}].observed_before_or_at",
        )
        _require(observed_start <= observed_end,
                 "page observation window must not be inverted")
        if previous_end is not None:
            _require(observed_start >= previous_end,
                     "later page observation cannot begin before prior response bound completed")
        previous_end = observed_end

        entity_ids = page.get("entity_ids")
        _require(isinstance(entity_ids, list)
                 and all(isinstance(v, str) and v for v in entity_ids),
                 f"transcript.pages[{idx-1}].entity_ids must be array of non-empty strings")
        flat_entities.extend(entity_ids)

    if pagination_model == "NonPaginated":
        only = pages[0]
        _require(only.get("continuation_in") is None and only.get("continuation_out") is None,
                 "NonPaginated transcript cannot carry continuation tokens")

    metadata = {
        "provider": provider,
        "resource_kind": resource_kind,
        "collection_ref": collection_ref,
        "query_scope_ref": query_scope_ref,
        "scope_ref": scope_ref,
        "pagination_model": pagination_model,
        "pagination_contract_ref": pagination_contract_ref,
        "pages_verified": len(pages),
        "temporal_declared_semantics": temporal_decl,
    }
    return flat_entities, metadata


def _expected_hak017_binding(hak017_record: dict[str, Any]) -> dict[str, Any]:
    completeness.validate_collection_record(hak017_record)
    return {
        "collection_digest": hak017_record["collection_digest"],
        "provider": hak017_record["source_collection"]["provider"],
        "resource_kind": hak017_record["source_collection"]["resource_kind"],
        "collection_ref": hak017_record["source_collection"]["collection_ref"],
        "query_scope_ref": hak017_record["source_collection"]["query_scope_ref"],
        "scope_ref": hak017_record["scope"]["scope_ref"],
        "pagination_model": hak017_record["pagination"]["model"],
        "pagination_contract_ref": hak017_record["pagination"]["contract_ref"],
    }


def _validate_join(metadata: dict[str, Any], dedup: dict[str, Any],
                   hak017_record: dict[str, Any]) -> dict[str, Any]:
    completeness.validate_collection_record(hak017_record)
    _require(hak017_record.get("collection_status") == "DeclaredComplete",
             "HAK-018 requires an exact HAK-017 DeclaredComplete witness")
    expected = _expected_hak017_binding(hak017_record)

    for field in (
        "provider", "resource_kind", "collection_ref", "query_scope_ref",
        "scope_ref", "pagination_model", "pagination_contract_ref",
    ):
        _require(metadata[field] == expected[field],
                 f"transcript {field} must match exact HAK-017 declaration")

    counts = hak017_record["counts"]
    _require(counts["deduplication"]["identity_ref"] == DEDUP_IDENTITY_REF,
             "HAK-017 dedup identity_ref is unsupported by HAK-018 v1")
    _require(counts["deduplication"]["rule_ref"] == DEDUP_RULE_REF,
             "HAK-017 dedup rule_ref is unsupported by HAK-018 v1")
    _require(counts["retained_raw_count"] == dedup["input_count"],
             "HAK-017 retained_raw_count differs from transcript replay")
    _require(counts["retained_unique_count"] == dedup["unique_count"],
             "HAK-017 retained_unique_count differs from dedup replay")
    _require(counts["duplicate_count"] == dedup["duplicate_count"],
             "HAK-017 duplicate_count differs from dedup replay")
    _require(hak017_record["pagination"]["pages_observed"] == metadata["pages_verified"],
             "HAK-017 pages_observed differs from transcript replay")

    return expected


def make_verification_record(transcript: dict[str, Any],
                             hak017_record: dict[str, Any]) -> dict[str, Any]:
    flat_entities, metadata = _validate_transcript(transcript)
    dedup = _compute_dedup_execution(flat_entities)
    binding = _validate_join(metadata, dedup, hak017_record)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "canonicalization_profile": _profile(),
        "digest_domain": DIGEST_DOMAIN,
        "verification_scope": VERIFICATION_SCOPE,
        "verification_status": VERIFICATION_STATUS,
        "provider_authentication": PROVIDER_AUTHENTICATION,
        "temporal_snapshot_verification": TEMPORAL_VERIFICATION,
        "continuation_source_verification": CONTINUATION_SOURCE_VERIFICATION,
        "raw_response_content_verification": RAW_RESPONSE_CONTENT_VERIFICATION,
        "entity_projection_verification": ENTITY_PROJECTION_VERIFICATION,
        "transcript_digest": compute_transcript_digest(transcript),
        "transcript": transcript,
        "hak017_binding": binding,
        "dedup_execution": dedup,
        "pages_verified": metadata["pages_verified"],
    }
    record["traversal_digest"] = compute_traversal_digest(record)
    validate_verification_record(record, hak017_record=hak017_record)
    return record


def validate_verification_record(record: dict[str, Any], *,
                                 hak017_record: dict[str, Any] | None = None) -> None:
    _require(isinstance(record, dict), "verification record must be object")
    _require(record.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(record.get("canonicalization_profile") == _profile(),
             "canonicalization_profile must bind exact inherited HAK-015 profile")
    _require(record.get("digest_domain") == DIGEST_DOMAIN, "digest_domain mismatch")
    _require(record.get("verification_scope") == VERIFICATION_SCOPE,
             "verification_scope must be RetainedTranscriptOnly")
    _require(record.get("verification_status") == VERIFICATION_STATUS,
             "verification_status must be VerifiedAgainstRetainedTranscript")
    _require(record.get("provider_authentication") == PROVIDER_AUTHENTICATION,
             "provider authentication is not established by retained-transcript replay")
    _require(record.get("temporal_snapshot_verification") == TEMPORAL_VERIFICATION,
             "temporal snapshot verification is not established by retained-transcript replay")
    _require(record.get("continuation_source_verification") == CONTINUATION_SOURCE_VERIFICATION,
             "continuation source verification is not established by retained-transcript replay")
    _require(record.get("raw_response_content_verification") == RAW_RESPONSE_CONTENT_VERIFICATION,
             "raw response content verification is not established by retained-transcript replay")
    _require(record.get("entity_projection_verification") == ENTITY_PROJECTION_VERIFICATION,
             "entity projection verification is not established by retained-transcript replay")

    transcript = record.get("transcript")
    _require(isinstance(transcript, dict), "transcript must be object")
    _require(record.get("transcript_digest") == compute_transcript_digest(transcript),
             "transcript_digest mismatch")
    flat_entities, metadata = _validate_transcript(transcript)
    expected_dedup = _compute_dedup_execution(flat_entities)
    _require(record.get("dedup_execution") == expected_dedup,
             "dedup_execution does not equal deterministic replay")
    _require(record.get("pages_verified") == metadata["pages_verified"],
             "pages_verified does not match transcript")

    binding = record.get("hak017_binding")
    _require(isinstance(binding, dict), "hak017_binding must be object")
    _require(hak017_record is not None,
             "exact HAK-017 declaration witness is required for HAK-018 verification")
    expected_binding = _validate_join(metadata, expected_dedup, hak017_record)
    _require(binding == expected_binding,
             "hak017_binding does not match exact loaded HAK-017 declaration")

    digest = record.get("traversal_digest")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
             "traversal_digest must be sha256:<64 lowercase hex>")
    _require(digest == compute_traversal_digest(record), "traversal_digest mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a HAK-018 retained traversal verification record"
    )
    parser.add_argument("record", type=Path)
    parser.add_argument("--hak017", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        record_value = canonical.parse_strict_json(args.record.read_bytes())
        if not isinstance(record_value, dict):
            raise TraversalVerificationError("record root must be object")
        hak017_value = None
        if args.hak017 is not None:
            hak017_value = canonical.parse_strict_json(args.hak017.read_bytes())
            if not isinstance(hak017_value, dict):
                raise TraversalVerificationError("HAK-017 root must be object")
        validate_verification_record(record_value, hak017_record=hak017_value)
        print(f"OK   {args.record} (HAK-018 retained traversal verification)")
        return 0
    except (OSError, TraversalVerificationError, completeness.CollectionCompletenessError,
            canonical.CanonicalJsonError) as exc:
        print(f"FAIL {args.record}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
