#!/usr/bin/env python3
"""HAK-017 population / collection completeness declarations.

Audit/evidence tooling only. This module prevents coverage over retained contexts
from being silently promoted into coverage over an entire provider population.
It validates internally consistent declarations and evidence references; it does
not resolve provider evidence, authenticate providers, prove truth, or grant authority.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import hak_canonical_json as canonical

SCHEMA_VERSION = "hak.collection-completeness.v1"
DIGEST_DOMAIN = SCHEMA_VERSION
PROFILE_ID = "hak.canonical-json.v1"
PROFILE_ARTIFACT_REF = (
    "git:Luminous-Dynamics/symthaea@cf440ce4f813bb30a6b1948e5caac96feda10610:"
    "docs/architecture/hak/canonical-json-v1.profile.json"
)
PROFILE_RAW_SHA256 = "sha256:f76152db3cce567bb4d24ca46b5ed94d95db698daa8882a6fcdb38828d40e830"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

POPULATION_EXTENTS = {"FullProviderPopulation", "DefinedSubset", "Unknown"}
CONSTRAINT_FIELDS = ("filtered", "sampled", "time_windowed", "permission_limited")
PAGINATION_MODELS = {"LinkHeader", "PageNumber", "Cursor", "NonPaginated", "Unknown"}
COMPLETENESS_STATES = {"DeclaredComplete", "Partial", "Unknown", "Failed", "NotApplicable"}
EXHAUSTION_KINDS = {
    "NoContinuationByProviderContract",
    "CursorExhausted",
    "NonPaginatedEndpoint",
    "ProviderExplicitComplete",
    "Unknown",
}
POSITIVE_EXHAUSTION_KINDS = EXHAUSTION_KINDS - {"Unknown"}


class CollectionCompletenessError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectionCompletenessError(message)


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def _count(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             f"{field} must be a non-negative integer")
    _require(value <= canonical.MAX_SAFE_INTEGER,
             f"{field} exceeds hak.canonical-json.v1 safe integer range")
    return value


def compute_collection_digest(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "collection_digest"}
    return canonical.hak_sha256(DIGEST_DOMAIN, payload)


def _positive_exhaustion_compatible(model: str, kind: str) -> bool:
    if kind == "ProviderExplicitComplete":
        return model != "Unknown"
    if model in {"LinkHeader", "PageNumber"}:
        return kind == "NoContinuationByProviderContract"
    if model == "Cursor":
        return kind == "CursorExhausted"
    if model == "NonPaginated":
        return kind == "NonPaginatedEndpoint"
    return False


def validate_collection_record(record: dict[str, Any]) -> None:
    _require(isinstance(record, dict), "collection-completeness record must be object")
    _require(record.get("schema_version") == SCHEMA_VERSION, "schema_version invalid")

    profile = record.get("canonicalization_profile")
    _require(profile == {
        "profile_id": PROFILE_ID,
        "artifact_ref": PROFILE_ARTIFACT_REF,
        "raw_sha256": PROFILE_RAW_SHA256,
    }, "canonicalization_profile must bind the exact inherited HAK-015 profile")
    _require(record.get("digest_domain") == DIGEST_DOMAIN, "digest_domain mismatch")

    source = record.get("source_collection")
    _require(isinstance(source, dict), "source_collection must be object")
    _text(source.get("provider"), "source_collection.provider")
    _text(source.get("resource_kind"), "source_collection.resource_kind")
    _text(source.get("collection_ref"), "source_collection.collection_ref")
    _text(source.get("query_scope_ref"), "source_collection.query_scope_ref")

    scope = record.get("scope")
    _require(isinstance(scope, dict), "scope must be object")
    extent = scope.get("population_extent")
    _require(extent in POPULATION_EXTENTS,
             f"scope.population_extent must be one of {sorted(POPULATION_EXTENTS)}")
    scope_ref = _text(scope.get("scope_ref"), "scope.scope_ref")
    constraints = scope.get("constraints")
    _require(isinstance(constraints, dict), "scope.constraints must be object")
    _require(set(constraints) == set(CONSTRAINT_FIELDS),
             f"scope.constraints must contain exactly {sorted(CONSTRAINT_FIELDS)}")
    for field in CONSTRAINT_FIELDS:
        _require(isinstance(constraints[field], bool), f"scope.constraints.{field} must be boolean")
    if extent == "FullProviderPopulation":
        _require(not any(constraints.values()),
                 "FullProviderPopulation cannot carry narrowing scope constraints")

    pagination = record.get("pagination")
    _require(isinstance(pagination, dict), "pagination must be object")
    model = pagination.get("model")
    _require(model in PAGINATION_MODELS, f"pagination.model must be one of {sorted(PAGINATION_MODELS)}")
    contract_ref = pagination.get("contract_ref")
    if model == "Unknown":
        _require(contract_ref is None, "Unknown pagination must not claim contract_ref")
    else:
        _text(contract_ref, "pagination.contract_ref")
    pages_observed = _count(pagination.get("pages_observed"), "pagination.pages_observed")
    _require(pages_observed > 0, "pagination.pages_observed must be positive")
    page_size = pagination.get("page_size")
    if page_size is not None:
        _require(_count(page_size, "pagination.page_size") > 0,
                 "pagination.page_size must be positive when known")

    exhaustion = pagination.get("exhaustion")
    _require(isinstance(exhaustion, dict), "pagination.exhaustion must be object")
    exhaustion_kind = exhaustion.get("kind")
    _require(exhaustion_kind in EXHAUSTION_KINDS,
             f"pagination.exhaustion.kind must be one of {sorted(EXHAUSTION_KINDS)}")
    refs = exhaustion.get("evidence_refs")
    _require(isinstance(refs, list) and all(isinstance(x, str) and x.strip() for x in refs),
             "pagination.exhaustion.evidence_refs must be an array of non-empty strings")
    _require(len(refs) == len(set(refs)), "pagination.exhaustion.evidence_refs must be unique")
    if exhaustion_kind == "Unknown":
        _require(not refs, "Unknown exhaustion must not claim positive evidence_refs")
    else:
        _require(bool(refs), "positive exhaustion declaration requires evidence_refs")
        _require(_positive_exhaustion_compatible(model, exhaustion_kind),
                 "positive exhaustion kind is incompatible with pagination model")

    counts = record.get("counts")
    _require(isinstance(counts, dict), "counts must be object")
    raw_count = _count(counts.get("retained_raw_count"), "counts.retained_raw_count")
    unique_count = _count(counts.get("retained_unique_count"), "counts.retained_unique_count")
    duplicate_count = _count(counts.get("duplicate_count"), "counts.duplicate_count")
    _require(raw_count == unique_count + duplicate_count,
             "count conservation violated: retained_raw_count must equal retained_unique_count + duplicate_count")

    dedup = counts.get("deduplication")
    _require(isinstance(dedup, dict), "counts.deduplication must be object")
    _require(set(dedup) == {"identity_ref", "rule_ref"},
             "counts.deduplication must contain exactly identity_ref and rule_ref")
    _text(dedup.get("identity_ref"), "counts.deduplication.identity_ref")
    _text(dedup.get("rule_ref"), "counts.deduplication.rule_ref")

    provider_total = counts.get("provider_reported_total")
    total_count: int | None
    if provider_total is None:
        total_count = None
    else:
        _require(isinstance(provider_total, dict), "counts.provider_reported_total must be object or null")
        _require(set(provider_total) == {"value", "scope_ref"},
                 "provider_reported_total must contain exactly value and scope_ref")
        total_count = _count(provider_total.get("value"), "counts.provider_reported_total.value")
        total_scope_ref = _text(provider_total.get("scope_ref"), "counts.provider_reported_total.scope_ref")
        _require(total_scope_ref == scope_ref,
                 "provider_reported_total.scope_ref must match the exact declared scope_ref")
        _require(unique_count <= total_count,
                 "retained_unique_count cannot exceed provider_reported_total.value")

    status = record.get("collection_status")
    _require(status in COMPLETENESS_STATES,
             f"collection_status must be one of {sorted(COMPLETENESS_STATES)}")
    reasons = record.get("reasons")
    _require(isinstance(reasons, list) and all(isinstance(x, str) and x.strip() for x in reasons),
             "reasons must be an array of non-empty strings")

    if status == "DeclaredComplete":
        _require(extent != "Unknown", "DeclaredComplete requires a defined population extent")
        _require(model != "Unknown", "DeclaredComplete requires known pagination semantics")
        _require(exhaustion_kind in POSITIVE_EXHAUSTION_KINDS,
                 "DeclaredComplete requires a positive exhaustion declaration")
        _require(bool(refs), "DeclaredComplete requires exhaustion evidence refs")
        if total_count is not None:
            _require(unique_count == total_count,
                     "DeclaredComplete with provider total requires retained_unique_count equality")
    elif status in {"Partial", "Unknown", "Failed", "NotApplicable"}:
        _require(bool(reasons), f"{status} requires at least one reason")

    digest = record.get("collection_digest")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
             "collection_digest must be sha256:<64 lowercase hex>")
    _require(digest == compute_collection_digest(record), "collection_digest mismatch")


def provider_population_declared_complete(record: dict[str, Any]) -> bool:
    """True only for a valid declared-complete record over the unconstrained provider population."""
    validate_collection_record(record)
    return (
        record["collection_status"] == "DeclaredComplete"
        and record["scope"]["population_extent"] == "FullProviderPopulation"
        and not any(record["scope"]["constraints"].values())
    )


def make_record(*, provider: str, resource_kind: str, collection_ref: str,
                query_scope_ref: str, population_extent: str, scope_ref: str,
                filtered: bool = False, sampled: bool = False,
                time_windowed: bool = False, permission_limited: bool = False,
                pagination_model: str, pagination_contract_ref: str | None,
                pages_observed: int, page_size: int | None,
                exhaustion_kind: str, exhaustion_evidence_refs: list[str],
                retained_raw_count: int, retained_unique_count: int,
                duplicate_count: int, dedup_identity_ref: str, dedup_rule_ref: str,
                provider_reported_total_count: int | None,
                provider_total_scope_ref: str | None,
                collection_status: str, reasons: list[str]) -> dict[str, Any]:
    if provider_reported_total_count is None:
        _require(provider_total_scope_ref is None,
                 "provider_total_scope_ref must be null when provider total is unavailable")
        provider_total = None
    else:
        _require(provider_total_scope_ref is not None,
                 "provider_total_scope_ref is required when provider total is available")
        provider_total = {
            "value": provider_reported_total_count,
            "scope_ref": provider_total_scope_ref,
        }

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "canonicalization_profile": {
            "profile_id": PROFILE_ID,
            "artifact_ref": PROFILE_ARTIFACT_REF,
            "raw_sha256": PROFILE_RAW_SHA256,
        },
        "digest_domain": DIGEST_DOMAIN,
        "source_collection": {
            "provider": provider,
            "resource_kind": resource_kind,
            "collection_ref": collection_ref,
            "query_scope_ref": query_scope_ref,
        },
        "scope": {
            "population_extent": population_extent,
            "scope_ref": scope_ref,
            "constraints": {
                "filtered": filtered,
                "sampled": sampled,
                "time_windowed": time_windowed,
                "permission_limited": permission_limited,
            },
        },
        "pagination": {
            "model": pagination_model,
            "contract_ref": pagination_contract_ref,
            "pages_observed": pages_observed,
            "page_size": page_size,
            "exhaustion": {"kind": exhaustion_kind, "evidence_refs": exhaustion_evidence_refs},
        },
        "counts": {
            "retained_raw_count": retained_raw_count,
            "retained_unique_count": retained_unique_count,
            "duplicate_count": duplicate_count,
            "deduplication": {
                "identity_ref": dedup_identity_ref,
                "rule_ref": dedup_rule_ref,
            },
            "provider_reported_total": provider_total,
        },
        "collection_status": collection_status,
        "reasons": reasons,
    }
    record["collection_digest"] = compute_collection_digest(record)
    validate_collection_record(record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate HAK-017 collection completeness declarations")
    parser.add_argument("record", type=Path)
    args = parser.parse_args(argv)
    try:
        value = canonical.parse_strict_json(args.record.read_bytes())
        if not isinstance(value, dict):
            raise CollectionCompletenessError("record root must be object")
        validate_collection_record(value)
        print(f"OK   {args.record} (HAK-017 collection completeness declaration)")
        return 0
    except (OSError, CollectionCompletenessError, canonical.CanonicalJsonError) as exc:
        print(f"FAIL {args.record}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
