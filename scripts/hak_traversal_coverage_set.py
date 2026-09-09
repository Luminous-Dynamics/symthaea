#!/usr/bin/env python3
"""HAK-018b coverage-set conservation over a verified retained traversal.

This accounts for every retained traversal page exactly once and derives selector
coverage across those retained pages. It does not authenticate provider content,
replay HAK-014 against raw response bytes, verify normalization-policy artifact
content, establish temporal snapshot consistency, or establish provider-population-
qualified coverage.
"""
from __future__ import annotations

import re
from typing import Any

import hak_canonical_json as canonical
import hak_collection_completeness as hak17
import hak_normalization_execute as hak14
import hak_selector_coverage as hak16
import hak_selector_coverage_profile_binding as hak16_profile
import hak_verified_traversal as hak18a

SCHEMA_VERSION = "hak.traversal-coverage-set.v1"
DIGEST_DOMAIN = SCHEMA_VERSION
COVERAGE_SCOPE = "VerifiedRetainedTraversalPagesOnly"
POPULATION_QUALIFICATION = "NotEstablished"
NORMALIZATION_INPUT_REPLAY = "NotEstablished"
NORMALIZATION_POLICY_CONTENT_VERIFICATION = "NotEstablished"
DISPOSITIONS = {"Assessed", "Unassessed"}
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_REF_RE = re.compile(r"^git:[^@]+@[0-9a-f]{40}:.+$")


class TraversalCoverageSetError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TraversalCoverageSetError(message)


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def _count(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             f"{field} must be a non-negative integer")
    _require(value <= canonical.MAX_SAFE_INTEGER,
             f"{field} exceeds hak.canonical-json.v1 safe integer range")
    return value


def _profile() -> dict[str, str]:
    return {
        "profile_id": hak17.PROFILE_ID,
        "artifact_ref": hak17.PROFILE_ARTIFACT_REF,
        "raw_sha256": hak17.PROFILE_RAW_SHA256,
    }


def compute_coverage_set_digest(record: dict[str, Any]) -> str:
    payload = {k: v for k, v in record.items() if k != "coverage_set_digest"}
    return canonical.hak_sha256(DIGEST_DOMAIN, payload)


def _result_for_selector(coverage: dict[str, Any], selector_path: str) -> dict[str, Any]:
    results = coverage.get("optional_results")
    _require(isinstance(results, list), "HAK-016 optional_results must be array")
    matches = [item for item in results if isinstance(item, dict) and item.get("path") == selector_path]
    _require(len(matches) == 1,
             "each assessed page must contain exactly one HAK-016 result for selector_path")
    return matches[0]


def _validate_assessed_witness(
    witness: dict[str, Any],
    page: dict[str, Any],
    selector_path: str,
    expected_resource_kind: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    _require(set(witness) == {"page_id", "disposition", "receipt", "coverage", "profile_binding"},
             "Assessed witness fields must match HAK-018b v1 exactly")
    receipt = witness.get("receipt")
    coverage = witness.get("coverage")
    profile_binding = witness.get("profile_binding")
    _require(isinstance(receipt, dict), "Assessed witness requires receipt")
    _require(isinstance(coverage, dict), "Assessed witness requires coverage")
    _require(isinstance(profile_binding, dict), "Assessed witness requires profile_binding")

    try:
        hak14.validate_receipt(receipt)
        hak16.validate_coverage_record(coverage, receipt)
        hak16_profile.validate_profile_binding(profile_binding, coverage)
    except Exception as exc:
        raise TraversalCoverageSetError(f"assessed witness upstream validation failed: {exc}") from exc

    raw = receipt.get("raw_source")
    _require(isinstance(raw, dict), "HAK-014 receipt raw_source missing")
    _require(raw.get("source_ref") == page.get("response_ref"),
             "HAK-014 raw source_ref must equal exact traversal response_ref")
    _require(raw.get("raw_response_digest") == page.get("raw_response_digest"),
             "HAK-014 raw response digest must equal exact traversal page digest")
    _require(receipt.get("resource_kind") == coverage.get("resource_kind"),
             "HAK-014 and HAK-016 resource_kind mismatch")
    _require(receipt.get("resource_kind") == expected_resource_kind,
             "HAK-014/016 resource_kind must equal exact traversal resource_kind")

    result = _result_for_selector(coverage, selector_path)
    coverage_policy = coverage.get("policy")
    receipt_policy = receipt.get("policy")
    _require(isinstance(coverage_policy, dict), "HAK-016 coverage policy missing")
    _require(isinstance(receipt_policy, dict), "HAK-014 receipt policy missing")
    policy_id = _text(coverage_policy.get("policy_id"), "coverage.policy.policy_id")
    policy_digest = coverage_policy.get("policy_digest")
    _require(isinstance(policy_digest, str) and SHA256_RE.fullmatch(policy_digest) is not None,
             "coverage.policy.policy_digest invalid")
    _require(receipt_policy.get("policy_id") == policy_id,
             "HAK-014/016 policy_id mismatch")
    _require(receipt_policy.get("policy_digest") == policy_digest,
             "HAK-014/016 policy_digest mismatch")
    artifact_ref = receipt_policy.get("artifact_ref")
    _require(isinstance(artifact_ref, str) and GIT_REF_RE.fullmatch(artifact_ref) is not None,
             "HAK-014 policy artifact_ref must be exact git ref")
    return result, {
        "policy_id": policy_id,
        "policy_digest": policy_digest,
        "artifact_ref": artifact_ref,
    }


def _derive_aggregate(page_entries: list[dict[str, Any]]) -> dict[str, Any]:
    assessed_pages = sum(item["disposition"] == "Assessed" for item in page_entries)
    unassessed_pages = sum(item["disposition"] == "Unassessed" for item in page_entries)
    failed_pages = sum(
        item["disposition"] == "Assessed" and item["coverage_state"] == "Failed"
        for item in page_entries
    )

    applicable = matches = missing = 0
    for item in page_entries:
        if item["disposition"] != "Assessed" or item["coverage_state"] == "Failed":
            continue
        applicable += _count(item["applicable"], "page.applicable")
        matches += _count(item["matches"], "page.matches")
        missing += _count(item["missing"], "page.missing")

    for value, name in ((applicable, "applicable"), (matches, "matches"), (missing, "missing")):
        _require(value <= canonical.MAX_SAFE_INTEGER,
                 f"aggregate {name} count exceeds canonical safe integer range")
    _require(applicable == matches + missing, "aggregate coverage conservation violated")

    if unassessed_pages:
        state = "Indeterminate"
    elif failed_pages:
        state = "Failed"
    elif applicable == 0:
        state = "NotApplicable"
    elif matches == applicable:
        state = "Present"
    elif matches == 0:
        state = "Absent"
    else:
        state = "PartiallyPresent"

    return {
        "coverage_state": state,
        "assessed_pages": assessed_pages,
        "unassessed_pages": unassessed_pages,
        "failed_pages": failed_pages,
        "observed_counts": {"applicable": applicable, "matches": matches, "missing": missing},
    }


def derive_coverage_set(
    *,
    selector_path: str,
    traversal_record: dict[str, Any],
    hak017_record: dict[str, Any],
    page_witnesses: list[dict[str, Any]],
) -> dict[str, Any]:
    _text(selector_path, "selector_path")
    try:
        hak18a.validate_verification_record(traversal_record, hak017_record=hak017_record)
    except Exception as exc:
        raise TraversalCoverageSetError(f"HAK-018a traversal validation failed: {exc}") from exc

    transcript = traversal_record.get("transcript")
    _require(isinstance(transcript, dict), "traversal transcript missing")
    resource_kind = _text(transcript.get("resource_kind"), "traversal resource_kind")
    pages = transcript.get("pages")
    _require(isinstance(pages, list) and pages, "traversal pages must be non-empty")
    _require(isinstance(page_witnesses, list), "page_witnesses must be array")

    witness_map: dict[str, dict[str, Any]] = {}
    for idx, witness in enumerate(page_witnesses):
        _require(isinstance(witness, dict), f"page_witnesses[{idx}] must be object")
        page_id = _text(witness.get("page_id"), f"page_witnesses[{idx}].page_id")
        _require(page_id not in witness_map, f"duplicate page witness: {page_id}")
        disposition = witness.get("disposition")
        _require(disposition in DISPOSITIONS,
                 f"page_witnesses[{idx}].disposition must be one of {sorted(DISPOSITIONS)}")
        if disposition == "Unassessed":
            _require(set(witness) == {"page_id", "disposition", "reason"},
                     "Unassessed witness fields must match HAK-018b v1 exactly")
            _text(witness.get("reason"), f"page_witnesses[{idx}].reason")
        witness_map[page_id] = witness

    expected_page_ids = [page["page_id"] for page in pages]
    _require(set(witness_map) == set(expected_page_ids),
             "page witness set must exactly equal traversal page set")

    page_entries: list[dict[str, Any]] = []
    common_policy: dict[str, str] | None = None
    for page in pages:
        page_id = page["page_id"]
        witness = witness_map[page_id]
        base = {
            "ordinal": page["ordinal"],
            "page_id": page_id,
            "response_ref": page["response_ref"],
            "raw_response_digest": page["raw_response_digest"],
        }
        if witness["disposition"] == "Unassessed":
            page_entries.append({**base, "disposition": "Unassessed", "reason": witness["reason"]})
            continue

        result, policy = _validate_assessed_witness(
            witness, page, selector_path, expected_resource_kind=resource_kind
        )
        if common_policy is None:
            common_policy = policy
        else:
            _require(policy == common_policy,
                     "all assessed pages must use the exact same recorded HAK-014/016 policy identity")

        receipt = witness["receipt"]
        coverage = witness["coverage"]
        profile_binding = witness["profile_binding"]
        page_entries.append({
            **base,
            "disposition": "Assessed",
            "receipt_digest": receipt["receipt_digest"],
            "normalization_execution_status": receipt["execution_status"],
            "coverage_digest": coverage["coverage_digest"],
            "profile_binding_digest": profile_binding["binding_digest"],
            "source_status": result["source_status"],
            "coverage_state": result["coverage_state"],
            "applicable": result["applicable"],
            "matches": result["matches"],
            "missing": result["missing"],
        })

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "canonicalization_profile": _profile(),
        "digest_domain": DIGEST_DOMAIN,
        "coverage_scope": COVERAGE_SCOPE,
        "population_qualification": POPULATION_QUALIFICATION,
        "normalization_input_replay": NORMALIZATION_INPUT_REPLAY,
        "normalization_policy_content_verification": NORMALIZATION_POLICY_CONTENT_VERIFICATION,
        "selector": {"path": selector_path, "coverage_policy": common_policy},
        "traversal_binding": {
            "traversal_digest": traversal_record["traversal_digest"],
            "hak017_collection_digest": hak017_record["collection_digest"],
            "pages_verified": traversal_record["pages_verified"],
        },
        "source_assurance": {
            "provider_authentication": traversal_record["provider_authentication"],
            "raw_response_content_verification": traversal_record["raw_response_content_verification"],
            "entity_projection_verification": traversal_record["entity_projection_verification"],
            "continuation_source_verification": traversal_record["continuation_source_verification"],
            "temporal_snapshot_verification": traversal_record["temporal_snapshot_verification"],
        },
        "pages": page_entries,
        "aggregate": _derive_aggregate(page_entries),
    }
    record["coverage_set_digest"] = compute_coverage_set_digest(record)
    return record


def validate_coverage_set(
    record: dict[str, Any],
    *,
    selector_path: str,
    traversal_record: dict[str, Any],
    hak017_record: dict[str, Any],
    page_witnesses: list[dict[str, Any]],
) -> None:
    _require(isinstance(record, dict), "coverage-set record must be object")
    _require(record.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(record.get("canonicalization_profile") == _profile(),
             "canonicalization_profile must bind exact inherited HAK-015 profile")
    _require(record.get("digest_domain") == DIGEST_DOMAIN, "digest_domain mismatch")
    _require(record.get("coverage_scope") == COVERAGE_SCOPE,
             "coverage_scope must remain verified retained traversal pages only")
    _require(record.get("population_qualification") == POPULATION_QUALIFICATION,
             "provider population qualification is not established by HAK-018b")
    _require(record.get("normalization_input_replay") == NORMALIZATION_INPUT_REPLAY,
             "HAK-018b does not establish HAK-014 replay against raw response bytes")
    _require(record.get("normalization_policy_content_verification") == NORMALIZATION_POLICY_CONTENT_VERIFICATION,
             "HAK-018b does not establish normalization-policy artifact content verification")
    digest = record.get("coverage_set_digest")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
             "coverage_set_digest invalid")
    _require(digest == compute_coverage_set_digest(record), "coverage_set_digest mismatch")

    expected = derive_coverage_set(
        selector_path=selector_path,
        traversal_record=traversal_record,
        hak017_record=hak017_record,
        page_witnesses=page_witnesses,
    )
    _require(record == expected,
             "coverage-set record is not the deterministic derivation of exact traversal/upstream witnesses")
