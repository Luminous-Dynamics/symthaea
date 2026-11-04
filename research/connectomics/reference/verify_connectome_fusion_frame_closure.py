#!/usr/bin/env python3
"""Independent, stdlib-only reference replay for the V0.1 frame-closure fixture.

This is a test oracle, not a production verifier or evidence custodian.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import itertools
import json
import sys
import unittest
from pathlib import Path
from typing import Any

EXECUTION_ROLE = "FUSION_EXECUTION_LEDGER"
CLAIM_ROLE = "CLAIM_CERTIFICATE_LEDGER"
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "connectome-fusion-frame-closure-omitted-use-v0.1.json"
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def domain_hash(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + canonical(value)).hexdigest()


def domain_hash_string(domain: str, value: str) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + value.encode("utf-8")).hexdigest()


def is_utc_timestamp(value: str) -> bool:
    try:
        if len(value) != 20 or not value.endswith("Z"):
            return False
        dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except (TypeError, ValueError):
        return False


def sorted_unique(values: list[str]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def semantic_use_material(scope_sha256: str, event: dict[str, Any]) -> dict[str, Any]:
    return {
        "scope_sha256": scope_sha256,
        "query_id": event["query_id"],
        "claim_id": event["claim_id"],
        "certificate_id": event["certificate_id"],
        "operation_id": event["operation_id"],
        "fusion_policy_sha256": event["fusion_policy_sha256"],
        "verifier_sha256": event["verifier_sha256"],
        "dependence_bundle_sha256": event["dependence_bundle_sha256"],
        "source_release_ids": sorted(event["source_release_ids"]),
        "evidence_ids": sorted(event["evidence_ids"]),
    }


def use_key(scope_sha256: str, event: dict[str, Any]) -> str:
    return domain_hash(
        "connectome-fusion-use-key-v0.1",
        semantic_use_material(scope_sha256, event),
    )


def event_semantic(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["operation_id"],
        event["claim_id"],
        event["query_id"],
        event["certificate_id"],
        event["fusion_policy_sha256"],
        event["verifier_sha256"],
        event["dependence_bundle_sha256"],
        tuple(sorted(event["source_release_ids"])),
        tuple(sorted(event["evidence_ids"])),
    )


def assertion_key_from_frame(assertion: dict[str, Any]) -> tuple[str, ...]:
    return (
        assertion["canonical_use_key"],
        assertion["claim_id"],
        assertion["query_id"],
        assertion["left_evidence_id"],
        assertion["right_evidence_id"],
        assertion["fusion_policy_sha256"],
        assertion["dependence_bundle_sha256"],
    )


def required_assertion_keys(event: dict[str, Any]) -> set[tuple[str, ...]]:
    ids = sorted(event["evidence_ids"])
    return {
        (
            event["canonical_use_key"],
            event["claim_id"],
            event["query_id"],
            left,
            right,
            event["fusion_policy_sha256"],
            event["dependence_bundle_sha256"],
        )
        for left, right in itertools.combinations(ids, 2)
    }


def refresh_event_identity(bundle: dict[str, Any], source: dict[str, Any]) -> None:
    for event in source["events"]:
        event["canonical_use_key"] = use_key(bundle["scope_sha256"], event)
    source["ordered_events_sha256"] = domain_hash(
        "connectome-fusion-events-v0.1", source["events"]
    )


def verify(bundle: dict[str, Any], raw: bytes = b"") -> dict[str, Any]:
    errors: list[str] = []
    scope = bundle["scope"]
    scope_sha256 = bundle["scope_sha256"]
    projection = bundle["fusion_policy_projection"]

    if bundle["schema_version"] != "0.1.0":
        errors.append("UNSUPPORTED_SCHEMA_VERSION")
    if not bundle["audit_id"].strip() or not scope["scope_id"].strip():
        errors.append("MISSING_STABLE_AUDIT_OR_SCOPE_ID")
    if (
        not is_utc_timestamp(scope["start_at"])
        or not is_utc_timestamp(scope["end_at"])
        or scope["start_at"] >= scope["end_at"]
    ):
        errors.append("INVALID_SCOPE_INTERVAL")
    if scope["event_time_basis"] not in {"EVENT_TIME", "INGESTION_TIME"}:
        errors.append("UNSUPPORTED_EVENT_TIME_BASIS")
    if not scope["query_ids"] or not sorted_unique(scope["query_ids"]):
        errors.append("QUERY_SCOPE_IDS_EMPTY_DUPLICATED_OR_UNSORTED")
    if not scope["release_ids"] or not sorted_unique(scope["release_ids"]):
        errors.append("RELEASE_SCOPE_IDS_EMPTY_DUPLICATED_OR_UNSORTED")
    if (
        bundle["reconciliation_policy"]["shared_control_policy"]
        != "FAIL_CLOSED_ON_DECLARED_SHARED_CONTROL"
    ):
        errors.append("UNSUPPORTED_SHARED_CONTROL_POLICY")

    if domain_hash("connectome-fusion-scope-v0.1", scope) != scope_sha256:
        errors.append("SCOPE_DIGEST_MISMATCH")
    projection_material = {
        "projection_id": projection["projection_id"],
        "rule": projection["rule"],
    }
    if (
        domain_hash("connectome-fusion-projection-v0.1", projection_material)
        != projection["projection_sha256"]
    ):
        errors.append("PROJECTION_DIGEST_MISMATCH")
    if projection["rule"] != "ALL_UNORDERED_EVIDENCE_PAIRS":
        errors.append("UNSUPPORTED_FUSION_POLICY_PROJECTION")

    required_roles = set(bundle["reconciliation_policy"]["required_source_roles"])
    if required_roles != {EXECUTION_ROLE, CLAIM_ROLE} or len(
        bundle["reconciliation_policy"]["required_source_roles"]
    ) != 2:
        errors.append("REQUIRED_SOURCE_ROLE_SET_UNSUPPORTED")

    source_checks = []
    role_counts: dict[str, int] = {}
    exports_by_role: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    custodians: set[str] = set()

    for source in bundle["source_exports"]:
        if source["source_id"] in source_ids:
            errors.append(f"DUPLICATE_SOURCE_ID:{source['source_id']}")
        source_ids.add(source["source_id"])
        if source["source_role"] not in {EXECUTION_ROLE, CLAIM_ROLE}:
            errors.append(f"UNSUPPORTED_SOURCE_ROLE:{source['source_role']}")
        role_counts[source["source_role"]] = role_counts.get(source["source_role"], 0) + 1
        if source["source_role"] in exports_by_role:
            errors.append(f"DUPLICATE_SOURCE_ROLE:{source['source_role']}")
        exports_by_role[source["source_role"]] = source

        event_ids = [event["event_id"] for event in source["events"]]
        sequences = [event["sequence"] for event in source["events"]]
        unique_use_keys = {event["canonical_use_key"] for event in source["events"]}
        contiguous = (
            len(source["events"]) == source["record_count"]
            and len(set(sequences)) == len(sequences)
            and sequences == list(
                range(source["sequence_start"], source["sequence_start"] + len(sequences))
            )
            and (
                source["sequence_end"] == source["sequence_start"]
                if not sequences
                else sequences[-1] == source["sequence_end"]
                and source["sequence_end"] - source["sequence_start"] + 1
                == source["record_count"]
            )
        )
        event_digest_valid = (
            domain_hash("connectome-fusion-events-v0.1", source["events"])
            == source["ordered_events_sha256"]
        )
        extraction_digest_valid = (
            domain_hash_string("connectome-fusion-extraction-v0.1", source["source_id"])
            == source["extraction_sha256"]
        )
        scope_match = source["scope_sha256"] == scope_sha256
        custodian_unique = source["custodian_ref"] not in custodians
        custodians.add(source["custodian_ref"])

        if not source["source_id"].strip() or not source["custodian_ref"].strip():
            errors.append(f"SOURCE_ID_OR_CUSTODIAN_MISSING:{source['source_id']}")
        if len(set(event_ids)) != len(event_ids):
            errors.append(f"DUPLICATE_EVENT_ID:{source['source_id']}")
        if source["record_count"] != len(source["events"]):
            errors.append(f"RECORD_COUNT_MISMATCH:{source['source_id']}")
        if source["unique_use_count"] != len(unique_use_keys):
            errors.append(f"UNIQUE_USE_COUNT_MISMATCH:{source['source_id']}")
        if not contiguous:
            errors.append(f"SEQUENCE_GAP_OR_ORDER_ERROR:{source['source_id']}")
        if not event_digest_valid:
            errors.append(f"EVENT_EXPORT_DIGEST_MISMATCH:{source['source_id']}")
        if not extraction_digest_valid:
            errors.append(f"EXTRACTION_DIGEST_MISMATCH:{source['source_id']}")
        if not scope_match:
            errors.append(f"SOURCE_SCOPE_DIGEST_MISMATCH:{source['source_id']}")

        for event in source["events"]:
            valid_identity = (
                bool(event["event_id"].strip())
                and bool(event["operation_id"].strip())
                and bool(event["claim_id"].strip())
                and bool(event["query_id"].strip())
                and bool(event["certificate_id"].strip())
                and len(event["evidence_ids"]) >= 2
                and bool(event["source_release_ids"])
                and is_utc_timestamp(event["event_at"])
                and is_utc_timestamp(event["ingested_at"])
                and event["ingested_at"] >= event["event_at"]
                and sorted_unique(event["source_release_ids"])
                and sorted_unique(event["evidence_ids"])
            )
            if not valid_identity:
                errors.append(
                    f"INVALID_EVENT_IDENTITY_OR_EVIDENCE_SET:{source['source_id']}:{event['event_id']}"
                )
            scoped_time = (
                event["event_at"]
                if scope["event_time_basis"] == "EVENT_TIME"
                else event["ingested_at"]
            )
            if (
                scoped_time < scope["start_at"]
                or scoped_time >= scope["end_at"]
                or event["query_id"] not in scope["query_ids"]
                or any(item not in scope["release_ids"] for item in event["source_release_ids"])
            ):
                errors.append(
                    f"EVENT_OUTSIDE_DECLARED_SCOPE:{source['source_id']}:{event['event_id']}"
                )
            if event["outcome"] not in {
                "CONTRIBUTED",
                "FAILED_BEFORE_CLAIM",
                "CANCELLED",
                "RETRY_OF_PRIOR_EVENT",
            }:
                errors.append(
                    f"UNSUPPORTED_EVENT_OUTCOME:{source['source_id']}:{event['event_id']}"
                )
            if use_key(scope_sha256, event) != event["canonical_use_key"]:
                errors.append(
                    f"CANONICAL_USE_KEY_MISMATCH:{source['source_id']}:{event['event_id']}"
                )

        source_checks.append(
            {
                "source_id": source["source_id"],
                "source_role": source["source_role"],
                "record_count_recomputed": len(source["events"]),
                "unique_use_count_recomputed": len(unique_use_keys),
                "sequence_contiguous": contiguous,
                "event_digest_valid": event_digest_valid,
                "extraction_digest_valid": extraction_digest_valid,
                "scope_match": scope_match,
                "coverage_qualified": source["coverage_disposition"]
                == "QUALIFIED_COMPLETE_WITHIN_DECLARED_SCOPE",
                "custodian_unique": custodian_unique,
                "shared_controls_empty": not source["shared_controls"],
            }
        )

    for role in (EXECUTION_ROLE, CLAIM_ROLE):
        if role_counts.get(role, 0) != 1:
            errors.append(f"REQUIRED_SOURCE_ROLE_MISSING_OR_DUPLICATED:{role}")

    frame = bundle["declared_frame"]
    if frame["assertion_count"] != len(frame["assertions"]):
        errors.append("FRAME_ASSERTION_COUNT_MISMATCH")
    if domain_hash("connectome-fusion-frame-v0.1", frame["assertions"]) != frame["frame_sha256"]:
        errors.append("FRAME_DIGEST_MISMATCH")

    frame_ids = [assertion["assertion_id"] for assertion in frame["assertions"]]
    frame_keys = [assertion_key_from_frame(a) for a in frame["assertions"]]
    if any(not item.strip() for item in frame_ids) or len(set(frame_ids)) != len(frame_ids):
        errors.append("FRAME_ASSERTION_ID_EMPTY_OR_DUPLICATED")
    if any(key[3] >= key[4] for key in frame_keys):
        errors.append("FRAME_PAIR_NOT_CANONICALLY_ORDERED")
    if len(set(frame_keys)) != len(frame_keys):
        errors.append("DUPLICATE_FRAME_ASSERTION")

    if errors:
        return {
            "disposition": "NOT_ESTIMABLE",
            "promotion_authorized": False,
            "errors": errors,
            "source_checks": source_checks,
            "source_population_match": False,
        }

    coverage_unknown = any(
        source["coverage_disposition"] != "QUALIFIED_COMPLETE_WITHIN_DECLARED_SCOPE"
        for source in bundle["source_exports"]
    )
    shared_control = (
        len(custodians) != len(bundle["source_exports"])
        or any(source["shared_controls"] for source in bundle["source_exports"])
    )

    execution_map: dict[str, tuple[Any, ...]] = {}
    claim_map: dict[str, tuple[Any, ...]] = {}
    for role, destination in ((EXECUTION_ROLE, execution_map), (CLAIM_ROLE, claim_map)):
        for event in exports_by_role[role]["events"]:
            if event["outcome"] == "CONTRIBUTED":
                destination[event["canonical_use_key"]] = event_semantic(event)
            elif role == CLAIM_ROLE:
                errors.append(f"NON_CONTRIBUTING_EVENT_IN_CLAIM_LEDGER:{event['event_id']}")

    execution_keys = set(execution_map)
    claim_keys = set(claim_map)
    execution_without_claim = sorted(execution_keys - claim_keys)
    claim_without_execution = sorted(claim_keys - execution_keys)
    source_population_match = execution_keys == claim_keys and all(
        execution_map[key] == claim_map[key] for key in execution_keys & claim_keys
    )
    if not source_population_match:
        errors.append("SOURCE_POPULATION_RECONCILIATION_MISMATCH")

    required_keys: set[tuple[str, ...]] = set()
    for event in exports_by_role[EXECUTION_ROLE]["events"]:
        if event["outcome"] == "CONTRIBUTED":
            required_keys.update(required_assertion_keys(event))
    frame_key_set = set(frame_keys)
    missing = sorted(required_keys - frame_key_set)
    orphan = sorted(frame_key_set - required_keys)

    if errors:
        disposition = "RECONCILIATION_CONFLICT"
    elif coverage_unknown or shared_control:
        disposition = "SOURCE_COVERAGE_UNKNOWN"
    elif missing:
        disposition = "FRAME_GAP_DETECTED"
    elif orphan:
        disposition = "RECONCILIATION_CONFLICT"
    elif not required_keys:
        disposition = "NOT_ESTIMABLE"
    else:
        disposition = "FRAME_CLOSED_WITHIN_DECLARED_SCOPE"

    return {
        "disposition": disposition,
        "promotion_authorized": disposition == "FRAME_CLOSED_WITHIN_DECLARED_SCOPE",
        "source_checks": source_checks,
        "source_population_match": source_population_match,
        "required_assertion_count": len(required_keys),
        "frame_assertion_count": len(frame_key_set),
        "missing_assertion_count": len(missing),
        "orphan_assertion_count": len(orphan),
        "execution_without_claim": execution_without_claim,
        "claim_without_execution": claim_without_execution,
        "errors": errors,
    }


class FrameClosureReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_known_answer_omitted_use(self) -> None:
        report = verify(self.bundle)
        self.assertEqual(report["disposition"], "FRAME_GAP_DETECTED")
        self.assertFalse(report["promotion_authorized"])
        self.assertEqual(report["required_assertion_count"], 2)
        self.assertEqual(report["frame_assertion_count"], 1)
        self.assertEqual(report["missing_assertion_count"], 1)
        self.assertEqual(report["orphan_assertion_count"], 0)

    def test_unknown_source_coverage_blocks_promotion(self) -> None:
        self.bundle["source_exports"][0]["coverage_disposition"] = "UNKNOWN"
        report = verify(self.bundle)
        self.assertEqual(report["disposition"], "SOURCE_COVERAGE_UNKNOWN")

    def test_shared_custodian_blocks_promotion(self) -> None:
        self.bundle["source_exports"][1]["custodian_ref"] = (
            self.bundle["source_exports"][0]["custodian_ref"]
        )
        report = verify(self.bundle)
        self.assertEqual(report["disposition"], "SOURCE_COVERAGE_UNKNOWN")

    def test_orphan_frame_assertion_is_conflict(self) -> None:
        assertion = self.bundle["declared_frame"]["assertions"][0]
        assertion["left_evidence_id"] = "evidence-B"
        assertion["right_evidence_id"] = "evidence-C"
        self.bundle["declared_frame"]["frame_sha256"] = domain_hash(
            "connectome-fusion-frame-v0.1",
            self.bundle["declared_frame"]["assertions"],
        )
        report = verify(self.bundle)
        self.assertEqual(report["disposition"], "RECONCILIATION_CONFLICT")
        self.assertEqual(report["missing_assertion_count"], 2)
        self.assertEqual(report["orphan_assertion_count"], 1)

    def test_event_outside_scope_is_not_estimable(self) -> None:
        source = self.bundle["source_exports"][0]
        source["events"][0]["event_at"] = "2026-11-05T01:00:00Z"
        source["ordered_events_sha256"] = domain_hash(
            "connectome-fusion-events-v0.1", source["events"]
        )
        report = verify(self.bundle)
        self.assertEqual(report["disposition"], "NOT_ESTIMABLE")
        self.assertTrue(
            any(error.startswith("EVENT_OUTSIDE_DECLARED_SCOPE") for error in report["errors"])
        )

    def test_source_event_disagreement_is_conflict(self) -> None:
        source = self.bundle["source_exports"][1]
        event = source["events"][1]
        event["evidence_ids"] = ["evidence-B", "evidence-D"]
        refresh_event_identity(self.bundle, source)
        report = verify(self.bundle)
        self.assertEqual(report["disposition"], "RECONCILIATION_CONFLICT")
        self.assertIn("SOURCE_POPULATION_RECONCILIATION_MISMATCH", report["errors"])


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FrameClosureReferenceTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    if len(sys.argv) != 2:
        print(
            "usage: verify_connectome_fusion_frame_closure.py <input.json> | --self-test",
            file=sys.stderr,
        )
        return 2

    path = Path(sys.argv[1])
    raw = path.read_bytes()
    bundle = json.loads(raw)
    report = verify(bundle, raw)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
