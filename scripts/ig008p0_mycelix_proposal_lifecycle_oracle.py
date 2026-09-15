#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

AUTHORITY = "MeasurementOnly"
CONFORMANCE = "CrossImplementationConformance"
SCHEMA = "mycelix-proposal-lifecycle-counterexamples-v1"
PROFILE_ID = "mycelix-proposal-lifecycle-observed-fca2c107-v1"
PROFILE_SHA256 = "7f42e2a8df25df94112d23f261d1f3ffe299d46d37cb3a5a6fe02aca0aa6c108"
EXPECTED_CORPUS_SHA256 = "13eaaa988c73d29d67bccf7381f6f72cabd4eb7be090f36f0b444978cc708324"
MYCELIX_EVIDENCE_HEAD = "6ddca81103c52408421e2e31da4b4dee0c0b2762"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
SAME_TREE_MAIN = "31ede2365b81365bb119cd9351b2739119974130"
SOURCE_FILES = {
    "mycelix-governance/zomes/proposals/coordinator/src/lib.rs": "eb8358353ee259ef9c3b46617a61d3439f1c714c",
    "mycelix-governance/zomes/proposals/integrity/src/lib.rs": "986bc0526aec8d37436efbe5ba798bc41705e3cf",
}


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("profile root must be object")
    return obj


def validate_profile(profile: dict) -> dict:
    if profile.get("schema") != "mycelix-proposal-lifecycle-observed-profile-v1":
        raise ValueError("unexpected proposal profile schema")
    if profile.get("authority_class") != "ObservedSourceBound":
        raise ValueError("proposal profile authority drift")
    if profile.get("profile_id") != PROFILE_ID or profile.get("profile_revision") != 1:
        raise ValueError("proposal profile identity drift")

    payload = copy.deepcopy(profile)
    claimed = payload.pop("profile_content_sha256", None)
    actual = digest(payload)
    if claimed != actual or actual != PROFILE_SHA256:
        raise ValueError("proposal profile commitment mismatch")

    source = profile.get("source_binding", {})
    if source.get("repository") != "Luminous-Dynamics/mycelix":
        raise ValueError("proposal source repository drift")
    if source.get("production_subject_sha") != PRODUCTION_SUBJECT:
        raise ValueError("proposal production subject drift")
    if source.get("tree_equivalent_current_main_sha") != SAME_TREE_MAIN:
        raise ValueError("proposal same-tree main reference drift")
    files = {
        item.get("path"): item.get("git_blob_sha1")
        for item in source.get("files", [])
        if isinstance(item, dict)
    }
    if files != SOURCE_FILES:
        raise ValueError("proposal source blob binding drift")

    creation = profile.get("creation", {})
    if creation.get("proposal_author_source") != "CallerSuppliedButIntegrityBoundToCommitter":
        raise ValueError("proposal author creation boundary drift")
    if creation.get("initial_status") != "Draft" or creation.get("initial_version") != 1:
        raise ValueError("proposal creation lifecycle boundary drift")
    if creation.get("proposal_by_id_link") != "CreatedToProposalCreationAction":
        raise ValueError("ProposalById creation semantics drift")

    lookup = profile.get("lookup_projection", {})
    expected_lookup = {
        "entrypoint": "get_proposal",
        "primary_lookup": "ProposalByIdLink",
        "primary_link_selection": "MaxLinkTimestamp",
        "update_refreshes_proposal_by_id_link": "NoneObserved",
        "linked_record_return": "ReturnsLinkedRecordWithoutUpdateTraversal",
        "fallback": "LocalChainScanLastMatchingProposalOnlyWhenLinkedLookupDoesNotReturnRecord",
        "explicit_competing_update_fork_rule": "NoneObserved",
    }
    if lookup != expected_lookup:
        raise ValueError("proposal lookup/projection semantics drift")

    update = profile.get("update_integrity", {})
    required = {
        "update_action_author_binding": "NoneObserved",
        "content_freeze_condition": "OriginalStatusNotDraft",
        "draft_to_active_content_mutation_structurally_rejected": False,
        "voting_starts_immutability": "NoneObserved",
        "voting_ends_immutability": "NoneObserved",
        "created_timestamp_immutability": "NoneObserved",
        "updated_timestamp_action_binding": "NoneObserved",
        "update_voting_period_order_check": "NoneObserved",
        "version_rule": "UpdatedEqualsOriginalPlusOne",
    }
    for key, value in required.items():
        if update.get(key) != value:
            raise ValueError(f"proposal update observation drift: {key}")

    gaps = profile.get("known_gaps")
    if gaps != [{"issue": 66, "class": "MutableProposalProjectionAuthorityGap", "status": "Observed"}]:
        raise ValueError("proposal known-gap set drift")

    return {
        "profile_id": PROFILE_ID,
        "content_sha256": PROFILE_SHA256,
        "authority_class": "ObservedSourceBound",
    }


def build_corpus(profile: dict) -> dict:
    ref = validate_profile(profile)
    lookup = profile["lookup_projection"]
    update = profile["update_integrity"]

    corpus = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "profile": ref,
        "issue": 66,
        "counterexamples": [
            {
                "id": "CE-PROP-01",
                "revision": 1,
                "kind": "ProjectionCurrentnessObservation",
                "inputs": {
                    "creation_status": "Draft",
                    "update_child_status": "Active",
                    "proposal_by_id_target": "CreationAction",
                    "linked_record_present": True,
                },
                "comparison": {
                    "primary_lookup": lookup["primary_lookup"],
                    "linked_record_return": lookup["linked_record_return"],
                    "fallback_reached": False,
                    "observed_read_status": "Draft",
                    "authoritative_currentness": "NotEstablished",
                },
                "non_claim": (
                    "Models the frozen source lookup semantics; does not claim every deployment "
                    "read has this history."
                ),
            },
            {
                "id": "CE-PROP-02",
                "revision": 1,
                "kind": "StructuralUpdateDifferential",
                "inputs": {
                    "original_status": "Draft",
                    "updated_status": "Active",
                    "id_author_unchanged": True,
                    "version_increment": 1,
                    "semantic_content_changed": True,
                },
                "comparison": {
                    "content_freeze_condition": update["content_freeze_condition"],
                    "condition_holds": False,
                    "draft_to_active_content_mutation_structurally_rejected": update[
                        "draft_to_active_content_mutation_structurally_rejected"
                    ],
                    "result": "ContentMutationNotRejectedByObservedUpdateCheck",
                },
                "non_claim": (
                    "Records the pure update-check predicate boundary; does not publish a live "
                    "proposal update."
                ),
            },
            {
                "id": "CE-PROP-03",
                "revision": 1,
                "kind": "UpdateAuthorityPredicateObservation",
                "inputs": {
                    "proposal_author": "did:mycelix:alice",
                    "update_action_author": "did:mycelix:bob",
                    "structural_fields_otherwise_valid": True,
                },
                "comparison": {
                    "update_action_author_binding": update["update_action_author_binding"],
                    "result": "IntegrityDoesNotEstablishUpdateAuthorAuthority",
                },
                "non_claim": (
                    "Records that proposal update integrity does not inspect the update action "
                    "author; no live unauthorized update is claimed."
                ),
            },
            {
                "id": "CE-PROP-04",
                "revision": 1,
                "kind": "TemporalIntegrityObservation",
                "inputs": {
                    "original_voting_starts": 1,
                    "original_voting_ends": 2,
                    "updated_voting_starts": 3,
                    "updated_voting_ends": 2,
                    "created_changed": True,
                    "updated_timestamp_arbitrary": True,
                },
                "comparison": {
                    "voting_starts_immutability": update["voting_starts_immutability"],
                    "voting_ends_immutability": update["voting_ends_immutability"],
                    "created_timestamp_immutability": update["created_timestamp_immutability"],
                    "updated_timestamp_action_binding": update["updated_timestamp_action_binding"],
                    "update_voting_period_order_check": update["update_voting_period_order_check"],
                    "result": "TemporalMutationNotRejectedByObservedUpdateCheck",
                },
                "non_claim": (
                    "Models missing update-level temporal predicates, not a live accepted DHT mutation."
                ),
            },
            {
                "id": "CE-PROP-05",
                "revision": 1,
                "kind": "ForkProjectionObservation",
                "inputs": {
                    "common_parent_status": "Draft",
                    "child_a": {"status": "Active", "version": 2},
                    "child_b": {"status": "Cancelled", "version": 2},
                },
                "comparison": {
                    "both_transition_shapes_allowed": True,
                    "explicit_competing_update_fork_rule": lookup[
                        "explicit_competing_update_fork_rule"
                    ],
                    "proposal_by_id_update_refresh": lookup[
                        "update_refreshes_proposal_by_id_link"
                    ],
                    "result": "NoObservedDeterministicAuthoritativeChildSelection",
                },
                "non_claim": (
                    "Records absence of an explicit authoritative fork projector in the legacy "
                    "profile; it does not claim a particular DHT arrival order."
                ),
            },
        ],
        "non_claims": [
            "no_live_unauthorized_update_claim",
            "no_deployment_exploit_claim",
            "no_authoritative_currentness_claim",
            "no_successor_stack_deployment_claim",
            "no_governance_safety_claim",
        ],
    }
    corpus["corpus_sha256"] = digest(corpus)
    return corpus


def self_test(profile: dict) -> dict:
    first = build_corpus(profile)
    second = build_corpus(profile)
    if canonical(first) != canonical(second):
        raise AssertionError("non-deterministic proposal conformance corpus")
    if first["corpus_sha256"] != EXPECTED_CORPUS_SHA256:
        raise AssertionError(
            f"cross-implementation proposal corpus mismatch: {first['corpus_sha256']}"
        )
    ids = {item["id"] for item in first["counterexamples"]}
    if ids != {"CE-PROP-01", "CE-PROP-02", "CE-PROP-03", "CE-PROP-04", "CE-PROP-05"}:
        raise AssertionError("proposal counterexample roster drift")

    return {
        "authority": AUTHORITY,
        "conformance_class": CONFORMANCE,
        "mycelix_evidence_head": MYCELIX_EVIDENCE_HEAD,
        "mycelix_production_subject": PRODUCTION_SUBJECT,
        "same_tree_main": SAME_TREE_MAIN,
        "profile_sha256": PROFILE_SHA256,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "counterexample_count": 5,
        "issue": 66,
        "self_test": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--corpus", action="store_true")
    args = parser.parse_args()
    profile = load(args.profile)
    if args.self_test:
        result = self_test(profile)
    elif args.corpus:
        result = build_corpus(profile)
    else:
        parser.error("choose --self-test or --corpus")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
