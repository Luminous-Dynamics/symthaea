#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROFILE_ID = "mycelix-constitution-parameter-observed-fca2c107-v1"
PROFILE_SHA256 = "770552d12489df1d2cdf8b0af676b01ea9a3da21940f70ed8a71910deaa35009"
CORPUS_SHA256 = "b37be9d2e3fd0cbec3696a19327c26dc4ba7062ec089ad92ad99bc28a11fea8e"
MYCELIX_EVIDENCE_HEAD = "3232d611d8833b03eba9f5412f5d7cb0cf89d4e1"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
SAME_TREE_MAIN = "31ede2365b81365bb119cd9351b2739119974130"
AUTHORITY = "MeasurementOnly"
CONFORMANCE = "CrossImplementationConformance"
SCHEMA = "mycelix-constitution-parameter-counterexamples-v1"


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("profile root must be object")
    return obj


def profile_digest(profile: dict) -> str:
    payload = dict(profile)
    payload.pop("profile_content_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def require_profile(profile: dict) -> None:
    if profile.get("profile_id") != PROFILE_ID or profile.get("profile_revision") != 1:
        raise ValueError("unexpected ConstitutionParameter profile identity")
    if profile.get("authority_class") != "ObservedSourceBound":
        raise ValueError("unexpected source authority")
    if profile_digest(profile) != PROFILE_SHA256 or profile.get("profile_content_sha256") != PROFILE_SHA256:
        raise ValueError("profile commitment mismatch")

    source = profile.get("source_binding", {})
    if source.get("production_subject_sha") != PRODUCTION_SUBJECT:
        raise ValueError("semantic production subject drift")
    if source.get("tree_equivalent_current_main_sha") != SAME_TREE_MAIN:
        raise ValueError("same-tree authoring head drift")

    dispatch = profile.get("execution_dispatch", {})
    if dispatch.get("payload_fields") != ["parameter", "value"]:
        raise ValueError("execution payload shape drift")
    if dispatch.get("proposal_id_included") is not False:
        raise ValueError("unexpected execution proposal-id semantics")
    if dispatch.get("qualified_authorization_ref_included") is not False:
        raise ValueError("unexpected execution authorization semantics")

    coordinator = profile.get("coordinator_update", {})
    expected_none = (
        "proposal_lookup",
        "proposal_status_check",
        "proposal_type_check",
        "exact_parameter_value_authorization_binding",
        "execution_authorization_binding",
        "caller_authority_check",
    )
    for field in expected_none:
        if coordinator.get(field) != "NoneObserved":
            raise ValueError(f"coordinator authority drift: {field}")

    gate = profile.get("set_parameter_gate", {})
    if gate.get("existing_parameter_without_proposal_id") != "Rejected":
        raise ValueError("existing-parameter containment drift")
    if gate.get("existing_parameter_with_nonempty_or_unverified_some_proposal_id") != "PresenceSatisfiesObservedCoordinatorGate":
        raise ValueError("proposal-presence gate drift")
    if gate.get("new_parameter_without_proposal_id") != "AllowedByObservedCoordinatorGate":
        raise ValueError("new-parameter bootstrap observation drift")
    if gate.get("proposal_id_authority_reconstruction") != "NoneObserved":
        raise ValueError("proposal authority reconstruction drift")

    integrity = profile.get("integrity", {})
    if integrity.get("create_checks") != ["NameNonEmpty", "ValueValidJson"]:
        raise ValueError("integrity create-check drift")
    if integrity.get("create_action_author_binding") != "NoneObserved":
        raise ValueError("integrity author-binding drift")
    if integrity.get("changed_by_proposal_authority_verification") != "NoneObserved":
        raise ValueError("integrity proposal-authority drift")

    projection = profile.get("storage_projection", {})
    if projection.get("link_selection") != "MaxLinkTimestamp":
        raise ValueError("parameter projection drift")
    if projection.get("explicit_authoritative_fork_rule") != "NoneObserved":
        raise ValueError("fork rule drift")
    if projection.get("timestamp_selection_is_authority") != "NotEstablished":
        raise ValueError("timestamp authority drift")


def build_corpus(profile: dict) -> dict:
    require_profile(profile)
    dispatch = profile["execution_dispatch"]
    gate = profile["set_parameter_gate"]
    integrity = profile["integrity"]
    projection = profile["storage_projection"]

    corpus = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "profile": {
            "profile_id": PROFILE_ID,
            "content_sha256": PROFILE_SHA256,
            "authority_class": "ObservedSourceBound",
        },
        "issue": 1002,
        "counterexamples": [
            {
                "id": "CE-CP-01",
                "revision": 1,
                "kind": "ExecutionDownstreamGateMismatch",
                "inputs": {
                    "execution_action": "UpdateParameter",
                    "execution_payload_fields": ["parameter", "value"],
                    "parameter_preexists": True,
                    "proposal_id_received_by_constitution": None,
                },
                "comparison": {
                    "execution_proposal_id_included": dispatch["proposal_id_included"],
                    "existing_parameter_without_proposal_id": gate["existing_parameter_without_proposal_id"],
                    "result": "ObservedExecutionDispatchCannotPassExistingParameterPresenceGate",
                },
                "non_claim": "Shows the frozen source-contract mismatch for an existing parameter; does not claim a live execution attempt.",
            },
            {
                "id": "CE-CP-02",
                "revision": 1,
                "kind": "BootstrapAuthorizationObservation",
                "inputs": {
                    "parameter_preexists": False,
                    "proposal_id": None,
                    "name_nonempty": True,
                    "value_valid_json": True,
                },
                "comparison": {
                    "new_parameter_without_proposal_id": gate["new_parameter_without_proposal_id"],
                    "changed_by_proposal": None,
                    "result": "NewParameterCreationAllowedWithoutProposalLinkage",
                },
                "non_claim": "Models the coordinator gate for a previously absent parameter; no live mutation is performed.",
            },
            {
                "id": "CE-CP-03",
                "revision": 1,
                "kind": "ProposalPresenceAuthorityObservation",
                "inputs": {
                    "parameter_preexists": True,
                    "proposal_id": "proposal:fixture",
                    "proposal_existence": "NotReconstructed",
                    "proposal_status": "NotReconstructed",
                    "proposal_type": "NotReconstructed",
                    "exact_parameter_value_authorization": "NotReconstructed",
                    "caller_authority": "NotReconstructed",
                },
                "comparison": {
                    "observed_gate": gate["existing_parameter_with_nonempty_or_unverified_some_proposal_id"],
                    "proposal_id_authority_reconstruction": gate["proposal_id_authority_reconstruction"],
                    "result": "ProposalIdPresencePassesObservedCoordinatorGateWithoutAuthorityReconstruction",
                },
                "non_claim": "Records that string presence satisfies this gate; it does not claim the fixture proposal exists or is authorized.",
            },
            {
                "id": "CE-CP-04",
                "revision": 1,
                "kind": "IntegrityAuthorityObservation",
                "inputs": {
                    "name": "governance.fixture",
                    "value": "{\"enabled\":true}",
                    "changed_by_proposal": None,
                    "create_action_author": "arbitrary-fixture-author",
                },
                "comparison": {
                    "create_checks": integrity["create_checks"],
                    "create_action_author_binding": integrity["create_action_author_binding"],
                    "changed_by_proposal_authority_verification": integrity["changed_by_proposal_authority_verification"],
                    "result": "IntegrityShapeValidityDoesNotEstablishParameterMutationAuthority",
                },
                "non_claim": "Pure structural fixture only; no DHT publication or unauthorized mutation is performed.",
            },
            {
                "id": "CE-CP-05",
                "revision": 1,
                "kind": "ProjectionAuthorityObservation",
                "inputs": {
                    "same_parameter_name": "quorum",
                    "publication_a": {"value": "0.60", "link_timestamp": 100},
                    "publication_b": {"value": "0.70", "link_timestamp": 200},
                },
                "comparison": {
                    "link_selection": projection["link_selection"],
                    "selected_fixture": "publication_b",
                    "explicit_authoritative_fork_rule": projection["explicit_authoritative_fork_rule"],
                    "timestamp_selection_is_authority": projection["timestamp_selection_is_authority"],
                    "result": "TimestampSelectedProjectionNotAuthoritativeForkResolution",
                },
                "non_claim": "Models the frozen read projection; does not claim a particular production fork exists.",
            },
        ],
        "non_claims": [
            "no_live_unauthorized_parameter_mutation",
            "no_deployment_exploit",
            "no_legal_constitutional_invalidity_claim",
            "no_authoritative_currentness_claim",
            "no_governance_safety_claim",
        ],
    }
    corpus["corpus_sha256"] = hashlib.sha256(canonical(corpus)).hexdigest()
    return corpus


def self_test(profile: dict) -> dict:
    first = build_corpus(profile)
    second = build_corpus(profile)
    if canonical(first) != canonical(second):
        raise AssertionError("non-deterministic independent ConstitutionParameter corpus")
    if first["corpus_sha256"] != CORPUS_SHA256:
        raise AssertionError(f"cross-implementation corpus drift: {first['corpus_sha256']}")
    ids = {item["id"] for item in first["counterexamples"]}
    if ids != {"CE-CP-01", "CE-CP-02", "CE-CP-03", "CE-CP-04", "CE-CP-05"}:
        raise AssertionError("unexpected counterexample roster")
    return {
        "authority": AUTHORITY,
        "conformance_class": CONFORMANCE,
        "schema": SCHEMA,
        "mycelix_evidence_head": MYCELIX_EVIDENCE_HEAD,
        "mycelix_production_subject": PRODUCTION_SUBJECT,
        "same_tree_main": SAME_TREE_MAIN,
        "profile_sha256": PROFILE_SHA256,
        "corpus_sha256": CORPUS_SHA256,
        "counterexample_count": 5,
        "issue": 1002,
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
