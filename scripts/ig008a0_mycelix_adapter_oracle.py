#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

AUTHORITY = "MeasurementOnly"
CONFORMANCE = "CrossImplementationConformance"
PROFILE_ID = "mycelix-voting-observed-fca2c107-v2"
PROFILE_SHA256 = "680af4668889c299b7e0d74531f44894a64a384be71778bca54d3f21ca80ac01"
MYCELIX_EVIDENCE_HEAD = "4b4e27910a98ad393e5a508e54e0f73c48c19107"
MYCELIX_PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
COORDINATOR_BLOB = "969b845e6186cbcad507c742a718060844f82eb2"
INTEGRITY_BLOB = "658562c8dfaf6a2f1b97a7bfd5cf0fc8a5ab6e66"
PREDECESSOR_PROFILE_SHA256 = "cbbbb3553ce465be989b5f096364ea97ccc9b1c5d6aae67d80177df8d8109763"
PREDECESSOR_CORPUS_SHA256 = "bb1cdcfe2205bcf6e6d718b536cbb73ba29a21c9da7dfa664798d02f9f866d90"
EXPECTED_CORPUS_SHA256 = "ee5e7649a773f564b443320689f465080d4641a0f4f09f13c0c49a7087d6dc10"
SCHEMA = "mycelix-observed-voting-counterexamples-v2"


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256(obj: object) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def load_object(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("profile root must be an object")
    return value


def validate_profile(profile: dict) -> dict:
    if profile.get("schema") != "mycelix-governance-observed-voting-profile-v2":
        raise ValueError("unexpected Mycelix observed-profile schema")
    if profile.get("authority_class") != "ObservedSourceBound":
        raise ValueError("Mycelix profile authority must remain ObservedSourceBound")
    if profile.get("profile_id") != PROFILE_ID or profile.get("profile_revision") != 2:
        raise ValueError("unexpected Mycelix profile identity")

    payload = copy.deepcopy(profile)
    claimed = payload.pop("profile_content_sha256", None)
    actual = sha256(payload)
    if claimed != actual or actual != PROFILE_SHA256:
        raise ValueError("Mycelix profile content commitment mismatch")

    source = profile.get("source_binding")
    if not isinstance(source, dict):
        raise ValueError("missing source binding")
    if source.get("repository") != "Luminous-Dynamics/mycelix":
        raise ValueError("unexpected source repository")
    if source.get("production_subject_sha") != MYCELIX_PRODUCTION_SUBJECT:
        raise ValueError("unexpected production subject")
    if source.get("observation_relation_to_v1") != "SameSourceSubjectMoreCompleteObservation":
        raise ValueError("unexpected observation lineage")

    files = {
        item.get("path"): item.get("git_blob_sha1")
        for item in source.get("files", [])
        if isinstance(item, dict)
    }
    expected_files = {
        "mycelix-governance/zomes/voting/coordinator/src/lib.rs": COORDINATOR_BLOB,
        "mycelix-governance/zomes/voting/integrity/src/lib.rs": INTEGRITY_BLOB,
    }
    if files != expected_files:
        raise ValueError("unexpected source-blob binding")

    gaps = {g.get("issue") for g in profile.get("known_gaps", []) if isinstance(g, dict)}
    if gaps != {851, 855, 856, 876, 877, 892}:
        raise ValueError("unexpected observed-gap set")

    direct = profile["vote_paths"]["phi_weighted_vote"]
    delegated = profile["vote_paths"]["delegated_phi_vote"]
    tally = profile["phi_tally"]

    expected_direct = {
        "proposal_window": "verify_voting_period_fail_closed",
        "phi_threshold_admission": "AvailablePhiMustMeetCallerTier;UnavailableSkipsGate",
        "duplicate_voter_guard": "AgentNamespacePlusPhiVoterLink",
    }
    for key, value in expected_direct.items():
        if direct.get(key) != value:
            raise ValueError(f"direct Phi observation drift: {key}")

    expected_delegated = {
        "proposal_window": "NoVerifyVotingPeriodCallObserved",
        "phi_threshold_admission": "NoMeetsThresholdCallObserved",
        "duplicate_voter_guard": "NoCoordinatorGuardObserved",
        "outgoing_allocation_conservation": "NotEstablishedAcrossApplicableActiveDelegations",
        "resolver_cycle_control": "VisitedSetPerResolutionTraversal",
        "authorship_binding": "VoterMustEqualCommittingAgent",
    }
    for key, value in expected_delegated.items():
        if delegated.get(key) != value:
            raise ValueError(f"delegated Phi observation drift: {key}")
    if delegated.get("agent_vote_limit_namespace") is not None:
        raise ValueError("unexpected delegated agent vote-limit namespace")
    if tally.get("defensive_voter_deduplication_observed") is not False:
        raise ValueError("unexpected tally deduplication claim")

    return {
        "profile_id": PROFILE_ID,
        "content_sha256": PROFILE_SHA256,
        "authority_class": "ObservedSourceBound",
    }


def basic_tally(record_count: int, per_record_for_weight: float) -> dict:
    eligible_voters = 10
    voter_count = record_count
    votes_for = record_count * per_record_for_weight
    participation_rate = voter_count / eligible_voters
    required_voter_count = max(math.ceil(eligible_voters * 0.15), 3)
    quorum_reached = participation_rate >= 0.15 and voter_count >= required_voter_count
    approval_rate = 1.0 if votes_for > 0.0 else 0.0
    return {
        "eligible_voters": eligible_voters,
        "voter_count": voter_count,
        "phi_votes_for": votes_for,
        "phi_votes_against": 0.0,
        "participation_rate": participation_rate,
        "required_voter_count": required_voter_count,
        "quorum_reached": quorum_reached,
        "approval_rate": approval_rate,
        "approved": quorum_reached and approval_rate >= 0.50,
    }


def build_corpus(profile: dict) -> dict:
    profile_ref = validate_profile(profile)

    corpus = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "profile": profile_ref,
        "predecessor": {
            "schema": "mycelix-observed-voting-counterexamples-v1",
            "corpus_sha256": PREDECESSOR_CORPUS_SHA256,
            "profile_sha256": PREDECESSOR_PROFILE_SHA256,
        },
        "counterexamples": [
            {
                "id": "CE-06",
                "revision": 1,
                "kind": "AuthorityGapCounterfactual",
                "issue": 876,
                "inputs": {
                    "tier": "Basic",
                    "eligible_voters": 10,
                    "same_voter_identity": "did:example:delegate",
                    "per_record_for_weight": 0.5,
                },
                "comparison": {
                    "one_record": basic_tally(1, 0.5),
                    "three_same_voter_records": basic_tally(3, 0.5),
                },
                "non_claim": (
                    "Models tally consequence if source-visible delegated duplicate gap permits "
                    "multiple linked records; not a live exploit claim."
                ),
            },
            {
                "id": "CE-07",
                "revision": 1,
                "kind": "InvariantGapCounterfactual",
                "issue": 877,
                "inputs": {
                    "delegator_source_weight": 0.5,
                    "allocations": [
                        {"delegate": "Bob", "percentage": 1.0},
                        {"delegate": "Carol", "percentage": 1.0},
                    ],
                    "scope_relation": "simultaneously_applicable",
                },
                "comparison": {
                    "bob_resolved_delegated_contribution": 0.5,
                    "carol_resolved_delegated_contribution": 0.5,
                    "total_represented_delegator_mass": 1.0,
                    "source_delegator_mass": 0.5,
                    "representation_multiple": 2.0,
                },
                "non_claim": (
                    "Isolates absence of an observed cross-resolution conservation rule; "
                    "does not choose exclusive or fractional successor semantics."
                ),
            },
            {
                "id": "CE-08",
                "revision": 1,
                "kind": "AdmissionGapCounterfactual",
                "issue": 892,
                "inputs": {"proposal_window_state": "Closed"},
                "comparison": {
                    "direct_phi": "RejectClosedWindowByDeclaredSourcePolicy",
                    "delegated_phi": "NoObservedWindowRejectionGate",
                },
                "non_claim": (
                    "NoObservedWindowRejectionGate is not equivalent to accepted execution; "
                    "other runtime failures remain possible."
                ),
            },
            {
                "id": "CE-09",
                "revision": 1,
                "kind": "AdmissionGapCounterfactual",
                "issue": 892,
                "inputs": {
                    "tier": "Major",
                    "phi_provenance": "Attested",
                    "phi_score": 0.2,
                    "required_phi_threshold": 0.4,
                },
                "comparison": {
                    "direct_phi": "RejectBelowTierPhiThresholdByDeclaredSourcePolicy",
                    "delegated_phi": "NoObservedPhiThresholdRejectionGate",
                },
                "non_claim": (
                    "NoObservedPhiThresholdRejectionGate is not equivalent to accepted execution; "
                    "this is a source-policy differential."
                ),
            },
        ],
        "non_claims": [
            "no_live_exploit_claim",
            "no_production_impact_estimate",
            "no_fairness_claim",
            "no_governance_safety_claim",
            "no_policy_migration_authority",
        ],
    }
    corpus["corpus_sha256"] = sha256(corpus)
    return corpus


def self_test(profile: dict) -> dict:
    first = build_corpus(profile)
    second = build_corpus(profile)
    if canonical(first) != canonical(second):
        raise AssertionError("non-deterministic Symthaea adapter corpus")
    if first["corpus_sha256"] != EXPECTED_CORPUS_SHA256:
        raise AssertionError("cross-implementation corpus commitment mismatch")

    by_id = {x["id"]: x for x in first["counterexamples"]}
    assert set(by_id) == {"CE-06", "CE-07", "CE-08", "CE-09"}
    assert by_id["CE-06"]["comparison"]["one_record"]["approved"] is False
    assert by_id["CE-06"]["comparison"]["three_same_voter_records"]["approved"] is True
    assert by_id["CE-07"]["comparison"]["representation_multiple"] == 2.0
    assert by_id["CE-08"]["comparison"]["delegated_phi"] == "NoObservedWindowRejectionGate"
    assert by_id["CE-09"]["comparison"]["delegated_phi"] == "NoObservedPhiThresholdRejectionGate"

    return {
        "authority": AUTHORITY,
        "conformance_class": CONFORMANCE,
        "mycelix_evidence_head": MYCELIX_EVIDENCE_HEAD,
        "mycelix_production_subject": MYCELIX_PRODUCTION_SUBJECT,
        "profile_sha256": PROFILE_SHA256,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "counterexample_count": 4,
        "self_test": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--corpus", action="store_true")
    args = parser.parse_args()

    profile = load_object(args.profile)
    if args.self_test:
        out = self_test(profile)
    elif args.corpus:
        out = build_corpus(profile)
    else:
        parser.error("choose --self-test or --corpus")
    print(json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
