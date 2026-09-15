#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

AUTHORITY = "MeasurementOnly"
CONFORMANCE = "CrossImplementationConformance"
SCHEMA = "mycelix-threshold-signing-counterexamples-v1"
PROFILE_ID = "mycelix-threshold-signing-observed-fca2c107-v1"
PROFILE_SHA256 = "c15dfd860b759747938af2a13129d729fa0af1e75284418c9ea6b9c172f643ac"
EXPECTED_CORPUS_SHA256 = "0f6532ae8e2c2e421da625592dbb3b38aa2b90c5342f46f3a305bdbec89b0269"
MYCELIX_EVIDENCE_HEAD = "a580915d588338077ce6196514c43e24052f86cd"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
SOURCE_FILES = {
    "mycelix-governance/zomes/threshold-signing/coordinator/src/lib.rs": "3449df8b03a4dd1774a5f22756d06931c72855b2",
    "mycelix-governance/zomes/threshold-signing/integrity/src/lib.rs": "3fec8344635600c044a494fa72bbbfe408fbe5ec",
    "mycelix-governance/zomes/proposals/coordinator/src/lib.rs": "eb8358353ee259ef9c3b46617a61d3439f1c714c",
    "mycelix-governance/zomes/execution/coordinator/src/lib.rs": "3dbb8a8f69b377e494ccf24164c94bd80f54e0ef",
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
    if profile.get("schema") != "mycelix-threshold-signing-observed-profile-v1":
        raise ValueError("unexpected threshold-signing profile schema")
    if profile.get("authority_class") != "ObservedSourceBound":
        raise ValueError("threshold-signing profile authority drift")
    if profile.get("profile_id") != PROFILE_ID or profile.get("profile_revision") != 1:
        raise ValueError("threshold-signing profile identity drift")

    payload = copy.deepcopy(profile)
    claimed = payload.pop("profile_content_sha256", None)
    actual = digest(payload)
    if claimed != actual or actual != PROFILE_SHA256:
        raise ValueError("threshold-signing profile commitment mismatch")

    source = profile.get("source_binding", {})
    if source.get("repository") != "Luminous-Dynamics/mycelix":
        raise ValueError("unexpected source repository")
    if source.get("production_subject_sha") != PRODUCTION_SUBJECT:
        raise ValueError("unexpected production subject")
    files = {
        item.get("path"): item.get("git_blob_sha1")
        for item in source.get("files", [])
        if isinstance(item, dict)
    }
    if files != SOURCE_FILES:
        raise ValueError("threshold-signing source bindings drift")

    producer = profile.get("producer_api", {})
    if producer.get("observed_externs") != ["create_committee"]:
        raise ValueError("producer extern census drift")
    if producer.get("consumer_expected_queries") != ["get_proposal_signature", "get_committee"]:
        raise ValueError("consumer query contract drift")
    if producer.get("get_proposal_signature") != "NoneObserved":
        raise ValueError("unexpected proposal-signature query observation")
    if producer.get("get_committee") != "NoneObserved":
        raise ValueError("unexpected committee query observation")

    integrity = profile.get("threshold_signature_integrity", {})
    required_none = (
        "cryptographic_signature_verification",
        "committee_lookup",
        "committee_active_epoch_check",
        "committee_threshold_check",
        "qualified_signer_membership_check",
        "committee_scope_check",
        "signed_subject_authorization_reconstruction",
        "verified_field_recomputed",
    )
    for key in required_none:
        if integrity.get(key) != "NoneObserved":
            raise ValueError(f"unexpected observed authority predicate: {key}")
    if integrity.get("create_validator") != "check_signature_validity":
        raise ValueError("signature-create validator drift")

    source_test = profile.get("source_test_observation", {})
    if source_test != {
        "fixture": "make_test_signature(Ecdsa)",
        "signature_bytes": "64ZeroBytes",
        "signed_content_hash": "32NonzeroFixtureBytes",
        "signer_count": 1,
        "signers": [1],
        "verified": False,
        "pure_validator_result": "AcceptedByCheckSignatureValidity",
        "evidence_class": "StructuralValidatorTest",
    }:
        raise ValueError("source structural fixture drift")

    link = profile.get("proposal_to_signature_link", {})
    if link != {
        "create_validation": "UnconditionalValidObserved",
        "exact_subject_authorization_reconstruction": "NoneObserved",
    }:
        raise ValueError("proposal-signature association observation drift")

    gaps = profile.get("known_gaps")
    if gaps != [
        {"issue": 959, "class": "ProducerConsumerApiContractGap", "status": "Observed"},
        {"issue": 960, "class": "ThresholdSignatureAuthorityGap", "status": "Observed"},
    ]:
        raise ValueError("threshold-signing known-gap set drift")

    return {
        "profile_id": PROFILE_ID,
        "content_sha256": PROFILE_SHA256,
        "authority_class": "ObservedSourceBound",
    }


def build_corpus(profile: dict) -> dict:
    ref = validate_profile(profile)
    integrity = profile["threshold_signature_integrity"]
    source_test = profile["source_test_observation"]
    link = profile["proposal_to_signature_link"]
    producer = profile["producer_api"]

    corpus = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "profile": ref,
        "issues": [959, 960],
        "counterexamples": [
            {
                "id": "CE-SIG-01",
                "revision": 1,
                "kind": "StructuralValidatorObservation",
                "inputs": {
                    "algorithm": "Ecdsa",
                    "signature_bytes": source_test["signature_bytes"],
                    "signed_content_hash": source_test["signed_content_hash"],
                    "signer_count": source_test["signer_count"],
                    "signers": source_test["signers"],
                    "verified": source_test["verified"],
                },
                "comparison": {
                    "pure_validator_result": source_test["pure_validator_result"],
                    "evidence_class": source_test["evidence_class"],
                    "cryptographic_validity": "NotEstablished",
                },
                "non_claim": (
                    "Models the exact source test shape and pure structural validator result; "
                    "does not claim a cryptographically valid signature."
                ),
            },
            {
                "id": "CE-SIG-02",
                "revision": 1,
                "kind": "StoredFlagAuthorityDifferential",
                "inputs": {
                    "otherwise_identical": True,
                    "verified_values": [False, True],
                },
                "comparison": {
                    "verified_field_recomputed": integrity["verified_field_recomputed"],
                    "structural_validity_depends_on_verified": False,
                    "result": "VerifiedFlagDoesNotAffectObservedStructuralValidity",
                },
                "non_claim": (
                    "Shows the observed pure validator does not derive cryptographic truth from "
                    "or recompute the stored verified flag."
                ),
            },
            {
                "id": "CE-SIG-03",
                "revision": 1,
                "kind": "CommitteeAuthorityPredicateObservation",
                "inputs": {
                    "modeled_committee_threshold": 3,
                    "fixture_signer_count": 1,
                },
                "comparison": {
                    "committee_lookup": integrity["committee_lookup"],
                    "committee_active_epoch_check": integrity["committee_active_epoch_check"],
                    "committee_threshold_check": integrity["committee_threshold_check"],
                    "qualified_signer_membership_check": integrity["qualified_signer_membership_check"],
                    "committee_scope_check": integrity["committee_scope_check"],
                    "result": "NoObservedCommitteeAuthorityPredicateInSignatureCreateValidation",
                },
                "non_claim": (
                    "Records absent source-visible committee authorization predicates; "
                    "does not claim a live threshold bypass."
                ),
            },
            {
                "id": "CE-SIG-04",
                "revision": 1,
                "kind": "AssociationAuthorityDifferential",
                "inputs": {"link_type": "ProposalToSignature"},
                "comparison": {
                    "create_validation": link["create_validation"],
                    "exact_subject_authorization_reconstruction": link[
                        "exact_subject_authorization_reconstruction"
                    ],
                    "result": "LinkDoesNotEstablishObservedProposalSignatureAuthorization",
                },
                "non_claim": (
                    "Treats the observed link as association only; no live proposal-signature "
                    "authorization claim is made."
                ),
            },
            {
                "id": "CE-SIG-05",
                "revision": 1,
                "kind": "ProducerConsumerApiContractObservation",
                "inputs": {
                    "consumer_expected_queries": producer["consumer_expected_queries"],
                },
                "comparison": {
                    "producer_observed_externs": producer["observed_externs"],
                    "get_proposal_signature": producer["get_proposal_signature"],
                    "get_committee": producer["get_committee"],
                    "result": "ExpectedSignatureQueryContractAbsentFromObservedProducer",
                },
                "non_claim": (
                    "Records the exact source API mismatch without asserting a particular "
                    "runtime failure mode."
                ),
            },
        ],
        "non_claims": [
            "no_forged_signature_acceptance_claim",
            "no_cryptographic_break",
            "no_live_threshold_bypass",
            "no_deployment_exploit",
            "no_governance_safety_claim",
        ],
    }
    corpus["corpus_sha256"] = digest(corpus)
    return corpus


def self_test(profile: dict) -> dict:
    first = build_corpus(profile)
    second = build_corpus(profile)
    if canonical(first) != canonical(second):
        raise AssertionError("non-deterministic threshold-signing conformance corpus")
    if first["corpus_sha256"] != EXPECTED_CORPUS_SHA256:
        raise AssertionError(
            f"cross-implementation threshold-signing corpus mismatch: {first['corpus_sha256']}"
        )
    ids = {item["id"] for item in first["counterexamples"]}
    if ids != {"CE-SIG-01", "CE-SIG-02", "CE-SIG-03", "CE-SIG-04", "CE-SIG-05"}:
        raise AssertionError("threshold-signing counterexample roster drift")

    return {
        "authority": AUTHORITY,
        "conformance_class": CONFORMANCE,
        "mycelix_evidence_head": MYCELIX_EVIDENCE_HEAD,
        "mycelix_production_subject": PRODUCTION_SUBJECT,
        "profile_sha256": PROFILE_SHA256,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "counterexample_count": 5,
        "issues": [959, 960],
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
        out = self_test(profile)
    elif args.corpus:
        out = build_corpus(profile)
    else:
        parser.error("choose --self-test or --corpus")

    print(json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
