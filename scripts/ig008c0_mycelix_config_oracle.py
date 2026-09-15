#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

PROFILE_ID = "mycelix-governance-config-observed-fca2c107-v1"
PROFILE_SHA256 = "de4435a69356557b1812f8beb46d654c66b9c957be9d18c64bd0431f92546d5a"
CORPUS_SHA256 = "3009e97529934fa8f470769de5615dcfda17bde0e942683e939d2b733430a216"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
SCHEMA = "mycelix-governance-config-counterexamples-v1"
EXPECTED_FILES = {
    "mycelix-governance/zomes/bridge/coordinator/src/consciousness_config.rs": "26da234e588bf26d0e25c10dbec34502e00c191a",
    "mycelix-governance/zomes/bridge/integrity/src/lib.rs": "61b20610216e2fd69c701ecaea5322c448eda119",
    "mycelix-governance/zomes/proposals/coordinator/src/lib.rs": "eb8358353ee259ef9c3b46617a61d3439f1c714c",
    "mycelix-governance/zomes/proposals/integrity/src/lib.rs": "986bc0526aec8d37436efbe5ba798bc41705e3cf",
}


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def payload_digest(profile: dict) -> str:
    payload = dict(profile)
    payload.pop("profile_content_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("profile root must be object")
    return value


def shape_valid(gates: dict[str, float]) -> bool:
    ordered = [gates[k] for k in ("basic", "proposal", "voting", "constitutional")]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in ordered):
        return False
    if not all(0.0 <= float(v) <= 1.0 for v in ordered):
        return False
    return all(float(a) <= float(b) for a, b in zip(ordered, ordered[1:]))


def validate_profile(profile: dict) -> None:
    if profile.get("profile_id") != PROFILE_ID or profile.get("profile_revision") != 1:
        raise ValueError("unexpected profile identity")
    if profile.get("authority_class") != "ObservedSourceBound":
        raise ValueError("profile authority drift")
    if profile.get("profile_content_sha256") != PROFILE_SHA256:
        raise ValueError("profile commitment field drift")
    if payload_digest(profile) != PROFILE_SHA256:
        raise ValueError("profile payload commitment mismatch")

    source = profile.get("source_binding", {})
    if source.get("repository") != "Luminous-Dynamics/mycelix":
        raise ValueError("repository drift")
    if source.get("production_subject_sha") != PRODUCTION_SUBJECT:
        raise ValueError("production subject drift")
    files = {
        item.get("path"): item.get("git_blob_sha1")
        for item in source.get("files", [])
        if isinstance(item, dict)
    }
    if files != EXPECTED_FILES:
        raise ValueError("source blob binding drift")

    declared = profile.get("declared_design", {})
    if declared.get("evidence_class") != "DeclaredDesign":
        raise ValueError("declared design evidence class drift")
    if declared.get("claimed_authorization") != [
        "ProposalExists", "ProposalApproved", "ProposalTypeConstitutional"
    ]:
        raise ValueError("declared authorization drift")

    observed = profile.get("observed_update_predicates", {})
    required_observed = {
        "proposal_record_requirement": "SomeRecord",
        "proposal_status_check": "NoneObserved",
        "proposal_type_check": "NoneObserved",
        "exact_action_binding": "NoneObserved",
        "caller_role_binding": "NoneObserved",
        "execution_signature_receipt_binding": "NoneObserved",
    }
    for key, expected in required_observed.items():
        if observed.get(key) != expected:
            raise ValueError(f"observed predicate drift: {key}")

    integrity = profile.get("integrity_authorization", {})
    if integrity.get("create_update_validator") != "check_consciousness_config":
        raise ValueError("integrity validator drift")
    if integrity.get("changed_by_proposal_binding") != "NoneObserved":
        raise ValueError("proposal binding drift")
    if integrity.get("entry_author_authorization") != "NoneObserved":
        raise ValueError("entry author authorization drift")

    effect = profile.get("policy_effect_observation", {}).get(
        "lower_gate_structurally_valid_test", {}
    )
    if effect != {
        "field": "consciousness_gate_basic",
        "value": 0.1,
        "result": "ValidUnderCheckConsciousnessConfig",
    }:
        raise ValueError("lower-gate observation drift")

    gaps = profile.get("known_gaps", [])
    if gaps != [{"issue": 943, "status": "Observed", "class": "AuthorizationPredicateGap"}]:
        raise ValueError("known gap drift")


def build_corpus(profile: dict) -> dict:
    validate_profile(profile)
    defaults = {"basic": 0.2, "proposal": 0.3, "voting": 0.4, "constitutional": 0.6}
    candidate = {"basic": 0.1, "proposal": 0.3, "voting": 0.4, "constitutional": 0.6}
    if not shape_valid(defaults) or not shape_valid(candidate):
        raise ValueError("gate fixture shape invalid")
    if not candidate["basic"] < defaults["basic"]:
        raise ValueError("gate fixture no longer lowers basic threshold")

    payload = {
        "schema": SCHEMA,
        "authority": "MeasurementOnly",
        "profile_ref": {
            "id": PROFILE_ID,
            "revision": 1,
            "content_sha256": PROFILE_SHA256,
        },
        "issue": 943,
        "fixtures": [
            {
                "id": "CE-CFG-01",
                "revision": 1,
                "classification": "SourceContractCounterexample",
                "premises": {
                    "proposal_id": "MIP-DRAFT-FIXTURE",
                    "modeled_proposal_state": "Draft",
                    "proposal_lookup_result": "SomeRecord",
                    "proposal_status_inspected": False,
                    "proposal_type_inspected": False,
                },
                "result": "AuthorizationContinuesAfterExistenceOnly",
                "non_claim": "NoLiveMutationExecuted",
            },
            {
                "id": "CE-CFG-02",
                "revision": 1,
                "classification": "PurePolicyEffectCounterexample",
                "premises": {
                    "default_gates": defaults,
                    "candidate_gates": candidate,
                    "required_shape": ["Finite", "UnitInterval", "Nondecreasing"],
                },
                "result": "StructurallyValidLowerRuntimeGate",
                "non_claim": "NoNormativeThresholdVerdict",
            },
            {
                "id": "CE-CFG-03",
                "revision": 1,
                "classification": "IntegrityAuthorityCounterexample",
                "premises": {
                    "changed_by_proposal": "MIP-FIXTURE",
                    "config_shape": "StructurallyValid",
                    "integrity_validator": "check_consciousness_config",
                    "proposal_authority_reconstruction": "NoneObserved",
                    "entry_author_authorization": "NoneObserved",
                },
                "result": "IntegrityAcceptsShapeWithoutObservedProposalAuthorityPredicate",
                "non_claim": "NoLiveDHTMutationExecuted",
            },
        ],
        "non_claims": [
            "no_live_config_mutation",
            "no_exploit_success",
            "no_deployment_currentness",
            "no_normative_threshold_verdict",
        ],
    }
    digest = hashlib.sha256(canonical(payload)).hexdigest()
    if digest != CORPUS_SHA256:
        raise ValueError(f"independent corpus commitment drift: {digest}")
    return {**payload, "corpus_sha256": digest}


def self_test(profile: dict) -> dict:
    corpus = build_corpus(profile)
    fixtures = {f["id"]: f for f in corpus["fixtures"]}
    if fixtures["CE-CFG-01"]["result"] != "AuthorizationContinuesAfterExistenceOnly":
        raise AssertionError("CE-CFG-01 drift")
    if fixtures["CE-CFG-02"]["result"] != "StructurallyValidLowerRuntimeGate":
        raise AssertionError("CE-CFG-02 drift")
    if fixtures["CE-CFG-03"]["result"] != "IntegrityAcceptsShapeWithoutObservedProposalAuthorityPredicate":
        raise AssertionError("CE-CFG-03 drift")
    return {
        "self_test": True,
        "authority": "MeasurementOnly",
        "conformance_class": "CrossImplementationConformance",
        "profile_sha256": PROFILE_SHA256,
        "corpus_sha256": corpus["corpus_sha256"],
        "counterexample_count": 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--corpus", action="store_true")
    args = parser.parse_args()
    profile = load(args.profile)
    result = self_test(profile) if args.self_test else build_corpus(profile)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
