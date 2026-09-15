#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

AUTHORITY = "MeasurementOnly"
CONFORMANCE = "CrossImplementationConformance"
PROFILE_ID = "mycelix-execution-observed-fca2c107-v1"
PROFILE_SHA256 = "c977bdcef9e5faac83351050999451432b618d5cc523bece804eba5dd1ae81f6"
MYCELIX_EVIDENCE_HEAD = "197714209c60503f0fba4143409da383bc9cbf83"
MYCELIX_PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
COORDINATOR_BLOB = "3dbb8a8f69b377e494ccf24164c94bd80f54e0ef"
INTEGRITY_BLOB = "657edaee9a314f100a0c4b1609a4596cf243e61d"
EXPECTED_CORPUS_SHA256 = "0c6669e44d6d18396ede43324f5cf3abbb25ddd3a2a9f59abb2c8a3699ba5fd4"
SCHEMA = "mycelix-observed-execution-counterexamples-v1"


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("profile root must be object")
    return obj


def validate_profile(profile: dict) -> dict:
    if profile.get("schema") != "mycelix-governance-observed-execution-profile-v1":
        raise ValueError("unexpected execution profile schema")
    if profile.get("authority_class") != "ObservedSourceBound":
        raise ValueError("execution profile authority drift")
    if profile.get("profile_id") != PROFILE_ID or profile.get("profile_revision") != 1:
        raise ValueError("unexpected profile identity")

    payload = copy.deepcopy(profile)
    claimed = payload.pop("profile_content_sha256", None)
    actual = digest(payload)
    if claimed != actual or actual != PROFILE_SHA256:
        raise ValueError("profile commitment mismatch")

    source = profile["source_binding"]
    if source.get("repository") != "Luminous-Dynamics/mycelix":
        raise ValueError("unexpected repository")
    if source.get("production_subject_sha") != MYCELIX_PRODUCTION_SUBJECT:
        raise ValueError("unexpected production subject")
    files = {x.get("path"): x.get("git_blob_sha1") for x in source.get("files", [])}
    if files != {
        "mycelix-governance/zomes/execution/coordinator/src/lib.rs": COORDINATOR_BLOB,
        "mycelix-governance/zomes/execution/integrity/src/lib.rs": INTEGRITY_BLOB,
    }:
        raise ValueError("unexpected execution source bindings")

    creation = profile["timelock_creation"]
    if creation.get("proposal_id_source") != "CallerSupplied":
        raise ValueError("proposal-id source drift")
    if creation.get("actions_source") != "CallerSupplied":
        raise ValueError("action source drift")
    if creation.get("duration_hours_source") != "CallerSupplied":
        raise ValueError("duration source drift")
    for key in ("proposal_lookup", "proposal_status_binding", "proposal_actions_binding", "policy_duration_binding"):
        if creation.get(key) != "NoneObserved":
            raise ValueError(f"unexpected creation authority predicate: {key}")

    ready = profile["readiness_transition"]
    if ready != {
        "entrypoint":"mark_timelock_ready",
        "required_source_status":"Pending",
        "caller_rule":"TimelockCreatorOnly",
        "threshold_signature_verification":"NoneObserved",
        "target_status":"Ready",
    }:
        raise ValueError("readiness observation drift")

    execution = profile["execution"]
    expected = {
        "entrypoint":"execute_timelock",
        "expiry_required":True,
        "executor_identity_rule":"ExecutorDidMustMatchCaller",
        "ready_signature_policy":"TrustReadyStateNoSignatureLookup",
        "pending_signature_policy":"LookupThresholdSignatureIfAvailableElseWarnAndContinue",
        "unavailable_signing_authority":"WarningOnlyExecutionContinues",
        "action_source":"TimelockStoredActions",
    }
    if execution != expected:
        raise ValueError("execution policy observation drift")

    dispatch = profile["action_dispatch"]
    if dispatch["TransferCredits"]["target"] != "governance_bridge::transfer_credits":
        raise ValueError("credit dispatch drift")
    if dispatch["UpdateParameter"]["target"] != "constitution::update_parameter":
        raise ValueError("parameter dispatch drift")
    if dispatch["EmitEvent"]["target"] != "emit_signal":
        raise ValueError("event dispatch drift")

    return {"profile_id": PROFILE_ID, "content_sha256": PROFILE_SHA256, "authority_class":"ObservedSourceBound"}


def build_corpus(profile: dict) -> dict:
    ref = validate_profile(profile)
    creation = profile["timelock_creation"]
    ready = profile["readiness_transition"]
    execution = profile["execution"]
    dispatch = profile["action_dispatch"]

    corpus = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "profile": ref,
        "counterexamples": [
            {
                "id":"CE-TL-01","revision":1,"kind":"AuthorityGapCounterfactual","issue":904,
                "inputs":{
                    "proposal_id":"MIP-UNBOUND-FIXTURE",
                    "actions":[{"type":"EmitEvent","event":"fixture","payload":{}}],
                    "duration_hours":1,
                },
                "comparison":{
                    "shape_checks":"Satisfied",
                    "proposal_lookup_predicate":creation["proposal_lookup"],
                    "proposal_status_predicate":creation["proposal_status_binding"],
                    "proposal_actions_binding_predicate":creation["proposal_actions_binding"],
                    "policy_duration_binding_predicate":creation["policy_duration_binding"],
                },
                "non_claim":"Shows absence of source-visible authorization predicates in timelock construction; does not execute the fixture.",
            },
            {
                "id":"CE-TL-02","revision":1,"kind":"AuthorityGapCounterfactual","issue":904,
                "inputs":{"source_status":"Pending","caller_relation":"TimelockCreator"},
                "comparison":{
                    "creator_check":"Satisfied",
                    "source_status_check":"Satisfied",
                    "threshold_signature_predicate":ready["threshold_signature_verification"],
                    "observed_target_status":ready["target_status"],
                },
                "non_claim":"Models local predicates of mark_timelock_ready; does not claim a live transition was performed.",
            },
            {
                "id":"CE-TL-03","revision":1,"kind":"ControlFlowDifferential","issue":904,
                "inputs":{"expired":True},
                "comparison":{
                    "Ready":{"threshold_signature_lookup":"NoneInBranch","execution_authority_assumption":"ReadyImpliesPreviouslyVerified"},
                    "Pending":{"threshold_signature_lookup":"Attempted","unavailable_authority_behavior":"WarnAndContinue"},
                },
                "non_claim":"Records source-control-flow differences, not an end-to-end exploit.",
            },
            {
                "id":"CE-TL-04","revision":1,"kind":"ExecutableSurfaceObservation","issue":904,
                "inputs":{"action_source":execution["action_source"]},
                "comparison":{
                    "TransferCredits":dispatch["TransferCredits"]["target"],
                    "UpdateParameter":dispatch["UpdateParameter"]["target"],
                    "EmitEvent":dispatch["EmitEvent"]["target"],
                },
                "non_claim":"Records dispatch reachability from timelock action parsing; downstream authorization remains unqualified.",
            },
            {
                "id":"CE-TL-05","revision":1,"kind":"FailOpenAuthorityObservation","issue":904,
                "inputs":{"source_status":"Pending","expired":True,"threshold_signing":"Unavailable"},
                "comparison":{
                    "warning":"threshold_signing_unavailable",
                    "signature_verification":"NotEstablished",
                    "source_control_flow":"ExecutionContinuesAfterWarning",
                },
                "non_claim":"Records the explicit graceful-degradation branch; does not execute governance actions.",
            },
        ],
        "non_claims":[
            "no_live_exploit_claim","no_downstream_authorization_claim","no_financial_mutation",
            "no_constitutional_mutation","no_deployment_currentness","no_governance_safety_claim"
        ],
    }
    corpus["corpus_sha256"] = digest(corpus)
    return corpus


def self_test(profile: dict) -> dict:
    a = build_corpus(profile)
    b = build_corpus(profile)
    if canonical(a) != canonical(b):
        raise AssertionError("non-deterministic execution corpus")
    if a["corpus_sha256"] != EXPECTED_CORPUS_SHA256:
        raise AssertionError("cross-implementation execution corpus mismatch")
    ids = {x["id"] for x in a["counterexamples"]}
    assert ids == {"CE-TL-01","CE-TL-02","CE-TL-03","CE-TL-04","CE-TL-05"}
    return {
        "authority":AUTHORITY,
        "conformance_class":CONFORMANCE,
        "mycelix_evidence_head":MYCELIX_EVIDENCE_HEAD,
        "mycelix_production_subject":MYCELIX_PRODUCTION_SUBJECT,
        "profile_sha256":PROFILE_SHA256,
        "corpus_sha256":EXPECTED_CORPUS_SHA256,
        "counterexample_count":5,
        "self_test":True,
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
