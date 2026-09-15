#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

AUTHORITY = "MeasurementOnly"
CONFORMANCE = "CrossImplementationConformance"
SCHEMA = "mycelix-constitution-bridge-sync-counterexamples-v1"
PROFILE_ID = "mycelix-constitution-bridge-sync-observed-fca2c107-v1"
PROFILE_SHA256 = "60daae86044098561fa8e41bcdf6f695ab41234760be4b7e235d2780271b681e"
EXPECTED_CORPUS_SHA256 = "2ca6d79212cfd0acae1e974c8390d564bab06821050d8eacbc32f437630ef60b"
MYCELIX_EVIDENCE_HEAD = "abb277cc39418e43e80d48e8a4a57bc7c939d9e4"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
SAME_TREE_MAIN = "31ede2365b81365bb119cd9351b2739119974130"

CONSTITUTION = {
    "path": "mycelix-governance/zomes/constitution/coordinator/src/lib.rs",
    "git_blob_sha1": "923a1ce789c8319c79df7f33a9241af50804ec55",
}
BRIDGE = [
    {"path":"mycelix-governance/zomes/bridge/coordinator/src/attestation.rs","git_blob_sha1":"6d7938084ba699b144ee5951966b1421c034f579"},
    {"path":"mycelix-governance/zomes/bridge/coordinator/src/consciousness.rs","git_blob_sha1":"d0acabf306594ab3ccfae220a9c9bd9aceed0ca6"},
    {"path":"mycelix-governance/zomes/bridge/coordinator/src/consciousness_config.rs","git_blob_sha1":"26da234e588bf26d0e25c10dbec34502e00c191a"},
    {"path":"mycelix-governance/zomes/bridge/coordinator/src/consensus.rs","git_blob_sha1":"3842dfa365953a01ca90bb79059da53f9cb00a6f"},
    {"path":"mycelix-governance/zomes/bridge/coordinator/src/cross_cluster.rs","git_blob_sha1":"3eb0ade8633d6e711fd266bca6eb2ebab616eac2"},
    {"path":"mycelix-governance/zomes/bridge/coordinator/src/lib.rs","git_blob_sha1":"fb278023c269a89f300504c18538ab85b09f1178"},
    {"path":"mycelix-governance/zomes/bridge/coordinator/src/query.rs","git_blob_sha1":"5a02812880a2fdaa8f3ed767a4686afc00c566f9"},
    {"path":"mycelix-governance/zomes/bridge/coordinator/src/validation.rs","git_blob_sha1":"9e13ba58939880738eccb997e54f729da7a11304"},
]


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest_without_commitment(profile: dict) -> str:
    payload = copy.deepcopy(profile)
    payload.pop("profile_content_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("profile root must be object")
    return obj


def validate_profile(profile: dict) -> dict:
    if profile.get("schema") != "mycelix-constitution-bridge-sync-observed-profile-v1":
        raise ValueError("unexpected sync profile schema")
    if profile.get("profile_id") != PROFILE_ID or profile.get("profile_revision") != 1:
        raise ValueError("sync profile identity drift")
    if profile.get("authority_class") != "ObservedSourceBound":
        raise ValueError("sync profile authority drift")
    actual = digest_without_commitment(profile)
    if profile.get("profile_content_sha256") != actual or actual != PROFILE_SHA256:
        raise ValueError("sync profile commitment mismatch")

    source = profile.get("source_binding", {})
    if source.get("repository") != "Luminous-Dynamics/mycelix":
        raise ValueError("sync source repository drift")
    if source.get("production_subject_sha") != PRODUCTION_SUBJECT:
        raise ValueError("sync production subject drift")
    if source.get("tree_equivalent_current_main_sha") != SAME_TREE_MAIN:
        raise ValueError("sync same-tree history drift")
    if source.get("constitution_coordinator") != CONSTITUTION:
        raise ValueError("constitution source binding drift")
    if source.get("bridge_coordinator_modules") != BRIDGE:
        raise ValueError("bridge module census binding drift")

    sync = profile.get("constitution_sync", {})
    if sync.get("target_zome") != "governance_bridge" or sync.get("target_function") != "update_phi_config":
        raise ValueError("sync target drift")
    if sync.get("call_semantics") != "BestEffort":
        raise ValueError("sync call semantics drift")
    if sync.get("parameter_write_precedes_sync") is not True:
        raise ValueError("sync ordering drift")
    if sync.get("sync_failure_effect") != "EmitPhiConfigSyncWarningAndRetainConstitutionParameterWrite":
        raise ValueError("sync failure semantics drift")
    if sync.get("explicit_unsynchronized_state_receipt") != "NoneObserved":
        raise ValueError("unsynchronized receipt observation drift")
    if sync.get("explicit_retry_reconciliation_contract") != "NoneObserved":
        raise ValueError("reconciliation contract observation drift")

    surface = profile.get("bridge_surface", {})
    if surface.get("coordinator_module_count") != 8:
        raise ValueError("bridge module count drift")
    if surface.get("target_symbol") != "update_phi_config":
        raise ValueError("bridge target-symbol drift")
    if surface.get("target_symbol_occurrences_in_bound_census") != 0:
        raise ValueError("bridge target-symbol absence theorem drift")
    if surface.get("visible_runtime_config_updater") != "update_consciousness_config":
        raise ValueError("visible runtime updater drift")

    auth = profile.get("authorization_boundary", {})
    if auth.get("visible_runtime_updater_issue") != 943:
        raise ValueError("visible updater issue binding drift")
    if auth.get("rename_target_to_visible_updater_is_sufficient_repair") is not False:
        raise ValueError("rename-is-repair boundary drift")

    gaps = profile.get("known_gaps")
    if gaps != [
        {"issue":944,"class":"ConstitutionBridgeSyncContractMismatch","status":"Observed"},
        {"issue":943,"class":"GovernanceConfigAuthorizationGap","status":"SeparateDependency"},
    ]:
        raise ValueError("sync known-gap set drift")

    return {
        "profile_id": PROFILE_ID,
        "content_sha256": PROFILE_SHA256,
        "authority_class": "ObservedSourceBound",
    }


def build_corpus(profile: dict) -> dict:
    ref = validate_profile(profile)
    surface = profile["bridge_surface"]
    sync = profile["constitution_sync"]
    auth = profile["authorization_boundary"]

    corpus = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "profile": ref,
        "issues": [943, 944],
        "counterexamples": [
            {
                "id": "CE-CS-01",
                "revision": 1,
                "kind": "TargetEntrypointCensusObservation",
                "inputs": {
                    "target_zome": "governance_bridge",
                    "target_function": "update_phi_config",
                    "bridge_coordinator_module_count": 8,
                },
                "comparison": {
                    "target_symbol_occurrences_in_bound_census": surface[
                        "target_symbol_occurrences_in_bound_census"
                    ],
                    "visible_runtime_config_updater": surface[
                        "visible_runtime_config_updater"
                    ],
                    "result": "TargetEntrypointAbsentFromObservedBridgeCoordinatorCensus",
                },
                "non_claim": (
                    "Records the exact frozen coordinator census; does not claim every deployed "
                    "bridge build has this surface."
                ),
            },
            {
                "id": "CE-CS-02",
                "revision": 1,
                "kind": "BestEffortSynchronizationObservation",
                "inputs": {
                    "constitution_parameter_write": "SucceededBeforeSyncAttempt",
                    "bridge_sync": "UnavailableOrFailed",
                    "call_semantics": "BestEffort",
                },
                "comparison": {
                    "sync_failure_effect": sync["sync_failure_effect"],
                    "runtime_sync_established": False,
                    "result": "ConstitutionParameterSuccessDoesNotEstablishRuntimeConfigSynchronization",
                },
                "non_claim": (
                    "Models source-visible control flow only; no live constitution/runtime "
                    "divergence is asserted."
                ),
            },
            {
                "id": "CE-CS-03",
                "revision": 1,
                "kind": "CrossMechanismAuthorizationBoundary",
                "inputs": {
                    "observed_target": "update_phi_config",
                    "visible_alternative": "update_consciousness_config",
                    "visible_alternative_issue": 943,
                },
                "comparison": {
                    "rename_target_to_visible_updater_is_sufficient_repair": auth[
                        "rename_target_to_visible_updater_is_sufficient_repair"
                    ],
                    "result": "EntrypointRenameAloneWouldBypassSeparateAuthorizationTheorem",
                },
                "non_claim": (
                    "Does not claim the visible updater is unusable after its independent "
                    "authorization theorem is corrected."
                ),
            },
            {
                "id": "CE-CS-04",
                "revision": 1,
                "kind": "ReconciliationEvidenceObservation",
                "inputs": {
                    "explicit_unsynchronized_state_receipt": sync[
                        "explicit_unsynchronized_state_receipt"
                    ],
                    "explicit_retry_reconciliation_contract": sync[
                        "explicit_retry_reconciliation_contract"
                    ],
                },
                "comparison": {
                    "content_bound_reconciliation_evidence": "NotEstablished",
                    "result": "NoObservedContentBoundReconciliationEvidence",
                },
                "non_claim": (
                    "Records this frozen helper's evidence surface; does not prove operational "
                    "reconciliation never occurs elsewhere."
                ),
            },
        ],
        "non_claims": [
            "no_live_runtime_divergence_claim",
            "no_live_config_mutation_claim",
            "no_deployment_exploit",
            "no_deployment_currentness",
            "no_governance_safety_claim",
        ],
    }
    corpus["corpus_sha256"] = hashlib.sha256(canonical(corpus)).hexdigest()
    return corpus


def self_test(profile: dict) -> dict:
    first = build_corpus(profile)
    second = build_corpus(profile)
    if canonical(first) != canonical(second):
        raise AssertionError("non-deterministic independent sync corpus")
    if first["corpus_sha256"] != EXPECTED_CORPUS_SHA256:
        raise AssertionError(
            f"cross-implementation sync corpus mismatch: {first['corpus_sha256']}"
        )
    ids = {item["id"] for item in first["counterexamples"]}
    if ids != {"CE-CS-01", "CE-CS-02", "CE-CS-03", "CE-CS-04"}:
        raise AssertionError("sync counterexample roster drift")
    return {
        "authority": AUTHORITY,
        "conformance_class": CONFORMANCE,
        "mycelix_evidence_head": MYCELIX_EVIDENCE_HEAD,
        "mycelix_production_subject": PRODUCTION_SUBJECT,
        "same_tree_main": SAME_TREE_MAIN,
        "profile_sha256": PROFILE_SHA256,
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "counterexample_count": 4,
        "issues": [943, 944],
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
