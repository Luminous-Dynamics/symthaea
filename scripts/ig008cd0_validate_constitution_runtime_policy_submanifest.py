#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SCHEMA = "symthaea-constitution-runtime-policy-downstream-manifest-v1"
MANIFEST_ID = "mycelix-constitution-runtime-policy-downstream-observed-fca2c107-v1"
EXPECTED_SHA256 = "157402e76abffd3849dc6d2001a7c0bbde296e80ab6a7f5c77c8f3e288302183"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
AUTHORITY = "ObservedCompositeSubslice"
CEILING = "ObservedConstitutionRuntimePolicySliceOnly"
COVERAGE = "SourceObservedMechanismRepresentedNotPropertySatisfied"
SCOPE = "ConstitutionRuntimePolicyDownstreamV1"

EXPECTED = {
    "ConstitutionParameter": {
        "role":"ConstitutionParameter",
        "profile_id":"mycelix-constitution-parameter-observed-fca2c107-v1",
        "profile_sha256":"770552d12489df1d2cdf8b0af676b01ea9a3da21940f70ed8a71910deaa35009",
        "corpus_sha256":"b37be9d2e3fd0cbec3696a19327c26dc4ba7062ec089ad92ad99bc28a11fea8e",
        "mycelix_evidence_head":"3232d611d8833b03eba9f5412f5d7cb0cf89d4e1",
        "symthaea_conformance_head":"3f6bd4210690b79734b71bd4168ce843aeda1036",
        "symthaea_pr":3349,
        "source_authority":"ObservedSourceBound",
        "symthaea_conformance":"CrossImplementationConformance",
        "issues":[1002],
    },
    "ConstitutionBridgeSync": {
        "role":"ConstitutionBridgeSync",
        "profile_id":"mycelix-constitution-bridge-sync-observed-fca2c107-v1",
        "profile_sha256":"60daae86044098561fa8e41bcdf6f695ab41234760be4b7e235d2780271b681e",
        "corpus_sha256":"2ca6d79212cfd0acae1e974c8390d564bab06821050d8eacbc32f437630ef60b",
        "mycelix_evidence_head":"abb277cc39418e43e80d48e8a4a57bc7c939d9e4",
        "symthaea_conformance_head":"594ac1a765131d242c01b71859ff28cb2fe71b1a",
        "symthaea_pr":3389,
        "source_authority":"ObservedSourceBound",
        "symthaea_conformance":"CrossImplementationConformance",
        "issues":[943,944],
    },
    "GovernanceConfig": {
        "role":"GovernanceConfig",
        "profile_id":"mycelix-governance-config-observed-fca2c107-v1",
        "profile_sha256":"de4435a69356557b1812f8beb46d654c66b9c957be9d18c64bd0431f92546d5a",
        "corpus_sha256":"3009e97529934fa8f470769de5615dcfda17bde0e942683e939d2b733430a216",
        "mycelix_evidence_head":"88922e7950ddf030026b5d4d2b06e0ed63727692",
        "symthaea_conformance_head":"c66296a29e27633f470d18568c9a7de3fae5c624",
        "symthaea_pr":3269,
        "source_authority":"ObservedSourceBound",
        "symthaea_conformance":"CrossImplementationConformance",
        "issues":[943],
    },
}

EDGES = {
    "execution_update_parameter_to_constitution_parameter",
    "constitution_parameter_storage_and_projection",
    "constitution_parameter_to_runtime_config_sync_attempt",
    "runtime_governance_config_mutation_entrypoint",
    "runtime_governance_config_integrity_shape",
}
UNESTABLISHED = {
    "AuthorizedConstitutionParameterMutation",
    "AtomicConstitutionRuntimePolicyMutation",
    "ConstitutionRuntimeConfigSynchronization",
    "ContentBoundReconciliation",
    "AuthorizedGovernanceConfigMutation",
    "RuntimeGovernanceConfigCurrentness",
    "DeploymentCurrentnessQualified",
    "GovernanceSafety",
}
NON_CLAIMS = {
    "no_complete_constitution_or_charter_amendment_coverage",
    "no_authorized_parameter_mutation_claim",
    "no_runtime_config_authorization_claim",
    "no_runtime_sync_success_claim",
    "no_live_divergence_claim",
    "no_deployment_currentness",
    "no_governance_safety_claim",
    "no_fairness_claim",
    "no_constitutional_legitimacy_claim",
}
FORBIDDEN = {
    "complete_constitution_coverage",
    "charter_amendments_covered",
    "authorized_parameter_mutation",
    "authorized_runtime_config",
    "runtime_sync_established",
    "deployment_current",
    "governance_safe",
    "constitutionally_legitimate",
}


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(manifest: dict) -> str:
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_content_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("submanifest root must be object")
    return obj


def walk_forbidden(obj: object, path: str = "$") -> None:
    if isinstance(obj, dict):
        bad = FORBIDDEN.intersection(obj)
        if bad:
            raise ValueError(f"forbidden verdict fields at {path}: {sorted(bad)}")
        for key, value in obj.items():
            walk_forbidden(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            walk_forbidden(value, f"{path}[{i}]")


def validate(m: dict) -> dict:
    if m.get("schema") != SCHEMA or m.get("manifest_id") != MANIFEST_ID or m.get("revision") != 1:
        raise ValueError("submanifest identity drift")
    if m.get("authority_class") != AUTHORITY or m.get("claim_ceiling") != CEILING:
        raise ValueError("submanifest authority/ceiling drift")
    if m.get("coverage_definition") != COVERAGE or m.get("scope") != SCOPE:
        raise ValueError("scope or coverage semantics drift")
    if m.get("production_subject") != PRODUCTION_SUBJECT:
        raise ValueError("production subject drift")
    walk_forbidden(m)

    mechanisms = m.get("mechanisms")
    if not isinstance(mechanisms, list) or len(mechanisms) != 3:
        raise ValueError("exactly three downstream mechanisms required")
    by_role = {x.get("role"): x for x in mechanisms if isinstance(x, dict)}
    if set(by_role) != set(EXPECTED):
        raise ValueError("downstream mechanism role set drift")
    for role, expected in EXPECTED.items():
        if by_role[role] != expected:
            raise ValueError(f"{role} evidence reference drift")

    if set(m.get("covered_mechanism_edges", [])) != EDGES:
        raise ValueError("covered mechanism-edge set drift")
    if set(m.get("unestablished_properties", [])) != UNESTABLISHED:
        raise ValueError("unestablished-property set drift")
    if m.get("known_issues") != [943,944,1002]:
        raise ValueError("known issue set/order drift")
    if not NON_CLAIMS.issubset(set(m.get("non_claims", []))):
        raise ValueError("required non-claims missing")

    actual = digest(m)
    if m.get("manifest_content_sha256") != actual or actual != EXPECTED_SHA256:
        raise ValueError(f"submanifest commitment drift: {actual}")

    return {
        "validated": True,
        "authority_class": AUTHORITY,
        "claim_ceiling": CEILING,
        "coverage_definition": COVERAGE,
        "scope": SCOPE,
        "manifest_id": MANIFEST_ID,
        "manifest_content_sha256": actual,
        "production_subject": PRODUCTION_SUBJECT,
        "mechanism_count": 3,
        "known_issues": [943,944,1002],
        "unestablished_property_count": len(UNESTABLISHED),
    }


def self_test(m: dict) -> dict:
    result = validate(m)
    baseline = digest(m)

    def identity_changes(mutator) -> None:
        c = copy.deepcopy(m)
        c.pop("manifest_content_sha256", None)
        mutator(c)
        if hashlib.sha256(canonical(c)).hexdigest() == baseline:
            raise AssertionError("semantic submanifest mutation did not change identity")

    identity_changes(lambda x: x["mechanisms"][0].update(profile_sha256="0"*64))
    identity_changes(lambda x: x["covered_mechanism_edges"].append("charter_amendment_application"))
    identity_changes(lambda x: x["unestablished_properties"].remove("GovernanceSafety"))

    invalid = [
        lambda x: x.update(authority_class="ObservedEndToEnd"),
        lambda x: x.update(claim_ceiling="AuthorizedConstitutionRuntimePolicy"),
        lambda x: x["mechanisms"].pop(),
        lambda x: x["known_issues"].clear(),
        lambda x: x.update(complete_constitution_coverage=True),
    ]
    for mutate in invalid:
        c = copy.deepcopy(m)
        mutate(c)
        try:
            validate(c)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid constitutional downstream mutation accepted")
    return {**result, "self_test": True}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    m = load(args.manifest)
    result = self_test(m) if args.self_test else validate(m)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
