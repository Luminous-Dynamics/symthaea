#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SCHEMA = "symthaea-observed-mechanism-manifest-v1"
MANIFEST_ID = "mycelix-governance-observed-fca2c107-bundle-v1"
MANIFEST_SHA256 = "a7bfe0a285dad6c83d2ac82a9edc7bba0c2a300532cba24a604411ca6c8e4e35"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
EXPECTED_COMPONENTS = {
    "voting": {
        "profile": {
            "id": "mycelix-voting-observed-fca2c107-v2",
            "revision": 2,
            "content_sha256": "680af4668889c299b7e0d74531f44894a64a384be71778bca54d3f21ca80ac01",
            "authority": "ObservedSourceBound",
        },
        "corpus_sha256": "ee5e7649a773f564b443320689f465080d4641a0f4f09f13c0c49a7087d6dc10",
        "evidence_head": "4b4e27910a98ad393e5a508e54e0f73c48c19107",
        "conformance_contract": "IG-008A0",
    },
    "execution": {
        "profile": {
            "id": "mycelix-execution-observed-fca2c107-v1",
            "revision": 1,
            "content_sha256": "c977bdcef9e5faac83351050999451432b618d5cc523bece804eba5dd1ae81f6",
            "authority": "ObservedSourceBound",
        },
        "corpus_sha256": "0c6669e44d6d18396ede43324f5cf3abbb25ddd3a2a9f59abb2c8a3699ba5fd4",
        "evidence_head": "197714209c60503f0fba4143409da383bc9cbf83",
        "conformance_contract": "IG-008E0",
    },
    "governance_config": {
        "profile": {
            "id": "mycelix-governance-config-observed-fca2c107-v1",
            "revision": 1,
            "content_sha256": "de4435a69356557b1812f8beb46d654c66b9c957be9d18c64bd0431f92546d5a",
            "authority": "ObservedSourceBound",
        },
        "corpus_sha256": "3009e97529934fa8f470769de5615dcfda17bde0e942683e939d2b733430a216",
        "evidence_head": "88922e7950ddf030026b5d4d2b06e0ed63727692",
        "conformance_contract": "IG-008C0",
    },
}
EXPECTED_SOURCE_BOUND = [
    "DirectAndDelegatedVotingObservedSurface",
    "VotingTallyTierAndDelegationObservedSemantics",
    "TimelockCreationReadinessSignatureBranchAndDispatchObservedSurface",
    "RuntimeGovernanceConsciousnessConfigReadAndMutationAuthorityObservedSurface",
]
EXPECTED_NOT_ESTABLISHED = [
    "CompleteProposalLifecycleAuthorization",
    "ThresholdSigningCryptographicAndCommitteeCorrectness",
    "CompleteCouncilAndGuardianAuthority",
    "ConstitutionAmendmentProcessCorrectness",
    "ActualFinanceAndFundAllocationMoneyMovement",
    "DownstreamFinanceCommonsAndCivicAuthorization",
    "IdentityDIDAndSybilSystemCorrectness",
    "CrossDNAAndNetworkFailureSemanticsBeyondFrozenObservations",
    "DeploymentCurrentness",
    "HumanAndAIBehavioralValidity",
    "GovernanceSafetyFairnessAndLegitimacy",
    "CrossComponentCausalCorrectnessBeyondExplicitlyModeledBindings",
]
EXPECTED_ISSUES = [851, 855, 856, 876, 877, 892, 900, 904, 943, 944]
TOP_KEYS = {
    "schema", "authority_class", "composition_class", "manifest_id",
    "manifest_revision", "production_subject", "components", "coverage",
    "composition_semantics", "required_conformance_contracts",
    "known_open_issues", "non_claims", "manifest_content_sha256",
}
FORBIDDEN_KEYS = {
    "end_to_end_qualified", "governance_safe", "mechanism_safe", "fair",
    "legitimate", "deployment_current", "behaviorally_valid", "complete",
}


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def payload_digest(manifest: dict) -> str:
    payload = copy.deepcopy(manifest)
    payload.pop("manifest_content_sha256", None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("manifest root must be object")
    return obj


def forbid_verdict_fields(obj: object, path: str = "$") -> None:
    if isinstance(obj, dict):
        bad = FORBIDDEN_KEYS.intersection(obj)
        if bad:
            raise ValueError(f"forbidden verdict field(s) at {path}: {sorted(bad)}")
        for key, value in obj.items():
            forbid_verdict_fields(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            forbid_verdict_fields(value, f"{path}[{index}]")


def validate(manifest: dict) -> dict:
    if set(manifest) != TOP_KEYS:
        raise ValueError("top-level manifest schema drift")
    if manifest["schema"] != SCHEMA:
        raise ValueError("schema drift")
    if manifest["manifest_id"] != MANIFEST_ID or manifest["manifest_revision"] != 1:
        raise ValueError("manifest identity drift")
    if manifest["authority_class"] != "MeasurementOnly":
        raise ValueError("static authority must remain MeasurementOnly")
    if manifest["composition_class"] != "PreparedComposition":
        raise ValueError("static composition cannot self-promote")
    forbid_verdict_fields(manifest)

    if manifest["production_subject"] != {
        "repository": "Luminous-Dynamics/mycelix",
        "sha": PRODUCTION_SUBJECT,
    }:
        raise ValueError("production subject drift")

    if manifest["components"] != EXPECTED_COMPONENTS:
        raise ValueError("component reference drift")
    for name, component in manifest["components"].items():
        if component["profile"]["authority"] != "ObservedSourceBound":
            raise ValueError(f"component authority drift: {name}")

    coverage = manifest["coverage"]
    if coverage.get("source_bound") != EXPECTED_SOURCE_BOUND:
        raise ValueError("source-bound coverage drift")
    if coverage.get("not_established") != EXPECTED_NOT_ESTABLISHED:
        raise ValueError("unmodeled coverage drift")
    if not coverage["not_established"]:
        raise ValueError("unmodeled coverage must remain explicit")

    semantics = manifest["composition_semantics"]
    if semantics != {
        "same_production_subject_required": True,
        "component_authority_required": "ObservedSourceBound",
        "cross_component_causality": "NotEstablishedByManifest",
        "end_to_end_completeness": "NotClaimed",
        "deployment_currentness": "Unqualified",
        "qualification_requires_all_component_conformance": True,
        "qualified_composition_ceiling": "ComposedCrossImplementationConformance",
    }:
        raise ValueError("composition semantics drift")

    if manifest["required_conformance_contracts"] != ["IG-008A0", "IG-008E0", "IG-008C0"]:
        raise ValueError("required conformance contract drift")
    if manifest["known_open_issues"] != EXPECTED_ISSUES:
        raise ValueError("known issue set drift")
    if manifest["non_claims"] != [
        "no_end_to_end_governance_qualification",
        "no_deployment_currentness",
        "no_mechanism_safety_claim",
        "no_fairness_or_legitimacy_claim",
        "no_behavioral_validity_claim",
    ]:
        raise ValueError("non-claim boundary drift")

    actual = payload_digest(manifest)
    if manifest["manifest_content_sha256"] != actual:
        raise ValueError(f"manifest commitment mismatch: {actual}")
    if actual != MANIFEST_SHA256:
        raise ValueError("manifest differs from frozen commitment")

    return {
        "validated": True,
        "authority_class": "MeasurementOnly",
        "composition_class": "PreparedComposition",
        "manifest_id": MANIFEST_ID,
        "manifest_revision": 1,
        "manifest_content_sha256": actual,
        "component_count": 3,
        "unmodeled_surface_count": len(EXPECTED_NOT_ESTABLISHED),
        "qualification_ceiling": "ComposedCrossImplementationConformance",
    }


def self_test(manifest: dict) -> dict:
    result = validate(manifest)
    baseline = payload_digest(manifest)

    def identity_changes(mutator) -> None:
        candidate = copy.deepcopy(manifest)
        candidate.pop("manifest_content_sha256", None)
        mutator(candidate)
        digest = hashlib.sha256(canonical(candidate)).hexdigest()
        if digest == baseline:
            raise AssertionError("semantic manifest mutation did not change identity")

    identity_changes(lambda m: m["components"]["voting"]["profile"].update(content_sha256="0" * 64))
    identity_changes(lambda m: m["components"]["execution"].update(evidence_head="0" * 40))
    identity_changes(lambda m: m["coverage"]["not_established"].pop())
    identity_changes(lambda m: m["known_open_issues"].append(999999))

    invalid = [
        lambda m: m.update(composition_class="EndToEndGovernanceQualified"),
        lambda m: m.update(authority_class="ExecutableQualified"),
        lambda m: m["coverage"].update(not_established=[]),
        lambda m: m["composition_semantics"].update(end_to_end_completeness="Established"),
        lambda m: m.update(governance_safe=True),
    ]
    for mutator in invalid:
        candidate = copy.deepcopy(manifest)
        mutator(candidate)
        try:
            validate(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid composition mutation accepted")

    return {**result, "self_test": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    manifest = load(args.manifest)
    result = self_test(manifest) if args.self_test else validate(manifest)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
