#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

V4_ID = "mycelix-observed-composite-fca2c107-v4"
V4_SHA = "993a17b4ab1e876beb7815f8e90fc137b8776c5dcce3b0d9e30da971b13a797f"
V5_ID = "mycelix-observed-composite-fca2c107-v5"
V5_SHA = "a63f9cd8a72824708925d6938ff6cffa2efda614794259cfe86189c27a4b44fe"
SUB_ID = "mycelix-constitution-runtime-policy-downstream-observed-fca2c107-v1"
SUB_SHA = "157402e76abffd3849dc6d2001a7c0bbde296e80ab6a7f5c77c8f3e288302183"
SUB_HEAD = "af73f6f0601c5ea3a2935fc62573078ce27a712f"
PRODUCTION = "fca2c107a1ea5108823ce617ba4111b6f7f77230"

EXPECTED_UNESTABLISHED = {
    "AuthorizedConstitutionParameterMutation",
    "AtomicConstitutionRuntimePolicyMutation",
    "ConstitutionRuntimeConfigSynchronization",
    "ContentBoundReconciliation",
    "AuthorizedGovernanceConfigMutation",
    "RuntimeGovernanceConfigCurrentness",
    "DeploymentCurrentnessQualified",
    "GovernanceSafety",
}
EXPECTED_EXCLUSIONS = {
    "charter_creation_and_currentness",
    "constitutional_amendment_lifecycle_and_application",
    "enhanced_immutable_core_amendment_requirements",
}
EXPECTED_UNCOVERED = [
    "treasury_credit_authorization_downstream",
    "deployment_currentness",
]
EXPECTED_REQUIRED_V5 = [
    "proposal_creation_and_lifecycle",
    "voting_and_tally",
    "delegation_and_participation_identity",
    "threshold_signing_authority",
    "timelock_and_readiness",
    "execution_dispatch",
    "constitution_runtime_policy_downstream",
    "treasury_credit_authorization_downstream",
    "deployment_currentness",
]
EXPECTED_CORRECTION = [{
    "predecessor_stage": "constitution_parameter_authorization_downstream",
    "successor_stage": "constitution_runtime_policy_downstream",
    "correction_class": "ScopeRefinementNotHistoricalInvalidation",
    "reason": (
        "ConstitutionParameter representation alone does not represent "
        "ConstitutionParameter-to-runtime synchronization and runtime "
        "GovernanceConfig mutation semantics."
    ),
}]
EXPECTED_SUB_REF = [{
    "role": "ConstitutionRuntimePolicyDownstream",
    "manifest_id": SUB_ID,
    "manifest_sha256": SUB_SHA,
    "production_subject": PRODUCTION,
    "authority_class": "ObservedCompositeSubslice",
    "claim_ceiling": "ObservedConstitutionRuntimePolicySliceOnly",
    "symthaea_evidence_head": SUB_HEAD,
    "symthaea_pr": 3392,
    "known_issues": [943, 944, 1002],
}]
REQUIRED_NONCLAIMS = {
    "no_observed_end_to_end_claim",
    "no_complete_constitution_or_charter_amendment_coverage",
    "no_authorized_constitution_parameter_mutation_claim",
    "no_runtime_config_authorization_claim",
    "no_runtime_sync_success_claim",
    "no_content_bound_reconciliation_claim",
    "no_deployment_currentness",
    "no_governance_safety_claim",
    "no_fairness_claim",
    "no_constitutional_legitimacy_claim",
}
FORBIDDEN_KEYS = {
    "observed_end_to_end",
    "complete_constitution_coverage",
    "charter_amendments_covered",
    "runtime_sync_established",
    "authorized_runtime_config",
    "deployment_current",
    "governance_safe",
    "constitutionally_legitimate",
}


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def digest(obj: dict, field: str) -> str:
    payload = copy.deepcopy(obj)
    payload.pop(field, None)
    return hashlib.sha256(canonical(payload)).hexdigest()


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: root must be object")
    return obj


def walk_forbidden(obj: object, path: str = "$") -> None:
    if isinstance(obj, dict):
        bad = FORBIDDEN_KEYS.intersection(obj)
        if bad:
            raise ValueError(f"forbidden verdict fields at {path}: {sorted(bad)}")
        for k, v in obj.items():
            walk_forbidden(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            walk_forbidden(v, f"{path}[{i}]")


def validate_v4(v4: dict) -> None:
    if v4.get("schema") != "symthaea-composite-mechanism-manifest-v4":
        raise ValueError("v4 schema drift")
    if v4.get("manifest_id") != V4_ID or v4.get("revision") != 4:
        raise ValueError("v4 identity drift")
    if v4.get("manifest_content_sha256") != V4_SHA or digest(v4, "manifest_content_sha256") != V4_SHA:
        raise ValueError("v4 content commitment drift")
    if v4.get("production_subject") != PRODUCTION:
        raise ValueError("v4 production subject drift")
    if v4.get("authority_class") != "ObservedCompositeSlice":
        raise ValueError("v4 authority drift")
    if v4.get("claim_ceiling") != "ObservedCompositeSliceOnly":
        raise ValueError("v4 ceiling drift")


def validate_submanifest(sub: dict) -> None:
    if sub.get("schema") != "symthaea-constitution-runtime-policy-downstream-manifest-v1":
        raise ValueError("submanifest schema drift")
    if sub.get("manifest_id") != SUB_ID or sub.get("revision") != 1:
        raise ValueError("submanifest identity drift")
    if sub.get("manifest_content_sha256") != SUB_SHA or digest(sub, "manifest_content_sha256") != SUB_SHA:
        raise ValueError("submanifest content commitment drift")
    if sub.get("production_subject") != PRODUCTION:
        raise ValueError("submanifest production subject drift")
    if sub.get("authority_class") != "ObservedCompositeSubslice":
        raise ValueError("submanifest authority drift")
    if sub.get("claim_ceiling") != "ObservedConstitutionRuntimePolicySliceOnly":
        raise ValueError("submanifest ceiling drift")
    if set(sub.get("unestablished_properties", [])) != EXPECTED_UNESTABLISHED:
        raise ValueError("submanifest unresolved-property set drift")


def validate(v4: dict, sub: dict, v5: dict) -> dict:
    validate_v4(v4)
    validate_submanifest(sub)
    walk_forbidden(v5)

    if v5.get("schema") != "symthaea-composite-mechanism-manifest-v5":
        raise ValueError("v5 schema drift")
    if v5.get("manifest_id") != V5_ID or v5.get("revision") != 5:
        raise ValueError("v5 identity drift")
    if v5.get("production_subject") != PRODUCTION:
        raise ValueError("v5 production subject drift")
    if v5.get("authority_class") != "ObservedCompositeSlice":
        raise ValueError("v5 authority promotion")
    if v5.get("claim_ceiling") != "ObservedCompositeSliceOnly":
        raise ValueError("v5 claim-ceiling promotion")
    if v5.get("coverage_definition") != "SourceObservedMechanismRepresentedNotPropertySatisfied":
        raise ValueError("v5 coverage semantics drift")

    if v5.get("predecessor_manifest") != {
        "manifest_id": V4_ID,
        "revision": 4,
        "manifest_content_sha256": V4_SHA,
    }:
        raise ValueError("v4 predecessor binding drift")

    if v5.get("components") != v4.get("components"):
        raise ValueError("v4 direct component set/content changed in v5")
    if v5.get("composed_submanifests") != EXPECTED_SUB_REF:
        raise ValueError("constitutional downstream submanifest ref drift")

    v4_covered = v4.get("covered_stages")
    v5_covered = v5.get("covered_stages")
    if not isinstance(v4_covered, list) or not isinstance(v5_covered, list):
        raise ValueError("covered stages must be lists")
    if v5_covered != v4_covered + ["constitution_runtime_policy_downstream_observed_slice"]:
        raise ValueError("v5 covered-stage delta is not exactly one submanifest slice")

    if v5.get("taxonomy_corrections") != EXPECTED_CORRECTION:
        raise ValueError("taxonomy correction drift")
    if v5.get("required_stages_for_end_to_end") != EXPECTED_REQUIRED_V5:
        raise ValueError("v5 required-stage registry drift")
    if v5.get("uncovered_stages") != EXPECTED_UNCOVERED:
        raise ValueError("v5 uncovered-stage set drift")
    if set(v5.get("represented_but_unestablished_properties", [])) != EXPECTED_UNESTABLISHED:
        raise ValueError("represented-but-unestablished property set drift")
    if set(v5.get("scope_exclusions", [])) != EXPECTED_EXCLUSIONS:
        raise ValueError("scope-exclusion set drift")
    if not REQUIRED_NONCLAIMS.issubset(set(v5.get("non_claims", []))):
        raise ValueError("required v5 nonclaims missing")

    # V4's old stage label must not survive as the canonical v5 registry item.
    if "constitution_parameter_authorization_downstream" in v5.get("required_stages_for_end_to_end", []):
        raise ValueError("superseded v4 downstream stage remains canonical in v5")

    # The nested submanifest is evidence authority for the refined stage; do not
    # flatten its three mechanisms into new v5 direct components.
    if len(v5.get("components", [])) != len(v4.get("components", [])):
        raise ValueError("v5 flattened nested constitutional components")

    actual = digest(v5, "manifest_content_sha256")
    if v5.get("manifest_content_sha256") != actual or actual != V5_SHA:
        raise ValueError(f"v5 content commitment drift: {actual}")

    return {
        "validated": True,
        "manifest_id": V5_ID,
        "revision": 5,
        "manifest_content_sha256": actual,
        "authority_class": "ObservedCompositeSlice",
        "claim_ceiling": "ObservedCompositeSliceOnly",
        "production_subject": PRODUCTION,
        "direct_component_count": len(v5["components"]),
        "submanifest_count": len(v5["composed_submanifests"]),
        "represented_but_unestablished_property_count": len(EXPECTED_UNESTABLISHED),
        "uncovered_stages": EXPECTED_UNCOVERED,
        "taxonomy_correction_class": "ScopeRefinementNotHistoricalInvalidation",
    }


def self_test(v4: dict, sub: dict, v5: dict) -> dict:
    result = validate(v4, sub, v5)
    baseline = digest(v5, "manifest_content_sha256")

    def changes(mutator) -> None:
        c = copy.deepcopy(v5)
        c.pop("manifest_content_sha256", None)
        mutator(c)
        if hashlib.sha256(canonical(c)).hexdigest() == baseline:
            raise AssertionError("semantic v5 mutation did not change identity")

    changes(lambda x: x["components"].pop())
    changes(lambda x: x["represented_but_unestablished_properties"].remove("GovernanceSafety"))
    changes(lambda x: x["scope_exclusions"].remove("constitutional_amendment_lifecycle_and_application"))
    changes(lambda x: x["taxonomy_corrections"][0].update(correction_class="HistoricalInvalidation"))

    invalid = [
        lambda x: x.update(authority_class="ObservedEndToEnd"),
        lambda x: x.update(claim_ceiling="GovernanceSafetyQualified"),
        lambda x: x["composed_submanifests"].clear(),
        lambda x: x["uncovered_stages"].clear(),
        lambda x: x.update(complete_constitution_coverage=True),
    ]
    for mutate in invalid:
        c = copy.deepcopy(v5)
        mutate(c)
        try:
            validate(v4, sub, c)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid v5 mutation accepted")

    return {**result, "self_test": True}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v4", type=Path, required=True)
    p.add_argument("--submanifest", type=Path, required=True)
    p.add_argument("--v5", type=Path, required=True)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    v4 = load(args.v4)
    sub = load(args.submanifest)
    v5 = load(args.v5)
    result = self_test(v4, sub, v5) if args.self_test else validate(v4, sub, v5)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
