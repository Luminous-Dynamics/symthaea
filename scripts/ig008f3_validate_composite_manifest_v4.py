#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from ig008f2_validate_composite_manifest_v3 import validate as validate_v3

SCHEMA = "symthaea-composite-mechanism-manifest-v4"
MANIFEST_ID = "mycelix-observed-composite-fca2c107-v4"
EXPECTED_SHA256 = "993a17b4ab1e876beb7815f8e90fc137b8776c5dcce3b0d9e30da971b13a797f"
V3_ID = "mycelix-observed-composite-fca2c107-v3"
V3_SHA256 = "3acdbd29d0fd631b863fa10a10eed0877af2336536578453acdc83f095632baf"
V2_SHA256 = "e3b42dbd7dacaa2f44de8bc330e85ff7d624f291d27c4a303fbaccf57091e541"
V1_SHA256 = "366b87944e74fe01369b78a8a7ea1c34f6dee05237583a47d86e5999b93b150c"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
AUTHORITY = "ObservedCompositeSlice"
CLAIM_CEILING = "ObservedCompositeSliceOnly"
COVERAGE_DEFINITION = "SourceObservedMechanismRepresentedNotPropertySatisfied"

TOP_KEYS = {
    "schema", "manifest_id", "revision", "authority_class", "claim_ceiling",
    "coverage_definition", "production_subject", "predecessor_manifest", "components",
    "covered_stages", "uncovered_stages", "required_stages_for_end_to_end",
    "non_claims", "manifest_content_sha256",
}

CONSTITUTION_COMPONENT = {
    "role": "ConstitutionParameter",
    "profile_id": "mycelix-constitution-parameter-observed-fca2c107-v1",
    "profile_sha256": "770552d12489df1d2cdf8b0af676b01ea9a3da21940f70ed8a71910deaa35009",
    "corpus_sha256": "b37be9d2e3fd0cbec3696a19327c26dc4ba7062ec089ad92ad99bc28a11fea8e",
    "mycelix_evidence_head": "3232d611d8833b03eba9f5412f5d7cb0cf89d4e1",
    "production_subject": PRODUCTION_SUBJECT,
    "evidence_authoring_tree_equivalent_head": "31ede2365b81365bb119cd9351b2739119974130",
    "source_authority": "ObservedSourceBound",
    "symthaea_conformance": "CrossImplementationConformance",
    "symthaea_pr": 3349,
}

ADDED_COVERAGE = {
    "constitution_parameter_execution_dispatch_observed_semantics",
    "constitution_parameter_coordinator_gate_observed_semantics",
    "constitution_parameter_integrity_observed_semantics",
    "constitution_parameter_projection_observed_semantics",
}
REMOVED_UNCOVERED = "constitution_parameter_authorization_downstream"
EXPECTED_REMAINING_UNCOVERED = {
    "treasury_credit_authorization_downstream",
    "deployment_currentness",
}
REQUIRED_NON_CLAIMS = {
    "no_observed_end_to_end_claim",
    "no_authorized_constitution_parameter_mutation_claim",
    "no_authoritative_constitution_parameter_currentness_claim",
    "no_deployment_currentness",
    "no_governance_safety_claim",
    "no_fairness_claim",
    "no_constitutional_legitimacy_claim",
}
FORBIDDEN_VERDICT_KEYS = {
    "safe", "secure", "fair", "authorized_parameter_mutation",
    "authoritative_parameter_current", "fork_resolved", "observed_end_to_end",
    "deployment_current", "constitutionally_legitimate",
}


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


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


def walk_forbidden(obj: object, path: str = "$") -> None:
    if isinstance(obj, dict):
        bad = FORBIDDEN_VERDICT_KEYS.intersection(obj)
        if bad:
            raise ValueError(f"forbidden verdict fields at {path}: {sorted(bad)}")
        for key, value in obj.items():
            walk_forbidden(value, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            walk_forbidden(value, f"{path}[{i}]")


def validate(manifest: dict, predecessor_v3: dict, predecessor_v2: dict, predecessor_v1: dict) -> dict:
    v3_result = validate_v3(predecessor_v3, predecessor_v2, predecessor_v1)
    if v3_result["manifest_content_sha256"] != V3_SHA256:
        raise ValueError("unexpected v3 predecessor commitment")
    if v3_result["predecessor_manifest_sha256"] != V2_SHA256:
        raise ValueError("v3 did not bind frozen v2")
    if v3_result["root_predecessor_manifest_sha256"] != V1_SHA256:
        raise ValueError("v3 did not bind frozen v1")

    if set(manifest) != TOP_KEYS:
        raise ValueError("top-level v4 manifest drift")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected v4 schema")
    if manifest.get("manifest_id") != MANIFEST_ID or manifest.get("revision") != 4:
        raise ValueError("unexpected v4 identity")
    if manifest.get("authority_class") != AUTHORITY:
        raise ValueError("v4 authority promotion or drift")
    if manifest.get("claim_ceiling") != CLAIM_CEILING:
        raise ValueError("v4 claim ceiling drift")
    if manifest.get("coverage_definition") != COVERAGE_DEFINITION:
        raise ValueError("coverage/assurance semantic drift")
    if manifest.get("production_subject") != PRODUCTION_SUBJECT:
        raise ValueError("v4 production subject drift")
    if manifest.get("predecessor_manifest") != {
        "manifest_id": V3_ID,
        "revision": 3,
        "manifest_content_sha256": V3_SHA256,
    }:
        raise ValueError("v4 predecessor binding drift")

    walk_forbidden(manifest)

    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != 5:
        raise ValueError("v4 requires exactly five frozen components")
    by_role = {item.get("role"): item for item in components if isinstance(item, dict)}
    expected_roles = {"ProposalLifecycle", "Voting", "ThresholdSigning", "Execution", "ConstitutionParameter"}
    if set(by_role) != expected_roles:
        raise ValueError("v4 component role set drift")

    predecessor_by_role = {
        item["role"]: item
        for item in predecessor_v3.get("components", [])
        if isinstance(item, dict) and "role" in item
    }
    for retained in ("ProposalLifecycle", "Voting", "ThresholdSigning", "Execution"):
        if by_role[retained] != predecessor_by_role.get(retained):
            raise ValueError(f"retained {retained} component changed from v3")
    if by_role["ConstitutionParameter"] != CONSTITUTION_COMPONENT:
        raise ValueError("ConstitutionParameter component commitment drift")

    for role, component in by_role.items():
        if component.get("production_subject") != PRODUCTION_SUBJECT:
            raise ValueError(f"{role} mixed production subject")
        if component.get("source_authority") != "ObservedSourceBound":
            raise ValueError(f"{role} source authority promotion")
        if component.get("symthaea_conformance") != "CrossImplementationConformance":
            raise ValueError(f"{role} conformance-class drift")

    predecessor_covered = set(predecessor_v3.get("covered_stages", []))
    covered = manifest.get("covered_stages")
    if not isinstance(covered, list) or len(set(covered)) != len(covered):
        raise ValueError("invalid v4 covered-stage list")
    if set(covered) != predecessor_covered | ADDED_COVERAGE:
        raise ValueError("v4 coverage did not change by exactly the ConstitutionParameter stages")

    predecessor_uncovered = set(predecessor_v3.get("uncovered_stages", []))
    uncovered = manifest.get("uncovered_stages")
    if not isinstance(uncovered, list) or len(set(uncovered)) != len(uncovered):
        raise ValueError("invalid v4 uncovered-stage list")
    if set(uncovered) != predecessor_uncovered - {REMOVED_UNCOVERED}:
        raise ValueError("v4 uncovered set changed beyond ConstitutionParameter")
    if set(uncovered) != EXPECTED_REMAINING_UNCOVERED:
        raise ValueError("unexpected v4 remaining uncovered stages")

    if manifest.get("required_stages_for_end_to_end") != predecessor_v3.get("required_stages_for_end_to_end"):
        raise ValueError("required-stage registry drift")
    if manifest["authority_class"] == "ObservedEndToEnd" or manifest["claim_ceiling"] == "ObservedEndToEnd":
        raise ValueError("ObservedEndToEnd forbidden while required coverage remains incomplete")

    non_claims = set(manifest.get("non_claims", []))
    if not REQUIRED_NON_CLAIMS.issubset(non_claims):
        raise ValueError("required v4 non-claims missing")

    actual = payload_digest(manifest)
    if manifest.get("manifest_content_sha256") != actual:
        raise ValueError(f"v4 content commitment mismatch: {actual}")
    if actual != EXPECTED_SHA256:
        raise ValueError("v4 differs from frozen commitment")

    return {
        "validated": True,
        "authority_class": AUTHORITY,
        "claim_ceiling": CLAIM_CEILING,
        "coverage_definition": COVERAGE_DEFINITION,
        "manifest_id": MANIFEST_ID,
        "manifest_content_sha256": actual,
        "predecessor_manifest_sha256": V3_SHA256,
        "v2_manifest_sha256": V2_SHA256,
        "root_predecessor_manifest_sha256": V1_SHA256,
        "production_subject": PRODUCTION_SUBJECT,
        "component_count": 5,
        "new_component": "ConstitutionParameter",
        "covered_stage_count": len(covered),
        "uncovered_stage_count": len(uncovered),
    }


def self_test(manifest: dict, predecessor_v3: dict, predecessor_v2: dict, predecessor_v1: dict) -> dict:
    result = validate(manifest, predecessor_v3, predecessor_v2, predecessor_v1)
    baseline = payload_digest(manifest)

    def identity_changes(mutator) -> None:
        candidate = copy.deepcopy(manifest)
        candidate.pop("manifest_content_sha256", None)
        mutator(candidate)
        if hashlib.sha256(canonical(candidate)).hexdigest() == baseline:
            raise AssertionError("semantic mutation did not change v4 identity")

    identity_changes(lambda m: m["predecessor_manifest"].update(manifest_content_sha256="0" * 64))
    identity_changes(lambda m: m["components"][-1].update(profile_sha256="1" * 64))
    identity_changes(lambda m: m["components"][-1].update(evidence_authoring_tree_equivalent_head="2" * 40))
    identity_changes(lambda m: m["covered_stages"].append("treasury_credit_authorization_downstream"))
    identity_changes(lambda m: m["uncovered_stages"].remove("deployment_currentness"))

    invalid_mutations = [
        lambda m: m["components"][-1].update(production_subject="different-subject"),
        lambda m: m["components"][-1].update(source_authority="ExecutableQualified"),
        lambda m: m["components"][0].update(profile_sha256="3" * 64),
        lambda m: m.update(authority_class="ObservedEndToEnd"),
        lambda m: m.update(claim_ceiling="ObservedEndToEnd"),
        lambda m: m["components"].pop(),
        lambda m: m["uncovered_stages"].clear(),
        lambda m: m.update(authorized_parameter_mutation=True),
        lambda m: m.update(authoritative_parameter_current=True),
    ]
    for mutate in invalid_mutations:
        forged = copy.deepcopy(manifest)
        mutate(forged)
        try:
            validate(forged, predecessor_v3, predecessor_v2, predecessor_v1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid v4 mutation accepted")

    return {**result, "self_test": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predecessor-v3", type=Path, required=True)
    parser.add_argument("--predecessor-v2", type=Path, required=True)
    parser.add_argument("--predecessor-v1", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    manifest = load(args.manifest)
    v3 = load(args.predecessor_v3)
    v2 = load(args.predecessor_v2)
    v1 = load(args.predecessor_v1)
    result = self_test(manifest, v3, v2, v1) if args.self_test else validate(manifest, v3, v2, v1)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
