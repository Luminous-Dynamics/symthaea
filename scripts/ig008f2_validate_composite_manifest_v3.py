#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from ig008f1_validate_composite_manifest_v2 import validate as validate_v2

SCHEMA = "symthaea-composite-mechanism-manifest-v3"
MANIFEST_ID = "mycelix-observed-composite-fca2c107-v3"
EXPECTED_SHA256 = "3acdbd29d0fd631b863fa10a10eed0877af2336536578453acdc83f095632baf"
V2_ID = "mycelix-observed-composite-fca2c107-v2"
V2_SHA256 = "e3b42dbd7dacaa2f44de8bc330e85ff7d624f291d27c4a303fbaccf57091e541"
V1_SHA256 = "366b87944e74fe01369b78a8a7ea1c34f6dee05237583a47d86e5999b93b150c"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
AUTHORITY = "ObservedCompositeSlice"
CLAIM_CEILING = "ObservedCompositeSliceOnly"
COVERAGE_DEFINITION = "SourceObservedMechanismRepresentedNotPropertySatisfied"

TOP_KEYS = {
    "schema",
    "manifest_id",
    "revision",
    "authority_class",
    "claim_ceiling",
    "coverage_definition",
    "production_subject",
    "predecessor_manifest",
    "components",
    "covered_stages",
    "uncovered_stages",
    "required_stages_for_end_to_end",
    "non_claims",
    "manifest_content_sha256",
}

PROPOSAL_COMPONENT = {
    "role": "ProposalLifecycle",
    "profile_id": "mycelix-proposal-lifecycle-observed-fca2c107-v1",
    "profile_sha256": "7f42e2a8df25df94112d23f261d1f3ffe299d46d37cb3a5a6fe02aca0aa6c108",
    "corpus_sha256": "13eaaa988c73d29d67bccf7381f6f72cabd4eb7be090f36f0b444978cc708324",
    "mycelix_evidence_head": "6ddca81103c52408421e2e31da4b4dee0c0b2762",
    "production_subject": PRODUCTION_SUBJECT,
    "evidence_authoring_tree_equivalent_head": "31ede2365b81365bb119cd9351b2739119974130",
    "source_authority": "ObservedSourceBound",
    "symthaea_conformance": "CrossImplementationConformance",
    "symthaea_pr": 3326,
}

ADDED_COVERAGE = {
    "proposal_creation_observed_semantics",
    "proposal_lookup_projection_observed_semantics",
    "proposal_update_integrity_observed_semantics",
    "proposal_status_update_observed_semantics",
}
REMOVED_UNCOVERED = "proposal_creation_and_lifecycle_as_independent_profile"
REQUIRED_NON_CLAIMS = {
    "no_observed_end_to_end_claim",
    "no_authoritative_proposal_currentness_claim",
    "no_deterministic_proposal_fork_resolution_claim",
    "no_successor_governance_stack_deployment_claim",
    "no_secure_threshold_signing_claim",
    "no_cryptographic_signature_validity_claim",
    "no_deployment_currentness",
    "no_governance_safety_claim",
    "no_fairness_claim",
    "no_constitutional_legitimacy_claim",
}
FORBIDDEN_VERDICT_KEYS = {
    "safe",
    "secure",
    "fair",
    "authoritative_current",
    "fork_resolved",
    "successor_deployed",
    "threshold_signing_secure",
    "cryptographically_valid",
    "observed_end_to_end",
    "deployment_current",
    "constitutionally_legitimate",
}


def canonical(obj: object) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
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


def validate(manifest: dict, predecessor_v2: dict, predecessor_v1: dict) -> dict:
    v2_result = validate_v2(predecessor_v2, predecessor_v1)
    if v2_result["manifest_content_sha256"] != V2_SHA256:
        raise ValueError("unexpected v2 predecessor commitment")
    if v2_result["predecessor_manifest_sha256"] != V1_SHA256:
        raise ValueError("v2 did not validate against the frozen v1 predecessor")

    if set(manifest) != TOP_KEYS:
        raise ValueError("top-level v3 manifest drift")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected v3 schema")
    if manifest.get("manifest_id") != MANIFEST_ID or manifest.get("revision") != 3:
        raise ValueError("unexpected v3 identity")
    if manifest.get("authority_class") != AUTHORITY:
        raise ValueError("v3 authority promotion or drift")
    if manifest.get("claim_ceiling") != CLAIM_CEILING:
        raise ValueError("v3 claim ceiling drift")
    if manifest.get("coverage_definition") != COVERAGE_DEFINITION:
        raise ValueError("coverage/assurance semantic drift")
    if manifest.get("production_subject") != PRODUCTION_SUBJECT:
        raise ValueError("v3 production subject drift")

    if manifest.get("predecessor_manifest") != {
        "manifest_id": V2_ID,
        "revision": 2,
        "manifest_content_sha256": V2_SHA256,
    }:
        raise ValueError("v3 predecessor binding drift")

    walk_forbidden(manifest)

    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != 4:
        raise ValueError("v3 requires exactly four frozen components")
    by_role = {item.get("role"): item for item in components if isinstance(item, dict)}
    if set(by_role) != {"ProposalLifecycle", "Voting", "ThresholdSigning", "Execution"}:
        raise ValueError("v3 component role set drift")

    predecessor_by_role = {
        item["role"]: item
        for item in predecessor_v2.get("components", [])
        if isinstance(item, dict) and "role" in item
    }
    for retained in ("Voting", "ThresholdSigning", "Execution"):
        if by_role[retained] != predecessor_by_role.get(retained):
            raise ValueError(f"retained {retained} component changed from v2")
    if by_role["ProposalLifecycle"] != PROPOSAL_COMPONENT:
        raise ValueError("ProposalLifecycle component commitment drift")

    for role, component in by_role.items():
        if component.get("production_subject") != PRODUCTION_SUBJECT:
            raise ValueError(f"{role} mixed production subject")
        if component.get("source_authority") != "ObservedSourceBound":
            raise ValueError(f"{role} source authority promotion")
        if component.get("symthaea_conformance") != "CrossImplementationConformance":
            raise ValueError(f"{role} conformance-class drift")

    predecessor_covered = set(predecessor_v2.get("covered_stages", []))
    covered = manifest.get("covered_stages")
    if not isinstance(covered, list) or len(set(covered)) != len(covered):
        raise ValueError("invalid v3 covered-stage list")
    if set(covered) != predecessor_covered | ADDED_COVERAGE:
        raise ValueError("v3 coverage did not change monotonically by exactly the Proposal stages")

    predecessor_uncovered = set(predecessor_v2.get("uncovered_stages", []))
    uncovered = manifest.get("uncovered_stages")
    if not isinstance(uncovered, list) or not uncovered or len(set(uncovered)) != len(uncovered):
        raise ValueError("v3 must retain explicit uncovered stages")
    expected_uncovered = predecessor_uncovered - {REMOVED_UNCOVERED}
    if set(uncovered) != expected_uncovered:
        raise ValueError("v3 uncovered set changed beyond the Proposal stage")

    if manifest.get("required_stages_for_end_to_end") != predecessor_v2.get("required_stages_for_end_to_end"):
        raise ValueError("required-stage registry drift")

    if manifest["authority_class"] == "ObservedEndToEnd" or manifest["claim_ceiling"] == "ObservedEndToEnd":
        raise ValueError("ObservedEndToEnd forbidden while required coverage remains incomplete")

    non_claims = set(manifest.get("non_claims", []))
    if not REQUIRED_NON_CLAIMS.issubset(non_claims):
        raise ValueError("required v3 non-claims missing")

    actual = payload_digest(manifest)
    if manifest.get("manifest_content_sha256") != actual:
        raise ValueError(f"v3 content commitment mismatch: {actual}")
    if actual != EXPECTED_SHA256:
        raise ValueError("v3 differs from frozen commitment")

    return {
        "validated": True,
        "authority_class": AUTHORITY,
        "claim_ceiling": CLAIM_CEILING,
        "coverage_definition": COVERAGE_DEFINITION,
        "manifest_id": MANIFEST_ID,
        "manifest_content_sha256": actual,
        "predecessor_manifest_sha256": V2_SHA256,
        "root_predecessor_manifest_sha256": V1_SHA256,
        "production_subject": PRODUCTION_SUBJECT,
        "component_count": 4,
        "new_component": "ProposalLifecycle",
        "covered_stage_count": len(covered),
        "uncovered_stage_count": len(uncovered),
    }


def self_test(manifest: dict, predecessor_v2: dict, predecessor_v1: dict) -> dict:
    result = validate(manifest, predecessor_v2, predecessor_v1)
    baseline = payload_digest(manifest)

    def identity_changes(mutator) -> None:
        candidate = copy.deepcopy(manifest)
        candidate.pop("manifest_content_sha256", None)
        mutator(candidate)
        if hashlib.sha256(canonical(candidate)).hexdigest() == baseline:
            raise AssertionError("semantic mutation did not change v3 identity")

    identity_changes(lambda m: m["predecessor_manifest"].update(manifest_content_sha256="0" * 64))
    identity_changes(lambda m: m["components"][0].update(profile_sha256="1" * 64))
    identity_changes(lambda m: m["components"][0].update(evidence_authoring_tree_equivalent_head="2" * 40))
    identity_changes(lambda m: m["covered_stages"].append("constitution_parameter_authorization_downstream"))
    identity_changes(lambda m: m["uncovered_stages"].remove("deployment_currentness"))

    invalid_mutations = [
        lambda m: m["components"][0].update(production_subject="different-subject"),
        lambda m: m["components"][0].update(source_authority="ExecutableQualified"),
        lambda m: m["components"][1].update(profile_sha256="3" * 64),
        lambda m: m.update(authority_class="ObservedEndToEnd"),
        lambda m: m.update(claim_ceiling="ObservedEndToEnd"),
        lambda m: m["components"].pop(0),
        lambda m: m["uncovered_stages"].clear(),
        lambda m: m.update(authoritative_current=True),
        lambda m: m.update(successor_deployed=True),
    ]
    for mutate in invalid_mutations:
        forged = copy.deepcopy(manifest)
        mutate(forged)
        try:
            validate(forged, predecessor_v2, predecessor_v1)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid v3 mutation accepted")

    return {**result, "self_test": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predecessor-v2", type=Path, required=True)
    parser.add_argument("--predecessor-v1", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    manifest = load(args.manifest)
    predecessor_v2 = load(args.predecessor_v2)
    predecessor_v1 = load(args.predecessor_v1)
    result = (
        self_test(manifest, predecessor_v2, predecessor_v1)
        if args.self_test
        else validate(manifest, predecessor_v2, predecessor_v1)
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
