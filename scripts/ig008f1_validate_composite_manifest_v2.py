#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from ig008f0_validate_composite_manifest import validate as validate_v1

SCHEMA = "symthaea-composite-mechanism-manifest-v2"
MANIFEST_ID = "mycelix-observed-composite-fca2c107-v2"
EXPECTED_SHA256 = "e3b42dbd7dacaa2f44de8bc330e85ff7d624f291d27c4a303fbaccf57091e541"
V1_ID = "mycelix-observed-composite-fca2c107-v1"
V1_SHA256 = "366b87944e74fe01369b78a8a7ea1c34f6dee05237583a47d86e5999b93b150c"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
AUTHORITY = "ObservedCompositeSlice"
CLAIM_CEILING = "ObservedCompositeSliceOnly"
COVERAGE_DEFINITION = "SourceObservedMechanismRepresentedNotPropertySatisfied"

THRESHOLD_COMPONENT = {
    "role": "ThresholdSigning",
    "profile_id": "mycelix-threshold-signing-observed-fca2c107-v1",
    "profile_sha256": "c15dfd860b759747938af2a13129d729fa0af1e75284418c9ea6b9c172f643ac",
    "corpus_sha256": "0f6532ae8e2c2e421da625592dbb3b38aa2b90c5342f46f3a305bdbec89b0269",
    "mycelix_evidence_head": "a580915d588338077ce6196514c43e24052f86cd",
    "production_subject": PRODUCTION_SUBJECT,
    "source_authority": "ObservedSourceBound",
    "symthaea_conformance": "CrossImplementationConformance",
    "symthaea_pr": 3296,
}

ADDED_COVERAGE = {
    "threshold_signing_producer_api_observed_semantics",
    "threshold_signature_integrity_observed_semantics",
}
REMOVED_UNCOVERED = "threshold_signing_as_independent_mechanism_profile"
REQUIRED_NON_CLAIMS = {
    "no_observed_end_to_end_claim",
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


def validate(manifest: dict, predecessor: dict) -> dict:
    predecessor_result = validate_v1(predecessor)
    if predecessor_result["manifest_content_sha256"] != V1_SHA256:
        raise ValueError("unexpected v1 predecessor commitment")

    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected v2 schema")
    if manifest.get("manifest_id") != MANIFEST_ID or manifest.get("revision") != 2:
        raise ValueError("unexpected v2 identity")
    if manifest.get("authority_class") != AUTHORITY:
        raise ValueError("v2 authority promotion or drift")
    if manifest.get("claim_ceiling") != CLAIM_CEILING:
        raise ValueError("v2 claim ceiling drift")
    if manifest.get("coverage_definition") != COVERAGE_DEFINITION:
        raise ValueError("coverage/assurance semantic drift")
    if manifest.get("production_subject") != PRODUCTION_SUBJECT:
        raise ValueError("v2 production subject drift")

    predecessor_ref = manifest.get("predecessor_manifest")
    if predecessor_ref != {
        "manifest_id": V1_ID,
        "revision": 1,
        "manifest_content_sha256": V1_SHA256,
    }:
        raise ValueError("v2 predecessor binding drift")

    walk_forbidden(manifest)

    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != 3:
        raise ValueError("v2 requires exactly three frozen components")
    by_role = {item.get("role"): item for item in components if isinstance(item, dict)}
    if set(by_role) != {"Voting", "ThresholdSigning", "Execution"}:
        raise ValueError("v2 component role set drift")

    predecessor_by_role = {
        item["role"]: item
        for item in predecessor.get("components", [])
        if isinstance(item, dict) and "role" in item
    }
    for retained in ("Voting", "Execution"):
        if by_role[retained] != predecessor_by_role.get(retained):
            raise ValueError(f"retained {retained} component changed from v1")
    if by_role["ThresholdSigning"] != THRESHOLD_COMPONENT:
        raise ValueError("ThresholdSigning component commitment drift")

    for role, component in by_role.items():
        if component.get("production_subject") != PRODUCTION_SUBJECT:
            raise ValueError(f"{role} mixed production subject")
        if component.get("source_authority") != "ObservedSourceBound":
            raise ValueError(f"{role} source authority promotion")
        if component.get("symthaea_conformance") != "CrossImplementationConformance":
            raise ValueError(f"{role} conformance-class drift")

    predecessor_covered = set(predecessor.get("covered_stages", []))
    covered = manifest.get("covered_stages")
    if not isinstance(covered, list) or len(set(covered)) != len(covered):
        raise ValueError("invalid v2 covered-stage list")
    if set(covered) != predecessor_covered | ADDED_COVERAGE:
        raise ValueError("v2 coverage did not change monotonically by exactly the signing stages")

    predecessor_uncovered = set(predecessor.get("uncovered_stages", []))
    uncovered = manifest.get("uncovered_stages")
    if not isinstance(uncovered, list) or not uncovered or len(set(uncovered)) != len(uncovered):
        raise ValueError("v2 must retain explicit uncovered stages")
    expected_uncovered = predecessor_uncovered - {REMOVED_UNCOVERED}
    if set(uncovered) != expected_uncovered:
        raise ValueError("v2 uncovered set changed beyond the threshold-signing stage")

    if manifest.get("required_stages_for_end_to_end") != predecessor.get("required_stages_for_end_to_end"):
        raise ValueError("required-stage registry drift")

    if manifest["authority_class"] == "ObservedEndToEnd" or manifest["claim_ceiling"] == "ObservedEndToEnd":
        raise ValueError("ObservedEndToEnd forbidden while required coverage remains incomplete")

    non_claims = set(manifest.get("non_claims", []))
    if not REQUIRED_NON_CLAIMS.issubset(non_claims):
        raise ValueError("required v2 non-claims missing")

    actual = payload_digest(manifest)
    if manifest.get("manifest_content_sha256") != actual:
        raise ValueError(f"v2 content commitment mismatch: {actual}")
    if actual != EXPECTED_SHA256:
        raise ValueError("v2 differs from frozen commitment")

    return {
        "validated": True,
        "authority_class": AUTHORITY,
        "claim_ceiling": CLAIM_CEILING,
        "coverage_definition": COVERAGE_DEFINITION,
        "manifest_id": MANIFEST_ID,
        "manifest_content_sha256": actual,
        "predecessor_manifest_sha256": V1_SHA256,
        "production_subject": PRODUCTION_SUBJECT,
        "component_count": 3,
        "new_component": "ThresholdSigning",
        "covered_stage_count": len(covered),
        "uncovered_stage_count": len(uncovered),
    }


def self_test(manifest: dict, predecessor: dict) -> dict:
    result = validate(manifest, predecessor)
    baseline = payload_digest(manifest)

    def identity_changes(mutator) -> None:
        candidate = copy.deepcopy(manifest)
        candidate.pop("manifest_content_sha256", None)
        mutator(candidate)
        if hashlib.sha256(canonical(candidate)).hexdigest() == baseline:
            raise AssertionError("semantic mutation did not change v2 identity")

    identity_changes(lambda m: m["predecessor_manifest"].update(manifest_content_sha256="0" * 64))
    identity_changes(lambda m: m["components"][1].update(profile_sha256="1" * 64))
    identity_changes(lambda m: m["covered_stages"].append("proposal_creation_and_lifecycle"))
    identity_changes(lambda m: m["uncovered_stages"].remove("deployment_currentness"))
    identity_changes(lambda m: m.update(coverage_definition="CoverageMeansSecure"))

    invalid_mutations = [
        lambda m: m["components"][0].update(production_subject="different-subject"),
        lambda m: m["components"][1].update(source_authority="ExecutableQualified"),
        lambda m: m.update(authority_class="ObservedEndToEnd"),
        lambda m: m.update(claim_ceiling="ObservedEndToEnd"),
        lambda m: m["components"].pop(1),
        lambda m: m["uncovered_stages"].clear(),
        lambda m: m.update(secure=True),
    ]
    for mutate in invalid_mutations:
        forged = copy.deepcopy(manifest)
        mutate(forged)
        try:
            validate(forged, predecessor)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid v2 mutation accepted")

    return {**result, "self_test": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    manifest = load(args.manifest)
    predecessor = load(args.predecessor)
    result = self_test(manifest, predecessor) if args.self_test else validate(manifest, predecessor)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
