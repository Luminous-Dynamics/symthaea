#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SCHEMA = "symthaea-composite-mechanism-manifest-v1"
MANIFEST_ID = "mycelix-observed-composite-fca2c107-v1"
EXPECTED_SHA256 = "366b87944e74fe01369b78a8a7ea1c34f6dee05237583a47d86e5999b93b150c"
PRODUCTION_SUBJECT = "fca2c107a1ea5108823ce617ba4111b6f7f77230"
AUTHORITY = "ObservedCompositeSlice"
CLAIM_CEILING = "ObservedCompositeSliceOnly"

EXPECTED_COMPONENTS = {
    "Voting": {
        "profile_id": "mycelix-voting-observed-fca2c107-v2",
        "profile_sha256": "680af4668889c299b7e0d74531f44894a64a384be71778bca54d3f21ca80ac01",
        "corpus_sha256": "ee5e7649a773f564b443320689f465080d4641a0f4f09f13c0c49a7087d6dc10",
        "mycelix_evidence_head": "4b4e27910a98ad393e5a508e54e0f73c48c19107",
        "production_subject": PRODUCTION_SUBJECT,
        "source_authority": "ObservedSourceBound",
        "symthaea_conformance": "CrossImplementationConformance",
        "symthaea_pr": 3233,
    },
    "Execution": {
        "profile_id": "mycelix-execution-observed-fca2c107-v1",
        "profile_sha256": "c977bdcef9e5faac83351050999451432b618d5cc523bece804eba5dd1ae81f6",
        "corpus_sha256": "0c6669e44d6d18396ede43324f5cf3abbb25ddd3a2a9f59abb2c8a3699ba5fd4",
        "mycelix_evidence_head": "197714209c60503f0fba4143409da383bc9cbf83",
        "production_subject": PRODUCTION_SUBJECT,
        "source_authority": "ObservedSourceBound",
        "symthaea_conformance": "CrossImplementationConformance",
        "symthaea_pr": 3254,
    },
}

EXPECTED_UNCOVERED = {
    "proposal_creation_and_lifecycle_as_independent_profile",
    "threshold_signing_as_independent_mechanism_profile",
    "constitution_parameter_authorization_downstream",
    "treasury_credit_authorization_downstream",
    "deployment_currentness",
}

FORBIDDEN_VERDICT_KEYS = {
    "safe",
    "secure",
    "fair",
    "sybil_proof",
    "end_to_end_safe",
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


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("manifest root must be object")
    return obj


def validate(manifest: dict) -> dict:
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected composite manifest schema")
    if manifest.get("manifest_id") != MANIFEST_ID or manifest.get("revision") != 1:
        raise ValueError("unexpected composite manifest identity")
    if manifest.get("authority_class") != AUTHORITY:
        raise ValueError("composite authority promotion or drift")
    if manifest.get("claim_ceiling") != CLAIM_CEILING:
        raise ValueError("composite claim ceiling drift")
    if manifest.get("production_subject") != PRODUCTION_SUBJECT:
        raise ValueError("unexpected composite production subject")

    walk_forbidden(manifest)

    components = manifest.get("components")
    if not isinstance(components, list) or len(components) != 2:
        raise ValueError("exactly two frozen components required")
    by_role = {c.get("role"): c for c in components if isinstance(c, dict)}
    if set(by_role) != set(EXPECTED_COMPONENTS):
        raise ValueError("component role set drift")
    for role, expected in EXPECTED_COMPONENTS.items():
        actual = dict(by_role[role])
        actual.pop("role", None)
        if actual != expected:
            raise ValueError(f"{role} component commitment drift")
        if by_role[role].get("production_subject") != manifest["production_subject"]:
            raise ValueError(f"{role} production subject mismatch")
        if by_role[role].get("source_authority") != "ObservedSourceBound":
            raise ValueError(f"{role} authority promotion")
        if by_role[role].get("symthaea_conformance") != "CrossImplementationConformance":
            raise ValueError(f"{role} conformance-class drift")

    covered = manifest.get("covered_stages")
    uncovered = manifest.get("uncovered_stages")
    required = manifest.get("required_stages_for_end_to_end")
    if not isinstance(covered, list) or not covered or len(set(covered)) != len(covered):
        raise ValueError("covered stages must be a non-empty unique list")
    if not isinstance(uncovered, list) or not uncovered or len(set(uncovered)) != len(uncovered):
        raise ValueError("ObservedCompositeSlice requires explicit uncovered stages")
    if set(uncovered) != EXPECTED_UNCOVERED:
        raise ValueError("uncovered-stage set drift")
    if not isinstance(required, list) or not required or len(set(required)) != len(required):
        raise ValueError("required end-to-end stage registry missing or invalid")

    # The current object is intentionally incomplete. End-to-end language is not
    # available while any declared required coverage remains outstanding.
    if manifest["authority_class"] == "ObservedEndToEnd" or manifest["claim_ceiling"] == "ObservedEndToEnd":
        raise ValueError("ObservedEndToEnd forbidden while coverage is incomplete")

    non_claims = set(manifest.get("non_claims", []))
    required_non_claims = {
        "no_observed_end_to_end_claim",
        "no_deployment_currentness",
        "no_governance_safety_claim",
        "no_fairness_claim",
        "no_constitutional_legitimacy_claim",
    }
    if not required_non_claims.issubset(non_claims):
        raise ValueError("required non-claims missing")

    actual = payload_digest(manifest)
    if manifest.get("manifest_content_sha256") != actual:
        raise ValueError(f"manifest content commitment mismatch: {actual}")
    if actual != EXPECTED_SHA256:
        raise ValueError("manifest differs from frozen commitment")

    return {
        "validated": True,
        "authority_class": AUTHORITY,
        "claim_ceiling": CLAIM_CEILING,
        "manifest_id": MANIFEST_ID,
        "manifest_content_sha256": actual,
        "production_subject": PRODUCTION_SUBJECT,
        "component_count": len(components),
        "covered_stage_count": len(covered),
        "uncovered_stage_count": len(uncovered),
    }


def self_test(manifest: dict) -> dict:
    out = validate(manifest)
    baseline = payload_digest(manifest)

    def identity_changes(mutator) -> None:
        candidate = copy.deepcopy(manifest)
        candidate.pop("manifest_content_sha256", None)
        mutator(candidate)
        if hashlib.sha256(canonical(candidate)).hexdigest() == baseline:
            raise AssertionError("semantic mutation did not change manifest identity")

    identity_changes(lambda m: m["components"][0].update(profile_sha256="0" * 64))
    identity_changes(lambda m: m["components"][1].update(corpus_sha256="1" * 64))
    identity_changes(lambda m: m["covered_stages"].append("proposal_creation_and_lifecycle"))
    identity_changes(lambda m: m["uncovered_stages"].remove("deployment_currentness"))
    identity_changes(lambda m: m.update(claim_ceiling="ObservedEndToEnd"))

    invalid_mutations = [
        lambda m: m["components"][1].update(production_subject="different-subject"),
        lambda m: m["components"][0].update(source_authority="ExecutableQualified"),
        lambda m: m.update(authority_class="ObservedEndToEnd"),
        lambda m: m["uncovered_stages"].clear(),
        lambda m: m.update(safe=True),
    ]
    for mutate in invalid_mutations:
        forged = copy.deepcopy(manifest)
        mutate(forged)
        try:
            validate(forged)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid composite mutation accepted")

    return {**out, "self_test": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    manifest = load(args.manifest)
    out = self_test(manifest) if args.self_test else validate(manifest)
    print(json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
