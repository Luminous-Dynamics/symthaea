#!/usr/bin/env python3
"""Independent V19 corpus-manifest validator for SCIP text-to-claim evaluation."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "symthaea.scip-text-claim-corpus-manifest-policy/v1"
MANIFEST_SCHEMA = "symthaea.scip-text-claim-corpus-manifest/v1"
VALIDATION_SCHEMA = "symthaea.scip-text-claim-corpus-manifest-validation/v1"
AUTHORITY = "manifest-contract-only"
V18_SHA256 = "96e2ec5e1fad213f4261405c22d1f80cb6f1d106fd5621481eb0e95a6cca4530"
POLICY_SEMANTIC_SHA256 = "9a30ba36b3e872aaec349647f71433579bc0932f24c96335bb32afe77498ffc0"

DIMENSIONS = (
    "entity-reference", "relation-direction", "numeric-value-and-unit",
    "polarity-and-negation", "quantifier-and-cardinality", "temporal-scope",
    "epistemic-modality", "attribution-and-source", "causal-strength",
    "unsupported-additions", "required-detail-coverage",
)
DISCOURSE_FAMILIES = (
    "long-distance-coreference", "cross-sentence-negation", "nested-attribution",
    "temporal-contrast", "multiple-entities-same-type", "multiple-numbers-same-unit",
    "causal-versus-correlational-contrast", "mixed-certain-and-uncertain-claims",
)
POLARITIES = ("positive", "negative")
SPLITS = ("calibration", "confirmatory")

POLICY_FIELDS = {
    "schema", "authority", "v18_preregistration_sha256", "dimensions",
    "discourse_families", "dimension_stratum", "discourse_stratum", "totals",
    "assignment", "case_identity", "manifest_identity", "leakage_controls",
    "claim_boundary",
}
DIMENSION_STRATUM_FIELDS = {
    "cases_per_polarity", "calibration_per_polarity", "confirmatory_per_polarity", "polarities"
}
DISCOURSE_STRATUM_FIELDS = {
    "cases_per_family", "calibration_per_family", "confirmatory_per_family"
}
TOTAL_FIELDS = {"calibration", "confirmatory", "all"}
ASSIGNMENT_FIELDS = {
    "algorithm", "seed_commitment_domain", "rank_domain", "seed_committed_before_case_generation",
    "seed_revealed_only_after_manifest_seal", "rank_tie_breaker", "seed_bytes", "assignment_key_bytes",
}
CASE_IDENTITY_FIELDS = {"algorithm", "domain", "split_excluded", "assignment_key_excluded"}
MANIFEST_IDENTITY_FIELDS = {"algorithm", "self_hash_field_absent"}
LEAKAGE_FIELDS = {
    "no_assignment_key_reuse", "no_case_id_reuse", "no_template_hash_overlap_between_splits",
    "no_named_entity_tuple_hash_overlap_between_splits", "no_numeric_tuple_hash_overlap_between_splits",
    "no_exact_sentence_hash_overlap_between_splits",
}
BOUNDARY_FIELDS = {
    "manifest_validation_does_not_establish_annotation_correctness",
    "manifest_validation_does_not_establish_extractor_quality",
    "manifest_validation_does_not_establish_surface_fidelity",
    "manifest_validation_does_not_authorize_confirmatory_execution",
}
MANIFEST_FIELDS = {
    "schema", "authority", "policy_semantic_sha256", "v18_preregistration_sha256",
    "seed_commitment_sha256", "cases",
}
CASE_FIELDS = {
    "assignment_key_sha256", "case_id", "kind", "dimension", "polarity", "discourse_family",
    "surface_sha256", "source_inventory_sha256", "expected_inventory_sha256",
    "annotation_receipt_sha256", "template_sha256", "named_entity_tuple_sha256",
    "numeric_tuple_sha256", "exact_sentence_sha256", "split",
}
CASE_ID_FIELDS = (
    "kind", "dimension", "polarity", "discourse_family", "surface_sha256",
    "source_inventory_sha256", "expected_inventory_sha256", "annotation_receipt_sha256",
    "template_sha256", "named_entity_tuple_sha256", "numeric_tuple_sha256",
    "exact_sentence_sha256",
)

class ManifestError(ValueError):
    pass


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ManifestError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"), object_pairs_hook=strict_object)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ManifestError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ManifestError("top-level value must be an object")
    return value


def fields(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object")
    actual = set(value)
    if actual != expected:
        raise ManifestError(
            f"{where} fields mismatch: missing={sorted(expected-actual)} extra={sorted(actual-expected)}"
        )
    return value


def require_true(value: Any, where: str) -> None:
    if value is not True:
        raise ManifestError(f"{where} must be true")


def require_hex(value: Any, where: str, *, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ManifestError(f"{where} must be 64 lowercase hex characters")
    if any(ch not in "0123456789abcdef" for ch in value) or value == "0" * 64:
        raise ManifestError(f"{where} must be non-zero lowercase hex")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_policy(policy: dict[str, Any]) -> None:
    fields(policy, POLICY_FIELDS, "policy")
    if policy["schema"] != POLICY_SCHEMA or policy["authority"] != AUTHORITY:
        raise ManifestError("policy schema/authority drift")
    if policy["v18_preregistration_sha256"] != V18_SHA256:
        raise ManifestError("policy V18 binding drift")
    if policy["dimensions"] != list(DIMENSIONS):
        raise ManifestError("policy dimension order drift")
    if policy["discourse_families"] != list(DISCOURSE_FAMILIES):
        raise ManifestError("policy discourse family order drift")

    dim = fields(policy["dimension_stratum"], DIMENSION_STRATUM_FIELDS, "policy.dimension_stratum")
    if dim != {
        "cases_per_polarity": 52, "calibration_per_polarity": 20,
        "confirmatory_per_polarity": 32, "polarities": list(POLARITIES),
    }:
        raise ManifestError("dimension stratum policy drift")
    discourse = fields(policy["discourse_stratum"], DISCOURSE_STRATUM_FIELDS, "policy.discourse_stratum")
    if discourse != {"cases_per_family": 27, "calibration_per_family": 11, "confirmatory_per_family": 16}:
        raise ManifestError("discourse stratum policy drift")
    totals = fields(policy["totals"], TOTAL_FIELDS, "policy.totals")
    if totals != {"calibration": 528, "confirmatory": 832, "all": 1360}:
        raise ManifestError("corpus totals policy drift")

    assignment = fields(policy["assignment"], ASSIGNMENT_FIELDS, "policy.assignment")
    if assignment["algorithm"] != "hmac-sha256-sort-within-stratum/v1":
        raise ManifestError("assignment algorithm drift")
    if assignment["seed_commitment_domain"] != "symthaea-scip-text-claim-seed-commitment-v1\0":
        raise ManifestError("seed commitment domain drift")
    if assignment["rank_domain"] != "symthaea-scip-text-claim-split-rank-v1\0":
        raise ManifestError("rank domain drift")
    if assignment["rank_tie_breaker"] != "assignment_key_sha256":
        raise ManifestError("rank tie-breaker drift")
    if assignment["seed_bytes"] != 32 or assignment["assignment_key_bytes"] != 32:
        raise ManifestError("assignment byte-width drift")
    require_true(assignment["seed_committed_before_case_generation"], "policy.assignment.seed_committed_before_case_generation")
    require_true(assignment["seed_revealed_only_after_manifest_seal"], "policy.assignment.seed_revealed_only_after_manifest_seal")

    case_identity = fields(policy["case_identity"], CASE_IDENTITY_FIELDS, "policy.case_identity")
    if case_identity["algorithm"] != "sha256-domain-separated-canonical-json/v1":
        raise ManifestError("case identity algorithm drift")
    if case_identity["domain"] != "symthaea-scip-text-claim-case-v1\0":
        raise ManifestError("case identity domain drift")
    require_true(case_identity["split_excluded"], "policy.case_identity.split_excluded")
    require_true(case_identity["assignment_key_excluded"], "policy.case_identity.assignment_key_excluded")

    manifest_identity = fields(policy["manifest_identity"], MANIFEST_IDENTITY_FIELDS, "policy.manifest_identity")
    if manifest_identity["algorithm"] != "sha256-canonical-json/v1":
        raise ManifestError("manifest identity algorithm drift")
    require_true(manifest_identity["self_hash_field_absent"], "policy.manifest_identity.self_hash_field_absent")

    for section, expected in (("leakage_controls", LEAKAGE_FIELDS), ("claim_boundary", BOUNDARY_FIELDS)):
        obj = fields(policy[section], expected, f"policy.{section}")
        for key in expected:
            require_true(obj[key], f"policy.{section}.{key}")

    actual_semantic_sha = semantic_sha256(policy)
    if actual_semantic_sha != POLICY_SEMANTIC_SHA256:
        raise ManifestError(f"policy semantic identity drifted: {actual_semantic_sha}")


def seed_commitment(seed: bytes, policy: dict[str, Any]) -> str:
    if len(seed) != policy["assignment"]["seed_bytes"]:
        raise ManifestError("seed must be exactly 32 bytes")
    domain = policy["assignment"]["seed_commitment_domain"].encode("utf-8")
    return hashlib.sha256(domain + seed).hexdigest()


def rank_for(seed: bytes, assignment_key_sha256: str, policy: dict[str, Any]) -> bytes:
    key_bytes = bytes.fromhex(assignment_key_sha256)
    domain = policy["assignment"]["rank_domain"].encode("utf-8")
    return hmac.new(seed, domain + key_bytes, hashlib.sha256).digest()


def case_identity(case: dict[str, Any], policy: dict[str, Any]) -> str:
    payload = {key: case[key] for key in CASE_ID_FIELDS}
    domain = policy["case_identity"]["domain"].encode("utf-8")
    return hashlib.sha256(domain + canonical_bytes(payload)).hexdigest()


def stratum_key(case: dict[str, Any]) -> tuple[str, ...]:
    if case["kind"] == "dimension":
        return ("dimension", case["dimension"], case["polarity"])
    return ("discourse", case["discourse_family"])


def validate_case(case: dict[str, Any], policy: dict[str, Any], index: int) -> None:
    fields(case, CASE_FIELDS, f"cases[{index}]")
    require_hex(case["assignment_key_sha256"], f"cases[{index}].assignment_key_sha256")
    require_hex(case["case_id"], f"cases[{index}].case_id")
    for name in (
        "surface_sha256", "source_inventory_sha256", "expected_inventory_sha256",
        "annotation_receipt_sha256", "template_sha256", "exact_sentence_sha256",
    ):
        require_hex(case[name], f"cases[{index}].{name}")
    require_hex(case["named_entity_tuple_sha256"], f"cases[{index}].named_entity_tuple_sha256", allow_none=True)
    require_hex(case["numeric_tuple_sha256"], f"cases[{index}].numeric_tuple_sha256", allow_none=True)

    if case["split"] not in SPLITS:
        raise ManifestError(f"cases[{index}].split invalid")
    if case["kind"] == "dimension":
        if case["dimension"] not in DIMENSIONS or case["polarity"] not in POLARITIES or case["discourse_family"] is not None:
            raise ManifestError(f"cases[{index}] invalid dimension stratum")
    elif case["kind"] == "discourse":
        if case["dimension"] is not None or case["polarity"] is not None or case["discourse_family"] not in DISCOURSE_FAMILIES:
            raise ManifestError(f"cases[{index}] invalid discourse stratum")
    else:
        raise ManifestError(f"cases[{index}].kind invalid")

    expected_case_id = case_identity(case, policy)
    if case["case_id"] != expected_case_id:
        raise ManifestError(f"cases[{index}].case_id does not match canonical post-adjudication content")


def expected_split_map(cases: list[dict[str, Any]], seed: bytes, policy: dict[str, Any]) -> dict[str, str]:
    by_stratum: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_stratum[stratum_key(case)].append(case)

    result: dict[str, str] = {}
    for key, items in by_stratum.items():
        ordered = sorted(items, key=lambda item: (rank_for(seed, item["assignment_key_sha256"], policy), item["assignment_key_sha256"]))
        calibration_quota = 20 if key[0] == "dimension" else 11
        for offset, item in enumerate(ordered):
            result[item["assignment_key_sha256"]] = "calibration" if offset < calibration_quota else "confirmatory"
    return result


def check_cross_split_overlap(cases: list[dict[str, Any]], field: str) -> None:
    seen: dict[str, str] = {}
    for case in cases:
        value = case[field]
        if value is None:
            continue
        prior = seen.get(value)
        if prior is None:
            seen[value] = case["split"]
        elif prior != case["split"]:
            raise ManifestError(f"{field} {value} appears in both splits")


def validate_manifest(manifest: dict[str, Any], policy: dict[str, Any], seed: bytes | None) -> dict[str, Any]:
    validate_policy(policy)
    fields(manifest, MANIFEST_FIELDS, "manifest")
    if manifest["schema"] != MANIFEST_SCHEMA or manifest["authority"] != AUTHORITY:
        raise ManifestError("manifest schema/authority drift")
    if manifest["policy_semantic_sha256"] != POLICY_SEMANTIC_SHA256:
        raise ManifestError("manifest policy binding drift")
    if manifest["v18_preregistration_sha256"] != V18_SHA256:
        raise ManifestError("manifest V18 binding drift")
    require_hex(manifest["seed_commitment_sha256"], "manifest.seed_commitment_sha256")
    cases = manifest["cases"]
    if not isinstance(cases, list) or len(cases) != 1360:
        raise ManifestError("manifest must contain exactly 1360 cases")

    assignment_keys: set[str] = set()
    case_ids: set[str] = set()
    stratum_counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    split_counts = defaultdict(int)
    for index, case in enumerate(cases):
        validate_case(case, policy, index)
        if case["assignment_key_sha256"] in assignment_keys:
            raise ManifestError("duplicate assignment_key_sha256")
        assignment_keys.add(case["assignment_key_sha256"])
        if case["case_id"] in case_ids:
            raise ManifestError("duplicate case_id")
        case_ids.add(case["case_id"])
        key = stratum_key(case)
        stratum_counts[key][case["split"]] += 1
        split_counts[case["split"]] += 1

    for dimension in DIMENSIONS:
        for polarity in POLARITIES:
            counts = stratum_counts[("dimension", dimension, polarity)]
            if counts["calibration"] != 20 or counts["confirmatory"] != 32:
                raise ManifestError(f"dimension stratum count drift: {dimension}/{polarity}")
    for family in DISCOURSE_FAMILIES:
        counts = stratum_counts[("discourse", family)]
        if counts["calibration"] != 11 or counts["confirmatory"] != 16:
            raise ManifestError(f"discourse stratum count drift: {family}")
    expected_strata = 11 * 2 + 8
    if len(stratum_counts) != expected_strata:
        raise ManifestError("unexpected stratum present")
    if split_counts["calibration"] != 528 or split_counts["confirmatory"] != 832:
        raise ManifestError("global split totals drift")

    for field in (
        "template_sha256", "named_entity_tuple_sha256", "numeric_tuple_sha256", "exact_sentence_sha256"
    ):
        check_cross_split_overlap(cases, field)

    split_verified = False
    if seed is not None:
        if seed_commitment(seed, policy) != manifest["seed_commitment_sha256"]:
            raise ManifestError("revealed seed does not match committed seed")
        expected = expected_split_map(cases, seed, policy)
        for case in cases:
            if case["split"] != expected[case["assignment_key_sha256"]]:
                raise ManifestError("manifest split disagrees with committed-seed reconstruction")
        split_verified = True

    manifest_sha = semantic_sha256(manifest)
    return {
        "schema": VALIDATION_SCHEMA,
        "authority": AUTHORITY,
        "policy_semantic_sha256": POLICY_SEMANTIC_SHA256,
        "manifest_sha256": manifest_sha,
        "case_count": 1360,
        "calibration_count": 528,
        "confirmatory_count": 832,
        "split_reconstruction_verified": split_verified,
        "annotation_correctness_established": False,
        "extractor_quality_established": False,
        "surface_fidelity_established": False,
        "confirmatory_execution_authorized": False,
    }


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def synthetic_manifest(policy: dict[str, Any], seed: bytes) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    counter = 0
    def make_case(kind: str, dimension: str | None, polarity: str | None, family: str | None) -> dict[str, Any]:
        nonlocal counter
        tag = f"case-{counter:04d}"
        counter += 1
        case = {
            "assignment_key_sha256": digest("assignment:" + tag),
            "case_id": "",
            "kind": kind,
            "dimension": dimension,
            "polarity": polarity,
            "discourse_family": family,
            "surface_sha256": digest("surface:" + tag),
            "source_inventory_sha256": digest("source:" + tag),
            "expected_inventory_sha256": digest("expected:" + tag),
            "annotation_receipt_sha256": digest("annotation:" + tag),
            "template_sha256": digest("template:" + tag),
            "named_entity_tuple_sha256": digest("entities:" + tag),
            "numeric_tuple_sha256": digest("numbers:" + tag),
            "exact_sentence_sha256": digest("sentence:" + tag),
            "split": "calibration",
        }
        case["case_id"] = case_identity(case, policy)
        return case

    for dimension in DIMENSIONS:
        for polarity in POLARITIES:
            for _ in range(52):
                cases.append(make_case("dimension", dimension, polarity, None))
    for family in DISCOURSE_FAMILIES:
        for _ in range(27):
            cases.append(make_case("discourse", None, None, family))

    expected = expected_split_map(cases, seed, policy)
    for case in cases:
        case["split"] = expected[case["assignment_key_sha256"]]
    return {
        "schema": MANIFEST_SCHEMA,
        "authority": AUTHORITY,
        "policy_semantic_sha256": POLICY_SEMANTIC_SHA256,
        "v18_preregistration_sha256": V18_SHA256,
        "seed_commitment_sha256": seed_commitment(seed, policy),
        "cases": cases,
    }


def self_test(policy: dict[str, Any]) -> None:
    seed = bytes(range(32))
    manifest = synthetic_manifest(policy, seed)
    structural = validate_manifest(manifest, policy, None)
    assert structural["split_reconstruction_verified"] is False
    verified = validate_manifest(manifest, policy, seed)
    assert verified["split_reconstruction_verified"] is True
    assert verified["case_count"] == 1360

    identity_subject = json.loads(json.dumps(manifest["cases"][0]))
    original_identity = case_identity(identity_subject, policy)
    identity_subject["split"] = (
        "confirmatory" if identity_subject["split"] == "calibration" else "calibration"
    )
    identity_subject["assignment_key_sha256"] = digest("alternate-assignment-key")
    assert case_identity(identity_subject, policy) == original_identity
    identity_subject["surface_sha256"] = digest("different-adjudicated-surface")
    assert case_identity(identity_subject, policy) != original_identity

    changed = json.loads(json.dumps(manifest))
    changed["cases"][0]["split"] = "confirmatory" if changed["cases"][0]["split"] == "calibration" else "calibration"
    try:
        validate_manifest(changed, policy, seed)
    except ManifestError:
        pass
    else:
        raise AssertionError("split tamper was accepted")

    changed = json.loads(json.dumps(manifest))
    changed["cases"][0]["assignment_key_sha256"] = changed["cases"][1]["assignment_key_sha256"]
    try:
        validate_manifest(changed, policy, None)
    except ManifestError:
        pass
    else:
        raise AssertionError("assignment-key duplication was accepted")

    print(f"PASS_CORPUS_MANIFEST_SELF_TEST policy_sha256={POLICY_SEMANTIC_SHA256} cases=1360")


def parse_seed(seed_hex: str | None) -> bytes | None:
    if seed_hex is None:
        return None
    if len(seed_hex) != 64 or any(ch not in "0123456789abcdef" for ch in seed_hex):
        raise ManifestError("--seed-hex must be exactly 64 lowercase hex characters")
    return bytes.fromhex(seed_hex)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?")
    parser.add_argument("--policy", default=str(Path(__file__).resolve().parent / "qualification" / "scip_text_claim_corpus_manifest_policy_v1.json"))
    parser.add_argument("--seed-hex")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        policy = load_json(Path(args.policy))
        validate_policy(policy)
        if args.self_test:
            self_test(policy)
        else:
            if not args.manifest:
                parser.error("manifest is required unless --self-test is used")
            result = validate_manifest(load_json(Path(args.manifest)), policy, parse_seed(args.seed_hex))
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
