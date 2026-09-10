#!/usr/bin/env python3
"""Strict trusted-main behavioral enforcement evidence selection.

V2 verifies GitHub detailed rule-suite records as hostile provider data. V3 adds
an independent-selection boundary: the trusted phase must choose the exact
upstream verification identities, root subject, rule-suite attempts, normalized
rule-suite observations, and actor before supplied evidence can become a
positive historical enforcement fact.

This remains non-attesting and non-scientific. Content identity is not signer
identity, and current admission remains a separate theorem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import trusted_main_effective_rules as effective_rules
import trusted_main_enforcement_evidence_v2 as v2
import trusted_main_ruleset as p0

SCHEMA = "symthaea.github-trusted-main-enforcement-evidence.v3"
DOMAIN = b"symthaea.github-trusted-main-enforcement-evidence.v3\0"
SELECTION_SCHEMA = "symthaea.github-trusted-main-enforcement-selection.v2"
SELECTION_DOMAIN = b"symthaea.github-trusted-main-enforcement-selection.v2\0"
OBSERVATION_SCHEMA = "symthaea.github-trusted-main-rule-suite-observation.v1"
OBSERVATION_DOMAIN = b"symthaea.github-trusted-main-rule-suite-observation.v1\0"
OPERATIONS = ("deletion", "force_push", "ordinary_direct_update")
COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DERIVED_FIELDS = {
    "schema", "repository", "repository_id", "target_ref", "root_sha", "ruleset_id",
    "policy_id", "structural_verification_id", "effective_rules_verification_id",
    "root_subject_id", "trusted_enforcement_selection_id", "selected_rule_suite_ids",
    "selected_rule_suite_observation_ids", "selected_actor_id", "v2_evidence_id",
    "disposition", "bypass_assurance", "evidence_selection_basis", "evidence_authority",
    "chronology_authority", "current_admission", "receipt_attestation",
    "scientific_authority", "evidence_id",
}
CANONICAL_DERIVED_LABELS = {
    "disposition": "EnforcementBehaviorallyCorroborated",
    "evidence_selection_basis": "trusted-phase-independent-expected-ids-and-observation-content",
    "evidence_authority": "server-rule-evaluation-readback-only",
    "chronology_authority": "provider-timestamp-observed-only",
    "current_admission": "not-evaluated",
    "receipt_attestation": "none",
    "scientific_authority": "none",
}
ALLOWED_BYPASS_ASSURANCE = {
    "no-configured-p0-bypass-and-distinct-non-bypass-failures-observed",
    "not-fully-established",
}


class EnforcementEvidenceV3Error(ValueError):
    """Fail-closed independent enforcement-evidence selection error."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + _canonical(value)).hexdigest()


def _expected_sha256(value: Any, *, where: str) -> str:
    try:
        return v2._sha256_id(value, where=where)
    except v2.EnforcementEvidenceError as exc:
        raise EnforcementEvidenceV3Error(str(exc)) from exc


def _expected_positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EnforcementEvidenceV3Error(f"{where}: positive integer required")
    return value


def _canonical_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EnforcementEvidenceV3Error(f"{where}: canonical non-empty string required")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise EnforcementEvidenceV3Error(f"{where}: control characters are forbidden")
    return value


def _require_mapping(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EnforcementEvidenceV3Error(f"{where}: object required")
    return value


def rule_suite_observation_id(observation: Any) -> str:
    """Content identity for the normalized V2 rule-suite observation used by V3."""
    if not isinstance(observation, dict):
        raise EnforcementEvidenceV3Error("rule-suite observation: object required")
    required = {
        "operation", "rule_type", "rule_suite_id", "actor_id", "actor_name",
        "before_sha", "attempted_after_sha", "pushed_at", "suite_result",
        "rule_enforcement", "rule_result", "ruleset_id", "evidence_basis",
    }
    if set(observation) != required:
        raise EnforcementEvidenceV3Error("rule-suite observation: closed normalized schema required")
    payload = {"schema": OBSERVATION_SCHEMA, **observation}
    return _content_id(OBSERVATION_DOMAIN, payload)


def validate_enforcement_evidence_v3(value: Any) -> dict[str, Any]:
    """Revalidate one derived V3 object, including selector and authority semantics."""
    result = _require_mapping(value, where="enforcement_v3")
    if set(result) != DERIVED_FIELDS:
        raise EnforcementEvidenceV3Error("enforcement_v3: closed derived schema required")
    if result["schema"] != SCHEMA:
        raise EnforcementEvidenceV3Error("enforcement_v3.schema: unsupported theorem schema")
    for field, expected in CANONICAL_DERIVED_LABELS.items():
        if result[field] != expected:
            raise EnforcementEvidenceV3Error(
                f"enforcement_v3.{field}: canonical authority/selection label required"
            )
    if result["bypass_assurance"] not in ALLOWED_BYPASS_ASSURANCE:
        raise EnforcementEvidenceV3Error("enforcement_v3.bypass_assurance: unsupported assurance label")

    _canonical_string(result["repository"], where="enforcement_v3.repository")
    _expected_positive_int(result["repository_id"], where="enforcement_v3.repository_id")
    target_ref = _canonical_string(result["target_ref"], where="enforcement_v3.target_ref")
    if not target_ref.startswith("refs/heads/"):
        raise EnforcementEvidenceV3Error("enforcement_v3.target_ref: full refs/heads/... ref required")
    root_sha = _canonical_string(result["root_sha"], where="enforcement_v3.root_sha")
    if COMMIT_SHA_RE.fullmatch(root_sha) is None:
        raise EnforcementEvidenceV3Error("enforcement_v3.root_sha: lowercase 40-hex commit SHA required")
    _expected_positive_int(result["ruleset_id"], where="enforcement_v3.ruleset_id")
    selected_actor_id = _expected_positive_int(
        result["selected_actor_id"], where="enforcement_v3.selected_actor_id"
    )

    for field in (
        "policy_id", "structural_verification_id", "effective_rules_verification_id",
        "root_subject_id", "trusted_enforcement_selection_id", "v2_evidence_id", "evidence_id",
    ):
        _expected_sha256(result[field], where=f"enforcement_v3.{field}")

    suite_ids = result["selected_rule_suite_ids"]
    if not isinstance(suite_ids, dict) or set(suite_ids) != set(OPERATIONS):
        raise EnforcementEvidenceV3Error(
            "enforcement_v3.selected_rule_suite_ids: exact operation set required"
        )
    normalized_suite_ids = {
        operation: _expected_positive_int(
            suite_ids[operation], where=f"enforcement_v3.selected_rule_suite_ids.{operation}"
        )
        for operation in OPERATIONS
    }
    if len(set(normalized_suite_ids.values())) != len(OPERATIONS):
        raise EnforcementEvidenceV3Error("enforcement_v3.selected_rule_suite_ids: IDs must be distinct")

    observation_ids = result["selected_rule_suite_observation_ids"]
    if not isinstance(observation_ids, dict) or set(observation_ids) != set(OPERATIONS):
        raise EnforcementEvidenceV3Error(
            "enforcement_v3.selected_rule_suite_observation_ids: exact operation set required"
        )
    normalized_observation_ids = {
        operation: _expected_sha256(
            observation_ids[operation],
            where=f"enforcement_v3.selected_rule_suite_observation_ids.{operation}",
        )
        for operation in OPERATIONS
    }
    if len(set(normalized_observation_ids.values())) != len(OPERATIONS):
        raise EnforcementEvidenceV3Error(
            "enforcement_v3.selected_rule_suite_observation_ids: IDs must be distinct"
        )

    selection_payload = {
        "schema": SELECTION_SCHEMA,
        "structural_verification_id": result["structural_verification_id"],
        "effective_rules_verification_id": result["effective_rules_verification_id"],
        "root_subject_id": result["root_subject_id"],
        "rule_suite_ids": normalized_suite_ids,
        "rule_suite_observation_ids": normalized_observation_ids,
        "actor_id": selected_actor_id,
    }
    if _content_id(SELECTION_DOMAIN, selection_payload) != result["trusted_enforcement_selection_id"]:
        raise EnforcementEvidenceV3Error("enforcement_v3: trusted selection identity mismatch")

    payload = dict(result)
    del payload["evidence_id"]
    if _content_id(DOMAIN, payload) != result["evidence_id"]:
        raise EnforcementEvidenceV3Error("enforcement_v3: content identity mismatch")
    return result


def derive_enforcement_evidence_v3(
    *,
    policy: Any,
    structural_verification: Any,
    effective_rules_verification: Any,
    root_subject: Any,
    direct_update_rule_suite: Any,
    force_push_rule_suite: Any,
    deletion_rule_suite: Any,
    expected_structural_verification_id: str,
    expected_effective_rules_verification_id: str,
    expected_root_subject_id: str,
    expected_direct_update_rule_suite_id: int,
    expected_force_push_rule_suite_id: int,
    expected_deletion_rule_suite_id: int,
    expected_direct_update_observation_id: str,
    expected_force_push_observation_id: str,
    expected_deletion_observation_id: str,
    expected_actor_id: int,
) -> dict[str, Any]:
    structural = _require_mapping(structural_verification, where="structural_verification")
    effective = _require_mapping(effective_rules_verification, where="effective_rules_verification")
    root = _require_mapping(root_subject, where="root_subject")
    direct = _require_mapping(direct_update_rule_suite, where="direct_update_rule_suite")
    force = _require_mapping(force_push_rule_suite, where="force_push_rule_suite")
    deletion = _require_mapping(deletion_rule_suite, where="deletion_rule_suite")

    # Schema tags are theorem identity, not decorative metadata.
    if structural.get("schema") != p0.VERIFICATION_SCHEMA:
        raise EnforcementEvidenceV3Error("structural_verification.schema: unsupported theorem schema")
    if effective.get("schema") != effective_rules.SCHEMA:
        raise EnforcementEvidenceV3Error("effective_rules_verification.schema: unsupported theorem schema")
    if root.get("schema") != v2.ROOT_SUBJECT_SCHEMA:
        raise EnforcementEvidenceV3Error("root_subject.schema: unsupported theorem schema")

    expected_structural_id = _expected_sha256(
        expected_structural_verification_id, where="expected_structural_verification_id"
    )
    expected_effective_id = _expected_sha256(
        expected_effective_rules_verification_id, where="expected_effective_rules_verification_id"
    )
    expected_root_id = _expected_sha256(expected_root_subject_id, where="expected_root_subject_id")
    expected_observation_ids = {
        "ordinary_direct_update": _expected_sha256(
            expected_direct_update_observation_id, where="expected_direct_update_observation_id"
        ),
        "force_push": _expected_sha256(
            expected_force_push_observation_id, where="expected_force_push_observation_id"
        ),
        "deletion": _expected_sha256(
            expected_deletion_observation_id, where="expected_deletion_observation_id"
        ),
    }

    selected_suite_ids = {
        "ordinary_direct_update": _expected_positive_int(
            expected_direct_update_rule_suite_id, where="expected_direct_update_rule_suite_id"
        ),
        "force_push": _expected_positive_int(
            expected_force_push_rule_suite_id, where="expected_force_push_rule_suite_id"
        ),
        "deletion": _expected_positive_int(
            expected_deletion_rule_suite_id, where="expected_deletion_rule_suite_id"
        ),
    }
    if len(set(selected_suite_ids.values())) != 3:
        raise EnforcementEvidenceV3Error("expected rule-suite IDs must be distinct")
    selected_actor_id = _expected_positive_int(expected_actor_id, where="expected_actor_id")

    observed_structural_id = structural.get("verification_id")
    observed_effective_id = effective.get("verification_id")
    normalized_policy = p0.normalize_policy(policy)
    normalized_root = v2._root_subject(root, normalized_policy)
    observed_root_id = v2._root_subject_id(normalized_root)

    if observed_structural_id != expected_structural_id:
        raise EnforcementEvidenceV3Error("structural verification bytes do not match independently selected identity")
    if observed_effective_id != expected_effective_id:
        raise EnforcementEvidenceV3Error("effective-rule verification bytes do not match independently selected identity")
    if observed_root_id != expected_root_id:
        raise EnforcementEvidenceV3Error("root subject bytes do not match independently selected identity")

    suites = {
        "ordinary_direct_update": direct,
        "force_push": force,
        "deletion": deletion,
    }
    for operation, suite in suites.items():
        if suite.get("id") != selected_suite_ids[operation]:
            raise EnforcementEvidenceV3Error(
                f"{operation}: supplied rule suite does not match independently selected suite ID"
            )
        if suite.get("actor_id") != selected_actor_id:
            raise EnforcementEvidenceV3Error(
                f"{operation}: supplied rule suite does not match independently selected actor ID"
            )

    base = v2.derive_enforcement_evidence(
        policy=policy,
        structural_verification=structural,
        effective_rules_verification=effective,
        root_subject=root,
        direct_update_rule_suite=direct,
        force_push_rule_suite=force,
        deletion_rule_suite=deletion,
    )
    if base["disposition"] != "EnforcementBehaviorallyCorroborated":
        raise EnforcementEvidenceV3Error(
            "V3 requires V2 EnforcementBehaviorallyCorroborated; "
            f"observed {base['disposition']!r}"
        )

    observed_observation_ids: dict[str, str] = {}
    for operation, expected_suite_id in selected_suite_ids.items():
        observation = base["operation_observations"].get(operation)
        if not isinstance(observation, dict):
            raise EnforcementEvidenceV3Error(f"{operation}: V2 positive observation missing")
        if observation.get("rule_suite_id") != expected_suite_id:
            raise EnforcementEvidenceV3Error(f"{operation}: V2 observation suite ID mismatch")
        if observation.get("actor_id") != selected_actor_id:
            raise EnforcementEvidenceV3Error(f"{operation}: V2 observation actor ID mismatch")
        observed_id = rule_suite_observation_id(observation)
        observed_observation_ids[operation] = observed_id
        if observed_id != expected_observation_ids[operation]:
            raise EnforcementEvidenceV3Error(
                f"{operation}: normalized rule-suite observation does not match independently selected content identity"
            )

    selection_payload = {
        "schema": SELECTION_SCHEMA,
        "structural_verification_id": expected_structural_id,
        "effective_rules_verification_id": expected_effective_id,
        "root_subject_id": expected_root_id,
        "rule_suite_ids": selected_suite_ids,
        "rule_suite_observation_ids": expected_observation_ids,
        "actor_id": selected_actor_id,
    }
    selection_id = _content_id(SELECTION_DOMAIN, selection_payload)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "repository": base["repository"],
        "repository_id": base["repository_id"],
        "target_ref": base["target_ref"],
        "root_sha": base["root_sha"],
        "ruleset_id": base["ruleset_id"],
        "policy_id": base["policy_id"],
        "structural_verification_id": expected_structural_id,
        "effective_rules_verification_id": expected_effective_id,
        "root_subject_id": expected_root_id,
        "trusted_enforcement_selection_id": selection_id,
        "selected_rule_suite_ids": selected_suite_ids,
        "selected_rule_suite_observation_ids": observed_observation_ids,
        "selected_actor_id": selected_actor_id,
        "v2_evidence_id": base["evidence_id"],
        "bypass_assurance": base["bypass_assurance"],
        **CANONICAL_DERIVED_LABELS,
    }
    result["evidence_id"] = _content_id(DOMAIN, result)
    return validate_enforcement_evidence_v3(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("structural_verification", type=Path)
    parser.add_argument("effective_rules_verification", type=Path)
    parser.add_argument("root_subject", type=Path)
    parser.add_argument("direct_update_rule_suite", type=Path)
    parser.add_argument("force_push_rule_suite", type=Path)
    parser.add_argument("deletion_rule_suite", type=Path)
    parser.add_argument("--expected-structural-verification-id", required=True)
    parser.add_argument("--expected-effective-rules-verification-id", required=True)
    parser.add_argument("--expected-root-subject-id", required=True)
    parser.add_argument("--expected-direct-update-rule-suite-id", required=True, type=int)
    parser.add_argument("--expected-force-push-rule-suite-id", required=True, type=int)
    parser.add_argument("--expected-deletion-rule-suite-id", required=True, type=int)
    parser.add_argument("--expected-direct-update-observation-id", required=True)
    parser.add_argument("--expected-force-push-observation-id", required=True)
    parser.add_argument("--expected-deletion-observation-id", required=True)
    parser.add_argument("--expected-actor-id", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = derive_enforcement_evidence_v3(
            policy=v2._load_json(args.policy),
            structural_verification=v2._load_json(args.structural_verification),
            effective_rules_verification=v2._load_json(args.effective_rules_verification),
            root_subject=v2._load_json(args.root_subject),
            direct_update_rule_suite=v2._load_json(args.direct_update_rule_suite),
            force_push_rule_suite=v2._load_json(args.force_push_rule_suite),
            deletion_rule_suite=v2._load_json(args.deletion_rule_suite),
            expected_structural_verification_id=args.expected_structural_verification_id,
            expected_effective_rules_verification_id=args.expected_effective_rules_verification_id,
            expected_root_subject_id=args.expected_root_subject_id,
            expected_direct_update_rule_suite_id=args.expected_direct_update_rule_suite_id,
            expected_force_push_rule_suite_id=args.expected_force_push_rule_suite_id,
            expected_deletion_rule_suite_id=args.expected_deletion_rule_suite_id,
            expected_direct_update_observation_id=args.expected_direct_update_observation_id,
            expected_force_push_observation_id=args.expected_force_push_observation_id,
            expected_deletion_observation_id=args.expected_deletion_observation_id,
            expected_actor_id=args.expected_actor_id,
        )
    except (OSError, p0.PolicyError, effective_rules.EffectiveRulesError, v2.EnforcementEvidenceError, EnforcementEvidenceV3Error) as exc:
        print(f"trusted-main enforcement evidence V3 invalid: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
