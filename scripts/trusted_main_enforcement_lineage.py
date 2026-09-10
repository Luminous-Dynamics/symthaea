#!/usr/bin/env python3
"""Bind trusted-main enforcement V3 to the exact positive V2 evidence it descends from.

V3 contains a `v2_evidence_id`, but an identifier cannot validate bytes that were
not supplied. This theorem therefore consumes both derived artifacts, validates
their positive semantics, independently selects both content identities, and
requires every V3-selected rule-suite/observation to be exactly the normalized
positive observation carried by V2.

This remains provider-readback evidence only. It does not authenticate GitHub,
establish external chronology, prove current admission, or grant scientific
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import trusted_main_enforcement_evidence_v2 as v2
import trusted_main_enforcement_evidence_v3 as v3
import trusted_main_provider_time as provider_time
import trusted_main_ruleset as p0

SCHEMA = "symthaea.github-trusted-main-enforcement-lineage.v1"
DOMAIN = b"symthaea.github-trusted-main-enforcement-lineage.v1\0"
MAX_JSON_BYTES = 4_000_000
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
OPERATIONS = ("deletion", "force_push", "ordinary_direct_update")
V2_POSITIVE_FIELDS = {
    "schema", "repository", "repository_id", "target_ref", "ruleset_id", "policy_id",
    "structural_verification_id", "effective_rules_verification_id", "root_subject_id",
    "root_sha", "operation_observations", "missing_operations", "disposition",
    "violations", "bypass_assurance", "evidence_authority", "chronology_authority",
    "current_admission", "scientific_authority", "evidence_id",
}
OBSERVATION_FIELDS = {
    "operation", "rule_type", "rule_suite_id", "actor_id", "actor_name", "before_sha",
    "attempted_after_sha", "pushed_at", "suite_result", "rule_enforcement", "rule_result",
    "ruleset_id", "evidence_basis",
}


class EnforcementLineageError(ValueError):
    """Fail-closed V2 -> V3 enforcement lineage error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise EnforcementLineageError(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise EnforcementLineageError(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except EnforcementLineageError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EnforcementLineageError(f"{path}: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical(value)).hexdigest()


def _sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise EnforcementLineageError(f"{where}: sha256 identity required")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EnforcementLineageError(f"{where}: positive integer required")
    return value


def _string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EnforcementLineageError(f"{where}: canonical non-empty string required")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise EnforcementLineageError(f"{where}: control characters are forbidden")
    return value


def _validate_positive_v2(value: Any, policy: Any) -> dict[str, Any]:
    p = p0.normalize_policy(policy)
    if not isinstance(value, dict) or set(value) != V2_POSITIVE_FIELDS:
        raise EnforcementLineageError("enforcement_v2: closed positive derived schema required")
    if value["schema"] != v2.SCHEMA:
        raise EnforcementLineageError("enforcement_v2.schema: unsupported theorem schema")
    if value["repository"] != p["repository"] or value["repository_id"] != p["repository_id"]:
        raise EnforcementLineageError("enforcement_v2: repository identity mismatch")
    if value["target_ref"] != p["target_ref"]:
        raise EnforcementLineageError("enforcement_v2: target ref mismatch")
    if value["policy_id"] != p0.policy_id(p):
        raise EnforcementLineageError("enforcement_v2: policy identity mismatch")
    ruleset_id = _positive_int(value["ruleset_id"], where="enforcement_v2.ruleset_id")
    root_sha = _string(value["root_sha"], where="enforcement_v2.root_sha")
    if v2.SHA40_RE.fullmatch(root_sha) is None:
        raise EnforcementLineageError("enforcement_v2.root_sha: lowercase 40-hex commit SHA required")
    for field in (
        "structural_verification_id", "effective_rules_verification_id", "root_subject_id",
        "evidence_id",
    ):
        _sha256(value[field], where=f"enforcement_v2.{field}")

    if value["disposition"] != "EnforcementBehaviorallyCorroborated":
        raise EnforcementLineageError("enforcement_v2: positive behavioral disposition required")
    if value["missing_operations"] != [] or value["violations"] != []:
        raise EnforcementLineageError("enforcement_v2: positive evidence cannot contain missing operations or violations")
    expected_bypass = (
        "no-configured-p0-bypass-and-distinct-non-bypass-failures-observed"
        if p["allowed_bypass_actors"] == []
        else "not-fully-established"
    )
    if value["bypass_assurance"] != expected_bypass:
        raise EnforcementLineageError("enforcement_v2: bypass assurance is inconsistent with selected policy")
    fixed = {
        "evidence_authority": "server-rule-evaluation-readback-only",
        "chronology_authority": "provider-timestamp-observed-only",
        "current_admission": "not-evaluated",
        "scientific_authority": "none",
    }
    for field, expected in fixed.items():
        if value[field] != expected:
            raise EnforcementLineageError(f"enforcement_v2.{field}: canonical authority ceiling required")

    observations = value["operation_observations"]
    if not isinstance(observations, dict) or set(observations) != set(OPERATIONS):
        raise EnforcementLineageError("enforcement_v2.operation_observations: exact operation set required")
    suite_ids: list[int] = []
    for operation in OPERATIONS:
        observation = observations[operation]
        if not isinstance(observation, dict) or set(observation) != OBSERVATION_FIELDS:
            raise EnforcementLineageError(f"enforcement_v2.{operation}: closed normalized observation required")
        if observation["operation"] != operation:
            raise EnforcementLineageError(f"enforcement_v2.{operation}: operation label mismatch")
        if observation["rule_type"] != v2.OPERATIONS[operation]:
            raise EnforcementLineageError(f"enforcement_v2.{operation}: rule type mismatch")
        suite_id = _positive_int(
            observation["rule_suite_id"], where=f"enforcement_v2.{operation}.rule_suite_id"
        )
        suite_ids.append(suite_id)
        _positive_int(observation["actor_id"], where=f"enforcement_v2.{operation}.actor_id")
        _string(observation["actor_name"], where=f"enforcement_v2.{operation}.actor_name")
        if observation["before_sha"] != root_sha:
            raise EnforcementLineageError(f"enforcement_v2.{operation}: root subject mismatch")
        attempted = _string(
            observation["attempted_after_sha"], where=f"enforcement_v2.{operation}.attempted_after_sha"
        )
        if v2.SHA40_RE.fullmatch(attempted) is None:
            raise EnforcementLineageError(f"enforcement_v2.{operation}: attempted SHA malformed")
        try:
            provider_time.parse_github_utc_instant(
                observation["pushed_at"], where=f"enforcement_v2.{operation}.pushed_at"
            )
        except provider_time.ProviderTimeError as exc:
            raise EnforcementLineageError(str(exc)) from exc
        expected_literals = {
            "suite_result": "fail",
            "rule_enforcement": "active",
            "rule_result": "fail",
            "evidence_basis": "github-rule-suite-detailed-readback",
        }
        for field, expected in expected_literals.items():
            if observation[field] != expected:
                raise EnforcementLineageError(
                    f"enforcement_v2.{operation}.{field}: canonical positive value required"
                )
        if observation["ruleset_id"] != ruleset_id:
            raise EnforcementLineageError(f"enforcement_v2.{operation}: ruleset identity mismatch")
    if len(set(suite_ids)) != len(OPERATIONS):
        raise EnforcementLineageError("enforcement_v2: rule-suite IDs must be distinct")

    payload = dict(value)
    observed_id = payload.pop("evidence_id")
    if v2._content_id(v2.DOMAIN, payload) != observed_id:
        raise EnforcementLineageError("enforcement_v2: content identity mismatch")
    return value


def derive_enforcement_lineage(
    *,
    policy: Any,
    enforcement_evidence_v2: Any,
    enforcement_evidence_v3: Any,
    expected_v2_evidence_id: str,
    expected_v3_evidence_id: str,
) -> dict[str, Any]:
    p = p0.normalize_policy(policy)
    v2_evidence = _validate_positive_v2(enforcement_evidence_v2, p)
    try:
        v3_evidence = v3.validate_enforcement_evidence_v3(enforcement_evidence_v3)
    except v3.EnforcementEvidenceV3Error as exc:
        raise EnforcementLineageError(f"enforcement_v3: revalidation failed: {exc}") from exc

    expected_v2 = _sha256(expected_v2_evidence_id, where="expected_v2_evidence_id")
    expected_v3 = _sha256(expected_v3_evidence_id, where="expected_v3_evidence_id")
    if v2_evidence["evidence_id"] != expected_v2:
        raise EnforcementLineageError("enforcement_v2: bytes do not match independently selected identity")
    if v3_evidence["evidence_id"] != expected_v3:
        raise EnforcementLineageError("enforcement_v3: bytes do not match independently selected identity")
    if v3_evidence["v2_evidence_id"] != v2_evidence["evidence_id"]:
        raise EnforcementLineageError("enforcement_v3.v2_evidence_id: does not bind supplied exact V2 evidence")

    shared_fields = (
        "repository", "repository_id", "target_ref", "ruleset_id", "policy_id",
        "structural_verification_id", "effective_rules_verification_id", "root_subject_id",
        "root_sha", "bypass_assurance",
    )
    for field in shared_fields:
        if v3_evidence[field] != v2_evidence[field]:
            raise EnforcementLineageError(f"V2/V3 lineage mismatch: {field}")

    observation_ids: dict[str, str] = {}
    for operation in OPERATIONS:
        observation = v2_evidence["operation_observations"][operation]
        observed_id = v3.rule_suite_observation_id(observation)
        observation_ids[operation] = observed_id
        if v3_evidence["selected_rule_suite_observation_ids"][operation] != observed_id:
            raise EnforcementLineageError(f"{operation}: V3 selected observation is not exact V2 observation")
        if v3_evidence["selected_rule_suite_ids"][operation] != observation["rule_suite_id"]:
            raise EnforcementLineageError(f"{operation}: V3 selected suite ID is not exact V2 suite")
        if v3_evidence["selected_actor_id"] != observation["actor_id"]:
            raise EnforcementLineageError(f"{operation}: V3 selected actor is not exact V2 actor")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "policy_id": p0.policy_id(p),
        "repository": p["repository"],
        "repository_id": p["repository_id"],
        "target_ref": p["target_ref"],
        "ruleset_id": v2_evidence["ruleset_id"],
        "root_sha": v2_evidence["root_sha"],
        "v2_evidence_id": v2_evidence["evidence_id"],
        "v3_evidence_id": v3_evidence["evidence_id"],
        "trusted_enforcement_selection_id": v3_evidence["trusted_enforcement_selection_id"],
        "selected_rule_suite_ids": v3_evidence["selected_rule_suite_ids"],
        "selected_rule_suite_observation_ids": observation_ids,
        "selected_actor_id": v3_evidence["selected_actor_id"],
        "disposition": "EnforcementV3BoundToExactPositiveV2",
        "lineage_basis": "independently-selected-v2-v3-content-and-exact-normalized-observations",
        "evidence_authority": "server-rule-evaluation-readback-only",
        "chronology_authority": "provider-valid-utc-instants-observed-only",
        "receipt_attestation": "none",
        "current_admission": "not-evaluated",
        "scientific_authority": "none",
    }
    result["lineage_id"] = _content_id(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("enforcement_evidence_v2", type=Path)
    parser.add_argument("enforcement_evidence_v3", type=Path)
    parser.add_argument("--expected-v2-evidence-id", required=True)
    parser.add_argument("--expected-v3-evidence-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = derive_enforcement_lineage(
            policy=_load_json(args.policy),
            enforcement_evidence_v2=_load_json(args.enforcement_evidence_v2),
            enforcement_evidence_v3=_load_json(args.enforcement_evidence_v3),
            expected_v2_evidence_id=args.expected_v2_evidence_id,
            expected_v3_evidence_id=args.expected_v3_evidence_id,
        )
    except (EnforcementLineageError, p0.PolicyError) as exc:
        print(f"trusted-main enforcement lineage invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
