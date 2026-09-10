#!/usr/bin/env python3
"""Bind selected P0 enforcement observations to one closed ruleset-version interval.

This is the narrow provider-history theorem from #1289. It composes independently
selected enforcement V3, exact reviewed P0 version-state V2, a stable complete
ruleset-history receipt, and the exact normalized observations selected by V3.

V1 deliberately requires both predecessor and successor version boundaries. A
first-ever or latest/open version is non-positive until a separate capture-fence
theorem can prove the missing side without assuming chronology.

A positive result establishes provider-order binding only. It does not establish
external chronology, provider/operator authentication, current admission, or
scientific authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import trusted_main_enforcement_evidence_v2 as enforcement_v2
import trusted_main_enforcement_evidence_v3 as enforcement_v3
import trusted_main_provider_time as provider_time
import trusted_main_ruleset as p0
import trusted_main_ruleset_history as history
import trusted_main_ruleset_version_state_v2 as version_state_v2

SCHEMA = "symthaea.github-trusted-main-ruleset-version-binding.v1"
DOMAIN = b"symthaea.github-trusted-main-ruleset-version-binding.v1\0"
OBSERVATIONS_SCHEMA = "symthaea.github-trusted-main-selected-rule-suite-observations.v1"
MAX_JSON_BYTES = 4_000_000
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
OPERATIONS = ("deletion", "force_push", "ordinary_direct_update")


class RulesetVersionBindingError(ValueError):
    """Fail-closed exact ruleset-version interval binding error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RulesetVersionBindingError(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise RulesetVersionBindingError(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except RulesetVersionBindingError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RulesetVersionBindingError(f"{path}: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(value: Any) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical(value)).hexdigest()


def _sha256(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise RulesetVersionBindingError(f"{where}: sha256 identity required")
    return value


def _positive_int(value: Any, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RulesetVersionBindingError(f"{where}: positive integer required")
    return value


def _validate_enforcement(value: Any, *, expected_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RulesetVersionBindingError("enforcement_v3: object required")
    required = {
        "schema", "repository", "repository_id", "target_ref", "root_sha", "ruleset_id",
        "policy_id", "structural_verification_id", "effective_rules_verification_id",
        "root_subject_id", "trusted_enforcement_selection_id", "selected_rule_suite_ids",
        "selected_rule_suite_observation_ids", "selected_actor_id", "v2_evidence_id",
        "disposition", "bypass_assurance", "evidence_selection_basis", "evidence_authority",
        "chronology_authority", "current_admission", "receipt_attestation",
        "scientific_authority", "evidence_id",
    }
    if set(value) != required or value.get("schema") != enforcement_v3.SCHEMA:
        raise RulesetVersionBindingError("enforcement_v3: closed supported theorem schema required")
    if value["disposition"] != "EnforcementBehaviorallyCorroborated":
        raise RulesetVersionBindingError("enforcement_v3: positive behavioral disposition required")
    expected = _sha256(expected_id, where="expected_enforcement_evidence_id")
    observed = _sha256(value["evidence_id"], where="enforcement_v3.evidence_id")
    payload = dict(value)
    del payload["evidence_id"]
    if enforcement_v3._content_id(enforcement_v3.DOMAIN, payload) != observed:
        raise RulesetVersionBindingError("enforcement_v3: content identity mismatch")
    if observed != expected:
        raise RulesetVersionBindingError("enforcement_v3: bytes do not match independently selected identity")
    for field in ("selected_rule_suite_ids", "selected_rule_suite_observation_ids"):
        mapping = value[field]
        if not isinstance(mapping, dict) or set(mapping) != set(OPERATIONS):
            raise RulesetVersionBindingError(f"enforcement_v3.{field}: exact operation set required")
    for operation in OPERATIONS:
        _positive_int(value["selected_rule_suite_ids"][operation], where=f"selected_rule_suite_ids.{operation}")
        _sha256(value["selected_rule_suite_observation_ids"][operation], where=f"selected_rule_suite_observation_ids.{operation}")
    return value


def _reverify_version_state(value: Any, policy: Any, *, expected_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RulesetVersionBindingError("version_state_v2: object required")
    required = {
        "schema", "policy_id", "repository", "repository_id", "target_ref", "ruleset_id",
        "version_id", "provider_updated_at", "provider_unix_nanos", "provider_time_schema",
        "provider_actor_id", "provider_actor_type", "state", "v1_version_state_id",
        "disposition", "provider_order_authority", "chronology_authority", "current_admission",
        "scientific_authority", "version_state_id",
    }
    if set(value) != required or value.get("schema") != version_state_v2.SCHEMA:
        raise RulesetVersionBindingError("version_state_v2: closed supported theorem schema required")
    expected = _sha256(expected_id, where="expected_version_state_id")
    raw = {
        "version_id": value["version_id"],
        "actor": {"id": value["provider_actor_id"], "type": value["provider_actor_type"]},
        "updated_at": value["provider_updated_at"],
        "state": value["state"],
    }
    try:
        regenerated = version_state_v2.verify_version_state_v2(raw, policy)
    except (version_state_v2.RulesetVersionStateV2Error, p0.PolicyError, ValueError) as exc:
        raise RulesetVersionBindingError(f"version_state_v2: revalidation failed: {exc}") from exc
    if regenerated != value:
        raise RulesetVersionBindingError("version_state_v2: supplied bytes differ from revalidated exact P0 state")
    if value["version_state_id"] != expected:
        raise RulesetVersionBindingError("version_state_v2: bytes do not match independently selected identity")
    return value


def _validate_observations(value: Any, enforcement: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or set(value) != {"schema", "observations"}:
        raise RulesetVersionBindingError("selected_observations: closed schema/observations object required")
    if value["schema"] != OBSERVATIONS_SCHEMA:
        raise RulesetVersionBindingError("selected_observations.schema: unsupported schema")
    observations = value["observations"]
    if not isinstance(observations, dict) or set(observations) != set(OPERATIONS):
        raise RulesetVersionBindingError("selected_observations.observations: exact operation set required")
    normalized: dict[str, dict[str, Any]] = {}
    for operation in OPERATIONS:
        observation = observations[operation]
        try:
            observation_id = enforcement_v3.rule_suite_observation_id(observation)
        except enforcement_v3.EnforcementEvidenceV3Error as exc:
            raise RulesetVersionBindingError(f"{operation}: invalid normalized observation: {exc}") from exc
        if observation_id != enforcement["selected_rule_suite_observation_ids"][operation]:
            raise RulesetVersionBindingError(f"{operation}: bytes do not match V3-selected observation identity")
        if observation["operation"] != operation:
            raise RulesetVersionBindingError(f"{operation}: normalized operation label mismatch")
        if observation["rule_suite_id"] != enforcement["selected_rule_suite_ids"][operation]:
            raise RulesetVersionBindingError(f"{operation}: selected rule-suite ID mismatch")
        if observation["actor_id"] != enforcement["selected_actor_id"]:
            raise RulesetVersionBindingError(f"{operation}: selected actor mismatch")
        if observation["before_sha"] != enforcement["root_sha"]:
            raise RulesetVersionBindingError(f"{operation}: root subject mismatch")
        if observation["ruleset_id"] != enforcement["ruleset_id"]:
            raise RulesetVersionBindingError(f"{operation}: ruleset ID mismatch")
        try:
            instant = provider_time.parse_github_utc_instant(observation["pushed_at"], where=f"{operation}.pushed_at")
        except provider_time.ProviderTimeError as exc:
            raise RulesetVersionBindingError(str(exc)) from exc
        normalized[operation] = {
            **observation,
            "provider_unix_nanos": instant.unix_nanos,
            "observation_id": observation_id,
        }
    return normalized


def derive_version_binding(
    *,
    policy: Any,
    history_verification: Any,
    selected_version_state: Any,
    enforcement_evidence_v3: Any,
    selected_observations: Any,
    expected_history_id: str,
    expected_version_state_id: str,
    expected_enforcement_evidence_id: str,
) -> dict[str, Any]:
    normalized_policy = p0.normalize_policy(policy)
    policy_id = p0.policy_id(normalized_policy)
    enforcement = _validate_enforcement(enforcement_evidence_v3, expected_id=expected_enforcement_evidence_id)
    version_state = _reverify_version_state(selected_version_state, policy, expected_id=expected_version_state_id)
    try:
        verified_history = history.validate_verified_history(history_verification)
    except history.RulesetHistoryError as exc:
        raise RulesetVersionBindingError(f"history_verification: {exc}") from exc
    expected_history = _sha256(expected_history_id, where="expected_history_id")
    if verified_history["history_id"] != expected_history:
        raise RulesetVersionBindingError("history_verification: bytes do not match independently selected identity")

    for label, artifact in (
        ("enforcement_v3", enforcement),
        ("version_state_v2", version_state),
        ("history_verification", verified_history),
    ):
        if artifact["policy_id"] != policy_id:
            raise RulesetVersionBindingError(f"{label}: policy identity mismatch")
        if artifact["repository"] != normalized_policy["repository"] or artifact["repository_id"] != normalized_policy["repository_id"]:
            raise RulesetVersionBindingError(f"{label}: repository identity mismatch")
        if artifact["target_ref"] != normalized_policy["target_ref"]:
            raise RulesetVersionBindingError(f"{label}: target ref mismatch")
    if not (enforcement["ruleset_id"] == version_state["ruleset_id"] == verified_history["ruleset_id"]):
        raise RulesetVersionBindingError("ruleset identity mismatch across composed artifacts")

    observations = _validate_observations(selected_observations, enforcement)
    versions = verified_history["versions"]
    matches = [i for i, entry in enumerate(versions) if entry["version_id"] == version_state["version_id"]]
    if len(matches) != 1:
        raise RulesetVersionBindingError("selected version_id is absent or non-unique in complete history")
    selected_index = matches[0]
    if selected_index == 0:
        raise RulesetVersionBindingError("selected version has no predecessor boundary; closed interval not established")
    if selected_index + 1 >= len(versions):
        raise RulesetVersionBindingError("selected version has no successor boundary; closed interval not established")

    selected_summary = versions[selected_index]
    for field in ("provider_updated_at", "provider_unix_nanos", "provider_actor_id", "provider_actor_type"):
        if selected_summary[field] != version_state[field]:
            raise RulesetVersionBindingError(f"selected version history summary does not match exact version state: {field}")
    predecessor = versions[selected_index - 1]
    successor = versions[selected_index + 1]

    resolutions: dict[str, dict[str, Any]] = {}
    for operation in OPERATIONS:
        attempt_nanos = observations[operation]["provider_unix_nanos"]
        eligible = [entry for entry in versions if entry["provider_unix_nanos"] <= attempt_nanos]
        if not eligible:
            raise RulesetVersionBindingError(f"{operation}: attempt predates first observed ruleset version")
        resolved = eligible[-1]
        if resolved["version_id"] != version_state["version_id"]:
            raise RulesetVersionBindingError(
                f"{operation}: provider ordering resolves attempt to version {resolved['version_id']}, "
                f"not selected P0 version {version_state['version_id']}"
            )
        resolutions[operation] = {
            "rule_suite_id": observations[operation]["rule_suite_id"],
            "observation_id": observations[operation]["observation_id"],
            "pushed_at": observations[operation]["pushed_at"],
            "provider_unix_nanos": attempt_nanos,
            "resolved_version_id": resolved["version_id"],
        }

    interval = {
        "version_id": version_state["version_id"],
        "predecessor_version_id": predecessor["version_id"],
        "start_inclusive_updated_at": version_state["provider_updated_at"],
        "start_inclusive_unix_nanos": version_state["provider_unix_nanos"],
        "successor_version_id": successor["version_id"],
        "end_exclusive_updated_at": successor["provider_updated_at"],
        "end_exclusive_unix_nanos": successor["provider_unix_nanos"],
        "boundary_assurance": "closed-predecessor-and-successor-version-boundaries",
    }
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "policy_id": policy_id,
        "repository": normalized_policy["repository"],
        "repository_id": normalized_policy["repository_id"],
        "target_ref": normalized_policy["target_ref"],
        "ruleset_id": version_state["ruleset_id"],
        "selected_version_state_id": version_state["version_state_id"],
        "selected_version_id": version_state["version_id"],
        "history_id": verified_history["history_id"],
        "enforcement_evidence_id": enforcement["evidence_id"],
        "trusted_enforcement_selection_id": enforcement["trusted_enforcement_selection_id"],
        "selected_rule_suite_observation_ids": enforcement["selected_rule_suite_observation_ids"],
        "selected_interval": interval,
        "attempt_resolutions": resolutions,
        "disposition": "P0RulesetVersionBindingOnly",
        "boundary_rule": "latest-version-with-updated-at-less-than-or-equal-to-attempt; next-version-boundary-exclusive",
        "history_completeness_basis": verified_history["completeness_basis"],
        "provider_order_authority": "github-provider-valid-utc-instants-only",
        "capture_authentication": "none",
        "chronology_authority": "not-externally-anchored",
        "operator_authentication": "none",
        "current_admission": "not-evaluated",
        "scientific_authority": "none",
    }
    result["binding_id"] = _content_id(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("history_verification", type=Path)
    parser.add_argument("selected_version_state", type=Path)
    parser.add_argument("enforcement_evidence_v3", type=Path)
    parser.add_argument("selected_observations", type=Path)
    parser.add_argument("--expected-history-id", required=True)
    parser.add_argument("--expected-version-state-id", required=True)
    parser.add_argument("--expected-enforcement-evidence-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = derive_version_binding(
            policy=_load_json(args.policy),
            history_verification=_load_json(args.history_verification),
            selected_version_state=_load_json(args.selected_version_state),
            enforcement_evidence_v3=_load_json(args.enforcement_evidence_v3),
            selected_observations=_load_json(args.selected_observations),
            expected_history_id=args.expected_history_id,
            expected_version_state_id=args.expected_version_state_id,
            expected_enforcement_evidence_id=args.expected_enforcement_evidence_id,
        )
    except (
        RulesetVersionBindingError, p0.PolicyError, history.RulesetHistoryError,
        version_state_v2.RulesetVersionStateV2Error, enforcement_v2.EnforcementEvidenceError,
        enforcement_v3.EnforcementEvidenceV3Error,
    ) as exc:
        print(f"trusted-main ruleset version binding invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
