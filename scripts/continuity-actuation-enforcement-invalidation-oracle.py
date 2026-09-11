#!/usr/bin/env python3
"""Independent decision oracle for actuation-enforcement campaign invalidation V1.

This classifies what remedy is required when an already-admitted campaign's world
changes. It does not admit evidence, authenticate roots, or grant execution authority.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

SCHEMA = "symthaea-continuity-actuation-enforcement-invalidation-oracle-v1"

HARD_FIELDS = (
    "backend_id",
    "backend_implementation_digest",
    "backend_generation",
    "enforcement_profile_id",
    "boundary_implementation_digest",
    "one_use_mechanism_digest",
    "enforcement_profile_generation",
    "authentication_profile_id",
    "authentication_implementation_digest",
    "authentication_root_id",
    "authentication_root_epoch",
    "harness_implementation_digest",
    "scenario_suite_manifest_digest",
    "environment_manifest_digest",
    "topology_dependency_manifest_digest",
    "hardware_firmware_manifest_digest",
    "toolchain_realization_digest",
    "obligation_schema_digest",
    "evidence_basis_mapping_digest",
    "evidence_manifest_digest",
    "verifier_semantics_digest",
    "verifier_adoption_id",
)

SPECIAL_SUCCESSOR_FIELDS = frozenset(
    {"authentication_root_id", "authentication_root_epoch", "verifier_adoption_id"}
)

FRESH_FIELDS = frozenset(
    {
        "verifier_challenge_current",
        "currentness_reestablished",
        "trusted_time_epoch_advanced_normally",
        "periodic_confirmation_due",
        "successor_readmitted_exact_campaign",
        "root_rotation_portable",
        "root_rotation_admission_id",
    }
)

ATTEMPT_FIELDS = frozenset(
    {
        "owner_authority_current",
        "distributed_context_unchanged",
        "subject_target_unchanged",
        "active_lkg_unchanged",
        "execution_attempt_unchanged",
        "actuation_fence_unchanged",
        "deny_or_emergency_stop",
        "permit_unconsumed",
        "resource_available",
    }
)

DIAGNOSTIC_FIELDS = frozenset(
    {
        "human_label_changed",
        "log_storage_location_changed",
        "report_format_changed",
        "comments_changed",
        "packaging_filename_changed",
    }
)

TOP_FIELDS = frozenset({"schema", "baseline", "current", "fresh", "attempt", "diagnostic"})
HEX64 = set("0123456789abcdef")


class InvalidationError(ValueError):
    pass


def exact_keys(obj: dict[str, Any], expected: set[str] | frozenset[str], where: str) -> None:
    got = set(obj)
    exp = set(expected)
    if got != exp:
        raise InvalidationError(
            f"{where}: exact fields required missing={sorted(exp-got)} extra={sorted(got-exp)}"
        )


def require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise InvalidationError(f"{field}: expected boolean")
    return value


def require_positive_u64(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > (1 << 64) - 1:
        raise InvalidationError(f"{field}: expected positive u64")
    return value


def require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value):
        raise InvalidationError(f"{field}: expected 64 lowercase hex characters")
    if value == "0" * 64:
        raise InvalidationError(f"{field}: zero identity forbidden")
    return value


def validate_hard_world(world: Any, where: str) -> dict[str, Any]:
    if not isinstance(world, dict):
        raise InvalidationError(f"{where}: expected object")
    exact_keys(world, set(HARD_FIELDS), where)
    for field in HARD_FIELDS:
        value = world[field]
        if field in {"backend_generation", "enforcement_profile_generation", "authentication_root_epoch"}:
            require_positive_u64(value, f"{where}.{field}")
        else:
            require_digest(value, f"{where}.{field}")
    return world


def validate_request(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise InvalidationError("request: expected object")
    exact_keys(obj, TOP_FIELDS, "request")
    if obj["schema"] != SCHEMA:
        raise InvalidationError(f"schema: expected {SCHEMA!r}")
    validate_hard_world(obj["baseline"], "baseline")
    validate_hard_world(obj["current"], "current")

    fresh = obj["fresh"]
    if not isinstance(fresh, dict):
        raise InvalidationError("fresh: expected object")
    exact_keys(fresh, FRESH_FIELDS, "fresh")
    for field in FRESH_FIELDS - {"root_rotation_admission_id"}:
        require_bool(fresh[field], f"fresh.{field}")
    rotation_id = fresh["root_rotation_admission_id"]
    if rotation_id is not None:
        require_digest(rotation_id, "fresh.root_rotation_admission_id")

    attempt = obj["attempt"]
    if not isinstance(attempt, dict):
        raise InvalidationError("attempt: expected object")
    exact_keys(attempt, ATTEMPT_FIELDS, "attempt")
    for field in ATTEMPT_FIELDS:
        require_bool(attempt[field], f"attempt.{field}")

    diagnostic = obj["diagnostic"]
    if not isinstance(diagnostic, dict):
        raise InvalidationError("diagnostic: expected object")
    exact_keys(diagnostic, DIAGNOSTIC_FIELDS, "diagnostic")
    for field in DIAGNOSTIC_FIELDS:
        require_bool(diagnostic[field], f"diagnostic.{field}")
    return obj


def decide(obj: Any) -> dict[str, Any]:
    req = validate_request(obj)
    baseline = req["baseline"]
    current = req["current"]
    fresh = req["fresh"]
    attempt = req["attempt"]

    changed = [field for field in HARD_FIELDS if baseline[field] != current[field]]
    ordinary_hard = [field for field in changed if field not in SPECIAL_SUCCESSOR_FIELDS]
    hard_reasons = [f"hard_identity_changed:{field}" for field in ordinary_hard]
    fresh_reasons: list[str] = []

    root_changed = any(
        baseline[field] != current[field]
        for field in ("authentication_root_id", "authentication_root_epoch")
    )
    adoption_changed = baseline["verifier_adoption_id"] != current["verifier_adoption_id"]

    if root_changed:
        explicit_rotation = (
            fresh["root_rotation_admission_id"] is not None
            and fresh["root_rotation_portable"]
            and fresh["successor_readmitted_exact_campaign"]
        )
        if explicit_rotation:
            fresh_reasons.append("authenticated_root_rotation_requires_fresh_readmission")
        else:
            hard_reasons.append("authentication_root_changed_without_portable_successor_admission")

    if adoption_changed:
        if fresh["successor_readmitted_exact_campaign"]:
            fresh_reasons.append("verifier_adoption_advanced_with_explicit_exact_campaign_readmission")
        else:
            hard_reasons.append("verifier_adoption_superseded_without_exact_campaign_readmission")

    if not fresh["verifier_challenge_current"]:
        fresh_reasons.append("verifier_challenge_not_current")
    if not fresh["currentness_reestablished"]:
        fresh_reasons.append("currentness_not_reestablished")
    if fresh["trusted_time_epoch_advanced_normally"]:
        fresh_reasons.append("trusted_time_epoch_advanced_requires_fresh_binding")
    if fresh["periodic_confirmation_due"]:
        fresh_reasons.append("periodic_currentness_confirmation_due")

    attempt_reasons: list[str] = []
    negative_attempt_expectations = {
        "owner_authority_current": "owner_authority_revoked_or_superseded",
        "distributed_context_unchanged": "distributed_context_changed",
        "subject_target_unchanged": "subject_or_target_changed",
        "active_lkg_unchanged": "active_lkg_changed",
        "execution_attempt_unchanged": "execution_attempt_or_trusted_epoch_changed",
        "actuation_fence_unchanged": "newer_actuation_fence_generation_exists",
        "permit_unconsumed": "one_use_permit_consumed",
        "resource_available": "resource_temporarily_unavailable",
    }
    for field, reason in negative_attempt_expectations.items():
        if not attempt[field]:
            attempt_reasons.append(reason)
    if attempt["deny_or_emergency_stop"]:
        attempt_reasons.append("deny_or_emergency_stop_active")

    # Precedence is constitutional in V1. No freshness challenge or transition-local
    # state may resurrect hard-invalidated evidence.
    if hard_reasons:
        classification = "FULL_CAMPAIGN_REQUALIFICATION"
        remedy = "rerun_all_enforcement_scenarios_under_new_campaign_root"
        reasons = hard_reasons
    elif fresh_reasons:
        classification = "FRESH_CURRENTNESS_READMISSION"
        remedy = "reestablish_currentness_and_exact_campaign_admission"
        reasons = fresh_reasons
    elif attempt_reasons:
        classification = "ATTEMPT_LOCAL_REQUALIFICATION"
        remedy = "deny_or_requalify_exact_transition_attempt"
        reasons = attempt_reasons
    else:
        classification = "CAMPAIGN_REUSABLE"
        remedy = "no_campaign_rerun_required"
        reasons = []

    return {
        "schema": SCHEMA,
        "classification": classification,
        "remedy": remedy,
        "reasons": sorted(set(reasons)),
        "hard_identity_change_count": len(changed),
        "diagnostic_changes": sorted(field for field, value in req["diagnostic"].items() if value),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", help="invalidation request JSON")
    args = parser.parse_args()
    try:
        with open(args.request, "r", encoding="utf-8") as fh:
            request = json.load(fh)
        result = decide(request)
    except (OSError, json.JSONDecodeError, InvalidationError) as exc:
        print(f"DENY: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
