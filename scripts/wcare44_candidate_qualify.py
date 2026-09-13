#!/usr/bin/env python3
"""Strict Stage-A front door for the WCARE-44 candidate kernel."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

import wcare44_candidate_kernel as kernel

PROTOCOL = "wcare44-authenticated-replication-aggregation-v1"
WCARE41 = "wcare41-authenticated-preregistration-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOKEN = re.compile(r"^[A-Za-z0-9._:-]+$")

AUTH_KEYS = {
    "protocol_version", "wcare40_plan_sha256", "wcare40_result_sha256",
    "wcare40_frontdoor_sha256", "wcare40_core_verifier_sha256",
    "builder_verifier", "temporal_verifier",
    "require_complete_builder_attestation_coverage",
    "require_temporal_preregistration", "evaluation_utc", "notes",
}
AUTH_REQUIRED = AUTH_KEYS - {"notes"}
BACKEND_KEYS = {"backend_id", "executable_sha256", "policy_sha256"}
BUILDER_KEYS = {
    "protocol_version", "subject_receipt_sha256", "subject_kind", "status",
    "wcare42_result_sha256", "wcare42_verifier_sha256",
    "wcare42_qualification_receipt_sha256", "verifier_execution_qualified",
    "synthetic", "notes",
}
BUILDER_REQUIRED = BUILDER_KEYS - {"notes"}
TEMPORAL_KEYS = {
    "protocol_version", "wcare40_result_sha256",
    "wcare41_authentication_plan_sha256", "status", "wcare43_result_sha256",
    "wcare43_verifier_sha256", "verifier_execution_qualified", "synthetic",
    "notes",
}
TEMPORAL_REQUIRED = TEMPORAL_KEYS - {"notes"}
FINAL_FALSE_FIELDS = (
    "child_verifier_lineage_established",
    "wcare42_executable_qualification_established",
    "wcare43_external_execution_lineage_established",
    "builder_authentication_established",
    "preregistration_temporal_precedence_established",
    "authenticated_preregistered_replication_established",
    "runtime_authority_granted",
)


class InvalidContract(Exception):
    pass


def load_object(path: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except Exception as exc:
        raise InvalidContract(f"{label}_json_invalid:{exc}") from exc
    if not isinstance(value, dict):
        raise InvalidContract(f"{label}_not_object")
    return value


def exact_keys(value: dict[str, Any], allowed: set[str], required: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise InvalidContract(f"{label}_unknown_fields:{','.join(unknown)}")
    if missing:
        raise InvalidContract(f"{label}_missing_fields:{','.join(missing)}")


def sha_field(value: dict[str, Any], field: str, label: str) -> None:
    item = value.get(field)
    if not isinstance(item, str) or not HEX64.fullmatch(item):
        raise InvalidContract(f"{label}_invalid_sha256:{field}")


def bool_field(value: dict[str, Any], field: str, label: str) -> None:
    if type(value.get(field)) is not bool:
        raise InvalidContract(f"{label}_invalid_boolean:{field}")


def validate_backend(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise InvalidContract(f"{label}_not_object")
    exact_keys(value, BACKEND_KEYS, BACKEND_KEYS, label)
    backend_id = value.get("backend_id")
    if not isinstance(backend_id, str) or not TOKEN.fullmatch(backend_id):
        raise InvalidContract(f"{label}_invalid_backend_id")
    sha_field(value, "executable_sha256", label)
    sha_field(value, "policy_sha256", label)


def validate_auth_plan(path: str) -> None:
    value = load_object(path, "wcare41_authentication_plan")
    exact_keys(value, AUTH_KEYS, AUTH_REQUIRED, "wcare41_authentication_plan")
    if value.get("protocol_version") != WCARE41:
        raise InvalidContract("wcare41_authentication_plan_protocol_mismatch")
    for field in (
        "wcare40_plan_sha256", "wcare40_result_sha256",
        "wcare40_frontdoor_sha256", "wcare40_core_verifier_sha256",
    ):
        sha_field(value, field, "wcare41_authentication_plan")
    validate_backend(value.get("builder_verifier"), "builder_verifier")
    validate_backend(value.get("temporal_verifier"), "temporal_verifier")
    if value.get("require_complete_builder_attestation_coverage") is not True:
        raise InvalidContract("wcare41_complete_builder_coverage_not_true")
    if value.get("require_temporal_preregistration") is not True:
        raise InvalidContract("wcare41_temporal_preregistration_not_true")


def validate_builder_observation(path: str) -> None:
    value = load_object(path, "builder_observation")
    exact_keys(value, BUILDER_KEYS, BUILDER_REQUIRED, "builder_observation")
    if value.get("protocol_version") != PROTOCOL:
        raise InvalidContract("builder_observation_protocol_mismatch")
    if value.get("subject_kind") not in {"BuilderProvenance", "BuilderRelation"}:
        raise InvalidContract("builder_observation_invalid_subject_kind")
    if value.get("status") not in {"ACCEPTED", "UNTRUSTED", "REJECTED", "INDETERMINATE"}:
        raise InvalidContract("builder_observation_invalid_status")
    for field in (
        "subject_receipt_sha256", "wcare42_result_sha256",
        "wcare42_verifier_sha256", "wcare42_qualification_receipt_sha256",
    ):
        sha_field(value, field, "builder_observation")
    bool_field(value, "verifier_execution_qualified", "builder_observation")
    bool_field(value, "synthetic", "builder_observation")


def validate_temporal_observation(path: str) -> None:
    value = load_object(path, "temporal_observation")
    exact_keys(value, TEMPORAL_KEYS, TEMPORAL_REQUIRED, "temporal_observation")
    if value.get("protocol_version") != PROTOCOL:
        raise InvalidContract("temporal_observation_protocol_mismatch")
    if value.get("status") not in {"ESTABLISHED", "NOT_ESTABLISHED", "INDETERMINATE", "INVALID"}:
        raise InvalidContract("temporal_observation_invalid_status")
    for field in (
        "wcare40_result_sha256", "wcare41_authentication_plan_sha256",
        "wcare43_result_sha256", "wcare43_verifier_sha256",
    ):
        sha_field(value, field, "temporal_observation")
    bool_field(value, "verifier_execution_qualified", "temporal_observation")
    bool_field(value, "synthetic", "temporal_observation")


def invalid(detail: str) -> tuple[dict[str, Any], int]:
    out = kernel.base_result()
    out["disposition"] = "CANDIDATE_INVALID"
    out["detail"] = f"frontdoor_contract_invalid:{detail}"
    out["builder_authentication_candidate_status"] = "INVALID"
    out["temporal_candidate_status"] = "INVALID"
    return out, 4


def main() -> int:
    args = kernel.parser().parse_args()
    try:
        validate_auth_plan(args.wcare41_authentication_plan)
        for path in args.builder_observation:
            validate_builder_observation(path)
        validate_temporal_observation(args.temporal_observation)
        result, code = kernel.evaluate(args)
        expected = {
            "CANDIDATE_MEASURED": 0,
            "CANDIDATE_INDETERMINATE": 3,
            "CANDIDATE_INVALID": 4,
        }.get(result.get("disposition"), 4)
        if code != expected:
            result, code = invalid("kernel_exit_disposition_mismatch")
        if any(result.get(field) is not False for field in FINAL_FALSE_FIELDS):
            result, code = invalid("kernel_attempted_final_promotion")
    except InvalidContract as exc:
        result, code = invalid(str(exc))
    except Exception as exc:
        result, code = invalid(f"unexpected_frontdoor_error:{type(exc).__name__}:{exc}")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
