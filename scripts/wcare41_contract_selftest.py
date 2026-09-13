#!/usr/bin/env python3
"""Dependency-free contract campaign for WCARE-41 claim boundaries.

This validates protocol/result algebra only. It does not simulate Ed25519 or an
external timestamp/transparency service and cannot establish real authentication.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL = "wcare41-authenticated-preregistration-v1"


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def classify_builder(required: list[str], accepted: list[str], indeterminate: list[str]) -> str:
    if len(required) != len(set(required)) or len(accepted) != len(set(accepted)) or len(indeterminate) != len(set(indeterminate)):
        return "INVALID"
    required_set, accepted_set, indeterminate_set = set(required), set(accepted), set(indeterminate)
    if not accepted_set <= required_set or not indeterminate_set <= required_set or accepted_set & indeterminate_set:
        return "INVALID"
    if indeterminate_set:
        return "INDETERMINATE"
    if not accepted_set:
        return "UNAUTHENTICATED"
    if accepted_set == required_set:
        return "AUTHENTICATED"
    return "PARTIAL"


def temporal_status(verifier_status: str, commitment_utc: str | None, replica_starts: list[str]) -> tuple[str, bool]:
    if verifier_status not in {"ESTABLISHED", "NOT_ESTABLISHED", "INDETERMINATE", "INVALID"}:
        return "INVALID", False
    if verifier_status != "ESTABLISHED":
        return verifier_status, False
    if commitment_utc is None or not replica_starts:
        return "INVALID", False
    commitment = parse_utc(commitment_utc)
    earliest = min(parse_utc(item) for item in replica_starts)
    if commitment >= earliest:
        return "NOT_ESTABLISHED", False
    return "ESTABLISHED", True


def validate_result_logic(
    *,
    w40_components: int,
    authenticated_components: int,
    builder_status: str,
    temporal: str,
    builder_established: bool,
    temporal_established: bool,
) -> None:
    assert 0 <= authenticated_components <= w40_components, "authentication_must_not_increase_independence"
    assert builder_established == (builder_status == "AUTHENTICATED"), "builder_establishment_status_mismatch"
    assert temporal_established == (temporal == "ESTABLISHED"), "temporal_establishment_status_mismatch"


def main() -> int:
    schema_paths = [
        "docs/release/evidence/WCARE41_AUTHENTICATION_PLAN_SCHEMA_V1.json",
        "docs/release/evidence/WCARE41_BUILDER_ATTESTATION_ENVELOPE_SCHEMA_V1.json",
        "docs/release/evidence/WCARE41_TEMPORAL_PROOF_PACKAGE_SCHEMA_V1.json",
        "docs/release/evidence/WCARE41_BUILDER_ISSUER_TRUST_POLICY_SCHEMA_V1.json",
        "docs/release/evidence/WCARE41_TEMPORAL_VERIFIER_POLICY_SCHEMA_V1.json",
        "docs/release/evidence/WCARE41_AUTHENTICATION_RESULT_SCHEMA_V1.json",
    ]
    schemas = {}
    for relative in schema_paths:
        value = json.loads((ROOT / relative).read_text())
        assert isinstance(value, dict), relative
        assert value.get("additionalProperties") is False, relative
        schemas[relative] = value

    protocol = (ROOT / "docs/release/evidence/WCARE41_AUTHENTICATED_PREREGISTRATION_PROTOCOL_V1.md").read_text()
    for marker in [
        "valid signature != trusted issuer != independent builder != correct subject",
        "self-declared timestamp != externally established preregistration",
        "0 <= I41 <= I40",
        "WCARE-41 does not trust a precomputed `ATTESTATION_ACCEPTED` JSON value",
        "WCARE-41 does not trust a precomputed `ESTABLISHED` label",
    ]:
        assert marker in protocol, marker

    plan_schema = schemas[schema_paths[0]]
    assert plan_schema["properties"]["require_complete_builder_attestation_coverage"]["const"] is True
    assert plan_schema["properties"]["require_temporal_preregistration"]["const"] is True

    envelope = schemas[schema_paths[1]]
    assert envelope["properties"]["domain"]["const"] == "builder-evidence-attestation"
    assert {item["properties"]["subject_kind"]["const"] for item in envelope["allOf"]} == {"BuilderProvenance", "BuilderRelation"}

    required = ["prov-a", "prov-b", "rel-a-b"]
    assert classify_builder(required, required, []) == "AUTHENTICATED"
    assert classify_builder(required, ["prov-a"], []) == "PARTIAL"
    assert classify_builder(required, [], []) == "UNAUTHENTICATED"
    assert classify_builder(required, ["prov-a"], ["prov-b"]) == "INDETERMINATE"
    assert classify_builder(required, ["prov-a", "prov-a"], []) == "INVALID"
    assert classify_builder(required, ["unknown"], []) == "INVALID"

    status, established = temporal_status(
        "ESTABLISHED",
        "2026-09-13T17:00:00Z",
        ["2026-09-13T18:00:00Z", "2026-09-13T18:05:00Z"],
    )
    assert status == "ESTABLISHED" and established
    late_status, late_established = temporal_status(
        "ESTABLISHED",
        "2026-09-13T18:00:00Z",
        ["2026-09-13T18:00:00Z", "2026-09-13T18:05:00Z"],
    )
    assert late_status == "NOT_ESTABLISHED" and not late_established
    assert temporal_status("NOT_ESTABLISHED", None, ["2026-09-13T18:00:00Z"]) == ("NOT_ESTABLISHED", False)

    validate_result_logic(
        w40_components=3,
        authenticated_components=2,
        builder_status="AUTHENTICATED",
        temporal="NOT_ESTABLISHED",
        builder_established=True,
        temporal_established=False,
    )
    validate_result_logic(
        w40_components=3,
        authenticated_components=1,
        builder_status="PARTIAL",
        temporal="ESTABLISHED",
        builder_established=False,
        temporal_established=True,
    )
    try:
        validate_result_logic(
            w40_components=2,
            authenticated_components=3,
            builder_status="AUTHENTICATED",
            temporal="ESTABLISHED",
            builder_established=True,
            temporal_established=True,
        )
    except AssertionError as exc:
        assert str(exc) == "authentication_must_not_increase_independence"
    else:
        raise AssertionError("monotonicity_violation_was_not_rejected")

    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "PASS_WCARE41_CONTRACT_SELFTEST",
        "cryptographic_authentication_executed": False,
        "external_temporal_verification_executed": False,
        "builder_temporal_orthogonality_verified": True,
        "late_commitment_rejected_as_preregistration": True,
        "duplicate_evidence_does_not_multiply_coverage": True,
        "authentication_monotonicity_verified": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
