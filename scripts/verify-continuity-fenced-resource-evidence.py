#!/usr/bin/env python3
"""Independent structural auditor for retained fenced-resource lab evidence.

This verifies bundle integrity and semantic identity reconstruction only. It does not
establish that the retained observations are truthful, verifier-qualified, or safe for
production actuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

BUNDLE_SCHEMA = "symthaea-continuity-fenced-resource-lab-evidence-bundle-v1"
OBSERVATIONS_SCHEMA = "symthaea-continuity-fenced-resource-lab-observations-v1"
RUN_CONTEXT_SCHEMA = "symthaea-continuity-fenced-resource-lab-run-context-v1"
CAMPAIGN_SCHEMA = "symthaea-continuity-actuation-enforcement-campaign-manifest-v1"

RECORD_DOMAIN = b"symthaea.continuity.fenced-resource-lab-observation.v1\0"
COMPLETE_DOMAIN = b"symthaea.continuity.fenced-resource-lab-complete-set.v1\0"
EVIDENCE_DOMAIN = b"symthaea.continuity.actuation-enforcement-campaign-evidence-manifest.v1\0"
PREIMAGE_DOMAIN = b"symthaea.continuity.actuation-enforcement-campaign-preimage.v1\0"

OBLIGATIONS = (
    ("boundary_identity", "static_implementation_inspection"),
    ("same_boundary_checks_and_mutates", "atomic_check_and_actuate_scenario"),
    ("durable_monotonic_fence", "crash_restart_scenario"),
    ("reject_stale_generation", "stale_holder_scenario"),
    ("reject_replay", "replay_scenario"),
    ("reject_deny_disposition", "deny_scenario"),
    ("emergency_stop_dominates", "emergency_stop_scenario"),
    ("one_use_permit_consumption", "one_use_consumption_scenario"),
    ("crash_recovery_preserves_fence", "crash_restart_scenario"),
)
OBLIGATION_NAMES = tuple(name for name, _ in OBLIGATIONS)
OBLIGATION_BASES = dict(OBLIGATIONS)

EXPECTED_PROPERTIES = (
    "paused_stale_holder_rejected",
    "one_use_replay_rejected",
    "concurrent_one_use_race_serialized",
    "same_generation_token_substitution_rejected",
    "committed_state_survives_reopen",
    "crash_before_commit_is_atomic",
    "crash_after_commit_is_durable",
    "emergency_stop_dominates",
    "deny_token_cannot_mutate",
    "cross_resource_token_rejected",
    "nine_obligation_campaign_shape_accepted",
)

BUNDLE_FILES = (
    "campaign-manifest.json",
    "campaign-observations.json",
    "campaign-run-context.json",
    "campaign-summary.json",
    "SHA256SUMS",
)
CHECKSUMMED_FILES = BUNDLE_FILES[:-1]

MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "complete_set_id",
        "enforcement_profile_id",
        "authentication_profile_id",
        "backend_id",
        "backend_implementation_digest",
        "backend_generation",
        "boundary_implementation_digest",
        "one_use_mechanism_digest",
        "enforcement_profile_generation",
        "campaign_nonce",
        "harness_implementation_digest",
        "scenario_suite_manifest_digest",
        "environment_manifest_digest",
        "topology_dependency_manifest_digest",
        "hardware_firmware_manifest_digest",
        "toolchain_realization_digest",
        "started_at_unix_ms",
        "ended_at_unix_ms",
        "records",
    }
)
MANIFEST_RECORD_FIELDS = frozenset({"obligation", "record_id", "observed_at_unix_ms"})
OBSERVATIONS_FIELDS = frozenset({"schema", "obligation_count", "observations"})
OBSERVATION_FIELDS = frozenset(
    {"obligation", "required_basis", "observed_at_unix_ms", "record_id", "evidence"}
)
RUN_CONTEXT_FIELDS = frozenset(
    {
        "schema",
        "subject_sha",
        "repository",
        "workflow",
        "run_id",
        "run_attempt",
        "job",
        "runner_os",
        "python",
        "sqlite",
        "platform",
    }
)
SUMMARY_FIELDS = frozenset(
    {
        "schema",
        "status",
        "final_generation",
        "final_value",
        "consumed_count",
        "campaign_manifest_sha256",
        "campaign_oracle",
        "obligation_bases",
        "properties",
        "subject_sha",
        "obligation_count",
    }
)
ORACLE_SUMMARY_FIELDS = frozenset(
    {
        "schema",
        "obligation_count",
        "backend_generation",
        "enforcement_profile_generation",
        "started_at_unix_ms",
        "ended_at_unix_ms",
        "evidence_manifest_sha256",
        "canonical_preimage_sha256",
    }
)

HEX32_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class AuditError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def exact_keys(obj: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    got = set(obj)
    if got != set(allowed):
        missing = sorted(set(allowed) - got)
        extra = sorted(got - set(allowed))
        raise AuditError(f"{where}: exact fields required missing={missing} extra={extra}")


def digest32(value: Any, field: str) -> bytes:
    if not isinstance(value, str) or not HEX32_RE.fullmatch(value):
        raise AuditError(f"{field}: expected 64 lowercase hex characters")
    raw = bytes.fromhex(value)
    if raw == b"\x00" * 32:
        raise AuditError(f"{field}: zero digest forbidden")
    return raw


def positive_u64(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > (1 << 64) - 1:
        raise AuditError(f"{field}: expected positive u64")
    return value


def load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"{path.name}: cannot load JSON: {exc}") from exc


def verify_file_set(root: pathlib.Path) -> None:
    observed = sorted(p.name for p in root.iterdir() if p.is_file())
    expected = sorted(BUNDLE_FILES)
    if observed != expected:
        raise AuditError(f"bundle file set mismatch expected={expected} observed={observed}")


def verify_sha256sums(root: pathlib.Path) -> str:
    lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if len(lines) != len(CHECKSUMMED_FILES):
        raise AuditError("SHA256SUMS: expected exactly four entries")
    seen: set[str] = set()
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or not HEX32_RE.fullmatch(parts[0]):
            raise AuditError(f"SHA256SUMS: invalid line {line!r}")
        digest, name = parts
        if name not in CHECKSUMMED_FILES or name in seen:
            raise AuditError(f"SHA256SUMS: unexpected or duplicate file {name!r}")
        seen.add(name)
        actual = sha256((root / name).read_bytes())
        if actual != digest:
            raise AuditError(f"SHA256SUMS: digest mismatch for {name}")
    if seen != set(CHECKSUMMED_FILES):
        raise AuditError("SHA256SUMS: incomplete file set")
    return sha256((root / "SHA256SUMS").read_bytes())


def observation_id(obligation: str, basis: str, evidence: Any) -> str:
    h = hashlib.sha256()
    h.update(RECORD_DOMAIN)
    for value in (obligation, basis):
        raw = value.encode("utf-8")
        h.update(len(raw).to_bytes(2, "little"))
        h.update(raw)
    h.update(hashlib.sha256(canonical(evidence)).digest())
    return h.hexdigest()


def audit_bundle(root: pathlib.Path, expected_subject: str | None = None) -> dict[str, Any]:
    if not root.is_dir():
        raise AuditError(f"not a directory: {root}")
    verify_file_set(root)
    sha256sums_sha256 = verify_sha256sums(root)

    manifest = load_json(root / "campaign-manifest.json")
    observations = load_json(root / "campaign-observations.json")
    context = load_json(root / "campaign-run-context.json")
    summary = load_json(root / "campaign-summary.json")
    for name, obj in (
        ("manifest", manifest),
        ("observations", observations),
        ("run context", context),
        ("summary", summary),
    ):
        if not isinstance(obj, dict):
            raise AuditError(f"{name}: expected object")

    exact_keys(manifest, MANIFEST_FIELDS, "manifest")
    exact_keys(observations, OBSERVATIONS_FIELDS, "observations")
    exact_keys(context, RUN_CONTEXT_FIELDS, "run context")
    exact_keys(summary, SUMMARY_FIELDS, "summary")

    if manifest["schema"] != CAMPAIGN_SCHEMA:
        raise AuditError("manifest: wrong schema")
    if observations["schema"] != OBSERVATIONS_SCHEMA:
        raise AuditError("observations: wrong schema")
    if context["schema"] != RUN_CONTEXT_SCHEMA:
        raise AuditError("run context: wrong schema")
    if summary["schema"] != BUNDLE_SCHEMA:
        raise AuditError("summary: wrong schema")

    subject = context["subject_sha"]
    if not isinstance(subject, str) or not GIT_SHA_RE.fullmatch(subject):
        raise AuditError("run context: invalid subject SHA")
    if summary["subject_sha"] != subject:
        raise AuditError("summary/run-context subject mismatch")
    if expected_subject is not None and subject != expected_subject.lower():
        raise AuditError(f"unexpected subject: expected={expected_subject.lower()} actual={subject}")

    if observations["obligation_count"] != len(OBLIGATIONS):
        raise AuditError("observations: wrong obligation_count")
    rows = observations["observations"]
    if not isinstance(rows, list) or len(rows) != len(OBLIGATIONS):
        raise AuditError("observations: expected exactly nine rows")

    by_name: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AuditError(f"observations[{i}]: expected object")
        exact_keys(row, OBSERVATION_FIELDS, f"observations[{i}]")
        obligation = row["obligation"]
        if obligation not in OBLIGATION_BASES or obligation in by_name:
            raise AuditError(f"observations[{i}]: unknown or duplicate obligation")
        expected_basis = OBLIGATION_BASES[obligation]
        if row["required_basis"] != expected_basis:
            raise AuditError(f"{obligation}: wrong evidence basis")
        positive_u64(row["observed_at_unix_ms"], f"{obligation}.observed_at_unix_ms")
        digest32(row["record_id"], f"{obligation}.record_id")
        if row["record_id"] in seen_ids:
            raise AuditError(f"{obligation}: duplicate record ID")
        seen_ids.add(row["record_id"])
        expected_id = observation_id(obligation, expected_basis, row["evidence"])
        if row["record_id"] != expected_id:
            raise AuditError(f"{obligation}: observation ID mismatch")
        by_name[obligation] = row

    if tuple(by_name.keys()) != OBLIGATION_NAMES:
        raise AuditError("observations: V1 obligation order mismatch")

    start = positive_u64(manifest["started_at_unix_ms"], "manifest.started_at_unix_ms")
    end = positive_u64(manifest["ended_at_unix_ms"], "manifest.ended_at_unix_ms")
    if end < start:
        raise AuditError("manifest: campaign interval reversed")

    records = manifest["records"]
    if not isinstance(records, list) or len(records) != len(OBLIGATIONS):
        raise AuditError("manifest: expected exactly nine records")
    record_ids: list[str] = []
    for i, (record, (obligation, _basis)) in enumerate(zip(records, OBLIGATIONS, strict=True)):
        if not isinstance(record, dict):
            raise AuditError(f"manifest.records[{i}]: expected object")
        exact_keys(record, MANIFEST_RECORD_FIELDS, f"manifest.records[{i}]")
        if record["obligation"] != obligation:
            raise AuditError("manifest: V1 obligation order mismatch")
        row = by_name[obligation]
        if record["record_id"] != row["record_id"] or record["observed_at_unix_ms"] != row["observed_at_unix_ms"]:
            raise AuditError(f"{obligation}: manifest/observation mismatch")
        if not (start <= row["observed_at_unix_ms"] <= end):
            raise AuditError(f"{obligation}: observation outside campaign interval")
        record_ids.append(row["record_id"])

    complete_set_id = sha256(COMPLETE_DOMAIN + b"".join(bytes.fromhex(x) for x in record_ids))
    if manifest["complete_set_id"] != complete_set_id:
        raise AuditError("manifest: complete_set_id mismatch")

    digest_fields = (
        "complete_set_id",
        "enforcement_profile_id",
        "authentication_profile_id",
        "backend_id",
        "backend_implementation_digest",
        "boundary_implementation_digest",
        "one_use_mechanism_digest",
        "campaign_nonce",
        "harness_implementation_digest",
        "scenario_suite_manifest_digest",
        "environment_manifest_digest",
        "topology_dependency_manifest_digest",
        "hardware_firmware_manifest_digest",
        "toolchain_realization_digest",
    )
    for field in digest_fields:
        digest32(manifest[field], f"manifest.{field}")
    backend_generation = positive_u64(manifest["backend_generation"], "manifest.backend_generation")
    enforcement_generation = positive_u64(
        manifest["enforcement_profile_generation"], "manifest.enforcement_profile_generation"
    )

    evidence_hasher = hashlib.sha256()
    evidence_hasher.update(EVIDENCE_DOMAIN)
    for obligation in OBLIGATION_NAMES:
        row = by_name[obligation]
        name = obligation.encode("ascii")
        evidence_hasher.update(len(name).to_bytes(2, "little"))
        evidence_hasher.update(name)
        evidence_hasher.update(bytes.fromhex(row["record_id"]))
        evidence_hasher.update(row["observed_at_unix_ms"].to_bytes(8, "little"))
    evidence_manifest_sha256 = evidence_hasher.digest()

    preimage = bytearray(PREIMAGE_DOMAIN)
    for field in (
        "complete_set_id",
        "enforcement_profile_id",
        "authentication_profile_id",
        "backend_id",
        "backend_implementation_digest",
    ):
        preimage.extend(bytes.fromhex(manifest[field]))
    preimage.extend(backend_generation.to_bytes(8, "little"))
    for field in ("boundary_implementation_digest", "one_use_mechanism_digest"):
        preimage.extend(bytes.fromhex(manifest[field]))
    preimage.extend(enforcement_generation.to_bytes(8, "little"))
    for field in (
        "campaign_nonce",
        "harness_implementation_digest",
        "scenario_suite_manifest_digest",
        "environment_manifest_digest",
        "topology_dependency_manifest_digest",
        "hardware_firmware_manifest_digest",
        "toolchain_realization_digest",
    ):
        preimage.extend(bytes.fromhex(manifest[field]))
    preimage.extend(start.to_bytes(8, "little"))
    preimage.extend(end.to_bytes(8, "little"))
    preimage.extend(evidence_manifest_sha256)
    canonical_preimage_sha256 = sha256(preimage)

    manifest_sha256 = sha256(canonical(manifest))
    if summary["campaign_manifest_sha256"] != manifest_sha256:
        raise AuditError("summary: campaign_manifest_sha256 mismatch")

    oracle = summary["campaign_oracle"]
    if not isinstance(oracle, dict):
        raise AuditError("summary.campaign_oracle: expected object")
    exact_keys(oracle, ORACLE_SUMMARY_FIELDS, "summary.campaign_oracle")
    expected_oracle = {
        "schema": CAMPAIGN_SCHEMA,
        "obligation_count": len(OBLIGATIONS),
        "backend_generation": backend_generation,
        "enforcement_profile_generation": enforcement_generation,
        "started_at_unix_ms": start,
        "ended_at_unix_ms": end,
        "evidence_manifest_sha256": evidence_manifest_sha256.hex(),
        "canonical_preimage_sha256": canonical_preimage_sha256,
    }
    if oracle != expected_oracle:
        raise AuditError("summary: campaign oracle reconstruction mismatch")

    if summary["status"] != "PASS":
        raise AuditError("summary: status is not PASS")
    if summary["obligation_count"] != len(OBLIGATIONS):
        raise AuditError("summary: wrong obligation_count")
    if summary["obligation_bases"] != OBLIGATION_BASES:
        raise AuditError("summary: obligation basis mapping mismatch")
    if summary["properties"] != list(EXPECTED_PROPERTIES):
        raise AuditError("summary: V1 property set/order mismatch")
    if summary["final_generation"] != 7 or summary["final_value"] != 13 or summary["consumed_count"] != 3:
        raise AuditError("summary: unexpected final V1 contained-lab state")

    return {
        "status": "PASS",
        "subject_sha": subject,
        "sha256sums_sha256": sha256sums_sha256,
        "complete_set_id": complete_set_id,
        "campaign_manifest_sha256": manifest_sha256,
        "evidence_manifest_sha256": evidence_manifest_sha256.hex(),
        "canonical_preimage_sha256": canonical_preimage_sha256,
        "obligation_count": len(OBLIGATIONS),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=pathlib.Path)
    parser.add_argument("--expected-subject")
    args = parser.parse_args()
    try:
        report = audit_bundle(args.evidence_dir, args.expected_subject)
    except AuditError as exc:
        print(f"DENY: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
