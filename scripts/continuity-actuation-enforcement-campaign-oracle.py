#!/usr/bin/env python3
"""Independent structural oracle for actuation-enforcement campaign manifests.

This script grants no authority and does not validate that evidence is truthful.
It only checks that one manifest is structurally closed, campaign-coherent, and
canonically encoded for independent parity with a future Rust implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Collection

SCHEMA = "symthaea-continuity-actuation-enforcement-campaign-manifest-v1"
PREIMAGE_DOMAIN = b"symthaea.continuity.actuation-enforcement-campaign-preimage.v1\0"

OBLIGATIONS = (
    "boundary_identity",
    "same_boundary_checks_and_mutates",
    "durable_monotonic_fence",
    "reject_stale_generation",
    "reject_replay",
    "reject_deny_disposition",
    "emergency_stop_dominates",
    "one_use_permit_consumption",
    "crash_recovery_preserves_fence",
)

DIGEST_FIELDS = (
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

GENERATION_FIELDS = (
    "backend_generation",
    "enforcement_profile_generation",
)

TOP_LEVEL_FIELDS = frozenset(
    (
        "schema",
        *DIGEST_FIELDS,
        *GENERATION_FIELDS,
        "started_at_unix_ms",
        "ended_at_unix_ms",
        "records",
    )
)
RECORD_FIELDS = frozenset(("obligation", "record_id", "observed_at_unix_ms"))


class ManifestError(ValueError):
    pass


def require_exact_keys(obj: dict[str, Any], allowed: Collection[str], context: str) -> None:
    observed = set(obj)
    allowed_set = set(allowed)
    missing = sorted(allowed_set - observed)
    extra = sorted(observed - allowed_set)
    if missing or extra:
        pieces = []
        if missing:
            pieces.append(f"missing={','.join(missing)}")
        if extra:
            pieces.append(f"unexpected={','.join(extra)}")
        raise ManifestError(f"{context}: exact field set required ({'; '.join(pieces)})")


def parse_digest(value: Any, field: str) -> bytes:
    if not isinstance(value, str) or len(value) != 64:
        raise ManifestError(f"{field}: expected 64 lowercase hex characters")
    if value.lower() != value:
        raise ManifestError(f"{field}: hex must be lowercase")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ManifestError(f"{field}: invalid hex") from exc
    if raw == b"\x00" * 32:
        raise ManifestError(f"{field}: zero digest is forbidden")
    return raw


def parse_u64(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestError(f"{field}: expected positive integer")
    if value > (1 << 64) - 1:
        raise ManifestError(f"{field}: exceeds u64")
    return value


def encode_u64(value: int) -> bytes:
    return value.to_bytes(8, "little", signed=False)


def validate_manifest(obj: Any) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(obj, dict):
        raise ManifestError("manifest must be a JSON object")
    require_exact_keys(obj, TOP_LEVEL_FIELDS, "manifest")
    if obj["schema"] != SCHEMA:
        raise ManifestError(f"schema: expected {SCHEMA!r}")

    digests = {field: parse_digest(obj[field], field) for field in DIGEST_FIELDS}
    backend_generation = parse_u64(obj["backend_generation"], "backend_generation")
    enforcement_profile_generation = parse_u64(
        obj["enforcement_profile_generation"], "enforcement_profile_generation"
    )
    start = parse_u64(obj["started_at_unix_ms"], "started_at_unix_ms")
    end = parse_u64(obj["ended_at_unix_ms"], "ended_at_unix_ms")
    if end < start:
        raise ManifestError("campaign end precedes campaign start")

    records = obj["records"]
    if not isinstance(records, list) or len(records) != len(OBLIGATIONS):
        raise ManifestError(f"records: expected exactly {len(OBLIGATIONS)} entries")

    by_obligation: dict[str, tuple[bytes, int]] = {}
    seen_record_ids: set[bytes] = set()
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ManifestError(f"records[{i}]: expected object")
        require_exact_keys(rec, RECORD_FIELDS, f"records[{i}]")
        obligation = rec["obligation"]
        if obligation not in OBLIGATIONS:
            raise ManifestError(f"records[{i}].obligation: unexpected {obligation!r}")
        if obligation in by_obligation:
            raise ManifestError(f"duplicate obligation: {obligation}")
        record_id = parse_digest(rec["record_id"], f"records[{i}].record_id")
        if record_id in seen_record_ids:
            raise ManifestError(f"duplicate record_id at records[{i}]")
        seen_record_ids.add(record_id)
        observed = parse_u64(
            rec["observed_at_unix_ms"],
            f"records[{i}].observed_at_unix_ms",
        )
        if not (start <= observed <= end):
            raise ManifestError(
                f"{obligation}: observation {observed} outside campaign [{start}, {end}]"
            )
        by_obligation[obligation] = (record_id, observed)

    missing = [name for name in OBLIGATIONS if name not in by_obligation]
    if missing:
        raise ManifestError(f"missing obligations: {', '.join(missing)}")

    evidence_hasher = hashlib.sha256()
    evidence_hasher.update(
        b"symthaea.continuity.actuation-enforcement-campaign-evidence-manifest.v1\0"
    )
    for obligation in OBLIGATIONS:
        record_id, observed = by_obligation[obligation]
        name = obligation.encode("ascii")
        evidence_hasher.update(len(name).to_bytes(2, "little"))
        evidence_hasher.update(name)
        evidence_hasher.update(record_id)
        evidence_hasher.update(encode_u64(observed))
    evidence_manifest_sha256 = evidence_hasher.digest()

    out = bytearray(PREIMAGE_DOMAIN)
    out.extend(digests["complete_set_id"])
    out.extend(digests["enforcement_profile_id"])
    out.extend(digests["authentication_profile_id"])
    out.extend(digests["backend_id"])
    out.extend(digests["backend_implementation_digest"])
    out.extend(encode_u64(backend_generation))
    out.extend(digests["boundary_implementation_digest"])
    out.extend(digests["one_use_mechanism_digest"])
    out.extend(encode_u64(enforcement_profile_generation))
    out.extend(digests["campaign_nonce"])
    out.extend(digests["harness_implementation_digest"])
    out.extend(digests["scenario_suite_manifest_digest"])
    out.extend(digests["environment_manifest_digest"])
    out.extend(digests["topology_dependency_manifest_digest"])
    out.extend(digests["hardware_firmware_manifest_digest"])
    out.extend(digests["toolchain_realization_digest"])
    out.extend(encode_u64(start))
    out.extend(encode_u64(end))
    out.extend(evidence_manifest_sha256)

    summary = {
        "schema": SCHEMA,
        "obligation_count": len(OBLIGATIONS),
        "backend_generation": backend_generation,
        "enforcement_profile_generation": enforcement_profile_generation,
        "started_at_unix_ms": start,
        "ended_at_unix_ms": end,
        "evidence_manifest_sha256": evidence_manifest_sha256.hex(),
        "canonical_preimage_sha256": hashlib.sha256(out).hexdigest(),
    }
    return bytes(out), summary


def self_test() -> None:
    def hx(n: int) -> str:
        return (bytes([n]) * 32).hex()

    base = {
        "schema": SCHEMA,
        **{field: hx(i + 1) for i, field in enumerate(DIGEST_FIELDS)},
        "backend_generation": 7,
        "enforcement_profile_generation": 3,
        "started_at_unix_ms": 1000,
        "ended_at_unix_ms": 2000,
        "records": [
            {
                "obligation": obligation,
                "record_id": hx(32 + i),
                "observed_at_unix_ms": 1100 + i,
            }
            for i, obligation in enumerate(OBLIGATIONS)
        ],
    }
    p1, s1 = validate_manifest(base)
    p2, s2 = validate_manifest({**base, "records": list(reversed(base["records"]))})
    assert p1 == p2 and s1 == s2, "input record ordering must not affect canonical preimage"

    bad = json.loads(json.dumps(base))
    bad["records"][0]["observed_at_unix_ms"] = 999
    try:
        validate_manifest(bad)
    except ManifestError:
        pass
    else:
        raise AssertionError("out-of-window observation accepted")

    bad = json.loads(json.dumps(base))
    bad["records"][1]["obligation"] = bad["records"][0]["obligation"]
    try:
        validate_manifest(bad)
    except ManifestError:
        pass
    else:
        raise AssertionError("duplicate obligation accepted")

    bad = json.loads(json.dumps(base))
    bad["environment_manifest_digest"] = "00" * 32
    try:
        validate_manifest(bad)
    except ManifestError:
        pass
    else:
        raise AssertionError("zero environment digest accepted")

    bad = json.loads(json.dumps(base))
    bad["records"][8]["record_id"] = bad["records"][7]["record_id"]
    try:
        validate_manifest(bad)
    except ManifestError:
        pass
    else:
        raise AssertionError("duplicate record id accepted")

    bad = json.loads(json.dumps(base))
    bad["shadow_policy"] = "permit"
    try:
        validate_manifest(bad)
    except ManifestError:
        pass
    else:
        raise AssertionError("unknown top-level field accepted")

    bad = json.loads(json.dumps(base))
    bad["records"][0]["shadow_result"] = "satisfied"
    try:
        validate_manifest(bad)
    except ManifestError:
        pass
    else:
        raise AssertionError("unknown record field accepted")

    changed = json.loads(json.dumps(base))
    changed["toolchain_realization_digest"] = hx(99)
    p3, _ = validate_manifest(changed)
    assert p1 != p3, "toolchain drift must change canonical preimage"

    changed = json.loads(json.dumps(base))
    changed["backend_generation"] += 1
    p4, _ = validate_manifest(changed)
    assert p1 != p4, "backend generation drift must change canonical preimage"

    changed = json.loads(json.dumps(base))
    changed["enforcement_profile_id"] = hx(99)
    p5, _ = validate_manifest(changed)
    assert p1 != p5, "enforcement-profile substitution must change canonical preimage"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", help="campaign JSON manifest")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--emit-preimage-hex", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("PASS: campaign oracle self-tests")
        return 0
    if not args.manifest:
        parser.error("manifest is required unless --self-test is used")

    try:
        with open(args.manifest, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        preimage, summary = validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        print(f"DENY: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, sort_keys=True))
    if args.emit_preimage_hex:
        print(preimage.hex())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
