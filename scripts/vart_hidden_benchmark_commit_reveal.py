#!/usr/bin/env python3
"""Salted commit/reveal tooling for hidden VART benchmark custody.

The public commitment manifest contains no fixture identifiers, seeds, nonces,
solutions, trap labels, or target defects. A private source with independent
256-bit nonces can be revealed after the campaign and verified byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any

SCHEMA_SOURCE = "symthaea.vart-hidden-benchmark-source.v1"
SCHEMA_PUBLIC = "symthaea.vart-hidden-benchmark-public-commitments.v1"
DOMAIN = b"SYMTHAEA-VART-HIDDEN-COMMIT-v1\x00"
HEX = set("0123456789abcdef")
FORBIDDEN_PUBLIC_KEYS = {
    "fixture_id",
    "fixture_ids",
    "seed",
    "seeds",
    "nonce",
    "nonce_hex",
    "plaintext",
    "expected_solution",
    "expected_solutions",
    "target_defect",
    "target_defects",
    "trap_label",
    "trap_labels",
}


class CommitRevealError(ValueError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: pathlib.Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CommitRevealError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CommitRevealError(f"{path} must contain a JSON object")
    return value, raw


def require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CommitRevealError(f"{name} must be a non-empty string")
    return value


def require_nonce(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        raise CommitRevealError(f"{name} must be 64 lowercase hex characters (256-bit nonce)")
    return value


def normalize_source(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("schema") != SCHEMA_SOURCE:
        raise CommitRevealError(f"source schema must be {SCHEMA_SOURCE}")
    campaign_id = require_nonempty_string(source.get("campaign_id"), "campaign_id")
    custodian_id = require_nonempty_string(source.get("custodian_id"), "custodian_id")

    fixtures = source.get("fixtures")
    seeds = source.get("seeds")
    if not isinstance(fixtures, list) or not fixtures:
        raise CommitRevealError("fixtures must be a non-empty list")
    if not isinstance(seeds, list) or not seeds:
        raise CommitRevealError("seeds must be a non-empty list")

    normalized_fixtures: list[dict[str, Any]] = []
    normalized_seeds: list[dict[str, Any]] = []
    nonces: set[str] = set()

    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, dict):
            raise CommitRevealError(f"fixtures[{index}] must be an object")
        fixture_id = require_nonempty_string(fixture.get("fixture_id"), f"fixtures[{index}].fixture_id")
        nonce = require_nonce(fixture.get("nonce_hex"), f"fixtures[{index}].nonce_hex")
        if nonce in nonces:
            raise CommitRevealError("every hidden fixture/seed must use an independent nonce")
        nonces.add(nonce)
        normalized_fixtures.append({"fixture_id": fixture_id, "nonce_hex": nonce})

    for index, seed in enumerate(seeds):
        if not isinstance(seed, dict):
            raise CommitRevealError(f"seeds[{index}] must be an object")
        value = seed.get("seed")
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 2**64 - 1:
            raise CommitRevealError(f"seeds[{index}].seed must be an unsigned 64-bit integer")
        nonce = require_nonce(seed.get("nonce_hex"), f"seeds[{index}].nonce_hex")
        if nonce in nonces:
            raise CommitRevealError("every hidden fixture/seed must use an independent nonce")
        nonces.add(nonce)
        normalized_seeds.append({"seed": value, "nonce_hex": nonce})

    return {
        "schema": SCHEMA_SOURCE,
        "campaign_id": campaign_id,
        "custodian_id": custodian_id,
        "fixtures": normalized_fixtures,
        "seeds": normalized_seeds,
    }


def commitment(campaign_id: str, kind: str, value: Any, nonce_hex: str) -> str:
    preimage = {
        "campaign_id": campaign_id,
        "kind": kind,
        "nonce_hex": nonce_hex,
        "value": value,
    }
    return sha256_bytes(DOMAIN + canonical_json_bytes(preimage))


def public_manifest_from_source(source: dict[str, Any], source_raw: bytes) -> dict[str, Any]:
    source = normalize_source(source)
    campaign_id = source["campaign_id"]
    fixture_commitments = sorted(
        commitment(campaign_id, "fixture", item["fixture_id"], item["nonce_hex"])
        for item in source["fixtures"]
    )
    seed_commitments = sorted(
        commitment(campaign_id, "seed", item["seed"], item["nonce_hex"])
        for item in source["seeds"]
    )
    return {
        "schema": SCHEMA_PUBLIC,
        "campaign_id": campaign_id,
        "custodian_id": source["custodian_id"],
        "fixture_count": len(fixture_commitments),
        "seed_count": len(seed_commitments),
        "fixture_commitments_sha256": fixture_commitments,
        "seed_commitments_sha256": seed_commitments,
        "private_source_sha256": sha256_bytes(source_raw),
        "commitment_domain": "SYMTHAEA-VART-HIDDEN-COMMIT-v1",
        "revealed": False,
        "confirmatory_execution_authorized": False,
        "claim_authorized": False,
    }


def validate_public_no_leak(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_PUBLIC_KEYS:
                raise CommitRevealError(f"public commitment manifest leaks forbidden field at {path}.{key}")
            validate_public_no_leak(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_public_no_leak(child, f"{path}[{index}]")


def validate_public_manifest(public: dict[str, Any]) -> None:
    if public.get("schema") != SCHEMA_PUBLIC:
        raise CommitRevealError(f"public schema must be {SCHEMA_PUBLIC}")
    require_nonempty_string(public.get("campaign_id"), "public.campaign_id")
    require_nonempty_string(public.get("custodian_id"), "public.custodian_id")
    if public.get("commitment_domain") != "SYMTHAEA-VART-HIDDEN-COMMIT-v1":
        raise CommitRevealError("unexpected commitment_domain")
    if public.get("revealed") is not False:
        raise CommitRevealError("prospective public commitment manifest must have revealed=false")
    if public.get("confirmatory_execution_authorized") is not False:
        raise CommitRevealError("commitment manifest cannot authorize confirmatory execution")
    if public.get("claim_authorized") is not False:
        raise CommitRevealError("commitment manifest cannot authorize claims")

    for field in ("fixture_commitments_sha256", "seed_commitments_sha256"):
        values = public.get(field)
        if not isinstance(values, list) or not values:
            raise CommitRevealError(f"{field} must be a non-empty list")
        if values != sorted(values) or len(values) != len(set(values)):
            raise CommitRevealError(f"{field} must be unique and lexicographically sorted")
        for value in values:
            if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
                raise CommitRevealError(f"invalid SHA-256 in {field}")

    if public.get("fixture_count") != len(public["fixture_commitments_sha256"]):
        raise CommitRevealError("fixture_count mismatch")
    if public.get("seed_count") != len(public["seed_commitments_sha256"]):
        raise CommitRevealError("seed_count mismatch")
    source_hash = public.get("private_source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in HEX for c in source_hash):
        raise CommitRevealError("private_source_sha256 must be lowercase SHA-256 hex")
    validate_public_no_leak(public)


def verify_reveal(public: dict[str, Any], source: dict[str, Any], source_raw: bytes) -> None:
    validate_public_manifest(public)
    expected = public_manifest_from_source(source, source_raw)
    fields = (
        "campaign_id",
        "custodian_id",
        "fixture_count",
        "seed_count",
        "fixture_commitments_sha256",
        "seed_commitments_sha256",
        "private_source_sha256",
        "commitment_domain",
    )
    for field in fields:
        if public.get(field) != expected.get(field):
            raise CommitRevealError(f"reveal mismatch for {field}")


def write_json(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def cmd_commit(args: argparse.Namespace) -> int:
    source, source_raw = read_json(pathlib.Path(args.source))
    public = public_manifest_from_source(source, source_raw)
    validate_public_manifest(public)
    write_json(pathlib.Path(args.public_out), public)
    print("VART_HIDDEN_BENCHMARK_COMMIT_READY")
    print(f"public_manifest_sha256={sha256_bytes(canonical_json_bytes(public) + b'\n')}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    public, _ = read_json(pathlib.Path(args.public))
    source, source_raw = read_json(pathlib.Path(args.reveal))
    verify_reveal(public, source, source_raw)
    print("VART_HIDDEN_BENCHMARK_REVEAL_VERIFIED")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    commit_parser = sub.add_parser("commit")
    commit_parser.add_argument("--source", required=True)
    commit_parser.add_argument("--public-out", required=True)
    commit_parser.set_defaults(func=cmd_commit)

    verify_parser = sub.add_parser("verify-reveal")
    verify_parser.add_argument("--public", required=True)
    verify_parser.add_argument("--reveal", required=True)
    verify_parser.set_defaults(func=cmd_verify)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        return args.func(args)
    except CommitRevealError as exc:
        print(f"VART_HIDDEN_BENCHMARK_COMMIT_REVEAL_REJECT: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
