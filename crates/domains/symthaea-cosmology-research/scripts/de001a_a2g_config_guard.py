#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DE-001A2G deterministic configuration-resolution guard.

This program performs no YAML parsing, no likelihood evaluation, no sampler
construction, and no optimization. It binds a qualified A2R receipt to the
same exact A0 configuration bytes and rejects configuration directives whose
meaning could depend on ambient files or environment variables.

Exit codes:
  0: PASS deterministic-resolution guard
  2: INVALID evidence/configuration
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROTOCOL = "DE-001A2G-CONFIG-RESOLUTION-GUARD-v1"
AUTHORITY = "configuration-resolution-determinism-only"
A0_PROTOCOL = "DE-001A0-BYTE-INTEGRITY-v1"
A2R_PROTOCOL = "DE-001A2R-PARAMETER-ROLE-BINDING-v1"
MAX_FILE_BYTES = 1024 * 1024

ARTIFACTS = {
    "reference-input-configuration": {
        "size": 2381,
        "sha256": "34499cb78ecaec78db44da9f06f61cd9c9ee497dc5c541b72b48cda54091c6ef",
        "a2r_key": "reference_input",
    },
    "reference-expanded-configuration": {
        "size": 3969,
        "sha256": "c4c23032d1695635aaea6eb47fabd909006ff32df0a27eadbf64b36c89f31ba1",
        "a2r_key": "reference_expanded",
    },
    "reference-minimizer-configuration": {
        "size": 2484,
        "sha256": "6b51048b359e4b9d646de09379ca273a8d6a1bc1b5a88f585a09d8f2e61f290c",
        "a2r_key": "reference_minimizer",
    },
}

FORBIDDEN_MARKERS = (
    ("defaults_directive", "!defaults"),
    ("path_directive", "!path"),
    ("environment_placeholder", "${"),
)


class InvalidGuard(RuntimeError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_regular(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise InvalidGuard(f"{path}: metadata failed: {exc}") from exc
    if path.is_symlink():
        raise InvalidGuard(f"{path}: symlinks are forbidden")
    if not path.is_file():
        raise InvalidGuard(f"{path}: not a regular file")
    if metadata.st_size > MAX_FILE_BYTES:
        raise InvalidGuard(f"{path}: exceeds {MAX_FILE_BYTES} bytes")
    data = path.read_bytes()
    if len(data) != metadata.st_size:
        raise InvalidGuard(f"{path}: size changed while reading")
    return data


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidGuard(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    data = read_regular(path)
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                InvalidGuard(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidGuard(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidGuard(f"{path}: top-level JSON must be an object")
    return data, value


def require_str(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise InvalidGuard(f"{key}: expected string")
    return result


def require_bool(value: dict[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise InvalidGuard(f"{key}: expected boolean")
    return result


def validate_a0(receipt_bytes: bytes, receipt: dict[str, Any]) -> None:
    if require_str(receipt, "protocol") != A0_PROTOCOL:
        raise InvalidGuard("A0 receipt protocol mismatch")
    if require_str(receipt, "verdict") != "PASS":
        raise InvalidGuard("A0 receipt is not PASS")
    if require_str(receipt, "scientific_claim") != "NONE":
        raise InvalidGuard("A0 receipt scientific_claim is not NONE")
    if receipt.get("errors") != []:
        raise InvalidGuard("A0 receipt contains errors")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise InvalidGuard("A0 artifacts is not an array")

    by_role: dict[str, dict[str, Any]] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise InvalidGuard("A0 artifact entry is not an object")
        role = entry.get("role")
        if not isinstance(role, str) or role in by_role:
            raise InvalidGuard("A0 artifact role missing or duplicated")
        by_role[role] = entry

    for role, expected in ARTIFACTS.items():
        entry = by_role.get(role)
        if entry is None:
            raise InvalidGuard(f"A0 receipt missing {role}")
        if entry.get("status") != "PASS":
            raise InvalidGuard(f"A0 {role} is not PASS")
        if entry.get("expected_size") != expected["size"] or entry.get("actual_size") != expected["size"]:
            raise InvalidGuard(f"A0 {role} size mismatch")
        if entry.get("expected_sha256") != expected["sha256"] or entry.get("actual_sha256") != expected["sha256"]:
            raise InvalidGuard(f"A0 {role} SHA-256 mismatch")

    if not receipt_bytes:
        raise InvalidGuard("A0 receipt is empty")


def validate_a2r(
    a2r_bytes: bytes,
    a2r: dict[str, Any],
    a0_sha256: str,
) -> None:
    if require_str(a2r, "protocol") != A2R_PROTOCOL:
        raise InvalidGuard("A2R protocol mismatch")
    if require_str(a2r, "verdict") != "PASS":
        raise InvalidGuard("A2R receipt is not PASS")
    if require_str(a2r, "scientific_claim") != "NONE":
        raise InvalidGuard("A2R scientific_claim is not NONE")
    if require_str(a2r, "authority") != "optimizer-parameter-role-binding-only":
        raise InvalidGuard("A2R authority mismatch")
    if require_bool(a2r, "a2_execution_authorized"):
        raise InvalidGuard("A2R unexpectedly authorizes A2")
    if require_str(a2r, "a0_receipt_sha256") != a0_sha256:
        raise InvalidGuard("A2R is not bound to this A0 receipt")

    hashes = a2r.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise InvalidGuard("A2R artifact_sha256 is not an object")
    for expected in ARTIFACTS.values():
        if hashes.get(expected["a2r_key"]) != expected["sha256"]:
            raise InvalidGuard(
                f"A2R artifact hash mismatch for {expected['a2r_key']}"
            )

    sampled = a2r.get("sampled_coordinate_order")
    if not isinstance(sampled, list) or not sampled:
        raise InvalidGuard("A2R sampled coordinate order is empty/invalid")
    if any(not isinstance(name, str) or not name for name in sampled):
        raise InvalidGuard("A2R sampled coordinate order contains invalid names")
    if len(sampled) != len(set(sampled)):
        raise InvalidGuard("A2R sampled coordinate order contains duplicates")

    if not a2r_bytes:
        raise InvalidGuard("A2R receipt is empty")


def validate_config(path: Path, role: str) -> tuple[bytes, dict[str, bool]]:
    expected = ARTIFACTS[role]
    if path.name != role:
        raise InvalidGuard(
            f"{role}: expected basename {role!r}, got {path.name!r}"
        )
    data = read_regular(path)
    if len(data) != expected["size"]:
        raise InvalidGuard(f"{role}: size mismatch")
    if sha256_hex(data) != expected["sha256"]:
        raise InvalidGuard(f"{role}: SHA-256 mismatch")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidGuard(f"{role}: not UTF-8") from exc

    marker_absence: dict[str, bool] = {}
    for marker_name, marker in FORBIDDEN_MARKERS:
        absent = marker not in text
        marker_absence[marker_name] = absent
        if not absent:
            raise InvalidGuard(
                f"{role}: forbidden ambient-resolution marker {marker!r} present"
            )
    return data, marker_absence


def write_json(value: dict[str, Any]) -> None:
    json.dump(
        value,
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a0_receipt", type=Path)
    parser.add_argument("a2r_receipt", type=Path)
    parser.add_argument("reference_input", type=Path)
    parser.add_argument("reference_expanded", type=Path)
    parser.add_argument("reference_minimizer", type=Path)
    args = parser.parse_args()

    try:
        a0_bytes, a0 = read_json(args.a0_receipt)
        a2r_bytes, a2r = read_json(args.a2r_receipt)
        validate_a0(a0_bytes, a0)
        a0_sha256 = sha256_hex(a0_bytes)
        validate_a2r(a2r_bytes, a2r, a0_sha256)

        config_paths = {
            "reference-input-configuration": args.reference_input,
            "reference-expanded-configuration": args.reference_expanded,
            "reference-minimizer-configuration": args.reference_minimizer,
        }
        config_hashes: dict[str, str] = {}
        marker_audit: dict[str, dict[str, bool]] = {}
        for role, path in config_paths.items():
            data, audit = validate_config(path, role)
            config_hashes[role] = sha256_hex(data)
            marker_audit[role] = audit

        write_json(
            {
                "protocol": PROTOCOL,
                "verdict": "PASS",
                "scientific_claim": "NONE",
                "authority": AUTHORITY,
                "a2m_execution_authorized": False,
                "a2_execution_authorized": False,
                "a0_receipt_sha256": a0_sha256,
                "a2r_receipt_sha256": sha256_hex(a2r_bytes),
                "configuration_sha256": config_hashes,
                "forbidden_markers": {
                    name: marker for name, marker in FORBIDDEN_MARKERS
                },
                "marker_absence": marker_audit,
                "resolution_policy": (
                    "A2M may consume this lineage only when all exact A0-qualified "
                    "input/expanded/minimizer configuration bytes contain no "
                    "!defaults, !path, or ${...} ambient-resolution markers"
                ),
            }
        )
        return 0
    except (InvalidGuard, OSError, ValueError, TypeError, KeyError) as exc:
        write_json(
            {
                "protocol": PROTOCOL,
                "verdict": "INVALID",
                "scientific_claim": "NONE",
                "authority": AUTHORITY,
                "a2m_execution_authorized": False,
                "a2_execution_authorized": False,
                "error": str(exc),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
