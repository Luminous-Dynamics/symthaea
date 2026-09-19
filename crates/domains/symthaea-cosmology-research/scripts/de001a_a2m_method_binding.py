#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DE-001A2M exact optimizer-method/options binding.

This program performs no likelihood evaluation, sampler construction, or
optimization. It consumes qualified A0/A2R/A2G evidence, parses the exact
authenticated DESI configuration bytes in-memory with Cobaya 3.6.2, and binds
the effective expanded sampler component/options for a later A2 execution gate.

Exit codes:
  0: PASS method/options binding
  2: INVALID evidence/configuration
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any

PROTOCOL = "DE-001A2M-OPTIMIZER-METHOD-BINDING-v1"
AUTHORITY = "optimizer-method-options-binding-only"
A0_PROTOCOL = "DE-001A0-BYTE-INTEGRITY-v1"
A2R_PROTOCOL = "DE-001A2R-PARAMETER-ROLE-BINDING-v1"
A2G_PROTOCOL = "DE-001A2G-CONFIG-RESOLUTION-GUARD-v1"
COBAYA_VERSION = "3.6.2"
COBAYA_SOURCE_COMMIT = "899f30a49f85de610dac321e91a1af50018e56aa"
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


class InvalidBinding(RuntimeError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def read_regular(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise InvalidBinding(f"{path}: metadata failed: {exc}") from exc
    if path.is_symlink():
        raise InvalidBinding(f"{path}: symlinks are forbidden")
    if not path.is_file():
        raise InvalidBinding(f"{path}: not a regular file")
    if metadata.st_size > MAX_FILE_BYTES:
        raise InvalidBinding(f"{path}: exceeds {MAX_FILE_BYTES} bytes")
    data = path.read_bytes()
    if len(data) != metadata.st_size:
        raise InvalidBinding(f"{path}: size changed while reading")
    return data


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidBinding(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    data = read_regular(path)
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                InvalidBinding(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidBinding(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidBinding(f"{path}: top-level JSON must be an object")
    return data, value


def require_str(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str):
        raise InvalidBinding(f"{key}: expected string")
    return result


def require_bool(value: dict[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise InvalidBinding(f"{key}: expected boolean")
    return result


def validate_config(path: Path, role: str) -> bytes:
    expected = ARTIFACTS[role]
    if path.name != role:
        raise InvalidBinding(
            f"{role}: expected basename {role!r}, got {path.name!r}"
        )
    data = read_regular(path)
    if len(data) != expected["size"]:
        raise InvalidBinding(f"{role}: size mismatch")
    if sha256_hex(data) != expected["sha256"]:
        raise InvalidBinding(f"{role}: SHA-256 mismatch")
    return data


def validate_a0(receipt_bytes: bytes, receipt: dict[str, Any]) -> None:
    if require_str(receipt, "protocol") != A0_PROTOCOL:
        raise InvalidBinding("A0 protocol mismatch")
    if require_str(receipt, "verdict") != "PASS":
        raise InvalidBinding("A0 receipt is not PASS")
    if require_str(receipt, "scientific_claim") != "NONE":
        raise InvalidBinding("A0 scientific_claim is not NONE")
    if receipt.get("errors") != []:
        raise InvalidBinding("A0 receipt contains errors")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise InvalidBinding("A0 artifacts is not an array")
    by_role: dict[str, dict[str, Any]] = {}
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise InvalidBinding("A0 artifact entry is not an object")
        role = entry.get("role")
        if not isinstance(role, str) or role in by_role:
            raise InvalidBinding("A0 role missing or duplicated")
        by_role[role] = entry
    for role, expected in ARTIFACTS.items():
        entry = by_role.get(role)
        if entry is None or entry.get("status") != "PASS":
            raise InvalidBinding(f"A0 role {role} missing/not PASS")
        if entry.get("expected_size") != expected["size"] or entry.get("actual_size") != expected["size"]:
            raise InvalidBinding(f"A0 role {role} size mismatch")
        if entry.get("expected_sha256") != expected["sha256"] or entry.get("actual_sha256") != expected["sha256"]:
            raise InvalidBinding(f"A0 role {role} SHA-256 mismatch")
    if not receipt_bytes:
        raise InvalidBinding("A0 receipt is empty")


def validate_a2r(a2r_bytes: bytes, a2r: dict[str, Any], a0_sha256: str) -> None:
    if require_str(a2r, "protocol") != A2R_PROTOCOL:
        raise InvalidBinding("A2R protocol mismatch")
    if require_str(a2r, "verdict") != "PASS":
        raise InvalidBinding("A2R receipt is not PASS")
    if require_str(a2r, "scientific_claim") != "NONE":
        raise InvalidBinding("A2R scientific_claim is not NONE")
    if require_str(a2r, "authority") != "optimizer-parameter-role-binding-only":
        raise InvalidBinding("A2R authority mismatch")
    if require_bool(a2r, "a2_execution_authorized"):
        raise InvalidBinding("A2R unexpectedly authorizes A2")
    if require_str(a2r, "a0_receipt_sha256") != a0_sha256:
        raise InvalidBinding("A2R does not bind this A0 receipt")
    hashes = a2r.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise InvalidBinding("A2R artifact_sha256 is not an object")
    for expected in ARTIFACTS.values():
        if hashes.get(expected["a2r_key"]) != expected["sha256"]:
            raise InvalidBinding(
                f"A2R artifact hash mismatch for {expected['a2r_key']}"
            )
    sampled = a2r.get("sampled_coordinate_order")
    if not isinstance(sampled, list) or not sampled:
        raise InvalidBinding("A2R sampled coordinate order is empty/invalid")
    if any(not isinstance(name, str) or not name for name in sampled):
        raise InvalidBinding("A2R sampled coordinate order contains invalid names")
    if len(sampled) != len(set(sampled)):
        raise InvalidBinding("A2R sampled coordinate order contains duplicates")
    if not a2r_bytes:
        raise InvalidBinding("A2R receipt is empty")


def validate_a2g(
    a2g_bytes: bytes,
    a2g: dict[str, Any],
    a0_sha256: str,
    a2r_sha256: str,
) -> None:
    if require_str(a2g, "protocol") != A2G_PROTOCOL:
        raise InvalidBinding("A2G protocol mismatch")
    if require_str(a2g, "verdict") != "PASS":
        raise InvalidBinding("A2G receipt is not PASS")
    if require_str(a2g, "scientific_claim") != "NONE":
        raise InvalidBinding("A2G scientific_claim is not NONE")
    if require_str(a2g, "authority") != "configuration-resolution-determinism-only":
        raise InvalidBinding("A2G authority mismatch")
    if require_bool(a2g, "a2m_execution_authorized"):
        raise InvalidBinding("A2G unexpectedly authorizes A2M execution")
    if require_bool(a2g, "a2_execution_authorized"):
        raise InvalidBinding("A2G unexpectedly authorizes A2")
    if require_str(a2g, "a0_receipt_sha256") != a0_sha256:
        raise InvalidBinding("A2G does not bind this A0 receipt")
    if require_str(a2g, "a2r_receipt_sha256") != a2r_sha256:
        raise InvalidBinding("A2G does not bind this A2R receipt")

    config_hashes = a2g.get("configuration_sha256")
    if not isinstance(config_hashes, dict):
        raise InvalidBinding("A2G configuration_sha256 is not an object")
    for role, expected in ARTIFACTS.items():
        if config_hashes.get(role) != expected["sha256"]:
            raise InvalidBinding(f"A2G configuration hash mismatch for {role}")

    marker_absence = a2g.get("marker_absence")
    if not isinstance(marker_absence, dict):
        raise InvalidBinding("A2G marker_absence is not an object")
    for role in ARTIFACTS:
        audit = marker_absence.get(role)
        if not isinstance(audit, dict) or not audit:
            raise InvalidBinding(f"A2G marker audit missing for {role}")
        if not all(value is True for value in audit.values()):
            raise InvalidBinding(f"A2G marker audit failed for {role}")
    if not a2g_bytes:
        raise InvalidBinding("A2G receipt is empty")


def load_yaml_in_memory(data: bytes, role: str) -> dict[str, Any]:
    from cobaya.yaml import yaml_load

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InvalidBinding(f"{role}: not UTF-8") from exc
    try:
        value = yaml_load(text)
    except Exception as exc:
        raise InvalidBinding(f"{role}: in-memory Cobaya YAML parse failed: {exc}") from exc
    if not isinstance(value, dict):
        raise InvalidBinding(f"{role}: YAML root is not a mapping")
    return value


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidBinding("non-finite float in optimizer metadata")
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    raise InvalidBinding(
        f"unsupported optimizer metadata type: {type(value).__name__}"
    )


def sampler_block(config: dict[str, Any], role: str) -> dict[str, Any] | None:
    value = config.get("sampler")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidBinding(f"{role}: sampler block is not a mapping")
    return json_safe(value)


def write_json(value: dict[str, Any]) -> None:
    json.dump(
        value,
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    sys.stdout.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("a0_receipt", type=Path)
    parser.add_argument("a2r_receipt", type=Path)
    parser.add_argument("a2g_receipt", type=Path)
    parser.add_argument("reference_input", type=Path)
    parser.add_argument("reference_expanded", type=Path)
    parser.add_argument("reference_minimizer", type=Path)
    args = parser.parse_args()

    try:
        if importlib.metadata.version("cobaya") != COBAYA_VERSION:
            raise InvalidBinding("runtime Cobaya version mismatch")

        a0_bytes, a0 = read_json(args.a0_receipt)
        a2r_bytes, a2r = read_json(args.a2r_receipt)
        a2g_bytes, a2g = read_json(args.a2g_receipt)
        validate_a0(a0_bytes, a0)
        a0_sha256 = sha256_hex(a0_bytes)
        validate_a2r(a2r_bytes, a2r, a0_sha256)
        a2r_sha256 = sha256_hex(a2r_bytes)
        validate_a2g(a2g_bytes, a2g, a0_sha256, a2r_sha256)

        input_bytes = validate_config(
            args.reference_input, "reference-input-configuration"
        )
        expanded_bytes = validate_config(
            args.reference_expanded, "reference-expanded-configuration"
        )
        minimizer_bytes = validate_config(
            args.reference_minimizer, "reference-minimizer-configuration"
        )

        input_config = load_yaml_in_memory(
            input_bytes, "reference-input-configuration"
        )
        expanded_config = load_yaml_in_memory(
            expanded_bytes, "reference-expanded-configuration"
        )
        minimizer_config = load_yaml_in_memory(
            minimizer_bytes, "reference-minimizer-configuration"
        )

        input_sampler = sampler_block(
            input_config, "reference-input-configuration"
        )
        expanded_sampler = sampler_block(
            expanded_config, "reference-expanded-configuration"
        )
        minimizer_sampler = sampler_block(
            minimizer_config, "reference-minimizer-configuration"
        )

        if not isinstance(expanded_sampler, dict) or len(expanded_sampler) != 1:
            raise InvalidBinding(
                "expanded configuration must contain exactly one sampler component"
            )
        sampler_name, sampler_options = next(iter(expanded_sampler.items()))
        if not isinstance(sampler_name, str) or not sampler_name:
            raise InvalidBinding("expanded sampler component name is invalid")
        if sampler_options is None:
            sampler_options = {}
        if not isinstance(sampler_options, dict):
            raise InvalidBinding("expanded sampler options are not a mapping")

        for label, block in (
            ("input", input_sampler),
            ("minimizer", minimizer_sampler),
        ):
            if block is not None and sampler_name not in block:
                raise InvalidBinding(
                    f"{label} sampler block does not contain expanded sampler {sampler_name!r}"
                )

        sampled = a2r["sampled_coordinate_order"]
        effective = {
            "sampler_name": sampler_name,
            "sampler_options": sampler_options,
        }
        declared_method = sampler_options.get("method")
        if declared_method is not None and not isinstance(declared_method, str):
            raise InvalidBinding("expanded sampler method field is not a string")

        result = {
            "protocol": PROTOCOL,
            "verdict": "PASS",
            "scientific_claim": "NONE",
            "authority": AUTHORITY,
            "a2_execution_authorized": False,
            "optimizer_execution_authorized": False,
            "cobaya_version": COBAYA_VERSION,
            "cobaya_source_commit": COBAYA_SOURCE_COMMIT,
            "a0_receipt_sha256": a0_sha256,
            "a2r_receipt_sha256": a2r_sha256,
            "a2g_receipt_sha256": sha256_hex(a2g_bytes),
            "configuration_sha256": {
                "reference-input-configuration": sha256_hex(input_bytes),
                "reference-expanded-configuration": sha256_hex(expanded_bytes),
                "reference-minimizer-configuration": sha256_hex(minimizer_bytes),
            },
            "sampled_coordinate_order": sampled,
            "effective_sampler": effective,
            "effective_sampler_sha256": sha256_hex(
                canonical_json_bytes(effective)
            ),
            "declared_method_field": declared_method,
            "sampler_blocks": {
                "input": input_sampler,
                "expanded": expanded_sampler,
                "minimizer": minimizer_sampler,
            },
            "sampler_block_sha256": {
                "input": (
                    sha256_hex(canonical_json_bytes(input_sampler))
                    if input_sampler is not None
                    else None
                ),
                "expanded": sha256_hex(
                    canonical_json_bytes(expanded_sampler)
                ),
                "minimizer": (
                    sha256_hex(canonical_json_bytes(minimizer_sampler))
                    if minimizer_sampler is not None
                    else None
                ),
            },
            "top_level_keys": {
                "input": list(input_config),
                "expanded": list(expanded_config),
                "minimizer": list(minimizer_config),
            },
            "method_policy": (
                "future A2 optimizer execution MUST bind this exact expanded "
                "sampler component and canonical option block; method/backend "
                "substitution is forbidden unless a separately qualified "
                "normalization/migration gate is introduced"
            ),
        }
        write_json(result)
        return 0
    except (
        InvalidBinding,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        ImportError,
        RuntimeError,
    ) as exc:
        write_json(
            {
                "protocol": PROTOCOL,
                "verdict": "INVALID",
                "scientific_claim": "NONE",
                "authority": AUTHORITY,
                "a2_execution_authorized": False,
                "optimizer_execution_authorized": False,
                "error": str(exc),
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
