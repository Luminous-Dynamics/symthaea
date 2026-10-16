#!/usr/bin/env python3
"""Verify exact-byte provenance for a research-only socioeconomic run bundle.

This standard-library utility supplements (but does not replace) the v1 receipt
schema/semantic validator. A bundle manifest inventories the actual run payloads;
the receipt binds the exact manifest bytes, while each other digest field must
point to an inventory entry whose bytes and length are recomputed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_SCHEMA = "civos.socioeconomic_interoperability.run_bundle_manifest.v1"
RECEIPT_SCHEMA = "civos.socioeconomic_interoperability.run_receipt.v1"
CLAIM_CEILING = "synthetic_deterministic_only"
HASH_POLICY = "sha256_exact_bytes_v1"
MANIFEST_DIGEST_POINTER = "/run/artifact_manifest_digest"
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
MANIFEST_FIELDS = {
    "schema", "status", "claim_ceiling", "hash_policy", "run_id",
    "receipt_path", "artifacts",
}
ARTIFACT_FIELDS = {
    "artifact_id", "role", "path", "digest", "byte_length",
    "receipt_pointers",
}
AUTHORITY_FALSE_FIELDS = (
    "is_real_world_observation",
    "can_authorize_real_actions",
    "can_reserve_real_resources",
    "can_settle_real_assets",
    "production_ledger_import_allowed",
)


class ValidationFailure(Exception):
    """A bundle failed a required structural, integrity, or boundary check."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationFailure(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValidationFailure(f"non-standard JSON numeric constant is forbidden: {value}")


def _strict_float(token: str) -> float:
    try:
        exact = Decimal(token)
        parsed = float(token)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise ValidationFailure(f"invalid JSON floating-point value: {token}") from exc
    if not exact.is_finite() or not math.isfinite(parsed):
        raise ValidationFailure(f"non-finite JSON floating-point value is forbidden: {token}")
    if parsed == 0.0 and exact != 0:
        raise ValidationFailure(f"JSON floating-point underflow is forbidden: {token}")
    return parsed


def _strict_json(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_strict_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"{label}: invalid UTF-8 JSON: {exc}") from exc


def _safe_file(root: Path, relative_path: Any, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValidationFailure(f"{label}: path must be a non-empty relative POSIX path")
    if "\x00" in relative_path:
        raise ValidationFailure(f"{label}: NUL bytes are forbidden in bundle paths")
    if "\\" in relative_path:
        raise ValidationFailure(f"{label}: backslashes are forbidden in bundle paths")
    relative = PurePosixPath(relative_path)
    if relative.as_posix() != relative_path:
        raise ValidationFailure(f"{label}: non-canonical relative path {relative_path!r}")
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValidationFailure(f"{label}: unsafe relative path {relative_path!r}")

    root_real = root.resolve(strict=True)
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValidationFailure(f"{label}: symlink components are forbidden: {relative_path!r}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValidationFailure(f"{label}: file is missing or unreadable: {relative_path!r}") from exc
    if not resolved.is_relative_to(root_real):
        raise ValidationFailure(f"{label}: path escapes the bundle root: {relative_path!r}")
    if not resolved.is_file():
        raise ValidationFailure(f"{label}: path is not a regular file: {relative_path!r}")
    return resolved


def _pointer_parts(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValidationFailure(f"invalid absolute JSON Pointer: {pointer!r}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _at_pointer(document: Any, pointer: str) -> Any:
    current = document
    for part in _pointer_parts(pointer):
        if isinstance(current, dict):
            if part not in current:
                raise ValidationFailure(f"JSON Pointer does not exist: {pointer}")
            current = current[part]
        elif isinstance(current, list):
            if not part.isdigit():
                raise ValidationFailure(f"array pointer component is not an index: {pointer}")
            index = int(part)
            if index >= len(current):
                raise ValidationFailure(f"array pointer index is out of range: {pointer}")
            current = current[index]
        else:
            raise ValidationFailure(f"JSON Pointer traverses a scalar: {pointer}")
    return current


def _digest_fields(document: Any, pointer: str = "") -> dict[str, str]:
    found: dict[str, str] = {}
    if isinstance(document, dict):
        for key, value in document.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            child_pointer = f"{pointer}/{escaped}"
            if "digest" in key.lower():
                if not isinstance(value, str) or DIGEST_PATTERN.fullmatch(value) is None:
                    raise ValidationFailure(f"{child_pointer}: digest field is not canonical sha256")
                found[child_pointer] = value
            else:
                found.update(_digest_fields(value, child_pointer))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            found.update(_digest_fields(value, f"{pointer}/{index}"))
    return found


def _expected_artifact_role(pointer: str) -> str | None:
    fixed = {
        "/run/scenario_registry_digest": "scenario_registry",
        "/run/world_seed_digest": "world_seed",
        "/run/environment_digest": "environment",
        "/run/shock_schedule_digest": "shock_schedule",
        "/run/agent_model_digest": "agent_model",
        "/profiles/governance_profile_digest": "governance_profile",
        "/profiles/settlement_finality_profile_digest": "settlement_finality_profile",
        "/world/initial_state_digest": "initial_state",
        "/world/final_state_digest": "final_state",
        "/world/resource_state_digest": "resource_state",
        "/evaluation/evaluator_digest": "evaluator",
        "/evaluation/counterexamples_digest": "counterexamples",
    }
    if pointer in fixed:
        return fixed[pointer]
    if re.fullmatch(r"/profiles/regime_assignments/\d+/profile_digest", pointer):
        return "regime_profile"
    if re.fullmatch(r"/profiles/bridge_profiles/\d+/profile_digest", pointer):
        return "bridge_profile"
    if re.fullmatch(r"/events/\d+/payload_digest", pointer):
        return "event_payload"
    return None


def validate_bundle(root_path: str | Path, manifest_relative_path: str) -> dict[str, Any]:
    root = Path(root_path).resolve(strict=True)
    if not root.is_dir():
        raise ValidationFailure("bundle root must be a directory")

    manifest_path = _safe_file(root, manifest_relative_path, "manifest")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _strict_json(manifest_bytes, "bundle manifest")
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise ValidationFailure(f"manifest fields must be exactly {sorted(MANIFEST_FIELDS)}")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValidationFailure("unsupported run-bundle manifest schema")
    if manifest.get("status") != "research" or manifest.get("claim_ceiling") != CLAIM_CEILING:
        raise ValidationFailure("bundle must declare research status and synthetic-only claim ceiling")
    if manifest.get("hash_policy") != HASH_POLICY:
        raise ValidationFailure("unsupported hash policy")
    if not isinstance(manifest.get("run_id"), str) or not manifest["run_id"].strip():
        raise ValidationFailure("manifest run_id must be a non-empty string")

    receipt_relative = manifest.get("receipt_path")
    if receipt_relative == manifest_relative_path:
        raise ValidationFailure("receipt and manifest must be distinct files")
    receipt_path = _safe_file(root, receipt_relative, "receipt")
    receipt = _strict_json(receipt_path.read_bytes(), "run receipt")
    if not isinstance(receipt, dict):
        raise ValidationFailure("run receipt root must be an object")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValidationFailure("receipt schema identifier is unsupported")
    if receipt.get("record_kind") != "synthetic_simulation_receipt":
        raise ValidationFailure("receipt record_kind must be synthetic_simulation_receipt")
    if receipt.get("claim_ceiling") != CLAIM_CEILING:
        raise ValidationFailure("receipt claim ceiling is missing or too permissive")
    receipt_run = receipt.get("run")
    if not isinstance(receipt_run, dict) or receipt_run.get("run_id") != manifest["run_id"]:
        raise ValidationFailure("manifest run_id must exactly match receipt run.run_id")

    boundary = receipt.get("authority_boundary")
    if not isinstance(boundary, dict):
        raise ValidationFailure("receipt authority_boundary must be an object")
    for field in AUTHORITY_FALSE_FIELDS:
        if boundary.get(field) is not False:
            raise ValidationFailure(f"authority_boundary.{field} must be exactly false")

    receipt_digests = _digest_fields(receipt)
    if MANIFEST_DIGEST_POINTER not in receipt_digests:
        raise ValidationFailure("receipt must contain /run/artifact_manifest_digest")
    expected_manifest_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    if _at_pointer(receipt, MANIFEST_DIGEST_POINTER) != expected_manifest_digest:
        raise ValidationFailure("receipt artifact_manifest_digest does not match exact manifest bytes")

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValidationFailure("manifest artifacts must be a non-empty array")

    artifact_ids: set[str] = set()
    artifact_paths: set[str] = set()
    bound_pointers: dict[str, str] = {}
    verified_artifacts = 0
    for index, artifact in enumerate(raw_artifacts):
        label = f"artifacts[{index}]"
        if not isinstance(artifact, dict) or set(artifact) != ARTIFACT_FIELDS:
            raise ValidationFailure(f"{label} fields must be exactly {sorted(ARTIFACT_FIELDS)}")
        artifact_id = artifact.get("artifact_id")
        role = artifact.get("role")
        relative_path = artifact.get("path")
        digest = artifact.get("digest")
        length = artifact.get("byte_length")
        pointers = artifact.get("receipt_pointers")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValidationFailure(f"{label}.artifact_id must be non-empty")
        if artifact_id in artifact_ids:
            raise ValidationFailure(f"duplicate artifact_id: {artifact_id}")
        artifact_ids.add(artifact_id)
        if not isinstance(role, str) or not role.strip():
            raise ValidationFailure(f"{label}.role must be non-empty")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValidationFailure(f"{label}.path must be non-empty")
        if relative_path in artifact_paths:
            raise ValidationFailure(f"duplicate artifact path: {relative_path}")
        if relative_path in {manifest_relative_path, receipt_relative}:
            raise ValidationFailure(f"{label}: manifest and receipt cannot be inventoried as payload artifacts")
        artifact_paths.add(relative_path)
        if not isinstance(digest, str) or DIGEST_PATTERN.fullmatch(digest) is None:
            raise ValidationFailure(f"{label}.digest must be canonical sha256")
        if type(length) is not int or length < 0:
            raise ValidationFailure(f"{label}.byte_length must be a non-negative integer")
        if not isinstance(pointers, list) or any(not isinstance(item, str) for item in pointers):
            raise ValidationFailure(f"{label}.receipt_pointers must be an array of strings")
        if len(pointers) != len(set(pointers)):
            raise ValidationFailure(f"{label}.receipt_pointers contains duplicates")

        payload_path = _safe_file(root, relative_path, label)
        payload_bytes = payload_path.read_bytes()
        actual_digest = "sha256:" + hashlib.sha256(payload_bytes).hexdigest()
        if len(payload_bytes) != length:
            raise ValidationFailure(f"{label}: byte length mismatch for {relative_path}")
        if actual_digest != digest:
            raise ValidationFailure(f"{label}: exact-byte SHA-256 mismatch for {relative_path}")
        verified_artifacts += 1

        for pointer in pointers:
            if pointer == MANIFEST_DIGEST_POINTER:
                raise ValidationFailure("manifest self-digest must not be bound as a payload artifact")
            if pointer in bound_pointers:
                raise ValidationFailure(f"receipt digest pointer is bound more than once: {pointer}")
            value = _at_pointer(receipt, pointer)
            if value != digest:
                raise ValidationFailure(f"{pointer}: receipt digest disagrees with artifact {artifact_id}")
            expected_role = _expected_artifact_role(pointer)
            if expected_role is None:
                raise ValidationFailure(f"{pointer}: no versioned artifact-role mapping exists")
            if role != expected_role:
                raise ValidationFailure(
                    f"{pointer}: expected artifact role {expected_role!r}, got {role!r}"
                )
            bound_pointers[pointer] = artifact_id

    if MANIFEST_DIGEST_POINTER not in receipt_digests:
        raise ValidationFailure("receipt must contain /run/artifact_manifest_digest")
    expected_payload_pointers = set(receipt_digests) - {MANIFEST_DIGEST_POINTER}
    actual_payload_pointers = set(bound_pointers)
    if actual_payload_pointers != expected_payload_pointers:
        missing = sorted(expected_payload_pointers - actual_payload_pointers)
        extra = sorted(actual_payload_pointers - expected_payload_pointers)
        raise ValidationFailure(
            f"receipt digest bindings are incomplete or unexpected; missing={missing}, extra={extra}"
        )

    return {
        "artifact_count": verified_artifacts,
        "receipt_digest_binding_count": len(bound_pointers),
        "manifest_digest": expected_manifest_digest,
        "claim_ceiling": CLAIM_CEILING,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="bundle root directory")
    parser.add_argument("--manifest", required=True, help="manifest path relative to bundle root")
    args = parser.parse_args()
    try:
        report = validate_bundle(args.root, args.manifest)
    except (ValidationFailure, OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        "PASS: exact-byte run bundle verified; "
        f"{report['artifact_count']} payload artifacts and "
        f"{report['receipt_digest_binding_count']} receipt digest bindings checked; "
        "research-only authority boundary enforced."
    )
    print("Claim boundary: provenance integrity only; no issuer authenticity or real-world truth proven.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
