#!/usr/bin/env python3
"""Verify and reconstruct durable Rustfmt remediation evidence chunks.

VerificationOnly: this tool proves archive integrity and reconstruction identity.
It never establishes product qualification, merge authority, or downstream unblock authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

SCHEMA = "symthaea.assurance.rustfmt-remediation-archive.v1"
DERIVATION_SCHEMA = "symthaea.assurance.rustfmt-remediation-derivation.v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def load_archive(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    archive = json.loads(path.read_text(encoding="utf-8"))
    if archive.get("schema") != SCHEMA:
        raise SystemExit(f"unexpected archive schema: {archive.get('schema')!r}")
    payload = archive.get("payload")
    if not isinstance(payload, dict):
        raise SystemExit("archive payload must be an object")
    expected_id = "sha256:" + sha256(canonical_json(payload))
    if archive.get("archive_id") != expected_id:
        raise SystemExit(
            f"archive_id mismatch: expected {expected_id}, got {archive.get('archive_id')!r}"
        )
    if payload.get("authority") != "DurableEvidenceOnly":
        raise SystemExit("archive authority must be DurableEvidenceOnly")
    if payload.get("qualification_result") != "NOT_ESTABLISHED":
        raise SystemExit("archive qualification_result must remain NOT_ESTABLISHED")
    if payload.get("archive_format") != "ordered-raw-chunks-v1":
        raise SystemExit("unsupported archive_format")
    return archive, payload


def load_derivation(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != DERIVATION_SCHEMA:
        raise SystemExit(f"unexpected derivation schema: {manifest.get('schema')!r}")
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise SystemExit("derivation payload must be an object")
    expected_id = "sha256:" + sha256(canonical_json(payload))
    if manifest.get("derivation_id") != expected_id:
        raise SystemExit(
            f"derivation_id mismatch: expected {expected_id}, got {manifest.get('derivation_id')!r}"
        )
    return manifest, payload


def verify_archive(
    repo_root: pathlib.Path,
    payload: dict[str, Any],
) -> bytes:
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise SystemExit("archive chunks must be a non-empty list")

    reconstructed = bytearray()
    expected_index = 0
    seen_paths: set[str] = set()
    for item in chunks:
        if not isinstance(item, dict):
            raise SystemExit("archive chunk entry must be an object")
        if item.get("index") != expected_index:
            raise SystemExit(
                f"chunk index mismatch: expected {expected_index}, got {item.get('index')!r}"
            )
        expected_index += 1
        path_value = item.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise SystemExit("chunk path must be non-empty")
        if path_value in seen_paths:
            raise SystemExit(f"duplicate chunk path: {path_value}")
        seen_paths.add(path_value)

        path = repo_root / path_value
        data = path.read_bytes()
        actual = {
            "bytes": len(data),
            "sha256": sha256(data),
            "git_blob_sha1": git_blob_sha1(data),
        }
        for field, value in actual.items():
            if item.get(field) != value:
                raise SystemExit(
                    f"chunk {item['index']} {field} mismatch: "
                    f"expected {item.get(field)!r}, got {value!r}"
                )
        reconstructed.extend(data)

    identity = payload.get("reconstructed")
    if not isinstance(identity, dict):
        raise SystemExit("missing reconstructed identity")
    data = bytes(reconstructed)
    actual = {
        "bytes": len(data),
        "sha256": sha256(data),
        "git_blob_sha1": git_blob_sha1(data),
    }
    for field, value in actual.items():
        if identity.get(field) != value:
            raise SystemExit(
                f"reconstructed {field} mismatch: expected {identity.get(field)!r}, got {value!r}"
            )
    return data


def cross_check_derivation(
    archive_payload: dict[str, Any],
    derivation_manifest: dict[str, Any],
    derivation_payload: dict[str, Any],
) -> None:
    if archive_payload.get("derivation_id") != derivation_manifest.get("derivation_id"):
        raise SystemExit("archive derivation_id does not match derivation manifest")

    formatted = derivation_payload.get("formatted")
    reconstructed = archive_payload.get("reconstructed")
    if not isinstance(formatted, dict) or not isinstance(reconstructed, dict):
        raise SystemExit("missing formatted/reconstructed identity")
    for field in ("bytes", "sha256", "git_blob_sha1"):
        if formatted.get(field) != reconstructed.get(field):
            raise SystemExit(
                f"archive/derivation {field} mismatch: "
                f"{reconstructed.get(field)!r} != {formatted.get(field)!r}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--repository-root", type=pathlib.Path, required=True)
    parser.add_argument("--derivation-manifest", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    archive, payload = load_archive(args.archive_manifest)
    data = verify_archive(args.repository_root, payload)

    if args.derivation_manifest is not None:
        derivation, derivation_payload = load_derivation(args.derivation_manifest)
        cross_check_derivation(payload, derivation, derivation_payload)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        if args.output.read_bytes() != data:
            raise SystemExit("reconstructed output round-trip mismatch")

    identity = payload["reconstructed"]
    receipt = {
        "schema": "symthaea.assurance.rustfmt-remediation-archive-verification.v1",
        "verification": "PASS",
        "authority": "VerificationOnly",
        "archive_id": archive["archive_id"],
        "derivation_id": payload["derivation_id"],
        "reconstructed_bytes": identity["bytes"],
        "reconstructed_sha256": identity["sha256"],
        "reconstructed_git_blob_sha1": identity["git_blob_sha1"],
        "derivation_cross_checked": args.derivation_manifest is not None,
        "output_written": args.output is not None,
        "qualification_result": "NOT_ESTABLISHED",
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
