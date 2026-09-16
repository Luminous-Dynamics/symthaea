#!/usr/bin/env python3
"""Verify an artifact-derived Rustfmt remediation and optional candidate product.

VerificationOnly: this tool never establishes source correctness, qualification,
merge authority, or downstream unblock authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

SCHEMA = "symthaea.assurance.rustfmt-remediation-derivation.v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def git(repo: pathlib.Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    return result.stdout


def load_manifest(path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise SystemExit(f"unexpected schema: {manifest.get('schema')!r}")
    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise SystemExit("manifest payload must be an object")
    expected_id = "sha256:" + sha256(canonical_json(payload))
    if manifest.get("derivation_id") != expected_id:
        raise SystemExit(
            f"derivation_id mismatch: expected {expected_id}, "
            f"got {manifest.get('derivation_id')!r}"
        )
    if payload.get("authority") != "RemediationDerived":
        raise SystemExit("authority must be RemediationDerived")
    if payload.get("qualification_result") != "NOT_ESTABLISHED":
        raise SystemExit("qualification_result must remain NOT_ESTABLISHED")
    return manifest, payload


def expected_formatted(payload: dict[str, Any]) -> dict[str, Any]:
    formatted = payload.get("formatted")
    if not isinstance(formatted, dict):
        raise SystemExit("missing formatted identity")
    required = {"path", "git_blob_sha1", "sha256", "bytes"}
    if set(formatted) != required:
        raise SystemExit(f"formatted identity fields drift: {sorted(formatted)}")
    return formatted


def verify_bytes(data: bytes, formatted: dict[str, Any], label: str) -> None:
    actual = {
        "git_blob_sha1": git_blob_sha1(data),
        "sha256": sha256(data),
        "bytes": len(data),
    }
    for field, value in actual.items():
        if value != formatted[field]:
            raise SystemExit(
                f"{label} {field} mismatch: expected {formatted[field]!r}, got {value!r}"
            )


def verify_formatted_file(path: pathlib.Path, formatted: dict[str, Any]) -> None:
    verify_bytes(path.read_bytes(), formatted, "formatted file")


def verify_candidate(
    repo: pathlib.Path,
    candidate: str,
    payload: dict[str, Any],
    formatted: dict[str, Any],
) -> None:
    contract = payload.get("next_product_contract")
    if not isinstance(contract, dict):
        raise SystemExit("missing next_product_contract")
    parent = contract.get("parent")
    expected_paths = contract.get("changed_paths_exact")
    if not isinstance(parent, str) or not isinstance(expected_paths, list):
        raise SystemExit("invalid next_product_contract")

    git(repo, "cat-file", "-e", f"{candidate}^{{commit}}")
    parents = str(git(repo, "rev-list", "--parents", "-n", "1", candidate)).strip().split()
    if len(parents) != 2:
        raise SystemExit(f"candidate must have exactly one parent: {parents!r}")
    if parents[1] != parent:
        raise SystemExit(f"candidate parent mismatch: expected {parent}, got {parents[1]}")

    changed_raw = bytes(
        git(
            repo,
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            parent,
            candidate,
            "--",
            text=False,
        )
    )
    changed = [
        item.decode("utf-8", errors="surrogateescape")
        for item in changed_raw.split(b"\0")
        if item
    ]
    if changed != expected_paths:
        raise SystemExit(
            f"candidate changed paths mismatch: expected {expected_paths!r}, got {changed!r}"
        )

    target = formatted["path"]
    blob = str(git(repo, "rev-parse", f"{candidate}:{target}")).strip()
    if blob != formatted["git_blob_sha1"]:
        raise SystemExit(
            f"candidate blob mismatch: expected {formatted['git_blob_sha1']}, got {blob}"
        )
    data = bytes(git(repo, "show", f"{candidate}:{target}", text=False))
    verify_bytes(data, formatted, "candidate file")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--formatted-file", type=pathlib.Path)
    parser.add_argument("--repository", type=pathlib.Path)
    parser.add_argument("--candidate")
    args = parser.parse_args()

    manifest, payload = load_manifest(args.manifest)
    formatted = expected_formatted(payload)

    if (args.repository is None) != (args.candidate is None):
        parser.error("--repository and --candidate must be supplied together")

    if args.formatted_file is not None:
        verify_formatted_file(args.formatted_file, formatted)

    if args.repository is not None and args.candidate is not None:
        verify_candidate(args.repository, args.candidate, payload, formatted)

    receipt = {
        "schema": "symthaea.assurance.rustfmt-remediation-derivation-verification.v1",
        "verification": "PASS",
        "authority": "VerificationOnly",
        "derivation_id": manifest["derivation_id"],
        "formatted_git_blob_sha1": formatted["git_blob_sha1"],
        "formatted_sha256": formatted["sha256"],
        "candidate_checked": args.candidate is not None,
        "artifact_bytes_checked": args.formatted_file is not None,
        "qualification_result": "NOT_ESTABLISHED",
    }
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
