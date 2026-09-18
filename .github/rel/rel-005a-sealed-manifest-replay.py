#!/usr/bin/env python3
"""Replay the sealed REL-005A execution-artifact manifest without scientific parsing.

Authority: SealedManifestReplayOnly.
This helper verifies the exact ten-file execution census, byte lengths, SHA-256
commitments, and manifest identity inside the ObservationSeal archive. It does
not parse the scientific observation JSON or execution logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

EXECUTION_HEAD = "6931639e53060809f9d459196a329c4fe983be7b"
EXPECTED = {
    "cargo-version.txt",
    "clippy-version.txt",
    "execution-exit-code.txt",
    "execution-v3-receipt.json",
    "execution-v3-static-audit.json",
    "execution.stderr.log",
    "execution.stdout.log",
    "rel-005a-execution-v3-observation.json",
    "rustc-version.txt",
    "rustfmt-version.txt",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def unique(root: pathlib.Path, basename: str) -> pathlib.Path:
    found = [p for p in root.rglob(basename) if p.is_file()]
    require(len(found) == 1, f"{basename}: census={len(found)}")
    return found[0]


def replay(root: pathlib.Path) -> dict[str, Any]:
    manifest_path = unique(root, "execution-artifact-manifest.json")
    manifest = load(manifest_path)
    require(
        manifest.get("schema") == "symthaea.rel.execution-artifact-manifest.v1",
        "manifest schema mismatch",
    )
    require(manifest.get("authority") == "ObservationSeal", "manifest authority mismatch")
    require(
        manifest.get("execution_subject_head") == EXECUTION_HEAD,
        "manifest execution subject mismatch",
    )

    entries = manifest.get("files")
    require(isinstance(entries, list), "manifest files must be a list")
    names = [entry.get("basename") for entry in entries]
    require(len(names) == len(set(names)), "duplicate manifest basename")
    require(set(names) == EXPECTED, f"manifest census mismatch: {sorted(names)}")

    verified = []
    for entry in entries:
        name = entry["basename"]
        path = unique(root, name)
        require(type(entry.get("byte_length")) is int, f"{name}: byte_length invalid")
        require(path.stat().st_size == entry["byte_length"], f"{name}: byte length mismatch")
        digest = entry.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64, f"{name}: sha256 invalid")
        require(sha(path) == digest, f"{name}: sha256 mismatch")
        verified.append(name)

    return {
        "schema": "symthaea.rel.sealed-manifest-replay-receipt.v1",
        "authority": "SealedManifestReplayOnly",
        "execution_subject_head": EXECUTION_HEAD,
        "manifest_sha256": sha(manifest_path),
        "manifest_entry_count": len(entries),
        "exact_file_census_verified": True,
        "byte_lengths_verified": True,
        "sha256_commitments_verified": True,
        "scientific_observation_fields_parsed": False,
        "execution_logs_parsed": False,
        "verified_basenames": verified,
        "claims": {
            "comparison_only_adjudicated": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.seal_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
