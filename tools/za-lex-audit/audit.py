#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify a pinned external za_lex snapshot without importing its data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any


def fail(message: str) -> ValueError:
    return ValueError(f"za-lex-audit: {message}")


def git_blob_sha1(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "za-lex-resource-audit/v1":
        raise fail("unsupported resource manifest schema")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("repository") != "https://github.com/NWU-MuST/za_lex":
        raise fail("unexpected upstream repository")
    commit = source.get("commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise fail("source commit must be a lowercase 40-character Git SHA")
    status = manifest.get("status")
    if not isinstance(status, dict):
        raise fail("status must be an object")
    if status.get("integration") != "not_integrated":
        raise fail("resource audit must not imply production integration")
    for field in (
        "model_training_rights",
        "speech_synthesis_rights",
        "singing_synthesis_rights",
    ):
        if status.get(field) != "not_assessed_and_not_inferred":
            raise fail(f"{field} must remain explicitly unassessed and not inferred")

    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise fail("file entries must be non-empty")
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise fail("file entry must be an object")
        path = entry.get("path")
        pure = PurePosixPath(path) if isinstance(path, str) else None
        if (
            pure is None
            or not path
            or pure.is_absolute()
            or any(part in ("", ".", "..") for part in pure.parts)
            or "\\" in path
            or path in seen
        ):
            raise fail(f"unsafe or duplicate manifest path: {path!r}")
        seen.add(path)
        blob = entry.get("git_blob_sha1")
        if not isinstance(blob, str) or len(blob) != 40 or any(c not in "0123456789abcdef" for c in blob):
            raise fail(f"{path}: invalid Git blob SHA-1")
        if type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 0:
            raise fail(f"{path}: invalid size")
        if not isinstance(entry.get("purpose"), str) or not entry["purpose"].strip():
            raise fail(f"{path}: purpose is required")

    profiles = manifest.get("language_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise fail("language_profiles must be non-empty")
    profile_ids: set[str] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            raise fail("language profile must be an object")
        profile_id = profile.get("id")
        if not isinstance(profile_id, str) or not profile_id or profile_id in profile_ids:
            raise fail(f"empty or duplicate profile ID: {profile_id!r}")
        profile_ids.add(profile_id)
        if profile.get("license") in (None, "", "unknown", "NOASSERTION"):
            raise fail(f"{profile_id}: license is not explicit")
        resources = profile.get("resource_files")
        if (
            not isinstance(resources, list)
            or not resources
            or any(not isinstance(resource, str) or not resource for resource in resources)
            or len(set(resources)) != len(resources)
        ):
            raise fail(f"{profile_id}: resource_files must be a non-empty unique string list")
        unknown = sorted(set(resources) - seen)
        if unknown:
            raise fail(f"{profile_id}: untracked resources: {unknown}")
        rights_notes = profile.get("rights_notes")
        capabilities = profile.get("capabilities")
        if (
            not isinstance(rights_notes, list)
            or not rights_notes
            or any(not isinstance(note, str) or not note.strip() for note in rights_notes)
            or not isinstance(capabilities, dict)
        ):
            raise fail(f"{profile_id}: rights notes and capabilities are required")
        if profile.get("qualification") != "research_only_not_integrated_not_singing_qualified":
            raise fail(f"{profile_id}: qualification must remain research-only and not singing-qualified")
        if capabilities.get("singing_synthesis") is not False:
            raise fail(f"{profile_id}: source profile cannot claim singing synthesis qualification")


def audit(root: Path, manifest: dict[str, Any], expected_commit: str) -> dict[str, Any]:
    validate_manifest(manifest)
    pinned = manifest["source"]["commit"]
    if expected_commit != pinned:
        raise fail(f"expected commit {expected_commit!r} does not match pinned {pinned!r}")
    if root.is_symlink() or not root.is_dir():
        raise fail("source root must be an existing real directory")
    root = root.resolve(strict=True)

    verified = []
    for entry in manifest["files"]:
        rel = PurePosixPath(entry["path"])
        candidate = root.joinpath(*rel.parts)
        current = root
        for part in rel.parts:
            current = current / part
            if current.is_symlink():
                raise fail(f"symlink rejected: {entry['path']}")
        try:
            path = candidate.resolve(strict=True)
        except OSError as exc:
            raise fail(f"missing file {entry['path']}: {exc}") from exc
        if not path.is_relative_to(root) or not path.is_file():
            raise fail(f"path escapes root or is not a regular file: {entry['path']}")
        content = path.read_bytes()
        observed_size = len(content)
        observed_sha = git_blob_sha1(content)
        if observed_size != entry["size_bytes"]:
            raise fail(f"{entry['path']}: size mismatch")
        if observed_sha != entry["git_blob_sha1"]:
            raise fail(f"{entry['path']}: Git blob SHA-1 mismatch")
        verified.append({"path": entry["path"], "git_blob_sha1": observed_sha, "size_bytes": observed_size})

    return {
        "schema": "za-lex-audit-report/v1",
        "repository": manifest["source"]["repository"],
        "source_commit": pinned,
        "verified_file_count": len(verified),
        "profile_ids": [p["id"] for p in manifest["language_profiles"]],
        "rights_review": manifest["status"]["rights_review"],
        "integration": "not_integrated",
        "verified_files": verified,
        "limitations": [
            "This verifies source identity and recorded license metadata, not legal interpretation.",
            "Historical rule-to-dictionary match figures are not independent speaker accuracy.",
            "No third-party dictionaries, phoneme inventories or rules are copied into the report.",
            "No voice-synthesis training, singing synthesis or model-weight rights are inferred.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = audit(args.root, manifest, args.expected_commit)
        encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
