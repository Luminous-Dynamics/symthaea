#!/usr/bin/env python3
"""Verify a local BoTorch source checkout against immutable casting baseline pins.

This checks git commit identity, clean-checkout state, and the Git blob SHA-1 of
all pinned files. Optional release artifacts are checked by SHA-256. Passing this
tool does not lock dependencies, run smoke tests, or make the benchmark executable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from validate_baseline_lock import DEFAULT_MANIFEST, loads_no_duplicate_keys, validate_lock


def git_blob_sha1_bytes(content: bytes) -> str:
    """Compute Git's blob-object SHA-1, not the SHA-1 of raw file bytes."""
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def verify_artifact(path: Path, expected_sha256: str, label: str) -> list[str]:
    errors: list[str] = []
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return [f"{label} cannot be read: {exc}"]
    if digest != expected_sha256:
        errors.append(f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {digest}")
    return errors


def verify_source_tree(
    source_root: Path,
    manifest: dict[str, Any],
    observed_commit: str,
    git_status_porcelain: str = "",
) -> list[str]:
    """Verify required source pins against an already-selected local checkout."""
    errors: list[str] = []
    release = manifest.get("candidate_release", {})
    expected_commit = release.get("source_commit")
    if observed_commit != expected_commit:
        errors.append(
            f"source checkout commit mismatch: expected {expected_commit}, observed {observed_commit}"
        )
    if git_status_porcelain.strip():
        errors.append("source checkout is dirty; use a clean checkout of the pinned tag/commit")

    root = source_root.resolve()
    expected_files: dict[str, str] = {}
    pyproject = release.get("pyproject", {})
    if isinstance(pyproject, dict) and isinstance(pyproject.get("path"), str):
        expected_files[pyproject["path"]] = pyproject.get("git_blob_sha1", "")
    for pin in manifest.get("method_pins", []):
        if isinstance(pin, dict) and isinstance(pin.get("source_path"), str):
            path = pin["source_path"]
            expected_hash = pin.get("source_blob_sha1", "")
            previous = expected_files.get(path)
            if previous is not None and previous != expected_hash:
                errors.append(f"conflicting expected blob hashes for {path}")
            expected_files[path] = expected_hash

    for relative_path, expected_hash in sorted(expected_files.items()):
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            errors.append(f"pinned source path escapes source tree: {relative_path}")
            continue
        try:
            content = candidate.read_bytes()
        except OSError:
            errors.append(f"pinned source file is missing or unreadable: {relative_path}")
            continue
        observed_hash = git_blob_sha1_bytes(content)
        if observed_hash != expected_hash:
            errors.append(
                f"Git blob SHA-1 mismatch for {relative_path}: "
                f"expected {expected_hash}, observed {observed_hash}"
            )
    return errors


def _git_output(source_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *args],
        check=False,
        text=True,
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise RuntimeError(detail)
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-tree", type=Path, required=True,
                        help="clean local checkout of meta-pytorch/botorch at the locked tag/commit")
    parser.add_argument("--sdist", type=Path, default=None, help="optional downloaded source archive")
    parser.add_argument("--wheel", type=Path, default=None, help="optional downloaded wheel")
    parser.add_argument("--require-artifacts", action="store_true",
                        help="require and verify both release archive and wheel")
    args = parser.parse_args(argv)

    try:
        manifest = loads_no_duplicate_keys(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"SOURCE_LOCK_INVALID: {exc}", file=sys.stderr)
        return 2

    structural_errors = validate_lock(manifest)
    if structural_errors:
        for error in structural_errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.require_artifacts and (args.sdist is None or args.wheel is None):
        print("ERROR: --require-artifacts requires both --sdist and --wheel", file=sys.stderr)
        return 2

    try:
        observed_commit = _git_output(args.source_tree, "rev-parse", "HEAD")
        git_status = _git_output(args.source_tree, "status", "--porcelain")
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"SOURCE_CHECKOUT_INVALID: {exc}", file=sys.stderr)
        return 2

    errors = verify_source_tree(args.source_tree, manifest, observed_commit, git_status)
    artifacts = manifest["candidate_release"]["release_artifacts"]
    if args.sdist is not None:
        errors.extend(verify_artifact(args.sdist, artifacts["sdist"]["sha256"], "sdist"))
    if args.wheel is not None:
        errors.extend(verify_artifact(args.wheel, artifacts["wheel"]["sha256"], "wheel"))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 3

    print("PINNED_BOTORCH_SOURCE_TREE_VERIFIED")
    if args.sdist is not None and args.wheel is not None:
        print("PINNED_BOTORCH_RELEASE_ARTIFACTS_VERIFIED")
    print("NOTE: source verification does not validate the runtime environment or execute a baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
