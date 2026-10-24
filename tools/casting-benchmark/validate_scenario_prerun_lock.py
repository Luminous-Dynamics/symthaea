#!/usr/bin/env python3
"""Fail-closed pre-run source-lock verifier for the synthetic casting bakeoff.

This gate checks exact Git blob identities from a frozen source commit and
checks the current worktree bytes before the benchmark runner is invoked.
It verifies source identity, not scientific validity, authorship, signatures,
environment identity, or SOTA performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_LOCK = Path("docs/research/CASTING_MULTIFIDELITY_POLICY_PRERUN_LOCK_V1.json")
EXPECTED_LOCK_ID = "CASTING-MF-POLICY-SCENARIOS-PRERUN-001"
EXPECTED_STATUS = "PRE_RUN_SOURCE_FREEZE_NOT_EXECUTION_EVIDENCE"
REQUIRED_PATHS = {
    "tools/casting-benchmark/multifidelity_stress.py",
    "tools/casting-benchmark/multifidelity_scenario_suite.py",
    "tools/casting-benchmark/multifidelity_scenario_policy_bakeoff.py",
    "tools/casting-benchmark/score_scenario_policy_bakeoff.py",
    "docs/research/CASTING_MULTIFIDELITY_SCENARIO_SUITE_V1.json",
    "docs/research/CASTING_MULTIFIDELITY_POLICY_SCENARIOS_V1.json",
    "tools/casting-benchmark/test_score_scenario_policy_bakeoff.py",
    "tools/casting-benchmark/test_multifidelity_scenario_policy_bakeoff.py",
}
LOCK_KEYS = {
    "schema_version", "lock_id", "title", "status", "source_commit",
    "hash_convention", "locked_files", "preregistration",
}
FILE_KEYS = {"path", "git_blob_sha1"}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode:
        raise ValueError(
            f"git {' '.join(args)} failed ({proc.returncode}): "
            f"{proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def git_blob_sha1(raw: bytes) -> str:
    """Return the SHA-1 Git computes for a blob, including its object header."""
    header = b"blob " + str(len(raw)).encode("ascii") + b"\0"
    return hashlib.sha1(header + raw).hexdigest()


def validate_lock(repo_root: Path, lock_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    lock_path = lock_path.resolve()
    try:
        lock_rel = lock_path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ValueError("lock path must be inside the repository") from exc

    top = _git(repo_root, "rev-parse", "--show-toplevel")
    if Path(top).resolve() != repo_root:
        raise ValueError("repo_root must be the Git repository root")

    raw_manifest = lock_path.read_bytes()
    manifest = json.loads(raw_manifest.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(manifest, dict) or set(manifest) != LOCK_KEYS:
        missing = sorted(LOCK_KEYS - set(manifest)) if isinstance(manifest, dict) else sorted(LOCK_KEYS)
        unknown = sorted(set(manifest) - LOCK_KEYS) if isinstance(manifest, dict) else []
        raise ValueError(f"lock top-level schema mismatch; missing={missing}, unknown={unknown}")
    if manifest["schema_version"] != "1.0.0":
        raise ValueError("unsupported lock schema_version")
    if manifest["lock_id"] != EXPECTED_LOCK_ID:
        raise ValueError("unexpected lock_id")
    if manifest["status"] != EXPECTED_STATUS:
        raise ValueError("unexpected lock status")
    if not isinstance(manifest["preregistration"], dict):
        raise ValueError("preregistration must be an object")

    source_commit = manifest["source_commit"]
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full lowercase 40-character SHA")
    _git(repo_root, "cat-file", "-e", f"{source_commit}^{{commit}}")
    head = _git(repo_root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", source_commit, head],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("locked source commit is not an ancestor of the execution HEAD")

    # A dirty tracked worktree can silently change benchmark logic after the lock.
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError("tracked worktree/index is dirty; commit or discard changes before a locked run")

    files = manifest["locked_files"]
    if not isinstance(files, list):
        raise ValueError("locked_files must be an array")
    entries: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != FILE_KEYS:
            raise ValueError("each locked_files entry must contain exactly path and git_blob_sha1")
        relative = entry["path"]
        expected = entry["git_blob_sha1"]
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"unsafe relative source path: {relative!r}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{40}", expected):
            raise ValueError(f"invalid Git blob SHA for {relative!r}")
        if relative in entries:
            raise ValueError(f"duplicate locked path: {relative}")
        entries[relative] = expected
    if set(entries) != REQUIRED_PATHS:
        raise ValueError(
            f"locked source set mismatch; missing={sorted(REQUIRED_PATHS-set(entries))}, "
            f"extra={sorted(set(entries)-REQUIRED_PATHS)}"
        )

    checked: list[dict[str, str]] = []
    for relative in sorted(entries):
        expected = entries[relative]
        path = repo_root / relative
        if not path.is_file():
            raise ValueError(f"locked source file is missing: {relative}")
        raw = path.read_bytes()
        worktree_sha = git_blob_sha1(raw)
        committed_sha = _git(repo_root, "rev-parse", f"{source_commit}:{relative}")
        if committed_sha != expected:
            raise ValueError(
                f"lock does not match pinned source commit for {relative}: "
                f"expected {expected}, commit has {committed_sha}"
            )
        if worktree_sha != expected:
            raise ValueError(
                f"worktree bytes do not match frozen source for {relative}: "
                f"expected {expected}, observed {worktree_sha}"
            )
        checked.append({"path": relative, "git_blob_sha1": worktree_sha})

    return {
        "schema_version": "1.0.0",
        "lock_id": manifest["lock_id"],
        "status": "PRE_RUN_LOCK_VERIFIED_NOT_EXECUTION_EVIDENCE",
        "lock_path": lock_rel,
        "lock_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "locked_source_commit": source_commit,
        "execution_head_sha": head,
        "tracked_worktree_clean": True,
        "locked_file_count": len(checked),
        "locked_files": checked,
        "claim_ceiling": manifest["preregistration"].get(
            "claim_ceiling", "NO_PERFORMANCE_CLAIM"
        ),
        "note": (
            "This confirms source identity only. It does not run the benchmark, "
            "verify Python dependencies, validate casting physics, or establish SOTA."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    lock_path = args.lock if args.lock.is_absolute() else root / args.lock
    try:
        result = validate_lock(root, lock_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"valid": False, "status": "PRE_RUN_LOCK_REFUSED", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"valid": True, **result}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
