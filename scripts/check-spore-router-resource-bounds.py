#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bound untrusted PR-head data before trusted-base routing policy parses it.

This preflight is fail-closed for focused routing: exceeding any bound means the
router must leave the expensive matrix enabled. It also rejects changed Git
object types that cannot safely inherit ordinary source-file ownership.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = "spore-router-resource-bounds-v1"
MAX_FOCUSED_FILES = 512
MAX_WORKFLOW_BYTES = 1024 * 1024
MAX_SCRIPT_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
MAX_PATH_BYTES = 16 * 1024
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_RAW_HEADER = re.compile(
    r"^:([0-7]{6}) ([0-7]{6}) ([0-9a-f]+) ([0-9a-f]+) ([A-Z][0-9]*)$"
)
ALLOWED_DIFF_MODES = {"000000", "100644", "100755"}

WORKFLOW_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/spore-boot-stack.yml",
)
SCRIPT_PATHS = (
    "scripts/check-spore-boot-stack.sh",
    "scripts/check-spore-focused-structural-coverage.sh",
)
CANDIDATE_MANIFESTS = (
    "crates/domains/symthaea-boot-protocol/Cargo.toml",
    "crates/domains/symthaea-boot-observer/Cargo.toml",
    "crates/domains/symthaea-quicken-fb/Cargo.toml",
    "crates/domains/symthaea-boot-control/Cargo.toml",
    "crates/domains/symthaea-boot-input/Cargo.toml",
    "crates/domains/symthaea-boot-ecology-live/Cargo.toml",
    "crates/domains/symthaea-boot-visual-clock/Cargo.toml",
    "crates/domains/symthaea-boot-presentation/Cargo.toml",
    "crates/core/symthaea-spore-continuity/Cargo.toml",
)


class BoundError(ValueError):
    pass


def run_git(repo: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise BoundError(
            f"git {' '.join(args)} failed: "
            + proc.stderr.decode("utf-8", "replace").strip()
        )
    return proc.stdout


def validate_oid(value: str, label: str) -> None:
    if not _OBJECT_ID.fullmatch(value):
        raise BoundError(f"{label} is not a canonical Git object id")


def blob_size(repo: Path, head: str, path: str, *, required: bool) -> int | None:
    raw = run_git(repo, "ls-tree", "-z", head, "--", path)
    if not raw:
        if required:
            raise BoundError(f"missing required routing input: {path}")
        return None
    records = raw.split(b"\0")
    if records and records[-1] == b"":
        records.pop()
    if len(records) != 1:
        raise BoundError(f"expected one Git entry for {path}, got {len(records)}")
    try:
        meta, actual_raw = records[0].split(b"\t", 1)
        mode_raw, kind_raw, oid_raw = meta.split(b" ", 2)
        actual = actual_raw.decode("utf-8", "strict")
        mode = mode_raw.decode("ascii")
        kind = kind_raw.decode("ascii")
        oid = oid_raw.decode("ascii")
    except (ValueError, UnicodeError) as error:
        raise BoundError(f"malformed tree entry for {path}") from error
    if actual != path or kind != "blob" or mode not in {"100644", "100755"}:
        raise BoundError(f"{path}: routing input is not an ordinary Git blob")
    validate_oid(oid, f"object id for {path}")
    size_raw = run_git(repo, "cat-file", "-s", oid).decode("ascii").strip()
    try:
        size = int(size_raw)
    except ValueError as error:
        raise BoundError(f"{path}: invalid Git object size {size_raw!r}") from error
    if size < 0:
        raise BoundError(f"{path}: negative Git object size")
    return size


def unique_merge_base(repo: Path, base: str, head: str) -> str:
    validate_oid(base, "base")
    validate_oid(head, "head")
    bases = [
        line
        for line in run_git(repo, "merge-base", "--all", base, head)
        .decode("ascii")
        .splitlines()
        if line
    ]
    if len(bases) != 1:
        raise BoundError(f"expected exactly one merge base, got {len(bases)}")
    return bases[0]


def bounded_raw_diff(
    repo: Path,
    merge_base: str,
    head: str,
    *,
    limit: int = MAX_FOCUSED_FILES,
) -> tuple[int, bool]:
    """Stream raw diff entries, bounding count and rejecting non-file object modes."""
    if limit < 1:
        raise BoundError("changed-file limit must be positive")
    proc = subprocess.Popen(
        [
            "git",
            "diff",
            "--raw",
            "-z",
            "--no-renames",
            merge_base,
            head,
            "--",
        ],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    count = 0
    token = bytearray()
    expecting_header = True
    pending_modes: tuple[str, str] | None = None
    exceeded = False

    def consume(raw_token: bytes) -> None:
        nonlocal count, expecting_header, pending_modes, exceeded
        if not raw_token:
            raise BoundError("empty token in raw Git diff")
        if expecting_header:
            try:
                text = raw_token.decode("ascii", "strict")
            except UnicodeDecodeError as error:
                raise BoundError("non-ASCII raw Git diff header") from error
            match = _RAW_HEADER.fullmatch(text)
            if match is None:
                raise BoundError(f"malformed raw Git diff header: {text!r}")
            old_mode, new_mode = match.group(1), match.group(2)
            if old_mode not in ALLOWED_DIFF_MODES or new_mode not in ALLOWED_DIFF_MODES:
                raise BoundError(
                    "changed path uses non-regular Git object mode: "
                    f"old={old_mode} new={new_mode}"
                )
            pending_modes = (old_mode, new_mode)
            expecting_header = False
            return

        # With --no-renames every raw record has exactly one pathname token.
        try:
            raw_token.decode("utf-8", "strict")
        except UnicodeDecodeError as error:
            raise BoundError("non-UTF-8 changed repository path") from error
        if len(raw_token) > MAX_PATH_BYTES:
            raise BoundError("changed repository path exceeds 16 KiB")
        assert pending_modes is not None
        pending_modes = None
        expecting_header = True
        count += 1
        if count > limit:
            exceeded = True

    try:
        while True:
            chunk = proc.stdout.read(8192)
            if not chunk:
                break
            for byte in chunk:
                if byte == 0:
                    consume(bytes(token))
                    token.clear()
                    if exceeded:
                        proc.kill()
                        break
                else:
                    token.append(byte)
                    # A raw header is tiny; a path is bounded independently.
                    if len(token) > MAX_PATH_BYTES:
                        raise BoundError("raw Git diff token exceeds 16 KiB")
            if exceeded:
                break

        if token and not exceeded:
            raise BoundError("raw Git diff ended without NUL terminator")
        if not expecting_header and not exceeded:
            raise BoundError("raw Git diff ended mid-record")
        if exceeded:
            proc.wait(timeout=5)
            return count, True

        stderr = proc.stderr.read().decode("utf-8", "replace").strip()
        returncode = proc.wait(timeout=5)
        if returncode != 0:
            raise BoundError(f"git diff failed: {stderr}")
        return count, False
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def check(repo: Path, base: str, head: str) -> dict[str, object]:
    actual_head = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    if actual_head != head:
        raise BoundError(f"checked-out HEAD {actual_head} != requested head {head}")
    merge_base = unique_merge_base(repo, base, head)

    sizes: dict[str, int | None] = {}
    for path in WORKFLOW_PATHS:
        size = blob_size(repo, head, path, required=True)
        assert size is not None
        if size > MAX_WORKFLOW_BYTES:
            raise BoundError(f"{path}: {size} bytes exceeds workflow bound")
        sizes[path] = size
    for path in SCRIPT_PATHS:
        size = blob_size(repo, head, path, required=False)
        if size is not None and size > MAX_SCRIPT_BYTES:
            raise BoundError(f"{path}: {size} bytes exceeds script bound")
        sizes[path] = size
    for path in CANDIDATE_MANIFESTS:
        size = blob_size(repo, head, path, required=False)
        if size is not None and size > MAX_MANIFEST_BYTES:
            raise BoundError(f"{path}: {size} bytes exceeds manifest bound")
        sizes[path] = size

    changed_count, exceeded = bounded_raw_diff(repo, merge_base, head)
    if exceeded:
        raise BoundError(
            f"changed-file count exceeds focused-only ceiling {MAX_FOCUSED_FILES}"
        )

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "source_commit": head,
        "merge_base": merge_base,
        "changed_file_count": changed_count,
        "max_focused_files": MAX_FOCUSED_FILES,
        "changed_git_modes_regular_only": True,
        "max_workflow_bytes": MAX_WORKFLOW_BYTES,
        "max_script_bytes": MAX_SCRIPT_BYTES,
        "max_manifest_bytes": MAX_MANIFEST_BYTES,
        "input_blob_sizes": sizes,
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory)
        run_git(repo, "init", "-q")
        run_git(repo, "config", "user.email", "ci@example.invalid")
        run_git(repo, "config", "user.name", "CI")
        for path in WORKFLOW_PATHS:
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("name: fixture\n", encoding="utf-8")
        run_git(repo, "add", ".")
        run_git(repo, "commit", "-q", "-m", "base")
        base = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()

        for i in range(3):
            target = repo / f"crates/domains/symthaea-boot-protocol/src/v{i}.rs"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("// fixture\n", encoding="utf-8")
        run_git(repo, "add", ".")
        run_git(repo, "commit", "-q", "-m", "head")
        head = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()
        merge_base = unique_merge_base(repo, base, head)
        count, exceeded = bounded_raw_diff(repo, merge_base, head, limit=2)
        assert count == 3 and exceeded is True

        # Blob-size theorem uses Git object size before reading bytes.
        huge = repo / WORKFLOW_PATHS[0]
        huge.write_bytes(b"x" * 1024)
        run_git(repo, "add", WORKFLOW_PATHS[0])
        run_git(repo, "commit", "-q", "-m", "large workflow")
        large_head = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()
        size = blob_size(repo, large_head, WORKFLOW_PATHS[0], required=True)
        assert size == 1024

        # Symlink/git-object-type theorem: ordinary pathname ownership is not
        # enough. A symlink under an otherwise authorized root must force full CI.
        symlink_base = large_head
        link = repo / "crates/domains/symthaea-boot-protocol/src/link.rs"
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to("v0.rs")
        run_git(repo, "add", str(link.relative_to(repo)))
        run_git(repo, "commit", "-q", "-m", "symlink")
        symlink_head = run_git(repo, "rev-parse", "HEAD").decode("ascii").strip()
        symlink_merge_base = unique_merge_base(repo, symlink_base, symlink_head)
        try:
            bounded_raw_diff(repo, symlink_merge_base, symlink_head)
        except BoundError as error:
            assert "non-regular Git object mode" in str(error)
        else:
            raise AssertionError("symlink change was accepted by resource preflight")

    print("spore-router-resource-bounds: self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if args.repo is None or args.base is None or args.head is None or args.receipt is None:
            raise BoundError("--repo, --base, --head, and --receipt are required")
        value = check(args.repo, args.base, args.head)
        args.receipt.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    except (OSError, UnicodeError, BoundError, AssertionError) as error:
        print(f"spore-router-resource-bounds: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
