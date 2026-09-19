#!/usr/bin/env python3
"""Fail-closed Git object guard for research qualification subjects.

This guard is trusted scheduler policy. It validates the candidate manifest and
all declared source paths as repository-contained regular Git objects before any
candidate code executes. Symlinks, gitlinks/submodules, path escape, rename
heuristics, and executable manifests are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from typing import Any

SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
PATH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
MANIFEST_ROOT = PurePosixPath(".github/research-qualifiers")
MAX_MANIFEST_BYTES = 64 * 1024
ALLOWED_MANIFEST_FIELDS = frozenset(
    {"schema", "program", "source_parent", "expected_rust", "packages", "source_paths"}
)
ALLOWED_SOURCE_ROOTS = frozenset(
    {("100644", "blob"), ("100755", "blob"), ("040000", "tree")}
)
ALLOWED_SOURCE_DESCENDANTS = frozenset(
    {("100644", "blob"), ("100755", "blob")}
)


class GuardError(RuntimeError):
    pass


def duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GuardError(f"duplicate manifest key: {key!r}")
        result[key] = value
    return result


def canonical_repo_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise GuardError(f"{label} must be a non-empty bounded string")
    if not PATH_RE.fullmatch(value) or ":" in value:
        raise GuardError(f"{label} contains unsupported characters: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith("./")
        or value.endswith("/")
        or "//" in value
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise GuardError(f"{label} must be canonical repository-relative path: {value!r}")
    return value


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_EXTERNAL_DIFF": ""},
    )
    if completed.returncode != 0:
        raise GuardError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def parse_tree_line(raw: str, expected_path: str) -> tuple[str, str, str]:
    if not raw or "\t" not in raw:
        raise GuardError(f"Git path is missing or ambiguous: {expected_path}")
    metadata, observed_path = raw.split("\t", 1)
    parts = metadata.split()
    if len(parts) != 3 or observed_path != expected_path:
        raise GuardError(f"unexpected ls-tree record for {expected_path!r}: {raw!r}")
    mode, object_type, object_id = parts
    if not SHA40_RE.fullmatch(object_id):
        raise GuardError(f"invalid Git object id for {expected_path}")
    return mode, object_type, object_id


def tree_entry(repo: Path, commit: str, path: str) -> tuple[str, str, str]:
    raw = git(repo, "ls-tree", commit, "--", path)
    lines = raw.splitlines() if raw else []
    if len(lines) != 1:
        raise GuardError(f"expected exactly one Git object for {path!r} at {commit}")
    return parse_tree_line(lines[0], path)


def require_repository_regular_path(root: Path, relative: str, label: str) -> Path:
    path = root / relative
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise GuardError(f"{label} does not exist in checkout: {relative}") from error
    if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise GuardError(f"{label} must be a regular file/directory, not a symlink or special file: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise GuardError(f"{label} resolves outside candidate checkout: {relative}") from error
    return path


def reject_special_descendants(repo: Path, commit: str, source_path: str) -> None:
    raw = git(repo, "ls-tree", "-r", commit, "--", source_path)
    if not raw:
        return
    for line in raw.splitlines():
        if "\t" not in line:
            raise GuardError(f"malformed recursive ls-tree record under {source_path!r}")
        metadata, descendant = line.split("\t", 1)
        parts = metadata.split()
        if len(parts) != 3:
            raise GuardError(f"malformed recursive ls-tree metadata under {source_path!r}")
        mode, object_type, object_id = parts
        if not SHA40_RE.fullmatch(object_id):
            raise GuardError(f"invalid descendant object id under {source_path!r}")
        if (mode, object_type) not in ALLOWED_SOURCE_DESCENDANTS:
            raise GuardError(
                f"source tree contains forbidden Git object {mode}/{object_type}: {descendant}"
            )


def load_manifest(candidate_root: Path, manifest_path: str) -> dict[str, Any]:
    path = require_repository_regular_path(candidate_root, manifest_path, "manifest")
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        raise GuardError("manifest must be a regular file no larger than 64 KiB")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=duplicate_rejector)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GuardError(f"manifest is not valid UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GuardError("manifest root must be an object")
    if set(payload) != ALLOWED_MANIFEST_FIELDS:
        raise GuardError(
            "manifest field set mismatch; "
            f"expected={sorted(ALLOWED_MANIFEST_FIELDS)}, observed={sorted(payload)}"
        )
    return payload


def run(args: argparse.Namespace) -> int:
    if not SHA40_RE.fullmatch(args.expected_head) or not SHA40_RE.fullmatch(args.expected_base):
        raise GuardError("expected head/base must be full lowercase 40-hex identities")

    candidate_root = Path(args.candidate_root).resolve(strict=True)
    if not candidate_root.is_dir():
        raise GuardError("candidate root is not a directory")
    manifest_path = canonical_repo_path(args.manifest, "manifest path")
    if PurePosixPath(manifest_path).parent != MANIFEST_ROOT:
        raise GuardError(f"manifest must live directly under {MANIFEST_ROOT}/")

    actual_head = git(candidate_root, "rev-parse", "HEAD")
    if actual_head != args.expected_head:
        raise GuardError(f"candidate HEAD mismatch: expected {args.expected_head}, observed {actual_head}")
    parents = git(candidate_root, "rev-list", "--parents", "-n", "1", "HEAD").split()
    if parents != [args.expected_head, args.expected_base]:
        raise GuardError(
            f"candidate must have exactly sole parent {args.expected_base}; observed {parents[1:]}"
        )

    changed = sorted(
        line
        for line in git(
            candidate_root,
            "diff",
            "--no-ext-diff",
            "--no-renames",
            "--name-only",
            args.expected_base,
            args.expected_head,
            "--",
        ).splitlines()
        if line
    )
    if changed != [manifest_path]:
        raise GuardError(f"candidate diff must contain exactly {manifest_path!r}; observed {changed!r}")

    manifest_entry = tree_entry(candidate_root, args.expected_head, manifest_path)
    if manifest_entry[:2] != ("100644", "blob"):
        raise GuardError(
            f"manifest Git object must be non-executable regular blob 100644; observed {manifest_entry[0]}/{manifest_entry[1]}"
        )

    manifest = load_manifest(candidate_root, manifest_path)
    if manifest.get("source_parent") != args.expected_base:
        raise GuardError("manifest source_parent does not equal exact candidate parent")

    source_paths = manifest.get("source_paths")
    if not isinstance(source_paths, list) or not source_paths or len(source_paths) > 32:
        raise GuardError("manifest source_paths must contain 1..32 entries")
    canonical_sources = [canonical_repo_path(path, "source path") for path in source_paths]
    if len(set(canonical_sources)) != len(canonical_sources):
        raise GuardError("manifest source_paths must not contain duplicates")

    for source_path in canonical_sources:
        require_repository_regular_path(candidate_root, source_path, "source path")
        base_entry = tree_entry(candidate_root, args.expected_base, source_path)
        head_entry = tree_entry(candidate_root, args.expected_head, source_path)
        if base_entry != head_entry:
            raise GuardError(f"source Git object changed in qualifier commit: {source_path}")
        if base_entry[:2] not in ALLOWED_SOURCE_ROOTS:
            raise GuardError(
                f"source path has forbidden Git object mode/type {base_entry[0]}/{base_entry[1]}: {source_path}"
            )
        reject_special_descendants(candidate_root, args.expected_base, source_path)

    print("research_qualification_subject_guard=PASS")
    return 0


def self_test() -> None:
    assert canonical_repo_path("crates/core/example", "test") == "crates/core/example"
    for bad in ("../escape", "./relative", "/absolute", "a//b", "a/./b", "a/../b", "a:b"):
        try:
            canonical_repo_path(bad, "test")
        except GuardError:
            continue
        raise AssertionError(f"non-canonical path accepted: {bad}")
    assert ("120000", "blob") not in ALLOWED_SOURCE_ROOTS
    assert ("160000", "commit") not in ALLOWED_SOURCE_ROOTS
    assert ("120000", "blob") not in ALLOWED_SOURCE_DESCENDANTS
    print("research_qualification_subject_guard_self_test=PASS")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--candidate-root")
    p.add_argument("--manifest")
    p.add_argument("--expected-head")
    p.add_argument("--expected-base")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    missing = [
        name
        for name in ("candidate_root", "manifest", "expected_head", "expected_base")
        if getattr(args, name) is None
    ]
    if missing:
        print(f"subject guard error: missing required arguments: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        return run(args)
    except (GuardError, OSError) as error:
        print(f"research qualification subject guard error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
