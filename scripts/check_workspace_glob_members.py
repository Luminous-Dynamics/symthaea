#!/usr/bin/env python3
"""Fail closed when a Cargo workspace member glob selects a directory without Cargo.toml."""
from __future__ import annotations

import fnmatch
import sys
import tomllib
from pathlib import Path


def is_excluded(path: str, excludes: list[str]) -> bool:
    normalized = path.replace("\\", "/").rstrip("/")
    for pattern in excludes:
        candidate = pattern.replace("\\", "/").rstrip("/")
        if normalized == candidate or fnmatch.fnmatchcase(normalized, candidate):
            return True
    return False


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "Cargo.toml"
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"workspace manifest parse failed: {error}", file=sys.stderr)
        return 2

    workspace = data.get("workspace")
    if not isinstance(workspace, dict):
        print("Cargo.toml has no [workspace] table", file=sys.stderr)
        return 2

    members = workspace.get("members")
    excludes = workspace.get("exclude", [])
    if not isinstance(members, list) or not isinstance(excludes, list):
        print("[workspace].members/exclude must be arrays", file=sys.stderr)
        return 2

    errors: list[str] = []
    checked: set[str] = set()
    excluded_matches: set[str] = set()
    for pattern in members:
        if not isinstance(pattern, str):
            errors.append(f"non-string workspace member pattern: {pattern!r}")
            continue
        matches = sorted(root.glob(pattern))
        if not matches:
            errors.append(f"workspace member pattern matched nothing: {pattern}")
            continue
        for match in matches:
            if not match.is_dir():
                errors.append(f"workspace member is not a directory: {match.relative_to(root)}")
                continue
            relative = match.relative_to(root).as_posix()
            if is_excluded(relative, excludes):
                excluded_matches.add(relative)
                continue
            checked.add(relative)
            if not (match / "Cargo.toml").is_file():
                errors.append(
                    f"workspace glob selects non-crate directory {relative!r}; "
                    "add a real Cargo.toml or explicitly exclude the directory"
                )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(
        "workspace glob integrity: "
        f"{len(checked)} buildable member directories; "
        f"{len(excluded_matches)} explicitly excluded paths"
    )
    for relative in sorted(excluded_matches):
        if not (root / relative / "Cargo.toml").is_file():
            print(f"excluded non-crate path (not built): {relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())