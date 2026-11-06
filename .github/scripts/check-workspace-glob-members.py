#!/usr/bin/env python3
"""Fail when root Cargo workspace globs include directories without manifests."""

from __future__ import annotations

import fnmatch
import glob
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def excluded(relative_path: str, patterns: list[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(relative_path, pattern)
        or relative_path.startswith(pattern.rstrip("/") + "/")
        for pattern in patterns
    )


def main() -> int:
    manifest_path = ROOT / "Cargo.toml"
    try:
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"workspace manifest could not be read: {exc}", file=sys.stderr)
        return 2

    workspace = manifest.get("workspace")
    if not isinstance(workspace, dict):
        print("root Cargo.toml has no [workspace] table", file=sys.stderr)
        return 2

    members = workspace.get("members", [])
    excludes = workspace.get("exclude", [])
    if not isinstance(members, list) or not all(isinstance(x, str) for x in members):
        print("[workspace].members must be an array of strings", file=sys.stderr)
        return 2
    if not isinstance(excludes, list) or not all(isinstance(x, str) for x in excludes):
        print("[workspace].exclude must be an array of strings", file=sys.stderr)
        return 2

    checked = 0
    skipped = 0
    failures: list[str] = []

    for pattern in members:
        matches = sorted(glob.glob(str(ROOT / pattern), recursive=True))
        if not matches:
            failures.append(f"workspace member pattern matched nothing: {pattern}")
            continue
        for raw_path in matches:
            candidate = Path(raw_path)
            relative = candidate.relative_to(ROOT).as_posix()
            if excluded(relative, excludes):
                skipped += 1
                continue
            if candidate.is_dir():
                checked += 1
                if not (candidate / "Cargo.toml").is_file():
                    failures.append(
                        f"{relative}: matched by {pattern!r} but has no Cargo.toml "
                        "and is not listed in [workspace].exclude"
                    )
            elif not candidate.is_file():
                failures.append(f"{relative}: workspace member does not exist")

    if failures:
        print("Cargo workspace member integrity FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        "Cargo workspace member integrity passed: "
        f"{checked} package directories checked, {skipped} explicitly excluded paths skipped"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
