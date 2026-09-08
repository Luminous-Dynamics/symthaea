#!/usr/bin/env python3
"""Convert `git diff --name-only -z --no-renames` bytes to canonical JSON paths."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


class ChangedPathError(ValueError):
    pass


def parse_name_only_z(data: bytes) -> list[str]:
    if not data:
        return []
    if not data.endswith(b"\0"):
        raise ChangedPathError("changed-path stream must end with NUL")

    raw_paths = data[:-1].split(b"\0")
    paths: list[str] = []
    seen: set[str] = set()

    for raw in raw_paths:
        if not raw:
            raise ChangedPathError("empty changed path forbidden")
        try:
            path = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ChangedPathError("changed path must be UTF-8") from exc
        if path in seen:
            raise ChangedPathError(f"duplicate changed path: {path}")
        seen.add(path)
        paths.append(path)

    paths.sort()
    return paths


def write_exclusive(path: Path, value: list[str]) -> None:
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise ChangedPathError("output exists; overwrite forbidden") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        paths = parse_name_only_z(args.input.read_bytes())
        write_exclusive(args.output, paths)
    except (OSError, ChangedPathError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"changed_path_count={len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
