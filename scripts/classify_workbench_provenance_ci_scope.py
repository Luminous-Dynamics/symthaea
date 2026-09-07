#!/usr/bin/env python3
"""Fail-closed classifier for Workbench-provenance-only pull-request diffs.

This module does not query GitHub and does not decide scientific authority. It only
answers whether every changed repository path is inside the narrow Workbench
provenance/qualification surface that has its own focused workflows.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "symthaea-workbench-provenance-ci-scope-v1"

_ALLOWED = tuple(
    re.compile(pattern)
    for pattern in (
        r"^\.github/workflows/workbench-[a-z0-9-]+\.yml$",
        r"^data/neuroscience/workbench_[a-z0-9_]+\.json$",
        r"^docs/neuroscience/WORKBENCH_[A-Z0-9_]+\.md$",
        r"^scripts/(?:workbench|verify_workbench|test_workbench|test_verify_workbench|check_workbench)_[a-z0-9_]+\.py$",
    )
)


class ScopeError(ValueError):
    pass


def _canonical_repo_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ScopeError("changed path: non-empty string required")
    if "\x00" in value or "\n" in value or "\r" in value or "\\" in value:
        raise ScopeError("changed path: control/backslash spelling forbidden")
    if value.startswith("/"):
        raise ScopeError("changed path: absolute path forbidden")
    pure = PurePosixPath(value)
    if value != "/".join(pure.parts):
        raise ScopeError("changed path: exact canonical POSIX spelling required")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ScopeError("changed path: traversal/noncanonical component forbidden")
    return value


def is_allowed_path(path: str) -> bool:
    return any(pattern.fullmatch(path) is not None for pattern in _ALLOWED)


def classify_paths(values: Any) -> dict[str, Any]:
    if not isinstance(values, list):
        raise ScopeError("changed paths: JSON array required")
    canonical: list[str] = []
    seen: set[str] = set()
    for raw in values:
        path = _canonical_repo_path(raw)
        if path in seen:
            raise ScopeError(f"changed paths: duplicate path: {path}")
        seen.add(path)
        canonical.append(path)

    canonical.sort()
    disallowed = [path for path in canonical if not is_allowed_path(path)]
    focused_only = bool(canonical) and not disallowed
    return {
        "schema": SCHEMA,
        "status": "workbench-provenance-only" if focused_only else "full-ci-required",
        "changed_path_count": len(canonical),
        "changed_paths": canonical,
        "disallowed_paths": disallowed,
        "global_ci_may_skip": focused_only,
        "focused_workbench_qualification_still_required": True,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ScopeError(f"JSON object: duplicate key: {key}")
        out[key] = value
    return out


def load_paths(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-json", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = classify_paths(load_paths(args.paths_json))
    except (OSError, json.JSONDecodeError, ScopeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["global_ci_may_skip"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
