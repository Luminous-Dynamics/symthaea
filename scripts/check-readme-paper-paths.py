#!/usr/bin/env python3
"""Fail if README paper/evidence paths drift from the canonical paper layout."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

FORBIDDEN_PREFIXES = (
    "papers/binius-hdc",
    "papers/cfc-zkp",
    "papers/triple-stack-fl",
)

REQUIRED_PATHS = (
    "papers/theory-foundations/binius-hdc",
    "papers/theory-foundations/binius-hdc/reproduce.sh",
    "papers/theory-foundations/cfc-zkp",
    "papers/evaluation/triple-stack-fl",
)


def main() -> int:
    text = README.read_text(encoding="utf-8")
    errors: list[str] = []

    for prefix in FORBIDDEN_PREFIXES:
        if prefix in text:
            errors.append(f"README still references superseded paper path: {prefix}")

    for relative in REQUIRED_PATHS:
        path = ROOT / relative
        if not path.exists():
            errors.append(f"canonical README paper/evidence target does not exist: {relative}")
        if relative not in text:
            errors.append(f"README does not reference canonical path: {relative}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("README paper/evidence paths: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
