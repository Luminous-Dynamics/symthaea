#!/usr/bin/env python3
"""Fail on Rust/Clippy warnings whose primary span is in qualified source paths."""

from __future__ import annotations

import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(
            "usage: strict_clippy_paths.py <cargo-jsonl> <qualified-path> [<qualified-path> ...]",
            file=sys.stderr,
        )
        return 2

    report_path = pathlib.Path(sys.argv[1])
    qualified = {pathlib.PurePosixPath(path).as_posix() for path in sys.argv[2:]}
    failures: list[tuple[str, str, str]] = []

    with report_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if event.get("reason") != "compiler-message":
                continue

            message = event.get("message") or {}
            if message.get("level") not in {"warning", "error"}:
                continue

            primary_paths = {
                pathlib.PurePosixPath(span.get("file_name", "")).as_posix()
                for span in message.get("spans", [])
                if span.get("is_primary")
            }
            matched = sorted(primary_paths & qualified)
            if not matched:
                continue

            code = (message.get("code") or {}).get("code") or message.get("level", "diagnostic")
            rendered = message.get("rendered") or message.get("message") or "compiler diagnostic"
            for path in matched:
                failures.append((path, code, rendered.rstrip()))

    if failures:
        print("strict focused Clippy diagnostics failed:", file=sys.stderr)
        for path, code, rendered in failures:
            print(f"\n[{path}] {code}\n{rendered}", file=sys.stderr)
        return 1

    print("strict focused Clippy diagnostics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
