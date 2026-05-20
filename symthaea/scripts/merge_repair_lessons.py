#!/usr/bin/env python3
"""Merge verified repair lessons into a stable deduplicated JSONL store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def lesson_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("task_name") or ""),
        str(record.get("signature") or ""),
        str(record.get("category") or ""),
        str(record.get("diagnostic") or ""),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"{path}:{line_no}: invalid JSONL: {error}") from error
    return records


def is_verified_lesson(record: dict[str, Any]) -> bool:
    return bool(record.get("broca_training_record")) and bool(record.get("fixed_source_preview"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--verified-only", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in read_jsonl(args.store):
        merged[lesson_key(record)] = record

    before = len(merged)
    read_count = 0
    skipped = 0
    for input_path in args.input:
        for record in read_jsonl(input_path):
            read_count += 1
            if args.verified_only and not is_verified_lesson(record):
                skipped += 1
                continue
            merged[lesson_key(record)] = record

    records = sorted(
        merged.values(),
        key=lambda record: (
            str(record.get("task_name") or ""),
            str(record.get("signature") or ""),
            str(record.get("category") or ""),
            str(record.get("diagnostic") or ""),
        ),
    )

    if not args.dry_run:
        args.store.parent.mkdir(parents=True, exist_ok=True)
        args.store.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )

    print(
        json.dumps(
            {
                "store": str(args.store),
                "read": read_count,
                "skipped": skipped,
                "before": before,
                "after": len(records),
                "added_or_replaced": max(0, len(records) - before),
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
