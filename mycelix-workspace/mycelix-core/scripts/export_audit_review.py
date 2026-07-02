#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Prepare a CSV of audit queue entries for human review.

Usage:
    python scripts/export_audit_review.py --queue 0TML/tests/results/audit_queue.jsonl \
        --output results/audit_review_queue.csv --limit 200

The generated CSV includes suggested columns for the Audit Guild to fill
in (`human_label`, `notes`) while preserving the original metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REVIEW_FIELDS = [
    "timestamp",
    "round",
    "node_id",
    "ml_prediction",
    "ml_confidence",
    "consensus_score",
    "pogq_score",
    "heuristic",
]


def load_queue(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Audit queue not found at {path}")

    entries: dict[tuple[int, int], dict] = {}
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            key = (int(record.get("round", -1)), int(record.get("node_id", -1)))
            if key in entries:
                # keep the earliest entry for the pair
                continue
            entries[key] = record
    return list(entries.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export audit queue for human review")
    parser.add_argument("--queue", default="0TML/tests/results/audit_queue.jsonl", help="Path to audit_queue.jsonl")
    parser.add_argument("--output", default="results/audit_review_queue.csv", help="Path to write the review CSV")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of rows")
    args = parser.parse_args()

    queue_path = Path(args.queue)
    output_path = Path(args.output)
    rows = load_queue(queue_path)

    rows.sort(key=lambda r: (r.get("round", 0), r.get("node_id", 0)))
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = REVIEW_FIELDS + ["human_label", "notes"]

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for record in rows:
            row = {key: record.get(key) for key in REVIEW_FIELDS}
            row.update({"human_label": "", "notes": ""})
            writer.writerow(row)

    print(f"✅ Wrote audit review CSV to {output_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
