#!/usr/bin/env python3
"""
Inspect HyperFeel metrics in experiment results.

Usage:
    python scripts/inspect_hyperfeel_results.py path/to/results.json

Prints a small table of HyperFeel compression metadata per baseline,
if available. Safe to run even when HyperFeel is disabled; it will
just report that no HyperFeel data was found.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str) -> int:
    p = Path(path)
    if not p.exists():
        print(f"Result file not found: {p}")
        return 1

    with p.open("r") as f:
        data = json.load(f)

    baselines = data.get("baselines", {})
    if not baselines:
        print("No baselines found in results.")
        return 0

    rows = []
    for name, history in baselines.items():
        hf = history.get("hyperfeel")
        if not hf:
            continue
        rows.append(
            (
                name,
                hf.get("vector_dimension"),
                hf.get("original_bytes"),
                hf.get("hypervector_bytes"),
                hf.get("compression_ratio"),
                hf.get("timestamp"),
            )
        )

    if not rows:
        print("No HyperFeel metadata found in baselines.")
        return 0

    header = (
        "baseline",
        "dim",
        "orig_bytes",
        "hv_bytes",
        "compression",
        "timestamp",
    )

    col_widths = [
        max(len(str(x)) for x in [h] + [r[i] for r in rows])
        for i, h in enumerate(header)
    ]

    def fmt_row(values):
        return "  ".join(
            str(v if v is not None else "-").ljust(col_widths[i])
            for i, v in enumerate(values)
        )

    print(fmt_row(header))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt_row(row))

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/inspect_hyperfeel_results.py path/to/results.json")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))

