#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Aggregate JSON result files (FEMNIST PoGQ / FLTrust) into Markdown tables.

Example:
    python tools/summarize_results.py \
        --inputs results/femnist_run/seed_*.json results/femnist_fltrust/summary.json \
        --out-md paper-submission/latex-submission/tables/femnist_results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple


def iter_rows(paths: List[Path]) -> List[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            rows.append(data)
        else:
            rows.extend(data)
    return rows


def mean_ci(values: List[float]) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], 0.0)
    m = mean(values)
    ci = 2 * stdev(values) / (len(values) ** 0.5)
    return (m, ci)


def render_table(rows: List[dict]) -> str:
    buckets: Dict[Tuple[str, float, str], Dict[str, List[float]]] = {}

    def add_row(dataset: str, ratio: float, method: str, tpr: float, fpr: float):
        key = (dataset, ratio, method)
        stats = buckets.setdefault(key, {"tpr": [], "fpr": []})
        stats["tpr"].append(tpr)
        stats["fpr"].append(fpr)

    def pretty(method: str) -> str:
        if not method:
            return "PoGQ"
        lower = method.lower()
        if "pogq" in lower:
            return "PoGQ"
        if "fltrust" in lower:
            return "FLTrust"
        if "meta" == lower:
            return "Meta"
        return method.capitalize()

    for row in rows:
        dataset = row.get("dataset", "femnist")
        ratio = row.get("bft_ratio")
        if "pogq" in row and "fltrust" in row:
            add_row(dataset, ratio, "PoGQ", row["pogq"]["tpr"], row["pogq"]["fpr"])
            add_row(dataset, ratio, "FLTrust", row["fltrust"]["tpr"], row["fltrust"]["fpr"])
            if "union" in row:
                add_row(dataset, ratio, "Union", row["union"]["tpr"], row["union"]["fpr"])
        else:
            method = pretty(row.get("method") or row.get("detector") or "PoGQ")
            add_row(dataset, ratio, method, row.get("tpr", 0.0), row.get("fpr", 0.0))
            if "meta" in row:
                meta = row["meta"]
                add_row(dataset, ratio, "Meta", meta.get("tpr", 0.0), meta.get("fpr", 0.0))

    lines = [
        "| Dataset | BFT | Method | TPR (mean±CI) | FPR (mean±CI) |",
        "|---|---:|---|---:|---:|",
    ]
    for (dataset, ratio, method), stats in sorted(buckets.items()):
        mt, ct = mean_ci(stats["tpr"])
        mf, cf = mean_ci(stats["fpr"])
        fmt = lambda m, c: f"{100*m:5.1f}% ± {100*c:4.1f}%"
        lines.append(
            f"| {dataset} | {ratio:.2f} | {method} | {fmt(mt, ct)} | {fmt(mf, cf)} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON result files")
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    paths = [Path(p) for p in args.inputs]
    rows = iter_rows(paths)
    table = render_table(rows)

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
