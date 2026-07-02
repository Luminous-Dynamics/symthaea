#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate FEMNIST PoGQ vs FLTrust diagnostic plots:
  1) Scatter: PoGQ quality vs FLTrust trust at multiple BFT ratios
  2) DET-style point plot: mean TPR/FPR for PoGQ, FLTrust, Union
  3) Stage attribution bars at 45% BFT

Input: results/femnist_fltrust/details.json produced by
       experiments/femnist_fltrust_vs_pogq.py (with --output ...details.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    raise ValueError("Expected a list of result rows")


def scatter_plot(rows: List[Dict], ratios: List[float], out_path: Path):
    cols = len(ratios)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4), sharex=True, sharey=True)
    if cols == 1:
        axes = [axes]
    for ax, ratio in zip(axes, ratios):
        subset = [
            detail
            for row in rows if row["bft_ratio"] == ratio
            for detail in row["details"]
        ]
        honest = [d for d in subset if not d["is_byzantine"]]
        byz = [d for d in subset if d["is_byzantine"]]
        ax.scatter(
            [d["fltrust_trust"] for d in honest],
            [d["pogq_quality"] for d in honest],
            alpha=0.6,
            label="Honest",
            color="#1f77b4",
            s=12,
        )
        ax.scatter(
            [d["fltrust_trust"] for d in byz],
            [d["pogq_quality"] for d in byz],
            alpha=0.7,
            label="Byzantine",
            color="#d62728",
            s=18,
        )
        ax.set_title(f"BFT={ratio:.2f}")
        ax.set_xlabel("FLTrust cosine (ReLU)")
        ax.set_ylabel("PoGQ quality")
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("FEMNIST: PoGQ Utility vs FLTrust Direction")
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def det_points(rows: List[Dict], ratios: List[float], out_path: Path):
    methods = ["pogq", "fltrust", "union"]
    colors = {"pogq": "#1f77b4", "fltrust": "#ff7f0e", "union": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for ratio in ratios:
        subset = [row for row in rows if row["bft_ratio"] == ratio]
        for method in methods:
            tprs = [row[method]["tpr"] for row in subset]
            fprs = [row[method]["fpr"] for row in subset]
            label = f"{method.capitalize()} @ {int(ratio*100)}% BFT"
            ax.scatter(np.mean(fprs), np.mean(tprs), color=colors[method], label=label, s=40)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("FEMNIST Detection Points (mean over seeds)")
    ax.set_xlim(-0.02, 0.25)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def stage_attribution(rows: List[Dict], ratio: float, out_path: Path):
    subset = [
        detail
        for row in rows if row["bft_ratio"] == ratio
        for detail in row["details"]
    ]
    if not subset:
        raise ValueError(f"No entries for BFT ratio {ratio}")
    stage1 = sum(1 for d in subset if d["fltrust_flag"])
    stage2 = sum(1 for d in subset if (not d["fltrust_flag"]) and d["pogq_flag"])
    union = sum(1 for d in subset if d["union_flag"])
    accepted = len(subset) - union
    values = [stage1, stage2, union, accepted]
    labels = ["Stage1 (FLTrust)", "Stage2 (PoGQ-only)", "Union total", "Accepted"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=["#ff7f0e", "#1f77b4", "#2ca02c", "#7f7f7f"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(value),
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Client count (per seed × clients)")
    ax.set_title(f"Stage Attribution at BFT={ratio:.2f}")
    plt.xticks(rotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot FEMNIST PoGQ vs FLTrust diagnostics")
    parser.add_argument("--details", type=Path, default=Path("results/femnist_fltrust/details.json"))
    parser.add_argument("--fig-dir", type=Path, default=Path("paper-submission/latex-submission/figures"))
    parser.add_argument("--scatter-ratios", default="0.35,0.45,0.50")
    parser.add_argument("--det-ratios", default="0.35,0.45,0.50")
    parser.add_argument("--stage-ratio", type=float, default=0.45)
    args = parser.parse_args()

    rows = load_rows(args.details)
    scatter_ratios = [float(r.strip()) for r in args.scatter_ratios.split(",")]
    det_ratios = [float(r.strip()) for r in args.det_ratios.split(",")]
    fig_dir = args.fig_dir

    scatter_plot(rows, scatter_ratios, fig_dir / "femnist_scatter.png")
    det_points(rows, det_ratios, fig_dir / "femnist_det_points.png")
    stage_attribution(rows, args.stage_ratio, fig_dir / "femnist_stage_attribution.png")


if __name__ == "__main__":
    main()
