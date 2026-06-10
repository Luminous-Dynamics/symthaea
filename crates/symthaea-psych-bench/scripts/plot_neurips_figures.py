#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Generate publication-ready NeurIPS figures from psych-bench JSON output.

Usage:
    python plot_neurips_figures.py /tmp/bench.json --output-dir /tmp/figs/
    cat /tmp/bench.json | python plot_neurips_figures.py --output-dir /tmp/figs/

Generates 4 PDF figures:
    fig_wm_capacity.pdf      - WM capacity degradation curves
    fig_fep_exploration.pdf   - FEP exploration/learning curves
    fig_ablation_heatmap.pdf  - Mechanistic ablation matrix heatmap
    fig_applied_tom.pdf       - Applied ToM behavioral prediction results
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# NeurIPS style defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (5.5, 4.0),
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLUMN_WIDTH = 5.5  # NeurIPS single column width in inches


def load_report(path):
    """Load BenchmarkReport JSON from file or stdin."""
    if path:
        with open(path) as f:
            return json.load(f)
    else:
        return json.load(sys.stdin)


def find_benchmark(report, name_substr):
    """Find a benchmark result by name substring."""
    for r in report["results"]:
        if name_substr in r["benchmark"]:
            return r
    return None


def get_metric(benchmark, key, default=0.0):
    """Extract a metric value from a benchmark result."""
    if benchmark is None:
        return {"mean": default, "std_dev": 0, "ci_lower": default, "ci_upper": default, "n": 0}
    return benchmark.get("metrics", {}).get(key, {"mean": default, "std_dev": 0, "ci_lower": default, "ci_upper": default, "n": 0})


def fig_wm_capacity(report, output_dir):
    """Figure 1: Working memory capacity degradation curves.

    X-axis: set size K
    Y-axis: accuracy (0-1)
    Two series: ChangeDetection (blue squares), Binding (red triangles)
    Dashed vertical line at Cowan's K ~ 4
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.5))

    cd = find_benchmark(report, "ChangeDetection")
    bd = find_benchmark(report, "Binding")

    # Extract set-size metrics
    set_sizes = [2, 4, 6, 8]

    cd_means, cd_ci_lo, cd_ci_hi = [], [], []
    bd_means, bd_ci_lo, bd_ci_hi = [], [], []

    for k in set_sizes:
        m = get_metric(cd, f"set_size_{k}::accuracy")
        cd_means.append(m["mean"])
        cd_ci_lo.append(m["mean"] - m["ci_lower"])
        cd_ci_hi.append(m["ci_upper"] - m["mean"])

        m = get_metric(bd, f"set_size_{k}::accuracy")
        bd_means.append(m["mean"])
        bd_ci_lo.append(m["mean"] - m["ci_lower"])
        bd_ci_hi.append(m["ci_upper"] - m["mean"])

    x = np.array(set_sizes)

    ax.errorbar(x, cd_means, yerr=[cd_ci_lo, cd_ci_hi],
                fmt="s-", color="#2196F3", markersize=6, capsize=3,
                label="Change Detection", linewidth=1.5)
    ax.errorbar(x, bd_means, yerr=[bd_ci_lo, bd_ci_hi],
                fmt="^-", color="#F44336", markersize=6, capsize=3,
                label="Feature Binding", linewidth=1.5)

    # Cowan's K
    ax.axvline(x=4, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(4.15, 0.95, "Cowan's K", fontsize=8, color="gray",
            transform=ax.get_xaxis_transform())

    ax.set_xlabel("Set Size (K)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Working Memory Capacity Degradation")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(set_sizes)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    path = output_dir / "fig_wm_capacity.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_fep_exploration(report, output_dir):
    """Figure 2: FEP exploration / learning curves.

    Shows metric values across CogBench tasks as a grouped bar chart,
    representing the learning/exploration profile.
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.5))

    tasks = [
        ("InstrumentalLearning", "reward_rate", "#4CAF50"),
        ("RestlessBandit", "reward_rate", "#FF9800"),
        ("ProbabilisticReasoning", "overall_accuracy", "#9C27B0"),
        ("Horizon", "horizon_6::directed_exploration", "#2196F3"),
        ("TwoStep", "beta3_model_basedness", "#F44336"),
    ]

    names, means, errors = [], [], []
    colors = []

    for task_name, metric_key, color in tasks:
        bench = find_benchmark(report, task_name)
        m = get_metric(bench, metric_key)
        names.append(task_name.replace("Learning", "\nLearning").replace("Reasoning", "\nReasoning"))
        means.append(m["mean"])
        errors.append(m["std_dev"])
        colors.append(color)

    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=errors, color=colors, capsize=4,
                  edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("FEP-Driven Cognitive Task Performance")
    ax.grid(True, axis="y", alpha=0.3)

    path = output_dir / "fig_fep_exploration.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_ablation_heatmap(report, output_dir):
    """Figure 3: Ablation heatmap.

    Rows: ablation conditions
    Columns: indicator score | downstream accuracy
    Color: green (high) -> red (low)
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.0))

    butlin = find_benchmark(report, "Butlin")

    ablations = [
        ("CfC off (RPT-1)", "disable_cfc_recurrence"),
        ("GWT off (GWT-3)", "disable_gwt_broadcast"),
        ("Metacog off (HOT-2)", "disable_metacognition"),
        ("PE off (PP-1)", "disable_prediction_learning"),
        ("AST off (AST-1)", "disable_attention_schema"),
    ]

    row_labels = []
    col_labels = ["Baseline\nIndicator", "Ablated\nIndicator", "Baseline\nAccuracy", "Ablated\nAccuracy"]
    data = []

    for label, name in ablations:
        row_labels.append(label)
        row = [
            get_metric(butlin, f"ablation::{name}::baseline_indicator")["mean"],
            get_metric(butlin, f"ablation::{name}::ablated_indicator")["mean"],
            get_metric(butlin, f"ablation::{name}::baseline_accuracy")["mean"],
            get_metric(butlin, f"ablation::{name}::ablated_accuracy")["mean"],
        ]
        data.append(row)

    data = np.array(data)

    # If all zeros (no symthaea-backend), show placeholder
    if data.max() == 0 and data.min() == 0:
        ax.text(0.5, 0.5, "Ablation data requires\n--features symthaea-backend",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray")
        ax.set_axis_off()
    else:
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)

        # Annotate cells
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = data[i, j]
                text_color = "white" if val < 0.3 or val > 0.8 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)

        fig.colorbar(im, ax=ax, shrink=0.8, label="Score")

    ax.set_title("Mechanistic Ablation Matrix")

    path = output_dir / "fig_ablation_heatmap.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_applied_tom(report, output_dir):
    """Figure 4: Applied ToM behavioral prediction results.

    Grouped bar chart: 5 ToMBench tasks x (action accuracy, action confidence)
    Human baseline comparison line (dashed)
    """
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.5))

    tasks = [
        ("FalseBelief", "false_belief_accuracy", 0.90),
        ("FauxPas", "faux_pas_accuracy", 0.85),
        ("Hinting", "hinting_accuracy", 0.80),
        ("Persuasion", "persuasion_detection", 0.85),
        ("StrangeStory", "overall_accuracy", 0.85),
    ]

    names, acc_means, acc_errs, conf_means, conf_errs = [], [], [], [], []
    human_baselines = []

    for task_name, acc_key, human_bl in tasks:
        bench = find_benchmark(report, task_name)
        m_acc = get_metric(bench, acc_key)
        m_conf = get_metric(bench, "action_confidence", default=0.0)

        names.append(task_name.replace("Belief", "\nBelief").replace("Story", "\nStory"))
        acc_means.append(m_acc["mean"])
        acc_errs.append(m_acc["std_dev"])
        conf_means.append(m_conf["mean"])
        conf_errs.append(m_conf["std_dev"])
        human_baselines.append(human_bl)

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width / 2, acc_means, width, yerr=acc_errs,
           label="Action Accuracy", color="#2196F3", capsize=3, alpha=0.85)
    ax.bar(x + width / 2, conf_means, width, yerr=conf_errs,
           label="Action Confidence", color="#FF9800", capsize=3, alpha=0.85)

    # Human baseline line
    ax.plot(x, human_baselines, "k--", linewidth=1.5, alpha=0.5, label="Human Baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("Applied Theory of Mind: Behavioral Predictions")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    path = output_dir / "fig_applied_tom.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate NeurIPS figures from psych-bench JSON output."
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="Path to JSON file (reads stdin if omitted)")
    parser.add_argument("--output-dir", "-o", default=".",
                        help="Output directory for PDF figures (default: .)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark report...")
    report = load_report(args.input)
    print(f"  {len(report.get('results', []))} benchmarks found")

    print("Generating figures...")
    fig_wm_capacity(report, output_dir)
    fig_fep_exploration(report, output_dir)
    fig_ablation_heatmap(report, output_dir)
    fig_applied_tom(report, output_dir)

    print(f"Done! {4} figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
