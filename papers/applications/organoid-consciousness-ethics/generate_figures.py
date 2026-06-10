#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate publication-quality figures for:

  "A Quantitative Framework for Consciousness-Gated Ethics
   in Neural Organoid Research"

Target journal: AJOB Neuroscience
Output: PNG (300 DPI) + PDF for each figure, saved to figures/

Usage:
    python generate_figures.py
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.6,
    "text.usetex": False,
})

# Muted academic palette
C_DARK = "#2c3e50"
C_BLUE = "#2980b9"
C_GREEN = "#27ae60"
C_ORANGE = "#e67e22"
C_RED = "#c0392b"
C_PURPLE = "#8e44ad"
C_GREY = "#7f8c8d"
C_LIGHT_GREY = "#bdc3c7"

TIER_COLORS = {
    0: "#d5e8d4",  # pale green
    1: "#dae8fc",  # pale blue
    2: "#fff2cc",  # pale yellow
    3: "#fce5cd",  # pale orange
    4: "#f8cecc",  # pale red
}


def _save(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.png + {name}.pdf")


# ===================================================================
# Figure 1 -- Framework Overview (conceptual flowchart)
# ===================================================================
def figure1_framework_overview():
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_aspect("equal")

    # --- Left column: 7 Consciousness Indicators ---
    indicators = [
        "Spontaneous\nActivity",
        "Oscillatory\nPatterns",
        "Synchronized\nBursting",
        "Integrated\nInformation (Phi)",
        "Evoked\nResponses",
        "Sleep-Wake\nCycles",
        "Learning\nCapacity",
    ]

    box_w, box_h = 1.7, 0.65
    left_x = 0.3
    top_y = 6.3
    spacing = 0.78

    for i, label in enumerate(indicators):
        y = top_y - i * spacing
        rect = FancyBboxPatch(
            (left_x, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05",
            facecolor="#dae8fc", edgecolor=C_DARK, linewidth=0.7,
        )
        ax.add_patch(rect)
        ax.text(
            left_x + box_w / 2, y, label,
            ha="center", va="center", fontsize=6.5, color=C_DARK,
        )

    # --- Middle: Tier Classification diamond ---
    diamond_cx, diamond_cy = 4.5, 3.5
    diamond_r = 0.7
    diamond = plt.Polygon(
        [
            (diamond_cx, diamond_cy + diamond_r),
            (diamond_cx + diamond_r * 1.3, diamond_cy),
            (diamond_cx, diamond_cy - diamond_r),
            (diamond_cx - diamond_r * 1.3, diamond_cy),
        ],
        closed=True, facecolor="#fff2cc", edgecolor=C_DARK, linewidth=0.8,
    )
    ax.add_patch(diamond)
    ax.text(
        diamond_cx, diamond_cy, "Tier\nClassification",
        ha="center", va="center", fontsize=7, fontweight="bold", color=C_DARK,
    )

    # Arrow: indicators -> diamond
    ax.annotate(
        "", xy=(diamond_cx - diamond_r * 1.3 - 0.05, diamond_cy),
        xytext=(left_x + box_w + 0.05, diamond_cy),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.0),
    )
    ax.text(
        (left_x + box_w + diamond_cx - diamond_r * 1.3) / 2,
        diamond_cy + 0.25,
        "7 indicators\nevaluated",
        ha="center", va="bottom", fontsize=6, color=C_GREY, style="italic",
    )

    # --- Right column: 5 Tiers ---
    tiers = [
        ("Tier 0", "No indicators", "Standard tissue\nprotocols"),
        ("Tier 1", "Activity only", "Enhanced\nmonitoring"),
        ("Tier 2", "Oscillations /\nsynchrony", "Ethics notification\n+ Phi tracking"),
        ("Tier 3", "Phi > threshold", "Protocol modification\n+ pain mitigation"),
        ("Tier 4", "High integration", "Experiment halt\n+ external review"),
    ]

    right_x = 6.6
    tier_w, tier_h = 2.8, 0.85
    tier_top = 6.1
    tier_spacing = 1.15

    for i, (tier_label, criteria, action) in enumerate(tiers):
        y = tier_top - i * tier_spacing
        color = list(TIER_COLORS.values())[i]
        rect = FancyBboxPatch(
            (right_x, y - tier_h / 2), tier_w, tier_h,
            boxstyle="round,pad=0.06",
            facecolor=color, edgecolor=C_DARK, linewidth=0.7,
        )
        ax.add_patch(rect)
        ax.text(
            right_x + 0.15, y + 0.12, tier_label,
            ha="left", va="center", fontsize=7, fontweight="bold", color=C_DARK,
        )
        ax.text(
            right_x + 0.15, y - 0.18, action,
            ha="left", va="center", fontsize=5.5, color=C_GREY,
        )
        ax.text(
            right_x + tier_w - 0.1, y + 0.12, criteria,
            ha="right", va="center", fontsize=5.5, color=C_DARK, style="italic",
        )

    # Arrow: diamond -> tiers
    mid_tier_y = tier_top - 2 * tier_spacing
    ax.annotate(
        "", xy=(right_x - 0.05, mid_tier_y),
        xytext=(diamond_cx + diamond_r * 1.3 + 0.05, diamond_cy),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.0),
    )

    # Title
    ax.text(
        5.0, 7.0,
        "Figure 1. Five-Tier Consciousness-Gated Ethics Framework",
        ha="center", va="top", fontsize=9, fontweight="bold", color=C_DARK,
    )

    _save(fig, "figure1_framework_overview")


# ===================================================================
# Figure 2 -- Consciousness Emergence Trajectory
# ===================================================================
def figure2_emergence_trajectory():
    # Data
    days = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])
    phi = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.011, 0.023, 0.038, 0.051, 0.069, 0.099, 0.126, 0.179, 0.238, 0.304,
    ])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Background bands for developmental stages
    stages = [
        (0, 25, "#dbeef4", "Proliferation"),
        (25, 50, "#d5f5e3", "Patterning"),
        (50, 75, "#fef9e7", "Synaptogenesis"),
        (75, 95, "#fdebd0", "Maturation"),
    ]
    for x0, x1, color, label in stages:
        ax.axvspan(x0, x1, facecolor=color, alpha=0.6, zorder=0)
        ax.text(
            (x0 + x1) / 2, 0.335, label,
            ha="center", va="bottom", fontsize=6.5, color=C_GREY, style="italic",
        )

    # Tier threshold lines
    thresholds = [
        (0.07, "T3 precautionary (0.07)", C_ORANGE, (4, 3)),
        (0.10, "T3 standard (0.10)", C_ORANGE, (2, 2)),
        (0.21, "T4 precautionary (0.21)", C_RED, (4, 3)),
        (0.30, "T4 standard (0.30)", C_RED, (2, 2)),
    ]
    for yval, label, color, dashes in thresholds:
        ax.axhline(yval, color=color, linestyle="--", linewidth=0.7, dashes=dashes, alpha=0.7)
        ax.text(91.5, yval, label, va="center", fontsize=5.5, color=color)

    # T1 label (activity-based, not Phi-based)
    ax.annotate(
        "T1 (activity onset)",
        xy=(41, 0.0), xytext=(20, 0.05),
        fontsize=6, color=C_BLUE,
        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=0.6),
    )

    # Main Phi trajectory
    ax.plot(days, phi, "o-", color=C_DARK, markersize=3, zorder=5, label="Phi trajectory")

    # Ethics halt line
    ax.axvline(90, color=C_RED, linestyle="-", linewidth=1.0, alpha=0.8)
    ax.text(89.5, 0.28, "Ethics\nHalt", ha="right", va="top", fontsize=6.5,
            color=C_RED, fontweight="bold")

    ax.set_xlim(-2, 115)
    ax.set_ylim(-0.01, 0.36)
    ax.set_xlabel("Developmental Day")
    ax.set_ylabel("Estimated Phi (Integrated Information)")
    ax.set_title("Figure 2. Consciousness Emergence Trajectory", fontweight="bold")

    ax.legend(loc="upper left", frameon=False)

    _save(fig, "figure2_emergence_trajectory")


# ===================================================================
# Figure 3 -- Threshold Sensitivity Analysis
# ===================================================================
def figure3_threshold_sensitivity():
    settings = ["Ultra-\nconservative", "Conservative", "Default", "Relaxed", "Permissive"]
    t1_days = [41, 41, 41, 41, 41]
    t2_days = [54, 54, 54, 54, 54]
    t3_days = [59, 67, 71, 75, 78]
    halt_days = [78, 86, 90, 200, 200]

    x = np.arange(len(settings))
    width = 0.18

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    bars_t1 = ax.bar(x - 1.5 * width, t1_days, width, label="First Tier 1",
                      color="#dae8fc", edgecolor=C_DARK, linewidth=0.5)
    bars_t2 = ax.bar(x - 0.5 * width, t2_days, width, label="First Tier 2",
                      color="#fff2cc", edgecolor=C_DARK, linewidth=0.5)
    bars_t3 = ax.bar(x + 0.5 * width, t3_days, width, label="First Tier 3",
                      color="#fce5cd", edgecolor=C_DARK, linewidth=0.5)
    bars_halt = ax.bar(x + 1.5 * width, halt_days, width, label="Halt Day",
                        color="#f8cecc", edgecolor=C_DARK, linewidth=0.5)

    # Mark "never" for halt days that exceed the simulation
    for i, hd in enumerate(halt_days):
        if hd >= 200:
            ax.text(
                x[i] + 1.5 * width, hd + 3, "never",
                ha="center", va="bottom", fontsize=6, color=C_RED, style="italic",
            )

    # Annotation: T1 and T2 identical across all settings
    ax.annotate(
        "T1 and T2 identical\nacross all settings",
        xy=(2 - 1.0 * width, 43), xytext=(0.2, 95),
        fontsize=6.5, color=C_GREY, style="italic",
        arrowprops=dict(arrowstyle="->", color=C_GREY, lw=0.6),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(settings, fontsize=7.5)
    ax.set_ylabel("Day")
    ax.set_ylim(0, 220)
    ax.set_title("Figure 3. Threshold Sensitivity Analysis", fontweight="bold")
    ax.legend(loc="upper left", frameon=False, ncol=2)

    _save(fig, "figure3_threshold_sensitivity")


# ===================================================================
# Figure 4 -- Multi-Seed Robustness (two panels)
# ===================================================================
def figure4_multiseed_robustness():
    halt_days = [80, 79, 83, 80, 79, 80, 81, 80, 80, 81,
                 79, 80, 81, 83, 80, 82, 80, 79, 79, 80]
    peak_phi = [0.2236, 0.2101, 0.2109, 0.2183, 0.2170,
                0.2108, 0.2213, 0.2197, 0.2197, 0.2186,
                0.2147, 0.2168, 0.2203, 0.2244, 0.2180,
                0.2171, 0.2149, 0.2134, 0.2192, 0.2107]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    # Panel A: Halt days
    bp_a = ax_a.boxplot(
        halt_days, vert=True, widths=0.4, patch_artist=True,
        boxprops=dict(facecolor="#dae8fc", edgecolor=C_DARK, linewidth=0.7),
        medianprops=dict(color=C_RED, linewidth=1.0),
        whiskerprops=dict(color=C_DARK, linewidth=0.7),
        capprops=dict(color=C_DARK, linewidth=0.7),
        flierprops=dict(marker="o", markersize=3, markerfacecolor=C_GREY),
    )
    # Strip (jittered) overlay
    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(halt_days))
    ax_a.scatter(
        np.ones(len(halt_days)) + jitter, halt_days,
        color=C_BLUE, s=12, alpha=0.6, zorder=5, edgecolors="none",
    )
    mean_h = np.mean(halt_days)
    std_h = np.std(halt_days, ddof=1)
    ax_a.text(
        0.95, 0.95,
        f"Mean = {mean_h:.1f} $\\pm$ {std_h:.1f} days\n(n = 20 seeds)",
        transform=ax_a.transAxes, ha="right", va="top",
        fontsize=7, color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_LIGHT_GREY, linewidth=0.5),
    )
    ax_a.set_ylabel("Halt Day")
    ax_a.set_title("A.  Halt Day Distribution", fontweight="bold", fontsize=9)
    ax_a.set_xticks([])

    # Panel B: Peak Phi
    bp_b = ax_b.boxplot(
        peak_phi, vert=True, widths=0.4, patch_artist=True,
        boxprops=dict(facecolor="#fce5cd", edgecolor=C_DARK, linewidth=0.7),
        medianprops=dict(color=C_RED, linewidth=1.0),
        whiskerprops=dict(color=C_DARK, linewidth=0.7),
        capprops=dict(color=C_DARK, linewidth=0.7),
        flierprops=dict(marker="o", markersize=3, markerfacecolor=C_GREY),
    )
    jitter_b = np.random.default_rng(7).uniform(-0.08, 0.08, len(peak_phi))
    ax_b.scatter(
        np.ones(len(peak_phi)) + jitter_b, peak_phi,
        color=C_ORANGE, s=12, alpha=0.6, zorder=5, edgecolors="none",
    )
    mean_p = np.mean(peak_phi)
    std_p = np.std(peak_phi, ddof=1)
    ax_b.text(
        0.95, 0.95,
        f"Mean = {mean_p:.3f} $\\pm$ {std_p:.3f}\n(n = 20 seeds)",
        transform=ax_b.transAxes, ha="right", va="top",
        fontsize=7, color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_LIGHT_GREY, linewidth=0.5),
    )
    ax_b.set_ylabel("Peak Phi at Halt")
    ax_b.set_title("B.  Peak Phi Distribution", fontweight="bold", fontsize=9)
    ax_b.set_xticks([])

    fig.suptitle("Figure 4. Multi-Seed Robustness (n = 20)", fontweight="bold", y=1.02)
    fig.tight_layout()

    _save(fig, "figure4_multiseed_robustness")


# ===================================================================
# Figure 5 -- Precautionary vs Standard Mode
# ===================================================================
def figure5_precautionary_vs_standard():
    # Step function data: (day, precautionary_tier, standard_tier)
    # We define the segments explicitly for step plotting.
    days_precautionary = [0, 41, 54, 66, 71, 84, 90, 95]
    tier_precautionary = [0, 1,  2,  3,  3,  4,  4,  4]

    days_standard      = [0, 41, 54, 71, 90, 95]
    tier_standard      = [0, 1,  2,  3,  4,  4]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    ax.step(
        days_precautionary, tier_precautionary,
        where="post", color=C_DARK, linewidth=1.5,
        label="Precautionary Mode (default)", zorder=5,
    )
    ax.step(
        days_standard, tier_standard,
        where="post", color=C_BLUE, linewidth=1.3, linestyle="--",
        label="Standard Mode", zorder=4,
    )

    # Background coloring by tier (based on precautionary)
    tier_bands = [
        (0, 41, 0), (41, 54, 1), (54, 66, 2),
        (66, 84, 3), (84, 95, 4),
    ]
    for d0, d1, tier in tier_bands:
        ax.axvspan(d0, d1, facecolor=list(TIER_COLORS.values())[tier], alpha=0.25, zorder=0)

    # Annotation: 5-day early warning
    ax.annotate(
        "",
        xy=(66, 2.5), xytext=(71, 2.5),
        arrowprops=dict(arrowstyle="<->", color=C_ORANGE, lw=1.2),
    )
    ax.text(
        68.5, 2.65, "5-day early\nwarning",
        ha="center", va="bottom", fontsize=6.5, color=C_ORANGE, fontweight="bold",
    )

    # Annotation: 6-day early halt
    ax.annotate(
        "",
        xy=(84, 3.5), xytext=(90, 3.5),
        arrowprops=dict(arrowstyle="<->", color=C_RED, lw=1.2),
    )
    ax.text(
        87, 3.65, "6-day early\nhalt",
        ha="center", va="bottom", fontsize=6.5, color=C_RED, fontweight="bold",
    )

    ax.set_xlim(-2, 97)
    ax.set_ylim(-0.3, 4.6)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["Tier 0", "Tier 1", "Tier 2", "Tier 3", "Tier 4"])
    ax.set_xlabel("Developmental Day")
    ax.set_ylabel("Ethics Tier")
    ax.set_title("Figure 5. Precautionary vs Standard Threshold Mode", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)

    _save(fig, "figure5_precautionary_vs_standard")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print(f"Generating figures into {OUTPUT_DIR}/")
    figure1_framework_overview()
    figure2_emergence_trajectory()
    figure3_threshold_sensitivity()
    figure4_multiseed_robustness()
    figure5_precautionary_vs_standard()
    print("Done. All figures generated.")
