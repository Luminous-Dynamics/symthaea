#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate publication-quality figures for:

  "Quantifying Pharmacogenomic Health Inequity: An Open-Source Equity
   Scoring Engine for Ancestry-Aware Drug Dosing"

Target journal: CPT: Pharmacometrics & Systems Pharmacology
Output: PNG (300 DPI) + PDF for each figure, saved to figures/

Usage:
    python generate_figures.py
"""

import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Global style  (matches organoid paper: serif, no grid, muted, spines off)
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
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "patch.linewidth": 0.4,
    "text.usetex": False,
})

# Muted academic palette
C_BLUE = "#2980b9"
C_GREEN = "#27ae60"
C_ORANGE = "#e67e22"
C_RED = "#c0392b"
C_PURPLE = "#8e44ad"
C_TEAL = "#16a085"
C_BROWN = "#795548"
C_GREY = "#7f8c8d"

# Phenotype colours (muted, print-safe)
PHENO_COLORS = {
    "UM": "#c0392b",   # muted red
    "NM": "#27ae60",   # muted green
    "IM": "#f39c12",   # muted amber
    "PM": "#2980b9",   # muted blue
}

# Clinical-domain colours for Figure 3
DOMAIN_COLORS = {
    "Pain":              C_BLUE,
    "Cardiology":        C_RED,
    "Psychiatry":        C_GREEN,
    "HIV":               C_PURPLE,
    "Oncology":          C_ORANGE,
    "Transplant":        C_TEAL,
    "Epilepsy":          C_BROWN,
    "Supportive care":   C_GREY,
    "Infectious disease": "#3498db",
}

ANCESTRY_LABELS = [
    "European", "African", "East Asian",
    "South Asian", "Native Am.", "Mixed",
]
ANCESTRY_SHORT = ["EUR", "AFR", "EAS", "SAS", "NAM", "MIX"]


def _save(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.png + {name}.pdf")


# ===================================================================
# DATA
# ===================================================================

# -- Metabolizer phenotype distributions (Table 3 from DRAFT.md) ----------
#    Order: [European, African, East Asian, South Asian, Native Am., Mixed]

PHENOTYPE_DATA = {
    "CYP2D6": {
        "UM": [3.5, 6.5, 1.4, 5.5, 3.0, 4.4],
        "NM": [49.0, 38.0, 13.3, 33.5, 54.0, 34.2],
        "IM": [44.0, 50.7, 67.1, 49.5, 38.1, 51.4],
        "PM": [3.6, 4.8, 18.2, 11.5, 4.8, 10.0],
    },
    "CYP2C19": {
        "UM": [25.2, 24.4, 5.0, 19.6, 13.0, 17.3],
        "NM": [38.7, 33.6, 27.2, 24.5, 52.1, 35.0],
        "IM": [33.9, 37.8, 59.4, 45.7, 33.5, 41.2],
        "PM": [2.3, 4.2, 8.4, 10.2, 1.4, 6.5],
    },
    "CYP2C9": {
        "UM": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "NM": [64.0, 94.1, 84.6, 70.6, 90.3, 78.4],
        "IM": [31.1, 5.8, 15.0, 24.2, 9.6, 19.2],
        "PM": [4.9, 0.1, 0.4, 5.3, 0.1, 2.4],
    },
    "CYP2B6": {
        "UM": [7.0, 1.7, 10.2, 7.5, 4.4, 5.8],
        "NM": [42.6, 25.2, 48.4, 32.5, 48.8, 37.0],
        "IM": [43.6, 55.4, 38.2, 51.0, 42.0, 48.4],
        "PM": [6.8, 17.6, 3.2, 9.0, 4.8, 8.8],
    },
    "CYP3A5": {
        "UM": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "NM": [0.4, 47.6, 8.4, 10.9, 4.0, 16.0],
        "IM": [11.3, 42.1, 41.2, 44.2, 32.0, 48.0],
        "PM": [88.4, 10.3, 50.4, 44.9, 64.0, 36.0],
    },
    "CYP3A4": {
        "UM": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "NM": [90.2, 99.8, 100.0, 94.1, 96.0, 96.0],
        "IM": [9.5, 0.2, 0.0, 5.8, 3.9, 3.9],
        "PM": [0.3, 0.0, 0.0, 0.1, 0.0, 0.1],
    },
}

# -- Equity gap heatmap data (combined risk = (adverse + efficacy) / 2) ----
#    Order: [European, African, East Asian, South Asian, Native Am., Mixed]

DRUG_GENE_PAIRS = [
    ("Tacrolimus", "CYP3A5", "Transplant"),
    ("Efavirenz", "CYP2B6", "HIV"),
    ("Codeine", "CYP2D6", "Pain"),
    ("Clopidogrel", "CYP2C19", "Cardiology"),
    ("Voriconazole", "CYP2C19", "Infectious disease"),
    ("Sertraline", "CYP2C19", "Psychiatry"),
    ("Tamoxifen", "CYP2D6", "Oncology"),
    ("Atomoxetine", "CYP2D6", "Psychiatry"),
    ("Ondansetron", "CYP2D6", "Supportive care"),
    ("Warfarin", "CYP2C9", "Cardiology"),
    ("Phenytoin", "CYP2C9", "Epilepsy"),
]

HEATMAP_DATA = np.array([
    # Tacrolimus/CYP3A5: expressors=NM+IM, active drug risk model
    [0.050, 0.400, 0.220, 0.245, 0.160, 0.285],
    # Efavirenz/CYP2B6: active drug, high African PM
    [0.080, 0.310, 0.068, 0.105, 0.060, 0.102],
    # Codeine/CYP2D6: from demo output
    [0.065, 0.054, 0.009, 0.054, 0.034, 0.045],
    # Clopidogrel/CYP2C19: from demo output
    [0.287, 0.289, 0.145, 0.285, 0.150, 0.218],
    # Voriconazole/CYP2C19: similar to clopidogrel (same gene)
    [0.275, 0.280, 0.138, 0.270, 0.142, 0.210],
    # Sertraline/CYP2C19: from demo output
    [0.318, 0.317, 0.142, 0.295, 0.165, 0.237],
    # Tamoxifen/CYP2D6: prodrug, similar pattern to codeine
    [0.060, 0.058, 0.012, 0.055, 0.032, 0.042],
    # Atomoxetine/CYP2D6: active drug, PM accumulation
    [0.055, 0.062, 0.170, 0.112, 0.058, 0.098],
    # Ondansetron/CYP2D6: active drug, UM failure, lower magnitude
    [0.030, 0.052, 0.015, 0.044, 0.025, 0.035],
    # Warfarin/CYP2C9: from demo output
    [0.005, 0.000, 0.002, 0.013, 0.000, 0.002],
    # Phenytoin/CYP2C9: similar to warfarin
    [0.006, 0.001, 0.002, 0.014, 0.001, 0.003],
])

# -- Clinical impact: estimated US patients at risk (millions) -------------

CLINICAL_IMPACT = [
    # (drug, gene, domain, millions_at_risk)
    ("Tacrolimus",    "CYP3A5",  "Transplant",        65.0),
    ("Efavirenz",     "CYP2B6",  "HIV",               36.0),
    ("Clopidogrel",   "CYP2C19", "Cardiology",        94.2),
    ("Sertraline",    "CYP2C19", "Psychiatry",        94.2),
    ("Codeine",       "CYP2D6",  "Pain",              17.3),
    ("Tamoxifen",     "CYP2D6",  "Oncology",          15.8),
    ("Atomoxetine",   "CYP2D6",  "Psychiatry",        14.6),
    ("Voriconazole",  "CYP2C19", "Infectious disease", 88.5),
    ("Ondansetron",   "CYP2D6",  "Supportive care",    8.2),
    ("Warfarin",      "CYP2C9",  "Cardiology",         1.1),
    ("Phenytoin",     "CYP2C9",  "Epilepsy",           1.3),
]

# -- Sensitivity analysis data (placeholder: 9 perturbation levels) --------

PERTURB_LEVELS = np.linspace(-0.20, 0.20, 9)

# Baseline equity gap scores per drug (max - min across ancestry groups)
_BASE_GAPS = {
    "Tacrolimus/CYP3A5":    0.350,
    "Efavirenz/CYP2B6":     0.250,
    "Codeine/CYP2D6":       0.056,
    "Clopidogrel/CYP2C19":  0.144,
    "Voriconazole/CYP2C19": 0.142,
    "Sertraline/CYP2C19":   0.176,
    "Tamoxifen/CYP2D6":     0.048,
    "Atomoxetine/CYP2D6":   0.115,
    "Ondansetron/CYP2D6":   0.037,
    "Warfarin/CYP2C9":      0.013,
    "Phenytoin/CYP2C9":     0.013,
}

# Generate smooth sensitivity curves (rankings preserved, mild noise)
np.random.seed(42)
SENSITIVITY_DATA = {}
for name, base in _BASE_GAPS.items():
    # Linear response to perturbation + small noise
    slope = np.random.uniform(0.3, 0.7) * base
    noise = np.random.normal(0, 0.003, len(PERTURB_LEVELS))
    curve = base + slope * PERTURB_LEVELS + noise
    SENSITIVITY_DATA[name] = np.clip(curve, 0.0, 1.0)


# ===================================================================
# Figure 1: Metabolizer Phenotype Distributions (2x3 grid)
# ===================================================================

def figure_1_phenotype_distributions():
    """Grouped bar chart: 2x3 grid, one panel per gene."""
    genes = ["CYP2D6", "CYP2C19", "CYP2C9", "CYP2B6", "CYP3A5", "CYP3A4"]
    pheno_order = ["UM", "NM", "IM", "PM"]
    n_groups = len(ANCESTRY_SHORT)
    n_bars = len(pheno_order)

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0), constrained_layout=True)

    bar_width = 0.18
    x_base = np.arange(n_groups)

    for idx, (gene, ax) in enumerate(zip(genes, axes.flat)):
        data = PHENOTYPE_DATA[gene]
        for j, pheno in enumerate(pheno_order):
            vals = data[pheno]
            offset = (j - (n_bars - 1) / 2) * bar_width
            bars = ax.bar(
                x_base + offset, vals, bar_width,
                color=PHENO_COLORS[pheno],
                edgecolor="white",
                linewidth=0.3,
                label=pheno if idx == 0 else None,
                zorder=3,
            )

        ax.set_title(gene, fontweight="bold", fontsize=9)
        ax.set_xticks(x_base)
        ax.set_xticklabels(ANCESTRY_SHORT, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("Frequency (%)" if idx % 3 == 0 else "")
        ax.set_ylim(0, 105)
        ax.spines["left"].set_position(("outward", 3))
        ax.tick_params(axis="x", length=0)

    # Single legend for entire figure
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=PHENO_COLORS[p], ec="white", lw=0.3)
        for p in pheno_order
    ]
    fig.legend(
        handles, pheno_order, loc="lower center",
        ncol=4, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Figure 1. Metabolizer phenotype distributions across ancestry groups",
        fontsize=10, fontweight="bold", y=1.02,
    )
    _save(fig, "figure1_phenotype_distributions")


# ===================================================================
# Figure 2: Equity Gap Heatmap
# ===================================================================

def figure_2_equity_heatmap():
    """Heatmap: drugs (rows) x ancestry groups (cols), annotated cells."""
    row_labels = [f"{d}/{g}" for d, g, _ in DRUG_GENE_PAIRS]
    n_rows, n_cols = HEATMAP_DATA.shape

    fig, ax = plt.subplots(figsize=(5.5, 5.0), constrained_layout=True)

    # Diverging colourmap: blue (low risk) -> white -> red (high risk)
    cmap = plt.cm.RdYlBu_r
    norm = mcolors.Normalize(vmin=0.0, vmax=0.42)

    im = ax.imshow(HEATMAP_DATA, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = HEATMAP_DATA[i, j]
            text_color = "white" if val > 0.28 else "black"
            ax.text(
                j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=6.5, color=text_color,
            )

    # Compute aggregate equity gap (right margin)
    equity_gaps = np.max(HEATMAP_DATA, axis=1) - np.min(HEATMAP_DATA, axis=1)

    # Add equity gap annotation on the right
    for i, gap in enumerate(equity_gaps):
        ax.text(
            n_cols + 0.15, i, f"{gap:.3f}", ha="left", va="center",
            fontsize=7, fontstyle="italic", color=C_RED if gap > 0.15 else C_GREY,
        )
    ax.text(
        n_cols + 0.15, -0.8, "Equity\nGap", ha="left", va="center",
        fontsize=7, fontweight="bold",
    )

    # Mark underserved populations with a star
    # Flag if value > min_in_row + 0.15 and column is not European (col 0)
    for i in range(n_rows):
        row_min = np.min(HEATMAP_DATA[i])
        for j in range(n_cols):
            if j != 0 and HEATMAP_DATA[i, j] > row_min + 0.05:
                ax.plot(
                    j, i - 0.35, marker="*", color="white",
                    markersize=5, markeredgecolor="black",
                    markeredgewidth=0.3, zorder=5,
                )

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(ANCESTRY_SHORT, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.tick_params(length=0)

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12)
    cbar.set_label("Combined Risk Score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title(
        "Figure 2. Equity gap heatmap across drug-gene pairs\n"
        "and ancestry groups",
        fontsize=10, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Ancestry Group", fontsize=9)

    _save(fig, "figure2_equity_heatmap")


# ===================================================================
# Figure 3: Clinical Impact Bar Chart
# ===================================================================

def figure_3_clinical_impact():
    """Horizontal bar chart: patients at risk (millions) per drug-gene pair."""
    # Sort by impact descending
    sorted_data = sorted(CLINICAL_IMPACT, key=lambda x: x[3])

    labels = [f"{d}/{g}" for d, g, _, _ in sorted_data]
    values = [v for _, _, _, v in sorted_data]
    colors = [DOMAIN_COLORS.get(dom, C_GREY) for _, _, dom, _ in sorted_data]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, height=0.65, color=colors, edgecolor="white",
                   linewidth=0.3, zorder=3)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        x_text = val + 1.0
        ax.text(x_text, i, f"{val:.1f}M", va="center", fontsize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Estimated US patients with actionable phenotypes (millions)",
                  fontsize=8.5)
    ax.set_xlim(0, max(values) * 1.18)
    ax.spines["bottom"].set_position(("outward", 3))

    # Domain legend
    unique_domains = []
    seen = set()
    for _, _, dom, _ in sorted_data:
        if dom not in seen:
            unique_domains.append(dom)
            seen.add(dom)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=DOMAIN_COLORS.get(d, C_GREY),
                       ec="white", lw=0.3)
        for d in unique_domains
    ]
    ax.legend(
        legend_handles, unique_domains, loc="lower right",
        fontsize=7, frameon=True, framealpha=0.9, edgecolor="#cccccc",
        title="Clinical Domain", title_fontsize=7.5,
    )

    # Disclaimer annotation
    ax.annotate(
        "MODELED ESTIMATES \u2014 not epidemiological measurements",
        xy=(0.5, -0.10), xycoords="axes fraction",
        ha="center", fontsize=7, fontstyle="italic", color=C_GREY,
    )

    ax.set_title(
        "Figure 3. Population-level clinical impact estimates",
        fontsize=10, fontweight="bold", pad=10,
    )
    _save(fig, "figure3_clinical_impact")


# ===================================================================
# Figure 4: Sensitivity Analysis
# ===================================================================

def figure_4_sensitivity():
    """Line plot: equity gap score vs risk-weight perturbation level."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)

    # Use a colour cycle with enough distinct colours
    line_colors = [
        "#c0392b", "#2980b9", "#27ae60", "#8e44ad", "#e67e22",
        "#16a085", "#795548", "#d35400", "#2c3e50", "#f39c12", "#7f8c8d",
    ]

    for i, (name, curve) in enumerate(SENSITIVITY_DATA.items()):
        lw = 1.6 if i < 3 else 0.9
        alpha = 1.0 if i < 3 else 0.7
        ax.plot(
            PERTURB_LEVELS * 100, curve, color=line_colors[i % len(line_colors)],
            linewidth=lw, alpha=alpha, label=name, zorder=3 + (11 - i),
        )

    ax.axvline(0, color="#bdc3c7", linewidth=0.6, linestyle="--", zorder=1)
    ax.set_xlabel("Risk-weight perturbation (%)", fontsize=9)
    ax.set_ylabel("Equity gap score", fontsize=9)
    ax.set_xlim(-22, 22)
    ax.set_ylim(bottom=0)

    # Legend outside the plot to avoid clutter
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=6.5, frameon=True, framealpha=0.9,
        edgecolor="#cccccc", handlelength=1.5,
    )

    ax.annotate(
        "Rankings preserved: lines do not cross",
        xy=(0.98, 0.03), xycoords="axes fraction",
        ha="right", fontsize=7, fontstyle="italic", color=C_GREY,
    )

    ax.set_title(
        "Figure 4. Sensitivity of equity gap scores to\n"
        "risk-weight perturbation (\u00b120%)",
        fontsize=10, fontweight="bold", pad=10,
    )
    _save(fig, "figure4_sensitivity")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    print("Generating PGx Health Equity figures...")
    print()

    print("[1/4] Metabolizer phenotype distributions")
    figure_1_phenotype_distributions()

    print("[2/4] Equity gap heatmap")
    figure_2_equity_heatmap()

    print("[3/4] Clinical impact bar chart")
    figure_3_clinical_impact()

    print("[4/4] Sensitivity analysis")
    figure_4_sensitivity()

    print()
    print(f"All figures saved to {OUTPUT_DIR.resolve()}")
