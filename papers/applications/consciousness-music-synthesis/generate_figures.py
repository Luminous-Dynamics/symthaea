#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

"""
Generate publication-quality figures for:

  "Consciousness-Driven Music Synthesis: Real-Time Emotional Expression
   from Integrated Information Dynamics"

Target journal: Frontiers in Computational Neuroscience
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
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Global style  (matches other papers: serif, no grid, muted, spines off)
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
    "patch.linewidth": 0.6,
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
C_DARK = "#2c3e50"
C_LIGHT_GREY = "#bdc3c7"


def _save(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=300)
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.png + {name}.pdf")


# ===================================================================
# DATA  (from analysis_results.txt)
# ===================================================================

FEATURES = [
    "onset_density", "rms_energy", "tempo_proxy", "spectral_centroid",
    "spectral_flux", "harmonic_tension", "zero_crossing_rate",
    "major_mode_weight", "note_range", "velocity_variance", "phrase_length",
]

FEATURE_LABELS = [
    "Onset density", "RMS energy", "Tempo proxy", "Spectral centroid",
    "Spectral flux", "Harmonic tension", "Zero-crossing rate",
    "Major mode weight", "Note range", "Velocity variance", "Phrase length",
]

# Feature-state correlations (Table 3)
R_AROUSAL = np.array([
    -0.113, +0.844, -0.022, +0.150, +0.839,
    -0.091, +0.150, -0.366, +0.139, +0.833, -0.113,
])

R_VALENCE = np.array([
    -0.299, +0.287, -0.387, -0.032, +0.270,
    -0.231, -0.032, +0.319, +0.351, +0.092, -0.299,
])

# Bonferroni-significant (p < 0.00227) flags for arousal and valence
SIG_AROUSAL = [False, True, True, False, True, True, False, False, False, True, False]
SIG_VALENCE = [False, False, False, True, False, False, True, False, False, True, False]

# Ablation study data
ABLATION_CONFIGS = [
    "Full system", "No FEP", "No feedback",
    "No Emot.\nAnchor", "No consciousness", "No reverb", "Null model",
]
ABLATION_DISCRIM = np.array([85.604, 148.326, 214.451, 134.629, 248.389, 162.404, 0.0])
ABLATION_R_A = np.array([0.479, 0.538, 0.777, 0.574, 0.787, 0.767, 0.0])
ABLATION_R_V = np.array([0.612, 0.668, 0.380, 0.543, 0.266, 0.455, 0.0])
ABLATION_DELTA = np.array([0.0, 73.3, 150.5, 57.3, 190.2, 89.7, -100.0])

# Phi partial correlations (controlling for A, V)
R_PHI = np.array([
    +0.113, -0.118, -0.060, -0.389, -0.022,
    -0.049, -0.389, +0.244, +0.120, -0.106, +0.113,
])

PHI_SIG = [False, False, True, True, True, True, True, False, False, False, False]


# ===================================================================
# FIGURE 1: Strange Loop Architecture
# ===================================================================

def figure1_strange_loop():
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box_kw = dict(
        boxstyle="round,pad=0.4", edgecolor=C_DARK,
        facecolor="white", linewidth=1.0,
    )
    highlight_kw = dict(
        boxstyle="round,pad=0.4", edgecolor=C_BLUE,
        facecolor="#eaf2f8", linewidth=1.2,
    )
    fep_kw = dict(
        boxstyle="round,pad=0.4", edgecolor=C_PURPLE,
        facecolor="#f5eef8", linewidth=1.2,
    )

    # Box positions (x_center, y_center)
    consciousness_pos = (2.5, 7.0)
    pipeline_pos = (2.5, 4.8)
    output_pos = (2.5, 2.6)
    feedback_pos = (7.0, 2.6)
    extraction_pos = (7.0, 4.8)
    fep_pos = (7.0, 7.0)

    # Draw boxes
    ax.text(*consciousness_pos,
            "Consciousness State\n"
            "MusicalState: A, V, Phi,\n"
            "DA, 5-HT, NE, PE",
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox=highlight_kw)

    ax.text(*pipeline_pos,
            "Synthesis Pipeline\n"
            "(8 phases: state update,\n"
            "notes, voices, ducking,\n"
            "spatial, reverb, clip, feedback)",
            ha="center", va="center", fontsize=7,
            bbox=box_kw)

    ax.text(*output_pos,
            "Stereo PCM Output\n"
            "44.1 kHz, 32 ms chunks",
            ha="center", va="center", fontsize=8,
            bbox=box_kw)

    ax.text(*extraction_pos,
            "Audio Feature Extraction\n"
            "6 features: centroid, flux,\n"
            "rhythm entropy, tension,\n"
            "RMS energy, ZCR",
            ha="center", va="center", fontsize=7,
            bbox=box_kw)

    ax.text(*feedback_pos,
            "Active Inference Agent\n"
            "16D hidden state,\n"
            "8 actions, TD learning",
            ha="center", va="center", fontsize=7,
            bbox=box_kw)

    ax.text(*fep_pos,
            "FEP Modulation\n"
            "Free energy minimization\n"
            "+ Emotion Anchor",
            ha="center", va="center", fontsize=7.5,
            bbox=fep_kw)

    # Arrows
    arrow_kw = dict(
        arrowstyle="->,head_width=0.15,head_length=0.12",
        color=C_DARK, linewidth=1.2,
        connectionstyle="arc3,rad=0",
    )

    # Consciousness -> Pipeline (down)
    ax.annotate("", xy=(2.5, 5.6), xytext=(2.5, 6.3),
                arrowprops=arrow_kw)

    # Pipeline -> Output (down)
    ax.annotate("", xy=(2.5, 3.3), xytext=(2.5, 4.0),
                arrowprops=arrow_kw)

    # Output -> Feedback extraction (right)
    ax.annotate("", xy=(5.4, 2.6), xytext=(4.1, 2.6),
                arrowprops=arrow_kw)

    # Feedback extraction -> Audio features (up)
    ax.annotate("", xy=(7.0, 3.9), xytext=(7.0, 3.3),
                arrowprops=arrow_kw)

    # Audio features -> FEP (up)
    ax.annotate("", xy=(7.0, 6.1), xytext=(7.0, 5.6),
                arrowprops=arrow_kw)

    # FEP -> Consciousness (left, closing the loop)
    ax.annotate("", xy=(4.3, 7.0), xytext=(5.4, 7.0),
                arrowprops={**arrow_kw, "color": C_PURPLE,
                            "connectionstyle": "arc3,rad=0"})

    # Label the loop
    ax.text(5.0, 7.55, "Strange Loop", ha="center", va="center",
            fontsize=8, fontstyle="italic", color=C_PURPLE)

    # Phase labels on arrows
    ax.text(1.4, 5.95, "17 mappings", ha="center", va="center",
            fontsize=6.5, color=C_GREY, fontstyle="italic")
    ax.text(1.3, 3.65, "render", ha="center", va="center",
            fontsize=6.5, color=C_GREY, fontstyle="italic")
    ax.text(4.75, 2.2, "self-listen", ha="center", va="center",
            fontsize=6.5, color=C_GREY, fontstyle="italic")
    ax.text(8.0, 3.6, "extract", ha="center", va="center",
            fontsize=6.5, color=C_GREY, fontstyle="italic")
    ax.text(8.0, 5.85, "surprise", ha="center", va="center",
            fontsize=6.5, color=C_GREY, fontstyle="italic")

    fig.suptitle("Figure 1. Strange Loop Architecture", fontsize=10,
                 fontweight="bold", y=0.02)

    _save(fig, "fig1_strange_loop")


# ===================================================================
# FIGURE 2: Feature-State Correlation Matrix
# ===================================================================

def figure2_correlation_matrix():
    data = np.column_stack([R_AROUSAL, R_VALENCE])
    sig_matrix = np.array([SIG_AROUSAL, SIG_VALENCE]).T  # 11x2

    fig, ax = plt.subplots(figsize=(3.5, 5.5))

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Arousal", "Valence"], fontsize=9)
    ax.set_yticks(range(len(FEATURE_LABELS)))
    ax.set_yticklabels(FEATURE_LABELS, fontsize=8)
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)

    # Annotate cells
    for i in range(len(FEATURE_LABELS)):
        for j in range(2):
            val = data[i, j]
            sig = sig_matrix[i, j]
            txt = f"{val:+.3f}"
            if sig:
                txt += " ***"
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=6.5, color=text_color, fontweight="bold" if sig else "normal")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label("Pearson r", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Remove default spines; the imshow has its own border
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Feature-State Correlations\n(N=60, Bonferroni p<0.00227)",
                 fontsize=9, pad=20)

    fig.suptitle("Figure 2.", fontsize=10, fontweight="bold", y=0.02)

    _save(fig, "fig2_correlation_matrix")


# ===================================================================
# FIGURE 3: Ablation Study
# ===================================================================

def figure3_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.0), sharey=False)

    x = np.arange(len(ABLATION_CONFIGS))
    width = 0.6

    # Severity-based colors: green for best, red for worst
    severity_colors = []
    for d in ABLATION_DELTA:
        if d == 0.0:
            severity_colors.append(C_GREEN)
        elif d < 0:
            severity_colors.append(C_GREY)
        elif d < 60:
            severity_colors.append(C_BLUE)
        elif d < 100:
            severity_colors.append(C_ORANGE)
        else:
            severity_colors.append(C_RED)

    # --- Left panel: State Discrimination (lower = better) ---
    ax = axes[0]
    bars = ax.bar(x, ABLATION_DISCRIM, width, color=severity_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_CONFIGS, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("State Discrimination\n(lower = better)", fontsize=8)
    ax.set_title("(a) Feature-Space Discrimination", fontsize=9)
    ax.axhline(y=ABLATION_DISCRIM[0], color=C_GREEN, linestyle="--", linewidth=0.7, alpha=0.6)

    # Add delta labels
    for i, (bar, delta) in enumerate(zip(bars, ABLATION_DELTA)):
        if delta != 0.0:
            label = f"{delta:+.0f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    label, ha="center", va="bottom", fontsize=6, color=C_DARK)

    # --- Right panel: Mean |r| (arousal + valence) ---
    ax = axes[1]
    mean_r = (np.abs(ABLATION_R_A) + np.abs(ABLATION_R_V)) / 2
    bars = ax.bar(x, mean_r, width, color=severity_colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ABLATION_CONFIGS, fontsize=6.5, rotation=45, ha="right")
    ax.set_ylabel("Mean |r|  (arousal + valence) / 2", fontsize=8)
    ax.set_title("(b) Emotion Correlation Strength", fontsize=9)
    ax.axhline(y=mean_r[0], color=C_GREEN, linestyle="--", linewidth=0.7, alpha=0.6)

    # Annotate bars with R(A) / R(V)
    for i, bar in enumerate(bars):
        if ABLATION_R_A[i] > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"A={ABLATION_R_A[i]:.2f}\nV={ABLATION_R_V[i]:.2f}",
                    ha="center", va="bottom", fontsize=5.5, color=C_DARK)

    fig.suptitle("Figure 3. Ablation Study (7 configurations)",
                 fontsize=10, fontweight="bold", y=0.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    _save(fig, "fig3_ablation")


# ===================================================================
# FIGURE 4: Phi as Musical Identity
# ===================================================================

def figure4_phi_identity():
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    x = np.arange(len(FEATURE_LABELS))
    colors = [C_BLUE if sig else C_LIGHT_GREY for sig in PHI_SIG]
    edgecolors = [C_DARK if sig else C_GREY for sig in PHI_SIG]

    bars = ax.bar(x, R_PHI, width=0.65, color=colors, edgecolor=edgecolors, linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_LABELS, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Partial r (Phi | A, V)", fontsize=9)
    ax.set_title("Phi Partial Correlations Controlling for Arousal and Valence",
                 fontsize=9, pad=8)
    ax.axhline(y=0, color=C_DARK, linewidth=0.5)

    # Significance annotations
    for i, (bar, sig, r) in enumerate(zip(bars, PHI_SIG, R_PHI)):
        y_pos = r + (0.025 if r >= 0 else -0.04)
        if sig:
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, "*",
                    ha="center", va="bottom" if r >= 0 else "top",
                    fontsize=10, fontweight="bold", color=C_DARK)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_BLUE, edgecolor=C_DARK, label="p < 0.05 (significant)"),
        Patch(facecolor=C_LIGHT_GREY, edgecolor=C_GREY, label="n.s."),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=False, fontsize=7)

    fig.suptitle("Figure 4.", fontsize=10, fontweight="bold", y=0.02)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    _save(fig, "fig4_phi_identity")


# ===================================================================
# FIGURE 5: Neuromodulator Mapping Table (Sankey-style diagram)
# ===================================================================

def figure5_neuromodulator_mappings():
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Source groups (left side) with colors
    sources = {
        "DA (Dopamine)":    (C_ORANGE, ["FM depth", "Spectral brightness"]),
        "5-HT (Serotonin)": (C_GREEN,  ["Low-pass rolloff"]),
        "NE (Norepinephrine)": (C_RED, ["Partial ratio", "Vibrato rate", "Upper-partial boost"]),
        "Arousal":          (C_BLUE,   ["Gain (loudness)", "Attack time", "Tempo / cadence"]),
        "Valence":          (C_PURPLE, ["Mode (major/minor)", "Gain modulation", "Partial detuning"]),
        "Phi":              (C_TEAL,   ["Polyphony gating", "Reverb room size",
                                        "Sustain envelope", "Micro-vibrato"]),
        "Prediction Error": (C_BROWN,  ["Spectral flux target"]),
    }

    # Layout: sources on left (x=0.5-2.5), targets on right (x=6.5-9.5)
    n_sources = len(sources)
    source_y_start = 9.0
    source_y_step = 1.2

    # Collect all targets for y-positioning
    all_targets = []
    target_source_map = {}
    for src, (color, targets) in sources.items():
        for t in targets:
            all_targets.append(t)
            target_source_map[t] = (src, color)

    n_targets = len(all_targets)
    target_y_start = 9.3
    target_y_step = 0.52

    # Draw source boxes
    source_positions = {}
    for i, (src, (color, targets)) in enumerate(sources.items()):
        y = source_y_start - i * source_y_step
        source_positions[src] = (1.5, y)

        bbox = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.15,
                    edgecolor=color, linewidth=1.0)
        ax.text(1.5, y, src, ha="center", va="center", fontsize=8,
                fontweight="bold", color=color, bbox=bbox)

    # Draw target labels and connecting lines
    for j, target in enumerate(all_targets):
        y = target_y_start - j * target_y_step
        src_name, color = target_source_map[target]
        src_x, src_y = source_positions[src_name]

        # Target text
        ax.text(8.5, y, target, ha="center", va="center", fontsize=7.5,
                color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.08,
                          edgecolor=color, linewidth=0.5))

        # Mapping number
        mapping_num = j + 1
        ax.text(6.3, y, f"#{mapping_num}", ha="center", va="center",
                fontsize=6, color=C_GREY)

        # Connecting curve
        ax.annotate(
            "", xy=(6.8, y), xytext=(3.0, src_y),
            arrowprops=dict(
                arrowstyle="->,head_width=0.08,head_length=0.06",
                color=color, alpha=0.45, linewidth=0.8,
                connectionstyle="arc3,rad=0.15",
            ),
        )

    # Title and citation column
    ax.text(5.0, 9.8, "Consciousness State  -->  Synthesis Parameter",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C_DARK)

    # Citation annotations (grouped)
    citations = [
        (0.5, 0.6, "Citations: DA: Berridge 2007; 5-HT/NE: Zentner 2008, Vassilakis 2005;"),
        (0.5, 0.25, "Arousal: Weber 1834, Eerola 2013, Gabrielsson 2010; Valence: Huron 2006,"),
        (0.5, -0.1, "Plomp & Levelt 1965; Phi: Tononi 2004; PE: Friston 2010"),
    ]
    for x_c, y_c, txt in citations:
        ax.text(x_c, y_c, txt, ha="left", va="center", fontsize=5.5,
                color=C_GREY, fontstyle="italic")

    fig.suptitle("Figure 5. Neuromodulator-to-Music Mappings (17 pathways)",
                 fontsize=10, fontweight="bold", y=0.02)

    _save(fig, "fig5_neuromodulator_mappings")


# ===================================================================
# MAIN
# ===================================================================

def main():
    print("Generating figures for consciousness-music-synthesis paper...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    figure1_strange_loop()
    figure2_correlation_matrix()
    figure3_ablation()
    figure4_phi_identity()
    figure5_neuromodulator_mappings()

    print()
    print("All figures generated.")


if __name__ == "__main__":
    main()
