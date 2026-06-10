# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#!/usr/bin/env python3
"""Generate the main B(t) figure for the Catastrophe as Genesis paper.

Full-page figure with 3 panels:
  A) B(t) baseline curve with CI bands + extinction event markers
  B) Counterfactual branches (6 events, HardCap regime) vs baseline
  C) Consciousness feasibility overlay with milestone markers

Reads embedded data directly (no dependency on Rust binary output).
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data paths ──
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "mycelix-multiworld-sim" / "data" / "biosphere"
OUT_DIR = Path(__file__).resolve().parent

def load_csv(filename):
    """Load a CSV file from the biosphere data directory."""
    rows = []
    with open(DATA_DIR / filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# ── B(t) calibration points (from cargo run output) ──
BT_POINTS = [
    (3500.0, 0.005, 0.001, 0.07),
    (2400.0, 0.034, 0.010, 0.11),
    (541.0, 0.127, 0.083, 0.24),
    (445.0, 0.175, 0.134, 0.23),
    (372.0, 0.20, 0.15, 0.27),
    (252.0, 0.246, 0.211, 0.29),
    (201.0, 0.30, 0.25, 0.36),
    (66.0, 0.647, 0.603, 0.69),
    (28.0, 0.80, 0.76, 0.84),
    (5.0, 0.88, 0.85, 0.91),
    (0.006, 0.949, 0.925, 0.94),
]

# ── Extinction events ──
EXTINCTIONS = [
    (2400, "GOE", 0.70),
    (717, "Sturtian\nSnowball", 0.50),
    (541, "End-\nEdiacaran", 0.50),
    (445, "End-\nOrdovician", 0.57),
    (372, "Late\nDevonian", 0.35),
    (252, "End-\nPermian", 0.57),
    (201, "End-\nTriassic", 0.34),
    (66, "K-Pg", 0.40),
]

# ── Counterfactual terminal values (HardCap) ──
CF_EVENTS = [
    ("GOE", 2400, 0.099),
    ("End-Ordovician", 445, 0.325),
    ("Late Devonian", 372, 0.475),
    ("End-Permian", 252, 0.585),
    ("End-Triassic", 201, 0.528),
    ("K-Pg", 66, 0.765),
]
BASELINE_TERMINAL = 0.949

# ── Consciousness milestones ──
CONSCIOUSNESS = [
    (406, 0.020, "Minimal\nconsciousness"),
    (311, 0.071, "Sentient\nexperience"),
    (123, 0.164, "Self-\nawareness"),
    (28, 0.374, "Meta-cognitive\nreflection"),
]

# ── Build interpolated B(t) curve ──
def interpolate_bt():
    """Create a smooth B(t) curve from calibration points."""
    ages = np.array([p[0] for p in BT_POINTS])
    vals = np.array([p[1] for p in BT_POINTS])
    ci_lo = np.array([p[2] for p in BT_POINTS])
    ci_hi = np.array([p[3] for p in BT_POINTS])

    # Log-space interpolation for smooth curve across 4 orders of magnitude
    log_ages = np.log10(ages + 0.001)
    fine_log_ages = np.linspace(log_ages.max(), log_ages.min(), 500)
    fine_ages = 10**fine_log_ages

    fine_vals = np.interp(fine_log_ages, log_ages[::-1], vals[::-1])
    fine_lo = np.interp(fine_log_ages, log_ages[::-1], ci_lo[::-1])
    fine_hi = np.interp(fine_log_ages, log_ages[::-1], ci_hi[::-1])

    return fine_ages, fine_vals, fine_lo, fine_hi

def main():
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10), gridspec_kw={'height_ratios': [3, 2, 2]})
    fig.patch.set_facecolor('white')

    ages, vals, ci_lo, ci_hi = interpolate_bt()

    # ── Panel A: B(t) baseline with CI bands + extinction markers ──
    ax = axes[0]
    ax.set_facecolor('#fafafa')
    ax.fill_between(ages, ci_lo, ci_hi, color='#4ECDC4', alpha=0.15, label='95% CI')
    ax.plot(ages, vals, color='#2C3E50', linewidth=1.8, label='$B(t)$ baseline', zorder=3)

    # Extinction markers
    for age, label, severity in EXTINCTIONS:
        ax.axvline(age, color='#E74C3C', alpha=0.25, linewidth=1, linestyle='-')
        y_pos = 0.92 if age > 300 else 0.85
        ax.annotate(label, xy=(age, y_pos), fontsize=5.5, color='#E74C3C',
                    ha='center', va='top', rotation=0)

    ax.set_xscale('log')
    ax.set_xlim(4000, 0.004)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Biosphere Coherence $B(t)$', fontsize=9)
    ax.set_title('A. Biosphere Coherence Index Across 3.8 Billion Years', fontsize=10, fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    ax.set_xticks([3000, 1000, 500, 200, 100, 50, 10, 1, 0.01])
    ax.set_xticklabels(['3 Ga', '1 Ga', '500 Ma', '200 Ma', '100 Ma', '50 Ma', '10 Ma', '1 Ma', '10 Ka'], fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.98, 0.05, 'Sepkoski validation: $r = 0.92$\nNon-circular: $r = 0.83$',
            transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ── Panel B: Counterfactual branches (HardCap) ──
    ax = axes[1]
    ax.set_facecolor('#fafafa')

    colors = ['#9B59B6', '#3498DB', '#1ABC9C', '#E67E22', '#E74C3C', '#2980B9']
    for i, (name, ext_age, terminal) in enumerate(CF_EVENTS):
        # Draw a horizontal line from extinction age to terminal, showing the cap
        ax.plot([ext_age, 0.005], [terminal, terminal], color=colors[i],
                linewidth=1.2, linestyle='--', alpha=0.7)
        ax.scatter([ext_age], [terminal], color=colors[i], s=20, zorder=5)
        ax.annotate(f'{name}\n$B_{{term}}={terminal:.2f}$', xy=(ext_age * 0.7, terminal),
                    fontsize=5.5, color=colors[i], va='center')

    # Baseline
    ax.axhline(BASELINE_TERMINAL, color='#2C3E50', linewidth=1.5, label=f'Baseline $B(t)={BASELINE_TERMINAL:.3f}$')
    ax.plot(ages, vals, color='#2C3E50', linewidth=1, alpha=0.3)

    ax.set_xscale('log')
    ax.set_xlim(4000, 0.004)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel('Terminal $B(t)$ at 3000 BCE', fontsize=9)
    ax.set_title('B. Counterfactual Branches (Hard Cap Regime)', fontsize=10, fontweight='bold', loc='left')
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)
    ax.set_xticks([3000, 1000, 500, 200, 100, 50, 10, 1, 0.01])
    ax.set_xticklabels(['3 Ga', '1 Ga', '500 Ma', '200 Ma', '100 Ma', '50 Ma', '10 Ma', '1 Ma', '10 Ka'], fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.98, 0.05, '6/6 events generative\n(HardCap regime)',
            transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ── Panel C: Consciousness feasibility ──
    ax = axes[2]
    ax.set_facecolor('#fafafa')

    # Feasibility curve (approximate from milestones)
    c_ages = [3500, 600, 500] + [m[0] for m in CONSCIOUSNESS] + [5, 0.006]
    c_vals = [0, 0, 0.005] + [m[1] for m in CONSCIOUSNESS] + [0.42, 0.46]
    c_ages_np = np.array(c_ages, dtype=float)
    c_vals_np = np.array(c_vals, dtype=float)

    ax.fill_between([4000, 0.004], [0.30, 0.30], [1.0, 1.0], color='#27AE60', alpha=0.08,
                    label='Meta-cognition zone ($F > 0.30$)')
    ax.axhline(0.30, color='#27AE60', linewidth=1, linestyle=':', alpha=0.5)
    ax.axhline(0.15, color='#F39C12', linewidth=0.8, linestyle=':', alpha=0.4)
    ax.axhline(0.05, color='#E67E22', linewidth=0.8, linestyle=':', alpha=0.4)
    ax.axhline(0.01, color='#E74C3C', linewidth=0.8, linestyle=':', alpha=0.4)

    ax.plot(c_ages_np, c_vals_np, color='#8E44AD', linewidth=1.8, label='Consciousness feasibility $F(t)$', zorder=3)

    # K-Pg suppressed (HardCap) — peaks at 0.24
    ax.axhline(0.24, color='#E74C3C', linewidth=1, linestyle='--', alpha=0.6,
              label='K-Pg suppressed peak ($F=0.24$, HardCap)')

    # Milestone markers
    for age, feas, label in CONSCIOUSNESS:
        ax.scatter([age], [feas], color='#8E44AD', s=30, zorder=5, edgecolors='white', linewidth=0.5)
        ax.annotate(label, xy=(age, feas), xytext=(age * 1.5, feas + 0.03),
                    fontsize=5.5, color='#8E44AD', ha='left',
                    arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=0.5))

    # The 0.06 gap annotation
    ax.annotate('', xy=(40, 0.30), xytext=(40, 0.24),
                arrowprops=dict(arrowstyle='<->', color='#E74C3C', lw=1.5))
    ax.text(50, 0.27, '$\\Delta = 0.06$\n(knife-edge)', fontsize=6, color='#E74C3C', ha='left', va='center')

    ax.set_xscale('log')
    ax.set_xlim(4000, 0.004)
    ax.set_ylim(-0.02, 0.55)
    ax.set_xlabel('Age (log scale)', fontsize=9)
    ax.set_ylabel('Consciousness Feasibility $F(t)$', fontsize=9)
    ax.set_title('C. Consciousness Emergence Timeline (Workspace Bottleneck)', fontsize=10, fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=6, framealpha=0.9)
    ax.set_xticks([3000, 1000, 500, 200, 100, 50, 10, 1, 0.01])
    ax.set_xticklabels(['3 Ga', '1 Ga', '500 Ma', '200 Ma', '100 Ma', '50 Ma', '10 Ma', '1 Ma', '10 Ka'], fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.98, 0.92, 'Bottleneck: Information Capacity\n(workspace) throughout 3.8 Ga',
            transform=ax.transAxes, fontsize=6.5, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(h_pad=1.5)
    out_path = OUT_DIR / 'fig1_biosphere_coherence.pdf'
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out_path}")

    # Also save PNG for preview
    fig.savefig(OUT_DIR / 'fig1_biosphere_coherence.png', dpi=150, bbox_inches='tight')
    print(f"PNG preview saved")

if __name__ == '__main__':
    main()
