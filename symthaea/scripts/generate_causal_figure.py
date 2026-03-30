#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate Figure 6: Causal Intervention Results

Visualizes the ablation study results showing that phenomenal-discriminating
heads have specific causal effects on phenomenal concept clustering.

Usage:
    cd /srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb
    nix-shell -p "python313.withPackages(ps: with ps; [numpy matplotlib seaborn])" \
        --run "python3 scripts/generate_causal_figure.py"
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

def load_results():
    """Load causal intervention results."""
    with open("/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/data/causal_intervention_results.json") as f:
        return json.load(f)

def create_figure(results, extended_results=None):
    """Create the causal intervention figure."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # === Panel A: Specificity comparison ===
    ax1 = axes[0]

    interventions = results["interventions"]
    names = [r["name"].split(" (")[0] for r in interventions]
    phen_changes = [r["changes"]["phenomenal"] * 100 for r in interventions]  # Convert to %
    func_changes = [r["changes"]["functional"] * 100 for r in interventions]
    specificities = [r["changes"]["specificity"] * 100 for r in interventions]

    # Color by type
    colors = []
    for r in interventions:
        if "Phenomenal" in r["name"]:
            colors.append("#e74c3c")  # Red for phenomenal
        elif "Functional" in r["name"]:
            colors.append("#3498db")  # Blue for functional
        else:
            colors.append("#95a5a6")  # Gray for control

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, phen_changes, width, label='Phenomenal Δ', color='#e74c3c', alpha=0.7)
    bars2 = ax1.bar(x + width/2, func_changes, width, label='Functional Δ', color='#3498db', alpha=0.7)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Clustering Change (%)')
    ax1.set_xlabel('Ablated Head')
    ax1.set_title('A. Effect of Head Ablation on Concept Clustering')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend(loc='lower left')

    # Add specificity annotations
    for i, (pc, fc, spec) in enumerate(zip(phen_changes, func_changes, specificities)):
        if spec > 0.5:
            ax1.annotate(f'+{spec:.1f}%', (i, max(pc, fc) + 0.3),
                        ha='center', fontsize=8, color='#27ae60')
        elif spec < -0.5:
            ax1.annotate(f'{spec:.1f}%', (i, min(pc, fc) - 0.5),
                        ha='center', fontsize=8, color='#e67e22')

    # === Panel B: Specificity by head type ===
    ax2 = axes[1]

    # Group by type
    phen_heads = [r for r in interventions if "Phenomenal" in r["name"]]
    ctrl_heads = [r for r in interventions if "Phenomenal" not in r["name"]]

    phen_specs = [r["changes"]["specificity"] * 100 for r in phen_heads]
    ctrl_specs = [r["changes"]["specificity"] * 100 for r in ctrl_heads]

    # Box plot style comparison
    positions = [1, 2]
    bp_data = [phen_specs, ctrl_specs]

    # Plot individual points
    for i, (data, pos, color) in enumerate(zip(bp_data, positions, ['#e74c3c', '#95a5a6'])):
        jitter = np.random.uniform(-0.1, 0.1, len(data))
        ax2.scatter([pos + j for j in jitter], data, c=color, s=100, alpha=0.7, zorder=3)
        # Mean line
        ax2.hlines(np.mean(data), pos - 0.3, pos + 0.3, colors=color, linewidth=3, zorder=4)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Specificity (%)\n(|Phen Δ| - |Func Δ|)')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(['Phenomenal\nHeads (n=3)', 'Control\nHeads (n=3)'])
    ax2.set_title('B. Causal Specificity by Head Type')

    # Add mean annotations
    phen_mean = np.mean(phen_specs)
    ctrl_mean = np.mean(ctrl_specs)
    ax2.annotate(f'Mean: {phen_mean:+.2f}%', (1, phen_mean + 0.3),
                ha='center', fontsize=9, fontweight='bold', color='#c0392b')
    ax2.annotate(f'Mean: {ctrl_mean:+.2f}%', (2, ctrl_mean - 0.5),
                ha='center', fontsize=9, fontweight='bold', color='#7f8c8d')

    # Significance annotation
    diff = phen_mean - ctrl_mean
    ax2.annotate(f'Difference: {diff:+.2f}%\n(Phenomenal heads\nmore specific)',
                xy=(1.5, max(phen_specs) + 0.5), ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()

    # Save
    output_dir = "/srv/luminous-dynamics/11-meta-consciousness/luminous-nix/symthaea-hlb/docs/papers/figures"
    plt.savefig(f"{output_dir}/fig6_causal_intervention.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/fig6_causal_intervention.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/fig6_causal_intervention.png")
    print(f"Saved: {output_dir}/fig6_causal_intervention.pdf")

    plt.close()


def main():
    print("Generating Figure 6: Causal Intervention Results...")
    results = load_results()
    create_figure(results)
    print("Done!")


if __name__ == "__main__":
    main()
