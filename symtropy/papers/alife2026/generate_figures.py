# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Generate figures for ALIFE 2026 paper."""

# Check if matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — generating ASCII figures instead")

def ablation_figure():
    """Figure 1: Ablation study bar chart."""
    conditions = ['FREE', 'E_ONLY', 'E+OFF', 'E+GRAD', 'E+O+R', 'FULL']
    clustering = [15.1, 15.1, 15.1, 1.8, 15.1, 13.5]
    collapse = [0, 0, 0, 72, 0, 45]

    if HAS_MPL:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        colors = ['#888888', '#888888', '#888888', '#cc4444', '#888888', '#4488cc']
        ax1.bar(conditions, clustering, color=colors)
        ax1.set_ylabel('Avg Nearest-Neighbor Distance')
        ax1.set_title('(a) Spatial Clustering')
        ax1.axhline(y=15.1, color='gray', linestyle='--', alpha=0.3, label='Baseline')

        ax2.bar(conditions, collapse, color=colors)
        ax2.set_ylabel('Collapse Rate (%)')
        ax2.set_title('(b) Agent Survival')

        plt.tight_layout()
        plt.savefig('fig_ablation.pdf', bbox_inches='tight')
        plt.savefig('fig_ablation.png', bbox_inches='tight', dpi=150)
        print("Generated fig_ablation.pdf")
    else:
        print("\nABLATION FIGURE (ASCII):")
        for c, cl, co in zip(conditions, clustering, collapse):
            bar = '█' * int(cl)
            print(f"  {c:>8} | {bar} {cl:.1f}  collapse={co}%")

def causal_figure():
    """Figure 2: Phi causal test."""
    conditions = ['Φ-COUP', 'H-COUP', 'R-COUP', 'C-COUP', 'Φ=0', 'DECOUP']
    clustering = [5.57, 5.65, 5.90, 5.57, 7.50, 5.57]
    ci = [0.48, 0.83, 0.87, 0.48, 3.41, 0.48]

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#4488cc', '#66aa66', '#cc8844', '#4488cc', '#cc4444', '#888888']
        bars = ax.bar(conditions, clustering, yerr=ci, color=colors, capsize=5)
        ax.set_ylabel('Avg Nearest-Neighbor Distance')
        ax.set_title('Φ Causal Test: Conscious vs Unconscious')
        ax.axhline(y=5.57, color='blue', linestyle='--', alpha=0.3, label='Φ>0 baseline')
        ax.axhline(y=7.50, color='red', linestyle='--', alpha=0.3, label='Φ=0')
        ax.legend()
        plt.tight_layout()
        plt.savefig('fig_causal.pdf', bbox_inches='tight')
        plt.savefig('fig_causal.png', bbox_inches='tight', dpi=150)
        print("Generated fig_causal.pdf")
    else:
        print("\nCAUSAL FIGURE (ASCII):")
        for c, cl, e in zip(conditions, clustering, ci):
            bar = '█' * int(cl * 3)
            print(f"  {c:>8} | {bar} {cl:.2f} ±{e:.2f}")

def phase_figure():
    """Figure 3: Phase diagram."""
    costs = [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50]
    densities = [6, 12, 24, 48]
    # C=cooperative(100), P=partial, X=extinct
    data = [
        [100, 100, 100, 100],  # 0.02
        [77, 75, 100, 100],    # 0.05
        [55, 53, 98, 100],     # 0.08
        [48, 65, 94, 100],     # 0.12
        [57, 76, 96, 100],     # 0.18
        [47, 77, 100, 100],    # 0.25
        [67, 74, 100, 100],    # 0.35
        [30, 89, 100, 100],    # 0.50
    ]

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(6, 5))
        data_arr = np.array(data)
        im = ax.imshow(data_arr, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(len(densities)))
        ax.set_xticklabels(densities)
        ax.set_yticks(range(len(costs)))
        ax.set_yticklabels([f'{c:.2f}' for c in costs])
        ax.set_xlabel('Population Density (agents in 100×100)')
        ax.set_ylabel('Energy Cost (J/tick)')
        ax.set_title('Phase Diagram: Survival Rate (%)')

        for i in range(len(costs)):
            for j in range(len(densities)):
                val = data[i][j]
                color = 'white' if val < 60 else 'black'
                ax.text(j, i, f'{val}%', ha='center', va='center', color=color, fontsize=9)

        plt.colorbar(im, label='Survival %')
        plt.tight_layout()
        plt.savefig('fig_phase.pdf', bbox_inches='tight')
        plt.savefig('fig_phase.png', bbox_inches='tight', dpi=150)
        print("Generated fig_phase.pdf")
    else:
        print("\nPHASE DIAGRAM (ASCII):")
        print(f"  {'Cost':>6} | " + " ".join(f"{d:>5}" for d in densities))
        print("  " + "-" * 30)
        for c, row in zip(costs, data):
            cells = " ".join(f"{v:>4}%" for v in row)
            print(f"  {c:>6.2f} | {cells}")

if __name__ == '__main__':
    ablation_figure()
    causal_figure()
    phase_figure()
    print("\nDone. Include fig_ablation.pdf, fig_causal.pdf, fig_phase.pdf in paper.")
