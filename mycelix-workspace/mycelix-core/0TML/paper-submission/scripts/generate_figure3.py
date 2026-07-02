#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate Figure 3: Mode 0 vs Mode 1 Direct A/B Comparison
Side-by-side confusion matrices showing detector inversion vs ground truth success
"""

import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Mode 0 (Peer-Comparison) - Complete Inversion
mode0_matrix = np.array([[7, 0], [13, 0]])  # TP, FN, FP, TN
mode0_title = "Mode 0: Peer-Comparison\n(Complete Detector Inversion)"

# Mode 1 (Ground Truth - PoGQ) - Perfect Discrimination
mode1_matrix = np.array([[7, 0], [0, 13]])  # TP, FN, FP, TN
mode1_title = "Mode 1: Ground Truth (PoGQ)\n(Perfect Discrimination)"

for ax, matrix, title, mode in [(ax1, mode0_matrix, mode0_title, "Mode 0"),
                                  (ax2, mode1_matrix, mode1_title, "Mode 1")]:
    # Color map: green for correct, red for errors
    if mode == "Mode 0":
        # Mode 0 has massive FP problem - use red for bottom-left
        colors = np.array([[0, 1], [2, 1]])  # 0=green (TP), 1=white (FN,TN), 2=red (FP)
        cmap_colors = ['green', 'white', 'red']
    else:
        # Mode 1 is perfect
        colors = np.array([[0, 1], [1, 0]])  # 0=green (TP,TN), 1=white (FN,FP)
        cmap_colors = ['green', 'white']

    im = ax.imshow(colors, cmap=plt.cm.colors.ListedColormap(cmap_colors),
                   alpha=0.6, vmin=0, vmax=2)

    # Add values
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            # Color code the text
            if i == 0 and j == 0:  # TP
                color = 'darkgreen'
                weight = 'bold'
            elif i == 1 and j == 0:  # FP
                color = 'darkred'
                weight = 'bold'
            elif i == 0 and j == 1:  # FN
                color = 'red'
                weight = 'normal'
            else:  # TN
                color = 'green'
                weight = 'normal'

            ax.text(j, i, value, ha="center", va="center",
                   color=color, fontsize=24, fontweight=weight)

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nByzantine', 'Predicted\nHonest'], fontsize=11)
    ax.set_yticklabels(['Actual\nByzantine', 'Actual\nHonest'], fontsize=11)

    # Title
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Annotations
    tp, fn = matrix[0, 0], matrix[0, 1]
    fp, tn = matrix[1, 0], matrix[1, 1]

    ax.text(0, -0.5, f'TP={tp}', ha='center', fontsize=10, color='green')
    ax.text(1, -0.5, f'FN={fn}', ha='center', fontsize=10, color='red' if fn > 0 else 'gray')
    ax.text(0, 1.5, f'FP={fp}', ha='center', fontsize=10, color='red' if fp > 0 else 'gray')
    ax.text(1, 1.5, f'TN={tn}', ha='center', fontsize=10, color='green' if tn > 0 else 'gray')

# Add overall comparison text
fig.text(0.5, 0.02,
         'Direct A/B Comparison at 35% BFT (Seed=42, Identical Data)\n'
         'Mode 0: 100% Detection, 100% FPR (All honest nodes flagged) | '
         'Mode 1: 100% Detection, 0% FPR (Perfect discrimination)',
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('figures/mode0_vs_mode1_comparison.pdf', dpi=300, bbox_inches='tight')
print("✅ Figure 3 generated: figures/mode0_vs_mode1_comparison.pdf")
