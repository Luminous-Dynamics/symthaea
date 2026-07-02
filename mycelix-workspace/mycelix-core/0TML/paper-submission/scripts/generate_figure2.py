#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate Figure 2: Confusion Matrix Grid Across BFT Levels
2×2 grid of confusion matrices at different Byzantine ratios
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

configs = [
    (20, 16, 4, 16, 0, 4, 0, "20% BFT", "100% Precision"),
    (35, 13, 7, 12, 1, 7, 0, "35% BFT", "87.5% Precision"),
    (45, 11, 9, 10, 1, 9, 0, "45% BFT", "90.0% Precision"),
    (50, 10, 10, 9, 1, 10, 0, "50% BFT", "90.9% Precision")
]

for idx, (bft, hn, bn, tn, fp, tp, fn, title, precision) in enumerate(configs):
    ax = axes[idx // 2, idx % 2]

    # Create confusion matrix
    matrix = np.array([[tp, fn], [fp, tn]])

    # Plot
    im = ax.imshow(matrix, cmap='YlGn', alpha=0.6, vmin=0, vmax=max(hn, bn))

    # Add values
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, matrix[i, j],
                          ha="center", va="center", color="black", fontsize=20,
                          fontweight='bold')

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nByzantine', 'Predicted\nHonest'], fontsize=10)
    ax.set_yticklabels(['Actual\nByzantine', 'Actual\nHonest'], fontsize=10)

    # Title
    ax.set_title(f'{title}\n{precision}', fontsize=12, fontweight='bold')

    # Annotations
    ax.text(0, -0.4, f'TP={tp}', ha='center', fontsize=9, color='green')
    ax.text(1, -0.4, f'FN={fn}', ha='center', fontsize=9, color='red')
    ax.text(0, 1.4, f'FP={fp}', ha='center', fontsize=9, color='orange')
    ax.text(1, 1.4, f'TN={tn}', ha='center', fontsize=9, color='green')

plt.tight_layout()
plt.savefig('figures/confusion_matrix_grid.pdf', dpi=300, bbox_inches='tight')
print("✅ Figure 2 generated: figures/confusion_matrix_grid.pdf")
