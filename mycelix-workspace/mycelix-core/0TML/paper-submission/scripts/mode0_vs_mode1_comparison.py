# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Visualization: Mode 0 vs Mode 1 Comparison at 35% BFT

Creates bar chart showing the catastrophic failure of peer-comparison (Mode 0)
vs the perfect performance of ground truth validation (Mode 1) at 35% BFT
with heterogeneous data.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def create_mode0_vs_mode1_comparison():
    """
    Create side-by-side bar chart comparing Mode 0 vs Mode 1.

    Shows:
    - Detection Rate (both 100%)
    - False Positive Rate (Mode 0: 100%, Mode 1: 0%)

    This visualization demonstrates the CATASTROPHIC failure of peer-comparison
    at 35% BFT with heterogeneous data.
    """

    # Data from test results
    mode0_detection = 100.0  # %
    mode0_fpr = 100.0        # %
    mode1_detection = 100.0  # %
    mode1_fpr = 0.0          # %

    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Mode 0 (Peer-Comparison) vs Mode 1 (Ground Truth) at 35% BFT\n'
        'Heterogeneous Data (Realistic Federated Learning)',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )

    # Color scheme
    mode0_color = '#e74c3c'  # Red (failure)
    mode1_color = '#27ae60'  # Green (success)
    bar_width = 0.35
    x = np.arange(2)  # Two metrics: Detection, FPR

    # Left subplot: Detection Rate
    ax1.set_title('Byzantine Detection Rate', fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Detection Rate (%)', fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Detection Rate'])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Detection bars
    bars1_mode0 = ax1.bar(0 - bar_width/2, mode0_detection, bar_width,
                          label='Mode 0 (Peer-Comparison)',
                          color=mode0_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars1_mode1 = ax1.bar(0 + bar_width/2, mode1_detection, bar_width,
                          label='Mode 1 (Ground Truth - PoGQ)',
                          color=mode1_color, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in [bars1_mode0, bars1_mode1]:
        for rect in bar:
            height = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width()/2., height + 2,
                    f'{height:.0f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add checkmarks/crosses
    ax1.text(0 - bar_width/2, mode0_detection + 8, '✓',
            ha='center', va='bottom', fontsize=20, color=mode0_color, fontweight='bold')
    ax1.text(0 + bar_width/2, mode1_detection + 8, '✓',
            ha='center', va='bottom', fontsize=20, color=mode1_color, fontweight='bold')

    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Right subplot: False Positive Rate
    ax2.set_title('False Positive Rate (FPR)\n⚠️ CATASTROPHIC FAILURE',
                 fontsize=14, fontweight='bold', pad=15, color='#c0392b')
    ax2.set_ylabel('False Positive Rate (%)', fontsize=12)
    ax2.set_ylim(0, 110)
    ax2.set_xticks([0])
    ax2.set_xticklabels(['FPR'])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # FPR bars
    bars2_mode0 = ax2.bar(0 - bar_width/2, mode0_fpr, bar_width,
                          label='Mode 0 (Peer-Comparison)',
                          color=mode0_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2_mode1 = ax2.bar(0 + bar_width/2, mode1_fpr, bar_width,
                          label='Mode 1 (Ground Truth - PoGQ)',
                          color=mode1_color, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    ax2.text(0 - bar_width/2, mode0_fpr + 2,
            f'{mode0_fpr:.0f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.text(0 + bar_width/2, max(5, mode1_fpr) + 2,
            f'{mode1_fpr:.0f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add checkmarks/crosses
    ax2.text(0 - bar_width/2, mode0_fpr + 8, '✗',
            ha='center', va='bottom', fontsize=24, color='#c0392b', fontweight='bold')
    ax2.text(0 + bar_width/2, max(5, mode1_fpr) + 8, '✓',
            ha='center', va='bottom', fontsize=20, color=mode1_color, fontweight='bold')

    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Add horizontal line at 10% (acceptable threshold)
    ax2.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(0.5, 11, 'Acceptable Threshold (10%)',
            ha='center', fontsize=9, color='orange', fontweight='bold')

    # Add annotations
    fig.text(0.5, -0.05,
            'Configuration: 20 clients (13 honest, 7 Byzantine = 35% BFT) | '
            'Heterogeneous MNIST data | Sign flip attack | Seed=42',
            ha='center', fontsize=10, style='italic')

    fig.text(0.5, -0.10,
            '⚠️ Mode 0 flags ALL 13 honest nodes (100% FPR) - Complete Detector Inversion',
            ha='center', fontsize=11, color='#c0392b', fontweight='bold')

    fig.text(0.5, -0.14,
            '✅ Mode 1 achieves perfect discrimination (0% FPR) using ground truth validation',
            ha='center', fontsize=11, color=mode1_color, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Save figure
    output_path = '/tmp/mode0_vs_mode1_35bft.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved visualization: {output_path}")

    # Also save as SVG for paper
    output_path_svg = '/tmp/mode0_vs_mode1_35bft.svg'
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved vector graphic: {output_path_svg}")

    plt.close()


def create_detailed_breakdown():
    """
    Create detailed breakdown showing TP, FP, TN, FN for both detectors.

    This gives a complete picture of the confusion matrix for each approach.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        'Mode 0 vs Mode 1: Complete Confusion Matrix Breakdown at 35% BFT',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    # Data
    categories = ['True\nPositives', 'False\nPositives', 'True\nNegatives', 'False\nNegatives']

    # Mode 0 results
    mode0_values = [7, 13, 0, 0]  # TP, FP, TN, FN
    mode0_colors = ['#27ae60', '#e74c3c', '#27ae60', '#e74c3c']  # Green for good, red for bad
    mode0_alpha = [0.8, 0.8, 0.8, 0.8]

    # Mode 1 results
    mode1_values = [7, 0, 13, 0]  # TP, FP, TN, FN
    mode1_colors = ['#27ae60', '#27ae60', '#27ae60', '#27ae60']  # All green!
    mode1_alpha = [0.8, 0.8, 0.8, 0.8]

    x = np.arange(len(categories))
    width = 0.6

    # Mode 0 subplot
    ax1.set_title('Mode 0: Peer-Comparison\n❌ COMPLETE INVERSION',
                 fontsize=14, fontweight='bold', pad=15, color='#c0392b')
    ax1.set_ylabel('Number of Clients', fontsize=12)
    ax1.set_ylim(0, 16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    bars1 = ax1.bar(x, mode0_values, width, color=mode0_colors, alpha=mode0_alpha,
                    edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, mode0_values)):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{val}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add problem annotation
    ax1.text(1, 14, '⚠️ All honest\nflagged!',
            ha='center', fontsize=10, color='#c0392b', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#c0392b', linewidth=2))

    # Mode 1 subplot
    ax2.set_title('Mode 1: Ground Truth (PoGQ)\n✅ PERFECT DISCRIMINATION',
                 fontsize=14, fontweight='bold', pad=15, color='#27ae60')
    ax2.set_ylabel('Number of Clients', fontsize=12)
    ax2.set_ylim(0, 16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    bars2 = ax2.bar(x, mode1_values, width, color=mode1_colors, alpha=mode1_alpha,
                    edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, mode1_values)):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{val}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add success annotation
    ax2.text(2, 14, '✅ Perfect!\n0 errors',
            ha='center', fontsize=10, color='#27ae60', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#27ae60', linewidth=2))

    # Add bottom annotations
    fig.text(0.5, 0.02,
            'Total: 20 clients (13 honest, 7 Byzantine) | Heterogeneous MNIST data | '
            'Sign flip attack',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figure
    output_path = '/tmp/mode0_vs_mode1_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved breakdown visualization: {output_path}")

    output_path_svg = '/tmp/mode0_vs_mode1_breakdown.svg'
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved vector graphic: {output_path_svg}")

    plt.close()


if __name__ == "__main__":
    print("Creating Mode 0 vs Mode 1 comparison visualizations...")
    print()

    print("Visualization 1: Detection Rate and FPR Comparison")
    create_mode0_vs_mode1_comparison()
    print()

    print("Visualization 2: Detailed Confusion Matrix Breakdown")
    create_detailed_breakdown()
    print()

    print("✅ All visualizations created successfully!")
    print()
    print("Output files:")
    print("  - /tmp/mode0_vs_mode1_35bft.png (main comparison)")
    print("  - /tmp/mode0_vs_mode1_35bft.svg (vector)")
    print("  - /tmp/mode0_vs_mode1_breakdown.png (detailed)")
    print("  - /tmp/mode0_vs_mode1_breakdown.svg (vector)")
