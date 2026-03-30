# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Visualization: Mode 1 (Ground Truth - PoGQ) Performance Across BFT Levels

Shows how Mode 1 detector performs as Byzantine ratio increases from 35% to 50%.
Demonstrates maintained high detection rate with gradually increasing FPR.
"""

import matplotlib.pyplot as plt
import numpy as np


def create_performance_across_bft():
    """
    Create line chart showing Mode 1 performance across BFT levels.

    Shows:
    - Detection Rate (stays at 100%)
    - False Positive Rate (increases gradually 0% → 10%)
    - Adaptive Threshold evolution
    """

    # Data from Mode 1 boundary validation tests
    bft_levels = [35, 40, 45, 50]
    detection_rates = [100.0, 100.0, 100.0, 100.0]
    fprs = [0.0, 8.3, 9.1, 10.0]
    adaptive_thresholds = [0.480175, 0.509740, 0.509740, 0.509740]

    # Configuration details for each test
    num_clients = 20
    num_byzantine = [7, 8, 9, 10]
    num_honest = [13, 12, 11, 10]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        'Mode 1 (Ground Truth - PoGQ) Performance Across BFT Levels\n'
        'Heterogeneous Data (Realistic Federated Learning)',
        fontsize=16,
        fontweight='bold',
        y=1.00
    )

    # Color scheme
    detection_color = '#27ae60'  # Green
    fpr_color = '#e74c3c'        # Red
    threshold_color = '#3498db'  # Blue

    # Left subplot: Detection Rate and FPR
    ax1.set_title('Detection Rate & False Positive Rate', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Byzantine Fault Tolerance (%)', fontsize=12)
    ax1.set_ylabel('Rate (%)', fontsize=12)
    ax1.set_xlim(32, 53)
    ax1.set_ylim(-5, 110)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Plot detection rate
    line1 = ax1.plot(bft_levels, detection_rates,
                     marker='o', markersize=10, linewidth=3,
                     color=detection_color, label='Detection Rate',
                     markeredgecolor='black', markeredgewidth=1.5)

    # Plot FPR
    line2 = ax1.plot(bft_levels, fprs,
                     marker='s', markersize=10, linewidth=3,
                     color=fpr_color, label='False Positive Rate',
                     markeredgecolor='black', markeredgewidth=1.5)

    # Add value labels
    for i, (bft, det, fpr) in enumerate(zip(bft_levels, detection_rates, fprs)):
        # Detection rate labels
        ax1.text(bft, det + 3, f'{det:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=detection_color)

        # FPR labels
        ax1.text(bft, fpr + 3, f'{fpr:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=fpr_color)

    # Add horizontal line at 10% (acceptable FPR threshold)
    ax1.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(52, 11, 'Acceptable FPR\nThreshold (10%)',
            ha='right', va='bottom', fontsize=9, color='orange', fontweight='bold')

    # Add vertical line at 35% (peer-comparison ceiling)
    ax1.axvline(x=35, color='purple', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(35.5, 100, 'Peer-Comparison\nCeiling (35%)',
            ha='left', va='top', fontsize=9, color='purple', fontweight='bold')

    ax1.legend(loc='lower left', fontsize=11, framealpha=0.9)

    # Right subplot: Adaptive Threshold Evolution
    ax2.set_title('Adaptive Threshold Evolution', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Byzantine Fault Tolerance (%)', fontsize=12)
    ax2.set_ylabel('Quality Score Threshold', fontsize=12)
    ax2.set_xlim(32, 53)
    ax2.set_ylim(0.45, 0.53)
    ax2.grid(True, alpha=0.3, linestyle='--')

    # Plot adaptive threshold
    line3 = ax2.plot(bft_levels, adaptive_thresholds,
                     marker='D', markersize=10, linewidth=3,
                     color=threshold_color, label='Adaptive Threshold',
                     markeredgecolor='black', markeredgewidth=1.5)

    # Add value labels
    for i, (bft, thresh) in enumerate(zip(bft_levels, adaptive_thresholds)):
        ax2.text(bft, thresh + 0.003, f'{thresh:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=threshold_color)

    # Add horizontal line at 0.5 (midpoint)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(52, 0.501, 'Midpoint (0.5)',
            ha='right', va='bottom', fontsize=8, color='gray', style='italic')

    # Add annotation about threshold convergence
    ax2.annotate('Threshold Convergence\n(40-50% BFT)',
                xy=(45, 0.509740), xytext=(42, 0.52),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5))

    ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Add bottom annotations
    fig.text(0.5, -0.02,
            'Configuration: 20 clients total | Heterogeneous MNIST data | '
            'Sign flip attack | Adaptive threshold enabled',
            ha='center', fontsize=10, style='italic')

    fig.text(0.5, -0.06,
            '✅ Mode 1 maintains 100% detection across all BFT levels | '
            'FPR increases gradually (0% → 10%) | Remains operational at 50% BFT',
            ha='center', fontsize=11, color=detection_color, fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Save figure
    output_path = '/tmp/mode1_performance_across_bft.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved visualization: {output_path}")

    output_path_svg = '/tmp/mode1_performance_across_bft.svg'
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved vector graphic: {output_path_svg}")

    plt.close()


def create_detailed_bft_comparison():
    """
    Create detailed bar chart comparing all metrics across BFT levels.

    Shows:
    - Detection Rate (TP, FN)
    - False Positive Rate (FP, TN)
    - Byzantine/Honest counts
    """

    # Data
    bft_levels = ['35%\n(7/20)', '40%\n(8/20)', '45%\n(9/20)', '50%\n(10/20)']

    # Counts
    true_positives = [7, 8, 9, 10]
    false_positives = [0, 1, 1, 1]
    true_negatives = [13, 11, 10, 9]
    false_negatives = [0, 0, 0, 0]

    # Create figure with stacked bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        'Mode 1 Detailed Performance Breakdown Across BFT Levels',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    x = np.arange(len(bft_levels))
    width = 0.6

    # Color scheme
    tp_color = '#27ae60'  # Green
    tn_color = '#2ecc71'  # Light green
    fp_color = '#e74c3c'  # Red
    fn_color = '#c0392b'  # Dark red

    # Left subplot: Byzantine Detection (TP + FN)
    ax1.set_title('Byzantine Node Detection\n(True Positives + False Negatives)',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Number of Byzantine Nodes', fontsize=12)
    ax1.set_ylim(0, 13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(bft_levels, fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Stacked bars: TP (bottom) + FN (top)
    bars1_tp = ax1.bar(x, true_positives, width, label='True Positives (Detected)',
                      color=tp_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars1_fn = ax1.bar(x, false_negatives, width, bottom=true_positives,
                      label='False Negatives (Missed)',
                      color=fn_color, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (tp, fn) in enumerate(zip(true_positives, false_negatives)):
        # TP label
        ax1.text(i, tp/2, f'{tp}',
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        # Total label
        total = tp + fn
        ax1.text(i, total + 0.3, f'{tp}/{total}\n100%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add annotation
    ax1.text(1.5, 11, '✅ Perfect Detection\n100% across all BFT levels',
            ha='center', fontsize=11, fontweight='bold', color=tp_color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=tp_color, linewidth=2))

    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Right subplot: Honest Node Classification (TN + FP)
    ax2.set_title('Honest Node Classification\n(True Negatives + False Positives)',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Number of Honest Nodes', fontsize=12)
    ax2.set_ylim(0, 16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bft_levels, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Stacked bars: TN (bottom) + FP (top)
    bars2_tn = ax2.bar(x, true_negatives, width, label='True Negatives (Correct)',
                      color=tn_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2_fp = ax2.bar(x, false_positives, width, bottom=true_negatives,
                      label='False Positives (Incorrect)',
                      color=fp_color, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, (tn, fp) in enumerate(zip(true_negatives, false_positives)):
        total = tn + fp

        # TN label
        if tn > 0:
            ax2.text(i, tn/2, f'{tn}',
                    ha='center', va='center', fontsize=14, fontweight='bold', color='white')

        # FP label (if any)
        if fp > 0:
            ax2.text(i, tn + fp/2, f'{fp}',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        # Total label with FPR
        fpr = (fp / total * 100) if total > 0 else 0
        ax2.text(i, total + 0.3, f'{tn}/{total}\nFPR: {fpr:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add annotation
    ax2.text(0, 14.5, '✅ Low FPR\n0-10% across all levels',
            ha='center', fontsize=11, fontweight='bold', color=tn_color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=tn_color, linewidth=2))

    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Add bottom annotation
    fig.text(0.5, 0.02,
            'Total: 20 clients per test | Heterogeneous MNIST data | '
            'Adaptive threshold enabled',
            ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save figure
    output_path = '/tmp/mode1_detailed_breakdown_bft.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved detailed breakdown: {output_path}")

    output_path_svg = '/tmp/mode1_detailed_breakdown_bft.svg'
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved vector graphic: {output_path_svg}")

    plt.close()


def create_multi_seed_validation_plot():
    """
    Create plot showing multi-seed validation results at 45% BFT.

    Shows statistical robustness across different random seeds.
    """

    # Data from multi-seed validation (seeds: 42, 123, 456)
    seeds = ['42', '123', '456']
    detection_rates = [100.0, 100.0, 100.0]
    fprs = [9.1, 0.0, 0.0]

    # Mean and std
    mean_detection = 100.0
    std_detection = 0.0
    mean_fpr = 3.0
    std_fpr = 4.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Multi-Seed Validation at 45% BFT (3 Seeds)\n'
        'Statistical Robustness of Mode 1 Performance',
        fontsize=16,
        fontweight='bold',
        y=1.00
    )

    x = np.arange(len(seeds))
    width = 0.5

    # Color scheme
    detection_color = '#27ae60'
    fpr_color = '#e74c3c'

    # Left subplot: Detection Rate across seeds
    ax1.set_title('Byzantine Detection Rate\nAcross Random Seeds',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel('Detection Rate (%)', fontsize=12)
    ax1.set_xlabel('Random Seed', fontsize=12)
    ax1.set_ylim(0, 110)
    ax1.set_xticks(x)
    ax1.set_xticklabels(seeds, fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    bars1 = ax1.bar(x, detection_rates, width,
                   color=detection_color, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, rate in enumerate(detection_rates):
        ax1.text(i, rate + 2, f'{rate:.0f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add mean line
    ax1.axhline(y=mean_detection, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(2.5, mean_detection + 3, f'Mean: {mean_detection:.0f}% ± {std_detection:.0f}%',
            ha='right', fontsize=10, color='blue', fontweight='bold')

    # Add annotation
    ax1.text(1, 95, '✅ Perfect Consistency\n100% across all seeds',
            ha='center', fontsize=11, fontweight='bold', color=detection_color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=detection_color, linewidth=2))

    # Right subplot: FPR across seeds
    ax2.set_title('False Positive Rate\nAcross Random Seeds',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('False Positive Rate (%)', fontsize=12)
    ax2.set_xlabel('Random Seed', fontsize=12)
    ax2.set_ylim(0, 15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(seeds, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    bars2 = ax2.bar(x, fprs, width,
                   color=fpr_color, alpha=0.8,
                   edgecolor='black', linewidth=1.5)

    # Add value labels
    for i, fpr in enumerate(fprs):
        ax2.text(i, fpr + 0.3, f'{fpr:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add mean line
    ax2.axhline(y=mean_fpr, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(2.5, mean_fpr + 0.5, f'Mean: {mean_fpr:.1f}% ± {std_fpr:.1f}%',
            ha='right', fontsize=10, color='blue', fontweight='bold')

    # Add acceptable threshold line
    ax2.axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(2.5, 10.5, 'Acceptable (10%)',
            ha='right', fontsize=9, color='orange', fontweight='bold')

    # Add annotation
    ax2.text(1, 12, '✅ Low Variance\n3.0% ± 4.3%',
            ha='center', fontsize=11, fontweight='bold', color='green',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='green', linewidth=2))

    # Add bottom annotation
    fig.text(0.5, -0.02,
            'Configuration: 20 clients (11 honest, 9 Byzantine = 45% BFT) | '
            'Heterogeneous data | Adaptive threshold',
            ha='center', fontsize=10, style='italic')

    fig.text(0.5, -0.06,
            '✅ Statistical Robustness Validated: Consistent performance across independent random seeds',
            ha='center', fontsize=11, color='green', fontweight='bold')

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Save figure
    output_path = '/tmp/mode1_multi_seed_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved multi-seed validation: {output_path}")

    output_path_svg = '/tmp/mode1_multi_seed_validation.svg'
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved vector graphic: {output_path_svg}")

    plt.close()


if __name__ == "__main__":
    print("Creating Mode 1 performance visualization suite...")
    print()

    print("Figure 1: Performance Across BFT Levels")
    create_performance_across_bft()
    print()

    print("Figure 2: Detailed Breakdown Across BFT Levels")
    create_detailed_bft_comparison()
    print()

    print("Figure 3: Multi-Seed Validation (Statistical Robustness)")
    create_multi_seed_validation_plot()
    print()

    print("✅ All Mode 1 performance visualizations created successfully!")
    print()
    print("Output files:")
    print("  - /tmp/mode1_performance_across_bft.png/svg")
    print("  - /tmp/mode1_detailed_breakdown_bft.png/svg")
    print("  - /tmp/mode1_multi_seed_validation.png/svg")
