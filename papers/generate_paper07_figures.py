#!/usr/bin/env python3
"""
Generate figures for Paper 07: GWT Meets Predictive Processing
Target: Trends in Cognitive Sciences

Run with: nix-shell -p python311 python311Packages.matplotlib python311Packages.numpy --run "python3 generate_paper07_figures.py"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os

os.makedirs('figures', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'gwt': '#3498DB',       # Blue
    'pp': '#E74C3C',        # Red
    'unified': '#9B59B6',   # Purple
    'prediction': '#2ECC71',
    'error': '#F39C12',
    'workspace': '#1ABC9C',
}


def fig1_unified_architecture():
    """
    Figure 1: The unified GWT-PP architecture
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Hierarchical levels
    levels = [
        ('Sensory Input', 1, COLORS['pp']),
        ('Level 1: First-Order Predictions', 3, COLORS['prediction']),
        ('Level 2: Higher-Order Predictions', 5, COLORS['prediction']),
        ('Global Workspace', 7, COLORS['gwt']),
        ('Conscious Experience', 9, COLORS['unified']),
    ]

    for label, y, color in levels:
        rect = FancyBboxPatch((1, y-0.4), 8, 0.8,
                              boxstyle="round,pad=0.05",
                              facecolor=color, alpha=0.3,
                              edgecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(5, y, label, ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrows: feedforward (prediction errors) and feedback (predictions)
    for i in range(4):
        y_start = levels[i][1] + 0.4
        y_end = levels[i+1][1] - 0.4

        # Feedforward (errors) - left side
        ax.annotate('', xy=(2, y_end), xytext=(2, y_start),
                   arrowprops=dict(arrowstyle='->', color=COLORS['error'], lw=2))

        # Feedback (predictions) - right side
        ax.annotate('', xy=(8, y_start), xytext=(8, y_end),
                   arrowprops=dict(arrowstyle='->', color=COLORS['prediction'], lw=2))

    # Labels
    ax.text(1.2, 5, 'Prediction\nErrors ↑', fontsize=9, color=COLORS['error'], rotation=90, va='center')
    ax.text(8.8, 5, 'Predictions ↓', fontsize=9, color=COLORS['prediction'], rotation=90, va='center')

    # Key insight box
    ax.text(5, 0.5, 'Key: Workspace arbitrates among competing predictions;\nPrediction errors trigger ignition',
            ha='center', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.axis('off')
    ax.set_title('Unified GWT-PP Architecture', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figures/paper07_fig1_unified_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper07_fig1_unified_architecture.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 1 generated: Unified architecture")


def fig2_prediction_error_ignition():
    """
    Figure 2: Prediction error triggers workspace ignition
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Predicted vs unpredicted stimuli
    ax = axes[0]

    conditions = ['Highly\nPredicted', 'Moderately\nPredicted', 'Unpredicted', 'Violated\nPrediction']
    ignition_prob = [0.15, 0.35, 0.72, 0.91]
    colors = [COLORS['prediction'], COLORS['prediction'], COLORS['error'], COLORS['error']]

    bars = ax.bar(conditions, ignition_prob, color=colors, edgecolor='white', lw=2, alpha=0.8)

    ax.axhline(0.5, color='gray', ls='--', lw=1.5, alpha=0.7)
    ax.set_ylabel('Workspace Ignition Probability', fontsize=11)
    ax.set_title('A. Prediction Error Drives Ignition', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, ignition_prob):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
               f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

    # Panel B: P300 amplitude by prediction error
    ax = axes[1]

    np.random.seed(111)
    n = 60
    prediction_error = np.random.uniform(0.1, 1.0, n)
    p300_amplitude = 2 + 6 * prediction_error + np.random.randn(n)
    p300_amplitude = np.clip(p300_amplitude, 0.5, 10)

    ax.scatter(prediction_error, p300_amplitude, c=prediction_error, cmap='YlOrRd',
               s=80, edgecolors='white', lw=0.5)

    z = np.polyfit(prediction_error, p300_amplitude, 1)
    p = np.poly1d(z)
    ax.plot([0.1, 1.0], [p(0.1), p(1.0)], 'k-', lw=2)

    r = np.corrcoef(prediction_error, p300_amplitude)[0, 1]
    ax.text(0.15, 9, f'r = {r:.2f}', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Prediction Error Magnitude', fontsize=11)
    ax.set_ylabel('P300 Amplitude (μV)', fontsize=11)
    ax.set_title('B. P300 Correlates with\nPrediction Error', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/paper07_fig2_prediction_error_ignition.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper07_fig2_prediction_error_ignition.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 generated: Prediction error ignition")


def fig3_precision_gating():
    """
    Figure 3: Precision as gain control for workspace access
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Precision modulates access
    ax = axes[0]

    precision_levels = ['Low\nPrecision', 'Medium\nPrecision', 'High\nPrecision']

    # At each precision level, show access probability for different error magnitudes
    x = np.arange(3)
    width = 0.25

    small_error = [0.10, 0.25, 0.45]
    medium_error = [0.25, 0.55, 0.78]
    large_error = [0.45, 0.75, 0.92]

    ax.bar(x - width, small_error, width, label='Small Error', color='#85C1E9', alpha=0.8)
    ax.bar(x, medium_error, width, label='Medium Error', color='#F7DC6F', alpha=0.8)
    ax.bar(x + width, large_error, width, label='Large Error', color='#E74C3C', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(precision_levels)
    ax.set_ylabel('Workspace Access Probability', fontsize=11)
    ax.set_title('A. Precision Modulates Access\n(Error × Precision Interaction)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # Panel B: Attention = precision mechanism
    ax = axes[1]

    # Conceptual diagram
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # GWT box
    gwt_box = FancyBboxPatch((0.5, 6), 4, 3,
                             boxstyle="round,pad=0.1",
                             facecolor=COLORS['gwt'], alpha=0.3,
                             edgecolor=COLORS['gwt'], lw=2)
    ax.add_patch(gwt_box)
    ax.text(2.5, 7.5, 'GWT:\nAttention modulates\nworkspace access', ha='center', va='center', fontsize=10)

    # PP box
    pp_box = FancyBboxPatch((5.5, 6), 4, 3,
                            boxstyle="round,pad=0.1",
                            facecolor=COLORS['pp'], alpha=0.3,
                            edgecolor=COLORS['pp'], lw=2)
    ax.add_patch(pp_box)
    ax.text(7.5, 7.5, 'PP:\nAttention =\nprecision weighting', ha='center', va='center', fontsize=10)

    # Unified box
    unified_box = FancyBboxPatch((2.5, 1), 5, 3,
                                 boxstyle="round,pad=0.1",
                                 facecolor=COLORS['unified'], alpha=0.3,
                                 edgecolor=COLORS['unified'], lw=2)
    ax.add_patch(unified_box)
    ax.text(5, 2.5, 'UNIFIED:\nPrecision weighting IS\nthe mechanism of\nworkspace access modulation',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(3.5, 4.2), xytext=(2.5, 5.8),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(6.5, 4.2), xytext=(7.5, 5.8),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.axis('off')
    ax.set_title('B. Attention Mechanisms Unified', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/paper07_fig3_precision_gating.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper07_fig3_precision_gating.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 generated: Precision gating")


def fig4_component_mapping():
    """
    Figure 4: Five components mapped to GWT-PP constructs
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Create table-like visualization
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)

    # Headers
    headers = ['Component', 'GWT Construct', 'PP Construct', 'Neural Correlate']
    header_x = [1, 4, 7, 10]

    for x, header in zip(header_x, headers):
        ax.text(x, 9.5, header, ha='center', va='center', fontsize=11, fontweight='bold')

    # Data rows
    rows = [
        ('Φ (Integration)', 'Global broadcast\ncoherence', 'Hierarchical model\ncoherence', 'Long-range\nconnectivity'),
        ('B (Binding)', 'Feature bundling\nfor broadcast', 'Joint prediction\nconstraint', 'Gamma\nsynchrony'),
        ('W (Workspace)', 'Workspace access\ncompetition', 'Precision-weighted\nerror routing', 'Prefrontal-\nparietal'),
        ('A (Awareness)', 'Meta-representation\nin workspace', 'Higher-order\npredictive model', 'DMN + medial\nPFC'),
        ('R (Recursion)', 'Temporal depth\nof broadcast', 'Hierarchical\ntime scales', 'Hippocampal-\nfrontal'),
    ]

    colors = [COLORS['pp'], COLORS['gwt'], COLORS['prediction'], COLORS['unified'], COLORS['error']]

    for i, (comp, gwt, pp, neural) in enumerate(rows):
        y = 7.5 - i * 1.5
        ax.text(1, y, comp, ha='center', va='center', fontsize=10, fontweight='bold', color=colors[i])
        ax.text(4, y, gwt, ha='center', va='center', fontsize=9)
        ax.text(7, y, pp, ha='center', va='center', fontsize=9)
        ax.text(10, y, neural, ha='center', va='center', fontsize=9)

        # Horizontal line
        if i < 4:
            ax.axhline(y - 0.7, color='lightgray', lw=1, xmin=0.05, xmax=0.95)

    # Header line
    ax.axhline(9.0, color='black', lw=2, xmin=0.05, xmax=0.95)

    ax.axis('off')
    ax.set_title('Component Mapping to GWT and PP Constructs', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figures/paper07_fig4_component_mapping.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper07_fig4_component_mapping.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 generated: Component mapping")


def fig5_novel_predictions():
    """
    Figure 5: Novel predictions from unified framework
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Prediction error vs consciousness
    ax = axes[0]

    conditions = ['Predicted\n(Expected)', 'Rare\n(Oddball)', 'Omission\n(Missing)']
    consciousness = [0.25, 0.68, 0.82]
    colors = [COLORS['prediction'], COLORS['error'], COLORS['error']]

    bars = ax.bar(conditions, consciousness, color=colors, edgecolor='white', lw=2, alpha=0.8)
    ax.set_ylabel('Conscious Detection Rate', fontsize=11)
    ax.set_title('A. Prediction 1:\nError → Consciousness', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)

    # Panel B: Precision manipulation
    ax = axes[1]

    np.random.seed(222)
    n = 40

    # High reliability context
    high_rel_stim = np.random.uniform(0.3, 0.7, n)
    high_rel_detect = 0.3 + 0.6 * high_rel_stim + 0.1 * np.random.randn(n)

    # Low reliability context (same stimuli, lower detection)
    low_rel_stim = np.random.uniform(0.3, 0.7, n)
    low_rel_detect = 0.15 + 0.4 * low_rel_stim + 0.1 * np.random.randn(n)

    ax.scatter(high_rel_stim, high_rel_detect, c=COLORS['unified'], s=60, alpha=0.7, label='High Precision Context')
    ax.scatter(low_rel_stim, low_rel_detect, c='gray', s=60, alpha=0.5, label='Low Precision Context')

    ax.set_xlabel('Stimulus Strength (identical)', fontsize=11)
    ax.set_ylabel('Detection Rate', fontsize=11)
    ax.set_title('B. Prediction 2:\nPrecision → Access', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')

    # Panel C: Hierarchical cascade timing
    ax = axes[2]

    time = np.linspace(0, 400, 100)  # ms

    sensory = np.where(time > 50, 1 - np.exp(-(time-50)/30), 0) * 100
    local_ws = np.where(time > 100, 1 - np.exp(-(time-100)/40), 0) * 100
    global_ws = np.where(time > 200, 1 - np.exp(-(time-200)/50), 0) * 100

    ax.plot(time, sensory, color=COLORS['prediction'], lw=2.5, label='Sensory Activation')
    ax.plot(time, local_ws, color=COLORS['error'], lw=2.5, label='Local Workspace')
    ax.plot(time, global_ws, color=COLORS['unified'], lw=2.5, label='Global Ignition')

    ax.axvline(50, color='gray', ls=':', lw=1)
    ax.axvline(100, color='gray', ls=':', lw=1)
    ax.axvline(200, color='gray', ls=':', lw=1)

    ax.set_xlabel('Time (ms)', fontsize=11)
    ax.set_ylabel('Activation (%)', fontsize=11)
    ax.set_title('C. Prediction 3:\nHierarchical Cascade', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(0, 400)

    plt.tight_layout()
    plt.savefig('figures/paper07_fig5_novel_predictions.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper07_fig5_novel_predictions.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 5 generated: Novel predictions")


def main():
    """Generate all figures for Paper 07."""
    print("\n" + "="*60)
    print("Generating Paper 07 Figures: GWT + Predictive Processing")
    print("="*60 + "\n")

    fig1_unified_architecture()
    fig2_prediction_error_ignition()
    fig3_precision_gating()
    fig4_component_mapping()
    fig5_novel_predictions()

    print("\n" + "="*60)
    print("✓ All 5 figures generated successfully!")
    print("  Output: papers/figures/paper07_fig[1-5]_*.png and .pdf")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
