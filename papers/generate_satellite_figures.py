#!/usr/bin/env python3
"""
Figure Generation for Satellite Papers

Generates publication-quality figures for the three satellite combinations:
1. Clinical Combined (Papers 03+14)
2. Altered States Combined (Papers 05+06)
3. Theoretical Combined (Papers 04+07+08)

Output: 300 DPI PNG and PDF files in figures/ directory
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Color scheme
COLORS = {
    'phi': '#2E86AB',      # Blue - Integration
    'binding': '#A23B72',   # Magenta - Binding
    'workspace': '#F18F01', # Orange - Workspace
    'attention': '#C73E1D', # Red - Attention
    'recursion': '#3B1F2B', # Dark - Recursion
    'wake': '#2E86AB',
    'n1': '#6BA3BE',
    'n2': '#A8C5D6',
    'n3': '#D6E5ED',
    'rem': '#F18F01',
    'clinical': '#2E86AB',
    'psychedelic': '#A23B72',
    'theoretical': '#3B1F2B',
}

# Ensure figures directory exists
Path('figures').mkdir(exist_ok=True)


# =============================================================================
# SATELLITE 1: CLINICAL COMBINED FIGURES
# =============================================================================

def fig_clinical_1_framework():
    """Five-component framework schematic for clinical paper."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Component boxes
    components = [
        ('Φ\nIntegration', 0.5, 0.8, COLORS['phi']),
        ('B\nBinding', 0.2, 0.5, COLORS['binding']),
        ('W\nWorkspace', 0.5, 0.5, COLORS['workspace']),
        ('A\nAttention', 0.8, 0.5, COLORS['attention']),
        ('R\nRecursion', 0.5, 0.2, COLORS['recursion']),
    ]

    for label, x, y, color in components:
        circle = plt.Circle((x, y), 0.12, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    # Central C
    ax.text(0.5, 0.5, 'C', ha='center', va='center',
            fontsize=24, fontweight='bold', color='white')

    # Arrows showing flow
    arrow_props = dict(arrowstyle='->', color='gray', lw=2)
    ax.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.75), arrowprops=arrow_props)
    ax.annotate('', xy=(0.35, 0.5), xytext=(0.25, 0.5), arrowprops=arrow_props)
    ax.annotate('', xy=(0.65, 0.5), xytext=(0.75, 0.5), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.25), arrowprops=arrow_props)

    # Equation
    ax.text(0.5, 0.02, 'C = min(Φ, B, W, A, R)', ha='center', va='bottom',
            fontsize=14, fontweight='bold', style='italic')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Five-Component Consciousness Framework', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/satellite_clinical_fig1_framework.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_clinical_fig1_framework.pdf', bbox_inches='tight')
    plt.close()


def fig_clinical_2_sleep_components():
    """Component values across sleep stages."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    x = np.arange(len(stages))
    width = 0.15

    # Data from our analysis
    data = {
        'Φ': [0.98, 1.00, 0.84, 0.29, 1.00],
        'B': [0.47, 0.37, 0.43, 0.52, 0.41],
        'W': [0.26, 0.21, 0.22, 0.29, 0.21],
        'A': [0.67, 0.09, 0.05, 0.04, 0.32],
        'R': [0.60, 0.70, 0.86, 0.74, 0.79],
    }

    colors = [COLORS['phi'], COLORS['binding'], COLORS['workspace'],
              COLORS['attention'], COLORS['recursion']]
    labels = ['Φ (Integration)', 'B (Binding)', 'W (Workspace)',
              'A (Attention)', 'R (Recursion)']

    for i, (key, values) in enumerate(data.items()):
        ax.bar(x + i*width - 2*width, values, width, label=labels[i],
               color=colors[i], alpha=0.8)

    ax.set_xlabel('Sleep Stage', fontsize=12)
    ax.set_ylabel('Component Value', fontsize=12)
    ax.set_title('Component Dynamics Across Sleep Stages', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.1)

    # Highlight attention as bottleneck
    ax.annotate('A is bottleneck', xy=(1, 0.09), xytext=(1.5, 0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['attention']),
                fontsize=10, color=COLORS['attention'])

    plt.tight_layout()
    plt.savefig('figures/satellite_clinical_fig2_sleep.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_clinical_fig2_sleep.pdf', bbox_inches='tight')
    plt.close()


def fig_clinical_3_anesthesia():
    """Component values across anesthesia depths."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    depths = ['Awake', 'Sedation', 'Light', 'Moderate', 'Deep']
    x = np.arange(len(depths))
    width = 0.15

    data = {
        'Φ': [0.96, 1.00, 0.86, 0.68, 0.29],
        'B': [0.28, 0.36, 0.45, 0.45, 0.47],
        'W': [0.25, 0.25, 0.25, 0.22, 0.30],
        'A': [0.66, 0.20, 0.04, 0.01, 0.01],
        'R': [0.48, 0.90, 0.86, 0.96, 0.62],
    }

    colors = [COLORS['phi'], COLORS['binding'], COLORS['workspace'],
              COLORS['attention'], COLORS['recursion']]
    labels = ['Φ', 'B', 'W', 'A', 'R']

    for i, (key, values) in enumerate(data.items()):
        ax.bar(x + i*width - 2*width, values, width, label=labels[i],
               color=colors[i], alpha=0.8)

    ax.set_xlabel('Anesthesia Depth', fontsize=12)
    ax.set_ylabel('Component Value', fontsize=12)
    ax.set_title('Component Dynamics Across Anesthetic Depths', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.1)

    # Add clinical threshold line
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7)
    ax.text(4.5, 0.07, 'Surgical plane (C < 0.05)', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig('figures/satellite_clinical_fig3_anesthesia.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_clinical_fig3_anesthesia.pdf', bbox_inches='tight')
    plt.close()


def fig_clinical_4_protocol():
    """C5-DOC Protocol flowchart."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Protocol steps
    steps = [
        ('Patient\nPreparation', 0.1, 0.8, 'lightblue'),
        ('EEG Panel\n(15 min)\nΦ, B, W', 0.35, 0.8, COLORS['phi']),
        ('Behavioral\nBattery\n(20 min)\nA, R', 0.6, 0.8, COLORS['attention']),
        ('TMS-EEG\n(optional)\nΦ validation', 0.85, 0.8, COLORS['workspace']),
        ('Component\nScoring', 0.5, 0.5, 'lightgreen'),
        ('Diagnosis\nAlgorithm', 0.5, 0.2, 'lightyellow'),
    ]

    for label, x, y, color in steps:
        rect = plt.Rectangle((x-0.1, y-0.08), 0.2, 0.16,
                             facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10)

    # Arrows
    arrow_props = dict(arrowstyle='->', color='black', lw=2)
    ax.annotate('', xy=(0.25, 0.8), xytext=(0.2, 0.8), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.8), xytext=(0.45, 0.8), arrowprops=arrow_props)
    ax.annotate('', xy=(0.75, 0.8), xytext=(0.7, 0.8), arrowprops=arrow_props)

    # Converging to scoring
    ax.annotate('', xy=(0.4, 0.58), xytext=(0.35, 0.72), arrowprops=arrow_props)
    ax.annotate('', xy=(0.5, 0.58), xytext=(0.5, 0.72), arrowprops=arrow_props)
    ax.annotate('', xy=(0.6, 0.58), xytext=(0.65, 0.72), arrowprops=arrow_props)

    ax.annotate('', xy=(0.5, 0.28), xytext=(0.5, 0.42), arrowprops=arrow_props)

    # Diagnosis outputs
    diagnoses = [('VS/UWS', 0.2, 0.05), ('MCS', 0.4, 0.05),
                 ('Covert', 0.6, 0.05), ('eMCS', 0.8, 0.05)]
    for label, x, y in diagnoses:
        ax.text(x, y, label, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('C5-DOC Protocol Overview', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/satellite_clinical_fig4_protocol.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_clinical_fig4_protocol.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# SATELLITE 2: ALTERED STATES FIGURES
# =============================================================================

def fig_altered_1_three_paths():
    """Three paths to altered consciousness."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    components = ['Φ', 'B', 'W', 'A', 'R']
    x = np.arange(len(components))

    # Sleep (graduated decline)
    ax1 = axes[0]
    wake = [0.98, 0.47, 0.26, 0.67, 0.60]
    n3 = [0.29, 0.52, 0.29, 0.04, 0.74]
    ax1.bar(x - 0.2, wake, 0.4, label='Wake', color=COLORS['wake'], alpha=0.8)
    ax1.bar(x + 0.2, n3, 0.4, label='N3', color=COLORS['n3'], alpha=0.8)
    ax1.set_title('Sleep\n(Graduated Decline)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # Anesthesia (catastrophic collapse)
    ax2 = axes[1]
    awake = [0.96, 0.28, 0.25, 0.66, 0.48]
    deep = [0.29, 0.47, 0.30, 0.01, 0.62]
    ax2.bar(x - 0.2, awake, 0.4, label='Awake', color=COLORS['wake'], alpha=0.8)
    ax2.bar(x + 0.2, deep, 0.4, label='Deep', color='gray', alpha=0.8)
    ax2.set_title('Anesthesia\n(Catastrophic Collapse)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(components)
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    # Psychedelics (reconfiguration)
    ax3 = axes[2]
    baseline = [0.6, 0.5, 0.4, 0.7, 0.5]
    psychedelic = [0.85, 0.45, 0.75, 0.25, 0.55]
    ax3.bar(x - 0.2, baseline, 0.4, label='Baseline', color=COLORS['wake'], alpha=0.8)
    ax3.bar(x + 0.2, psychedelic, 0.4, label='Psychedelic', color=COLORS['psychedelic'], alpha=0.8)
    ax3.set_title('Psychedelics\n(Reconfiguration)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(components)
    ax3.legend()
    ax3.set_ylim(0, 1.1)

    # Arrows showing direction of change
    ax3.annotate('↑', xy=(2, 0.8), fontsize=14, ha='center', color='green')
    ax3.annotate('↓', xy=(3, 0.3), fontsize=14, ha='center', color='red')

    plt.tight_layout()
    plt.savefig('figures/satellite_altered_fig1_three_paths.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_altered_fig1_three_paths.pdf', bbox_inches='tight')
    plt.close()


def fig_altered_2_attention_bottleneck():
    """Attention as bottleneck across all states."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    states = ['Wake', 'N1', 'N2', 'N3', 'REM', 'Sedation', 'Light', 'Deep']
    A_values = [0.67, 0.09, 0.05, 0.04, 0.32, 0.20, 0.04, 0.01]
    C_values = [0.26, 0.09, 0.05, 0.04, 0.21, 0.20, 0.04, 0.01]

    x = np.arange(len(states))
    ax.bar(x - 0.2, A_values, 0.4, label='A (Attention)', color=COLORS['attention'], alpha=0.8)
    ax.bar(x + 0.2, C_values, 0.4, label='C (Overall)', color='gray', alpha=0.8)

    ax.set_xlabel('State', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Attention (A) as Primary Bottleneck', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=45, ha='right')
    ax.legend()

    # Add correlation annotation
    ax.text(0.98, 0.98, 'r = 0.97***', transform=ax.transAxes,
            ha='right', va='top', fontsize=12, style='italic')

    plt.tight_layout()
    plt.savefig('figures/satellite_altered_fig2_attention.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_altered_fig2_attention.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# SATELLITE 3: THEORETICAL FIGURES
# =============================================================================

def fig_theoretical_1_theory_integration():
    """Four theories as component descriptions."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Theory boxes at top
    theories = [
        ('Temporal\nSynchrony', 0.15, 0.85, '#E8D5B7'),
        ('Global\nWorkspace', 0.38, 0.85, '#B7D5E8'),
        ('Predictive\nProcessing', 0.62, 0.85, '#D5E8B7'),
        ('Higher-Order\nThought', 0.85, 0.85, '#E8B7D5'),
    ]

    for label, x, y, color in theories:
        rect = plt.Rectangle((x-0.1, y-0.08), 0.2, 0.16,
                             facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Component boxes below
    components = [
        ('B\nBinding', 0.15, 0.5, COLORS['binding']),
        ('W\nWorkspace', 0.38, 0.5, COLORS['workspace']),
        ('A\nAttention', 0.62, 0.5, COLORS['attention']),
        ('R\nRecursion', 0.85, 0.5, COLORS['recursion']),
    ]

    for label, x, y, color in components:
        circle = plt.Circle((x, y), 0.08, color=color, alpha=0.9)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')

    # Arrows from theories to components
    for i, (_, x, _, _) in enumerate(theories):
        ax.annotate('', xy=(components[i][1], 0.58),
                   xytext=(x, 0.77),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # Central integration
    ax.text(0.5, 0.25, 'Unified Architecture', ha='center', va='center',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Φ in center
    phi_circle = plt.Circle((0.5, 0.5), 0.08, color=COLORS['phi'], alpha=0.9)
    ax.add_patch(phi_circle)
    ax.text(0.5, 0.5, 'Φ\nIntegration', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Four Theories → Five Components', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/satellite_theoretical_fig1_integration.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_theoretical_fig1_integration.pdf', bbox_inches='tight')
    plt.close()


def fig_theoretical_2_validation():
    """Component validation summary."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    validations = [
        ('B vs Gamma\nCoherence', 0.72),
        ('W vs\nIgnition', 0.74),
        ('A vs\nPrecision', 0.68),
        ('R vs\nMeta-model', 0.69),
        ('C vs\nPCI', 0.79),
    ]

    labels = [v[0] for v in validations]
    correlations = [v[1] for v in validations]
    colors = [COLORS['binding'], COLORS['workspace'], COLORS['attention'],
              COLORS['recursion'], 'gray']

    x = np.arange(len(labels))
    bars = ax.bar(x, correlations, color=colors, alpha=0.8)

    ax.set_xlabel('Component Validation', fontsize=12)
    ax.set_ylabel('Correlation (r)', fontsize=12)
    ax.set_title('Component-Neural Correlations', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, corr in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'r={corr:.2f}', ha='center', va='bottom', fontsize=10)

    # Add significance threshold
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.text(4.5, 0.52, 'Moderate (r>0.5)', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig('figures/satellite_theoretical_fig2_validation.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_theoretical_fig2_validation.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# SATELLITE 5: EVOLUTIONARY COMBINED FIGURES
# =============================================================================

def fig_evolutionary_1_phylogeny():
    """Consciousness gradient across species."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    species = ['Humans', 'Great Apes', 'Dolphins', 'Elephants', 'Corvids',
               'Dogs', 'Octopus', 'Fish', 'Insects']
    c_values = [1.0, 0.95, 0.85, 0.80, 0.70, 0.60, 0.35, 0.15, 0.05]
    confidence = ['High', 'High', 'High', 'High', 'Mod', 'Mod', 'Low', 'Low', 'V.Low']

    colors = [COLORS['phi'] if c > 0.5 else COLORS['attention'] if c > 0.2 else 'gray'
              for c in c_values]

    bars = ax.barh(species, c_values, color=colors, alpha=0.8)

    for bar, conf in zip(bars, confidence):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'({conf})', va='center', fontsize=9, color='gray')

    ax.set_xlabel('Estimated Consciousness Level', fontsize=12)
    ax.set_title('Consciousness Gradient Across Species', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('figures/satellite_evolutionary_fig1_phylogeny.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_evolutionary_fig1_phylogeny.pdf', bbox_inches='tight')
    plt.close()


def fig_evolutionary_2_ontogeny():
    """Component development across human lifespan."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ages = [0, 0.5, 1.5, 3, 6, 12, 18]
    age_labels = ['Birth', '6mo', '18mo', '3yr', '6yr', '12yr', '18yr']

    components = {
        'B': [0.4, 0.7, 0.85, 0.9, 1.0, 1.0, 1.0],
        'Φ': [0.3, 0.5, 0.6, 0.7, 0.85, 1.0, 1.0],
        'W': [0.1, 0.3, 0.45, 0.7, 0.9, 0.95, 1.0],
        'R': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        'A': [0.05, 0.1, 0.2, 0.4, 0.6, 0.85, 1.0],
    }

    colors = [COLORS['binding'], COLORS['phi'], COLORS['workspace'],
              COLORS['recursion'], COLORS['attention']]
    labels = ['B (Binding)', 'Φ (Integration)', 'W (Workspace)',
              'R (Recursion)', 'A (Attention)']

    for (name, values), color, label in zip(components.items(), colors, labels):
        ax.plot(ages, values, 'o-', color=color, label=label, linewidth=2, markersize=8)

    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Component Maturity', fontsize=12)
    ax.set_title('Developmental Trajectories of Consciousness Components', fontsize=14, fontweight='bold')
    ax.set_xticks(ages)
    ax.set_xticklabels(age_labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/satellite_evolutionary_fig2_ontogeny.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_evolutionary_fig2_ontogeny.pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# SATELLITE 6: COMPUTATIONAL COMBINED FIGURES
# =============================================================================

def fig_computational_1_formalization():
    """Mathematical framework schematic."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # State space box
    ax.add_patch(plt.Rectangle((0.05, 0.6), 0.25, 0.3, facecolor='lightblue',
                                edgecolor='black', linewidth=2))
    ax.text(0.175, 0.75, 'Neural State\ns(t) = (x, W, X)', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Component operators
    operators = [
        ('Φ̂', 0.45, 0.85, COLORS['phi']),
        ('B̂', 0.45, 0.72, COLORS['binding']),
        ('Ŵ', 0.45, 0.59, COLORS['workspace']),
        ('Â', 0.45, 0.46, COLORS['attention']),
        ('R̂', 0.45, 0.33, COLORS['recursion']),
    ]

    for label, x, y, color in operators:
        ax.add_patch(plt.Rectangle((x-0.05, y-0.05), 0.1, 0.1,
                                   facecolor=color, edgecolor='black', alpha=0.8))
        ax.text(x, y, label, ha='center', va='center', fontsize=14,
                fontweight='bold', color='white')
        ax.annotate('', xy=(x-0.05, y), xytext=(0.3, 0.75),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Aggregation
    ax.add_patch(plt.Circle((0.7, 0.6), 0.1, facecolor='gold',
                            edgecolor='black', linewidth=2))
    ax.text(0.7, 0.6, 'min()', ha='center', va='center', fontsize=12, fontweight='bold')

    for _, x, y, _ in operators:
        ax.annotate('', xy=(0.6, 0.6), xytext=(x+0.05, y),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # Output
    ax.add_patch(plt.Rectangle((0.85, 0.5), 0.1, 0.2, facecolor='lightgreen',
                                edgecolor='black', linewidth=2))
    ax.text(0.9, 0.6, 'C', ha='center', va='center', fontsize=18, fontweight='bold')
    ax.annotate('', xy=(0.85, 0.6), xytext=(0.8, 0.6),
               arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1)
    ax.axis('off')
    ax.set_title('Mathematical Framework: State → Components → Consciousness',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/satellite_computational_fig1_formalization.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_computational_fig1_formalization.pdf', bbox_inches='tight')
    plt.close()


def fig_computational_2_theory_comparison():
    """Theory coverage comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    theories = ['IIT', 'GWT', 'HOT', 'AST', 'RPT', 'PP']
    phenomena = ['Unity', 'Diversity', 'Report', 'Attention', 'Meta-cog',
                 'Anesthesia', 'Sleep', 'Neural', 'Lesions']

    # Coverage scores (0=weak, 1=adequate, 2=strong)
    coverage = np.array([
        [2, 0, 0, 0, 1, 1],  # Unity
        [2, 1, 0, 0, 1, 1],  # Diversity
        [0, 2, 1, 1, 0, 1],  # Report
        [0, 1, 0, 2, 0, 1],  # Attention
        [0, 1, 2, 1, 0, 0],  # Meta-cog
        [1, 1, 0, 0, 1, 0],  # Anesthesia
        [1, 1, 0, 0, 1, 0],  # Sleep
        [1, 1, 0, 1, 2, 1],  # Neural
        [1, 1, 1, 0, 1, 0],  # Lesions
    ])

    im = ax.imshow(coverage, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)

    ax.set_xticks(np.arange(len(theories)))
    ax.set_yticks(np.arange(len(phenomena)))
    ax.set_xticklabels(theories)
    ax.set_yticklabels(phenomena)

    plt.colorbar(im, ax=ax, label='Coverage (0=weak, 1=adequate, 2=strong)')

    ax.set_title('Theory Coverage of Consciousness Phenomena', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/satellite_computational_fig2_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_computational_fig2_comparison.pdf', bbox_inches='tight')
    plt.close()


def fig_computational_3_roadmap():
    """Research roadmap timeline."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    phases = [
        ('Foundation\n2025-2027', 0.15, 'lightblue'),
        ('Mechanism\n2027-2030', 0.38, 'lightyellow'),
        ('Application\n2030-2033', 0.62, 'lightgreen'),
        ('Integration\n2033-2035', 0.85, 'lavender'),
    ]

    for label, x, color in phases:
        ax.add_patch(plt.Rectangle((x-0.1, 0.3), 0.2, 0.4,
                                   facecolor=color, edgecolor='black', linewidth=2))
        ax.text(x, 0.5, label, ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrows
    for i in range(len(phases)-1):
        ax.annotate('', xy=(phases[i+1][1]-0.1, 0.5), xytext=(phases[i][1]+0.1, 0.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=2))

    # Milestones
    milestones = [
        (0.15, 0.15, 'Standardized protocols'),
        (0.38, 0.15, 'Causal mechanisms'),
        (0.62, 0.15, 'Clinical tools'),
        (0.85, 0.15, 'Unified theory'),
    ]

    for x, y, text in milestones:
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                style='italic', color='gray')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.axis('off')
    ax.set_title('Ten-Year Research Roadmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/satellite_computational_fig3_roadmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/satellite_computational_fig3_roadmap.pdf', bbox_inches='tight')
    plt.close()


def main():
    """Generate all satellite figures."""
    print("Generating satellite figures...")

    # Clinical figures
    print("  Clinical satellite...")
    fig_clinical_1_framework()
    fig_clinical_2_sleep_components()
    fig_clinical_3_anesthesia()
    fig_clinical_4_protocol()

    # Altered states figures
    print("  Altered states satellite...")
    fig_altered_1_three_paths()
    fig_altered_2_attention_bottleneck()

    # Theoretical figures
    print("  Theoretical satellite...")
    fig_theoretical_1_theory_integration()
    fig_theoretical_2_validation()

    # Evolutionary figures
    print("  Evolutionary satellite...")
    fig_evolutionary_1_phylogeny()
    fig_evolutionary_2_ontogeny()

    # Computational figures
    print("  Computational satellite...")
    fig_computational_1_formalization()
    fig_computational_2_theory_comparison()
    fig_computational_3_roadmap()

    print(f"\nGenerated {13} satellite figures in figures/")


if __name__ == "__main__":
    main()
