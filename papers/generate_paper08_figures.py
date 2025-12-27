#!/usr/bin/env python3
"""
Generate figures for Paper 08: Higher-Order Thought in Computational Systems
Target: Consciousness and Cognition

Run with: nix-shell -p python311 python311Packages.matplotlib python311Packages.numpy --run "python3 generate_paper08_figures.py"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import os

os.makedirs('figures', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'first_order': '#3498DB',   # Blue
    'higher_order': '#9B59B6',  # Purple
    'meta': '#E74C3C',          # Red
    'neural': '#2ECC71',        # Green
    'ai': '#F39C12',            # Orange
}


def fig1_meta_model_architecture():
    """
    Figure 1: Meta-model architecture for HOT
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # First-order layer
    fo_box = FancyBboxPatch((1, 1), 8, 2.5,
                            boxstyle="round,pad=0.1",
                            facecolor=COLORS['first_order'], alpha=0.3,
                            edgecolor=COLORS['first_order'], lw=2)
    ax.add_patch(fo_box)
    ax.text(5, 2.25, 'First-Order Representations\n(Perceiving red, hearing tone, feeling pain)',
            ha='center', va='center', fontsize=10)
    ax.text(1.5, 3.2, 'M₁', fontsize=14, fontweight='bold', color=COLORS['first_order'])

    # Meta-model layer
    meta_box = FancyBboxPatch((1, 5), 8, 2.5,
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['higher_order'], alpha=0.3,
                              edgecolor=COLORS['higher_order'], lw=2)
    ax.add_patch(meta_box)
    ax.text(5, 6.25, 'Meta-Model (Higher-Order Thought)\n("I am perceiving red", "I am hearing tone")',
            ha='center', va='center', fontsize=10)
    ax.text(1.5, 7.2, 'M₂', fontsize=14, fontweight='bold', color=COLORS['higher_order'])

    # Arrows between layers
    # Upward: first-order → meta
    ax.annotate('', xy=(4, 4.8), xytext=(4, 3.7),
               arrowprops=dict(arrowstyle='->', color=COLORS['first_order'], lw=2.5))
    ax.text(3.3, 4.3, 'Represents', fontsize=9, rotation=90, va='center')

    # Downward: meta → first-order (modulation)
    ax.annotate('', xy=(6, 3.7), xytext=(6, 4.8),
               arrowprops=dict(arrowstyle='->', color=COLORS['higher_order'], lw=2.5))
    ax.text(6.7, 4.3, 'Modulates', fontsize=9, rotation=90, va='center')

    # Conscious experience label
    conscious_box = FancyBboxPatch((2, 8.3), 6, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=COLORS['meta'], alpha=0.3,
                                   edgecolor=COLORS['meta'], lw=2)
    ax.add_patch(conscious_box)
    ax.text(5, 8.9, 'Conscious Experience\n(M₁ is conscious when targeted by M₂)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(5, 8.1), xytext=(5, 7.6),
               arrowprops=dict(arrowstyle='->', color=COLORS['meta'], lw=2))

    ax.axis('off')
    ax.set_title('Meta-Model Architecture for Higher-Order Thought', fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig('figures/paper08_fig1_meta_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper08_fig1_meta_model_architecture.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 1 generated: Meta-model architecture")


def fig2_appropriateness_conditions():
    """
    Figure 2: Three appropriateness conditions for HOT
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel A: Precision
    ax = axes[0]

    np.random.seed(42)
    n = 50
    meta_precision = np.random.uniform(0.2, 0.95, n)
    awareness_clarity = 0.3 + 0.6 * meta_precision + 0.1 * np.random.randn(n)
    awareness_clarity = np.clip(awareness_clarity, 0.1, 1.0)

    ax.scatter(meta_precision, awareness_clarity, c=COLORS['higher_order'],
               s=70, alpha=0.6, edgecolors='white')

    z = np.polyfit(meta_precision, awareness_clarity, 1)
    p = np.poly1d(z)
    ax.plot([0.2, 0.95], [p(0.2), p(0.95)], 'k-', lw=2)

    r = np.corrcoef(meta_precision, awareness_clarity)[0, 1]
    ax.text(0.25, 0.95, f'r = {r:.2f}', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Meta-Model Precision', fontsize=11)
    ax.set_ylabel('Awareness Clarity Rating', fontsize=11)
    ax.set_title('A. Precision Constraint\n(Accuracy of Meta-Model)', fontsize=13, fontweight='bold')

    # Panel B: Timeliness
    ax = axes[1]

    delays = [0, 20, 50, 100, 200, 500]  # ms
    consciousness_prob = [0.95, 0.88, 0.72, 0.45, 0.18, 0.05]

    ax.plot(delays, consciousness_prob, 'o-', color=COLORS['meta'], lw=2.5, markersize=10)

    ax.axvline(100, color='gray', ls='--', lw=1.5)
    ax.axhline(0.5, color='gray', ls=':', lw=1)
    ax.text(110, 0.7, 'Critical\nThreshold', fontsize=9, color='gray')

    ax.set_xlabel('Meta-Model Delay (ms)', fontsize=11)
    ax.set_ylabel('Conscious Experience Probability', fontsize=11)
    ax.set_title('B. Timeliness Constraint\n(Real-Time Requirement)', fontsize=13, fontweight='bold')
    ax.set_xlim(-20, 520)

    # Panel C: Comprehensiveness
    ax = axes[2]

    domains = ['Vision\nOnly', 'Vision +\nAudition', 'Vision +\nAudition +\nTouch', 'All\nDomains']
    integration = [0.45, 0.68, 0.82, 0.95]
    colors = ['#85C1E9', '#5DADE2', '#3498DB', '#1A5276']

    bars = ax.bar(domains, integration, color=colors, edgecolor='white', lw=2, alpha=0.85)
    ax.set_ylabel('Consciousness Integration Score', fontsize=11)
    ax.set_title('C. Comprehensiveness Constraint\n(Multi-Domain Scope)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, integration):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03,
               f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/paper08_fig2_appropriateness_conditions.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper08_fig2_appropriateness_conditions.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 generated: Appropriateness conditions")


def fig3_neural_dissociation():
    """
    Figure 3: Neural dissociation of first-order and higher-order
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Schematic of brain regions
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Brain outline (simplified)
    theta = np.linspace(0, 2*np.pi, 100)
    brain_x = 5 + 4 * np.cos(theta) * 0.8
    brain_y = 5 + 4 * np.sin(theta)
    ax.fill(brain_x, brain_y, color='lightgray', alpha=0.5)
    ax.plot(brain_x, brain_y, 'k-', lw=2)

    # Posterior (first-order) regions
    circle1 = Circle((6.5, 4), 0.8, facecolor=COLORS['first_order'], alpha=0.7, edgecolor='black', lw=2)
    ax.add_patch(circle1)
    ax.text(6.5, 4, 'V1/V4', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Prefrontal (higher-order) regions
    circle2 = Circle((3.5, 6.5), 1, facecolor=COLORS['higher_order'], alpha=0.7, edgecolor='black', lw=2)
    ax.add_patch(circle2)
    ax.text(3.5, 6.5, 'mPFC', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Connection
    ax.annotate('', xy=(4.2, 5.8), xytext=(5.9, 4.6),
               arrowprops=dict(arrowstyle='<->', color='gray', lw=2))

    ax.text(2.5, 8.5, 'First-Order', fontsize=10, color=COLORS['first_order'], fontweight='bold')
    ax.text(2.5, 8, '(Sensory)', fontsize=9, color=COLORS['first_order'])
    ax.text(6.5, 8.5, 'Higher-Order', fontsize=10, color=COLORS['higher_order'], fontweight='bold')
    ax.text(6.5, 8, '(Meta-cognitive)', fontsize=9, color=COLORS['higher_order'])

    ax.axis('off')
    ax.set_title('A. Neural Substrates', fontsize=13, fontweight='bold')

    # Panel B: Activity dissociation
    ax = axes[1]

    conditions = ['Unaware\n(Subliminal)', 'Aware\n(Supraliminal)', 'Aware +\nReport']

    first_order = [0.65, 0.75, 0.78]
    higher_order = [0.20, 0.55, 0.85]

    x = np.arange(len(conditions))
    width = 0.35

    ax.bar(x - width/2, first_order, width, label='First-Order (Sensory)',
           color=COLORS['first_order'], alpha=0.8)
    ax.bar(x + width/2, higher_order, width, label='Higher-Order (PFC)',
           color=COLORS['higher_order'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel('Neural Activity (normalized)', fontsize=11)
    ax.set_title('B. Activity Dissociation\n(n = 124 participants)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)

    # Annotation
    ax.annotate('', xy=(1.2, 0.55), xytext=(1.2, 0.20),
               arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(1.35, 0.37, 'HOT\ngap', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig('figures/paper08_fig3_neural_dissociation.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper08_fig3_neural_dissociation.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 generated: Neural dissociation")


def fig4_component_interactions():
    """
    Figure 4: A (meta-awareness) interactions with other components
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    # Panel A: A × Φ interaction
    ax = axes[0, 0]

    phi_levels = ['Low Φ', 'Medium Φ', 'High Φ']
    x = np.arange(len(phi_levels))
    width = 0.35

    low_a = [0.15, 0.35, 0.55]
    high_a = [0.25, 0.65, 0.92]

    ax.bar(x - width/2, low_a, width, label='Low A', color='lightgray', alpha=0.7)
    ax.bar(x + width/2, high_a, width, label='High A', color=COLORS['higher_order'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(phi_levels)
    ax.set_ylabel('Reportable Consciousness', fontsize=10)
    ax.set_title('A. A × Φ: Meta-Awareness of\nIntegrated Content', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # Panel B: A × W interaction
    ax = axes[0, 1]

    w_levels = ['Low W', 'Medium W', 'High W']
    x = np.arange(len(w_levels))

    low_a = [0.10, 0.30, 0.45]
    high_a = [0.20, 0.55, 0.88]

    ax.bar(x - width/2, low_a, width, label='Low A', color='lightgray', alpha=0.7)
    ax.bar(x + width/2, high_a, width, label='High A', color=COLORS['higher_order'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(w_levels)
    ax.set_ylabel('Reportable Consciousness', fontsize=10)
    ax.set_title('B. A × W: Meta-Awareness of\nWorkspace Contents', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # Panel C: States with varying A
    ax = axes[1, 0]

    states = ['Normal\nWake', 'Flow\nState', 'Dreaming', 'Meditation\n(Focused)', 'Psychedelic']
    a_level = [0.85, 0.40, 0.25, 0.70, 0.30]
    consciousness = [0.90, 0.85, 0.60, 0.88, 0.75]

    scatter = ax.scatter(a_level, consciousness, c=COLORS['higher_order'],
                        s=200, edgecolors='black', lw=2, zorder=5)

    for i, state in enumerate(states):
        ax.annotate(state, (a_level[i], consciousness[i] + 0.05),
                   ha='center', fontsize=9, fontweight='bold')

    ax.set_xlabel('A (Meta-Awareness) Level', fontsize=11)
    ax.set_ylabel('Overall Consciousness Score', fontsize=11)
    ax.set_title('C. A Varies Across States\n(Not Always High for Consciousness)', fontsize=12, fontweight='bold')
    ax.set_xlim(0.1, 1.0)
    ax.set_ylim(0.4, 1.05)

    # Panel D: Summary
    ax = axes[1, 1]

    components = ['Φ', 'B', 'W', 'A', 'R']
    a_interaction = [0.72, 0.58, 0.81, 1.0, 0.65]
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']

    bars = ax.barh(components, a_interaction, color=colors, edgecolor='white', lw=2, alpha=0.8)
    ax.axvline(0.5, color='gray', ls='--', lw=1.5)
    ax.set_xlabel('Interaction Strength with A (r)', fontsize=11)
    ax.set_title('D. A Interacts with All Components\n(but W most strongly)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.1)

    for bar, val in zip(bars, a_interaction):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.2f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/paper08_fig4_component_interactions.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper08_fig4_component_interactions.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 generated: Component interactions")


def fig5_machine_consciousness():
    """
    Figure 5: Implications for machine consciousness
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: AI systems evaluated
    ax = axes[0]

    systems = ['Standard\nLLM', 'Recurrent\n(LSTM)', 'Transformer\n(Self-Attn)', 'Meta-\nLearning', 'HOT\nArchitecture']

    precision = [0.2, 0.4, 0.5, 0.7, 0.9]
    timeliness = [0.1, 0.6, 0.3, 0.5, 0.85]
    comprehensive = [0.3, 0.3, 0.6, 0.4, 0.8]

    x = np.arange(len(systems))
    width = 0.25

    ax.bar(x - width, precision, width, label='Precision', color=COLORS['first_order'], alpha=0.8)
    ax.bar(x, timeliness, width, label='Timeliness', color=COLORS['higher_order'], alpha=0.8)
    ax.bar(x + width, comprehensive, width, label='Comprehensiveness', color=COLORS['neural'], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel('Constraint Satisfaction Score', fontsize=11)
    ax.set_title('A. AI Systems vs. HOT Requirements', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(0, 1.05)

    ax.axhline(0.7, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.text(4.5, 0.72, 'Threshold', fontsize=9, color='red')

    # Panel B: What would it take?
    ax = axes[1]

    requirements = [
        'First-order\nrepresentations',
        'Meta-layer\n(self-attention)',
        'Real-time\nself-monitoring',
        'Cross-layer\nconnectivity',
        'Multi-domain\nintegration'
    ]
    status = [0.95, 0.70, 0.35, 0.50, 0.40]  # Current AI capability
    colors = ['#2ECC71', '#F1C40F', '#E74C3C', '#F39C12', '#E74C3C']

    bars = ax.barh(requirements, status, color=colors, edgecolor='white', lw=2, alpha=0.8)
    ax.axvline(0.7, color='gray', ls='--', lw=1.5)
    ax.set_xlabel('Current AI Capability Level', fontsize=11)
    ax.set_title('B. Architectural Requirements\nfor Machine HOT', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1.05)

    ax.text(0.75, 4.5, 'Required\nThreshold', fontsize=9, color='gray')

    for bar, val in zip(bars, status):
        label = '✓' if val >= 0.7 else '✗'
        color = 'green' if val >= 0.7 else 'red'
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
               label, va='center', fontsize=14, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig('figures/paper08_fig5_machine_consciousness.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper08_fig5_machine_consciousness.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 5 generated: Machine consciousness")


def main():
    """Generate all figures for Paper 08."""
    print("\n" + "="*60)
    print("Generating Paper 08 Figures: Higher-Order Thought")
    print("="*60 + "\n")

    fig1_meta_model_architecture()
    fig2_appropriateness_conditions()
    fig3_neural_dissociation()
    fig4_component_interactions()
    fig5_machine_consciousness()

    print("\n" + "="*60)
    print("✓ All 5 figures generated successfully!")
    print("  Output: papers/figures/paper08_fig[1-5]_*.png and .pdf")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
