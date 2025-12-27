#!/usr/bin/env python3
"""
Generate figures for Paper 10: Comparative Framework Analysis
Target: Neuroscience & Biobehavioral Reviews

Run with: nix-shell -p python311 python311Packages.matplotlib python311Packages.numpy --run "python3 generate_paper10_figures.py"
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'iit': '#E74C3C',
    'gwt': '#3498DB',
    'hot': '#9B59B6',
    'rpt': '#2ECC71',
    'ast': '#F39C12',
    'pp': '#1ABC9C',
    'fcm': '#E91E63',
}


def fig1_theory_comparison_heatmap():
    """
    Figure 1: Heatmap of theory vs phenomena scores
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    theories = ['IIT', 'GNW', 'HOT', 'RPT', 'AST', 'PP', 'FCM']
    phenomena = ['Masking', 'Rivalry', 'Change\nBlind.', 'Blindsight',
                 'Anesthesia', 'Sleep', 'Dreaming', 'Psyched.',
                 'DOC', 'Split-\nbrain', 'PFC\nDamage', 'Graded\nC',
                 'Temporal\nBind.', 'Feature\nBind.', 'Report\nDissoc.']

    # Scores matrix (theories x phenomena)
    scores = np.array([
        [1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 2, 1, 1, 0],  # IIT
        [2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2],  # GNW
        [1, 1, 1, 2, 1, 1, 0, 0, 1, 1, 2, 0, 0, 0, 2],  # HOT
        [2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1],  # RPT
        [1, 2, 2, 1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1],  # AST
        [2, 2, 1, 1, 1, 1, 2, 2, 1, 0, 1, 1, 1, 1, 1],  # PP
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # FCM
    ])

    im = ax.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)

    ax.set_xticks(np.arange(len(phenomena)))
    ax.set_yticks(np.arange(len(theories)))
    ax.set_xticklabels(phenomena, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(theories, fontsize=11)

    # Add text annotations
    for i in range(len(theories)):
        for j in range(len(phenomena)):
            text = ax.text(j, i, scores[i, j], ha='center', va='center',
                          color='white' if scores[i, j] == 2 else 'black', fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(['Fails', 'Partial', 'Explains'])

    ax.set_title('Theory Performance Across Phenomena\n(0 = Fails, 1 = Partial, 2 = Explains)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/paper10_fig1_theory_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper10_fig1_theory_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 1 generated: Theory comparison heatmap")


def fig2_desiderata_scores():
    """
    Figure 2: Theory scores on theoretical desiderata
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    theories = ['IIT', 'GNW', 'HOT', 'RPT', 'AST', 'PP', 'FCM']
    colors = [COLORS['iit'], COLORS['gwt'], COLORS['hot'], COLORS['rpt'],
              COLORS['ast'], COLORS['pp'], COLORS['fcm']]

    # Panel A: Bar chart of total scores
    ax = axes[0]

    empirical_scores = [19, 23, 14, 21, 17, 18, 30]
    theoretical_scores = [13, 13, 8, 14, 11, 11, 15]

    x = np.arange(len(theories))
    width = 0.35

    bars1 = ax.bar(x - width/2, empirical_scores, width, label='Empirical (max 30)',
                   color=[c for c in colors], alpha=0.8, edgecolor='white', lw=2)
    bars2 = ax.bar(x + width/2, theoretical_scores, width, label='Theoretical (max 16)',
                   color=[c for c in colors], alpha=0.4, edgecolor='white', lw=2)

    ax.set_xticks(x)
    ax.set_xticklabels(theories, fontsize=11)
    ax.set_ylabel('Total Score', fontsize=11)
    ax.set_title('A. Theory Scores Summary', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 35)

    # Panel B: Radar plot of desiderata
    ax = axes[1]
    ax = fig.add_subplot(122, projection='polar')

    desiderata = ['Falsifiability', 'Parsimony', 'Mechanism', 'Clinical',
                  'Cross-species', 'Development', 'Neural Impl.', 'Scope']
    n_des = len(desiderata)
    angles = np.linspace(0, 2*np.pi, n_des, endpoint=False).tolist()
    angles += angles[:1]

    # Select subset of theories for clarity
    iit_vals = [2, 1, 2, 2, 2, 1, 2, 1] + [2]
    gwt_vals = [2, 2, 1, 2, 2, 1, 2, 1] + [2]
    fcm_vals = [2, 1, 2, 2, 2, 2, 2, 2] + [2]

    ax.plot(angles, iit_vals, 'o-', color=COLORS['iit'], lw=2, label='IIT')
    ax.plot(angles, gwt_vals, 's-', color=COLORS['gwt'], lw=2, label='GNW')
    ax.plot(angles, fcm_vals, '^-', color=COLORS['fcm'], lw=2, label='FCM')

    ax.fill(angles, fcm_vals, color=COLORS['fcm'], alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(desiderata, fontsize=9)
    ax.set_ylim(0, 2.2)
    ax.set_title('B. Desiderata Profiles', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/paper10_fig2_desiderata_scores.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper10_fig2_desiderata_scores.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 generated: Desiderata scores")


def fig3_theory_complementarity():
    """
    Figure 3: How theories map to FCM components
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # FCM components as central pentagon
    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, 6)
    component_names = ['Φ', 'B', 'W', 'A', 'R']
    comp_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']

    center = (5, 5)
    radius = 2

    for i, (name, color) in enumerate(zip(component_names, comp_colors)):
        x = center[0] + radius * np.cos(angles[i])
        y = center[1] + radius * np.sin(angles[i])

        circle = plt.Circle((x, y), 0.5, facecolor=color, edgecolor='white', lw=2, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, name, ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Theory labels around the outside
    theory_positions = {
        'IIT': (5, 8.5),
        'GWT': (8, 6),
        'HOT': (8, 3.5),
        'RPT': (2, 3.5),
        'PP': (2, 6),
    }

    theory_components = {
        'IIT': [0],       # Φ
        'GWT': [2],       # W
        'HOT': [3],       # A
        'RPT': [1, 4],    # B, R
        'PP': [2, 4],     # W, R
    }

    for theory, pos in theory_positions.items():
        ax.text(pos[0], pos[1], theory, ha='center', va='center',
               fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        # Draw lines to relevant components
        for comp_idx in theory_components[theory]:
            comp_x = center[0] + radius * np.cos(angles[comp_idx])
            comp_y = center[1] + radius * np.sin(angles[comp_idx])
            ax.annotate('', xy=(comp_x, comp_y), xytext=pos,
                       arrowprops=dict(arrowstyle='->', color=comp_colors[comp_idx],
                                      lw=2, alpha=0.6))

    ax.text(5, 1.5, 'FCM: Integration of all major theories\nEach theory contributes its strongest aspect',
            ha='center', fontsize=11, style='italic')

    ax.axis('off')
    ax.set_title('Theory Contributions to FCM Components', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('figures/paper10_fig3_theory_complementarity.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper10_fig3_theory_complementarity.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 generated: Theory complementarity")


def fig4_coverage_comparison():
    """
    Figure 4: Coverage comparison across domains
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Stacked bar by domain
    ax = axes[0]

    theories = ['IIT', 'GNW', 'HOT', 'RPT', 'AST', 'PP', 'FCM']
    domains = ['Perceptual', 'State', 'Clinical', 'Binding']
    domain_colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']

    scores = {
        'Perceptual': [5, 8, 5, 6, 6, 6, 8],
        'State': [6, 6, 2, 5, 4, 6, 8],
        'Clinical': [6, 5, 5, 4, 4, 3, 8],
        'Binding': [2, 4, 2, 6, 3, 3, 6],
    }

    x = np.arange(len(theories))
    width = 0.6

    bottom = np.zeros(len(theories))
    for domain, color in zip(domains, domain_colors):
        ax.bar(x, scores[domain], width, label=domain, bottom=bottom, color=color, alpha=0.8)
        bottom += scores[domain]

    ax.set_xticks(x)
    ax.set_xticklabels(theories, fontsize=11)
    ax.set_ylabel('Total Score', fontsize=11)
    ax.set_title('A. Coverage by Domain', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')

    # Panel B: Percentage coverage
    ax = axes[1]

    max_scores = {'Perceptual': 8, 'State': 8, 'Clinical': 8, 'Binding': 6}
    total_max = sum(max_scores.values())

    coverage = [(5+6+6+2)/total_max, (8+6+5+4)/total_max, (5+2+5+2)/total_max,
                (6+5+4+6)/total_max, (6+4+4+3)/total_max, (6+6+3+3)/total_max, 1.0]
    coverage = [c * 100 for c in coverage]

    colors = [COLORS['iit'], COLORS['gwt'], COLORS['hot'], COLORS['rpt'],
              COLORS['ast'], COLORS['pp'], COLORS['fcm']]

    bars = ax.barh(theories, coverage, color=colors, edgecolor='white', lw=2, alpha=0.8)

    ax.axvline(50, color='gray', ls='--', lw=1.5)
    ax.set_xlabel('Phenomena Coverage (%)', fontsize=11)
    ax.set_title('B. Overall Coverage', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 110)

    for bar, val in zip(bars, coverage):
        ax.text(val + 2, bar.get_y() + bar.get_height()/2,
               f'{val:.0f}%', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/paper10_fig4_coverage_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper10_fig4_coverage_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 generated: Coverage comparison")


def main():
    """Generate all figures for Paper 10."""
    print("\n" + "="*60)
    print("Generating Paper 10 Figures: Comparative Framework")
    print("="*60 + "\n")

    fig1_theory_comparison_heatmap()
    fig2_desiderata_scores()
    fig3_theory_complementarity()
    fig4_coverage_comparison()

    print("\n" + "="*60)
    print("✓ All 4 figures generated successfully!")
    print("  Output: papers/figures/paper10_fig[1-4]_*.png and .pdf")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
