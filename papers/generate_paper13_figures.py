#!/usr/bin/env python3
"""
Generate figures for Paper 13: Cross-Species Consciousness
Target: Philosophical Transactions of the Royal Society B

Run with: nix-shell -p python311 python311Packages.matplotlib python311Packages.numpy --run "python3 generate_paper13_figures.py"
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'phi': '#E74C3C',
    'binding': '#3498DB',
    'workspace': '#2ECC71',
    'awareness': '#9B59B6',
    'recursion': '#F39C12',
    'mammal': '#E74C3C',
    'bird': '#3498DB',
    'cephalopod': '#2ECC71',
    'fish': '#F39C12',
    'insect': '#9B59B6',
}


def fig1_consciousness_gradient():
    """
    Figure 1: The consciousness gradient across species
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    species = ['Humans', 'Great Apes', 'Dolphins', 'Elephants', 'Corvids',
               'Dogs', 'Octopus', 'Rodents', 'Fish', 'Insects']

    # Component levels (0-1 scale)
    data = {
        'Humans':     [1.0, 1.0, 1.0, 1.0, 1.0],
        'Great Apes': [0.95, 0.95, 0.95, 0.90, 0.95],
        'Dolphins':   [0.90, 0.75, 0.90, 0.85, 0.90],
        'Elephants':  [0.90, 0.70, 0.75, 0.85, 0.85],
        'Corvids':    [0.75, 0.70, 0.75, 0.65, 0.85],
        'Dogs':       [0.75, 0.85, 0.70, 0.40, 0.70],
        'Octopus':    [0.65, 0.45, 0.50, 0.30, 0.65],
        'Rodents':    [0.65, 0.80, 0.50, 0.25, 0.50],
        'Fish':       [0.40, 0.50, 0.25, 0.10, 0.30],
        'Insects':    [0.25, 0.25, 0.25, 0.05, 0.10],
    }

    components = ['Φ', 'B', 'W', 'A', 'R']
    comp_colors = [COLORS['phi'], COLORS['binding'], COLORS['workspace'],
                   COLORS['awareness'], COLORS['recursion']]

    x = np.arange(len(species))
    width = 0.15

    for i, (comp, color) in enumerate(zip(components, comp_colors)):
        values = [data[s][i] for s in species]
        ax.bar(x + i*width - 2*width, values, width, label=comp, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Component Level (0-1)', fontsize=12)
    ax.set_title('Consciousness Gradient Across Species', fontsize=14, fontweight='bold')
    ax.legend(title='Component', fontsize=9, title_fontsize=10)
    ax.set_ylim(0, 1.15)

    # Add taxonomic groupings
    ax.axvspan(-0.5, 0.5, alpha=0.05, color='red')
    ax.axvspan(0.5, 3.5, alpha=0.05, color='blue')
    ax.axvspan(3.5, 5.5, alpha=0.05, color='green')

    plt.tight_layout()
    plt.savefig('figures/paper13_fig1_consciousness_gradient.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper13_fig1_consciousness_gradient.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 1 generated: Consciousness gradient")


def fig2_phylogenetic_tree():
    """
    Figure 2: Component emergence across phylogeny
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Simplified phylogenetic tree with component annotations
    # Main trunk
    ax.plot([1, 1], [1, 8], 'k-', lw=3)

    # Major branches
    branches = [
        (1, 7, 4, 8.5, 'Mammals', COLORS['mammal']),
        (1, 5.5, 4, 6.5, 'Birds', COLORS['bird']),
        (1, 4, 4, 4.5, 'Reptiles', 'gray'),
        (1, 2.5, 4, 3, 'Cephalopods', COLORS['cephalopod']),
        (1, 1.5, 4, 1.5, 'Insects', COLORS['insect']),
    ]

    for x1, y1, x2, y2, label, color in branches:
        ax.plot([x1, x2], [y1, y2], '-', color=color, lw=2.5)
        ax.text(x2 + 0.2, y2, label, fontsize=11, color=color, fontweight='bold', va='center')

    # Sub-branches for mammals
    ax.plot([4, 6], [8.5, 9.2], '-', color=COLORS['mammal'], lw=2)
    ax.plot([4, 6], [8.5, 8.5], '-', color=COLORS['mammal'], lw=2)
    ax.plot([4, 6], [8.5, 7.8], '-', color=COLORS['mammal'], lw=2)

    ax.text(6.2, 9.2, 'Primates (All 5)', fontsize=9, va='center')
    ax.text(6.2, 8.5, 'Cetaceans (All 5)', fontsize=9, va='center')
    ax.text(6.2, 7.8, 'Carnivores (4)', fontsize=9, va='center')

    # Component markers
    ax.text(8, 9.2, 'Φ B W A R', fontsize=8, family='monospace', color='green')
    ax.text(8, 8.5, 'Φ B W A R', fontsize=8, family='monospace', color='green')
    ax.text(8, 7.8, 'Φ B W - R', fontsize=8, family='monospace', color='orange')

    # Sub-branches for birds
    ax.plot([4, 6], [6.5, 7], '-', color=COLORS['bird'], lw=2)
    ax.plot([4, 6], [6.5, 6], '-', color=COLORS['bird'], lw=2)

    ax.text(6.2, 7, 'Corvids (4-5)', fontsize=9, va='center')
    ax.text(6.2, 6, 'Other birds (2-3)', fontsize=9, va='center')

    ax.text(8, 7, 'Φ B W a R', fontsize=8, family='monospace', color='orange')
    ax.text(8, 6, 'Φ B w - -', fontsize=8, family='monospace', color='red')

    # Legend
    ax.text(0.5, 0.5, 'Capital = strong, lowercase = moderate, - = absent/minimal',
            fontsize=9, style='italic')

    ax.axis('off')
    ax.set_title('Component Emergence Across Phylogeny', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/paper13_fig2_phylogenetic_tree.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper13_fig2_phylogenetic_tree.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 generated: Phylogenetic tree")


def fig3_neural_substrates():
    """
    Figure 3: Neural substrates for components across taxa
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    components = ['Φ (Integration)', 'B (Binding)', 'W (Workspace)',
                  'A (Awareness)', 'R (Recursion)']
    comp_colors = [COLORS['phi'], COLORS['binding'], COLORS['workspace'],
                   COLORS['awareness'], COLORS['recursion']]

    taxa = ['Mammals', 'Birds', 'Cephalopods', 'Fish', 'Insects']
    taxa_colors = [COLORS['mammal'], COLORS['bird'], COLORS['cephalopod'],
                   COLORS['fish'], COLORS['insect']]

    # Data: presence/strength of neural substrate (0-1)
    substrates = {
        'Φ (Integration)': [0.95, 0.80, 0.60, 0.40, 0.25],
        'B (Binding)': [0.90, 0.75, 0.45, 0.50, 0.25],
        'W (Workspace)': [0.85, 0.70, 0.50, 0.25, 0.25],
        'A (Awareness)': [0.80, 0.55, 0.25, 0.10, 0.05],
        'R (Recursion)': [0.85, 0.75, 0.55, 0.30, 0.10],
    }

    neural_structures = {
        'Φ (Integration)': ['Corpus callosum', 'Pallial connections', 'Vertical lobe', 'Telencephalon', 'Central complex'],
        'B (Binding)': ['GABAergic interneurons', 'Pallial interneurons', 'Unknown', 'Tectal circuits', 'Mushroom bodies'],
        'W (Workspace)': ['PFC', 'NCL', 'Vertical lobe?', 'None clear', 'Mushroom bodies?'],
        'A (Awareness)': ['mPFC/DMN', 'Nidopallium?', 'Unknown', 'None', 'None'],
        'R (Recursion)': ['Hippocampus', 'Hippocampus homolog', 'Vertical lobe', 'Pallium', 'Mushroom bodies'],
    }

    for idx, (comp, color) in enumerate(zip(components, comp_colors)):
        ax = axes[idx]
        values = substrates[comp]
        structures = neural_structures[comp]

        bars = ax.barh(taxa, values, color=taxa_colors, edgecolor='white', lw=2, alpha=0.8)

        ax.set_xlim(0, 1.1)
        ax.set_title(comp, fontsize=12, fontweight='bold', color=color)

        # Add structure labels
        for i, (bar, struct) in enumerate(zip(bars, structures)):
            ax.text(values[i] + 0.02, bar.get_y() + bar.get_height()/2,
                   struct, va='center', fontsize=8, style='italic')

    # Remove empty subplot
    axes[5].axis('off')

    plt.suptitle('Neural Substrates for Components Across Taxa', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/paper13_fig3_neural_substrates.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper13_fig3_neural_substrates.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 generated: Neural substrates")


def fig4_consciousness_space():
    """
    Figure 4: Species in consciousness space (2D projection)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 2D projection: Integration (Φ) vs Meta-awareness (A)
    species_data = {
        'Humans': (1.0, 1.0),
        'Great Apes': (0.95, 0.90),
        'Dolphins': (0.90, 0.85),
        'Elephants': (0.90, 0.85),
        'Corvids': (0.75, 0.65),
        'Dogs': (0.75, 0.40),
        'Octopus': (0.65, 0.30),
        'Parrots': (0.70, 0.50),
        'Rodents': (0.65, 0.25),
        'Fish': (0.40, 0.10),
        'Insects': (0.25, 0.05),
    }

    colors_map = {
        'Humans': 'red', 'Great Apes': 'red', 'Dolphins': 'blue',
        'Elephants': 'purple', 'Corvids': 'cyan', 'Dogs': 'orange',
        'Octopus': 'green', 'Parrots': 'cyan', 'Rodents': 'orange',
        'Fish': 'yellow', 'Insects': 'gray',
    }

    for species, (phi, a) in species_data.items():
        ax.scatter(phi, a, s=200, c=colors_map[species], edgecolors='black', lw=2, zorder=5)
        ax.annotate(species, (phi + 0.02, a + 0.02), fontsize=10)

    # Add quadrant labels
    ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5)
    ax.axvline(0.5, color='gray', ls='--', lw=1, alpha=0.5)

    ax.text(0.75, 0.25, 'High Integration\nLow Meta-awareness', ha='center', fontsize=9, style='italic')
    ax.text(0.25, 0.75, 'Low Integration\nHigh Meta-awareness\n(unlikely)', ha='center', fontsize=9, style='italic')
    ax.text(0.75, 0.75, 'Rich Consciousness', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.25, 0.25, 'Minimal/No\nConsciousness', ha='center', fontsize=10)

    ax.set_xlabel('Φ (Integration)', fontsize=12)
    ax.set_ylabel('A (Meta-Awareness)', fontsize=12)
    ax.set_title('Species in Consciousness Space', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('figures/paper13_fig4_consciousness_space.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper13_fig4_consciousness_space.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 generated: Consciousness space")


def fig5_ethical_gradient():
    """
    Figure 5: Ethical implications gradient
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Primates/\nCetaceans', 'Elephants/\nCorvids', 'Other\nMammals',
                  'Birds', 'Cephalopods', 'Fish', 'Arthropods']
    consciousness_prob = [0.95, 0.85, 0.70, 0.55, 0.45, 0.30, 0.15]
    annual_affected = [0.001, 0.01, 5, 20, 1, 100, 1000]  # billions

    # Normalize bubble size
    sizes = [np.log10(a + 1) * 200 + 100 for a in annual_affected]

    colors = plt.cm.RdYlGn(consciousness_prob)

    scatter = ax.scatter(range(len(categories)), consciousness_prob,
                        s=sizes, c=colors, edgecolors='black', lw=2, alpha=0.8)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Estimated Consciousness Probability', fontsize=12)
    ax.set_title('Ethical Priority: Probability × Numbers Affected', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)

    # Add size legend
    ax.text(6.5, 0.95, 'Bubble size =\nlog(billions affected)', fontsize=9, ha='center')

    # Protection recommendations
    ax.axhline(0.7, color='green', ls='--', lw=2, alpha=0.7)
    ax.axhline(0.4, color='orange', ls='--', lw=2, alpha=0.7)
    ax.text(6.8, 0.72, 'Strong protection', fontsize=9, color='green')
    ax.text(6.8, 0.42, 'Precautionary protection', fontsize=9, color='orange')

    plt.tight_layout()
    plt.savefig('figures/paper13_fig5_ethical_gradient.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper13_fig5_ethical_gradient.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 5 generated: Ethical gradient")


def main():
    """Generate all figures for Paper 13."""
    print("\n" + "="*60)
    print("Generating Paper 13 Figures: Cross-Species Consciousness")
    print("="*60 + "\n")

    fig1_consciousness_gradient()
    fig2_phylogenetic_tree()
    fig3_neural_substrates()
    fig4_consciousness_space()
    fig5_ethical_gradient()

    print("\n" + "="*60)
    print("✓ All 5 figures generated successfully!")
    print("  Output: papers/figures/paper13_fig[1-5]_*.png and .pdf")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
