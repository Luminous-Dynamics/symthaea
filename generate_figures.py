#!/usr/bin/env python3
"""
Generate Publication-Ready Figures for Dimensional Φ Analysis
==============================================================

Creates high-quality figures for Nature/Science journal submission:
1. Dimensional curve (Φ vs dimension, 1D-7D with asymptotic fit)
2. Topology rankings (complete 19-topology bar chart)
3. Category comparison (boxplot by topology type)
4. Non-orientability dimension effect (1D vs 2D twist comparison)

Author: Tristan Stoltz & Claude
Date: December 28, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy.optimize import curve_fit

# Set publication-quality defaults
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 13

# Color palette (colorblind-friendly)
COLORS = {
    'hypercube': '#0173B2',      # Blue
    'uniform': '#DE8F05',         # Orange
    'original': '#029E73',        # Green
    'tier1': '#CC78BC',          # Purple
    'tier2': '#CA9161',          # Brown
    'tier3': '#ECE133',          # Yellow
    'baseline': '#949494',       # Gray
    'champion': '#D55E00',       # Red-orange
}

# ============================================================================
# Figure 1: Dimensional Sweep Curve (1D-7D)
# ============================================================================

def asymptotic_model(k, phi_max, A, alpha):
    """Asymptotic model: Φ(k) = Φ_max - A·exp(-α·k)"""
    return phi_max - A * np.exp(-alpha * k)

def create_dimensional_curve():
    """Create dimensional sweep figure with asymptotic fit"""

    # Data: 1D-7D hypercubes
    dimensions = np.array([1, 2, 3, 4, 5, 6, 7])
    phi_values = np.array([1.0000, 0.5011, 0.4960, 0.4976, 0.4987, 0.4990, 0.4991])
    std_devs = np.array([0.0000, 0.0002, 0.0002, 0.0001, 0.0001, 0.0001, 0.0000])

    # Exclude 1D (K₂ edge case) from fit
    fit_dims = dimensions[1:]
    fit_phi = phi_values[1:]

    # Fit asymptotic model
    popt, _ = curve_fit(asymptotic_model, fit_dims, fit_phi,
                        p0=[0.5, 0.004, 0.3], maxfev=10000)
    phi_max_fit, A_fit, alpha_fit = popt

    # Generate smooth curve
    k_smooth = np.linspace(2, 10, 200)
    phi_smooth = asymptotic_model(k_smooth, *popt)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot 1D point separately (edge case)
    ax.errorbar(1, phi_values[0], yerr=std_devs[0],
                fmt='s', color=COLORS['baseline'], markersize=8,
                capsize=4, capthick=1.5, elinewidth=1.5,
                label='1D K₂ (edge case)', zorder=3)

    # Plot 2D-7D data points
    ax.errorbar(fit_dims, fit_phi, yerr=std_devs[1:],
                fmt='o', color=COLORS['hypercube'], markersize=8,
                capsize=4, capthick=1.5, elinewidth=1.5,
                label='Measured Φ (2D-7D)', zorder=3)

    # Plot asymptotic fit
    ax.plot(k_smooth, phi_smooth, '--', color=COLORS['champion'],
            linewidth=2, label=f'Asymptotic fit: Φ_max = {phi_max_fit:.3f}', zorder=2)

    # Add asymptote line
    ax.axhline(y=phi_max_fit, color=COLORS['baseline'], linestyle=':',
               linewidth=1.5, alpha=0.7, label=f'Φ_max ≈ 0.5', zorder=1)

    # Highlight 3D and 4D points
    ax.scatter([3], [phi_values[2]], s=200, facecolors='none',
              edgecolors=COLORS['uniform'], linewidths=2.5, zorder=4,
              label='3D (biological brains)')
    ax.scatter([4], [phi_values[3]], s=200, marker='*',
              color=COLORS['champion'], edgecolors='black', linewidths=0.5,
              zorder=5, label='4D champion (Φ = 0.4976)')

    # Formatting
    ax.set_xlabel('Hypercube Dimension (k)', fontweight='bold')
    ax.set_ylabel('Integrated Information (Φ)', fontweight='bold')
    ax.set_title('Dimensional Scaling of Integrated Information\n' +
                 'Asymptotic Convergence to Φ_max ≈ 0.5', fontweight='bold')
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(['1D\n(K₂)', '2D\n(Square)', '3D\n(Cube)',
                        '4D\n(Tesseract)', '5D\n(Penteract)',
                        '6D\n(Hexeract)', '7D\n(Hepteract)'], fontsize=8)
    ax.set_ylim(0.48, 1.02)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', framealpha=0.95)

    # Add annotations
    ax.annotate('99.2% of Φ_max', xy=(3, phi_values[2]),
                xytext=(1.5, 0.52), fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['uniform']),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    ax.annotate(f'Δ = +{((phi_values[6]-phi_values[2])/phi_values[2]*100):.2f}%\n(3D→7D)',
                xy=(7, phi_values[6]), xytext=(5.5, 0.51), fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLORS['hypercube']),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

    plt.tight_layout()
    plt.savefig('figure_1_dimensional_curve.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_1_dimensional_curve.pdf', bbox_inches='tight')
    print("✅ Figure 1 saved: figure_1_dimensional_curve.png/.pdf")
    return fig

# ============================================================================
# Figure 2: Complete Topology Rankings
# ============================================================================

def create_topology_rankings():
    """Create comprehensive topology rankings bar chart"""

    # Data: 19 topologies (ranked by RealHV Φ)
    topologies = [
        'Hypercube 4D', 'Hypercube 3D', 'Ring', 'Torus (3×3)', 'Klein Bottle',
        'Dense Network', 'Lattice', 'Modular', 'Small-World', 'Line',
        'Scale-Free', 'Hyperbolic', 'Binary Tree', 'Quantum (3:1:1)',
        'Star', 'Quantum (1:1:1)', 'Random', 'Fractal', 'Möbius Strip'
    ]

    phi_values = [
        0.4976, 0.4960, 0.4954, 0.4953, 0.4940,
        0.4888, 0.4855, 0.4812, 0.4786, 0.4768,
        0.4753, 0.4718, 0.4712, 0.4650,
        0.4553, 0.4432, 0.4358, 0.4345, 0.3729
    ]

    std_devs = [
        0.0001, 0.0002, 0.0000, 0.0001, 0.0002,
        0.0000, 0.0000, 0.0000, 0.0060, 0.0000,
        0.0030, 0.0000, 0.0000, 0.0007,
        0.0004, 0.0009, 0.0005, 0.0001, 0.0000
    ]

    # Category colors
    colors = [
        COLORS['tier3'], COLORS['tier3'], COLORS['uniform'], COLORS['tier1'], COLORS['tier2'],
        COLORS['original'], COLORS['original'], COLORS['original'], COLORS['tier1'], COLORS['original'],
        COLORS['tier2'], COLORS['tier2'], COLORS['original'], COLORS['tier3'],
        COLORS['original'], COLORS['tier3'], COLORS['baseline'], COLORS['tier3'], COLORS['tier1']
    ]

    # Highlight champions
    colors[0] = COLORS['champion']  # Hypercube 4D
    colors[1] = COLORS['hypercube']  # Hypercube 3D

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Horizontal bar chart (reversed for descending order)
    y_pos = np.arange(len(topologies))
    bars = ax.barh(y_pos, phi_values, xerr=std_devs,
                   color=colors, edgecolor='black', linewidth=0.8,
                   capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})

    # Add value labels
    for i, (val, err) in enumerate(zip(phi_values, std_devs)):
        ax.text(val + 0.005, i, f'{val:.4f}',
               va='center', ha='left', fontsize=7, fontweight='bold')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(topologies, fontsize=9)
    ax.set_xlabel('Integrated Information (Φ)', fontweight='bold')
    ax.set_title('Complete Topology Rankings (n=19)\nRealHV Φ Method with Standard Deviations',
                fontweight='bold', pad=15)
    ax.set_xlim(0.35, 0.52)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.invert_yaxis()  # Highest at top

    # Add vertical line at random baseline
    ax.axvline(x=phi_values[16], color=COLORS['baseline'], linestyle='--',
              linewidth=2, alpha=0.7, label='Random baseline')

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['champion'], label='Champion (Hypercube 4D)', edgecolor='black'),
        mpatches.Patch(color=COLORS['hypercube'], label='Hypercube 3D', edgecolor='black'),
        mpatches.Patch(color=COLORS['uniform'], label='Uniform Manifolds', edgecolor='black'),
        mpatches.Patch(color=COLORS['original'], label='Original 8', edgecolor='black'),
        mpatches.Patch(color=COLORS['tier1'], label='Tier 1 Exotic', edgecolor='black'),
        mpatches.Patch(color=COLORS['tier2'], label='Tier 2 Exotic', edgecolor='black'),
        mpatches.Patch(color=COLORS['tier3'], label='Tier 3 Exotic', edgecolor='black'),
        mpatches.Patch(color=COLORS['baseline'], label='Random Baseline', edgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)

    # Add ranking medals
    medal_positions = [0, 1, 2]
    medals = ['🥇', '🥈', '🥉']
    for pos, medal in zip(medal_positions, medals):
        ax.text(0.36, pos, medal, fontsize=16, va='center')

    plt.tight_layout()
    plt.savefig('figure_2_topology_rankings.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_2_topology_rankings.pdf', bbox_inches='tight')
    print("✅ Figure 2 saved: figure_2_topology_rankings.png/.pdf")
    return fig

# ============================================================================
# Figure 3: Category Comparison (Boxplot)
# ============================================================================

def create_category_comparison():
    """Create boxplot comparing topology categories"""

    # Data organized by category
    categories = {
        'Hypercubes\n(3D-4D)': [0.4960, 0.4976],
        'Uniform\nManifolds': [0.4954, 0.4953, 0.4940],
        'Original\n(8-node)': [0.4888, 0.4855, 0.4812, 0.4768, 0.4712, 0.4553, 0.4358],
        'Tier 1\nExotic': [0.4953, 0.4786, 0.3729],  # Torus, Small-World, Möbius
        'Tier 2\nExotic': [0.4940, 0.4753, 0.4718],  # Klein, Scale-Free, Hyperbolic
        'Tier 3\nExotic': [0.4976, 0.4650, 0.4432, 0.4345],  # Hypercube 4D, Quantum x2, Fractal
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for boxplot
    data = list(categories.values())
    labels = list(categories.keys())

    # Create boxplot
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
                    medianprops=dict(color='black', linewidth=2),
                    boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5))

    # Color boxes by category
    category_colors = [COLORS['hypercube'], COLORS['uniform'], COLORS['original'],
                      COLORS['tier1'], COLORS['tier2'], COLORS['tier3']]
    for patch, color in zip(bp['boxes'], category_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (category_data, x_pos) in enumerate(zip(data, range(1, len(data)+1))):
        x = np.random.normal(x_pos, 0.04, size=len(category_data))
        ax.scatter(x, category_data, alpha=0.6, s=40, color='black', zorder=3)

    # Add statistics as text
    for i, (label, category_data) in enumerate(categories.items(), 1):
        mean_val = np.mean(category_data)
        std_val = np.std(category_data)
        ax.text(i, 0.375, f'μ={mean_val:.4f}\nσ={std_val:.4f}',
               ha='center', fontsize=7, bbox=dict(boxstyle='round',
               facecolor='white', alpha=0.8))

    # Formatting
    ax.set_ylabel('Integrated Information (Φ)', fontweight='bold')
    ax.set_title('Φ Distribution by Topology Category\nBoxplot with Individual Data Points',
                fontweight='bold')
    ax.set_ylim(0.36, 0.51)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add random baseline
    ax.axhline(y=0.4358, color=COLORS['baseline'], linestyle='--',
              linewidth=2, alpha=0.7, label='Random baseline (Φ=0.4358)')

    ax.legend(loc='upper right', framealpha=0.95)

    plt.tight_layout()
    plt.savefig('figure_3_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_3_category_comparison.pdf', bbox_inches='tight')
    print("✅ Figure 3 saved: figure_3_category_comparison.png/.pdf")
    return fig

# ============================================================================
# Figure 4: Non-Orientability Dimension Effect
# ============================================================================

def create_non_orientability_figure():
    """Create figure showing 1D vs 2D twist effects"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left panel: 1D Non-orientability (Möbius Strip) ----

    topologies_1d = ['Ring\n(1D)', 'Möbius Strip\n(1D twist)']
    phi_1d = [0.4954, 0.3729]
    colors_1d = [COLORS['uniform'], 'red']

    bars1 = ax1.bar(topologies_1d, phi_1d, color=colors_1d,
                    edgecolor='black', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Integrated Information (Φ)', fontweight='bold')
    ax1.set_title('1D Non-Orientability Effect\nMöbius Strip vs Ring',
                  fontweight='bold')
    ax1.set_ylim(0, 0.55)
    ax1.axhline(y=0.4358, color=COLORS['baseline'], linestyle='--',
                linewidth=2, alpha=0.7, label='Random baseline')
    ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels and percentage change
    for bar, val in zip(bars1, phi_1d):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add arrow showing decrease
    ax1.annotate('', xy=(1, phi_1d[1]), xytext=(0, phi_1d[0]),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax1.text(0.5, 0.43, f'-24.7%\nCATASTROPHIC\nFAILURE',
            ha='center', fontsize=9, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax1.legend(loc='upper right', framealpha=0.95)

    # ---- Right panel: 2D Non-orientability (Klein Bottle) ----

    topologies_2d = ['Torus\n(2D)', 'Klein Bottle\n(2D twist)']
    phi_2d = [0.4953, 0.4940]
    colors_2d = [COLORS['uniform'], COLORS['tier2']]

    bars2 = ax2.bar(topologies_2d, phi_2d, color=colors_2d,
                    edgecolor='black', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Integrated Information (Φ)', fontweight='bold')
    ax2.set_title('2D Non-Orientability Effect\nKlein Bottle vs Torus',
                  fontweight='bold')
    ax2.set_ylim(0.48, 0.50)
    ax2.axhline(y=0.4358, color=COLORS['baseline'], linestyle='--',
                linewidth=2, alpha=0.7, label='Random baseline')
    ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

    # Add value labels and percentage change
    for bar, val in zip(bars2, phi_2d):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add arrow showing minimal decrease
    ax2.annotate('', xy=(1, phi_2d[1]), xytext=(0, phi_2d[0]),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['tier2']))
    ax2.text(0.5, 0.4947, f'-0.26%\nMAINTAINS\nHIGH Φ',
            ha='center', fontsize=9, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax2.legend(loc='upper right', framealpha=0.95)

    # Overall title
    fig.suptitle('Dimension-Dependent Non-Orientability Effect\n' +
                'Local Uniformity > Global Orientability',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('figure_4_non_orientability.png', dpi=300, bbox_inches='tight')
    plt.savefig('figure_4_non_orientability.pdf', bbox_inches='tight')
    print("✅ Figure 4 saved: figure_4_non_orientability.png/.pdf")
    return fig

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n🎨 Generating Publication-Ready Figures")
    print("=" * 60)

    # Create output directory
    import os
    os.makedirs('figures', exist_ok=True)
    os.chdir('figures')

    # Generate all figures
    print("\n📊 Creating figures...")
    fig1 = create_dimensional_curve()
    plt.close()

    fig2 = create_topology_rankings()
    plt.close()

    fig3 = create_category_comparison()
    plt.close()

    fig4 = create_non_orientability_figure()
    plt.close()

    print("\n" + "=" * 60)
    print("✅ ALL FIGURES GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📁 Output files (PNG + PDF):")
    print("   - figure_1_dimensional_curve.png/.pdf")
    print("   - figure_2_topology_rankings.png/.pdf")
    print("   - figure_3_category_comparison.png/.pdf")
    print("   - figure_4_non_orientability.png/.pdf")
    print("\n📂 Location: ./figures/")
    print("\n💡 Ready for manuscript submission!\n")
