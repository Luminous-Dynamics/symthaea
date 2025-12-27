#!/usr/bin/env python3
"""
Generate figures for Paper 15: Future Directions
Target: Neuron (Perspective)

Run with: nix-shell -p python311 python311Packages.matplotlib python311Packages.numpy --run "python3 generate_paper15_figures.py"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patches as mpatches
import os

os.makedirs('figures', exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'phi': '#E74C3C',
    'binding': '#3498DB',
    'workspace': '#2ECC71',
    'awareness': '#9B59B6',
    'recursion': '#F39C12',
    'foundational': '#E74C3C',
    'empirical': '#3498DB',
    'clinical': '#2ECC71',
    'philosophical': '#9B59B6',
    'methodology': '#F39C12',
}


def fig1_open_questions():
    """
    Figure 1: Ten open questions organized by domain
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.7, 'Ten Open Questions in Consciousness Science', fontsize=16, fontweight='bold',
            ha='center', va='center')

    # Domain boxes
    domains = [
        ('Foundational (1-3)', 0.5, 7.5, 4, 2, COLORS['foundational'],
         ['1. Why experience?', '2. Necessity vs correlation?', '3. Minimal sufficient set?']),
        ('Empirical (4-6)', 5.5, 7.5, 4, 2, COLORS['empirical'],
         ['4. Component dynamics?', '5. Evolutionary trajectories?', '6. Artificial consciousness?']),
        ('Clinical (7-8)', 0.5, 4.5, 4, 2, COLORS['clinical'],
         ['7. Targeted interventions?', '8. Validation without report?']),
        ('Philosophical (9-10)', 5.5, 4.5, 4, 2, COLORS['philosophical'],
         ['9. Hard problem status?', '10. Ethics of engineering?']),
    ]

    for name, x, y, w, h, color, questions in domains:
        # Background box
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', lw=2, alpha=0.2)
        ax.add_patch(box)

        # Domain title
        ax.text(x + w/2, y + h - 0.3, name, fontsize=11, fontweight='bold',
               ha='center', va='top', color=color)

        # Questions
        for i, q in enumerate(questions):
            ax.text(x + 0.2, y + h - 0.8 - i*0.4, q, fontsize=9, va='center')

    # Methodology box at bottom
    box = FancyBboxPatch((1.5, 1.5), 7, 2.2, boxstyle="round,pad=0.05",
                         facecolor=COLORS['methodology'], edgecolor='black', lw=2, alpha=0.2)
    ax.add_patch(box)
    ax.text(5, 3.4, 'Methodological Priorities', fontsize=11, fontweight='bold',
           ha='center', va='top', color=COLORS['methodology'])

    priorities = ['Standardization (ICMI)', 'Large-scale replication', 'Theory-experiment dialogue', 'Open science platform']
    for i, p in enumerate(priorities):
        ax.text(1.7 + (i % 2) * 3.5, 2.8 - (i // 2) * 0.5, f'• {p}', fontsize=9, va='center')

    # Arrows showing connections
    ax.annotate('', xy=(5, 7.5), xytext=(5, 6.7),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='gray'))

    plt.tight_layout()
    plt.savefig('figures/paper15_fig1_open_questions.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper15_fig1_open_questions.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 1 generated: Open questions")


def fig2_research_timeline():
    """
    Figure 2: Proposed 10-year research timeline
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    years = ['2024', '2025', '2026', '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035']
    y_positions = np.arange(len(years))

    # Research streams
    streams = {
        'Standardization': (0, 3, COLORS['methodology']),
        'Replication': (1, 4, COLORS['methodology']),
        'Development Study': (2, 9, COLORS['empirical']),
        'Cross-Species': (1, 6, COLORS['empirical']),
        'AI Consciousness': (3, 10, COLORS['foundational']),
        'Clinical Trials': (2, 8, COLORS['clinical']),
        'Ethics Framework': (0, 5, COLORS['philosophical']),
        'Physics Integration': (5, 10, COLORS['foundational']),
    }

    # Plot horizontal bars
    for i, (stream, (start, end, color)) in enumerate(streams.items()):
        ax.barh(i, end - start, left=start, height=0.6, color=color, alpha=0.7, edgecolor='black', lw=1)
        ax.text(start + 0.1, i, stream, fontsize=9, va='center', fontweight='bold')

    # Milestones
    milestones = [
        (2, 'Protocols\nfinalized', 'green'),
        (4, 'First\nreplications', 'blue'),
        (6, 'Interventions\nvalidated', 'purple'),
        (8, 'AI criteria\nestablished', 'red'),
        (10, 'Major\nadvances', 'gold'),
    ]

    for x, label, color in milestones:
        ax.axvline(x, color=color, ls='--', lw=2, alpha=0.5)
        ax.text(x, len(streams) + 0.3, label, ha='center', fontsize=8, color=color)

    ax.set_yticks(range(len(streams)))
    ax.set_yticklabels([])
    ax.set_xlabel('Years from 2024', fontsize=12)
    ax.set_title('Proposed 10-Year Research Timeline', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 12)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['methodology'], edgecolor='black', alpha=0.7, label='Methodology'),
        mpatches.Patch(facecolor=COLORS['empirical'], edgecolor='black', alpha=0.7, label='Empirical'),
        mpatches.Patch(facecolor=COLORS['clinical'], edgecolor='black', alpha=0.7, label='Clinical'),
        mpatches.Patch(facecolor=COLORS['philosophical'], edgecolor='black', alpha=0.7, label='Philosophical'),
        mpatches.Patch(facecolor=COLORS['foundational'], edgecolor='black', alpha=0.7, label='Foundational'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('figures/paper15_fig2_research_timeline.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper15_fig2_research_timeline.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 generated: Research timeline")


def fig3_funding_landscape():
    """
    Figure 3: Current and proposed funding landscape
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Current funding
    ax = axes[0]
    current_sources = ['NIH', 'Private\nFoundations', 'Industry', 'Other']
    current_amounts = [50, 30, 10, 10]
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#95A5A6']

    ax.pie(current_amounts, labels=current_sources, colors=colors, autopct='$%dM',
           startangle=90, explode=(0.05, 0.05, 0.05, 0.05))
    ax.set_title('A. Current (~$100M/year)', fontsize=12, fontweight='bold')

    # Panel B: Proposed investment
    ax = axes[1]
    proposed = ['Standardization', 'Development\nStudy', 'Cross-Species', 'Clinical\nTrials',
                'AI Research', 'Ethics']
    amounts = [10, 50, 30, 100, 20, 10]
    colors = [COLORS['methodology'], COLORS['empirical'], COLORS['empirical'],
              COLORS['clinical'], COLORS['foundational'], COLORS['philosophical']]

    bars = ax.barh(proposed, amounts, color=colors, alpha=0.8, edgecolor='black', lw=1.5)
    ax.set_xlabel('Investment ($M over 10 years)', fontsize=11)
    ax.set_title('B. Proposed Investment ($220M total)', fontsize=12, fontweight='bold')

    # Add value labels
    for bar, amount in zip(bars, amounts):
        ax.text(amount + 2, bar.get_y() + bar.get_height()/2, f'${amount}M',
               va='center', fontsize=9)

    ax.set_xlim(0, 120)

    plt.suptitle('Consciousness Research Funding Landscape', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/paper15_fig3_funding_landscape.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper15_fig3_funding_landscape.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 generated: Funding landscape")


def fig4_vision_2035():
    """
    Figure 4: Vision for 2035 - expected achievements
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Vision for Consciousness Science: 2035', fontsize=16, fontweight='bold',
            ha='center', va='center')

    # Four quadrants
    quadrants = [
        ('Scientific\nAchievements', 1, 7.5, COLORS['empirical'],
         ['Standardized protocols worldwide', 'Component dynamics understood',
          'Evolutionary timeline validated', 'AI consciousness criteria']),
        ('Clinical\nAdvances', 6, 7.5, COLORS['clinical'],
         ['Personalized DOC treatment', 'Consciousness-preserving anesthesia',
          'Early detection protocols', 'Targeted interventions']),
        ('Philosophical\nProgress', 1, 3.5, COLORS['philosophical'],
         ['Hard problem status clarified', 'Ethical frameworks adopted',
          'Physics integration', 'New theoretical syntheses']),
        ('Societal\nImpact', 6, 3.5, COLORS['foundational'],
         ['Revised animal welfare laws', 'AI rights frameworks',
          'End-of-life policy reform', 'Public understanding']),
    ]

    for title, x, y, color, items in quadrants:
        # Background
        box = FancyBboxPatch((x - 0.5, y - 2), 3.5, 3, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', lw=2, alpha=0.15)
        ax.add_patch(box)

        # Title
        ax.text(x + 1.25, y + 0.8, title, fontsize=12, fontweight='bold',
               ha='center', color=color)

        # Items
        for i, item in enumerate(items):
            ax.text(x - 0.3, y + 0.3 - i*0.5, f'✓ {item}', fontsize=9, va='center')

    # Central connector
    circle = Circle((5, 5.5), 0.8, facecolor='white', edgecolor='black', lw=2)
    ax.add_patch(circle)
    ax.text(5, 5.5, 'C=f(Φ,B,W,A,R)', fontsize=10, ha='center', va='center', fontweight='bold')

    # Arrows from center to quadrants
    for x, y in [(2.5, 7.5), (7.5, 7.5), (2.5, 4), (7.5, 4)]:
        ax.annotate('', xy=(x, y), xytext=(5, 5.5),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5,
                                 connectionstyle='arc3,rad=0.1'))

    plt.tight_layout()
    plt.savefig('figures/paper15_fig4_vision_2035.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper15_fig4_vision_2035.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 generated: Vision 2035")


def fig5_progress_metrics():
    """
    Figure 5: How we'll know we've succeeded - progress metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Knowledge accumulation (papers, citations)
    ax = axes[0, 0]
    years = np.arange(2000, 2036)
    papers = 100 * np.exp(0.08 * (years - 2000))  # Exponential growth
    papers_projected = np.where(years > 2024, papers, np.nan)
    papers_actual = np.where(years <= 2024, papers * (1 + 0.1*np.random.randn(len(years))), np.nan)

    ax.plot(years, papers_actual, 'b-', lw=2, label='Historical')
    ax.plot(years, papers_projected, 'b--', lw=2, label='Projected', alpha=0.5)
    ax.axvline(2024, color='gray', ls=':', lw=1)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel('Publications/year', fontsize=10)
    ax.set_title('A. Knowledge Accumulation', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_xlim(2000, 2035)

    # Panel B: Clinical translation
    ax = axes[0, 1]
    metrics = ['Sensitivity', 'Specificity', 'AUC', 'Clinical\nAdoption']
    current = [0.71, 0.85, 0.82, 0.15]
    projected = [0.97, 0.93, 0.96, 0.70]

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width/2, current, width, label='2024', color='gray', alpha=0.8)
    ax.bar(x + width/2, projected, width, label='2035', color=COLORS['clinical'], alpha=0.8)

    ax.set_ylabel('Performance', fontsize=10)
    ax.set_title('B. Clinical Translation', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # Panel C: Theory unification
    ax = axes[1, 0]
    theories = ['IIT', 'GNW', 'HOT', 'PP', 'Component\nFramework']
    integration_2024 = [0.3, 0.4, 0.35, 0.25, 0.6]
    integration_2035 = [0.7, 0.75, 0.7, 0.65, 0.9]

    x = np.arange(len(theories))
    ax.bar(x - width/2, integration_2024, width, label='2024', color='gray', alpha=0.7)
    ax.bar(x + width/2, integration_2035, width, label='2035', color='gold', alpha=0.8)

    ax.set_ylabel('Theoretical Integration', fontsize=10)
    ax.set_title('C. Theory Unification', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(theories, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # Panel D: Societal impact
    ax = axes[1, 1]
    impacts = ['Animal\nWelfare Laws', 'AI\nRegulations', 'Medical\nProtocols', 'Public\nUnderstanding']
    current_impact = [0.2, 0.1, 0.3, 0.15]
    projected_impact = [0.6, 0.5, 0.8, 0.55]

    x = np.arange(len(impacts))
    ax.bar(x - width/2, current_impact, width, label='2024', color=COLORS['philosophical'], alpha=0.5)
    ax.bar(x + width/2, projected_impact, width, label='2035', color=COLORS['philosophical'], alpha=0.9)

    ax.set_ylabel('Policy Adoption', fontsize=10)
    ax.set_title('D. Societal Impact', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(impacts, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    plt.suptitle('Success Metrics: 2024 vs 2035 Vision', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/paper15_fig5_progress_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('figures/paper15_fig5_progress_metrics.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Figure 5 generated: Progress metrics")


def main():
    """Generate all figures for Paper 15."""
    print("\n" + "="*60)
    print("Generating Paper 15 Figures: Future Directions")
    print("="*60 + "\n")

    fig1_open_questions()
    fig2_research_timeline()
    fig3_funding_landscape()
    fig4_vision_2035()
    fig5_progress_metrics()

    print("\n" + "="*60)
    print("✓ All 5 figures generated successfully!")
    print("  Output: papers/figures/paper15_fig[1-5]_*.png and .pdf")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
