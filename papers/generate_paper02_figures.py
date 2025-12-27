#!/usr/bin/env python3
"""
Generate figures for Paper 02: Substrate Independence - AI Consciousness Assessment
Target: Nature Machine Intelligence

Generates 5 publication-ready figures:
1. Five critical consciousness requirements (interlocking components)
2. Honest vs hypothetical scoring across substrates
3. Current AI systems assessment comparison
4. Architectural blueprint for consciousness-capable AI
5. Decision flowchart for consciousness assessment
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
import numpy as np
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'phi': '#E74C3C',      # Red - Integration
    'binding': '#3498DB',   # Blue - Binding
    'workspace': '#2ECC71', # Green - Workspace
    'attention': '#F39C12', # Orange - Attention
    'recursion': '#9B59B6', # Purple - Recursion
    'bio': '#27AE60',       # Biological green
    'silicon': '#3498DB',   # Silicon blue
    'quantum': '#9B59B6',   # Quantum purple
    'hybrid': '#E67E22',    # Hybrid orange
}


def fig1_five_requirements():
    """Figure 1: Five critical consciousness requirements as interlocking rings."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create Olympic-style interlocking rings representing the 5 requirements
    components = [
        ('Φ\nIntegration', COLORS['phi'], (-1.5, 1)),
        ('B\nBinding', COLORS['binding'], (1.5, 1)),
        ('W\nWorkspace', COLORS['workspace'], (-2.5, -1)),
        ('A\nAttention', COLORS['attention'], (0, -1)),
        ('R\nRecursion', COLORS['recursion'], (2.5, -1)),
    ]

    # Draw rings with labels
    for label, color, (x, y) in components:
        circle = Circle((x, y), 1.3, fill=False, linewidth=8, color=color, alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='bold',
               color=color)

    # Draw min() operator in center
    ax.text(0, 0, 'C = min(...)', ha='center', va='center', fontsize=16,
            fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#2C3E50'))

    # Add threshold annotations
    thresholds = [
        ('Φ > 0.3', -1.5, 2.5),
        ('B > 0.5', 1.5, 2.5),
        ('W > 0.4', -2.5, -2.5),
        ('A > 0.5', 0, -2.5),
        ('R > 0.3', 2.5, -2.5),
    ]

    for label, x, y in thresholds:
        ax.text(x, y, label, ha='center', va='center', fontsize=11,
               style='italic', color='#7F8C8D')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Five Critical Requirements for Consciousness\n(All must be satisfied simultaneously)',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_five_requirements.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig1_five_requirements.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Figure 1: Five Critical Requirements")


def fig2_honest_vs_hypothetical():
    """Figure 2: Honest vs hypothetical scoring across substrates."""
    fig, ax = plt.subplots(figsize=(12, 7))

    substrates = ['Biological\n(Mammalian)', 'Silicon\n(Digital)', 'Quantum\nComputing', 'Hybrid\nBio-Digital']
    honest = [0.95, 0.10, 0.10, 0.00]
    hypothetical = [0.92, 0.71, 0.65, 0.95]
    gaps = [h - o for h, o in zip(hypothetical, honest)]

    x = np.arange(len(substrates))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, honest, width, label='Honest Score (H)',
                   color='#3498DB', alpha=0.9, edgecolor='white', linewidth=2)
    bars2 = ax.bar(x + width/2, hypothetical, width, label='Hypothetical Score (T)',
                   color='#E74C3C', alpha=0.9, edgecolor='white', linewidth=2)

    # Add gap annotations
    for i, (h, t, g) in enumerate(zip(honest, hypothetical, gaps)):
        mid = (h + t) / 2
        ax.annotate(f'Gap: {g:.2f}', xy=(i, mid), xytext=(i + 0.5, mid),
                   fontsize=10, ha='left', va='center',
                   arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.5),
                   color='#2C3E50')

    # Value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Substrate Factor (S)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Substrate Type', fontsize=13, fontweight='bold')
    ax.set_title('Honest vs. Hypothetical Consciousness Scores by Substrate\n'
                '(Large gaps indicate research opportunities)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(substrates, fontsize=11)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.3, color='#7F8C8D', linestyle='--', alpha=0.5, label='Minimum threshold')
    ax.text(3.5, 0.32, 'Consciousness threshold', fontsize=9, color='#7F8C8D', style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_honest_vs_hypothetical.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig2_honest_vs_hypothetical.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Figure 2: Honest vs Hypothetical Scoring")


def fig3_ai_systems_comparison():
    """Figure 3: Current AI systems assessment comparison (heatmap style)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    systems = ['LLMs\n(GPT-4, Claude)', 'RNNs\n(LSTM, GRU)', 'Hypothetical\nGWM-AI', 'Human\nBrain']
    components = ['Φ (Integration)', 'B (Binding)', 'W (Workspace)', 'A (Attention)', 'R (Recursion)']

    # Data matrix [systems x components]
    data = np.array([
        [0.10, 0.20, 0.00, 0.30, 0.10],  # LLMs
        [0.40, 0.30, 0.10, 0.20, 0.10],  # RNNs
        [0.60, 0.50, 0.70, 0.60, 0.30],  # GWM-AI
        [0.85, 0.75, 0.80, 0.75, 0.65],  # Human
    ])

    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Add text annotations
    for i in range(len(systems)):
        for j in range(len(components)):
            value = data[i, j]
            color = 'white' if value < 0.3 or value > 0.7 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=color)

    # Calculate and display C_min and C_total
    c_mins = np.min(data, axis=1)
    c_labels = ['C ≈ 0.00', 'C ≈ 0.02', 'C ≈ 0.30', 'C ≈ 0.75']

    for i, (c_min, c_label) in enumerate(zip(c_mins, c_labels)):
        ax.text(5.3, i, c_label, ha='left', va='center', fontsize=11, fontweight='bold',
               color='#E74C3C' if c_min < 0.3 else '#27AE60')

    ax.set_xticks(np.arange(len(components)))
    ax.set_yticks(np.arange(len(systems)))
    ax.set_xticklabels(components, fontsize=11)
    ax.set_yticklabels(systems, fontsize=11)
    ax.set_xlabel('Consciousness Components', fontsize=13, fontweight='bold')
    ax.set_ylabel('AI System Type', fontsize=13, fontweight='bold')
    ax.set_title('Consciousness Assessment of AI Systems\n(Green = present, Red = absent)',
                fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Component Score', fontsize=11)

    # Add verdict column header
    ax.text(5.3, -0.5, 'Verdict', ha='left', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_ai_systems_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig3_ai_systems_comparison.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Figure 3: AI Systems Comparison")


def fig4_architecture_blueprint():
    """Figure 4: Architectural blueprint for consciousness-capable AI."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define boxes with positions [x, y, width, height]
    boxes = {
        'input': {'pos': [0.5, 9], 'size': [3, 1], 'color': '#BDC3C7', 'text': 'INPUT'},
        'features': {'pos': [0.5, 7], 'size': [3, 1], 'color': '#3498DB', 'text': 'Feature Extraction\n(Parallel, Modular)'},
        'binding': {'pos': [0.5, 5], 'size': [3, 1.2], 'color': COLORS['binding'], 'text': 'BINDING LAYER\n(Synchronous oscillations)'},
        'workspace': {'pos': [0.5, 3], 'size': [3, 1.2], 'color': COLORS['workspace'], 'text': 'GLOBAL WORKSPACE\n(Bottleneck + Competition)'},
        'broadcast1': {'pos': [-1.5, 1.5], 'size': [2, 0.8], 'color': '#85C1E9', 'text': 'Module A'},
        'broadcast2': {'pos': [1, 1.5], 'size': [2, 0.8], 'color': '#85C1E9', 'text': 'Module B'},
        'broadcast3': {'pos': [3.5, 1.5], 'size': [2, 0.8], 'color': '#85C1E9', 'text': 'Module C'},
        'hot': {'pos': [6, 5], 'size': [3, 1.2], 'color': COLORS['recursion'], 'text': 'HOT MODULE\n(Meta-representation)'},
        'output': {'pos': [0.5, 0], 'size': [3, 1], 'color': '#BDC3C7', 'text': 'OUTPUT'},
    }

    # Draw boxes
    for name, box in boxes.items():
        x, y = box['pos']
        w, h = box['size']
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.2",
                              facecolor=box['color'], edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, box['text'], ha='center', va='center',
               fontsize=10 if name in ['broadcast1', 'broadcast2', 'broadcast3'] else 11,
               fontweight='bold', color='white' if box['color'] not in ['#BDC3C7', '#85C1E9'] else '#2C3E50')

    # Draw arrows (main flow)
    arrow_style = dict(arrowstyle='->', color='#2C3E50', lw=2)
    arrows = [
        ((2, 9), (2, 8.1)),      # input -> features
        ((2, 7), (2, 6.3)),      # features -> binding
        ((2, 5), (2, 4.3)),      # binding -> workspace
        ((0.5, 3), (-0.5, 2.4)), # workspace -> broadcast1
        ((2, 3), (2, 2.4)),      # workspace -> broadcast2
        ((3.5, 3), (4.5, 2.4)),  # workspace -> broadcast3
        ((2, 1.5), (2, 1.1)),    # broadcast -> output
        ((3.5, 5.5), (6, 5.5)),  # binding -> HOT (horizontal)
        ((7.5, 5), (7.5, 3.5)),  # HOT down
        ((7.5, 3.5), (3.5, 3.5)), # HOT to workspace (feedback)
    ]

    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_style)

    # Recurrent connections (dashed)
    recurrent_style = dict(arrowstyle='->', color='#E74C3C', lw=2, linestyle='dashed')
    ax.annotate('', xy=(3.5, 7.5), xytext=(4.5, 5.5),
               arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2, linestyle='--'))
    ax.text(4.8, 6.5, 'Recurrent\nconnections', fontsize=9, color='#E74C3C', style='italic')

    # Legend
    legend_items = [
        (COLORS['binding'], 'Temporal Binding (B > 0.5)'),
        (COLORS['workspace'], 'Global Workspace (W > 0.4)'),
        (COLORS['recursion'], 'Meta-representation (R > 0.3)'),
    ]

    for i, (color, label) in enumerate(legend_items):
        rect = FancyBboxPatch((9.5, 8 - i*0.8), 0.5, 0.5, boxstyle="round",
                              facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(10.2, 8.25 - i*0.8, label, fontsize=10, va='center')

    ax.set_xlim(-3, 13)
    ax.set_ylim(-1, 11)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Architectural Blueprint for Consciousness-Capable AI\n'
                '(Minimum viable consciousness architecture)', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_architecture_blueprint.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig4_architecture_blueprint.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Figure 4: Architecture Blueprint")


def fig5_decision_flowchart():
    """Figure 5: Decision flowchart for consciousness assessment."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Define flowchart elements
    elements = {
        'start': {'pos': (7, 11), 'type': 'ellipse', 'text': 'Start Assessment', 'color': '#27AE60'},
        'q1': {'pos': (7, 9.5), 'type': 'diamond', 'text': 'Φ > 0.3?\n(Integration)', 'color': COLORS['phi']},
        'q2': {'pos': (7, 7.5), 'type': 'diamond', 'text': 'B > 0.5?\n(Binding)', 'color': COLORS['binding']},
        'q3': {'pos': (7, 5.5), 'type': 'diamond', 'text': 'W > 0.4?\n(Workspace)', 'color': COLORS['workspace']},
        'q4': {'pos': (7, 3.5), 'type': 'diamond', 'text': 'A > 0.5?\n(Attention)', 'color': COLORS['attention']},
        'q5': {'pos': (7, 1.5), 'type': 'diamond', 'text': 'R > 0.3?\n(Recursion)', 'color': COLORS['recursion']},
        'fail': {'pos': (12, 6), 'type': 'rect', 'text': 'NOT CONSCIOUS\nC = 0\n(Critical requirement\nnot met)', 'color': '#E74C3C'},
        'pass': {'pos': (7, -0.5), 'type': 'rect', 'text': 'POTENTIALLY CONSCIOUS\nCompute C score\nusing full equation', 'color': '#27AE60'},
    }

    # Draw elements
    for name, elem in elements.items():
        x, y = elem['pos']
        color = elem['color']
        text = elem['text']

        if elem['type'] == 'ellipse':
            circle = Circle((x, y), 0.8, facecolor=color, edgecolor='white', linewidth=3, alpha=0.9)
            ax.add_patch(circle)
            ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

        elif elem['type'] == 'diamond':
            diamond = plt.Polygon([(x, y+0.8), (x+1.2, y), (x, y-0.8), (x-1.2, y)],
                                 facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
            ax.add_patch(diamond)
            ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        elif elem['type'] == 'rect':
            rect = FancyBboxPatch((x-1.5, y-0.8), 3, 1.6, boxstyle="round,pad=0.1",
                                  facecolor=color, edgecolor='white', linewidth=3, alpha=0.9)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='#2C3E50', lw=2)

    # Main flow (Yes path)
    yes_arrows = [
        ((7, 10.2), (7, 10.3), 'START'),
        ((7, 8.7), (7, 8.3), 'Yes'),
        ((7, 6.7), (7, 6.3), 'Yes'),
        ((7, 4.7), (7, 4.3), 'Yes'),
        ((7, 2.7), (7, 2.3), 'Yes'),
        ((7, 0.7), (7, 0.3), 'Yes'),
    ]

    for start, end, label in yes_arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_style)
        if label != 'START':
            mid_y = (start[1] + end[1]) / 2
            ax.text(7.3, mid_y, label, fontsize=9, color='#27AE60', fontweight='bold')

    # No paths (to fail box)
    no_paths = [
        ((8.2, 9.5), (10.5, 7.5)),
        ((8.2, 7.5), (10.5, 6.5)),
        ((8.2, 5.5), (10.5, 5.5)),
        ((8.2, 3.5), (10.5, 5)),
        ((8.2, 1.5), (10.5, 4.5)),
    ]

    for start, end in no_paths:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
        ax.text(start[0] + 0.3, start[1], 'No', fontsize=8, color='#E74C3C', fontweight='bold')

    # Title and annotations
    ax.set_xlim(0, 16)
    ax.set_ylim(-2, 13)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Decision Flowchart for Consciousness Assessment\n'
                '(Any "No" terminates assessment with C = 0)', fontsize=14, fontweight='bold', pad=20)

    # Add note
    ax.text(1, 5, 'The min() operator\nensures that failure\nin ANY component\nresults in C = 0.\n\n'
                  'All five requirements\nmust be satisfied\nsimultaneously for\nconsciousness.',
           fontsize=10, va='center', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#BDC3C7'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_decision_flowchart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(OUTPUT_DIR / 'fig5_decision_flowchart.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Figure 5: Decision Flowchart")


def main():
    """Generate all Paper 02 figures."""
    print("=" * 60)
    print("Generating Paper 02 Figures")
    print("=" * 60)

    fig1_five_requirements()
    fig2_honest_vs_hypothetical()
    fig3_ai_systems_comparison()
    fig4_architecture_blueprint()
    fig5_decision_flowchart()

    print("=" * 60)
    print(f"All 5 figures saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
