# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Visualization: 0TML System Architecture Diagram

Shows the complete Hybrid-Trust Architecture with:
- Mode 0 (Peer-Comparison): Failed at 35% BFT
- Mode 1 (Ground Truth - PoGQ): Succeeds at 35-50% BFT
- BFT Estimation & Fail-Safe Mechanism
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np


def create_architecture_diagram():
    """
    Create system architecture diagram showing Mode 0, Mode 1, and fail-safe.
    """

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    fig.suptitle(
        'Hybrid-Trust Architecture: Mode 0 vs Mode 1 Byzantine Detection',
        fontsize=18,
        fontweight='bold',
        y=0.98
    )

    # Color scheme
    mode0_color = '#e74c3c'      # Red (failed)
    mode1_color = '#27ae60'      # Green (success)
    failsafe_color = '#f39c12'   # Orange
    client_color = '#3498db'     # Blue
    server_color = '#9b59b6'     # Purple

    # ===== FEDERATED LEARNING CLIENTS (Top) =====
    y_clients = 10.5

    # Honest clients
    for i in range(4):
        x = 2 + i * 1.2
        circle = Circle((x, y_clients), 0.35, color=client_color, alpha=0.7,
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y_clients, 'H', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    ax.text(3.2, y_clients - 0.8, 'Honest Clients (13)',
           ha='center', fontsize=10, fontweight='bold')

    # Byzantine clients
    for i in range(3):
        x = 7.5 + i * 1.2
        circle = Circle((x, y_clients), 0.35, color=mode0_color, alpha=0.7,
                       edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y_clients, 'B', ha='center', va='center',
               fontsize=10, fontweight='bold', color='white')

    ax.text(8.6, y_clients - 0.8, 'Byzantine Clients (7)',
           ha='center', fontsize=10, fontweight='bold', color=mode0_color)

    ax.text(5.9, y_clients + 0.7, '35% BFT (7/20)',
           ha='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Arrows from clients to detectors
    arrow1 = FancyArrowPatch((3.2, y_clients - 1.3), (2, 7.5),
                            arrowstyle='->', lw=3, color='gray', alpha=0.6)
    ax.add_patch(arrow1)

    arrow2 = FancyArrowPatch((8.6, y_clients - 1.3), (14, 7.5),
                            arrowstyle='->', lw=3, color='gray', alpha=0.6)
    ax.add_patch(arrow2)

    # ===== MODE 0: PEER-COMPARISON (Left Side) =====
    y_mode0 = 6

    # Mode 0 box
    mode0_box = FancyBboxPatch((0.5, y_mode0 - 1.5), 3, 3,
                              boxstyle="round,pad=0.15",
                              edgecolor=mode0_color, facecolor='white',
                              linewidth=3, alpha=0.9)
    ax.add_patch(mode0_box)

    ax.text(2, y_mode0 + 1.2, 'Mode 0', ha='center', fontsize=14,
           fontweight='bold', color=mode0_color)
    ax.text(2, y_mode0 + 0.7, 'Peer-Comparison', ha='center', fontsize=11,
           style='italic')

    # Mode 0 components
    ax.text(2, y_mode0 + 0.2, '• Cosine Similarity', ha='center', fontsize=9)
    ax.text(2, y_mode0 - 0.2, '• Magnitude Z-Score', ha='center', fontsize=9)
    ax.text(2, y_mode0 - 0.6, '• No Validation Set', ha='center', fontsize=9,
           style='italic', color='red')

    # Mode 0 results box
    result0_box = FancyBboxPatch((0.5, y_mode0 - 3.5), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                edgecolor=mode0_color, facecolor=mode0_color,
                                linewidth=2, alpha=0.2)
    ax.add_patch(result0_box)

    ax.text(2, y_mode0 - 2.5, '❌ FAILED', ha='center', fontsize=12,
           fontweight='bold', color=mode0_color)
    ax.text(2, y_mode0 - 3, 'Detection: 100%', ha='center', fontsize=9)
    ax.text(2, y_mode0 - 3.3, 'FPR: 100% (!)', ha='center', fontsize=9,
           fontweight='bold', color='red')

    # Arrow to verdict
    arrow3 = FancyArrowPatch((2, y_mode0 - 3.7), (2, 0.8),
                            arrowstyle='->', lw=3, color=mode0_color, alpha=0.6)
    ax.add_patch(arrow3)

    # Mode 0 verdict
    verdict0 = FancyBboxPatch((0.5, 0.2), 3, 0.5,
                             boxstyle="round,pad=0.1",
                             edgecolor=mode0_color, facecolor=mode0_color,
                             linewidth=2, alpha=0.3)
    ax.add_patch(verdict0)
    ax.text(2, 0.45, 'UNUSABLE', ha='center', fontsize=11,
           fontweight='bold', color=mode0_color)

    # ===== MODE 1: GROUND TRUTH (Right Side) =====
    y_mode1 = 6

    # Mode 1 box
    mode1_box = FancyBboxPatch((12.5, y_mode1 - 1.5), 3, 3,
                              boxstyle="round,pad=0.15",
                              edgecolor=mode1_color, facecolor='white',
                              linewidth=3, alpha=0.9)
    ax.add_patch(mode1_box)

    ax.text(14, y_mode1 + 1.2, 'Mode 1', ha='center', fontsize=14,
           fontweight='bold', color=mode1_color)
    ax.text(14, y_mode1 + 0.7, 'Ground Truth (PoGQ)', ha='center', fontsize=11,
           style='italic')

    # Mode 1 components
    ax.text(14, y_mode1 + 0.2, '• Validation Loss', ha='center', fontsize=9)
    ax.text(14, y_mode1 - 0.2, '• Adaptive Threshold', ha='center', fontsize=9)
    ax.text(14, y_mode1 - 0.6, '• Quality Score', ha='center', fontsize=9,
           style='italic', color='green')

    # Validation set (small box)
    val_box = FancyBboxPatch((13.2, y_mode1 - 1.2), 1.6, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor=server_color, facecolor=server_color,
                            linewidth=1.5, alpha=0.3)
    ax.add_patch(val_box)
    ax.text(14, y_mode1 - 0.95, 'Server Val Set', ha='center', fontsize=8,
           fontweight='bold')

    # Mode 1 results box
    result1_box = FancyBboxPatch((12.5, y_mode1 - 3.5), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                edgecolor=mode1_color, facecolor=mode1_color,
                                linewidth=2, alpha=0.2)
    ax.add_patch(result1_box)

    ax.text(14, y_mode1 - 2.5, '✅ SUCCESS', ha='center', fontsize=12,
           fontweight='bold', color=mode1_color)
    ax.text(14, y_mode1 - 3, 'Detection: 100%', ha='center', fontsize=9)
    ax.text(14, y_mode1 - 3.3, 'FPR: 0%', ha='center', fontsize=9,
           fontweight='bold', color='green')

    # Arrow to verdict
    arrow4 = FancyArrowPatch((14, y_mode1 - 3.7), (14, 0.8),
                            arrowstyle='->', lw=3, color=mode1_color, alpha=0.6)
    ax.add_patch(arrow4)

    # Mode 1 verdict
    verdict1 = FancyBboxPatch((12.5, 0.2), 3, 0.5,
                             boxstyle="round,pad=0.1",
                             edgecolor=mode1_color, facecolor=mode1_color,
                             linewidth=2, alpha=0.3)
    ax.add_patch(verdict1)
    ax.text(14, 0.45, 'IDEAL', ha='center', fontsize=11,
           fontweight='bold', color=mode1_color)

    # ===== BFT ESTIMATION & FAIL-SAFE (Center Bottom) =====
    y_failsafe = 2.8

    # Fail-safe box
    failsafe_box = FancyBboxPatch((5.5, y_failsafe - 1), 5, 2,
                                 boxstyle="round,pad=0.15",
                                 edgecolor=failsafe_color, facecolor='white',
                                 linewidth=3, alpha=0.9, linestyle='--')
    ax.add_patch(failsafe_box)

    ax.text(8, y_failsafe + 0.7, 'BFT Estimation & Fail-Safe', ha='center',
           fontsize=12, fontweight='bold', color=failsafe_color)

    # BFT estimation formula
    ax.text(8, y_failsafe + 0.2, 'ρ̂ = 0.5×(detected/N) + 0.3×(conf/N) + 0.2×(low_rep/N)',
           ha='center', fontsize=8, family='monospace')

    # Fail-safe logic
    ax.text(8, y_failsafe - 0.2, 'if ρ̂ > 0.35 → HALT (Mode 0 unsafe)',
           ha='center', fontsize=9, color='red', fontweight='bold')
    ax.text(8, y_failsafe - 0.5, 'if ρ̂ > 0.50 → HALT (Mode 1 unsafe)',
           ha='center', fontsize=9, color='orange', fontweight='bold')

    # Arrows from modes to fail-safe
    arrow5 = FancyArrowPatch((3.5, y_mode0 - 2), (5.7, y_failsafe + 0.5),
                            arrowstyle='->', lw=2, color=failsafe_color,
                            alpha=0.5, linestyle='--')
    ax.add_patch(arrow5)

    arrow6 = FancyArrowPatch((12.5, y_mode1 - 2), (10.3, y_failsafe + 0.5),
                            arrowstyle='->', lw=2, color=failsafe_color,
                            alpha=0.5, linestyle='--')
    ax.add_patch(arrow6)

    # ===== KEY INSIGHTS (Bottom) =====
    # Key insights box
    insights_box = FancyBboxPatch((4.5, -0.5), 7, 0.6,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='lightyellow',
                                 linewidth=2, alpha=0.7)
    ax.add_patch(insights_box)

    ax.text(8, -0.1, '🔑 Key Insight: Peer-comparison (Mode 0) inverts at 35% BFT (100% FPR) |',
           ha='center', fontsize=10, fontweight='bold')
    ax.text(8, -0.35, 'Ground truth (Mode 1) achieves perfect discrimination (0% FPR)',
           ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_path = '/tmp/system_architecture_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved architecture diagram: {output_path}")

    output_path_svg = '/tmp/system_architecture_diagram.svg'
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved vector graphic: {output_path_svg}")

    plt.close()


def create_simplified_architecture():
    """
    Create simplified architecture showing just the core components.
    """

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    fig.suptitle(
        'Byzantine Detection Architecture: Ground Truth vs Peer-Comparison',
        fontsize=16,
        fontweight='bold',
        y=0.96
    )

    # Color scheme
    honest_color = '#3498db'
    byzantine_color = '#e74c3c'
    peer_color = '#f39c12'
    ground_truth_color = '#27ae60'

    # ===== CLIENTS (Top) =====
    y_client = 8.5

    # Honest client
    honest_box = FancyBboxPatch((1, y_client), 3, 1,
                               boxstyle="round,pad=0.1",
                               edgecolor=honest_color, facecolor=honest_color,
                               linewidth=2, alpha=0.3)
    ax.add_patch(honest_box)
    ax.text(2.5, y_client + 0.5, 'Honest Client', ha='center',
           fontsize=11, fontweight='bold')
    ax.text(2.5, y_client + 0.15, 'Local Data: Unique', ha='center', fontsize=8)

    # Byzantine client
    byz_box = FancyBboxPatch((10, y_client), 3, 1,
                            boxstyle="round,pad=0.1",
                            edgecolor=byzantine_color, facecolor=byzantine_color,
                            linewidth=2, alpha=0.3)
    ax.add_patch(byz_box)
    ax.text(11.5, y_client + 0.5, 'Byzantine Client', ha='center',
           fontsize=11, fontweight='bold', color=byzantine_color)
    ax.text(11.5, y_client + 0.15, 'Attack: Sign Flip', ha='center',
           fontsize=8, color='red')

    # Gradients
    ax.text(2.5, y_client - 0.4, '∇θ (honest)', ha='center', fontsize=10,
           style='italic', family='serif')
    ax.text(11.5, y_client - 0.4, '∇θ (Byzantine)', ha='center', fontsize=10,
           style='italic', family='serif', color=byzantine_color)

    # ===== DETECTION METHODS (Middle) =====
    y_detect = 5.5

    # Peer-comparison
    peer_box = FancyBboxPatch((0.5, y_detect - 0.5), 5.5, 2.5,
                             boxstyle="round,pad=0.15",
                             edgecolor=peer_color, facecolor='white',
                             linewidth=3)
    ax.add_patch(peer_box)

    ax.text(3.25, y_detect + 1.7, 'Peer-Comparison', ha='center',
           fontsize=13, fontweight='bold', color=peer_color)
    ax.text(3.25, y_detect + 1.2, '(Mode 0)', ha='center', fontsize=10, style='italic')

    ax.text(3.25, y_detect + 0.7, 'Compare to peers:', ha='center', fontsize=9,
           fontweight='bold')
    ax.text(3.25, y_detect + 0.3, '• Cosine similarity', ha='center', fontsize=8)
    ax.text(3.25, y_detect, '• Magnitude distribution', ha='center', fontsize=8)

    ax.text(3.25, y_detect - 0.5, '❌ Result: 100% FPR', ha='center',
           fontsize=10, fontweight='bold', color='red')

    # Arrow from clients to peer
    arrow1 = FancyArrowPatch((2.5, y_client - 0.7), (2, y_detect + 2.2),
                            arrowstyle='->', lw=2, color='gray')
    ax.add_patch(arrow1)
    arrow2 = FancyArrowPatch((11.5, y_client - 0.7), (4.5, y_detect + 2.2),
                            arrowstyle='->', lw=2, color='gray')
    ax.add_patch(arrow2)

    # Ground truth
    gt_box = FancyBboxPatch((8, y_detect - 0.5), 5.5, 2.5,
                           boxstyle="round,pad=0.15",
                           edgecolor=ground_truth_color, facecolor='white',
                           linewidth=3)
    ax.add_patch(gt_box)

    ax.text(10.75, y_detect + 1.7, 'Ground Truth', ha='center',
           fontsize=13, fontweight='bold', color=ground_truth_color)
    ax.text(10.75, y_detect + 1.2, '(Mode 1 - PoGQ)', ha='center',
           fontsize=10, style='italic')

    ax.text(10.75, y_detect + 0.7, 'Validation loss:', ha='center',
           fontsize=9, fontweight='bold')
    ax.text(10.75, y_detect + 0.3, '• Measure quality', ha='center', fontsize=8)
    ax.text(10.75, y_detect, '• Adaptive threshold', ha='center', fontsize=8)

    ax.text(10.75, y_detect - 0.5, '✅ Result: 0% FPR', ha='center',
           fontsize=10, fontweight='bold', color='green')

    # Arrow from clients to ground truth
    arrow3 = FancyArrowPatch((2.5, y_client - 0.7), (9.5, y_detect + 2.2),
                            arrowstyle='->', lw=2, color='gray')
    ax.add_patch(arrow3)
    arrow4 = FancyArrowPatch((11.5, y_client - 0.7), (12, y_detect + 2.2),
                            arrowstyle='->', lw=2, color='gray')
    ax.add_patch(arrow4)

    # Validation set for ground truth
    val_box = FancyBboxPatch((9.5, y_detect - 1.2), 2.5, 0.5,
                            boxstyle="round,pad=0.05",
                            edgecolor='purple', facecolor='purple',
                            linewidth=1.5, alpha=0.2)
    ax.add_patch(val_box)
    ax.text(10.75, y_detect - 0.95, 'Server Val Set', ha='center',
           fontsize=8, fontweight='bold')

    # ===== PROBLEM & SOLUTION (Bottom) =====
    y_bottom = 2.5

    # Problem box
    problem_box = FancyBboxPatch((0.5, y_bottom), 6, 1.5,
                                boxstyle="round,pad=0.1",
                                edgecolor='red', facecolor='mistyrose',
                                linewidth=2)
    ax.add_patch(problem_box)

    ax.text(3.5, y_bottom + 1.1, '❌ Problem: Heterogeneous Data',
           ha='center', fontsize=11, fontweight='bold', color='red')
    ax.text(3.5, y_bottom + 0.7, 'Honest clients have diverse gradients',
           ha='center', fontsize=9)
    ax.text(3.5, y_bottom + 0.4, '→ Low peer similarity', ha='center',
           fontsize=8, style='italic')
    ax.text(3.5, y_bottom + 0.1, '→ Peer-comparison flags them', ha='center',
           fontsize=8, style='italic', color='red')

    # Solution box
    solution_box = FancyBboxPatch((7.5, y_bottom), 6, 1.5,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='green', facecolor='honeydew',
                                 linewidth=2)
    ax.add_patch(solution_box)

    ax.text(10.5, y_bottom + 1.1, '✅ Solution: External Reference',
           ha='center', fontsize=11, fontweight='bold', color='green')
    ax.text(10.5, y_bottom + 0.7, 'Measure against actual task', ha='center',
           fontsize=9)
    ax.text(10.5, y_bottom + 0.4, '→ Quality = loss improvement', ha='center',
           fontsize=8, style='italic')
    ax.text(10.5, y_bottom + 0.1, '→ Perfect discrimination', ha='center',
           fontsize=8, style='italic', color='green')

    # ===== KEY FINDING (Very Bottom) =====
    finding_box = FancyBboxPatch((2, 0.5), 10, 1.2,
                                boxstyle="round,pad=0.15",
                                edgecolor='black', facecolor='lightyellow',
                                linewidth=3)
    ax.add_patch(finding_box)

    ax.text(7, 1.3, '🔬 Empirical Finding at 35% BFT:',
           ha='center', fontsize=12, fontweight='bold')
    ax.text(7, 0.95, 'Peer-Comparison: 100% Detection BUT 100% FPR (ALL honest flagged) ❌',
           ha='center', fontsize=10)
    ax.text(7, 0.65, 'Ground Truth: 100% Detection AND 0% FPR (Perfect) ✅',
           ha='center', fontsize=10)

    plt.tight_layout()

    # Save
    output_path = '/tmp/simplified_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved simplified architecture: {output_path}")

    output_path_svg = '/tmp/simplified_architecture.svg'
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight', facecolor='white')
    print(f"✓ Saved vector graphic: {output_path_svg}")

    plt.close()


if __name__ == "__main__":
    print("Creating system architecture diagrams...")
    print()

    print("Architecture Diagram 1: Complete System")
    create_architecture_diagram()
    print()

    print("Architecture Diagram 2: Simplified View")
    create_simplified_architecture()
    print()

    print("✅ All architecture diagrams created successfully!")
    print()
    print("Output files:")
    print("  - /tmp/system_architecture_diagram.png/svg")
    print("  - /tmp/simplified_architecture.png/svg")
