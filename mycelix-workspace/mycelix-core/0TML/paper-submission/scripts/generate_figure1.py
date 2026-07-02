#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate Figure 1: Mode 1 Performance Across BFT Levels
Dual-axis line chart showing detection rate and FPR across Byzantine ratios
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Table 1
bft_ratios = [20, 25, 30, 35, 40, 45, 50]
detection_rates = [100, 100, 100, 100, 100, 100, 100]
fprs = [0, 0, 0, 7.7, 7.7, 9.1, 10.0]
detection_stds = [0, 0, 0, 0, 0, 0, 0]  # Perfect consistency
fpr_stds = [0, 0, 0, 4.3, 4.3, 4.3, 4.3]  # From multi-seed validation

fig, ax1 = plt.subplots(figsize=(7, 5))

# Detection rate (left axis)
color = 'tab:red'
ax1.set_xlabel('Byzantine Ratio (%)', fontsize=12)
ax1.set_ylabel('Detection Rate (%)', color=color, fontsize=12)
line1 = ax1.plot(bft_ratios, detection_rates, 'o-', color=color, linewidth=2,
                 markersize=8, label='Detection Rate')
ax1.fill_between(bft_ratios,
                 np.array(detection_rates) - detection_stds,
                 np.array(detection_rates) + detection_stds,
                 color=color, alpha=0.2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([0, 110])
ax1.grid(axis='y', alpha=0.3)

# FPR (right axis)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('False Positive Rate (%)', color=color, fontsize=12)
line2 = ax2.plot(bft_ratios, fprs, 's-', color=color, linewidth=2,
                 markersize=8, label='False Positive Rate')
ax2.fill_between(bft_ratios,
                 np.array(fprs) - fpr_stds,
                 np.array(fprs) + fpr_stds,
                 color=color, alpha=0.2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim([0, 15])

# 33% barrier line
ax1.axvline(x=33, color='gray', linestyle='--', linewidth=1.5,
            label='33% Peer-Comparison Ceiling')

# 10% FPR target line
ax2.axhline(y=10, color='gray', linestyle=':', linewidth=1.5,
            label='10% FPR Target')

# Legend
lines = line1 + line2 + [plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5),
                          plt.Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5)]
labels = ['Detection Rate (100%)', 'False Positive Rate',
          '33% Ceiling', '10% Target']
ax1.legend(lines, labels, loc='center left', fontsize=10)

plt.title('Mode 1 Performance: Breaking the 33% Barrier', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/mode1_performance_bft.pdf', dpi=300, bbox_inches='tight')
print("✅ Figure 1 generated: figures/mode1_performance_bft.pdf")
