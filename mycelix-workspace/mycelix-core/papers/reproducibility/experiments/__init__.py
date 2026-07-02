# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mycelix Paper Reproducibility - Experiment Modules

This package contains scripts to reproduce all experimental results
from the Mycelix MLSys 2026 paper.

Available experiments:
- table1_byzantine_detection: Detection rate vs Byzantine ratio
- table2_latency: Aggregation latency comparison
- figure1_detection_vs_ratio: Detection rate line plot
- figure2_convergence: Model accuracy convergence
- figure3_scalability: Node scaling analysis
"""

from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
REPRODUCIBILITY_DIR = EXPERIMENTS_DIR.parent

__all__ = [
    'table1_byzantine_detection',
    'table2_latency',
    'figure1_detection_vs_ratio',
    'figure2_convergence',
    'figure3_scalability',
]
