# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Analysis Modules for Non-IID Validation Experiments

This package provides analysis tools for evaluating Byzantine detection
performance under Non-IID data distributions:

- convergence: Model convergence analysis under Non-IID conditions
- detection: Byzantine detection accuracy analysis
- fairness: Fairness metrics for data-imbalanced scenarios
"""

from .convergence import analyze_convergence, ConvergenceAnalyzer
from .detection import analyze_detection, DetectionAnalyzer
from .fairness import analyze_fairness, FairnessAnalyzer

__all__ = [
    "analyze_convergence",
    "ConvergenceAnalyzer",
    "analyze_detection",
    "DetectionAnalyzer",
    "analyze_fairness",
    "FairnessAnalyzer",
]
