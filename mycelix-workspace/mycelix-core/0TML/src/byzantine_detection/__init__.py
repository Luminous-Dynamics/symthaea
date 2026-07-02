# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Detection Module - Dimension-Aware & Hybrid Detection Strategies

This module provides adaptive Byzantine detection for federated learning
that adjusts its strategy based on gradient characteristics.

Week 1-2: Dimension-aware and label-skew-aware detection
Week 3: Multi-signal hybrid detection with ensemble voting
"""

# Week 1-2: Gradient analysis and label skew detection
from .gradient_analyzer import (
    GradientDimensionalityAnalyzer,
    GradientProfile,
)

# Week 3: Hybrid detection components
from .temporal_detector import (
    TemporalConsistencyDetector,
    TemporalStatistics,
)
from .magnitude_detector import (
    MagnitudeDistributionDetector,
    MagnitudeStatistics,
)
from .ensemble_voting import (
    EnsembleVotingSystem,
    EnsembleDecision,
)
from .hybrid_detector import (
    HybridByzantineDetector,
)

__all__ = [
    # Week 1-2 exports
    "GradientDimensionalityAnalyzer",
    "GradientProfile",
    # Week 3 exports
    "TemporalConsistencyDetector",
    "TemporalStatistics",
    "MagnitudeDistributionDetector",
    "MagnitudeStatistics",
    "EnsembleVotingSystem",
    "EnsembleDecision",
    "HybridByzantineDetector",
]
