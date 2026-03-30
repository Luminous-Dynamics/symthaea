# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Utilities for combining heuristic and hybrid Byzantine detection signals.
"""

from typing import Optional


def combine_detection_votes(
    base_flag: bool,
    hybrid_score: Optional[float],
    threshold: float,
) -> bool:
    """
    Merge a legacy detection flag with the hybrid ensemble confidence.

    Args:
        base_flag: Existing heuristic/PoGQ classification.
        hybrid_score: Ensemble confidence from HybridByzantineDetector (0..1)
            or None when the hybrid detector is disabled.
        threshold: Minimum ensemble confidence required to flag a node.
    """
    if hybrid_score is None:
        return base_flag
    if hybrid_score >= threshold:
        return True
    return base_flag
