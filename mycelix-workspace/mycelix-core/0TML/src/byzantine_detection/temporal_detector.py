# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Temporal Consistency Detector

Tracks gradient behavior over multiple rounds to distinguish between:
- Honest nodes with label skew: Consistent gradient patterns over time
- Byzantine attackers: Erratic, inconsistent behavior across rounds

Week 3 Component - Multi-Signal Hybrid Detection
"""

from typing import Dict, List, Optional
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class TemporalStatistics:
    """Statistics about a node's temporal behavior"""

    node_id: int
    num_observations: int

    # Cosine similarity statistics over time
    mean_cosine: float
    std_cosine: float
    min_cosine: float
    max_cosine: float

    # Gradient magnitude statistics over time
    mean_magnitude: float
    std_magnitude: float
    min_magnitude: float
    max_magnitude: float

    # Consistency metrics
    cosine_variance: float
    magnitude_variance: float

    # Temporal confidence
    temporal_confidence: float  # [0, 1] where 1 = highly confident Byzantine


class TemporalConsistencyDetector:
    """
    Track gradient behavior over multiple rounds to detect erratic Byzantine patterns.

    Hypothesis:
    - Honest nodes with label skew: Consistently produce similar gradients
      (same local data distribution) with stable cosine similarity and magnitude
    - Byzantine attackers: Show inconsistent behavior across rounds with
      high variance in cosine similarities and/or gradient magnitudes

    Detection Method:
    - Track rolling window of gradient characteristics per node
    - Compute variance in cosine similarities and magnitudes
    - High variance → Likely Byzantine (erratic behavior)
    - Low variance → Likely honest (consistent behavior)
    """

    def __init__(
        self,
        window_size: int = 5,
        cosine_variance_threshold: float = 0.1,
        magnitude_variance_threshold: float = 0.5,
        min_observations: int = 3
    ):
        """
        Args:
            window_size: Number of recent rounds to track (default: 5)
            cosine_variance_threshold: Variance threshold for cosine similarities
            magnitude_variance_threshold: Relative variance threshold for magnitudes
            min_observations: Minimum observations before making confident decisions
        """
        self.window_size = window_size
        self.cosine_variance_threshold = cosine_variance_threshold
        self.magnitude_variance_threshold = magnitude_variance_threshold
        self.min_observations = min_observations

        # History tracking per node
        self.gradient_history: Dict[int, List[torch.Tensor]] = {}
        self.cosine_history: Dict[int, List[float]] = {}
        self.magnitude_history: Dict[int, List[float]] = {}

    def update(
        self,
        node_id: int,
        gradient: torch.Tensor,
        mean_cosine_similarity: float
    ) -> None:
        """
        Update temporal history for a node.

        Args:
            node_id: Node identifier
            gradient: Current gradient tensor
            mean_cosine_similarity: Mean cosine similarity to other gradients
        """
        # Initialize history if first observation
        if node_id not in self.gradient_history:
            self.gradient_history[node_id] = []
            self.cosine_history[node_id] = []
            self.magnitude_history[node_id] = []

        # Append current observation
        self.gradient_history[node_id].append(gradient.detach().clone())
        self.cosine_history[node_id].append(mean_cosine_similarity)
        self.magnitude_history[node_id].append(torch.norm(gradient).item())

        # Maintain rolling window (keep most recent observations)
        if len(self.gradient_history[node_id]) > self.window_size:
            self.gradient_history[node_id].pop(0)
            self.cosine_history[node_id].pop(0)
            self.magnitude_history[node_id].pop(0)

    def compute_temporal_confidence(self, node_id: int) -> float:
        """
        Compute confidence that node is Byzantine based on temporal patterns.

        High variance in cosine similarities and/or magnitudes indicates
        erratic behavior characteristic of Byzantine attacks.

        Args:
            node_id: Node identifier

        Returns:
            Confidence score [0, 1] where 1 = highly confident Byzantine,
            0 = highly confident honest, 0.5 = neutral/insufficient data
        """
        if node_id not in self.cosine_history:
            return 0.5  # No history - neutral

        if len(self.cosine_history[node_id]) < self.min_observations:
            return 0.5  # Not enough observations - abstain

        cosines = self.cosine_history[node_id]
        magnitudes = self.magnitude_history[node_id]

        # Measure consistency via variance
        cosine_variance = np.var(cosines) if len(cosines) > 1 else 0.0
        magnitude_variance = np.var(magnitudes) if len(magnitudes) > 1 else 0.0

        # High variance = erratic behavior = likely Byzantine
        # Low variance = consistent behavior = likely honest

        # Normalize cosine variance to [0, 1] confidence
        cosine_confidence = min(1.0, cosine_variance / self.cosine_variance_threshold)

        # Normalize magnitude variance to [0, 1] confidence
        # Use relative variance (coefficient of variation squared)
        mean_mag = np.mean(magnitudes)
        if mean_mag > 0:
            relative_magnitude_var = magnitude_variance / (mean_mag ** 2)
            magnitude_confidence = min(
                1.0,
                relative_magnitude_var / self.magnitude_variance_threshold
            )
        else:
            magnitude_confidence = 0.5  # Cannot determine

        # Average the two signals
        temporal_confidence = (cosine_confidence + magnitude_confidence) / 2.0

        return temporal_confidence

    def get_temporal_statistics(self, node_id: int) -> Optional[TemporalStatistics]:
        """
        Get detailed temporal statistics for a node.

        Args:
            node_id: Node identifier

        Returns:
            TemporalStatistics object or None if no history
        """
        if node_id not in self.cosine_history:
            return None

        cosines = self.cosine_history[node_id]
        magnitudes = self.magnitude_history[node_id]

        if len(cosines) == 0:
            return None

        cosine_variance = np.var(cosines) if len(cosines) > 1 else 0.0
        magnitude_variance = np.var(magnitudes) if len(magnitudes) > 1 else 0.0

        return TemporalStatistics(
            node_id=node_id,
            num_observations=len(cosines),
            mean_cosine=np.mean(cosines),
            std_cosine=np.std(cosines),
            min_cosine=np.min(cosines),
            max_cosine=np.max(cosines),
            mean_magnitude=np.mean(magnitudes),
            std_magnitude=np.std(magnitudes),
            min_magnitude=np.min(magnitudes),
            max_magnitude=np.max(magnitudes),
            cosine_variance=cosine_variance,
            magnitude_variance=magnitude_variance,
            temporal_confidence=self.compute_temporal_confidence(node_id)
        )

    def reset(self) -> None:
        """Clear all temporal history."""
        self.gradient_history.clear()
        self.cosine_history.clear()
        self.magnitude_history.clear()

    def reset_node(self, node_id: int) -> None:
        """Clear temporal history for a specific node."""
        if node_id in self.gradient_history:
            del self.gradient_history[node_id]
        if node_id in self.cosine_history:
            del self.cosine_history[node_id]
        if node_id in self.magnitude_history:
            del self.magnitude_history[node_id]
