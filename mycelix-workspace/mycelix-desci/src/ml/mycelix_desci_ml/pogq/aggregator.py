# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Gradient Aggregator

Advanced aggregation strategies for federated learning.
"""

import numpy as np
from typing import List, Optional
from .validator import GradientUpdate, QualityScore


class GradientAggregator:
    """Aggregates gradients from multiple participants"""

    def __init__(self, strategy: str = "weighted_average"):
        """Initialize aggregator

        Args:
            strategy: Aggregation strategy ("weighted_average", "median", "trimmed_mean")
        """
        self.strategy = strategy

    def aggregate(
        self,
        gradients: List[GradientUpdate],
        scores: Optional[List[QualityScore]] = None,
    ) -> np.ndarray:
        """Aggregate gradients

        Args:
            gradients: List of gradient updates
            scores: Optional quality scores for weighting

        Returns:
            Aggregated gradient
        """
        if self.strategy == "weighted_average":
            return self._weighted_average(gradients, scores)
        elif self.strategy == "median":
            return self._median_aggregation(gradients)
        elif self.strategy == "trimmed_mean":
            return self._trimmed_mean(gradients)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_average(
        self,
        gradients: List[GradientUpdate],
        scores: Optional[List[QualityScore]],
    ) -> np.ndarray:
        """Weighted average aggregation"""
        if scores is None:
            # Equal weights
            return np.mean([g.gradients for g in gradients], axis=0)

        total_weight = sum(s.score for s in scores)
        aggregated = np.zeros_like(gradients[0].gradients)

        for grad, score in zip(gradients, scores):
            weight = score.score / total_weight
            aggregated += weight * grad.gradients

        return aggregated

    def _median_aggregation(self, gradients: List[GradientUpdate]) -> np.ndarray:
        """Median aggregation (Byzantine-resistant)"""
        return np.median([g.gradients for g in gradients], axis=0)

    def _trimmed_mean(
        self, gradients: List[GradientUpdate], trim_ratio: float = 0.2
    ) -> np.ndarray:
        """Trimmed mean aggregation"""
        grad_arrays = np.array([g.gradients for g in gradients])
        n_trim = int(len(gradients) * trim_ratio)

        # Sort and trim extremes
        sorted_grads = np.sort(grad_arrays, axis=0)
        trimmed = sorted_grads[n_trim : len(gradients) - n_trim]

        return np.mean(trimmed, axis=0)
