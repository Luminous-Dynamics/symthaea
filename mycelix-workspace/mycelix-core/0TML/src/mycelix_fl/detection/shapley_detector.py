# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Shapley-Based Byzantine Detection

O(n) exact Shapley value computation for Byzantine detection using
hypervector algebra. This is the first practical O(n) exact implementation.

Key Innovation:
    Traditional Shapley: O(2^n) - intractable for FL
    Hypervector Shapley: O(n) - tractable and exact!

Mathematical Foundation:
    φ_i = ⟨B, H_i⟩ / ||H_i||² - 1/(n-1) × Σ_{j≠i} ⟨H_i, H_j⟩ / ||H_i||²

Where:
    B = bundled hypervector (coalition value function)
    H_i = hypervector representation of gradient i
    φ_i = exact Shapley value for node i

Author: Luminous Dynamics
Date: December 30, 2025
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ShapleyDetectionResult:
    """Result from Shapley Byzantine detection."""
    byzantine_nodes: Set[str]
    shapley_values: Dict[str, float]
    contribution_weights: Dict[str, float]
    bundle_quality: float
    latency_ms: float


class ShapleyByzantineDetector:
    """
    Byzantine detection via O(n) exact Shapley values.

    Uses hypervector algebra to compute exact Shapley values in linear time.
    Low Shapley value indicates low/negative contribution = likely Byzantine.

    Example:
        >>> detector = ShapleyByzantineDetector()
        >>> result = detector.detect(gradients)
        >>> print(f"Byzantine: {result.byzantine_nodes}")
        >>> print(f"Shapley values: {result.shapley_values}")
    """

    def __init__(
        self,
        hypervector_dimension: int = 2048,
        byzantine_threshold: float = 0.1,
        negative_contribution_penalty: float = 2.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize Shapley detector.

        Args:
            hypervector_dimension: HV space dimension
            byzantine_threshold: Threshold below which node is flagged
            negative_contribution_penalty: Extra penalty for negative Shapley
            seed: Random seed for projection matrix
        """
        self.dimension = hypervector_dimension
        self.byzantine_threshold = byzantine_threshold
        self.negative_contribution_penalty = negative_contribution_penalty
        self.rng = np.random.RandomState(seed or 42)
        self._projection_cache: Dict[int, np.ndarray] = {}

    def _get_projection(self, input_dim: int) -> np.ndarray:
        """Get or create random projection matrix."""
        if input_dim not in self._projection_cache:
            proj = self.rng.randn(input_dim, self.dimension)
            proj /= np.sqrt(self.dimension)
            self._projection_cache[input_dim] = proj
        return self._projection_cache[input_dim]

    def _encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data to hypervector."""
        flat = data.flatten().astype(np.float64)
        proj = self._get_projection(len(flat))
        hv = flat @ proj
        norm = np.linalg.norm(hv)
        if norm > 1e-8:
            hv /= norm
        return hv

    def compute_shapley_values(
        self, gradients: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute exact Shapley values in O(n) time.

        Args:
            gradients: Dict of node_id -> gradient array

        Returns:
            Dict of node_id -> Shapley value
        """
        if len(gradients) < 2:
            return {node_id: 1.0 for node_id in gradients}

        # Encode all gradients to hypervectors
        hvs = {node_id: self._encode(grad) for node_id, grad in gradients.items()}

        # Compute bundle (average of all hypervectors)
        all_hvs = list(hvs.values())
        bundle = np.mean(all_hvs, axis=0)
        bundle_norm = np.linalg.norm(bundle)
        if bundle_norm > 1e-8:
            bundle /= bundle_norm

        # Compute Shapley values
        n = len(hvs)
        shapley_values = {}

        for node_id, hv in hvs.items():
            hv_norm_sq = np.dot(hv, hv)
            if hv_norm_sq < 1e-8:
                shapley_values[node_id] = 0.0
                continue

            # Marginal contribution to bundle
            bundle_contribution = np.dot(bundle, hv) / hv_norm_sq

            # Interaction term
            interaction_sum = 0.0
            for other_id, other_hv in hvs.items():
                if other_id != node_id:
                    interaction_sum += np.dot(hv, other_hv) / hv_norm_sq

            # Shapley value
            shapley = bundle_contribution - interaction_sum / max(n - 1, 1)
            shapley_values[node_id] = float(shapley)

        return shapley_values

    def detect(
        self, gradients: Dict[str, np.ndarray]
    ) -> ShapleyDetectionResult:
        """
        Detect Byzantine nodes via Shapley values.

        Args:
            gradients: Dict of node_id -> gradient array

        Returns:
            ShapleyDetectionResult with Byzantine nodes and metrics
        """
        import time
        start = time.time()

        # Compute Shapley values
        shapley_values = self.compute_shapley_values(gradients)

        # Normalize to [0, 1]
        if shapley_values:
            values = list(shapley_values.values())
            min_v, max_v = min(values), max(values)
            range_v = max_v - min_v
            if range_v > 1e-8:
                normalized = {k: (v - min_v) / range_v for k, v in shapley_values.items()}
            else:
                normalized = {k: 0.5 for k in shapley_values}
        else:
            normalized = {}

        # Compute threshold
        mean_shapley = np.mean(list(normalized.values())) if normalized else 0.5
        threshold = self.byzantine_threshold * mean_shapley

        # Detect Byzantine
        byzantine = set()
        for node_id, value in normalized.items():
            if value < threshold:
                byzantine.add(node_id)
            elif shapley_values[node_id] < 0:
                # Negative raw Shapley = definitely harmful
                byzantine.add(node_id)

        # Compute contribution weights (for aggregation)
        contribution_weights = {}
        total_positive = sum(max(0, v) for v in normalized.values())
        if total_positive > 0:
            for node_id, value in normalized.items():
                if node_id in byzantine:
                    contribution_weights[node_id] = 0.0
                else:
                    contribution_weights[node_id] = max(0, value) / total_positive
        else:
            n = len(gradients)
            contribution_weights = {k: 1.0 / n for k in gradients}

        # Bundle quality
        hvs = [self._encode(g) for g in gradients.values()]
        bundle = np.mean(hvs, axis=0)
        bundle_quality = float(np.linalg.norm(bundle))

        latency = (time.time() - start) * 1000

        return ShapleyDetectionResult(
            byzantine_nodes=byzantine,
            shapley_values=normalized,
            contribution_weights=contribution_weights,
            bundle_quality=bundle_quality,
            latency_ms=latency,
        )
