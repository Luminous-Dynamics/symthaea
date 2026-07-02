#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
BOBA: Label-Skew Aware Byzantine-Robust Aggregation
====================================================

BOBA accounts for non-IID data distributions by considering client class
histograms. Clients with similar class distributions are weighted higher.

Reference:
"Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing"

Author: Luminous Dynamics
Date: November 8, 2025
Version: Tier-2 (for non-IID ablation study)
"""

import numpy as np
from typing import List, Optional
from scipy.spatial.distance import jensenshannon
import logging

logger = logging.getLogger(__name__)


class BOBA:
    """
    BOBA: Label-skew aware robust aggregation

    Uses Jensen-Shannon divergence on client class histograms to compute
    similarity weights. Byzantine-robust for heterogeneous (non-IID) data.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        normalize_weights: bool = True
    ):
        """
        Args:
            similarity_threshold: Minimum JS similarity to include client
            normalize_weights: Whether to normalize final weights
        """
        self.similarity_threshold = similarity_threshold
        self.normalize_weights = normalize_weights

    def _compute_js_similarity(
        self,
        hist1: np.ndarray,
        hist2: np.ndarray
    ) -> float:
        """
        Compute Jensen-Shannon similarity (1 - JS divergence)

        Args:
            hist1, hist2: Probability distributions (class histograms)

        Returns:
            JS similarity in [0, 1]
        """
        # Ensure normalized
        hist1 = hist1 / (np.sum(hist1) + 1e-8)
        hist2 = hist2 / (np.sum(hist2) + 1e-8)

        # JS divergence
        js_div = jensenshannon(hist1, hist2, base=2)

        # Convert to similarity
        js_sim = 1.0 - js_div

        return float(js_sim)

    def compute_similarity_matrix(
        self,
        client_class_histograms: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute pairwise JS similarity matrix

        Args:
            client_class_histograms: List of class probability distributions

        Returns:
            n × n similarity matrix
        """
        n = len(client_class_histograms)
        similarity = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                similarity[i, j] = self._compute_js_similarity(
                    client_class_histograms[i],
                    client_class_histograms[j]
                )

        return similarity

    def compute_boba_weights(
        self,
        client_class_histograms: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute BOBA weights based on JS similarity

        Clients with higher average similarity to others get higher weights.

        Returns:
            Weight vector (one per client)
        """
        n = len(client_class_histograms)

        # Compute similarity matrix
        similarity = self.compute_similarity_matrix(client_class_histograms)

        # Average similarity to all other clients
        weights = np.mean(similarity, axis=1)

        # Threshold: reject clients below similarity threshold
        weights = np.where(weights >= self.similarity_threshold, weights, 0.0)

        # Normalize if requested
        if self.normalize_weights and np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        return weights

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        client_class_histograms: Optional[List[np.ndarray]] = None,
        **context
    ) -> np.ndarray:
        """
        Aggregate with BOBA label-skew aware weighting

        Args:
            client_updates: List of client gradients
            client_class_histograms: List of class probability distributions
                                    (required for BOBA)
            **context: Additional context (ignored)

        Returns:
            BOBA weighted aggregated gradient
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        if client_class_histograms is None:
            logger.warning("BOBA: No class histograms provided, falling back to mean")
            return np.mean(client_updates, axis=0)

        if len(client_class_histograms) != len(client_updates):
            raise ValueError("Mismatch: {} gradients, {} histograms".format(
                len(client_updates), len(client_class_histograms)
            ))

        # Compute BOBA weights
        weights = self.compute_boba_weights(client_class_histograms)

        # If all weights zero, fall back to mean
        if np.sum(weights) < 1e-8:
            logger.warning("BOBA: All clients rejected, using mean")
            return np.mean(client_updates, axis=0)

        # Weighted aggregation
        aggregated = np.zeros_like(client_updates[0])
        for weight, gradient in zip(weights, client_updates):
            aggregated += weight * gradient

        logger.info(f"BOBA: Aggregated {np.sum(weights > 0)}/{len(client_updates)} clients "
                   f"(weight range: [{np.min(weights):.3f}, {np.max(weights):.3f}])")

        return aggregated

    def explain(self):
        """Return configuration"""
        return {
            "defense": "boba",
            "similarity_threshold": self.similarity_threshold,
            "normalize_weights": self.normalize_weights
        }
