#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Krum, Multi-Krum, and Bulyan: Distance-Based Byzantine Defense
===============================================================

Krum: Select the gradient that is closest to its neighbors
Multi-Krum: Select top-k closest gradients and average
Bulyan: Krum selection + trimmed mean aggregation

Reference:
"Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (Krum)
"The Hidden Vulnerability of Distributed Learning in Byzantium" (Bulyan)

Author: Luminous Dynamics
Date: November 8, 2025
"""

import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class Krum:
    """
    Krum: Select gradient with minimum sum of squared distances to neighbors

    Byzantine-robust for f < n/2 - 1 where n is total clients and f is Byzantine.
    """

    def __init__(self, f: int = 2):
        """
        Args:
            f: Maximum number of Byzantine clients (default 2)
        """
        self.f = f

    def _compute_distances(self, gradients: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise squared L2 distances"""
        n = len(gradients)
        X = np.vstack(gradients)

        # Pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sum((X[i] - X[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def select_krum_index(self, gradients: List[np.ndarray]) -> int:
        """
        Select Krum gradient index

        Returns:
            Index of selected gradient
        """
        n = len(gradients)
        m = n - self.f - 2  # Number of neighbors to consider

        if m <= 0:
            logger.warning(f"Krum: Too many Byzantine (f={self.f}, n={n}), using median")
            return n // 2

        # Compute distances
        distances = self._compute_distances(gradients)

        # For each gradient, sum distances to m closest neighbors
        scores = np.zeros(n)
        for i in range(n):
            # Sort distances for this gradient
            sorted_distances = np.sort(distances[i])
            # Sum m closest (excluding self at index 0)
            scores[i] = np.sum(sorted_distances[1:m+1])

        # Select minimum score
        selected = int(np.argmin(scores))
        return selected

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """
        Aggregate via Krum selection

        Args:
            client_updates: List of client gradients
            **context: Additional context (ignored)

        Returns:
            Selected Krum gradient
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        if len(client_updates) == 1:
            return client_updates[0]

        selected_idx = self.select_krum_index(client_updates)
        return client_updates[selected_idx]

    def explain(self):
        """Return configuration"""
        return {
            "defense": "krum",
            "f": self.f
        }


class MultiKrum(Krum):
    """
    Multi-Krum: Select top-k gradients by Krum score and average

    More robust than single Krum by averaging multiple selections.
    """

    def __init__(self, f: int = 2, k: int = 3):
        """
        Args:
            f: Maximum number of Byzantine clients
            k: Number of gradients to select
        """
        super().__init__(f)
        self.k = k

    def select_multikrum_indices(self, gradients: List[np.ndarray]) -> List[int]:
        """
        Select top-k gradients by Krum score

        Returns:
            List of k selected indices
        """
        n = len(gradients)
        m = n - self.f - 2

        if m <= 0:
            return list(range(min(self.k, n)))

        # Compute distances
        distances = self._compute_distances(gradients)

        # Compute scores
        scores = np.zeros(n)
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            scores[i] = np.sum(sorted_distances[1:m+1])

        # Select k minimum scores
        selected = np.argsort(scores)[:self.k]
        return list(selected)

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """
        Aggregate via Multi-Krum

        Args:
            client_updates: List of client gradients
            **context: Additional context (ignored)

        Returns:
            Average of selected gradients
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        if len(client_updates) == 1:
            return client_updates[0]

        k = min(self.k, len(client_updates))
        selected_indices = self.select_multikrum_indices(client_updates)[:k]

        selected_gradients = [client_updates[i] for i in selected_indices]
        return np.mean(selected_gradients, axis=0)

    def explain(self):
        """Return configuration"""
        return {
            "defense": "multi_krum",
            "f": self.f,
            "k": self.k
        }


class Bulyan(MultiKrum):
    """
    Bulyan: Multi-Krum selection + trimmed mean aggregation

    Most robust combination: Krum selection eliminates outliers,
    then trimmed mean provides final robustness.

    Byzantine-robust for f < n/4.
    """

    def __init__(self, f: int = 2, trim_ratio: float = 0.25):
        """
        Args:
            f: Maximum number of Byzantine clients
            trim_ratio: Fraction to trim after selection (default 0.25)
        """
        # Select 2f+1 gradients
        k = 2 * f + 1
        super().__init__(f, k)
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """
        Aggregate via Bulyan (Multi-Krum + Trimmed Mean)

        Args:
            client_updates: List of client gradients
            **context: Additional context (ignored)

        Returns:
            Bulyan aggregated gradient
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        if len(client_updates) == 1:
            return client_updates[0]

        # Step 1: Multi-Krum selection
        selected_indices = self.select_multikrum_indices(client_updates)
        selected_gradients = [client_updates[i] for i in selected_indices]

        # Step 2: Trimmed mean
        n_selected = len(selected_gradients)
        X = np.vstack(selected_gradients)

        n_trim = int(n_selected * self.trim_ratio)
        if n_trim == 0:
            return np.mean(X, axis=0)

        # Sort each coordinate
        X_sorted = np.sort(X, axis=0)

        # Trim and average
        X_trimmed = X_sorted[n_trim:-n_trim] if n_trim > 0 else X_sorted

        return np.mean(X_trimmed, axis=0)

    def explain(self):
        """Return configuration"""
        return {
            "defense": "bulyan",
            "f": self.f,
            "k": self.k,
            "trim_ratio": self.trim_ratio
        }
