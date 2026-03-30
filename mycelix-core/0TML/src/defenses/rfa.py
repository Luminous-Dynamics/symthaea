#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
RFA (Robust Federated Averaging): Geometric Median via Weiszfeld Algorithm
===========================================================================

RFA computes the geometric median of client gradients, which is robust to
Byzantine outliers. Uses iterative Weiszfeld algorithm for efficiency.

Reference:
"Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing"

Author: Luminous Dynamics
Date: November 8, 2025
"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class RobustFederatedAveraging:
    """
    Robust Federated Averaging (RFA) via geometric median

    Computes the geometric median of gradients using Weiszfeld's algorithm,
    which is provably robust to Byzantine failures.
    """

    def __init__(
        self,
        max_iters: int = 20,
        tol: float = 1e-5,
        weiszfeld_epsilon: float = 1e-6
    ):
        """
        Args:
            max_iters: Maximum Weiszfeld iterations
            tol: Convergence tolerance
            weiszfeld_epsilon: Small constant to avoid division by zero
        """
        self.max_iters = max_iters
        self.tol = tol
        self.eps = weiszfeld_epsilon

    def geometric_median_weiszfeld(
        self,
        gradients: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute geometric median using Weiszfeld algorithm

        The geometric median minimizes sum of L2 distances to all points.
        Weiszfeld is an iteratively reweighted least squares algorithm.
        """
        X = np.vstack(gradients)
        n = len(gradients)

        # Initialize at mean
        median = np.mean(X, axis=0)

        for iteration in range(self.max_iters):
            # Compute distances
            distances = np.linalg.norm(X - median, axis=1)

            # Weights (inverse distances)
            weights = 1.0 / (distances + self.eps)
            weights /= np.sum(weights)

            # Update median
            new_median = np.sum(weights[:, np.newaxis] * X, axis=0)

            # Check convergence
            delta = np.linalg.norm(new_median - median)
            if delta < self.tol:
                logger.debug(f"RFA converged in {iteration+1} iterations")
                break

            median = new_median

        return median

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """
        Aggregate gradients using geometric median

        Args:
            client_updates: List of client gradients
            **context: Additional context (ignored)

        Returns:
            Geometric median of gradients
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        if len(client_updates) == 1:
            return client_updates[0]

        return self.geometric_median_weiszfeld(client_updates)

    def explain(self):
        """Return configuration"""
        return {
            "defense": "rfa",
            "max_iters": self.max_iters,
            "tol": self.tol
        }
