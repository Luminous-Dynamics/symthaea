#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Coordinate-wise Median: Simple Baseline Defense
================================================

Computes the median of each gradient coordinate independently.
Simple and efficient baseline for Byzantine robustness.

Phase 5 adds CoordinateMedianSafe with three safety guards:
1. Min-clients guard: Fall back to trimmed-mean if N < 2f+3
2. Norm clamp: Clip updates to c × median_norm
3. Direction check: Drop clients with cosine < -0.2 to robust center

Author: Luminous Dynamics
Date: November 8, 2025
"""

import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class CoordinateMedian:
    """
    Coordinate-wise median aggregation

    Robust baseline: compute median per coordinate.
    Byzantine-robust up to 50% Byzantine fraction.
    """

    def __init__(self):
        pass

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """
        Aggregate via coordinate-wise median

        Args:
            client_updates: List of client gradients
            **context: Additional context (ignored)

        Returns:
            Coordinate-wise median
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        X = np.vstack(client_updates)
        return np.median(X, axis=0)

    def explain(self):
        """Return configuration"""
        return {"defense": "coord_median"}


class CoordinateMedianSafe:
    """
    Coordinate-wise median with Phase 5 safety guards

    Guards:
    1. Min-clients guard: Fall back to trimmed-mean if N < 2f+3
    2. Norm clamp: Clip each update to c × median_norm (default c=3)
    3. Direction check: Drop clients with cosine < threshold to robust center

    Byzantine-robust up to 50% Byzantine fraction with enhanced safety.
    """

    def __init__(
        self,
        byzantine_fraction: float = 0.33,
        norm_clamp_factor: float = 3.0,
        direction_threshold: float = -0.2,
        trim_ratio: float = 0.1,
        enable_min_clients_guard: bool = True,
        enable_norm_clamp: bool = True,
        enable_direction_check: bool = True
    ):
        """
        Args:
            byzantine_fraction: Expected Byzantine fraction (default 0.33)
            norm_clamp_factor: Norm clipping factor (default 3.0)
            direction_threshold: Cosine threshold for direction check (default -0.2)
            trim_ratio: Trimmed-mean ratio for fallback (default 0.1)
            enable_min_clients_guard: Enable min-clients guard
            enable_norm_clamp: Enable norm clamp guard
            enable_direction_check: Enable direction check guard
        """
        self.byzantine_fraction = byzantine_fraction
        self.norm_clamp_factor = norm_clamp_factor
        self.direction_threshold = direction_threshold
        self.trim_ratio = trim_ratio
        self.enable_min_clients_guard = enable_min_clients_guard
        self.enable_norm_clamp = enable_norm_clamp
        self.enable_direction_check = enable_direction_check

        # Import trimmed mean for fallback
        from .trimmed_mean import TrimmedMean
        self.trimmed_mean_fallback = TrimmedMean(trim_ratio=trim_ratio)

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """
        Aggregate via coordinate-wise median with safety guards

        Args:
            client_updates: List of client gradients
            **context: Additional context (e.g., client_ids)

        Returns:
            Safely aggregated gradient
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        n_clients = len(client_updates)
        f = int(n_clients * self.byzantine_fraction)

        # Guard 1: Min-clients guard
        min_clients_required = 2 * f + 3
        if self.enable_min_clients_guard and n_clients < min_clients_required:
            logger.warning(
                f"Min-clients guard triggered: N={n_clients} < 2f+3={min_clients_required}. "
                f"Falling back to trimmed-mean."
            )
            return self.trimmed_mean_fallback.aggregate(client_updates)

        # Convert to matrix
        X = np.vstack(client_updates)

        # Guard 2: Norm clamp
        if self.enable_norm_clamp:
            norms = np.linalg.norm(X, axis=1)
            median_norm = np.median(norms)
            max_norm = self.norm_clamp_factor * median_norm

            # Clip gradients exceeding max_norm
            for i in range(len(X)):
                if norms[i] > max_norm:
                    X[i] = X[i] * (max_norm / norms[i])
                    logger.debug(
                        f"Norm clamp: Client {i} norm {norms[i]:.3f} > {max_norm:.3f}, clipped"
                    )

        # Guard 3: Direction check (before computing final median)
        if self.enable_direction_check:
            # Compute robust center (preliminary median without direction filter)
            preliminary_center = np.median(X, axis=0)

            # Check cosine similarity to center
            center_norm = np.linalg.norm(preliminary_center)
            if center_norm > 1e-8:  # Avoid division by zero
                kept_indices = []
                for i in range(len(X)):
                    grad_norm = np.linalg.norm(X[i])
                    if grad_norm > 1e-8:
                        cosine = np.dot(X[i], preliminary_center) / (grad_norm * center_norm)
                        if cosine >= self.direction_threshold:
                            kept_indices.append(i)
                        else:
                            logger.debug(
                                f"Direction check: Client {i} cosine {cosine:.3f} < {self.direction_threshold}, dropped"
                            )
                    else:
                        kept_indices.append(i)  # Keep zero gradients

                if len(kept_indices) > 0:
                    X = X[kept_indices]
                else:
                    logger.warning("Direction check dropped all clients, using preliminary center")
                    return preliminary_center

        # Final coordinate-wise median
        result = np.median(X, axis=0)
        return result

    def explain(self):
        """Return configuration"""
        return {
            "defense": "coord_median_safe",
            "byzantine_fraction": self.byzantine_fraction,
            "norm_clamp_factor": self.norm_clamp_factor,
            "direction_threshold": self.direction_threshold,
            "trim_ratio": self.trim_ratio,
            "guards": {
                "min_clients": self.enable_min_clients_guard,
                "norm_clamp": self.enable_norm_clamp,
                "direction_check": self.enable_direction_check
            }
        }
