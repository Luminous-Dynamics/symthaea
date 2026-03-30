#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Trimmed Mean: Coordinate-wise Robust Aggregation
=================================================

Computes trimmed mean per coordinate: drop top β and bottom β percentile,
then average the remaining values. Robust to Byzantine outliers.

Reference:
"Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"

Author: Luminous Dynamics
Date: November 8, 2025
"""

import numpy as np
from typing import List


class TrimmedMean:
    """
    Trimmed Mean aggregation

    For each coordinate, sort values, trim top/bottom β percentile,
    then average. Byzantine-robust for f < (1-2β)n.
    """

    def __init__(self, trim_ratio: float = 0.1):
        """
        Args:
            trim_ratio: Fraction to trim from each end (default 10%)
        """
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """
        Aggregate via trimmed mean

        Args:
            client_updates: List of client gradients
            **context: Additional context (ignored)

        Returns:
            Trimmed mean
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        n = len(client_updates)
        X = np.vstack(client_updates)

        # Number to trim from each end
        n_trim = int(n * self.trim_ratio)

        if n_trim == 0:
            # Not enough samples to trim
            return np.mean(X, axis=0)

        # Sort each coordinate
        X_sorted = np.sort(X, axis=0)

        # Trim and average
        X_trimmed = X_sorted[n_trim:-n_trim] if n_trim > 0 else X_sorted

        return np.mean(X_trimmed, axis=0)

    def explain(self):
        """Return configuration"""
        return {
            "defense": "trimmed_mean",
            "trim_ratio": self.trim_ratio
        }
