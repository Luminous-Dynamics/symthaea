#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FedAvg: Vanilla Federated Averaging (Baseline)
===============================================

Simple unweighted averaging of client gradients.
NOT Byzantine-robust - should fail under attack.
Included as baseline for comparison.

Reference:
"Communication-Efficient Learning of Deep Networks from Decentralized Data"

Author: Luminous Dynamics
Date: November 8, 2025
"""

import numpy as np
from typing import List


class FedAvg:
    """
    Vanilla Federated Averaging

    Simple mean of all client gradients.
    NO Byzantine robustness - should fail under attack.
    """

    def __init__(self):
        """FedAvg has no parameters"""
        pass

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        **context
    ) -> np.ndarray:
        """
        Aggregate via simple mean

        Args:
            client_updates: List of client gradients
            **context: Additional context (ignored)

        Returns:
            Arithmetic mean of gradients
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        return np.mean(client_updates, axis=0)

    def explain(self):
        """Return configuration"""
        return {
            "defense": "fedavg",
            "byzantine_robust": False,
            "note": "Vanilla baseline - should fail under attack"
        }
