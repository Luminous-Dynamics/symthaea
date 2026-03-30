#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FoolsGold: Anti-Sybil Defense via Historical Cosine Similarity
===============================================================

FoolsGold detects Sybil attacks by tracking historical gradient similarities.
Clients with similar gradient patterns are down-weighted.

Reference:
"Defending Against Sybils in Federated Learning"

Author: Luminous Dynamics
Date: November 8, 2025
"""

import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FoolsGold:
    """
    FoolsGold: Anti-Sybil defense via gradient similarity tracking

    Detects colluding clients by their historical gradient cosine similarity.
    Down-weights clients that submit similar gradients (likely Sybils).
    """

    def __init__(
        self,
        history_window: int = 10,
        alpha: float = 1.0
    ):
        """
        Args:
            history_window: Number of rounds to track per client
            alpha: Weighting parameter (controls down-weighting severity)
        """
        self.history_window = history_window
        self.alpha = alpha

        # Historical gradients per client
        self.client_history: Dict[str, List[np.ndarray]] = defaultdict(list)

    def update_history(
        self,
        client_id: str,
        gradient: np.ndarray
    ):
        """Update client gradient history"""
        self.client_history[client_id].append(gradient)

        # Keep only recent window
        if len(self.client_history[client_id]) > self.history_window:
            self.client_history[client_id] = \
                self.client_history[client_id][-self.history_window:]

    def compute_similarity_matrix(
        self,
        client_ids: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix based on historical gradients

        Returns:
            n × n similarity matrix
        """
        n = len(client_ids)
        similarity = np.eye(n)

        for i, client_i in enumerate(client_ids):
            if client_i not in self.client_history:
                continue

            for j, client_j in enumerate(client_ids):
                if i >= j or client_j not in self.client_history:
                    continue

                # Compute average cosine similarity over history
                history_i = self.client_history[client_i]
                history_j = self.client_history[client_j]

                sims = []
                for grad_i in history_i:
                    for grad_j in history_j:
                        norm_i = np.linalg.norm(grad_i)
                        norm_j = np.linalg.norm(grad_j)

                        if norm_i > 1e-8 and norm_j > 1e-8:
                            sim = np.dot(grad_i, grad_j) / (norm_i * norm_j)
                            sims.append(sim)

                if sims:
                    avg_sim = np.mean(sims)
                    similarity[i, j] = avg_sim
                    similarity[j, i] = avg_sim

        return similarity

    def compute_foolsgold_weights(
        self,
        client_ids: List[str]
    ) -> np.ndarray:
        """
        Compute FoolsGold weights based on historical similarity

        Clients with high similarity to others are down-weighted
        (assumed to be Sybils).

        Returns:
            Weight vector (one per client)
        """
        n = len(client_ids)

        # Compute similarity matrix
        similarity = self.compute_similarity_matrix(client_ids)

        # Compute importance scores (inverse of max similarity to others)
        importance = np.zeros(n)
        for i in range(n):
            # Max similarity to any other client
            max_sim = np.max(similarity[i, :i].tolist() + similarity[i, i+1:].tolist())
            importance[i] = 1.0 / (1.0 + max_sim)

        # Apply alpha weighting
        weights = importance ** self.alpha

        # Normalize
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

        return weights

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        client_ids: Optional[List[str]] = None,
        **context
    ) -> np.ndarray:
        """
        Aggregate with FoolsGold anti-Sybil weighting

        Args:
            client_updates: List of client gradients
            client_ids: List of client identifiers
            **context: Additional context (ignored)

        Returns:
            FoolsGold weighted aggregated gradient
        """
        if len(client_updates) == 0:
            raise ValueError("No gradients to aggregate")

        # Generate client IDs if not provided
        if client_ids is None:
            client_ids = [f"client_{i}" for i in range(len(client_updates))]

        # Update history
        for client_id, gradient in zip(client_ids, client_updates):
            self.update_history(client_id, gradient)

        # Compute FoolsGold weights
        weights = self.compute_foolsgold_weights(client_ids)

        # Weighted aggregation
        aggregated = np.zeros_like(client_updates[0])
        for weight, gradient in zip(weights, client_updates):
            aggregated += weight * gradient

        logger.info(f"FoolsGold: Aggregated {len(client_updates)} clients "
                   f"(weight range: [{np.min(weights):.3f}, {np.max(weights):.3f}])")

        return aggregated

    def explain(self):
        """Return configuration"""
        return {
            "defense": "foolsgold",
            "history_window": self.history_window,
            "alpha": self.alpha,
            "n_clients_tracked": len(self.client_history)
        }
