#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FLTrust: Direction-Based Byzantine Defense with Server Validation Set
======================================================================

FLTrust uses a clean server-side validation set to compute a trusted gradient.
Client gradients are accepted if their direction aligns with the server gradient.

Reference:
"FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping"

Author: Luminous Dynamics
Date: November 8, 2025
"""

import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FLTrust:
    """
    FLTrust: Trust-bootstrapped Byzantine defense

    Uses server-side clean validation set to compute reference gradient.
    Aggregates client gradients weighted by cosine similarity to reference.
    """

    def __init__(
        self,
        server_lr: float = 0.001,
        clip_threshold: float = 1.0,
        normalize: bool = True
    ):
        """
        Args:
            server_lr: Server learning rate for validation gradient
            clip_threshold: Clip weights to prevent domination
            normalize: Whether to normalize client weights
        """
        self.server_lr = server_lr
        self.clip_threshold = clip_threshold
        self.normalize = normalize

        self.server_gradient: Optional[np.ndarray] = None

    def set_server_gradient(
        self,
        server_gradient: np.ndarray
    ):
        """
        Set server reference gradient from clean validation set

        Args:
            server_gradient: Gradient computed on server validation data
        """
        self.server_gradient = server_gradient
        logger.info(f"FLTrust: Server gradient set (norm={np.linalg.norm(server_gradient):.3f})")

    def compute_trust_score(
        self,
        client_gradient: np.ndarray
    ) -> float:
        """
        Compute trust score: ReLU(cosine similarity with server gradient)

        Args:
            client_gradient: Client gradient vector

        Returns:
            Trust score in [0, 1]
        """
        if self.server_gradient is None:
            raise RuntimeError("Server gradient not set. Call set_server_gradient() first")

        # Cosine similarity
        dot_product = np.dot(client_gradient, self.server_gradient)
        norm_client = np.linalg.norm(client_gradient)
        norm_server = np.linalg.norm(self.server_gradient)

        if norm_client < 1e-8 or norm_server < 1e-8:
            return 0.0

        cosine_sim = dot_product / (norm_client * norm_server)

        # ReLU: only positive alignment
        trust_score = max(0.0, cosine_sim)

        return float(trust_score)

    def aggregate(
        self,
        client_updates: List[np.ndarray],
        server_gradient: Optional[np.ndarray] = None,
        **context
    ) -> np.ndarray:
        """
        Aggregate client gradients with FLTrust weighting

        Args:
            client_updates: List of client gradients
            server_gradient: Optional server gradient (if not already set)
            **context: Additional context (ignored)

        Returns:
            Trust-weighted aggregated gradient
        """
        if server_gradient is not None:
            self.set_server_gradient(server_gradient)

        if self.server_gradient is None:
            raise RuntimeError("Server gradient required for FLTrust")

        if len(client_updates) == 0:
            return self.server_gradient

        # Compute trust scores
        trust_scores = []
        for gradient in client_updates:
            score = self.compute_trust_score(gradient)
            trust_scores.append(score)

        trust_scores = np.array(trust_scores)

        # Clip trust scores
        trust_scores = np.clip(trust_scores, 0, self.clip_threshold)

        # Normalize if requested
        if self.normalize and np.sum(trust_scores) > 0:
            trust_scores = trust_scores / np.sum(trust_scores)

        # If all scores are zero, fall back to server gradient
        if np.sum(trust_scores) < 1e-8:
            logger.warning("FLTrust: All clients rejected, using server gradient")
            return self.server_gradient

        # Weighted aggregation
        aggregated = np.zeros_like(self.server_gradient)
        for score, gradient in zip(trust_scores, client_updates):
            aggregated += score * gradient

        # Scale by server gradient norm (FLTrust normalization)
        server_norm = np.linalg.norm(self.server_gradient)
        agg_norm = np.linalg.norm(aggregated)

        if agg_norm > 1e-8:
            aggregated = aggregated * (server_norm / agg_norm)

        logger.info(f"FLTrust: Aggregated with {np.sum(trust_scores > 0)}/{len(client_updates)} clients")

        return aggregated

    def explain(self):
        """Return configuration"""
        return {
            "defense": "fltrust",
            "server_lr": self.server_lr,
            "clip_threshold": self.clip_threshold,
            "normalize": self.normalize,
            "server_gradient_set": self.server_gradient is not None
        }
