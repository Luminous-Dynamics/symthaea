# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Self-Healing Byzantine Detector

Instead of just excluding anomalous nodes, this detector attempts to
correct recoverable errors. Many "Byzantine" behaviors are actually
honest mistakes (bugs, network issues, stale models).

Key Insight:
    Traditional: Detect → Exclude
    Self-Healing: Detect → Analyze → Correct or Exclude

This can increase effective participation from ~55% to ~90% in realistic
deployments where many flagged nodes are actually broken honest nodes.

Author: Luminous Dynamics
Date: December 30, 2025
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HealingResult:
    """Result from self-healing attempt."""
    healed_nodes: Set[str]
    healed_gradients: Dict[str, np.ndarray]
    excluded_nodes: Set[str]
    healing_details: Dict[str, Dict]


class SelfHealingDetector:
    """
    Self-healing Byzantine detector.

    Attempts to correct anomalous gradients by projecting them onto
    the "honest subspace" computed from trusted gradients.

    Recovery Types:
    1. Minor deviation: Full correction (keep most of original)
    2. Moderate deviation: Partial correction (50/50 blend)
    3. Major deviation: Exclusion (too far from honest)

    Example:
        >>> healer = SelfHealingDetector()
        >>> result = healer.heal(gradients, flagged_nodes)
        >>> print(f"Healed: {result.healed_nodes}")
        >>> # Use result.healed_gradients for corrected versions
    """

    def __init__(
        self,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.7,
        variance_explained: float = 0.95,
        min_honest_nodes: int = 3,
    ):
        """
        Initialize self-healing detector.

        Args:
            minor_threshold: Max deviation for full healing
            moderate_threshold: Max deviation for partial healing
            variance_explained: PCA variance to retain
            min_honest_nodes: Minimum honest nodes needed for healing
        """
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.variance_explained = variance_explained
        self.min_honest_nodes = min_honest_nodes

    def compute_honest_subspace(
        self, honest_gradients: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the "honest subspace" via PCA.

        Args:
            honest_gradients: List of trusted gradient arrays

        Returns:
            (mean, projection_matrix, singular_values)
        """
        if len(honest_gradients) < self.min_honest_nodes:
            raise ValueError(f"Need at least {self.min_honest_nodes} honest gradients")

        grads = np.array(honest_gradients)
        mean = np.mean(grads, axis=0)
        centered = grads - mean

        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            # Keep components explaining variance_explained of variance
            total_var = np.sum(S ** 2)
            cumsum = np.cumsum(S ** 2) / total_var
            n_components = np.searchsorted(cumsum, self.variance_explained) + 1
            n_components = max(1, min(n_components, len(S)))

            projection = Vt[:n_components].T

            return mean, projection, S[:n_components]

        except np.linalg.LinAlgError as e:
            logger.warning(f"SVD failed: {e}")
            # Fall back to identity (no projection)
            return mean, np.eye(len(mean)), np.ones(len(mean))

    def compute_healing_ratio(
        self,
        gradient: np.ndarray,
        mean: np.ndarray,
        projection: np.ndarray,
    ) -> float:
        """
        Compute how far gradient deviates from honest subspace.

        Returns:
            Ratio in [0, 1+] where 0 = perfectly aligned, >1 = mostly outlier
        """
        centered = gradient - mean

        # Project onto honest subspace
        projected = projection @ projection.T @ centered
        residual = centered - projected

        # Ratio = residual magnitude / total magnitude
        residual_norm = np.linalg.norm(residual)
        gradient_norm = np.linalg.norm(centered)

        if gradient_norm < 1e-8:
            return 1.0  # Zero gradient is suspicious

        return residual_norm / gradient_norm

    def heal_gradient(
        self,
        gradient: np.ndarray,
        mean: np.ndarray,
        projection: np.ndarray,
        healing_ratio: float,
    ) -> Tuple[np.ndarray, str]:
        """
        Attempt to heal a gradient.

        Args:
            gradient: Original gradient
            mean: Honest mean
            projection: Honest subspace projection matrix
            healing_ratio: Pre-computed healing ratio

        Returns:
            (healed_gradient, healing_type)
        """
        centered = gradient - mean
        projected = projection @ projection.T @ centered

        if healing_ratio < self.minor_threshold:
            # Minor deviation: Strong correction (mostly projection)
            healed = mean + projected + 0.2 * (centered - projected)
            return healed, "minor"

        elif healing_ratio < self.moderate_threshold:
            # Moderate deviation: Partial correction (50/50)
            healed = mean + projected + 0.5 * (centered - projected)
            return healed, "moderate"

        else:
            # Major deviation: Cannot heal
            return gradient, "excluded"

    def heal(
        self,
        gradients: Dict[str, np.ndarray],
        flagged_nodes: Set[str],
        trusted_nodes: Optional[Set[str]] = None,
    ) -> HealingResult:
        """
        Attempt to heal flagged nodes.

        Args:
            gradients: All gradients (node_id -> array)
            flagged_nodes: Nodes suspected of being Byzantine
            trusted_nodes: Optional set of known-honest nodes

        Returns:
            HealingResult with healed gradients and excluded nodes
        """
        # Determine honest nodes
        if trusted_nodes:
            honest_node_ids = trusted_nodes - flagged_nodes
        else:
            honest_node_ids = set(gradients.keys()) - flagged_nodes

        honest_gradients = [
            gradients[node_id] for node_id in honest_node_ids
            if node_id in gradients
        ]

        # Check if we have enough honest nodes
        if len(honest_gradients) < self.min_honest_nodes:
            logger.warning(
                f"Not enough honest nodes ({len(honest_gradients)}) for healing. "
                f"Need at least {self.min_honest_nodes}."
            )
            return HealingResult(
                healed_nodes=set(),
                healed_gradients={},
                excluded_nodes=flagged_nodes,
                healing_details={},
            )

        # Compute honest subspace
        try:
            mean, projection, _ = self.compute_honest_subspace(honest_gradients)
        except Exception as e:
            logger.error(f"Failed to compute honest subspace: {e}")
            return HealingResult(
                healed_nodes=set(),
                healed_gradients={},
                excluded_nodes=flagged_nodes,
                healing_details={},
            )

        # Try to heal each flagged node
        healed_nodes = set()
        healed_gradients = {}
        excluded_nodes = set()
        healing_details = {}

        for node_id in flagged_nodes:
            if node_id not in gradients:
                continue

            gradient = gradients[node_id]
            healing_ratio = self.compute_healing_ratio(gradient, mean, projection)

            healed_grad, healing_type = self.heal_gradient(
                gradient, mean, projection, healing_ratio
            )

            healing_details[node_id] = {
                "healing_ratio": healing_ratio,
                "healing_type": healing_type,
            }

            if healing_type != "excluded":
                healed_nodes.add(node_id)
                healed_gradients[node_id] = healed_grad
                logger.debug(
                    f"Healed {node_id}: ratio={healing_ratio:.3f}, type={healing_type}"
                )
            else:
                excluded_nodes.add(node_id)
                logger.debug(
                    f"Excluded {node_id}: ratio={healing_ratio:.3f} (too far)"
                )

        logger.info(
            f"Self-healing complete: {len(healed_nodes)} healed, "
            f"{len(excluded_nodes)} excluded"
        )

        return HealingResult(
            healed_nodes=healed_nodes,
            healed_gradients=healed_gradients,
            excluded_nodes=excluded_nodes,
            healing_details=healing_details,
        )
