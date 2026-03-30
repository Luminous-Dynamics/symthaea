# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Magnitude Distribution Detector

Analyzes gradient magnitude patterns to detect Byzantine attacks that manipulate
gradient norms while potentially keeping directions similar.

Week 3 Component - Multi-Signal Hybrid Detection
"""

from typing import List
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class MagnitudeStatistics:
    """Statistics about gradient magnitude distribution"""

    # Current gradient statistics
    gradient_norm: float
    mean_norm: float
    std_norm: float
    min_norm: float
    max_norm: float

    # Z-score analysis
    z_score: float
    is_outlier: bool  # |z_score| > 3

    # Magnitude confidence
    magnitude_confidence: float  # [0, 1] where 1 = highly confident Byzantine


class MagnitudeDistributionDetector:
    """
    Analyze gradient magnitude patterns to detect norm-based Byzantine attacks.

    Hypothesis:
    - Under label skew: Different labels → different gradient directions,
      BUT similar gradient magnitudes (same learning rate, loss scale)
    - Byzantine attacks may have abnormal magnitudes:
      * Noise injection: Very large magnitudes
      * Gradient suppression: Very small magnitudes
      * Sign-flipping: Normal magnitude, flipped direction (caught by similarity)

    Detection Method:
    - Compute Z-score: How many standard deviations from mean?
    - Z-score > 3 is statistically unusual (>99.7% of normal distribution)
    - High |Z-score| → Likely Byzantine (abnormal magnitude)
    - Low |Z-score| → Likely honest (normal magnitude)
    """

    def __init__(
        self,
        z_score_threshold: float = 3.0,
        min_samples: int = 3
    ):
        """
        Args:
            z_score_threshold: Standard deviations from mean to flag as outlier
            min_samples: Minimum number of gradients needed for reliable statistics
        """
        self.z_score_threshold = z_score_threshold
        self.min_samples = min_samples

    def compute_magnitude_confidence(
        self,
        gradient: torch.Tensor,
        other_gradients: List[torch.Tensor]
    ) -> float:
        """
        Compute confidence that gradient is Byzantine based on magnitude analysis.

        Uses Z-score to measure how unusual the gradient magnitude is compared
        to other gradients in the same round.

        Args:
            gradient: Target gradient to analyze
            other_gradients: Reference gradients from other nodes

        Returns:
            Confidence score [0, 1] where 1 = highly confident Byzantine,
            0 = highly confident honest, 0.5 = neutral/insufficient data
        """
        if len(other_gradients) < self.min_samples:
            return 0.5  # Not enough samples - abstain

        # Compute gradient norms
        grad_norm = torch.norm(gradient).item()
        other_norms = [torch.norm(g).item() for g in other_gradients]

        mean_norm = np.mean(other_norms)
        std_norm = np.std(other_norms)

        if std_norm == 0:
            # All other gradients have same norm
            if abs(grad_norm - mean_norm) < 1e-6:
                return 0.0  # Same as others - honest
            else:
                return 1.0  # Different from uniform others - Byzantine

        # Compute Z-score: how many standard deviations away?
        z_score = abs(grad_norm - mean_norm) / std_norm

        # Convert Z-score to confidence [0, 1]
        # z = 0: confidence = 0 (perfectly normal)
        # z = 3: confidence = 1 (highly unusual)
        # z > 3: confidence = 1 (cap at 1)
        confidence = min(1.0, z_score / self.z_score_threshold)

        return confidence

    def get_magnitude_statistics(
        self,
        gradient: torch.Tensor,
        other_gradients: List[torch.Tensor]
    ) -> MagnitudeStatistics:
        """
        Get detailed magnitude statistics for a gradient.

        Args:
            gradient: Target gradient to analyze
            other_gradients: Reference gradients from other nodes

        Returns:
            MagnitudeStatistics object with detailed analysis
        """
        grad_norm = torch.norm(gradient).item()

        if len(other_gradients) == 0:
            # No reference gradients
            return MagnitudeStatistics(
                gradient_norm=grad_norm,
                mean_norm=grad_norm,
                std_norm=0.0,
                min_norm=grad_norm,
                max_norm=grad_norm,
                z_score=0.0,
                is_outlier=False,
                magnitude_confidence=0.5
            )

        other_norms = [torch.norm(g).item() for g in other_gradients]
        mean_norm = np.mean(other_norms)
        std_norm = np.std(other_norms)

        # Compute Z-score
        if std_norm > 0:
            z_score = (grad_norm - mean_norm) / std_norm
        else:
            z_score = 0.0 if abs(grad_norm - mean_norm) < 1e-6 else float('inf')

        is_outlier = abs(z_score) > self.z_score_threshold

        return MagnitudeStatistics(
            gradient_norm=grad_norm,
            mean_norm=mean_norm,
            std_norm=std_norm,
            min_norm=np.min(other_norms),
            max_norm=np.max(other_norms),
            z_score=z_score,
            is_outlier=is_outlier,
            magnitude_confidence=self.compute_magnitude_confidence(
                gradient, other_gradients
            )
        )

    def detect_norm_based_attacks(
        self,
        gradients: List[torch.Tensor],
        attack_types: List[str] = ["noise", "suppression", "scaling"]
    ) -> dict:
        """
        Detect specific norm-based attack patterns.

        Args:
            gradients: List of gradients to analyze
            attack_types: Types of attacks to check for

        Returns:
            Dictionary mapping attack types to detection results
        """
        norms = [torch.norm(g).item() for g in gradients]
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        max_norm = np.max(norms)
        min_norm = np.min(norms)

        results = {}

        # Noise injection: Abnormally large gradients
        if "noise" in attack_types:
            noise_threshold = mean_norm + 3 * std_norm
            noise_attacks = [i for i, norm in enumerate(norms) if norm > noise_threshold]
            results["noise"] = {
                "detected": len(noise_attacks) > 0,
                "suspected_nodes": noise_attacks,
                "threshold": noise_threshold
            }

        # Gradient suppression: Abnormally small gradients
        if "suppression" in attack_types:
            suppression_threshold = max(0, mean_norm - 3 * std_norm)
            suppression_attacks = [
                i for i, norm in enumerate(norms)
                if norm < suppression_threshold
            ]
            results["suppression"] = {
                "detected": len(suppression_attacks) > 0,
                "suspected_nodes": suppression_attacks,
                "threshold": suppression_threshold
            }

        # Scaling attacks: Unusual magnitude range
        if "scaling" in attack_types:
            if max_norm > 0:
                magnitude_range = (max_norm - min_norm) / max_norm
                # High range indicates diverse magnitudes (potential attack)
                results["scaling"] = {
                    "detected": magnitude_range > 0.5,
                    "magnitude_range": magnitude_range,
                    "max_norm": max_norm,
                    "min_norm": min_norm
                }
            else:
                results["scaling"] = {
                    "detected": False,
                    "magnitude_range": 0.0,
                    "max_norm": 0.0,
                    "min_norm": 0.0
                }

        return results
