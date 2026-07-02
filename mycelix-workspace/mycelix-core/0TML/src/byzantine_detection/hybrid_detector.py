# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Hybrid Byzantine Detector

Orchestrates multiple detection signals (similarity, temporal, magnitude) to make
robust Byzantine detection decisions using ensemble voting.

Week 3 - Multi-Signal Hybrid Detection System
"""

from typing import List, Dict, Optional
import torch
import torch.nn.functional as F

from .temporal_detector import TemporalConsistencyDetector, TemporalStatistics
from .magnitude_detector import MagnitudeDistributionDetector, MagnitudeStatistics
from .ensemble_voting import EnsembleVotingSystem, EnsembleDecision
from .gradient_analyzer import GradientProfile


class HybridByzantineDetector:
    """
    Multi-signal Byzantine detection using ensemble of detection methods.

    Combines three detection signals:
    1. Gradient Similarity: Cosine similarity analysis (Week 2 foundation)
    2. Temporal Consistency: Behavioral patterns over time (Week 3 new)
    3. Magnitude Distribution: Gradient norm analysis (Week 3 new)

    Architecture:
    - Each signal provides confidence score [0, 1]
    - Ensemble voting combines signals with tunable weights
    - Final decision based on ensemble confidence threshold

    Usage:
        detector = HybridByzantineDetector()

        # Each round
        for gradient, node_id in gradients:
            decision = detector.analyze_gradient(
                gradient, other_gradients, node_id, profile
            )
            if decision.is_byzantine:
                # Flag as Byzantine
    """

    def __init__(
        self,
        # Temporal detector parameters
        temporal_window_size: int = 5,
        temporal_cosine_var_threshold: float = 0.1,
        temporal_magnitude_var_threshold: float = 0.5,
        temporal_min_observations: int = 3,
        # Magnitude detector parameters
        magnitude_z_score_threshold: float = 3.0,
        magnitude_min_samples: int = 3,
        # Ensemble voting parameters
        similarity_weight: float = 0.5,
        temporal_weight: float = 0.3,
        magnitude_weight: float = 0.2,
        ensemble_threshold: float = 0.6
    ):
        """
        Initialize hybrid detector with configurable parameters.

        Args:
            temporal_window_size: Rolling window size for temporal tracking
            temporal_cosine_var_threshold: Variance threshold for cosine consistency
            temporal_magnitude_var_threshold: Variance threshold for magnitude consistency
            temporal_min_observations: Minimum observations before temporal decisions
            magnitude_z_score_threshold: Z-score threshold for magnitude outliers
            magnitude_min_samples: Minimum samples for magnitude statistics
            similarity_weight: Ensemble weight for similarity signal
            temporal_weight: Ensemble weight for temporal signal
            magnitude_weight: Ensemble weight for magnitude signal
            ensemble_threshold: Decision threshold for Byzantine classification
        """
        # Initialize detectors
        self.temporal_detector = TemporalConsistencyDetector(
            window_size=temporal_window_size,
            cosine_variance_threshold=temporal_cosine_var_threshold,
            magnitude_variance_threshold=temporal_magnitude_var_threshold,
            min_observations=temporal_min_observations
        )

        self.magnitude_detector = MagnitudeDistributionDetector(
            z_score_threshold=magnitude_z_score_threshold,
            min_samples=magnitude_min_samples
        )

        self.ensemble = EnsembleVotingSystem(
            similarity_weight=similarity_weight,
            temporal_weight=temporal_weight,
            magnitude_weight=magnitude_weight,
            decision_threshold=ensemble_threshold
        )

    def analyze_gradient(
        self,
        gradient: torch.Tensor,
        other_gradients: List[torch.Tensor],
        node_id: int,
        profile: GradientProfile
    ) -> EnsembleDecision:
        """
        Analyze a single gradient using all detection signals.

        Args:
            gradient: Target gradient to analyze
            other_gradients: Reference gradients from other nodes
            node_id: Node identifier for temporal tracking
            profile: Gradient profile from analyzer (contains thresholds)

        Returns:
            EnsembleDecision with all signal confidences and final decision
        """
        # Signal 1: Gradient Similarity (Week 2 foundation)
        similarity_conf = self._compute_similarity_confidence(
            gradient, other_gradients, profile
        )

        # Compute mean cosine similarity for temporal tracking
        mean_cosine = self._compute_mean_cosine(gradient, other_gradients)

        # Update temporal history BEFORE computing confidence
        # (confidence depends on history)
        self.temporal_detector.update(node_id, gradient, mean_cosine)

        # Signal 2: Temporal Consistency (Week 3 new)
        temporal_conf = self.temporal_detector.compute_temporal_confidence(node_id)

        # Signal 3: Magnitude Distribution (Week 3 new)
        magnitude_conf = self.magnitude_detector.compute_magnitude_confidence(
            gradient, other_gradients
        )

        # Ensemble Decision
        decision = self.ensemble.make_decision(
            similarity_conf, temporal_conf, magnitude_conf
        )

        return decision

    def batch_analyze(
        self,
        gradients: List[torch.Tensor],
        node_ids: List[int],
        profile: GradientProfile
    ) -> List[EnsembleDecision]:
        """
        Analyze multiple gradients in batch.

        Args:
            gradients: List of gradients to analyze
            node_ids: List of node identifiers (must match gradient order)
            profile: Gradient profile from analyzer

        Returns:
            List of EnsembleDecisions (one per gradient)
        """
        if len(gradients) != len(node_ids):
            raise ValueError(
                f"Mismatch: {len(gradients)} gradients but {len(node_ids)} node_ids"
            )

        decisions = []
        for i, (gradient, node_id) in enumerate(zip(gradients, node_ids)):
            # Get reference gradients (all except current)
            other_gradients = [g for j, g in enumerate(gradients) if j != i]

            decision = self.analyze_gradient(
                gradient, other_gradients, node_id, profile
            )
            decisions.append(decision)

        return decisions

    def _compute_similarity_confidence(
        self,
        gradient: torch.Tensor,
        other_gradients: List[torch.Tensor],
        profile: GradientProfile
    ) -> float:
        """
        Compute Byzantine confidence based on gradient similarity (Week 2 signal).

        Args:
            gradient: Target gradient
            other_gradients: Reference gradients
            profile: Contains recommended cos_min/cos_max thresholds

        Returns:
            Confidence [0, 1] where 1 = highly confident Byzantine
        """
        if len(other_gradients) == 0:
            return 0.5  # Cannot determine - abstain

        # Flatten gradients
        grad_flat = gradient.flatten()
        other_flats = [g.flatten() for g in other_gradients]

        # Compute mean cosine similarity
        cosines = [
            F.cosine_similarity(grad_flat.unsqueeze(0), other_flat.unsqueeze(0), dim=1).item()
            for other_flat in other_flats
        ]
        mean_cosine = sum(cosines) / len(cosines)

        # Check if within expected range
        cos_min = profile.recommended_cos_min
        cos_max = profile.recommended_cos_max

        if cos_min <= mean_cosine <= cos_max:
            # Within expected range - likely honest
            return 0.0
        elif mean_cosine < cos_min:
            # Below minimum - suspicious
            distance = cos_min - mean_cosine
            # Normalize to [0, 1] confidence (assume 0.5 is max meaningful distance)
            confidence = min(1.0, distance / 0.5)
            return confidence
        else:
            # Above maximum - suspicious (unusually similar)
            distance = mean_cosine - cos_max
            confidence = min(1.0, distance / 0.5)
            return confidence

    def _compute_mean_cosine(
        self,
        gradient: torch.Tensor,
        other_gradients: List[torch.Tensor]
    ) -> float:
        """
        Compute mean cosine similarity to other gradients.

        Args:
            gradient: Target gradient
            other_gradients: Reference gradients

        Returns:
            Mean cosine similarity
        """
        if len(other_gradients) == 0:
            return 0.0

        grad_flat = gradient.flatten()
        other_flats = [g.flatten() for g in other_gradients]

        cosines = [
            F.cosine_similarity(grad_flat.unsqueeze(0), other_flat.unsqueeze(0), dim=1).item()
            for other_flat in other_flats
        ]

        return sum(cosines) / len(cosines)

    def get_temporal_statistics(self, node_id: int) -> Optional[TemporalStatistics]:
        """Get temporal statistics for a node."""
        return self.temporal_detector.get_temporal_statistics(node_id)

    def get_magnitude_statistics(
        self,
        gradient: torch.Tensor,
        other_gradients: List[torch.Tensor]
    ) -> MagnitudeStatistics:
        """Get magnitude statistics for a gradient."""
        return self.magnitude_detector.get_magnitude_statistics(
            gradient, other_gradients
        )

    def reset(self) -> None:
        """Reset all temporal history."""
        self.temporal_detector.reset()

    def reset_node(self, node_id: int) -> None:
        """Reset temporal history for a specific node."""
        self.temporal_detector.reset_node(node_id)

    def __str__(self) -> str:
        return (
            f"HybridByzantineDetector(\\n"
            f"  Temporal Detector: window={self.temporal_detector.window_size}, "
            f"min_obs={self.temporal_detector.min_observations}\\n"
            f"  Magnitude Detector: z_threshold={self.magnitude_detector.z_score_threshold}\\n"
            f"  Ensemble: {self.ensemble}\\n"
            f")"
        )
