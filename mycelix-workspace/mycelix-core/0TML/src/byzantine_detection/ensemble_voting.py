# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Ensemble Voting System

Combines multiple Byzantine detection signals using weighted voting to make
robust decisions that balance different detection methods.

Week 3 Component - Multi-Signal Hybrid Detection
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EnsembleDecision:
    """Result of ensemble voting for a single gradient"""

    # Individual signal confidences
    similarity_confidence: float  # From gradient cosine similarity analysis
    temporal_confidence: float  # From temporal consistency tracking
    magnitude_confidence: float  # From magnitude distribution analysis

    # Ensemble result
    ensemble_confidence: float  # Weighted average of all signals
    is_byzantine: bool  # Final decision based on threshold
    decision_threshold: float  # Threshold used for decision

    # Signal weights used
    similarity_weight: float
    temporal_weight: float
    magnitude_weight: float

    def __str__(self) -> str:
        return (
            f"EnsembleDecision(\\n"
            f"  Similarity: {self.similarity_confidence:.3f} (weight={self.similarity_weight:.2f})\\n"
            f"  Temporal: {self.temporal_confidence:.3f} (weight={self.temporal_weight:.2f})\\n"
            f"  Magnitude: {self.magnitude_confidence:.3f} (weight={self.magnitude_weight:.2f})\\n"
            f"  Ensemble: {self.ensemble_confidence:.3f}\\n"
            f"  Decision: {'BYZANTINE' if self.is_byzantine else 'HONEST'} "
            f"(threshold={self.decision_threshold:.2f})\\n"
            f")"
        )


class EnsembleVotingSystem:
    """
    Combine multiple Byzantine detection signals with weighted voting.

    Design Philosophy:
    - Each detection method provides a confidence score [0, 1]
    - Scores are combined via weighted average
    - Final decision based on threshold (default: 0.6)
    - Weights are tunable based on empirical performance

    Default Weights Rationale:
    - Similarity (0.5): Primary signal, proven effective in Week 2
    - Temporal (0.3): Strong attack indicator, secondary importance
    - Magnitude (0.2): Complementary signal, catches specific attacks
    """

    def __init__(
        self,
        similarity_weight: float = 0.5,
        temporal_weight: float = 0.3,
        magnitude_weight: float = 0.2,
        decision_threshold: float = 0.6,
        normalize_weights: bool = True
    ):
        """
        Args:
            similarity_weight: Weight for gradient similarity signal
            temporal_weight: Weight for temporal consistency signal
            magnitude_weight: Weight for magnitude distribution signal
            decision_threshold: Threshold for Byzantine decision [0, 1]
            normalize_weights: Whether to normalize weights to sum to 1
        """
        self.similarity_weight = similarity_weight
        self.temporal_weight = temporal_weight
        self.magnitude_weight = magnitude_weight
        self.decision_threshold = decision_threshold

        # Normalize weights if requested
        if normalize_weights:
            total_weight = similarity_weight + temporal_weight + magnitude_weight
            if total_weight > 0:
                self.similarity_weight /= total_weight
                self.temporal_weight /= total_weight
                self.magnitude_weight /= total_weight

    def compute_ensemble_confidence(
        self,
        similarity_conf: float,
        temporal_conf: float,
        magnitude_conf: float
    ) -> float:
        """
        Compute weighted average of detection confidences.

        Args:
            similarity_conf: Confidence from gradient similarity analysis [0, 1]
            temporal_conf: Confidence from temporal consistency [0, 1]
            magnitude_conf: Confidence from magnitude analysis [0, 1]

        Returns:
            Ensemble confidence score [0, 1] where 1 = highly confident Byzantine
        """
        # Weighted sum
        weighted_sum = (
            similarity_conf * self.similarity_weight +
            temporal_conf * self.temporal_weight +
            magnitude_conf * self.magnitude_weight
        )

        # Total weight (should be ~1.0 if normalized)
        total_weight = (
            self.similarity_weight +
            self.temporal_weight +
            self.magnitude_weight
        )

        # Compute weighted average
        if total_weight > 0:
            ensemble_confidence = weighted_sum / total_weight
        else:
            # No weights set - use simple average
            ensemble_confidence = (
                similarity_conf + temporal_conf + magnitude_conf
            ) / 3.0

        return ensemble_confidence

    def make_decision(
        self,
        similarity_conf: float,
        temporal_conf: float,
        magnitude_conf: float,
        threshold: Optional[float] = None
    ) -> EnsembleDecision:
        """
        Make Byzantine detection decision using ensemble voting.

        Args:
            similarity_conf: Confidence from gradient similarity [0, 1]
            temporal_conf: Confidence from temporal consistency [0, 1]
            magnitude_conf: Confidence from magnitude analysis [0, 1]
            threshold: Override default decision threshold (optional)

        Returns:
            EnsembleDecision with detailed breakdown and final decision
        """
        # Compute ensemble confidence
        ensemble_conf = self.compute_ensemble_confidence(
            similarity_conf, temporal_conf, magnitude_conf
        )

        # Use provided threshold or default
        decision_thresh = threshold if threshold is not None else self.decision_threshold

        # Make binary decision
        is_byzantine = ensemble_conf >= decision_thresh

        return EnsembleDecision(
            similarity_confidence=similarity_conf,
            temporal_confidence=temporal_conf,
            magnitude_confidence=magnitude_conf,
            ensemble_confidence=ensemble_conf,
            is_byzantine=is_byzantine,
            decision_threshold=decision_thresh,
            similarity_weight=self.similarity_weight,
            temporal_weight=self.temporal_weight,
            magnitude_weight=self.magnitude_weight
        )

    def batch_decisions(
        self,
        similarity_confs: List[float],
        temporal_confs: List[float],
        magnitude_confs: List[float],
        threshold: Optional[float] = None
    ) -> List[EnsembleDecision]:
        """
        Make decisions for multiple gradients in batch.

        Args:
            similarity_confs: List of similarity confidences
            temporal_confs: List of temporal confidences
            magnitude_confs: List of magnitude confidences
            threshold: Override default decision threshold (optional)

        Returns:
            List of EnsembleDecision objects (one per gradient)
        """
        if not (len(similarity_confs) == len(temporal_confs) == len(magnitude_confs)):
            raise ValueError(
                f"Confidence lists must have same length: "
                f"similarity={len(similarity_confs)}, "
                f"temporal={len(temporal_confs)}, "
                f"magnitude={len(magnitude_confs)}"
            )

        decisions = []
        for sim_conf, temp_conf, mag_conf in zip(
            similarity_confs, temporal_confs, magnitude_confs
        ):
            decision = self.make_decision(
                sim_conf, temp_conf, mag_conf, threshold
            )
            decisions.append(decision)

        return decisions

    def analyze_decision_distribution(
        self,
        decisions: List[EnsembleDecision]
    ) -> Dict[str, float]:
        """
        Analyze distribution of ensemble decisions.

        Args:
            decisions: List of ensemble decisions

        Returns:
            Dictionary with summary statistics
        """
        if len(decisions) == 0:
            return {}

        # Extract ensemble confidences
        ensemble_confs = [d.ensemble_confidence for d in decisions]

        # Count Byzantine vs Honest
        num_byzantine = sum(1 for d in decisions if d.is_byzantine)
        num_honest = len(decisions) - num_byzantine

        # Confidence statistics
        mean_conf = np.mean(ensemble_confs)
        std_conf = np.std(ensemble_confs)
        min_conf = np.min(ensemble_confs)
        max_conf = np.max(ensemble_confs)

        # Signal contributions
        mean_similarity = np.mean([d.similarity_confidence for d in decisions])
        mean_temporal = np.mean([d.temporal_confidence for d in decisions])
        mean_magnitude = np.mean([d.magnitude_confidence for d in decisions])

        return {
            "num_decisions": len(decisions),
            "num_byzantine": num_byzantine,
            "num_honest": num_honest,
            "byzantine_rate": num_byzantine / len(decisions),
            "mean_ensemble_confidence": mean_conf,
            "std_ensemble_confidence": std_conf,
            "min_ensemble_confidence": min_conf,
            "max_ensemble_confidence": max_conf,
            "mean_similarity_conf": mean_similarity,
            "mean_temporal_conf": mean_temporal,
            "mean_magnitude_conf": mean_magnitude
        }

    def tune_weights(
        self,
        similarity_confs: List[float],
        temporal_confs: List[float],
        magnitude_confs: List[float],
        ground_truth: List[bool],
        weight_grid: Optional[List[tuple]] = None
    ) -> tuple:
        """
        Tune ensemble weights to maximize accuracy on labeled data.

        Args:
            similarity_confs: Similarity confidences for each sample
            temporal_confs: Temporal confidences for each sample
            magnitude_confs: Magnitude confidences for each sample
            ground_truth: True labels (True = Byzantine, False = Honest)
            weight_grid: List of (sim_w, temp_w, mag_w) tuples to try

        Returns:
            (best_weights, best_accuracy) tuple
        """
        if weight_grid is None:
            # Default grid search
            weight_grid = [
                (0.5, 0.3, 0.2),  # Default
                (0.6, 0.3, 0.1),  # More similarity emphasis
                (0.4, 0.4, 0.2),  # Balanced similarity/temporal
                (0.7, 0.2, 0.1),  # Heavy similarity
                (0.4, 0.3, 0.3),  # Balanced all three
            ]

        best_accuracy = 0.0
        best_weights = (0.5, 0.3, 0.2)

        for sim_w, temp_w, mag_w in weight_grid:
            # Temporarily set weights
            orig_sim_w = self.similarity_weight
            orig_temp_w = self.temporal_weight
            orig_mag_w = self.magnitude_weight

            self.similarity_weight = sim_w
            self.temporal_weight = temp_w
            self.magnitude_weight = mag_w

            # Make decisions with current weights
            decisions = self.batch_decisions(
                similarity_confs, temporal_confs, magnitude_confs
            )

            # Compute accuracy
            predictions = [d.is_byzantine for d in decisions]
            correct = sum(1 for pred, truth in zip(predictions, ground_truth) if pred == truth)
            accuracy = correct / len(ground_truth)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = (sim_w, temp_w, mag_w)

            # Restore original weights
            self.similarity_weight = orig_sim_w
            self.temporal_weight = orig_temp_w
            self.magnitude_weight = orig_mag_w

        return best_weights, best_accuracy

    def __str__(self) -> str:
        return (
            f"EnsembleVotingSystem(\\n"
            f"  Weights: similarity={self.similarity_weight:.2f}, "
            f"temporal={self.temporal_weight:.2f}, "
            f"magnitude={self.magnitude_weight:.2f}\\n"
            f"  Threshold: {self.decision_threshold:.2f}\\n"
            f")"
        )
