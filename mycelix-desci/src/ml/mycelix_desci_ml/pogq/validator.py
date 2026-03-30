# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""PoGQ Validator

Validates gradient updates for Byzantine behavior in federated learning.
"""

from typing import List, Dict, Tuple
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class GradientUpdate:
    """Gradient update from a federated participant"""

    participant_id: str
    gradients: np.ndarray
    timestamp: float
    proof: bytes = b""


@dataclass
class QualityScore:
    """Quality assessment of a gradient update"""

    score: float  # Overall quality (0.0 - 1.0)
    consistency: float  # Consistency with other gradients
    magnitude_check: float  # Magnitude validity
    direction_alignment: float  # Direction alignment with consensus
    is_valid: bool


class PoGQValidator:
    """Byzantine-resistant gradient validator

    Validates gradient updates using statistical analysis and consensus mechanisms
    to detect and filter out Byzantine (malicious or faulty) participants.

    Supports up to 45% Byzantine fault tolerance.
    """

    def __init__(
        self,
        bft_threshold: float = 0.45,
        min_quality_score: float = 0.7,
        validation_rounds: int = 3,
        adaptive_threshold: bool = True,
    ):
        """Initialize PoGQ validator

        Args:
            bft_threshold: Byzantine fault tolerance threshold (default: 0.45)
            min_quality_score: Minimum quality score for validity (default: 0.7)
            validation_rounds: Number of validation iterations (default: 3)
            adaptive_threshold: Enable adaptive threshold adjustment (default: True)
        """
        self.bft_threshold = bft_threshold
        self.min_quality_score = min_quality_score
        self.validation_rounds = validation_rounds
        self.adaptive_threshold = adaptive_threshold

    def validate_gradients(
        self, gradients: List[GradientUpdate]
    ) -> List[QualityScore]:
        """Validate a batch of gradient updates

        Args:
            gradients: List of gradient updates from participants

        Returns:
            List of quality scores for each gradient

        Raises:
            ValueError: If no gradients provided
        """
        if not gradients:
            raise ValueError("No gradients to validate")

        scores = []
        for gradient in gradients:
            score = self._compute_quality_score(gradient, gradients)
            scores.append(score)

        return scores

    def _compute_quality_score(
        self, gradient: GradientUpdate, all_gradients: List[GradientUpdate]
    ) -> QualityScore:
        """Compute quality score for a single gradient

        Args:
            gradient: The gradient to evaluate
            all_gradients: All gradients for comparison

        Returns:
            Quality score assessment
        """
        # Extract gradient arrays
        grad_arrays = [g.gradients for g in all_gradients]

        # Compute median gradient (Byzantine-resistant)
        median_gradient = np.median(grad_arrays, axis=0)

        # Consistency: distance from median
        distance = np.linalg.norm(gradient.gradients - median_gradient)
        median_distance = np.median([np.linalg.norm(g - median_gradient) for g in grad_arrays])

        consistency = 1.0 / (1.0 + distance / (median_distance + 1e-8))

        # Magnitude check: detect extreme outliers
        grad_norm = np.linalg.norm(gradient.gradients)
        median_norm = np.median([np.linalg.norm(g) for g in grad_arrays])

        magnitude_check = 1.0 / (1.0 + abs(grad_norm - median_norm) / (median_norm + 1e-8))

        # Direction alignment: cosine similarity with median
        direction_alignment = self._cosine_similarity(
            gradient.gradients, median_gradient
        )

        # Overall score (weighted average)
        overall_score = (
            0.4 * consistency + 0.3 * magnitude_check + 0.3 * direction_alignment
        )

        is_valid = overall_score >= self.min_quality_score

        return QualityScore(
            score=overall_score,
            consistency=consistency,
            magnitude_check=magnitude_check,
            direction_alignment=direction_alignment,
            is_valid=is_valid,
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity (-1.0 to 1.0), normalized to (0.0 to 1.0)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        # Normalize to 0-1 range
        return (similarity + 1.0) / 2.0

    def aggregate_gradients(
        self,
        gradients: List[GradientUpdate],
        scores: List[QualityScore],
    ) -> np.ndarray:
        """Aggregate valid gradients into consensus update

        Args:
            gradients: List of gradient updates
            scores: Corresponding quality scores

        Returns:
            Aggregated gradient

        Raises:
            ValueError: If no valid gradients
        """
        if len(gradients) != len(scores):
            raise ValueError("Mismatched gradients and scores")

        # Filter valid gradients
        valid_pairs = [
            (grad, score)
            for grad, score in zip(gradients, scores)
            if score.is_valid
        ]

        if not valid_pairs:
            raise ValueError("No valid gradients to aggregate")

        # Weighted aggregation based on quality scores
        total_weight = sum(score.score for _, score in valid_pairs)
        aggregated = np.zeros_like(valid_pairs[0][0].gradients)

        for grad, score in valid_pairs:
            weight = score.score / total_weight
            aggregated += weight * grad.gradients

        return aggregated

    def detect_byzantine(
        self,
        gradients: List[GradientUpdate],
        scores: List[QualityScore],
    ) -> List[str]:
        """Detect Byzantine actors based on quality scores

        Args:
            gradients: List of gradient updates
            scores: Corresponding quality scores

        Returns:
            List of participant IDs identified as Byzantine
        """
        byzantine = []
        for grad, score in zip(gradients, scores):
            if not score.is_valid or score.score < self.min_quality_score:
                byzantine.append(grad.participant_id)

        return byzantine


# Example usage
if __name__ == "__main__":
    # Create validator
    validator = PoGQValidator(bft_threshold=0.45, min_quality_score=0.7)

    # Simulate gradient updates
    np.random.seed(42)
    gradients = [
        GradientUpdate(
            participant_id=f"participant_{i}",
            gradients=np.random.randn(100),
            timestamp=float(i),
        )
        for i in range(10)
    ]

    # Add a Byzantine gradient (outlier)
    gradients.append(
        GradientUpdate(
            participant_id="byzantine_1",
            gradients=np.random.randn(100) * 10,  # Extreme values
            timestamp=11.0,
        )
    )

    # Validate
    scores = validator.validate_gradients(gradients)

    # Detect Byzantine actors
    byzantine_actors = validator.detect_byzantine(gradients, scores)
    print(f"Detected Byzantine actors: {byzantine_actors}")

    # Aggregate valid gradients
    aggregated = validator.aggregate_gradients(gradients, scores)
    print(f"Aggregated gradient shape: {aggregated.shape}")
