# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Byzantine Detector

Advanced Byzantine detection algorithms.
"""

from typing import List, Set
import numpy as np
from .validator import GradientUpdate, QualityScore


class ByzantineDetector:
    """Detects Byzantine actors in federated learning"""

    def __init__(self, detection_threshold: float = 0.7):
        """Initialize detector

        Args:
            detection_threshold: Threshold for Byzantine detection
        """
        self.detection_threshold = detection_threshold
        self.history: List[Set[str]] = []

    def detect(
        self,
        gradients: List[GradientUpdate],
        scores: List[QualityScore],
    ) -> Set[str]:
        """Detect Byzantine actors

        Args:
            gradients: List of gradient updates
            scores: Quality scores

        Returns:
            Set of Byzantine participant IDs
        """
        byzantine = set()

        for grad, score in zip(gradients, scores):
            if not score.is_valid or score.score < self.detection_threshold:
                byzantine.add(grad.participant_id)

        # Update history
        self.history.append(byzantine)

        return byzantine

    def get_persistent_byzantine(self, rounds: int = 3) -> Set[str]:
        """Get participants consistently flagged as Byzantine

        Args:
            rounds: Number of recent rounds to consider

        Returns:
            Set of persistently Byzantine participants
        """
        if len(self.history) < rounds:
            return set()

        recent = self.history[-rounds:]
        # Find intersection (participants flagged in all rounds)
        persistent = set.intersection(*recent) if recent else set()

        return persistent

    def reset_history(self):
        """Reset detection history"""
        self.history.clear()
