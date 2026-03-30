# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen 5 Layer 7: Self-Healing Mechanism
======================================

Automatic recovery from high Byzantine fractions through:
- BFT estimation from score distributions
- Gradient quarantine with exponential recovery
- Adaptive thresholds during healing
- Multi-phase recovery protocol

Key Features:
- Activates when BFT > 45% tolerance
- Reduces Byzantine influence via weight quarantine
- Gradual restoration after sustained honest behavior
- No manual intervention required

Algorithm:
    1. Continuously estimate Byzantine fraction from scores
    2. If BFT > tolerance: activate healing (quarantine Byzantine gradients)
    3. Monitor behavior with reduced Byzantine influence
    4. Gradually restore weights (exponential recovery)
    5. If BFT < threshold for sustained period: return to normal

Author: Luminous Dynamics
Date: November 12, 2025
"""

from typing import List, Dict, Set, Tuple, Optional, Deque
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class HealingConfig:
    """Configuration for self-healing mechanism."""

    bft_tolerance: float = 0.45  # BFT above which healing activates
    healing_threshold: float = 0.40  # BFT below which healing deactivates
    score_threshold: float = 0.5  # Byzantine/honest boundary
    quarantine_weight: float = 0.1  # Weight for quarantined gradients (10%)
    recovery_rate: float = 0.05  # Exponential recovery (5% per round)
    window_size: int = 100  # BFT estimation window
    sustained_rounds: int = 5  # Rounds of low BFT before deactivating


class BFTEstimator:
    """
    Byzantine Fraction Tolerance estimator.

    Estimates current Byzantine fraction from detection score distribution.

    Simple approach: fraction of scores below threshold
    (More sophisticated: EM algorithm for Gaussian mixture)
    """

    def __init__(self, threshold: float = 0.5, window_size: int = 100):
        """
        Initialize BFT estimator.

        Args:
            threshold: Score threshold for Byzantine classification
            window_size: Number of recent scores to consider
        """
        self.threshold = threshold
        self.window_size = window_size
        self.score_history: Deque[float] = deque(maxlen=window_size)

    def update(self, scores: List[float]):
        """Update score history with new batch."""
        self.score_history.extend(scores)

    def estimate_bft(self) -> float:
        """
        Estimate current Byzantine fraction.

        Returns:
            Estimated BFT in [0, 1]
        """
        if len(self.score_history) == 0:
            return 0.0

        byzantine_count = sum(1 for s in self.score_history if s < self.threshold)
        return byzantine_count / len(self.score_history)

    def is_under_attack(self, tolerance: float = 0.45) -> bool:
        """
        Check if BFT exceeds tolerance.

        Args:
            tolerance: Maximum acceptable Byzantine fraction

        Returns:
            True if estimated BFT > tolerance
        """
        return self.estimate_bft() > tolerance

    def get_score_distribution(self) -> Dict[str, float]:
        """Get statistics on score distribution."""
        if len(self.score_history) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        scores = list(self.score_history)
        return {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
        }


class GradientQuarantine:
    """
    Gradient quarantine system for suspected Byzantine gradients.

    Reduces influence of Byzantine gradients without removing them completely.
    Implements exponential recovery for gradual weight restoration.
    """

    def __init__(
        self,
        quarantine_weight: float = 0.1,
        recovery_rate: float = 0.05,
    ):
        """
        Initialize quarantine system.

        Args:
            quarantine_weight: Weight for quarantined gradients (0.1 = 10%)
            recovery_rate: Exponential recovery rate (0.05 = 5% per round)
        """
        self.quarantine_weight = quarantine_weight
        self.recovery_rate = recovery_rate

        # Tracking
        self.quarantined_indices: Set[int] = set()
        self.current_weights: Dict[int, float] = {}
        self.original_weights: Dict[int, float] = {}
        self.quarantine_start: Dict[int, int] = {}  # Round when quarantined

    def quarantine_gradient(
        self, idx: int, round_num: int, original_weight: float = 1.0
    ):
        """
        Quarantine a gradient by reducing its weight.

        Args:
            idx: Gradient index
            round_num: Current round number
            original_weight: Original weight before quarantine
        """
        if idx not in self.quarantined_indices:
            self.quarantined_indices.add(idx)
            self.original_weights[idx] = original_weight
            self.quarantine_start[idx] = round_num

        self.current_weights[idx] = self.quarantine_weight

    def release_gradient(self, idx: int):
        """Release gradient from quarantine."""
        if idx in self.quarantined_indices:
            self.quarantined_indices.remove(idx)
            self.current_weights[idx] = self.original_weights.get(idx, 1.0)
            if idx in self.quarantine_start:
                del self.quarantine_start[idx]

    def apply_recovery(self, round_num: int):
        """
        Gradually restore weights for quarantined gradients.

        Exponential recovery: w(t+1) = w(t) × (1 + α)

        Args:
            round_num: Current round number
        """
        for idx in list(self.quarantined_indices):
            current = self.current_weights[idx]
            original = self.original_weights[idx]

            # Exponential recovery
            new_weight = current * (1 + self.recovery_rate)

            # Cap at original weight
            if new_weight >= original:
                new_weight = original
                self.release_gradient(idx)  # Fully recovered

            self.current_weights[idx] = new_weight

    def get_weight(self, idx: int, default: float = 1.0) -> float:
        """
        Get current weight for gradient.

        Args:
            idx: Gradient index
            default: Default weight if not quarantined

        Returns:
            Current weight
        """
        return self.current_weights.get(idx, default)

    def get_quarantine_duration(self, idx: int, current_round: int) -> Optional[int]:
        """
        Get how long gradient has been quarantined.

        Args:
            idx: Gradient index
            current_round: Current round number

        Returns:
            Duration in rounds, or None if not quarantined
        """
        if idx not in self.quarantine_start:
            return None
        return current_round - self.quarantine_start[idx]

    def get_stats(self) -> Dict:
        """Get quarantine statistics."""
        return {
            "quarantined_count": len(self.quarantined_indices),
            "average_weight": (
                np.mean(list(self.current_weights.values()))
                if self.current_weights
                else 1.0
            ),
            "min_weight": (
                np.min(list(self.current_weights.values()))
                if self.current_weights
                else 1.0
            ),
            "max_weight": (
                np.max(list(self.current_weights.values()))
                if self.current_weights
                else 1.0
            ),
            "quarantined_indices": list(self.quarantined_indices),
        }

    def reset(self):
        """Reset quarantine state."""
        self.quarantined_indices.clear()
        self.current_weights.clear()
        self.original_weights.clear()
        self.quarantine_start.clear()


class SelfHealingMechanism:
    """
    Self-healing mechanism for automatic recovery from high Byzantine activity.

    Multi-phase recovery protocol:
        Phase 1 (DETECTION): Identify when BFT > tolerance
        Phase 2 (QUARANTINE): Reduce Byzantine gradient weights
        Phase 3 (MONITORING): Observe behavior under reduced influence
        Phase 4 (RECOVERY): Gradually restore weights
        Phase 5 (NORMAL): Return to normal operation when BFT < threshold

    Example:
        >>> healer = SelfHealingMechanism()
        >>> scores = [0.85, 0.90, 0.15, 0.20, 0.88]  # 40% Byzantine
        >>> decisions, details = healer.process_batch(scores, round_num=0)
        >>> print(details["is_healing"])  # False (below tolerance)
        >>>
        >>> # Simulate attack
        >>> scores = [0.15, 0.20, 0.10, 0.18, 0.12]  # 100% Byzantine
        >>> decisions, details = healer.process_batch(scores, round_num=1)
        >>> print(details["is_healing"])  # True (healing activated)
    """

    def __init__(self, config: Optional[HealingConfig] = None):
        """
        Initialize self-healing mechanism.

        Args:
            config: Healing configuration parameters
        """
        self.config = config or HealingConfig()

        # Components
        self.bft_estimator = BFTEstimator(
            threshold=self.config.score_threshold,
            window_size=self.config.window_size,
        )
        self.quarantine = GradientQuarantine(
            quarantine_weight=self.config.quarantine_weight,
            recovery_rate=self.config.recovery_rate,
        )

        # State
        self.is_healing = False
        self.healing_start_round: Optional[int] = None
        self.adaptive_threshold = self.config.score_threshold
        self.low_bft_rounds = 0  # Consecutive rounds with low BFT

        # Statistics
        self.stats = {
            "total_rounds": 0,
            "healing_activations": 0,
            "total_healing_rounds": 0,
            "max_bft_observed": 0.0,
            "gradients_quarantined": 0,
            "gradients_released": 0,
        }

    def process_batch(
        self,
        scores: List[float],
        round_num: int,
    ) -> Tuple[List[str], Dict]:
        """
        Process batch of gradients with self-healing.

        Args:
            scores: Detection scores for gradients
            round_num: Current round number

        Returns:
            (decisions, details) where:
                decisions: List of "honest" or "byzantine" per gradient
                details: Healing state and statistics
        """
        # Update BFT estimation
        self.bft_estimator.update(scores)
        current_bft = self.bft_estimator.estimate_bft()

        # Update stats
        self.stats["total_rounds"] += 1
        self.stats["max_bft_observed"] = max(
            self.stats["max_bft_observed"], current_bft
        )

        # Check if healing needed
        if not self.is_healing and current_bft > self.config.bft_tolerance:
            self._activate_healing(round_num)
            self.low_bft_rounds = 0

        # Check if healing can stop (sustained low BFT)
        if self.is_healing:
            if current_bft < self.config.healing_threshold:
                self.low_bft_rounds += 1
                if self.low_bft_rounds >= self.config.sustained_rounds:
                    self._deactivate_healing()
            else:
                self.low_bft_rounds = 0  # Reset counter

        # Make decisions
        decisions = []
        for idx, score in enumerate(scores):
            # Use adaptive threshold if healing
            threshold = (
                self.adaptive_threshold
                if self.is_healing
                else self.config.score_threshold
            )

            decision = "honest" if score >= threshold else "byzantine"
            decisions.append(decision)

            # Quarantine if healing and Byzantine
            if self.is_healing and decision == "byzantine":
                self.quarantine.quarantine_gradient(idx, round_num)
                self.stats["gradients_quarantined"] += 1

        # Apply recovery to quarantined gradients
        if self.is_healing:
            initial_count = len(self.quarantine.quarantined_indices)
            self.quarantine.apply_recovery(round_num)
            final_count = len(self.quarantine.quarantined_indices)
            self.stats["gradients_released"] += initial_count - final_count
            self.stats["total_healing_rounds"] += 1

        # Prepare details
        details = {
            "is_healing": self.is_healing,
            "current_bft": current_bft,
            "adaptive_threshold": self.adaptive_threshold,
            "quarantine_stats": self.quarantine.get_stats(),
            "healing_round": (
                round_num - self.healing_start_round if self.is_healing else None
            ),
            "low_bft_rounds": self.low_bft_rounds,
            "score_distribution": self.bft_estimator.get_score_distribution(),
        }

        return (decisions, details)

    def _activate_healing(self, round_num: int):
        """Activate healing protocol."""
        self.is_healing = True
        self.healing_start_round = round_num
        self.stats["healing_activations"] += 1

        # Adjust threshold (be more aggressive)
        current_bft = self.bft_estimator.estimate_bft()
        delta = 0.1 * (current_bft - self.config.bft_tolerance)
        self.adaptive_threshold = self.config.score_threshold - min(delta, 0.1)

    def _deactivate_healing(self):
        """Deactivate healing protocol and return to normal."""
        self.is_healing = False
        self.healing_start_round = None
        self.adaptive_threshold = self.config.score_threshold
        self.low_bft_rounds = 0

        # Release all quarantined gradients
        for idx in list(self.quarantine.quarantined_indices):
            self.quarantine.release_gradient(idx)
            self.stats["gradients_released"] += 1

    def get_healing_state(self) -> Dict:
        """Get current healing state."""
        return {
            "is_healing": self.is_healing,
            "healing_round": (
                self.stats["total_rounds"] - self.healing_start_round
                if self.is_healing and self.healing_start_round is not None
                else None
            ),
            "current_bft": self.bft_estimator.estimate_bft(),
            "adaptive_threshold": self.adaptive_threshold,
            "low_bft_streak": self.low_bft_rounds,
        }

    def get_stats(self) -> Dict:
        """Get healing statistics."""
        stats = self.stats.copy()
        stats["is_currently_healing"] = self.is_healing
        stats["quarantine_active_count"] = len(self.quarantine.quarantined_indices)
        stats["current_bft_estimate"] = self.bft_estimator.estimate_bft()
        stats["healing_activation_rate"] = (
            self.stats["healing_activations"] / max(1, self.stats["total_rounds"])
        )
        return stats

    def reset(self):
        """Reset healing mechanism to initial state."""
        self.is_healing = False
        self.healing_start_round = None
        self.adaptive_threshold = self.config.score_threshold
        self.low_bft_rounds = 0
        self.bft_estimator = BFTEstimator(
            threshold=self.config.score_threshold,
            window_size=self.config.window_size,
        )
        self.quarantine = GradientQuarantine(
            quarantine_weight=self.config.quarantine_weight,
            recovery_rate=self.config.recovery_rate,
        )
        self.stats = {
            "total_rounds": 0,
            "healing_activations": 0,
            "total_healing_rounds": 0,
            "max_bft_observed": 0.0,
            "gradients_quarantined": 0,
            "gradients_released": 0,
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "HEALING" if self.is_healing else "NORMAL"
        bft = self.bft_estimator.estimate_bft()
        return (
            f"SelfHealingMechanism(status={status}, "
            f"BFT={bft:.1%}, "
            f"quarantined={len(self.quarantine.quarantined_indices)})"
        )
