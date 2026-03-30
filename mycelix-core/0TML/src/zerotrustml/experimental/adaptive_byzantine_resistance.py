# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Adaptive Byzantine Resistance (Phase 5 Enhancement)

Multi-level reputation system with dynamic threshold adjustment,
reputation recovery, and Byzantine-resistant aggregation algorithms.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio

# Credits integration (optional)
try:
    from .zerotrustml_credits_integration import ZeroTrustMLCreditsIntegration
except ImportError:
    ZeroTrustMLCreditsIntegration = None


class ReputationLevel(Enum):
    """Reputation threshold levels"""
    BLACKLISTED = 0.0  # Permanently blocked
    CRITICAL = 0.3     # Under review
    WARNING = 0.5      # Degraded service
    NORMAL = 0.7       # Full service
    TRUSTED = 0.9      # Enhanced privileges
    ELITE = 0.95       # Top contributors


@dataclass
class GradientReputationMetrics:
    """Per-gradient reputation tracking"""
    gradient_id: str
    timestamp: datetime
    validation_passed: bool
    pogq_score: float
    anomaly_score: float
    contribution_value: float  # How much it improved the model


@dataclass
class NodeReputationProfile:
    """Comprehensive node reputation profile"""
    node_id: int

    # Multi-level reputation scores
    overall_reputation: float = 0.7  # Aggregated score
    gradient_reputation: float = 0.7  # Quality of gradients
    network_reputation: float = 0.7  # Network behavior
    temporal_reputation: float = 0.7  # Time-based reliability

    # Statistics
    total_gradients: int = 0
    valid_gradients: int = 0
    invalid_gradients: int = 0
    contributions: int = 0

    # Temporal tracking
    first_seen: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Recovery tracking
    recovery_attempts: int = 0
    last_recovery: Optional[datetime] = None

    # Gradient history (last 100)
    gradient_history: List[GradientReputationMetrics] = field(default_factory=list)


class DynamicThresholdManager:
    """
    Manages dynamic reputation thresholds based on network conditions

    Adjusts thresholds based on:
    - Overall network health
    - Attack detection rate
    - False positive rate
    - Network capacity
    """

    def __init__(
        self,
        base_threshold: float = 0.5,
        min_threshold: float = 0.3,
        max_threshold: float = 0.8
    ):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        # Network state
        self.total_validations = 0
        self.byzantine_detections = 0
        self.false_positives = 0

        # Adaptive parameters
        self.current_threshold = base_threshold
        self.adjustment_rate = 0.01

    def update_network_state(
        self,
        validation_passed: bool,
        is_false_positive: bool = False
    ):
        """Update network state and adjust threshold"""
        self.total_validations += 1

        if not validation_passed:
            self.byzantine_detections += 1

        if is_false_positive:
            self.false_positives += 1

        # Adjust threshold based on network health
        self._adjust_threshold()

    def _adjust_threshold(self):
        """Dynamically adjust threshold based on network conditions"""
        if self.total_validations < 100:
            return  # Not enough data

        # Calculate metrics
        attack_rate = self.byzantine_detections / self.total_validations
        false_positive_rate = self.false_positives / max(1, self.byzantine_detections)

        # Adjust threshold
        if attack_rate > 0.3:  # High attack rate -> stricter threshold
            self.current_threshold = min(
                self.max_threshold,
                self.current_threshold + self.adjustment_rate
            )
        elif attack_rate < 0.05 and false_positive_rate < 0.01:
            # Low attack rate + low false positives -> relax threshold
            self.current_threshold = max(
                self.min_threshold,
                self.current_threshold - self.adjustment_rate
            )

    def get_threshold(self, node_history: Optional[NodeReputationProfile] = None) -> float:
        """Get threshold for a specific node (or global)"""
        threshold = self.current_threshold

        if node_history:
            # Adjust based on node's track record
            if node_history.consecutive_successes > 50:
                threshold *= 0.9  # Relax threshold for trusted nodes
            elif node_history.consecutive_failures > 5:
                threshold *= 1.2  # Stricter threshold for problematic nodes

        return np.clip(threshold, self.min_threshold, self.max_threshold)


class ReputationRecoveryManager:
    """
    Manages reputation recovery for nodes with degraded reputation

    Allows nodes to recover from temporary issues through:
    - Probationary periods with stricter monitoring
    - Gradual reputation restoration
    - Time-based reputation decay for old failures
    """

    def __init__(
        self,
        recovery_window: int = 100,  # Number of gradients to evaluate
        recovery_threshold: float = 0.95,  # Success rate needed
        max_recovery_attempts: int = 3
    ):
        self.recovery_window = recovery_window
        self.recovery_threshold = recovery_threshold
        self.max_recovery_attempts = max_recovery_attempts

    def can_attempt_recovery(self, profile: NodeReputationProfile) -> bool:
        """Check if node is eligible for recovery attempt"""
        if profile.recovery_attempts >= self.max_recovery_attempts:
            return False

        # Must wait at least 24 hours between recovery attempts
        if profile.last_recovery:
            time_since_last = datetime.now() - profile.last_recovery
            if time_since_last < timedelta(hours=24):
                return False

        # Must have been blacklisted or in critical state
        if profile.overall_reputation > ReputationLevel.CRITICAL.value:
            return False

        return True

    def start_recovery(self, profile: NodeReputationProfile):
        """Start a recovery probationary period"""
        profile.recovery_attempts += 1
        profile.last_recovery = datetime.now()
        # Reset consecutive counters
        profile.consecutive_failures = 0
        profile.consecutive_successes = 0

    def evaluate_recovery(self, profile: NodeReputationProfile) -> Tuple[bool, float]:
        """
        Evaluate recovery progress

        Returns:
            (should_recover, new_reputation)
        """
        recent_gradients = profile.gradient_history[-self.recovery_window:]

        if len(recent_gradients) < self.recovery_window:
            return False, profile.overall_reputation  # Not enough data yet

        # Calculate success rate in recovery window
        successes = sum(1 for g in recent_gradients if g.validation_passed)
        success_rate = successes / len(recent_gradients)

        if success_rate >= self.recovery_threshold:
            # Recovery successful!
            target = max(profile.overall_reputation, 0.7)
            return True, target

        return False, profile.overall_reputation


class ByzantineResistantAggregator:
    """
    Byzantine-resistant gradient aggregation algorithms

    Implements:
    - Krum: Select most representative gradient
    - Trimmed Mean: Remove outliers and average
    - Coordinate-wise Median: Median per parameter
    """

    @staticmethod
    def krum(
        gradients: List[np.ndarray],
        reputations: List[float],
        num_byzantine: int = 2
    ) -> np.ndarray:
        """
        Krum aggregation: Select gradient with smallest distance to neighbors

        Args:
            gradients: List of gradient arrays
            reputations: Reputation scores for each gradient
            num_byzantine: Estimated number of Byzantine nodes

        Returns:
            Selected gradient (most representative)
        """
        n = len(gradients)
        k = n - num_byzantine - 2  # Number of neighbors to consider

        if k <= 0:
            # Fallback to reputation-weighted average
            return ByzantineResistantAggregator.weighted_average(gradients, reputations)

        # Flatten gradients for distance calculation
        flat_gradients = [g.flatten() for g in gradients]

        # Calculate pairwise distances
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(flat_gradients[i] - flat_gradients[j])
                    distances.append(dist)

            # Sum of k smallest distances (weighted by reputation)
            distances = np.array(sorted(distances)[:k])
            score = np.sum(distances) * reputations[i]  # Weight by reputation
            scores.append(score)

        # Select gradient with minimum score
        best_idx = np.argmin(scores)
        return gradients[best_idx]

    @staticmethod
    def trimmed_mean(
        gradients: List[np.ndarray],
        reputations: List[float],
        trim_ratio: float = 0.2
    ) -> np.ndarray:
        """
        Trimmed mean: Remove extreme values and average

        Args:
            gradients: List of gradient arrays
            reputations: Reputation scores
            trim_ratio: Fraction of values to trim from each end

        Returns:
            Trimmed mean gradient
        """
        # Stack gradients
        stacked = np.stack([g.flatten() for g in gradients])

        # Weight by reputation
        weights = np.array(reputations) / np.sum(reputations)

        # Sort along gradient axis
        sorted_indices = np.argsort(stacked, axis=0)

        # Calculate trim size
        n = len(gradients)
        trim_size = int(n * trim_ratio)

        # Estimate Byzantine count from low-reputation participants
        estimated_byzantine = sum(1 for rep in reputations if rep < 0.5)
        trim_size = min(max(trim_size, estimated_byzantine), (n - 1) // 2)

        if trim_size == 0:
            # No trimming, just weighted average
            return np.average(stacked, axis=0, weights=weights).reshape(gradients[0].shape)

        # If trimming removes all entries, fall back to median
        if trim_size * 2 >= n:
            return np.median(stacked, axis=0).reshape(gradients[0].shape)

        # Sort gradients per-coordinate and remove extremes symmetrically
        sorted_stack = np.take_along_axis(stacked, sorted_indices, axis=0)
        trimmed = sorted_stack[trim_size:n - trim_size]

        # Use simple mean for the surviving coordinates (weights complicate axes)
        result = np.mean(trimmed, axis=0)
        return result.reshape(gradients[0].shape)

    @staticmethod
    def coordinate_median(
        gradients: List[np.ndarray],
        reputations: List[float]
    ) -> np.ndarray:
        """
        Coordinate-wise median: Median of each parameter independently

        Args:
            gradients: List of gradient arrays
            reputations: Reputation scores (used for weighted median)

        Returns:
            Median gradient
        """
        # Stack gradients
        stacked = np.stack([g.flatten() for g in gradients])

        # Weighted median (approximate with weighted percentile)
        weights = np.array(reputations)
        weights = weights / np.sum(weights)

        # Calculate weighted median per coordinate
        result = np.array([
            ByzantineResistantAggregator._weighted_median(stacked[:, i], weights)
            for i in range(stacked.shape[1])
        ])

        return result.reshape(gradients[0].shape)

    @staticmethod
    def weighted_average(
        gradients: List[np.ndarray],
        reputations: List[float]
    ) -> np.ndarray:
        """Simple reputation-weighted average (baseline)"""
        weights = np.array(reputations) / np.sum(reputations)
        stacked = np.stack([g.flatten() for g in gradients])
        result = np.average(stacked, axis=0, weights=weights)
        return result.reshape(gradients[0].shape)

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted median"""
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, 0.5)

        return sorted_values[median_idx]


class AdaptiveByzantineResistance:
    """
    Complete adaptive Byzantine resistance system

    Combines:
    - Multi-level reputation tracking
    - Dynamic threshold adjustment
    - Reputation recovery
    - Byzantine-resistant aggregation
    """

    def __init__(self, credits_integration: Optional[Any] = None):
        self.node_profiles: Dict[int, NodeReputationProfile] = {}
        self.threshold_manager = DynamicThresholdManager()
        self.recovery_manager = ReputationRecoveryManager()
        self.aggregator = ByzantineResistantAggregator()
        self.credits_integration = credits_integration  # Optional credits system

    def get_or_create_profile(self, node_id: int) -> NodeReputationProfile:
        """Get existing profile or create new one"""
        if node_id not in self.node_profiles:
            self.node_profiles[node_id] = NodeReputationProfile(node_id=node_id)
        return self.node_profiles[node_id]

    def update_reputation(
        self,
        node_id: int,
        gradient: np.ndarray,
        validation_passed: bool,
        pogq_score: float,
        anomaly_score: float = 0.0,
        contribution_value: float = 0.0,
        detector_node_id: Optional[int] = None
    ):
        """Update node reputation based on gradient validation"""
        profile = self.get_or_create_profile(node_id)

        # Track previous reputation level for Byzantine detection
        previous_level = self.get_reputation_level(node_id)

        # Create gradient metrics
        gradient_metrics = GradientReputationMetrics(
            gradient_id=f"{node_id}_{profile.total_gradients}",
            timestamp=datetime.now(),
            validation_passed=validation_passed,
            pogq_score=pogq_score,
            anomaly_score=anomaly_score,
            contribution_value=contribution_value
        )

        # Update history
        profile.gradient_history.append(gradient_metrics)
        if len(profile.gradient_history) > 100:
            profile.gradient_history.pop(0)

        # Update statistics
        profile.total_gradients += 1
        profile.last_activity = datetime.now()

        if validation_passed:
            profile.valid_gradients += 1
            profile.consecutive_successes += 1
            profile.consecutive_failures = 0
        else:
            profile.invalid_gradients += 1
            profile.consecutive_failures += 1
            profile.consecutive_successes = 0

        # Update multi-level reputation
        self._update_multi_level_reputation(profile)

        # Update network threshold
        self.threshold_manager.update_network_state(validation_passed)

        # Issue Byzantine detection credits if reputation dropped to CRITICAL/BLACKLISTED
        if self.credits_integration and detector_node_id is not None:
            current_level = self.get_reputation_level(node_id)

            # Byzantine detected if transitioned to CRITICAL or BLACKLISTED
            if current_level in [ReputationLevel.CRITICAL, ReputationLevel.BLACKLISTED]:
                if previous_level not in [ReputationLevel.CRITICAL, ReputationLevel.BLACKLISTED]:
                    # This is a new Byzantine detection (reputation just dropped)
                    detector_reputation = self.get_or_create_profile(detector_node_id).overall_reputation
                    asyncio.create_task(self.credits_integration.on_byzantine_detection(
                        detector_node_id=f"node_{detector_node_id}",
                        detected_node_id=f"node_{node_id}",
                        reputation_level=self._reputation_score_to_level(detector_reputation),
                        evidence={
                            "consecutive_failures": profile.consecutive_failures,
                            "validation_passed": validation_passed,
                            "pogq_score": pogq_score,
                            "anomaly_score": anomaly_score,
                            "previous_reputation": previous_level.value,
                            "current_reputation": current_level.value
                        }
                    ))

    def _update_multi_level_reputation(self, profile: NodeReputationProfile):
        """Update all reputation levels"""
        # Gradient reputation (quality of contributions)
        if profile.total_gradients > 0:
            profile.gradient_reputation = profile.valid_gradients / profile.total_gradients

        # Network reputation (responsiveness, uptime)
        time_active = (datetime.now() - profile.first_seen).total_seconds()
        if time_active > 0:
            activity_rate = profile.total_gradients / (time_active / 3600)  # Per hour
            profile.network_reputation = min(1.0, activity_rate / 10.0)  # Target: 10/hour

        # Temporal reputation (recent behavior)
        recent_window = profile.gradient_history[-20:]  # Last 20 gradients
        if recent_window:
            recent_success_rate = sum(1 for g in recent_window if g.validation_passed) / len(recent_window)
            profile.temporal_reputation = recent_success_rate

        # Overall reputation (weighted average)
        profile.overall_reputation = (
            0.5 * profile.gradient_reputation +
            0.2 * profile.network_reputation +
            0.3 * profile.temporal_reputation
        )

        # Apply penalties for consecutive failures
        if profile.consecutive_failures > 10:
            profile.overall_reputation *= 0.5  # Severe penalty
        elif profile.consecutive_failures > 5:
            profile.overall_reputation *= 0.7

    def get_reputation_level(self, node_id: int) -> ReputationLevel:
        """Get categorical reputation level"""
        profile = self.get_or_create_profile(node_id)
        rep = profile.overall_reputation

        if rep >= ReputationLevel.ELITE.value:
            return ReputationLevel.ELITE
        elif rep >= ReputationLevel.TRUSTED.value:
            return ReputationLevel.TRUSTED
        elif rep >= ReputationLevel.NORMAL.value:
            return ReputationLevel.NORMAL
        elif rep >= ReputationLevel.WARNING.value:
            return ReputationLevel.WARNING
        elif rep >= ReputationLevel.CRITICAL.value:
            return ReputationLevel.CRITICAL
        else:
            return ReputationLevel.BLACKLISTED

    def _reputation_score_to_level(self, reputation_score: float) -> str:
        """Convert reputation score (0.0-1.0) to level string for credits system"""
        if reputation_score < 0.30:
            return "BLACKLISTED"
        elif reputation_score < 0.50:
            return "CRITICAL"
        elif reputation_score < 0.70:
            return "WARNING"
        elif reputation_score < 0.90:
            return "NORMAL"
        elif reputation_score < 0.95:
            return "TRUSTED"
        else:
            return "ELITE"

    async def aggregate_gradients(
        self,
        gradients: Dict[int, np.ndarray],
        algorithm: str = "krum"
    ) -> np.ndarray:
        """
        Aggregate gradients using Byzantine-resistant algorithm

        Args:
            gradients: Dict mapping node_id -> gradient
            algorithm: "krum", "trimmed_mean", "median", or "weighted"

        Returns:
            Aggregated gradient
        """
        if not gradients:
            raise ValueError("No gradients to aggregate")

        # Get reputations
        node_ids = list(gradients.keys())
        gradient_arrays = list(gradients.values())
        reputations = [self.get_or_create_profile(nid).overall_reputation for nid in node_ids]

        # Select aggregation algorithm
        if algorithm == "krum":
            return self.aggregator.krum(gradient_arrays, reputations, num_byzantine=2)
        elif algorithm == "trimmed_mean":
            return self.aggregator.trimmed_mean(gradient_arrays, reputations, trim_ratio=0.2)
        elif algorithm == "median":
            return self.aggregator.coordinate_median(gradient_arrays, reputations)
        else:  # weighted average (baseline)
            return self.aggregator.weighted_average(gradient_arrays, reputations)

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        if not self.node_profiles:
            return {"total_nodes": 0}

        reputations = [p.overall_reputation for p in self.node_profiles.values()]

        return {
            "total_nodes": len(self.node_profiles),
            "avg_reputation": np.mean(reputations),
            "median_reputation": np.median(reputations),
            "min_reputation": np.min(reputations),
            "max_reputation": np.max(reputations),
            "current_threshold": self.threshold_manager.current_threshold,
            "byzantine_detection_rate": (
                self.threshold_manager.byzantine_detections /
                max(1, self.threshold_manager.total_validations)
            ),
            "reputation_distribution": {
                level.name: sum(
                    1 for p in self.node_profiles.values()
                    if self.get_reputation_level(p.node_id) == level
                )
                for level in ReputationLevel
            }
        }
