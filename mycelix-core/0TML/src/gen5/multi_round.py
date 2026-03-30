# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen 5 Layer 6: Multi-Round Temporal Detection
==============================================

Temporal pattern detection across multiple FL rounds.

Key Features:
- Sleeper agent detection (honest → Byzantine)
- Coordination detection (synchronized attacks)
- Reputation evolution tracking (Bayesian)
- Temporal anomaly detection (statistical)

Algorithm:
    1. Track per-agent score history over rounds
    2. Detect behavior changes (CUSUM change point detection)
    3. Detect coordination (cross-correlation analysis)
    4. Update reputation (Beta-Binomial Bayesian updates)
    5. Flag temporal anomalies (z-score outliers)

Author: Luminous Dynamics
Date: November 12, 2025
"""

from typing import Dict, List, Tuple, Optional, Deque
from collections import deque, defaultdict
from dataclasses import dataclass
import numpy as np


@dataclass
class MultiRoundConfig:
    """Configuration for multi-round temporal detection."""

    window_size: int = 50  # History window (rounds)
    sleeper_threshold: float = 0.3  # Behavior change threshold
    coordination_threshold: float = 0.7  # Correlation threshold
    z_threshold: float = 3.0  # Anomaly detection (3-sigma)
    lag_tolerance: int = 2  # Max lag for coordination (rounds)
    cusum_slack: float = 0.1  # CUSUM slack parameter (k)
    cusum_threshold: float = 5.0  # CUSUM detection threshold (h)
    alpha_prior: float = 1.0  # Beta distribution α prior
    beta_prior: float = 1.0  # Beta distribution β prior


@dataclass
class TemporalAnomaly:
    """Temporal anomaly detection result."""

    agent_id: str
    round: int
    anomaly_type: str  # "sleeper" | "spike" | "drift"
    magnitude: float
    details: Dict


class ReputationTracker:
    """
    Bayesian reputation tracking using Beta-Binomial conjugate prior.

    Reputation ∈ [0.0, 1.0]:
    - 1.0: Fully trusted (consistently honest)
    - 0.5: Neutral (no history or equal evidence)
    - 0.0: Fully distrusted (consistently Byzantine)

    Update Rule:
        α_new = α_old + (score >= 0.5)  # Honest evidence
        β_new = β_old + (score < 0.5)   # Byzantine evidence
        reputation = α / (α + β)
    """

    def __init__(
        self, alpha_prior: float = 1.0, beta_prior: float = 1.0
    ):
        """
        Initialize with Beta(α, β) prior.

        Args:
            alpha_prior: Prior belief in honesty (default: 1.0 = uniform)
            beta_prior: Prior belief in Byzantine (default: 1.0 = uniform)
        """
        self.agents: Dict[str, Tuple[float, float]] = {}
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def update_reputation(self, agent_id: str, score: float):
        """
        Update reputation with new evidence.

        Args:
            agent_id: Agent identifier
            score: Detection score ∈ [0.0, 1.0]
                  - score >= 0.5: Honest evidence
                  - score < 0.5: Byzantine evidence
        """
        if agent_id not in self.agents:
            self.agents[agent_id] = (self.alpha_prior, self.beta_prior)

        alpha, beta = self.agents[agent_id]

        # Bayesian update
        if score >= 0.5:  # Honest evidence
            alpha += 1.0
        else:  # Byzantine evidence
            beta += 1.0

        self.agents[agent_id] = (alpha, beta)

    def get_reputation(self, agent_id: str) -> float:
        """
        Get current reputation (posterior mean).

        Returns:
            Reputation ∈ [0.0, 1.0]
            - Returns 0.5 for unknown agents (neutral)
        """
        if agent_id not in self.agents:
            return 0.5  # Neutral for new agents

        alpha, beta = self.agents[agent_id]
        return alpha / (alpha + beta)

    def get_reputation_variance(self, agent_id: str) -> float:
        """
        Get reputation uncertainty (posterior variance).

        Returns:
            Variance of Beta(α, β) posterior
            - High variance: Uncertain reputation
            - Low variance: Confident reputation
        """
        if agent_id not in self.agents:
            return 0.25  # Maximum variance for Beta(1, 1)

        alpha, beta = self.agents[agent_id]
        # Variance of Beta distribution
        variance = (alpha * beta) / (
            (alpha + beta) ** 2 * (alpha + beta + 1)
        )
        return variance

    def get_reputation_confidence_interval(
        self, agent_id: str, alpha: float = 0.1
    ) -> Tuple[float, float]:
        """
        Get reputation confidence interval.

        Uses Beta distribution quantiles.

        Args:
            agent_id: Agent identifier
            alpha: Significance level (default: 0.1 for 90% CI)

        Returns:
            (lower, upper) confidence bounds
        """
        if agent_id not in self.agents:
            return (0.0, 1.0)  # Wide interval for unknown

        a, b = self.agents[agent_id]

        # Approximate Beta quantiles using normal approximation
        # For large α+β, Beta(α,β) ≈ N(μ, σ²)
        mean = a / (a + b)
        variance = (a * b) / ((a + b) ** 2 * (a + b + 1))
        std = np.sqrt(variance)

        # 90% CI ≈ mean ± 1.645 × std
        z_score = 1.645  # For α=0.1 (90% CI)
        lower = max(0.0, mean - z_score * std)
        upper = min(1.0, mean + z_score * std)

        return (lower, upper)


class MultiRoundDetector:
    """
    Multi-round temporal detection system.

    Tracks agent behavior over multiple FL rounds to detect:
    - Sleeper agents (sudden behavior changes)
    - Coordinated attacks (synchronized Byzantine behavior)
    - Temporal anomalies (statistical outliers)
    - Reputation evolution (Bayesian trust tracking)

    Example:
        >>> detector = MultiRoundDetector(
        ...     window_size=50,
        ...     sleeper_threshold=0.3,
        ...     coordination_threshold=0.7
        ... )
        >>>
        >>> # Track over multiple rounds
        >>> for round_idx in range(100):
        ...     for agent_id, gradient in gradients:
        ...         score = ensemble.score(gradient)
        ...         detector.update_agent_history(agent_id, round_idx, score)
        >>>
        >>> # Detect anomalies
        >>> sleepers = detector.get_sleeper_agents()
        >>> coordinated = detector.get_coordinated_groups()
        >>> reputations = detector.get_all_reputations()
    """

    def __init__(self, config: Optional[MultiRoundConfig] = None):
        """
        Initialize multi-round detector.

        Args:
            config: Configuration parameters
        """
        self.config = config or MultiRoundConfig()

        # History tracking
        self.agent_histories: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.config.window_size)
        )
        self.agent_rounds: Dict[str, Deque[int]] = defaultdict(
            lambda: deque(maxlen=self.config.window_size)
        )

        # Reputation tracking
        self.reputation_tracker = ReputationTracker(
            alpha_prior=self.config.alpha_prior,
            beta_prior=self.config.beta_prior,
        )

        # CUSUM state for sleeper detection
        self.cusum_state: Dict[str, float] = defaultdict(float)

        # Statistics
        self.stats = {
            "total_rounds": 0,
            "total_updates": 0,
            "sleepers_detected": 0,
            "coordination_groups": 0,
            "anomalies_detected": 0,
        }

    def update_agent_history(
        self, agent_id: str, round_idx: int, score: float
    ):
        """
        Update agent's history with new score.

        Args:
            agent_id: Agent identifier
            round_idx: Current round number
            score: Detection score for this round
        """
        self.agent_histories[agent_id].append(score)
        self.agent_rounds[agent_id].append(round_idx)

        # Update reputation
        self.reputation_tracker.update_reputation(agent_id, score)

        # Update CUSUM for sleeper detection
        self._update_cusum(agent_id, score)

        # Update stats
        self.stats["total_updates"] += 1
        if round_idx > self.stats["total_rounds"]:
            self.stats["total_rounds"] = round_idx

    def _update_cusum(self, agent_id: str, score: float):
        """
        Update CUSUM statistic for sleeper detection.

        CUSUM detects mean shifts in time series.

        Algorithm:
            CUSUM_t = max(0, CUSUM_{t-1} + (x_t - μ - k))

        Where:
            x_t: Current score
            μ: Baseline mean (FIXED from first 20 rounds)
            k: Slack parameter (allowance for noise)
        """
        history = list(self.agent_histories[agent_id])

        if len(history) < 10:  # Need minimum history
            return

        # Use FIXED baseline from first 20 rounds (or max available if less)
        # Once we have 20 rounds, the baseline never changes
        baseline_window = min(20, len(history))
        baseline_mean = np.mean(history[:baseline_window])

        # CUSUM update (detect shift away from baseline)
        deviation = abs(score - baseline_mean) - self.config.cusum_slack

        current_cusum = self.cusum_state[agent_id]
        self.cusum_state[agent_id] = max(0.0, current_cusum + deviation)

    def detect_sleeper_agent(
        self, agent_id: str
    ) -> Tuple[bool, float, Optional[int]]:
        """
        Detect if agent is a sleeper (sudden behavior change).

        Uses CUSUM change point detection.

        Returns:
            (is_sleeper, change_magnitude, change_point_round)
        """
        if agent_id not in self.agent_histories:
            return (False, 0.0, None)

        history = list(self.agent_histories[agent_id])
        rounds = list(self.agent_rounds[agent_id])

        if len(history) < 20:  # Need sufficient history
            return (False, 0.0, None)

        # Check CUSUM threshold
        cusum_value = self.cusum_state.get(agent_id, 0.0)

        if cusum_value > self.config.cusum_threshold:
            # Detect change point (when CUSUM started increasing)
            # Simple approach: find point where mean shifts
            mid_point = len(history) // 2
            mean_first_half = np.mean(history[:mid_point])
            mean_second_half = np.mean(history[mid_point:])

            change_magnitude = abs(mean_second_half - mean_first_half)

            if change_magnitude > self.config.sleeper_threshold:
                change_point_round = rounds[mid_point]
                self.stats["sleepers_detected"] += 1
                return (True, change_magnitude, change_point_round)

        return (False, 0.0, None)

    def detect_coordination(
        self, agent_ids: Optional[List[str]] = None
    ) -> List[Tuple[str, str, float]]:
        """
        Detect coordinated attacks (agents with correlated behavior).

        Uses Pearson correlation on score time series.

        Args:
            agent_ids: List of agents to check (default: all agents)

        Returns:
            List of (agent1, agent2, correlation) tuples
            where correlation > coordination_threshold
        """
        if agent_ids is None:
            agent_ids = list(self.agent_histories.keys())

        coordinated_pairs = []

        # Pairwise correlation
        for i, agent1 in enumerate(agent_ids):
            for agent2 in agent_ids[i + 1 :]:
                # Get overlapping time series
                history1 = list(self.agent_histories[agent1])
                history2 = list(self.agent_histories[agent2])

                # Need sufficient overlap
                min_len = min(len(history1), len(history2))
                if min_len < 10:
                    continue

                # Align to most recent common rounds
                scores1 = history1[-min_len:]
                scores2 = history2[-min_len:]

                # Compute Pearson correlation
                if len(scores1) >= 2:  # Need at least 2 points
                    correlation = np.corrcoef(scores1, scores2)[0, 1]

                    # Check threshold
                    if (
                        not np.isnan(correlation)
                        and correlation > self.config.coordination_threshold
                    ):
                        coordinated_pairs.append(
                            (agent1, agent2, correlation)
                        )

        if coordinated_pairs:
            self.stats["coordination_groups"] += len(coordinated_pairs)

        return coordinated_pairs

    def detect_temporal_anomaly(
        self, agent_id: str
    ) -> Tuple[bool, float]:
        """
        Detect statistical anomaly in agent behavior.

        Uses z-score (standard deviations from mean).

        Returns:
            (is_anomaly, z_score)
        """
        if agent_id not in self.agent_histories:
            return (False, 0.0)

        history = list(self.agent_histories[agent_id])

        if len(history) < 10:  # Need sufficient history
            return (False, 0.0)

        # Current score
        current_score = history[-1]

        # Historical statistics (exclude current)
        historical_scores = history[:-1]
        mean = np.mean(historical_scores)
        std = np.std(historical_scores)

        # Avoid division by zero
        if std < 1e-6:
            return (False, 0.0)

        # Z-score
        z_score = abs(current_score - mean) / std

        # Check threshold
        is_anomaly = z_score > self.config.z_threshold

        if is_anomaly:
            self.stats["anomalies_detected"] += 1

        return (is_anomaly, z_score)

    def get_sleeper_agents(self) -> List[Dict]:
        """
        Get all detected sleeper agents.

        Returns:
            List of sleeper agent info dicts
        """
        sleepers = []

        for agent_id in self.agent_histories.keys():
            is_sleeper, magnitude, change_point = self.detect_sleeper_agent(
                agent_id
            )

            if is_sleeper:
                sleepers.append(
                    {
                        "agent_id": agent_id,
                        "change_magnitude": magnitude,
                        "change_point_round": change_point,
                        "current_reputation": self.reputation_tracker.get_reputation(
                            agent_id
                        ),
                    }
                )

        return sleepers

    def get_coordinated_groups(self) -> List[Dict]:
        """
        Get all detected coordination groups.

        Returns:
            List of coordination group info dicts
        """
        pairs = self.detect_coordination()

        groups = []
        for agent1, agent2, correlation in pairs:
            groups.append(
                {
                    "agents": [agent1, agent2],
                    "correlation": correlation,
                    "reputation_agent1": self.reputation_tracker.get_reputation(
                        agent1
                    ),
                    "reputation_agent2": self.reputation_tracker.get_reputation(
                        agent2
                    ),
                }
            )

        return groups

    def get_all_reputations(self) -> Dict[str, float]:
        """Get reputation scores for all agents."""
        return {
            agent_id: self.reputation_tracker.get_reputation(agent_id)
            for agent_id in self.agent_histories.keys()
        }

    def get_agent_summary(self, agent_id: str) -> Dict:
        """
        Get comprehensive summary for an agent.

        Returns:
            Dict with history, reputation, anomalies
        """
        if agent_id not in self.agent_histories:
            return {"error": "Agent not found"}

        history = list(self.agent_histories[agent_id])
        rounds = list(self.agent_rounds[agent_id])

        is_sleeper, change_mag, change_point = self.detect_sleeper_agent(
            agent_id
        )
        is_anomaly, z_score = self.detect_temporal_anomaly(agent_id)

        reputation = self.reputation_tracker.get_reputation(agent_id)
        rep_variance = self.reputation_tracker.get_reputation_variance(
            agent_id
        )
        rep_ci = self.reputation_tracker.get_reputation_confidence_interval(
            agent_id
        )

        return {
            "agent_id": agent_id,
            "history_length": len(history),
            "rounds": rounds,
            "scores": history,
            "mean_score": np.mean(history) if history else 0.0,
            "std_score": np.std(history) if history else 0.0,
            "reputation": reputation,
            "reputation_variance": rep_variance,
            "reputation_ci": rep_ci,
            "is_sleeper": is_sleeper,
            "sleeper_change_magnitude": change_mag,
            "sleeper_change_point": change_point,
            "is_temporal_anomaly": is_anomaly,
            "anomaly_z_score": z_score,
        }

    def get_stats(self) -> Dict:
        """Get detection statistics."""
        stats = self.stats.copy()

        # Add derived metrics
        stats["total_agents"] = len(self.agent_histories)
        stats["avg_history_length"] = (
            np.mean([len(h) for h in self.agent_histories.values()])
            if self.agent_histories
            else 0.0
        )

        return stats

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "total_rounds": 0,
            "total_updates": 0,
            "sleepers_detected": 0,
            "coordination_groups": 0,
            "anomalies_detected": 0,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MultiRoundDetector("
            f"agents={len(self.agent_histories)}, "
            f"rounds={self.stats['total_rounds']}, "
            f"sleepers={self.stats['sleepers_detected']}, "
            f"coordinated={self.stats['coordination_groups']})"
        )
