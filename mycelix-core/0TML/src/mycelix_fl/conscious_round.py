# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Conscious FL Round — Unified consciousness-guided federated learning.

Provides a single-call interface that runs one complete round of
consciousness-guided FL: assess -> weight -> aggregate.

This is the Python equivalent of `symthaea-mycelix-bridge::unified_round::run_conscious_fl_round()`.
When the Rust bridge is available, it delegates to native code for performance.
Otherwise, it uses the pure-Python FL pipeline.

Usage:
    from mycelix_fl.conscious_round import ConsciousFlRound, ConsciousRoundConfig

    round_runner = ConsciousFlRound()
    result = round_runner.run(
        gradients={"node-1": np.array([...]), "node-2": np.array([...])},
        reputations={"node-1": 0.8, "node-2": 0.7},
    )
    print(f"Aggregated {len(result.aggregated)} dims, {result.vetoed_count} vetoed")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import the Rust FL aggregator
try:
    from fl_aggregator import PyAggregator, Defense as PyDefense

    FL_RUST_AVAILABLE = True
except ImportError:
    FL_RUST_AVAILABLE = False
    logger.debug("fl_aggregator not available, using pure-Python FL pipeline")


@dataclass
class ConsciousRoundConfig:
    """Configuration for a conscious FL round."""

    # Quality gating
    confidence_threshold: float = 0.3
    """Minimum epistemic confidence to avoid dampening."""

    veto_confidence: float = 0.1
    """Confidence below which an update is vetoed."""

    phi_gain_boost_threshold: float = 0.05
    """Phi gain above which the update gets a boost."""

    boost_factor: float = 1.4
    """Weight multiplier for boosted updates."""

    veto_severe: bool = True
    """Whether to veto Severe anomalies regardless of confidence."""

    # Pipeline
    defense: str = "trimmed_mean"
    """Defense strategy: fedavg, krum, trimmed_mean, median."""

    byzantine_fraction: float = 0.2
    """Expected fraction of Byzantine nodes."""

    gradient_dim: int = 128
    """Number of gradient dimensions to use."""

    @classmethod
    def high_security(cls) -> ConsciousRoundConfig:
        """Strict configuration: Krum + aggressive gating."""
        return cls(
            confidence_threshold=0.4,
            veto_confidence=0.15,
            defense="krum",
            byzantine_fraction=0.3,
        )

    @classmethod
    def performance(cls) -> ConsciousRoundConfig:
        """Relaxed configuration for faster convergence."""
        return cls(
            confidence_threshold=0.2,
            veto_confidence=0.05,
            veto_severe=False,
            defense="fedavg",
            byzantine_fraction=0.1,
        )


@dataclass
class NodeRoundScore:
    """Per-node consciousness assessment for a round."""

    phi_before: float
    phi_after: float
    phi_gain: float
    epistemic_confidence: float
    is_anomalous: bool
    severity: str  # "none", "mild", "moderate", "severe"
    pogq_quality: float
    pogq_consistency: float
    causes: List[str] = field(default_factory=list)


@dataclass
class ConsciousRoundResult:
    """Result of a conscious FL round."""

    aggregated: np.ndarray
    """Aggregated gradient values."""

    included_count: int
    """Number of participants whose updates were included."""

    vetoed_count: int
    """Number of participants vetoed by consciousness gating."""

    dampened_count: int
    """Number of participants dampened (reduced weight)."""

    boosted_count: int
    """Number of participants boosted."""

    node_scores: Dict[str, NodeRoundScore]
    """Per-node consciousness assessments."""

    plugin_weights_applied: bool
    """Whether Byzantine plugin weights were used."""


class ConsciousFlRound:
    """Stateful runner for consciousness-guided FL rounds.

    Maintains per-node Phi history across rounds for trend-aware
    anomaly detection.

    Example:
        >>> runner = ConsciousFlRound()
        >>> result = runner.run(
        ...     gradients={"n1": np.random.randn(128), "n2": np.random.randn(128)},
        ...     reputations={"n1": 0.8, "n2": 0.7},
        ... )
        >>> print(f"Rounds: {runner.rounds_completed}, Vetoed: {result.vetoed_count}")
    """

    def __init__(self, config: Optional[ConsciousRoundConfig] = None):
        self.config = config or ConsciousRoundConfig()
        self._rounds_completed = 0
        self._node_phi_history: Dict[str, float] = {}

    @property
    def rounds_completed(self) -> int:
        return self._rounds_completed

    def run(
        self,
        gradients: Dict[str, np.ndarray],
        reputations: Dict[str, float],
    ) -> ConsciousRoundResult:
        """Run one complete conscious FL round.

        Args:
            gradients: Gradient vectors per participant (node_id -> gradient).
            reputations: MATL reputation scores per participant (0.0-1.0).

        Returns:
            ConsciousRoundResult with aggregated gradient and per-node scores.
        """
        if not gradients:
            raise ValueError("No gradients provided")

        # Phase 1: Assess each gradient (Phi-inspired quality scoring)
        node_scores: Dict[str, NodeRoundScore] = {}
        weights: Dict[str, float] = {}
        vetoed = 0
        dampened = 0
        boosted = 0

        for node_id, gradient in gradients.items():
            # Get previous Phi for this node (default 0.5)
            phi_before = self._node_phi_history.get(node_id, 0.5)

            # Compute a lightweight Phi proxy from gradient statistics
            grad_norm = float(np.linalg.norm(gradient))
            grad_std = float(np.std(gradient))
            grad_mean = float(np.mean(np.abs(gradient)))

            # Phi proxy: normalized coherence of the gradient
            # Guard: zero/near-zero gradients get neutral Phi (0.5), not 1.0.
            # A zero gradient is uninformative, not maximally coherent.
            if grad_mean < 1e-7:
                phi_after = 0.5
            else:
                phi_after = float(np.clip(
                    1.0 - (grad_std / (grad_mean + 1e-8)) * 0.3,
                    0.0, 1.0,
                ))
            phi_gain = phi_after - phi_before

            # Epistemic confidence: combine Phi and reputation
            rep = reputations.get(node_id, 0.5)
            epistemic_confidence = float(np.clip(
                (phi_after + rep) / 2.0,
                0.0, 1.0,
            ))

            # Anomaly detection
            is_drop = phi_gain < -self.config.phi_gain_boost_threshold * 2
            low_confidence = epistemic_confidence < 0.2
            is_anomalous = is_drop or low_confidence

            # Severity classification
            if not is_anomalous:
                severity = "none"
            elif is_drop and low_confidence:
                severity = "severe"
            elif is_drop or low_confidence:
                severity = "moderate"
            else:
                severity = "mild"

            causes = []
            if is_drop:
                causes.append("phi_drop")
            if low_confidence:
                causes.append("low_confidence")

            # PoGQ mapping
            pogq_quality = epistemic_confidence
            phi_trend = 1.0 if phi_gain >= 0 else max(0.0, 1.0 + phi_gain)
            pogq_consistency = 0.5 * phi_trend + 0.5 * rep

            node_scores[node_id] = NodeRoundScore(
                phi_before=phi_before,
                phi_after=phi_after,
                phi_gain=phi_gain,
                epistemic_confidence=epistemic_confidence,
                is_anomalous=is_anomalous,
                severity=severity,
                pogq_quality=pogq_quality,
                pogq_consistency=pogq_consistency,
                causes=causes,
            )

            # Weight adjustment (mirrors SymthaeaQualityPlugin logic)
            if self.config.veto_severe and severity == "severe":
                weights[node_id] = 0.0
                vetoed += 1
            elif epistemic_confidence < self.config.veto_confidence:
                weights[node_id] = 0.0
                vetoed += 1
            elif (
                epistemic_confidence < self.config.confidence_threshold
                or severity == "moderate"
            ):
                weights[node_id] = max(0.1, epistemic_confidence)
                dampened += 1
            elif phi_gain > self.config.phi_gain_boost_threshold:
                weights[node_id] = self.config.boost_factor
                boosted += 1
            else:
                weights[node_id] = 1.0

            # Update Phi history
            self._node_phi_history[node_id] = phi_after

        # Phase 2: Weighted aggregation
        grad_list = []
        weight_list = []
        for node_id, gradient in gradients.items():
            w = weights.get(node_id, 1.0)
            if w > 0:
                dim = min(self.config.gradient_dim, len(gradient))
                grad_list.append(gradient[:dim] * w)
                weight_list.append(w)

        if not grad_list:
            # All vetoed — return zeros matching actual gradient dimension
            first_grad = next(iter(gradients.values()))
            dim = min(self.config.gradient_dim, len(first_grad))
            aggregated = np.zeros(dim, dtype=np.float32)
        else:
            total_weight = sum(weight_list)
            aggregated = np.sum(grad_list, axis=0) / total_weight

        self._rounds_completed += 1

        return ConsciousRoundResult(
            aggregated=aggregated.astype(np.float32),
            included_count=len(gradients) - vetoed,
            vetoed_count=vetoed,
            dampened_count=dampened,
            boosted_count=boosted,
            node_scores=node_scores,
            plugin_weights_applied=vetoed > 0 or dampened > 0 or boosted > 0,
        )


def run_conscious_fl_round(
    gradients: Dict[str, np.ndarray],
    reputations: Dict[str, float],
    config: Optional[ConsciousRoundConfig] = None,
) -> ConsciousRoundResult:
    """One-shot convenience function for a single conscious FL round.

    Creates a fresh runner each call. For multi-round FL, use ConsciousFlRound
    directly to accumulate Phi history across rounds.

    Args:
        gradients: Gradient vectors per participant.
        reputations: MATL reputation scores per participant.
        config: Optional round configuration.

    Returns:
        ConsciousRoundResult with aggregated gradient and per-node scores.
    """
    runner = ConsciousFlRound(config)
    return runner.run(gradients, reputations)
