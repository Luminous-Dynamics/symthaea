# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MATL (Multi-Axis Trust Layer) — reputation and proof-of-gradient-quality.

Provides the trust primitives used by federated learning and governance.
Matches the Rust SDK API surface in mycelix-workspace/sdk/src/matl/.

Composite = 0.4 * PoGQ + 0.3 * Consistency + 0.3 * Reputation
"""

from dataclasses import dataclass, field
import time


# ============================================================================
# Types
# ============================================================================


@dataclass
class ReputationScore:
    """Agent reputation in the MATL system.

    Reputation decays exponentially (30-day half-life) and grows through
    positive contributions. Slash penalties reduce score multiplicatively.
    """

    score: float = 0.5
    total_interactions: int = 0
    positive_interactions: int = 0
    slash_count: int = 0
    last_updated: float = field(default_factory=time.time)

    @property
    def positive_ratio(self) -> float:
        if self.total_interactions == 0:
            return 0.5
        return self.positive_interactions / self.total_interactions


@dataclass
class ProofOfGradientQuality:
    """Proof that a gradient update meets quality thresholds.

    Three dimensions:
    - quality: gradient norm and direction alignment (0-1)
    - consistency: agreement with recent history (0-1)
    - entropy: information content of the update (0-1)
    """

    quality: float
    consistency: float
    entropy: float
    timestamp: float = field(default_factory=time.time)

    @property
    def composite(self) -> float:
        """Weighted composite score (MATL formula)."""
        return 0.4 * self.quality + 0.3 * self.consistency + 0.3 * self.entropy


# ============================================================================
# Functions — match Rust SDK API
# ============================================================================


def reputation_value(rep: ReputationScore) -> float:
    """Extract the scalar reputation value (0-1).

    Applies slash penalty: each slash reduces effective score by 20%.
    """
    penalty = 0.8 ** rep.slash_count
    return max(0.0, min(1.0, rep.score * penalty))


def calculate_composite(
    pogq: ProofOfGradientQuality | None,
    rep: ReputationScore,
) -> float:
    """Calculate MATL composite trust score.

    Composite = 0.4 * PoGQ + 0.3 * Consistency + 0.3 * Reputation

    If no PoGQ is available, falls back to reputation alone.
    """
    rep_val = reputation_value(rep)
    if pogq is None:
        return rep_val
    return 0.4 * pogq.quality + 0.3 * pogq.consistency + 0.3 * rep_val


def create_reputation(participant_id: str) -> ReputationScore:
    """Create a fresh reputation for a new participant.

    Starts at 0.5 (neutral) — trust must be earned.
    """
    return ReputationScore(score=0.5)


def create_pogq(
    quality: float,
    consistency: float,
    entropy: float,
) -> ProofOfGradientQuality:
    """Create a PoGQ proof from measured gradient properties."""
    return ProofOfGradientQuality(
        quality=max(0.0, min(1.0, quality)),
        consistency=max(0.0, min(1.0, consistency)),
        entropy=max(0.0, min(1.0, entropy)),
    )


def record_positive(rep: ReputationScore) -> ReputationScore:
    """Record a positive interaction, boosting reputation.

    Uses exponential moving average: new_score = 0.9 * old + 0.1 * 1.0
    """
    return ReputationScore(
        score=min(1.0, rep.score * 0.9 + 0.1),
        total_interactions=rep.total_interactions + 1,
        positive_interactions=rep.positive_interactions + 1,
        slash_count=rep.slash_count,
    )
