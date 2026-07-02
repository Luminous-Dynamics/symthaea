# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Consciousness Bridge — Maps 0TML FL quality scores to Mycelix consciousness profiles.

This module bridges the gap between 0TML's Python FL system and Mycelix's
on-chain consciousness gating. It translates FL participant quality signals
(Byzantine detection scores, gradient quality, reputation) into the 4D
consciousness profile format used by mycelix-bridge-common.

The bridge enables:
1. FL quality scores → Mycelix reputation dimension updates
2. Byzantine detection flags → consciousness tier demotion
3. Gradient quality metrics → engagement dimension scoring
4. Cross-round reputation decay and recovery

Usage:
    from mycelix_fl.consciousness_bridge import ConsciousnessBridge

    bridge = ConsciousnessBridge()
    profile = bridge.compute_fl_consciousness_profile(
        participant_id="node-1",
        quality_score=0.85,
        byzantine_flags=0,
        rounds_participated=12,
        total_rounds=20,
    )
    # profile.tier => "Citizen" (0.40-0.60 range)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Consciousness tier thresholds (matching mycelix-bridge-common/consciousness_profile.rs)
TIER_OBSERVER = 0.0
TIER_PARTICIPANT = 0.30
TIER_CITIZEN = 0.40
TIER_STEWARD = 0.60
TIER_GUARDIAN = 0.80
HYSTERESIS_MARGIN = 0.05

# Dimension weights (matching Rust: 0.25 identity + 0.25 reputation + 0.30 community + 0.20 engagement)
WEIGHT_IDENTITY = 0.25
WEIGHT_REPUTATION = 0.25
WEIGHT_COMMUNITY = 0.30
WEIGHT_ENGAGEMENT = 0.20


@dataclass
class ConsciousnessProfile:
    """4D consciousness profile matching mycelix-bridge-common format."""
    identity: float = 0.5       # MFA assurance level (default: moderate)
    reputation: float = 0.5     # Cross-hApp aggregated reputation
    community: float = 0.5      # Peer trust attestations
    engagement: float = 0.0     # Domain-specific participation

    @property
    def combined_score(self) -> float:
        return (
            WEIGHT_IDENTITY * self.identity
            + WEIGHT_REPUTATION * self.reputation
            + WEIGHT_COMMUNITY * self.community
            + WEIGHT_ENGAGEMENT * self.engagement
        )

    @property
    def tier(self) -> str:
        score = self.combined_score
        if score >= TIER_GUARDIAN:
            return "Guardian"
        elif score >= TIER_STEWARD:
            return "Steward"
        elif score >= TIER_CITIZEN:
            return "Citizen"
        elif score >= TIER_PARTICIPANT:
            return "Participant"
        return "Observer"

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "reputation": self.reputation,
            "community": self.community,
            "engagement": self.engagement,
            "combined_score": self.combined_score,
            "tier": self.tier,
        }


@dataclass
class ParticipantHistory:
    """Tracks FL participant history for reputation computation."""
    rounds_participated: int = 0
    rounds_vetoed: int = 0
    byzantine_flags: int = 0
    cumulative_quality: float = 0.0
    last_active: float = 0.0


class ConsciousnessBridge:
    """
    Maps 0TML FL quality metrics to Mycelix consciousness profiles.

    This bridge maintains per-participant history and computes consciousness
    profiles that can be submitted to the Mycelix on-chain reputation system.
    """

    def __init__(
        self,
        reputation_decay_rate: float = 0.95,
        byzantine_penalty: float = 0.15,
        quality_weight: float = 0.6,
        participation_weight: float = 0.4,
    ):
        self.reputation_decay_rate = reputation_decay_rate
        self.byzantine_penalty = byzantine_penalty
        self.quality_weight = quality_weight
        self.participation_weight = participation_weight
        self._history: Dict[str, ParticipantHistory] = {}

    def _get_history(self, participant_id: str) -> ParticipantHistory:
        if participant_id not in self._history:
            self._history[participant_id] = ParticipantHistory()
        return self._history[participant_id]

    def record_round_result(
        self,
        participant_id: str,
        quality_score: float,
        was_vetoed: bool = False,
        byzantine_detected: bool = False,
    ) -> None:
        """Record the result of a participant's FL round contribution."""
        hist = self._get_history(participant_id)
        hist.rounds_participated += 1
        hist.cumulative_quality = (
            hist.cumulative_quality * self.reputation_decay_rate + quality_score
        )
        hist.last_active = time.time()

        if was_vetoed:
            hist.rounds_vetoed += 1
        if byzantine_detected:
            hist.byzantine_flags += 1

    def compute_fl_consciousness_profile(
        self,
        participant_id: str,
        quality_score: float,
        byzantine_flags: int = 0,
        rounds_participated: int = 0,
        total_rounds: int = 1,
    ) -> ConsciousnessProfile:
        """
        Compute a Mycelix-compatible consciousness profile from FL metrics.

        Maps:
        - quality_score (0-1) → engagement dimension
        - byzantine_flags → reputation penalty
        - participation_rate → community dimension
        - identity is pass-through (set externally via DID)
        """
        hist = self._get_history(participant_id)

        # Engagement: weighted average of current quality and historical
        if hist.rounds_participated > 0:
            historical_avg = hist.cumulative_quality / hist.rounds_participated
            engagement = (
                self.quality_weight * quality_score
                + self.participation_weight * historical_avg
            )
        else:
            engagement = quality_score * 0.8  # first-round discount

        # Reputation: penalized by byzantine flags
        base_reputation = 0.5
        penalty = min(byzantine_flags * self.byzantine_penalty, 0.45)
        reputation = max(0.0, base_reputation - penalty)

        # Also penalize from veto history
        if hist.rounds_participated > 0:
            veto_rate = hist.rounds_vetoed / hist.rounds_participated
            reputation -= veto_rate * 0.2

        reputation = max(0.0, min(1.0, reputation))

        # Community: participation rate
        participation_rate = rounds_participated / max(total_rounds, 1)
        community = min(1.0, participation_rate * 1.2)  # slight boost for active participants

        return ConsciousnessProfile(
            identity=0.5,  # Set externally via DID system
            reputation=reputation,
            community=community,
            engagement=min(1.0, max(0.0, engagement)),
        )

    def batch_compute_profiles(
        self,
        participants: Dict[str, Dict],
        total_rounds: int = 1,
    ) -> Dict[str, ConsciousnessProfile]:
        """
        Compute consciousness profiles for all participants in a round.

        Args:
            participants: Dict of participant_id -> {quality_score, byzantine_flags, rounds_participated}
            total_rounds: Total rounds in the FL session

        Returns:
            Dict of participant_id -> ConsciousnessProfile
        """
        profiles = {}
        for pid, metrics in participants.items():
            profiles[pid] = self.compute_fl_consciousness_profile(
                participant_id=pid,
                quality_score=metrics.get("quality_score", 0.5),
                byzantine_flags=metrics.get("byzantine_flags", 0),
                rounds_participated=metrics.get("rounds_participated", 0),
                total_rounds=total_rounds,
            )
        return profiles

    def should_gate_participant(
        self,
        participant_id: str,
        required_tier: str = "Participant",
    ) -> bool:
        """
        Check if a participant meets the consciousness tier requirement.

        This mirrors the on-chain gate_consciousness() check from mycelix-bridge-common.
        """
        hist = self._get_history(participant_id)
        if hist.rounds_participated == 0:
            return required_tier == "Observer"

        profile = self.compute_fl_consciousness_profile(
            participant_id=participant_id,
            quality_score=hist.cumulative_quality / max(hist.rounds_participated, 1),
            byzantine_flags=hist.byzantine_flags,
            rounds_participated=hist.rounds_participated,
            total_rounds=hist.rounds_participated,  # conservative estimate
        )

        tier_order = ["Observer", "Participant", "Citizen", "Steward", "Guardian"]
        current_idx = tier_order.index(profile.tier)
        required_idx = tier_order.index(required_tier)
        return current_idx >= required_idx
