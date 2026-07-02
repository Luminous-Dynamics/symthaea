#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
BFT Fail-Safe Mechanism for Mode 0 (Peer-Comparison)

This module implements Byzantine percentage estimation and network halt logic
to prevent catastrophic failure when BFT exceeds the peer-comparison ceiling (~35%).

Critical Safety Feature: Peer-comparison Byzantine detection relies on honest majority.
Beyond 35% BFT, Byzantine nodes can coordinate to appear more consistent than honest
nodes with label skew, leading to inversion (honest flagged, Byzantine accepted).
"""

from typing import Dict, List, Tuple


class BFTFailSafe:
    """
    Fail-safe mechanism for Mode 0 (Peer-Comparison) Byzantine detection.

    Monitors Byzantine percentage and halts network gracefully when exceeding
    peer-comparison capability (~35% BFT).
    """

    def __init__(self, bft_limit: float = 0.35, warning_threshold: float = 0.30):
        """
        Initialize fail-safe monitor.

        Args:
            bft_limit: BFT percentage where network halts (default: 0.35)
            warning_threshold: BFT percentage where warnings start (default: 0.30)
        """
        self.bft_limit = bft_limit
        self.warning_threshold = warning_threshold
        self.halt_triggered = False
        self.halt_reason = None

    def estimate_bft_percentage(
        self,
        node_reputations: Dict[int, float],
        detection_flags: Dict[int, bool],
        confidence_scores: Dict[int, float]
    ) -> float:
        """
        Estimate current Byzantine percentage based on detection results.

        Uses multiple signals:
        1. Number of nodes flagged as Byzantine
        2. Confidence scores of detections
        3. Reputation degradation patterns

        Args:
            node_reputations: Reputation scores for each node (0-1)
            detection_flags: Byzantine detection flags (True = detected)
            confidence_scores: Ensemble confidence scores (0-1)

        Returns:
            Estimated BFT percentage (0-1)
        """
        total_nodes = len(node_reputations)

        if total_nodes == 0:
            return 0.0

        # Signal 1: Direct detection count
        detected_count = sum(1 for flagged in detection_flags.values() if flagged)

        # Signal 2: Weighted by confidence (high confidence = more certain)
        weighted_detected = sum(
            confidence_scores.get(node_id, 0.0)
            for node_id, flagged in detection_flags.items()
            if flagged
        )

        # Signal 3: Reputation-based estimate (reputation < 0.3 = likely Byzantine)
        low_reputation_count = sum(
            1 for rep in node_reputations.values() if rep < 0.3
        )

        # Combined estimate (weighted average)
        direct_estimate = detected_count / total_nodes
        weighted_estimate = weighted_detected / total_nodes
        reputation_estimate = low_reputation_count / total_nodes

        # Weight: 50% direct, 30% weighted, 20% reputation
        combined_estimate = (
            0.5 * direct_estimate +
            0.3 * weighted_estimate +
            0.2 * reputation_estimate
        )

        return combined_estimate

    def check_safety(
        self,
        current_bft_estimate: float,
        round_num: int
    ) -> Tuple[bool, str]:
        """
        Check if network is operating safely within Mode 0 capability.

        Args:
            current_bft_estimate: Current BFT percentage estimate (0-1)
            round_num: Current training round

        Returns:
            Tuple of (is_safe, message)
            - is_safe: False if network should halt
            - message: Explanation of status
        """
        # Already halted
        if self.halt_triggered:
            return False, f"Network halted: {self.halt_reason}"

        # Warning zone (30-35%)
        if self.warning_threshold <= current_bft_estimate < self.bft_limit:
            message = (
                f"⚠️  WARNING (Round {round_num}): BFT estimate at {current_bft_estimate*100:.1f}%\n"
                f"    Approaching peer-comparison limit ({self.bft_limit*100:.0f}%)\n"
                f"    Mode 0 performance degrading - consider Mode 1 (Ground Truth)"
            )
            return True, message

        # Danger zone (>35%)
        if current_bft_estimate >= self.bft_limit:
            self.halt_triggered = True
            self.halt_reason = f"BFT estimate ({current_bft_estimate*100:.1f}%) exceeds limit ({self.bft_limit*100:.0f}%)"

            message = (
                f"🛑 NETWORK HALT (Round {round_num})\n"
                f"\n"
                f"BFT Estimate: {current_bft_estimate*100:.1f}% > {self.bft_limit*100:.0f}% limit\n"
                f"\n"
                f"REASON:\n"
                f"  Peer-comparison Byzantine detection is unreliable beyond ~35% BFT.\n"
                f"  Without honest majority, Byzantine nodes can coordinate to appear\n"
                f"  more consistent than honest nodes with label skew.\n"
                f"\n"
                f"RISK:\n"
                f"  - Catastrophic inversion (honest flagged, Byzantine accepted)\n"
                f"  - Silent model corruption\n"
                f"  - Loss of training integrity\n"
                f"\n"
                f"RECOMMENDATION:\n"
                f"  Switch to Mode 1 (Ground Truth - PoGQ) for >35% BFT scenarios.\n"
                f"  Mode 1 is resilient to >50% BFT with server's private test set.\n"
                f"\n"
                f"Network halted gracefully. No gradients accepted.\n"
            )

            return False, message

        # Safe zone (<30%)
        return True, f"✅ Safe: BFT estimate {current_bft_estimate*100:.1f}% (Round {round_num})"

    def should_halt(self) -> bool:
        """Check if halt has been triggered."""
        return self.halt_triggered

    def get_halt_reason(self) -> str:
        """Get reason for halt if triggered."""
        return self.halt_reason or "No halt triggered"

    def reset(self):
        """Reset fail-safe state (for testing)."""
        self.halt_triggered = False
        self.halt_reason = None


def demonstrate_failsafe():
    """Demonstration of fail-safe behavior at different BFT levels."""

    print("="*70)
    print("BFT Fail-Safe Demonstration")
    print("="*70)
    print()

    failsafe = BFTFailSafe(bft_limit=0.35, warning_threshold=0.30)

    # Scenario 1: Safe operation (20% BFT)
    print("Scenario 1: 20% BFT (4/20 nodes Byzantine)")
    print("-" * 70)

    node_reputations = {i: (0.2 if i < 4 else 1.0) for i in range(20)}
    detection_flags = {i: (i < 4) for i in range(20)}
    confidence_scores = {i: (0.8 if i < 4 else 0.1) for i in range(20)}

    bft_est = failsafe.estimate_bft_percentage(node_reputations, detection_flags, confidence_scores)
    is_safe, message = failsafe.check_safety(bft_est, round_num=5)

    print(f"Estimated BFT: {bft_est*100:.1f}%")
    print(message)
    print()

    # Scenario 2: Warning zone (33% BFT)
    print("Scenario 2: 33% BFT (7/20 nodes Byzantine)")
    print("-" * 70)

    failsafe.reset()
    node_reputations = {i: (0.25 if i < 7 else 0.95) for i in range(20)}
    detection_flags = {i: (i < 7) for i in range(20)}
    confidence_scores = {i: (0.75 if i < 7 else 0.15) for i in range(20)}

    bft_est = failsafe.estimate_bft_percentage(node_reputations, detection_flags, confidence_scores)
    is_safe, message = failsafe.check_safety(bft_est, round_num=5)

    print(f"Estimated BFT: {bft_est*100:.1f}%")
    print(message)
    print()

    # Scenario 3: Danger zone (40% BFT) - HALT
    print("Scenario 3: 40% BFT (8/20 nodes Byzantine) - NETWORK HALT")
    print("-" * 70)

    failsafe.reset()
    node_reputations = {i: (0.20 if i < 8 else 0.90) for i in range(20)}
    detection_flags = {i: (i < 8) for i in range(20)}
    confidence_scores = {i: (0.85 if i < 8 else 0.10) for i in range(20)}

    bft_est = failsafe.estimate_bft_percentage(node_reputations, detection_flags, confidence_scores)
    is_safe, message = failsafe.check_safety(bft_est, round_num=5)

    print(f"Estimated BFT: {bft_est*100:.1f}%")
    print(message)
    print()

    print("="*70)
    print("Fail-Safe Status:")
    print(f"  Halt Triggered: {failsafe.should_halt()}")
    print(f"  Reason: {failsafe.get_halt_reason()}")
    print("="*70)


if __name__ == "__main__":
    demonstrate_failsafe()
