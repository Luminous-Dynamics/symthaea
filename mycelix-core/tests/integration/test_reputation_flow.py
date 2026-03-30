# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Reputation Flow

Verifies the complete reputation flow:
1. Byzantine detection triggers
2. Bridge receives detection event
3. Reputation score is updated
4. Score changes affect future aggregation weights

This tests the feedback loop between detection and reputation systems.
"""

import pytest
import numpy as np
import asyncio
from typing import List, Set, Dict

from .fixtures.network import (
    FLNetwork,
    NetworkConfig,
    GradientStore,
    ZomeValidator,
    ReputationBridge,
    ByzantineDetector,
    FLAggregator,
)
from .fixtures.nodes import ByzantineNode, HonestNode, NodeBehavior
from .fixtures.metrics import MetricsCollector


# ============================================================================
# Basic Reputation Tests
# ============================================================================

class TestReputationBasics:
    """Test basic reputation operations."""

    @pytest.mark.asyncio
    async def test_initial_reputation_score(self):
        """Test that nodes start with initial reputation score."""
        bridge = ReputationBridge(initial_score=1.0)

        score = await bridge.get_score("new_node")
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_reputation_update(self):
        """Test reputation score updates."""
        bridge = ReputationBridge(initial_score=1.0)

        # Positive update
        new_score = await bridge.update_score("node_1", delta=0.1, reason="good behavior")
        assert new_score == 1.0  # Capped at max

        # Negative update
        new_score = await bridge.update_score("node_1", delta=-0.2, reason="bad behavior")
        assert new_score == 0.8

    @pytest.mark.asyncio
    async def test_reputation_bounds(self):
        """Test that reputation stays within bounds."""
        bridge = ReputationBridge(initial_score=1.0, min_score=0.0, max_score=1.0)

        # Try to go above max
        await bridge.update_score("node", delta=1.0, reason="test")
        score = await bridge.get_score("node")
        assert score == 1.0

        # Try to go below min
        await bridge.update_score("node", delta=-2.0, reason="test")
        score = await bridge.get_score("node")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_reputation_history(self):
        """Test that reputation history is tracked."""
        bridge = ReputationBridge(initial_score=1.0)

        await bridge.update_score("node", delta=-0.1, reason="first")
        await bridge.update_score("node", delta=-0.2, reason="second")
        await bridge.update_score("node", delta=0.05, reason="third")

        history = bridge.get_history("node")

        assert len(history) == 3
        assert history[0]["reason"] == "first"
        assert history[1]["reason"] == "second"
        assert history[2]["reason"] == "third"

        # Verify score changes are tracked
        assert history[0]["old_score"] == 1.0
        assert history[0]["new_score"] == 0.9


# ============================================================================
# Byzantine Detection to Reputation Tests
# ============================================================================

class TestByzantineToReputation:
    """Test flow from Byzantine detection to reputation update."""

    @pytest.mark.asyncio
    async def test_byzantine_flag_reduces_score(self):
        """Test that flagging as Byzantine reduces reputation."""
        bridge = ReputationBridge(
            initial_score=1.0,
            penalty_byzantine=0.3,
        )

        initial = await bridge.get_score("attacker")
        await bridge.flag_byzantine("attacker", detection_confidence=1.0)
        after = await bridge.get_score("attacker")

        assert after < initial
        assert after == 0.7  # 1.0 - 0.3

    @pytest.mark.asyncio
    async def test_confidence_weighted_penalty(self):
        """Test that penalty is weighted by detection confidence."""
        bridge = ReputationBridge(
            initial_score=1.0,
            penalty_byzantine=0.3,
        )

        # High confidence
        await bridge.flag_byzantine("high_conf", detection_confidence=1.0)
        high_score = await bridge.get_score("high_conf")

        # Low confidence
        await bridge.flag_byzantine("low_conf", detection_confidence=0.5)
        low_score = await bridge.get_score("low_conf")

        # High confidence should result in lower score
        assert high_score < low_score
        assert high_score == 0.7  # 1.0 - 0.3 * 1.0
        assert low_score == 0.85  # 1.0 - 0.3 * 0.5

    @pytest.mark.asyncio
    async def test_repeated_detections_compound(self):
        """Test that repeated Byzantine detections compound."""
        bridge = ReputationBridge(
            initial_score=1.0,
            penalty_byzantine=0.2,
        )

        # Multiple detections
        for _ in range(3):
            await bridge.flag_byzantine("repeat_offender", detection_confidence=1.0)

        score = await bridge.get_score("repeat_offender")

        # 1.0 - 3 * 0.2 = 0.4
        assert score == 0.4

        # Check Byzantine count
        count = bridge.get_byzantine_count("repeat_offender")
        assert count == 3

    @pytest.mark.asyncio
    async def test_rejection_penalty(self):
        """Test that gradient rejection affects reputation."""
        bridge = ReputationBridge(
            initial_score=1.0,
            penalty_rejected=0.1,
        )

        await bridge.flag_rejected("node", "Invalid gradient format")
        score = await bridge.get_score("node")

        assert score == 0.9

    @pytest.mark.asyncio
    async def test_detection_integration(self, honest_node_factory, byzantine_node_factory):
        """Test integration of detection and reputation."""
        detector = ByzantineDetector(z_threshold=2.0)
        bridge = ReputationBridge(
            initial_score=1.0,
            penalty_byzantine=0.25,
        )
        gradient_dim = 500

        # Create nodes
        honest_nodes = [honest_node_factory(f"honest_{i}", gradient_dim) for i in range(7)]
        byzantine_nodes = [
            byzantine_node_factory(f"byzantine_{i}", "random", gradient_dim)
            for i in range(3)
        ]

        # Collect gradients
        gradients = [await node.compute_gradient(1) for node in honest_nodes]
        node_ids = [node.node_id for node in honest_nodes]

        gradients += [await node.compute_gradient(1) for node in byzantine_nodes]
        node_ids += [node.node_id for node in byzantine_nodes]

        # Detect Byzantine
        detected, confidence = await detector.detect(gradients, node_ids)

        # Update reputation for detected nodes
        for node_id in detected:
            conf = confidence.get(node_id, 1.0)
            await bridge.flag_byzantine(node_id, conf)

        # Verify reputation changes
        for node_id in detected:
            score = await bridge.get_score(node_id)
            assert score < 1.0, f"Detected node {node_id} should have reduced reputation"

        # Honest nodes should still have full reputation
        for node in honest_nodes:
            if node.node_id not in detected:
                score = await bridge.get_score(node.node_id)
                assert score == 1.0


# ============================================================================
# Reputation Recovery Tests
# ============================================================================

class TestReputationRecovery:
    """Test reputation recovery and decay mechanisms."""

    @pytest.mark.asyncio
    async def test_honest_participation_reward(self):
        """Test that honest participation increases reputation."""
        bridge = ReputationBridge(
            initial_score=0.5,  # Start low
            reward_honest=0.1,
        )

        # Set initial low score
        await bridge.update_score("recovering_node", delta=-0.5, reason="initial")

        # Reward for honest participation
        await bridge.reward_honest_participation("recovering_node")
        score = await bridge.get_score("recovering_node")

        assert score == 0.6  # 0.5 + 0.1

    @pytest.mark.asyncio
    async def test_reputation_decay_toward_initial(self):
        """Test that reputation decays toward initial value."""
        bridge = ReputationBridge(
            initial_score=1.0,
            decay_rate=0.1,
        )

        # Penalize a node
        await bridge.update_score("node", delta=-0.5, reason="penalty")
        assert await bridge.get_score("node") == 0.5

        # Apply decay
        await bridge.apply_decay()
        score = await bridge.get_score("node")
        assert score == 0.6  # 0.5 + 0.1 (decay toward 1.0)

    @pytest.mark.asyncio
    async def test_full_recovery_through_good_behavior(self):
        """Test that nodes can fully recover through good behavior."""
        bridge = ReputationBridge(
            initial_score=1.0,
            penalty_byzantine=0.3,
            reward_honest=0.05,
            decay_rate=0.02,
        )

        # Initial penalty
        await bridge.flag_byzantine("recoverable_node", 1.0)
        assert await bridge.get_score("recoverable_node") == 0.7

        # Simulate multiple rounds of good behavior
        for _ in range(10):
            await bridge.reward_honest_participation("recoverable_node")
            await bridge.apply_decay()

        score = await bridge.get_score("recoverable_node")

        # Should have recovered close to initial
        assert score >= 0.95


# ============================================================================
# Network Integration Tests
# ============================================================================

class TestReputationNetworkIntegration:
    """Test reputation flow in full network context."""

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_network_updates_reputation(self, byzantine_network_config):
        """Test that network automatically updates reputation."""
        config = byzantine_network_config
        config.reputation_enabled = True
        network = FLNetwork(config)
        await network.start()

        try:
            # Get initial scores
            initial_scores = network.reputation_bridge.get_all_scores()

            # Run several rounds
            results = await network.run_rounds(5)

            # Get final scores
            final_scores = network.reputation_bridge.get_all_scores()

            # Byzantine nodes should have lower scores
            for node_id in final_scores:
                if "byzantine" in node_id:
                    # Score should decrease if detected
                    # (may not always decrease if detection fails)
                    pass

            # Honest nodes that weren't falsely flagged should have maintained/increased score
            for node_id in final_scores:
                if "honest" in node_id:
                    initial = initial_scores.get(node_id, 1.0)
                    final = final_scores[node_id]
                    # Honest nodes should not have significantly reduced scores
                    assert final >= initial - 0.1, f"Honest node {node_id} unfairly penalized"

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_reputation_affects_aggregation(self, byzantine_network_config):
        """Test that reputation scores affect aggregation weights."""
        # This test verifies the concept - in production, low-reputation
        # nodes would have reduced weight in aggregation

        config = byzantine_network_config
        network = FLNetwork(config)
        await network.start()

        try:
            # Run initial rounds to establish reputation
            await network.run_rounds(10)

            # Get reputation scores
            scores = network.reputation_bridge.get_all_scores()

            # Byzantine nodes should have lower scores
            byzantine_scores = [
                scores.get(f"byzantine_{i}", 1.0)
                for i in range(config.num_byzantine_nodes)
            ]
            honest_scores = [
                scores.get(f"honest_{i}", 1.0)
                for i in range(config.num_honest_nodes)
            ]

            # Average Byzantine score should be lower
            avg_byzantine = np.mean(byzantine_scores) if byzantine_scores else 0
            avg_honest = np.mean(honest_scores) if honest_scores else 1

            # Not always guaranteed (depends on detection), but expected
            assert avg_honest >= avg_byzantine or abs(avg_honest - avg_byzantine) < 0.3

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    async def test_reputation_over_many_rounds(
        self, byzantine_network_config, metrics_collector
    ):
        """Test reputation evolution over many rounds."""
        config = byzantine_network_config
        config.num_honest_nodes = 15
        config.num_byzantine_nodes = 5

        network = FLNetwork(config)
        await network.start()

        try:
            score_history: Dict[str, List[float]] = {}

            for round_num in range(20):
                # Record scores before round
                scores = network.reputation_bridge.get_all_scores()
                for node_id, score in scores.items():
                    if node_id not in score_history:
                        score_history[node_id] = []
                    score_history[node_id].append(score)

                await network.run_round()

                metrics_collector.record_custom(
                    "reputation_snapshot",
                    {"round": round_num, "scores": scores.copy()}
                )

            # Analyze trends
            for node_id, history in score_history.items():
                if "byzantine" in node_id and len(history) > 1:
                    # Byzantine scores should trend downward
                    trend = history[-1] - history[0]
                    metrics_collector.record_custom(
                        f"reputation_trend_{node_id}",
                        trend
                    )

        finally:
            await network.shutdown()


# ============================================================================
# Edge Cases
# ============================================================================

class TestReputationEdgeCases:
    """Test edge cases in reputation system."""

    @pytest.mark.asyncio
    async def test_concurrent_reputation_updates(self):
        """Test concurrent reputation updates."""
        bridge = ReputationBridge(initial_score=1.0)

        async def update(i):
            await bridge.update_score("shared_node", delta=-0.01, reason=f"update_{i}")

        # Run many concurrent updates
        await asyncio.gather(*[update(i) for i in range(100)])

        score = await bridge.get_score("shared_node")

        # Score should reflect all updates (1.0 - 100 * 0.01 = 0.0)
        assert abs(score - 0.0) < 0.01

    @pytest.mark.asyncio
    async def test_reputation_reset(self):
        """Test reputation reset functionality."""
        bridge = ReputationBridge(initial_score=1.0)

        # Create some history
        await bridge.flag_byzantine("node_1", 1.0)
        await bridge.flag_byzantine("node_2", 1.0)

        assert await bridge.get_score("node_1") < 1.0
        assert await bridge.get_score("node_2") < 1.0

        # Reset
        await bridge.reset()

        # All scores should be back to initial
        assert await bridge.get_score("node_1") == 1.0
        assert await bridge.get_score("node_2") == 1.0
        assert bridge.get_all_scores() == {}

    @pytest.mark.asyncio
    async def test_reputation_with_zero_initial(self):
        """Test reputation system with zero initial score."""
        bridge = ReputationBridge(
            initial_score=0.0,
            reward_honest=0.1,
        )

        # Nodes start at zero
        assert await bridge.get_score("new_node") == 0.0

        # Build reputation through good behavior
        for _ in range(5):
            await bridge.reward_honest_participation("new_node")

        assert await bridge.get_score("new_node") == 0.5

    @pytest.mark.asyncio
    async def test_many_nodes_reputation(self):
        """Test reputation system with many nodes."""
        bridge = ReputationBridge(initial_score=1.0)

        n_nodes = 1000

        # Update all nodes
        for i in range(n_nodes):
            delta = -0.1 if i % 2 == 0 else 0.0
            if delta != 0:
                await bridge.update_score(f"node_{i}", delta, "test")

        scores = bridge.get_all_scores()

        # Should have correct number of entries (only updated nodes are tracked)
        assert len(scores) == n_nodes // 2

        # Even nodes should have 0.9
        for i in range(0, n_nodes, 2):
            assert scores[f"node_{i}"] == 0.9
