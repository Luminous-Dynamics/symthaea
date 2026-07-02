# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Network Partition and Recovery

Tests the system's ability to:
- Handle network partitions
- Recover from node failures
- Maintain consistency during disruptions
- Resume normal operation after recovery
"""

import pytest
import numpy as np
import asyncio
import time
from typing import List, Dict, Set, Any, Optional

from .fixtures.network import (
    FLNetwork,
    NetworkConfig,
    GradientStore,
    ReputationBridge,
    FLAggregator,
)
from .fixtures.nodes import HonestNode, ByzantineNode, SlowNode, IntermittentNode
from .fixtures.metrics import MetricsCollector, TestReport


# ============================================================================
# Network Partition Simulation
# ============================================================================

class PartitionedNetwork:
    """
    Simulates network partitions by controlling which nodes can communicate.
    """

    def __init__(self, network: FLNetwork):
        self.network = network
        self._partitions: List[Set[str]] = []
        self._isolated_nodes: Set[str] = set()
        self._original_nodes = set(network.get_all_nodes().keys())

    def create_partition(self, partition_a: Set[str], partition_b: Set[str]):
        """Create a network partition between two groups."""
        self._partitions = [partition_a, partition_b]
        # Nodes in neither partition are isolated
        self._isolated_nodes = self._original_nodes - partition_a - partition_b

    def heal_partition(self):
        """Heal all partitions."""
        self._partitions = []
        self._isolated_nodes = set()

    def isolate_node(self, node_id: str):
        """Isolate a single node from the network."""
        self._isolated_nodes.add(node_id)

    def reconnect_node(self, node_id: str):
        """Reconnect an isolated node."""
        self._isolated_nodes.discard(node_id)

    def get_active_nodes(self, from_perspective: Optional[str] = None) -> Set[str]:
        """Get nodes visible from a given perspective."""
        if from_perspective and from_perspective in self._isolated_nodes:
            return {from_perspective}  # Isolated node only sees itself

        if not self._partitions:
            return self._original_nodes - self._isolated_nodes

        # Find which partition the perspective node is in
        for partition in self._partitions:
            if from_perspective is None or from_perspective in partition:
                return partition - self._isolated_nodes

        return set()


# ============================================================================
# Recovery Configuration
# ============================================================================

@pytest.fixture
def recovery_config():
    """Configuration for recovery tests."""
    return NetworkConfig(
        num_honest_nodes=15,
        num_byzantine_nodes=0,
        gradient_dimension=1000,
        aggregation_strategy="trimmed_mean",
        byzantine_threshold=0.33,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        min_participants=5,
        timeout_ms=10000,
    )


@pytest.fixture
def fault_tolerant_config():
    """Configuration for fault tolerance tests."""
    return NetworkConfig(
        num_honest_nodes=20,
        num_byzantine_nodes=0,
        gradient_dimension=1000,
        aggregation_strategy="multi_krum",
        min_participants=10,
        timeout_ms=15000,
    )


# ============================================================================
# Node Failure Tests
# ============================================================================

class TestNodeFailure:
    """Test handling of individual node failures."""

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_single_node_dropout(self, recovery_config, metrics_collector):
        """Test system continues when single node drops out."""
        network = FLNetwork(recovery_config)
        await network.start()

        try:
            # Run initial round successfully
            result1 = await network.run_round()
            assert result1["success"]
            assert result1["n_participants"] == 15

            # Simulate node dropout by removing from network
            nodes = network.get_all_nodes()
            dropout_node = "honest_0"

            # Mark node as having high dropout probability
            if dropout_node in nodes:
                nodes[dropout_node].dropout_probability = 1.0

            # Run round with dropped node
            result2 = await network.run_round()

            # Should still succeed with n-1 participants
            assert result2["success"]
            # May have fewer participants due to dropout

            metrics_collector.record_custom(
                "single_dropout",
                {
                    "before_participants": result1["n_participants"],
                    "after_participants": result2["n_participants"],
                }
            )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_multiple_node_dropouts(self, recovery_config, metrics_collector):
        """Test system handles multiple simultaneous dropouts."""
        network = FLNetwork(recovery_config)
        await network.start()

        try:
            # Baseline round
            result1 = await network.run_round()
            assert result1["success"]

            # Simulate 30% node dropout
            nodes = network.get_all_nodes()
            dropout_count = int(len(nodes) * 0.3)

            for i, (node_id, node) in enumerate(nodes.items()):
                if i < dropout_count:
                    node.dropout_probability = 1.0

            # Run round with dropouts
            result2 = await network.run_round()

            # Should still succeed if above min_participants
            if result2["n_participants"] >= recovery_config.min_participants:
                assert result2["success"]
            else:
                # May fail due to insufficient participants
                pass

            metrics_collector.record_custom(
                "multiple_dropout",
                {
                    "dropout_count": dropout_count,
                    "remaining_participants": result2.get("n_participants", 0),
                    "success": result2["success"],
                }
            )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_node_recovery_after_dropout(
        self, recovery_config, metrics_collector
    ):
        """Test node can rejoin after dropout."""
        network = FLNetwork(recovery_config)
        await network.start()

        try:
            # Initial round
            result1 = await network.run_round()
            initial_participants = result1["n_participants"]

            # Cause dropout
            nodes = network.get_all_nodes()
            dropout_node_id = "honest_5"
            nodes[dropout_node_id].dropout_probability = 1.0

            # Round with dropout
            result2 = await network.run_round()

            # Recovery - node comes back
            nodes[dropout_node_id].dropout_probability = 0.0

            # Round after recovery
            result3 = await network.run_round()

            # Should be back to full participation
            assert result3["success"]

            metrics_collector.record_custom(
                "recovery_after_dropout",
                {
                    "initial": initial_participants,
                    "during_dropout": result2.get("n_participants", 0),
                    "after_recovery": result3["n_participants"],
                }
            )

        finally:
            await network.shutdown()


# ============================================================================
# Network Partition Tests
# ============================================================================

class TestNetworkPartition:
    """Test handling of network partitions."""

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_partition_detection(self, fault_tolerant_config, metrics_collector):
        """Test that system detects reduced participation during partition."""
        network = FLNetwork(fault_tolerant_config)
        await network.start()

        try:
            # Normal operation
            result1 = await network.run_round()
            assert result1["success"]
            normal_participants = result1["n_participants"]

            # Simulate partition by disabling half the nodes
            nodes = network.get_all_nodes()
            partition_size = len(nodes) // 2

            for i, (node_id, node) in enumerate(nodes.items()):
                if i < partition_size:
                    node.dropout_probability = 1.0

            # Operation during partition
            result2 = await network.run_round()

            # Should have roughly half participants
            partition_participants = result2.get("n_participants", 0)

            assert partition_participants < normal_participants
            assert partition_participants >= fault_tolerant_config.min_participants

            metrics_collector.record_custom(
                "partition_impact",
                {
                    "normal": normal_participants,
                    "during_partition": partition_participants,
                }
            )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_partition_recovery(self, fault_tolerant_config, metrics_collector):
        """Test recovery after network partition heals."""
        network = FLNetwork(fault_tolerant_config)
        await network.start()

        try:
            # Phase 1: Normal operation
            results_normal = []
            for _ in range(3):
                result = await network.run_round()
                results_normal.append(result)
            assert all(r["success"] for r in results_normal)

            # Phase 2: Create partition
            nodes = network.get_all_nodes()
            for i, (node_id, node) in enumerate(nodes.items()):
                if i < len(nodes) // 2:
                    node.dropout_probability = 1.0

            results_partition = []
            for _ in range(3):
                result = await network.run_round()
                results_partition.append(result)

            # Phase 3: Heal partition
            for node in nodes.values():
                node.dropout_probability = 0.0

            results_recovery = []
            for _ in range(3):
                result = await network.run_round()
                results_recovery.append(result)
            assert all(r["success"] for r in results_recovery)

            # Verify recovery
            normal_avg = np.mean([r["n_participants"] for r in results_normal])
            partition_avg = np.mean([
                r.get("n_participants", 0) for r in results_partition
            ])
            recovery_avg = np.mean([r["n_participants"] for r in results_recovery])

            assert recovery_avg >= normal_avg * 0.9  # Should recover to near normal

            metrics_collector.record_custom(
                "partition_recovery",
                {
                    "normal_avg": normal_avg,
                    "partition_avg": partition_avg,
                    "recovery_avg": recovery_avg,
                }
            )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_asymmetric_partition(self, fault_tolerant_config, metrics_collector):
        """Test asymmetric partition (70/30 split)."""
        network = FLNetwork(fault_tolerant_config)
        await network.start()

        try:
            nodes = network.get_all_nodes()
            total_nodes = len(nodes)

            # 70/30 split
            majority_size = int(total_nodes * 0.7)

            # Disable minority partition
            for i, (node_id, node) in enumerate(nodes.items()):
                if i >= majority_size:
                    node.dropout_probability = 1.0

            # Run rounds with majority partition
            results = []
            for _ in range(5):
                result = await network.run_round()
                results.append(result)

            # Majority partition should still function
            success_rate = sum(1 for r in results if r["success"]) / len(results)
            assert success_rate >= 0.8

            avg_participants = np.mean([
                r.get("n_participants", 0) for r in results
            ])
            assert avg_participants >= majority_size * 0.8

        finally:
            await network.shutdown()


# ============================================================================
# Consistency During Recovery Tests
# ============================================================================

class TestConsistencyDuringRecovery:
    """Test data consistency during recovery scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_gradient_consistency_across_recovery(
        self, recovery_config, metrics_collector
    ):
        """Test gradient aggregation remains consistent across recovery."""
        network = FLNetwork(recovery_config)
        await network.start()

        try:
            # Collect gradients from stable rounds
            stable_aggregates = []
            for _ in range(5):
                result = await network.run_round()
                if result["success"]:
                    stable_aggregates.append(result["aggregated_gradient"])

            # Cause disruption
            nodes = network.get_all_nodes()
            for i, node in enumerate(nodes.values()):
                if i % 3 == 0:
                    node.dropout_probability = 0.5

            # Run disrupted rounds
            disrupted_aggregates = []
            for _ in range(5):
                result = await network.run_round()
                if result["success"]:
                    disrupted_aggregates.append(result["aggregated_gradient"])

            # Recover
            for node in nodes.values():
                node.dropout_probability = 0.0

            # Post-recovery rounds
            recovery_aggregates = []
            for _ in range(5):
                result = await network.run_round()
                if result["success"]:
                    recovery_aggregates.append(result["aggregated_gradient"])

            # Check consistency: aggregates should follow similar direction
            if stable_aggregates and recovery_aggregates:
                stable_direction = np.mean(stable_aggregates, axis=0)
                recovery_direction = np.mean(recovery_aggregates, axis=0)

                similarity = np.dot(stable_direction, recovery_direction) / (
                    np.linalg.norm(stable_direction) * np.linalg.norm(recovery_direction)
                )

                # Directions should be similar (same convergence direction)
                assert similarity > 0.5, f"Inconsistent recovery: similarity={similarity}"

                metrics_collector.record_custom(
                    "consistency_check",
                    {"pre_post_similarity": similarity}
                )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_reputation_consistency_across_recovery(
        self, recovery_config, metrics_collector
    ):
        """Test reputation scores remain consistent across recovery."""
        config = recovery_config
        config.num_byzantine_nodes = 3
        config.num_honest_nodes = 12

        network = FLNetwork(config)
        await network.start()

        try:
            # Build up reputation over several rounds
            for _ in range(10):
                await network.run_round()

            pre_disruption_scores = network.reputation_bridge.get_all_scores().copy()

            # Cause disruption (but no new Byzantine behavior)
            nodes = network.get_all_nodes()
            for i, node in enumerate(nodes.values()):
                if i < 5:
                    node.dropout_probability = 0.8

            for _ in range(5):
                await network.run_round()

            # Recover
            for node in nodes.values():
                node.dropout_probability = 0.0

            for _ in range(5):
                await network.run_round()

            post_recovery_scores = network.reputation_bridge.get_all_scores()

            # Scores should not have dramatically changed for honest nodes
            for node_id in pre_disruption_scores:
                if "honest" in node_id:
                    pre = pre_disruption_scores.get(node_id, 1.0)
                    post = post_recovery_scores.get(node_id, 1.0)

                    # Honest nodes shouldn't lose significant reputation
                    assert post >= pre - 0.2, (
                        f"Node {node_id} unfairly penalized during recovery"
                    )

        finally:
            await network.shutdown()


# ============================================================================
# Slow Node and Timeout Tests
# ============================================================================

class TestSlowNodesAndTimeouts:
    """Test handling of slow nodes and timeouts."""

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_slow_node_handling(self, recovery_config, metrics_collector):
        """Test that slow nodes don't block the system."""
        network = FLNetwork(recovery_config)
        await network.start()

        try:
            # Replace some nodes with slow nodes
            nodes = network.get_all_nodes()

            # Add artificial latency to some nodes
            for i, node in enumerate(nodes.values()):
                if i < 3:
                    node.latency_ms = 500  # 500ms delay

            # Run rounds
            results = []
            for _ in range(5):
                start = time.time()
                result = await network.run_round()
                duration = (time.time() - start) * 1000
                results.append({"result": result, "duration": duration})

            # All rounds should complete
            assert all(r["result"]["success"] for r in results)

            # Duration should not be excessive
            avg_duration = np.mean([r["duration"] for r in results])
            assert avg_duration < recovery_config.timeout_ms

            metrics_collector.record_custom(
                "slow_node_handling",
                {"avg_round_duration_ms": avg_duration}
            )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_intermittent_node_handling(
        self, recovery_config, metrics_collector
    ):
        """Test handling of intermittent node failures."""
        network = FLNetwork(recovery_config)
        await network.start()

        try:
            nodes = network.get_all_nodes()

            # Make some nodes intermittent
            for i, node in enumerate(nodes.values()):
                if i < 4:
                    node.dropout_probability = 0.3  # 30% chance of failure per round

            # Run many rounds
            n_rounds = 20
            results = []
            for _ in range(n_rounds):
                result = await network.run_round()
                results.append(result)

            # Most rounds should succeed
            success_count = sum(1 for r in results if r["success"])
            success_rate = success_count / n_rounds

            assert success_rate >= 0.8, f"Too many failures: {success_rate}"

            # Participant count should vary
            participant_counts = [
                r.get("n_participants", 0) for r in results if r["success"]
            ]
            if participant_counts:
                std_participants = np.std(participant_counts)
                assert std_participants > 0, "Should have variable participation"

            metrics_collector.record_custom(
                "intermittent_handling",
                {
                    "success_rate": success_rate,
                    "avg_participants": np.mean(participant_counts) if participant_counts else 0,
                    "std_participants": std_participants if participant_counts else 0,
                }
            )

        finally:
            await network.shutdown()


# ============================================================================
# Comprehensive Recovery Test
# ============================================================================

class TestComprehensiveRecovery:
    """Comprehensive recovery scenario test."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.network
    async def test_full_recovery_scenario(
        self, fault_tolerant_config, metrics_collector, test_report
    ):
        """Test complete recovery scenario with multiple disruptions."""
        network = FLNetwork(fault_tolerant_config)
        await network.start()

        test_report.add_tag("recovery")
        test_report.add_tag("comprehensive")

        try:
            phases = []

            # Phase 1: Stable operation (10 rounds)
            phase1_results = await network.run_rounds(10)
            phases.append({
                "name": "stable",
                "success_rate": sum(1 for r in phase1_results if r["success"]) / 10,
                "avg_participants": np.mean([
                    r.get("n_participants", 0) for r in phase1_results
                ]),
            })

            # Phase 2: 30% node failure (10 rounds)
            nodes = network.get_all_nodes()
            failure_nodes = list(nodes.keys())[:6]
            for node_id in failure_nodes:
                nodes[node_id].dropout_probability = 1.0

            phase2_results = await network.run_rounds(10)
            phases.append({
                "name": "30%_failure",
                "success_rate": sum(1 for r in phase2_results if r["success"]) / 10,
                "avg_participants": np.mean([
                    r.get("n_participants", 0) for r in phase2_results
                ]),
            })

            # Phase 3: Partial recovery (5 rounds)
            for node_id in failure_nodes[:3]:
                nodes[node_id].dropout_probability = 0.0

            phase3_results = await network.run_rounds(5)
            phases.append({
                "name": "partial_recovery",
                "success_rate": sum(1 for r in phase3_results if r["success"]) / 5,
                "avg_participants": np.mean([
                    r.get("n_participants", 0) for r in phase3_results
                ]),
            })

            # Phase 4: Full recovery (10 rounds)
            for node in nodes.values():
                node.dropout_probability = 0.0

            phase4_results = await network.run_rounds(10)
            phases.append({
                "name": "full_recovery",
                "success_rate": sum(1 for r in phase4_results if r["success"]) / 10,
                "avg_participants": np.mean([
                    r.get("n_participants", 0) for r in phase4_results
                ]),
            })

            # Verify recovery
            assert phases[0]["success_rate"] >= 0.9, "Initial stable phase should succeed"
            assert phases[3]["success_rate"] >= 0.9, "Full recovery should succeed"
            assert phases[3]["avg_participants"] >= phases[0]["avg_participants"] * 0.9

            metrics_collector.record_custom("recovery_phases", phases)
            test_report.passed = True

        except Exception as e:
            test_report.fail(str(e))
            raise

        finally:
            await network.shutdown()
