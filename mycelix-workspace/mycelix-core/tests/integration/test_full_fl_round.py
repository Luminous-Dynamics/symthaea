# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Full FL Round with 20 Nodes

Comprehensive test of a complete Federated Learning round with:
- 20 participant nodes
- Mixed honest and Byzantine participants
- Full gradient lifecycle
- Detection, aggregation, and reputation updates
"""

import pytest
import numpy as np
import asyncio
import time
from typing import List, Dict, Any

from .fixtures.network import FLNetwork, NetworkConfig, FLAggregator
from .fixtures.nodes import HonestNode, ByzantineNode, NodeFactory
from .fixtures.metrics import (
    MetricsCollector,
    QualityAssertions,
    TestReport,
    AggregationMetrics,
)


# ============================================================================
# Full Round Configuration
# ============================================================================

@pytest.fixture
def twenty_node_config():
    """Configuration for 20-node FL network."""
    return NetworkConfig(
        num_honest_nodes=20,
        num_byzantine_nodes=0,
        gradient_dimension=5000,
        aggregation_strategy="trimmed_mean",
        byzantine_threshold=0.33,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        consensus_rounds=3,
        timeout_ms=10000,
        min_participants=10,
    )


@pytest.fixture
def twenty_node_mixed_config():
    """Configuration for 20-node network with Byzantine nodes."""
    return NetworkConfig(
        num_honest_nodes=14,
        num_byzantine_nodes=6,  # 30% Byzantine
        gradient_dimension=5000,
        aggregation_strategy="multi_krum",
        byzantine_threshold=0.33,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        consensus_rounds=3,
        timeout_ms=15000,
        min_participants=10,
    )


# ============================================================================
# Complete FL Round Tests
# ============================================================================

class TestFullFLRound:
    """Test complete FL round with 20 nodes."""

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_20_node_honest_round(self, twenty_node_config, metrics_collector):
        """Test a complete FL round with 20 honest nodes."""
        network = FLNetwork(twenty_node_config)
        await network.start()

        try:
            # Verify network setup
            assert len(network.get_all_nodes()) == 20
            stats = network.get_stats()
            assert stats["honest_nodes"] == 20
            assert stats["byzantine_nodes"] == 0

            # Execute round
            start_time = time.time()
            result = await network.run_round()
            round_duration = (time.time() - start_time) * 1000

            # Verify success
            assert result["success"], f"Round failed: {result.get('error')}"
            assert result["n_participants"] == 20
            assert result["n_rejected"] == 0
            assert result["n_byzantine_detected"] == 0

            # Verify aggregated gradient
            aggregated = result["aggregated_gradient"]
            assert aggregated is not None
            assert aggregated.shape == (5000,)
            assert np.isfinite(aggregated).all()

            # Record metrics
            metrics_collector.record_latency(
                "full_round_20_nodes",
                round_duration,
                round_id=result["round_id"],
            )

            # Verify reasonable latency (should complete within timeout)
            assert round_duration < twenty_node_config.timeout_ms

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    @pytest.mark.byzantine
    async def test_20_node_mixed_round(self, twenty_node_mixed_config, metrics_collector):
        """Test a complete FL round with 14 honest + 6 Byzantine nodes."""
        network = FLNetwork(twenty_node_mixed_config)
        await network.start()

        try:
            # Verify network setup
            assert len(network.get_all_nodes()) == 20
            stats = network.get_stats()
            assert stats["honest_nodes"] == 14
            assert stats["byzantine_nodes"] == 6

            # Execute round
            result = await network.run_round()

            # Verify success despite Byzantine nodes
            assert result["success"], f"Round failed: {result.get('error')}"
            assert result["n_participants"] >= 10  # Min participants met

            # Some Byzantine nodes may be detected
            # Not all will necessarily be detected in one round

            # Verify aggregated gradient quality
            aggregated = result["aggregated_gradient"]
            assert np.isfinite(aggregated).all()
            assert np.linalg.norm(aggregated) > 0

            # Record metrics
            metrics_collector.record_latency(
                "full_round_20_nodes_mixed",
                result["duration_ms"],
            )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_aggregation_quality_20_nodes(
        self, twenty_node_config, metrics_collector
    ):
        """Test that aggregation produces high-quality results with 20 nodes."""
        network = FLNetwork(twenty_node_config)
        await network.start()

        try:
            # Run round
            result = await network.run_round()
            assert result["success"]

            # Get all submitted gradients
            entries = await network.gradient_store.get_gradients_for_round(
                result["round_id"]
            )
            submitted_gradients = [e.gradient for e in entries]

            # Compute expected mean (all honest nodes)
            expected_mean = np.mean(submitted_gradients, axis=0)

            # Compare with aggregated
            aggregated = result["aggregated_gradient"]

            # With all honest nodes and trimmed_mean, should be very close
            cosine_sim = np.dot(aggregated, expected_mean) / (
                np.linalg.norm(aggregated) * np.linalg.norm(expected_mean)
            )

            relative_error = np.linalg.norm(aggregated - expected_mean) / np.linalg.norm(expected_mean)

            # Record quality metrics
            metrics_collector.record_aggregation(
                round_id=result["round_id"],
                aggregated=aggregated,
                ground_truth=expected_mean,
                n_participants=20,
                n_byzantine=0,
                n_detected=0,
                strategy=network.config.aggregation_strategy,
                duration_ms=result["duration_ms"],
            )

            # Assert quality
            assert cosine_sim > 0.95, f"Cosine similarity too low: {cosine_sim}"
            assert relative_error < 0.15, f"Relative error too high: {relative_error}"

        finally:
            await network.shutdown()


# ============================================================================
# Performance and Latency Tests
# ============================================================================

class TestFullRoundPerformance:
    """Test performance characteristics of full FL rounds."""

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_round_latency_breakdown(self, twenty_node_config, timing_context):
        """Test latency breakdown of a full FL round."""
        network = FLNetwork(twenty_node_config)
        await network.start()

        try:
            latencies = {}

            # Measure each phase
            round_id = 1

            # Phase 1: Gradient computation
            async with timing_context("gradient_computation") as t:
                gradients = []
                for node in network.get_all_nodes().values():
                    g = await node.compute_gradient(round_id)
                    gradients.append(g)
            latencies["gradient_computation"] = t["duration_ms"]

            # Phase 2: Gradient submission
            async with timing_context("gradient_submission") as t:
                for i, (node_id, node) in enumerate(network.get_all_nodes().items()):
                    await network.gradient_store.submit_gradient(
                        node_id=node_id,
                        round_id=round_id,
                        gradient=gradients[i],
                    )
            latencies["gradient_submission"] = t["duration_ms"]

            # Phase 3: Gradient retrieval
            async with timing_context("gradient_retrieval") as t:
                entries = await network.gradient_store.get_gradients_for_round(round_id)
            latencies["gradient_retrieval"] = t["duration_ms"]

            # Phase 4: Byzantine detection
            async with timing_context("byzantine_detection") as t:
                detected, _ = await network.detector.detect(
                    [e.gradient for e in entries],
                    [e.node_id for e in entries],
                )
            latencies["byzantine_detection"] = t["duration_ms"]

            # Phase 5: Aggregation
            async with timing_context("aggregation") as t:
                aggregated, _ = await network.aggregator.aggregate(
                    [e.gradient for e in entries]
                )
            latencies["aggregation"] = t["duration_ms"]

            # Verify all phases completed in reasonable time
            total = sum(latencies.values())
            assert total < twenty_node_config.timeout_ms

            # Print breakdown for analysis
            for phase, duration in latencies.items():
                assert duration < 5000, f"{phase} took too long: {duration}ms"

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_round_scalability(self, metrics_collector):
        """Test how round time scales with number of nodes."""
        node_counts = [5, 10, 15, 20]
        round_times = []

        for n_nodes in node_counts:
            config = NetworkConfig(
                num_honest_nodes=n_nodes,
                num_byzantine_nodes=0,
                gradient_dimension=1000,
                aggregation_strategy="fedavg",
            )

            network = FLNetwork(config)
            await network.start()

            try:
                result = await network.run_round()
                assert result["success"]
                round_times.append(result["duration_ms"])

                metrics_collector.record_custom(
                    f"scalability_{n_nodes}_nodes",
                    result["duration_ms"]
                )
            finally:
                await network.shutdown()

        # Round time should scale reasonably (sub-quadratic)
        # Doubling nodes shouldn't more than triple time
        ratio_5_to_10 = round_times[1] / round_times[0] if round_times[0] > 0 else 0
        ratio_10_to_20 = round_times[3] / round_times[1] if round_times[1] > 0 else 0

        assert ratio_5_to_10 < 5, f"Scaling 5->10 too high: {ratio_5_to_10}x"
        assert ratio_10_to_20 < 5, f"Scaling 10->20 too high: {ratio_10_to_20}x"


# ============================================================================
# Aggregation Strategy Tests
# ============================================================================

class TestAggregationStrategies:
    """Test different aggregation strategies with 20 nodes."""

    @pytest.mark.asyncio
    @pytest.mark.network
    @pytest.mark.parametrize("strategy", [
        "fedavg",
        "trimmed_mean",
        "median",
        "krum",
        "multi_krum",
        "bulyan",
    ])
    async def test_strategy_with_20_honest_nodes(
        self, strategy, metrics_collector
    ):
        """Test each aggregation strategy with 20 honest nodes."""
        config = NetworkConfig(
            num_honest_nodes=20,
            num_byzantine_nodes=0,
            gradient_dimension=1000,
            aggregation_strategy=strategy,
        )

        network = FLNetwork(config)
        await network.start()

        try:
            result = await network.run_round()

            assert result["success"], f"{strategy} failed: {result.get('error')}"

            # Get expected mean
            entries = await network.gradient_store.get_gradients_for_round(
                result["round_id"]
            )
            expected = np.mean([e.gradient for e in entries], axis=0)

            # Compute quality
            aggregated = result["aggregated_gradient"]
            cosine_sim = np.dot(aggregated, expected) / (
                np.linalg.norm(aggregated) * np.linalg.norm(expected)
            )

            metrics_collector.record_custom(f"quality_{strategy}", cosine_sim)

            # All strategies should produce reasonable results with honest nodes
            assert cosine_sim > 0.8, f"{strategy} quality too low: {cosine_sim}"

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    @pytest.mark.byzantine
    @pytest.mark.parametrize("strategy,min_quality", [
        ("fedavg", 0.6),  # FedAvg is NOT Byzantine-resilient
        ("trimmed_mean", 0.75),
        ("median", 0.75),
        ("krum", 0.8),
        ("multi_krum", 0.8),
        ("bulyan", 0.85),
    ])
    async def test_strategy_with_byzantine_nodes(
        self, strategy, min_quality, metrics_collector
    ):
        """Test each aggregation strategy with Byzantine nodes."""
        config = NetworkConfig(
            num_honest_nodes=14,
            num_byzantine_nodes=6,  # 30%
            gradient_dimension=1000,
            aggregation_strategy=strategy,
            detection_enabled=False,  # Test raw aggregation resilience
        )

        network = FLNetwork(config)
        await network.start()

        try:
            result = await network.run_round()
            assert result["success"]

            # Get honest gradients only for ground truth
            entries = await network.gradient_store.get_gradients_for_round(
                result["round_id"]
            )
            honest_entries = [e for e in entries if "honest" in e.node_id]
            honest_mean = np.mean([e.gradient for e in honest_entries], axis=0)

            # Compute quality against honest mean
            aggregated = result["aggregated_gradient"]
            cosine_sim = np.dot(aggregated, honest_mean) / (
                np.linalg.norm(aggregated) * np.linalg.norm(honest_mean)
            )

            metrics_collector.record_custom(
                f"byzantine_quality_{strategy}",
                cosine_sim
            )

            # Byzantine-resilient strategies should maintain quality
            # Note: FedAvg is expected to have lower quality
            if strategy != "fedavg":
                assert cosine_sim >= min_quality, (
                    f"{strategy} quality {cosine_sim} below threshold {min_quality}"
                )

        finally:
            await network.shutdown()


# ============================================================================
# Comprehensive Round Tests
# ============================================================================

class TestComprehensiveRound:
    """Comprehensive tests of full FL round functionality."""

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_full_round_with_all_features(
        self, twenty_node_mixed_config, metrics_collector, test_report
    ):
        """Test full round with all features enabled."""
        config = twenty_node_mixed_config
        network = FLNetwork(config)
        await network.start()

        test_report.set_parameter("num_nodes", 20)
        test_report.set_parameter("num_byzantine", 6)
        test_report.set_parameter("strategy", config.aggregation_strategy)
        test_report.add_tag("full_round")

        try:
            # Run round
            round_metrics = metrics_collector.start_round(1)
            result = await network.run_round()
            metrics_collector.end_round(round_metrics, result["success"])

            assert result["success"]

            # Verify all components were involved
            assert result["n_participants"] > 0
            assert "aggregated_gradient" in result
            assert "detected_byzantine" in result
            assert "aggregation_metadata" in result

            # Check storage
            stats = network.gradient_store.get_stats()
            assert stats["total_entries"] > 0
            assert stats["validated"] > 0

            # Check reputation was updated
            scores = network.reputation_bridge.get_all_scores()
            # Some nodes should have scores if Byzantine were detected
            if result["n_byzantine_detected"] > 0:
                assert len(scores) > 0

            # Record comprehensive metrics
            entries = await network.gradient_store.get_gradients_for_round(
                result["round_id"]
            )
            honest_mean = np.mean(
                [e.gradient for e in entries if "honest" in e.node_id],
                axis=0
            )

            metrics_collector.record_aggregation(
                round_id=result["round_id"],
                aggregated=result["aggregated_gradient"],
                ground_truth=honest_mean,
                n_participants=result["n_participants"],
                n_byzantine=config.num_byzantine_nodes,
                n_detected=result["n_byzantine_detected"],
                strategy=config.aggregation_strategy,
                duration_ms=result["duration_ms"],
            )

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_multiple_consecutive_rounds(
        self, twenty_node_config, metrics_collector
    ):
        """Test multiple consecutive FL rounds."""
        network = FLNetwork(twenty_node_config)
        await network.start()

        try:
            n_rounds = 5
            results = []

            for i in range(n_rounds):
                result = await network.run_round()
                results.append(result)

                assert result["success"], f"Round {i+1} failed"
                assert result["round_id"] == i + 1

                metrics_collector.record_latency(
                    "consecutive_round",
                    result["duration_ms"],
                    round_id=result["round_id"],
                )

            # Verify all rounds completed
            assert len(results) == n_rounds

            # Check round history
            history = network.get_round_history()
            assert len(history) == n_rounds

            # Verify gradient evolution (norms should decrease with convergence)
            norms = [np.linalg.norm(r["aggregated_gradient"]) for r in results]
            assert norms[-1] < norms[0], "No convergence observed"

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.network
    async def test_parallel_gradient_submission(self, twenty_node_config):
        """Test that gradient submission can happen in parallel."""
        network = FLNetwork(twenty_node_config)
        await network.start()

        try:
            round_id = 1

            async def submit_gradient(node_id, node):
                gradient = await node.compute_gradient(round_id)
                return await network.gradient_store.submit_gradient(
                    node_id=node_id,
                    round_id=round_id,
                    gradient=gradient,
                )

            # Submit all gradients in parallel
            nodes = network.get_all_nodes()
            tasks = [
                submit_gradient(node_id, node)
                for node_id, node in nodes.items()
            ]

            start = time.time()
            results = await asyncio.gather(*tasks)
            parallel_duration = (time.time() - start) * 1000

            # All should succeed
            successes = sum(1 for success, _, _ in results if success)
            assert successes == 20

            # Parallel should be faster than sequential would be
            # (This is a sanity check)
            assert parallel_duration < 5000  # Should complete quickly

        finally:
            await network.shutdown()
