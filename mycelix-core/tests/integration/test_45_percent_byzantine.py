# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Maximum Byzantine Tolerance (45%)

Tests the system at the theoretical limits of Byzantine fault tolerance.
With 45% Byzantine nodes, the system should still maintain:
- Aggregation quality above threshold
- Detection accuracy
- System stability

This is a stress test for Byzantine resilience mechanisms.
"""

import pytest
import numpy as np
import asyncio
from typing import List, Dict, Set, Any

from .fixtures.network import FLNetwork, NetworkConfig, FLAggregator
from .fixtures.nodes import ByzantineNode, HonestNode, AttackType
from .fixtures.metrics import MetricsCollector, QualityAssertions, TestReport


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture
def max_byzantine_45_config():
    """Configuration with 45% Byzantine nodes."""
    return NetworkConfig(
        num_honest_nodes=11,
        num_byzantine_nodes=9,  # 45% of 20
        gradient_dimension=2000,
        aggregation_strategy="bulyan",  # Most resilient strategy
        byzantine_threshold=0.45,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
        consensus_rounds=5,
        timeout_ms=20000,
        min_participants=10,
    )


@pytest.fixture
def extreme_byzantine_config():
    """Configuration pushing beyond 45% (should struggle)."""
    return NetworkConfig(
        num_honest_nodes=10,
        num_byzantine_nodes=10,  # 50% - at theoretical limit
        gradient_dimension=1000,
        aggregation_strategy="bulyan",
        byzantine_threshold=0.5,
        detection_enabled=True,
        reputation_enabled=True,
        holochain_mock=True,
    )


# ============================================================================
# Basic 45% Byzantine Tests
# ============================================================================

class TestMaxByzantineTolerance:
    """Test system behavior at 45% Byzantine ratio."""

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    @pytest.mark.stress
    async def test_45_percent_single_round(
        self, max_byzantine_45_config, metrics_collector
    ):
        """Test a single round with 45% Byzantine nodes."""
        config = max_byzantine_45_config
        network = FLNetwork(config)
        await network.start()

        try:
            # Verify configuration
            stats = network.get_stats()
            byzantine_ratio = stats["byzantine_nodes"] / stats["total_nodes"]
            assert abs(byzantine_ratio - 0.45) < 0.01

            # Run round
            result = await network.run_round()

            # Should still succeed
            assert result["success"], f"Round failed: {result.get('error')}"

            # Check aggregation quality
            entries = await network.gradient_store.get_gradients_for_round(
                result["round_id"]
            )
            honest_grads = [e.gradient for e in entries if "honest" in e.node_id]

            if honest_grads:
                honest_mean = np.mean(honest_grads, axis=0)
                aggregated = result["aggregated_gradient"]

                quality = np.dot(aggregated, honest_mean) / (
                    np.linalg.norm(aggregated) * np.linalg.norm(honest_mean) + 1e-10
                )

                metrics_collector.record_aggregation(
                    round_id=result["round_id"],
                    aggregated=aggregated,
                    ground_truth=honest_mean,
                    n_participants=result["n_participants"],
                    n_byzantine=config.num_byzantine_nodes,
                    n_detected=result["n_byzantine_detected"],
                    strategy=config.aggregation_strategy,
                    duration_ms=result["duration_ms"],
                )

                # With Bulyan at 45%, quality should be above 0.5
                assert quality > 0.5, f"Quality too low at 45% Byzantine: {quality}"

        finally:
            await network.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    @pytest.mark.stress
    async def test_45_percent_multiple_rounds(
        self, max_byzantine_45_config, metrics_collector
    ):
        """Test multiple rounds with 45% Byzantine nodes."""
        config = max_byzantine_45_config
        network = FLNetwork(config)
        await network.start()

        try:
            n_rounds = 20
            qualities = []
            detection_rates = []

            true_byzantines = {
                f"byzantine_{i}" for i in range(config.num_byzantine_nodes)
            }

            for i in range(n_rounds):
                result = await network.run_round()

                if not result["success"]:
                    continue

                # Measure quality
                entries = await network.gradient_store.get_gradients_for_round(
                    result["round_id"]
                )
                honest_grads = [e.gradient for e in entries if "honest" in e.node_id]

                if honest_grads:
                    honest_mean = np.mean(honest_grads, axis=0)
                    agg = result["aggregated_gradient"]
                    quality = np.dot(agg, honest_mean) / (
                        np.linalg.norm(agg) * np.linalg.norm(honest_mean) + 1e-10
                    )
                    qualities.append(quality)

                # Measure detection
                detected = set(result.get("detected_byzantine", []))
                tp = len(detected & true_byzantines)
                detection_rates.append(tp / len(true_byzantines))

            # Verify sustained quality
            assert len(qualities) >= n_rounds * 0.9, "Too many failures"

            avg_quality = np.mean(qualities)
            min_quality = np.min(qualities)

            assert avg_quality > 0.6, f"Average quality {avg_quality} too low"
            assert min_quality > 0.4, f"Minimum quality {min_quality} too low"

            metrics_collector.record_custom(
                "45_percent_quality",
                {
                    "avg": avg_quality,
                    "min": min_quality,
                    "std": np.std(qualities),
                }
            )

        finally:
            await network.shutdown()


# ============================================================================
# Attack Strategy Tests at 45%
# ============================================================================

class TestAttackStrategiesAt45Percent:
    """Test various attack strategies at maximum Byzantine ratio."""

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    @pytest.mark.stress
    @pytest.mark.parametrize("attack_type", [
        "random",
        "sign_flip",
        "scaling",
        "little",
        "empire",
    ])
    async def test_attack_at_45_percent(
        self, attack_type, metrics_collector
    ):
        """Test resilience against specific attack at 45% Byzantine."""
        gradient_dim = 1000
        n_honest = 11
        n_byzantine = 9

        # Create nodes
        honest_nodes = [
            HonestNode(f"honest_{i}", gradient_dim)
            for i in range(n_honest)
        ]

        byzantine_nodes = [
            ByzantineNode(
                f"byzantine_{i}",
                gradient_dim,
                attack_type=attack_type,
                attack_params={
                    "n_byzantine": n_byzantine,
                    "n_total": n_honest + n_byzantine,
                },
            )
            for i in range(n_byzantine)
        ]

        # Collect honest gradients first
        honest_gradients = [await node.compute_gradient(1) for node in honest_nodes]
        honest_mean = np.mean(honest_gradients, axis=0)

        # Let Byzantine nodes observe
        for byz in byzantine_nodes:
            for hg in honest_gradients:
                byz.observe_gradient(hg)

        # Collect Byzantine gradients
        byzantine_gradients = [await node.compute_gradient(1) for node in byzantine_nodes]

        # Combine all gradients
        all_gradients = honest_gradients + byzantine_gradients

        # Test different aggregation strategies
        strategies = ["bulyan", "multi_krum", "trimmed_mean"]
        results = {}

        for strategy in strategies:
            aggregator = FLAggregator(strategy=strategy, byzantine_threshold=0.45)
            aggregated, _ = await aggregator.aggregate(all_gradients)

            quality = np.dot(aggregated, honest_mean) / (
                np.linalg.norm(aggregated) * np.linalg.norm(honest_mean) + 1e-10
            )
            results[strategy] = quality

        # Bulyan should perform best at high Byzantine ratios
        assert results["bulyan"] >= max(results.values()) - 0.1

        metrics_collector.record_custom(
            f"attack_{attack_type}_quality",
            results
        )

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    @pytest.mark.stress
    async def test_coordinated_attack_at_45_percent(self, metrics_collector):
        """Test coordinated Byzantine attack at 45%."""
        gradient_dim = 1000
        n_honest = 11
        n_byzantine = 9

        honest_nodes = [HonestNode(f"honest_{i}", gradient_dim) for i in range(n_honest)]

        # All Byzantine nodes coordinate with "empire" attack
        byzantine_nodes = [
            ByzantineNode(
                f"byzantine_{i}",
                gradient_dim,
                attack_type="empire",
                attack_params={
                    "n_byzantine": n_byzantine,
                    "n_total": n_honest + n_byzantine,
                    "target_shift": np.ones(gradient_dim) * -10,  # Coordinated target
                },
            )
            for i in range(n_byzantine)
        ]

        # Gather gradients
        honest_gradients = [await node.compute_gradient(1) for node in honest_nodes]
        honest_mean = np.mean(honest_gradients, axis=0)

        for byz in byzantine_nodes:
            for hg in honest_gradients:
                byz.observe_gradient(hg)

        byzantine_gradients = [await node.compute_gradient(1) for node in byzantine_nodes]
        all_gradients = honest_gradients + byzantine_gradients

        # Test resilience
        aggregator = FLAggregator(strategy="bulyan", byzantine_threshold=0.45)
        aggregated, _ = await aggregator.aggregate(all_gradients)

        quality = np.dot(aggregated, honest_mean) / (
            np.linalg.norm(aggregated) * np.linalg.norm(honest_mean) + 1e-10
        )

        # Should maintain some quality even under coordinated attack
        assert quality > 0.3, f"Coordinated attack broke aggregation: quality={quality}"

        # The aggregated gradient should not be close to attack target
        target = np.ones(gradient_dim) * -10
        target_similarity = np.dot(aggregated, target) / (
            np.linalg.norm(aggregated) * np.linalg.norm(target) + 1e-10
        )

        assert target_similarity < 0.5, "Attack succeeded in shifting aggregate"


# ============================================================================
# Aggregation Strategy Comparison at 45%
# ============================================================================

class TestAggregationAt45Percent:
    """Compare aggregation strategies at 45% Byzantine."""

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    @pytest.mark.stress
    async def test_strategy_comparison(self, metrics_collector, test_report):
        """Compare all aggregation strategies at 45% Byzantine."""
        gradient_dim = 1000
        n_honest = 11
        n_byzantine = 9

        test_report.set_parameter("byzantine_ratio", 0.45)
        test_report.add_tag("strategy_comparison")

        # Create nodes
        honest_nodes = [HonestNode(f"honest_{i}", gradient_dim) for i in range(n_honest)]
        byzantine_nodes = [
            ByzantineNode(f"byzantine_{i}", gradient_dim, attack_type="random")
            for i in range(n_byzantine)
        ]

        honest_gradients = [await node.compute_gradient(1) for node in honest_nodes]
        byzantine_gradients = [await node.compute_gradient(1) for node in byzantine_nodes]
        all_gradients = honest_gradients + byzantine_gradients

        honest_mean = np.mean(honest_gradients, axis=0)

        strategies = [
            "fedavg",
            "trimmed_mean",
            "median",
            "krum",
            "multi_krum",
            "bulyan",
            "geometric_median",
        ]

        results = {}

        for strategy in strategies:
            try:
                aggregator = FLAggregator(strategy=strategy, byzantine_threshold=0.45)
                aggregated, _ = await aggregator.aggregate(all_gradients)

                quality = np.dot(aggregated, honest_mean) / (
                    np.linalg.norm(aggregated) * np.linalg.norm(honest_mean) + 1e-10
                )
                results[strategy] = float(quality)
            except Exception as e:
                results[strategy] = f"Error: {str(e)}"

        # Record results
        metrics_collector.record_custom("strategy_comparison_45_percent", results)

        # Verify Byzantine-resilient strategies outperform FedAvg
        if isinstance(results.get("fedavg"), float) and isinstance(results.get("bulyan"), float):
            assert results["bulyan"] > results["fedavg"], (
                "Bulyan should outperform FedAvg at 45% Byzantine"
            )

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    @pytest.mark.stress
    async def test_bulyan_properties_at_45_percent(self, metrics_collector):
        """Test specific Bulyan properties at 45% Byzantine."""
        gradient_dim = 1000
        n_honest = 11
        n_byzantine = 9

        # Run multiple trials
        n_trials = 10
        qualities = []

        for trial in range(n_trials):
            honest_nodes = [
                HonestNode(f"honest_{i}", gradient_dim, noise_scale=0.01)
                for i in range(n_honest)
            ]
            byzantine_nodes = [
                ByzantineNode(f"byzantine_{i}", gradient_dim, attack_type="random")
                for i in range(n_byzantine)
            ]

            honest_gradients = [await node.compute_gradient(trial + 1) for node in honest_nodes]
            byzantine_gradients = [await node.compute_gradient(trial + 1) for node in byzantine_nodes]

            all_gradients = honest_gradients + byzantine_gradients
            np.random.shuffle(all_gradients)  # Randomize order

            honest_mean = np.mean(honest_gradients, axis=0)

            aggregator = FLAggregator(strategy="bulyan", byzantine_threshold=0.45)
            aggregated, metadata = await aggregator.aggregate(all_gradients)

            quality = np.dot(aggregated, honest_mean) / (
                np.linalg.norm(aggregated) * np.linalg.norm(honest_mean) + 1e-10
            )
            qualities.append(quality)

        # Bulyan should be consistent
        avg_quality = np.mean(qualities)
        std_quality = np.std(qualities)

        assert avg_quality > 0.5, f"Bulyan average quality too low: {avg_quality}"
        assert std_quality < 0.3, f"Bulyan too variable: std={std_quality}"

        metrics_collector.record_custom(
            "bulyan_45_percent_stats",
            {
                "avg_quality": avg_quality,
                "std_quality": std_quality,
                "min_quality": min(qualities),
                "max_quality": max(qualities),
            }
        )


# ============================================================================
# Boundary Tests
# ============================================================================

class TestByzantineBoundaries:
    """Test behavior at Byzantine ratio boundaries."""

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    @pytest.mark.stress
    @pytest.mark.parametrize("byzantine_ratio", [0.30, 0.33, 0.40, 0.45, 0.49])
    async def test_quality_vs_byzantine_ratio(
        self, byzantine_ratio, metrics_collector
    ):
        """Test quality degradation as Byzantine ratio increases."""
        total_nodes = 20
        n_byzantine = int(total_nodes * byzantine_ratio)
        n_honest = total_nodes - n_byzantine
        gradient_dim = 1000

        honest_nodes = [HonestNode(f"honest_{i}", gradient_dim) for i in range(n_honest)]
        byzantine_nodes = [
            ByzantineNode(f"byzantine_{i}", gradient_dim, attack_type="random")
            for i in range(n_byzantine)
        ]

        honest_gradients = [await node.compute_gradient(1) for node in honest_nodes]
        byzantine_gradients = [await node.compute_gradient(1) for node in byzantine_nodes]
        all_gradients = honest_gradients + byzantine_gradients

        honest_mean = np.mean(honest_gradients, axis=0)

        aggregator = FLAggregator(strategy="bulyan", byzantine_threshold=byzantine_ratio + 0.05)
        aggregated, _ = await aggregator.aggregate(all_gradients)

        quality = np.dot(aggregated, honest_mean) / (
            np.linalg.norm(aggregated) * np.linalg.norm(honest_mean) + 1e-10
        )

        metrics_collector.record_custom(
            f"quality_at_{int(byzantine_ratio*100)}_percent",
            quality
        )

        # Quality should decrease with Byzantine ratio but stay above threshold
        if byzantine_ratio <= 0.33:
            assert quality > 0.7, f"Quality too low at {byzantine_ratio*100}%"
        elif byzantine_ratio <= 0.45:
            assert quality > 0.5, f"Quality too low at {byzantine_ratio*100}%"
        else:
            # Above theoretical limit, quality may degrade significantly
            pass

    @pytest.mark.asyncio
    @pytest.mark.byzantine
    @pytest.mark.stress
    async def test_beyond_theoretical_limit(
        self, extreme_byzantine_config, metrics_collector
    ):
        """Test behavior beyond the theoretical Byzantine limit (50%)."""
        config = extreme_byzantine_config
        network = FLNetwork(config)
        await network.start()

        try:
            # This should struggle but not crash
            result = await network.run_round()

            # May or may not succeed
            metrics_collector.record_custom(
                "beyond_limit_result",
                {
                    "success": result["success"],
                    "byzantine_ratio": 0.5,
                    "error": result.get("error"),
                }
            )

            if result["success"]:
                # If it succeeds, quality may be low
                entries = await network.gradient_store.get_gradients_for_round(
                    result["round_id"]
                )
                honest_grads = [e.gradient for e in entries if "honest" in e.node_id]

                if honest_grads:
                    honest_mean = np.mean(honest_grads, axis=0)
                    agg = result["aggregated_gradient"]
                    quality = np.dot(agg, honest_mean) / (
                        np.linalg.norm(agg) * np.linalg.norm(honest_mean) + 1e-10
                    )

                    metrics_collector.record_custom("beyond_limit_quality", quality)
                    # No assertion on quality - we're beyond guarantees

        finally:
            await network.shutdown()


# ============================================================================
# Long-Running 45% Byzantine Test
# ============================================================================

class TestLongRunning45Percent:
    """Long-running tests at 45% Byzantine ratio."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.byzantine
    @pytest.mark.stress
    async def test_50_rounds_at_45_percent(
        self, max_byzantine_45_config, metrics_collector, test_report
    ):
        """Test 50 rounds at 45% Byzantine ratio."""
        config = max_byzantine_45_config
        network = FLNetwork(config)
        await network.start()

        test_report.set_parameter("byzantine_ratio", 0.45)
        test_report.set_parameter("n_rounds", 50)
        test_report.add_tag("long_running_45_percent")

        try:
            n_rounds = 50
            qualities = []
            successes = 0

            for i in range(n_rounds):
                result = await network.run_round()

                if result["success"]:
                    successes += 1

                    entries = await network.gradient_store.get_gradients_for_round(
                        result["round_id"]
                    )
                    honest_grads = [e.gradient for e in entries if "honest" in e.node_id]

                    if honest_grads:
                        honest_mean = np.mean(honest_grads, axis=0)
                        agg = result["aggregated_gradient"]
                        quality = np.dot(agg, honest_mean) / (
                            np.linalg.norm(agg) * np.linalg.norm(honest_mean) + 1e-10
                        )
                        qualities.append(quality)

            success_rate = successes / n_rounds
            avg_quality = np.mean(qualities) if qualities else 0
            min_quality = np.min(qualities) if qualities else 0

            # At 45%, should maintain high success rate
            assert success_rate >= 0.9, f"Success rate {success_rate} too low"

            # Quality should be maintained
            assert avg_quality >= 0.5, f"Average quality {avg_quality} too low"
            assert min_quality >= 0.3, f"Minimum quality {min_quality} too low"

            metrics_collector.record_custom(
                "50_rounds_45_percent_summary",
                {
                    "success_rate": success_rate,
                    "avg_quality": avg_quality,
                    "min_quality": min_quality,
                    "std_quality": np.std(qualities) if qualities else 0,
                }
            )

            test_report.passed = True

        except Exception as e:
            test_report.fail(str(e))
            raise

        finally:
            await network.shutdown()
