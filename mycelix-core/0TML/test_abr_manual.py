# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Manual test script for Adaptive Byzantine Resistance
Verifies the implementation without pytest
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import asyncio
from zerotrustml.experimental.adaptive_byzantine_resistance import (
    AdaptiveByzantineResistance,
    ReputationLevel,
    DynamicThresholdManager,
    ReputationRecoveryManager,
    ByzantineResistantAggregator
)


def test_multi_level_reputation():
    """Test multi-level reputation tracking"""
    print("\n=== Test: Multi-Level Reputation ===")

    abr = AdaptiveByzantineResistance()

    # Test initial reputation
    profile = abr.get_or_create_profile(node_id=1)
    assert profile.overall_reputation == 0.7, "Initial reputation should be 0.7"
    print("✓ Initial reputation correct")

    # Submit valid gradients
    for i in range(10):
        gradient = np.random.randn(100)
        abr.update_reputation(
            node_id=1,
            gradient=gradient,
            validation_passed=True,
            pogq_score=0.9,
            anomaly_score=0.0
        )

    profile = abr.get_or_create_profile(1)
    assert profile.overall_reputation > 0.7, "Reputation should improve"
    assert profile.gradient_reputation == 1.0, "Should have 100% success rate"
    print(f"✓ Reputation improved to {profile.overall_reputation:.3f}")
    print(f"✓ Gradient reputation: {profile.gradient_reputation:.3f}")

    # Test reputation degradation
    for i in range(10):
        gradient = np.random.randn(100)
        abr.update_reputation(
            node_id=2,
            gradient=gradient,
            validation_passed=False,
            pogq_score=0.2,
            anomaly_score=0.9
        )

    profile2 = abr.get_or_create_profile(2)
    assert profile2.overall_reputation < 0.7, "Reputation should degrade"
    print(f"✓ Reputation degraded to {profile2.overall_reputation:.3f}")


def test_reputation_levels():
    """Test categorical reputation levels"""
    print("\n=== Test: Reputation Levels ===")

    abr = AdaptiveByzantineResistance()

    test_cases = [
        (0.96, ReputationLevel.ELITE),
        (0.92, ReputationLevel.TRUSTED),
        (0.75, ReputationLevel.NORMAL),
        (0.55, ReputationLevel.WARNING),
        (0.35, ReputationLevel.CRITICAL),
        (0.2, ReputationLevel.BLACKLISTED)
    ]

    for reputation, expected_level in test_cases:
        node_id = int(reputation * 1000)  # Unique ID
        profile = abr.get_or_create_profile(node_id)
        profile.overall_reputation = reputation

        level = abr.get_reputation_level(node_id)
        assert level == expected_level, f"Reputation {reputation} should be {expected_level}"
        print(f"✓ {reputation:.2f} → {level.name}")


def test_dynamic_thresholds():
    """Test dynamic threshold adjustment"""
    print("\n=== Test: Dynamic Thresholds ===")

    manager = DynamicThresholdManager(base_threshold=0.5)
    initial_threshold = manager.current_threshold

    # Simulate high attack rate
    for i in range(100):
        is_byzantine = (i % 10) < 4  # 40% Byzantine
        manager.update_network_state(validation_passed=not is_byzantine)

    assert manager.current_threshold > initial_threshold, "Threshold should increase under attack"
    print(f"✓ Threshold increased: {initial_threshold:.3f} → {manager.current_threshold:.3f}")

    # Test safe conditions
    manager2 = DynamicThresholdManager(base_threshold=0.6)
    initial2 = manager2.current_threshold

    for i in range(100):
        is_byzantine = (i % 100) < 2  # 2% Byzantine
        manager2.update_network_state(validation_passed=not is_byzantine)

    assert manager2.current_threshold < initial2, "Threshold should decrease when safe"
    print(f"✓ Threshold decreased: {initial2:.3f} → {manager2.current_threshold:.3f}")


def test_byzantine_resistant_aggregation():
    """Test Byzantine-resistant aggregation algorithms"""
    print("\n=== Test: Byzantine-Resistant Aggregation ===")

    aggregator = ByzantineResistantAggregator()

    # Create honest and Byzantine gradients
    honest_gradients = [np.ones(10) + np.random.randn(10) * 0.1 for _ in range(7)]
    byzantine_gradients = [np.ones(10) * 10 + np.random.randn(10) for _ in range(3)]

    all_gradients = honest_gradients + byzantine_gradients
    reputations = [0.9] * 7 + [0.3] * 3

    # Test Krum
    result_krum = aggregator.krum(all_gradients, reputations, num_byzantine=3)
    assert np.allclose(result_krum, np.ones(10), atol=0.5), "Krum should filter Byzantine"
    print(f"✓ Krum: mean={np.mean(result_krum):.3f} (expected ~1.0)")

    # Test Trimmed Mean
    result_trimmed = aggregator.trimmed_mean(all_gradients, reputations, trim_ratio=0.2)
    assert np.allclose(result_trimmed, np.ones(10), atol=0.5), "Trimmed mean should filter outliers"
    print(f"✓ Trimmed Mean: mean={np.mean(result_trimmed):.3f} (expected ~1.0)")

    # Test Coordinate Median
    result_median = aggregator.coordinate_median(all_gradients, reputations)
    assert np.allclose(result_median, np.ones(10), atol=0.5), "Median should be robust"
    print(f"✓ Coordinate Median: mean={np.mean(result_median):.3f} (expected ~1.0)")

    # Test Weighted Average (baseline)
    honest_only = honest_gradients[:5]
    honest_reps = [0.9] * 5
    result_weighted = aggregator.weighted_average(honest_only, honest_reps)
    assert np.allclose(result_weighted, np.ones(10), atol=0.3), "Weighted average should work for honest"
    print(f"✓ Weighted Average: mean={np.mean(result_weighted):.3f} (expected ~1.0)")


async def test_end_to_end():
    """Test complete integrated system"""
    print("\n=== Test: End-to-End Integration ===")

    abr = AdaptiveByzantineResistance()

    # Create nodes
    honest_nodes = [1, 2, 3, 4, 5]
    byzantine_nodes = [6, 7]

    # Build reputation
    for round_num in range(10):
        # Honest nodes
        for node_id in honest_nodes:
            gradient = np.random.randn(100) * 0.1
            abr.update_reputation(
                node_id=node_id,
                gradient=gradient,
                validation_passed=True,
                pogq_score=0.9,
                anomaly_score=0.1
            )

        # Byzantine nodes
        for node_id in byzantine_nodes:
            gradient = np.random.randn(100) * 10
            abr.update_reputation(
                node_id=node_id,
                gradient=gradient,
                validation_passed=False,
                pogq_score=0.2,
                anomaly_score=0.9
            )

    # Verify reputations
    for node_id in honest_nodes:
        level = abr.get_reputation_level(node_id)
        assert level in [ReputationLevel.TRUSTED, ReputationLevel.ELITE], \
            f"Honest node {node_id} should be TRUSTED/ELITE, got {level}"
    print(f"✓ Honest nodes have high reputation")

    for node_id in byzantine_nodes:
        level = abr.get_reputation_level(node_id)
        assert level in [ReputationLevel.BLACKLISTED, ReputationLevel.CRITICAL], \
            f"Byzantine node {node_id} should be BLACKLISTED/CRITICAL, got {level}"
    print(f"✓ Byzantine nodes have low reputation")

    # Aggregate gradients
    gradients = {
        **{nid: np.random.randn(100) * 0.1 for nid in honest_nodes},
        **{nid: np.random.randn(100) * 10 for nid in byzantine_nodes}
    }

    result = await abr.aggregate_gradients(gradients, algorithm="krum")

    # Result should not be influenced by Byzantine gradients
    assert np.abs(np.mean(result)) < 1.0, "Result should not be influenced by outliers"
    print(f"✓ Aggregation robust: mean={np.mean(result):.3f}")

    # Test statistics
    stats = abr.get_statistics()
    assert stats["total_nodes"] == 7, "Should have 7 nodes"
    print(f"✓ Statistics: {stats['total_nodes']} nodes, avg reputation={stats['avg_reputation']:.3f}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("ADAPTIVE BYZANTINE RESISTANCE - MANUAL TESTS")
    print("=" * 60)

    try:
        test_multi_level_reputation()
        test_reputation_levels()
        test_dynamic_thresholds()
        test_byzantine_resistant_aggregation()

        # Run async test
        asyncio.run(test_end_to_end())

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
