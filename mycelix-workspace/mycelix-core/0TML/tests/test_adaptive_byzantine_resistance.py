# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Adaptive Byzantine Resistance (Phase 5 Enhancement 2)
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import asyncio

from zerotrustml.experimental.adaptive_byzantine_resistance import (
    AdaptiveByzantineResistance,
    ReputationLevel,
    DynamicThresholdManager,
    ReputationRecoveryManager,
    ByzantineResistantAggregator
)


@pytest.fixture
def abr_system():
    """Create fresh ABR system for each test"""
    return AdaptiveByzantineResistance()


class TestMultiLevelReputation:
    """Test multi-level reputation tracking"""

    def test_initial_reputation(self, abr_system):
        """Test that new nodes start with default reputation"""
        profile = abr_system.get_or_create_profile(node_id=1)

        assert profile.overall_reputation == 0.7
        assert profile.gradient_reputation == 0.7
        assert profile.network_reputation == 0.7
        assert profile.temporal_reputation == 0.7
        assert profile.total_gradients == 0

    def test_reputation_improves_with_valid_gradients(self, abr_system):
        """Test reputation increases with valid gradients"""
        node_id = 1
        gradient = np.random.randn(100)

        # Submit 10 valid gradients
        for _ in range(10):
            abr_system.update_reputation(
                node_id=node_id,
                gradient=gradient,
                validation_passed=True,
                pogq_score=0.8,
                anomaly_score=0.0,
                contribution_value=0.1
            )

        profile = abr_system.get_or_create_profile(node_id)
        assert profile.overall_reputation > 0.7  # Improved from default
        assert profile.gradient_reputation == 1.0  # 10/10 valid
        assert profile.consecutive_successes == 10

    def test_reputation_degrades_with_invalid_gradients(self, abr_system):
        """Test reputation decreases with invalid gradients"""
        node_id = 2
        gradient = np.random.randn(100)

        # Submit 10 invalid gradients
        for _ in range(10):
            abr_system.update_reputation(
                node_id=node_id,
                gradient=gradient,
                validation_passed=False,
                pogq_score=0.2,
                anomaly_score=0.9
            )

        profile = abr_system.get_or_create_profile(node_id)
        assert profile.overall_reputation < 0.7  # Degraded from default
        assert profile.gradient_reputation == 0.0  # 0/10 valid
        assert profile.consecutive_failures == 10

    def test_reputation_levels(self, abr_system):
        """Test categorical reputation levels"""
        # Elite node (0.95+)
        abr_system.node_profiles[1] = abr_system.get_or_create_profile(1)
        abr_system.node_profiles[1].overall_reputation = 0.96
        assert abr_system.get_reputation_level(1) == ReputationLevel.ELITE

        # Trusted node (0.9+)
        abr_system.node_profiles[2] = abr_system.get_or_create_profile(2)
        abr_system.node_profiles[2].overall_reputation = 0.92
        assert abr_system.get_reputation_level(2) == ReputationLevel.TRUSTED

        # Normal node (0.7+)
        abr_system.node_profiles[3] = abr_system.get_or_create_profile(3)
        abr_system.node_profiles[3].overall_reputation = 0.75
        assert abr_system.get_reputation_level(3) == ReputationLevel.NORMAL

        # Warning node (0.5+)
        abr_system.node_profiles[4] = abr_system.get_or_create_profile(4)
        abr_system.node_profiles[4].overall_reputation = 0.55
        assert abr_system.get_reputation_level(4) == ReputationLevel.WARNING

        # Critical node (0.3+)
        abr_system.node_profiles[5] = abr_system.get_or_create_profile(5)
        abr_system.node_profiles[5].overall_reputation = 0.35
        assert abr_system.get_reputation_level(5) == ReputationLevel.CRITICAL

        # Blacklisted node (<0.3)
        abr_system.node_profiles[6] = abr_system.get_or_create_profile(6)
        abr_system.node_profiles[6].overall_reputation = 0.2
        assert abr_system.get_reputation_level(6) == ReputationLevel.BLACKLISTED


class TestDynamicThresholds:
    """Test dynamic threshold adjustment"""

    def test_threshold_increases_under_attack(self):
        """Test threshold increases when attack rate is high"""
        manager = DynamicThresholdManager(base_threshold=0.5)
        initial_threshold = manager.current_threshold

        # Simulate high attack rate (40% Byzantine)
        for i in range(100):
            is_byzantine = (i % 10) < 4  # 40% Byzantine
            manager.update_network_state(validation_passed=not is_byzantine)

        # Threshold should increase
        assert manager.current_threshold > initial_threshold

    def test_threshold_decreases_when_safe(self):
        """Test threshold decreases in safe conditions"""
        manager = DynamicThresholdManager(base_threshold=0.6)
        initial_threshold = manager.current_threshold

        # Simulate low attack rate (2% Byzantine)
        for i in range(100):
            is_byzantine = (i % 100) < 2  # 2% Byzantine
            manager.update_network_state(validation_passed=not is_byzantine)

        # Threshold should decrease
        assert manager.current_threshold < initial_threshold

    def test_threshold_respects_bounds(self):
        """Test threshold stays within min/max bounds"""
        manager = DynamicThresholdManager(
            base_threshold=0.5,
            min_threshold=0.3,
            max_threshold=0.8
        )

        # Extreme attack scenario
        for _ in range(1000):
            manager.update_network_state(validation_passed=False)

        assert manager.current_threshold <= manager.max_threshold

        # Extremely safe scenario
        manager2 = DynamicThresholdManager(
            base_threshold=0.5,
            min_threshold=0.3,
            max_threshold=0.8
        )
        for _ in range(1000):
            manager2.update_network_state(validation_passed=True)

        assert manager2.current_threshold >= manager2.min_threshold


class TestReputationRecovery:
    """Test reputation recovery mechanisms"""

    def test_recovery_eligibility(self, abr_system):
        """Test recovery eligibility checks"""
        recovery = ReputationRecoveryManager()
        profile = abr_system.get_or_create_profile(1)

        # New node not eligible (reputation too high)
        assert not recovery.can_attempt_recovery(profile)

        # Degrade reputation to critical
        profile.overall_reputation = 0.25
        assert recovery.can_attempt_recovery(profile)

        # After max attempts, not eligible
        profile.recovery_attempts = 3
        assert not recovery.can_attempt_recovery(profile)

    def test_recovery_process(self, abr_system):
        """Test complete recovery process"""
        recovery = ReputationRecoveryManager(recovery_window=20, recovery_threshold=0.95)
        profile = abr_system.get_or_create_profile(1)

        # Degrade reputation
        profile.overall_reputation = 0.25

        # Start recovery
        recovery.start_recovery(profile)
        assert profile.recovery_attempts == 1
        assert profile.last_recovery is not None

        # Submit good gradients during recovery
        gradient = np.random.randn(100)
        for _ in range(20):
            abr_system.update_reputation(
                node_id=1,
                gradient=gradient,
                validation_passed=True,
                pogq_score=0.9,
                anomaly_score=0.0
            )

        # Evaluate recovery
        should_recover, new_reputation = recovery.evaluate_recovery(profile)
        assert should_recover  # 100% success rate > 95% threshold
        assert new_reputation >= profile.overall_reputation


class TestByzantineResistantAggregation:
    """Test Byzantine-resistant aggregation algorithms"""

    def create_test_gradients(self, num_honest: int, num_byzantine: int):
        """Create mix of honest and Byzantine gradients"""
        # Honest gradients cluster around [1, 1, ..., 1]
        honest_gradients = [
            np.ones(10) + np.random.randn(10) * 0.1
            for _ in range(num_honest)
        ]

        # Byzantine gradients are outliers
        byzantine_gradients = [
            np.ones(10) * 10 + np.random.randn(10)  # 10x larger
            for _ in range(num_byzantine)
        ]

        all_gradients = honest_gradients + byzantine_gradients

        # Honest nodes have high reputation, Byzantine low
        reputations = [0.9] * num_honest + [0.3] * num_byzantine

        return all_gradients, reputations, np.ones(10)  # Expected result

    def test_krum_aggregation(self):
        """Test Krum algorithm filters Byzantine gradients"""
        aggregator = ByzantineResistantAggregator()
        gradients, reputations, expected = self.create_test_gradients(
            num_honest=7, num_byzantine=3
        )

        result = aggregator.krum(gradients, reputations, num_byzantine=3)

        # Result should be close to honest gradients (around 1.0)
        assert np.allclose(result, expected, atol=0.5)

    def test_trimmed_mean_aggregation(self):
        """Test trimmed mean filters outliers"""
        aggregator = ByzantineResistantAggregator()
        gradients, reputations, expected = self.create_test_gradients(
            num_honest=8, num_byzantine=2
        )

        result = aggregator.trimmed_mean(gradients, reputations, trim_ratio=0.2)

        # Result should be close to honest gradients
        assert np.allclose(result, expected, atol=0.5)

    def test_coordinate_median_aggregation(self):
        """Test coordinate-wise median"""
        aggregator = ByzantineResistantAggregator()
        gradients, reputations, expected = self.create_test_gradients(
            num_honest=7, num_byzantine=3
        )

        result = aggregator.coordinate_median(gradients, reputations)

        # Median should be robust to outliers
        assert np.allclose(result, expected, atol=0.5)

    def test_weighted_average_baseline(self):
        """Test weighted average (baseline, not Byzantine-resistant)"""
        aggregator = ByzantineResistantAggregator()

        # All honest gradients
        gradients = [np.ones(10) + np.random.randn(10) * 0.1 for _ in range(5)]
        reputations = [0.9] * 5

        result = aggregator.weighted_average(gradients, reputations)

        # Should be close to 1.0 for honest gradients
        assert np.allclose(result, np.ones(10), atol=0.3)


@pytest.mark.asyncio
class TestIntegratedAdaptiveByzantine:
    """Test complete integrated system"""

    async def test_end_to_end_aggregation(self, abr_system):
        """Test complete workflow with reputation tracking and aggregation"""
        # Create nodes with different behaviors
        honest_nodes = [1, 2, 3, 4, 5]  # 5 honest
        byzantine_nodes = [6, 7]  # 2 Byzantine

        gradient_shape = (100,)

        # Build reputation history
        for round_num in range(10):
            # Honest nodes submit good gradients
            for node_id in honest_nodes:
                gradient = np.random.randn(*gradient_shape) * 0.1
                abr_system.update_reputation(
                    node_id=node_id,
                    gradient=gradient,
                    validation_passed=True,
                    pogq_score=0.9,
                    anomaly_score=0.1
                )

            # Byzantine nodes submit bad gradients
            for node_id in byzantine_nodes:
                gradient = np.random.randn(*gradient_shape) * 10  # Large outliers
                abr_system.update_reputation(
                    node_id=node_id,
                    gradient=gradient,
                    validation_passed=False,
                    pogq_score=0.2,
                    anomaly_score=0.9
                )

        # Verify reputations
        for node_id in honest_nodes:
            level = abr_system.get_reputation_level(node_id)
            assert level in [ReputationLevel.TRUSTED, ReputationLevel.ELITE]

        for node_id in byzantine_nodes:
            level = abr_system.get_reputation_level(node_id)
            assert level in [ReputationLevel.BLACKLISTED, ReputationLevel.CRITICAL]

        # Aggregate gradients
        gradients = {
            **{nid: np.random.randn(*gradient_shape) * 0.1 for nid in honest_nodes},
            **{nid: np.random.randn(*gradient_shape) * 10 for nid in byzantine_nodes}
        }

        result = await abr_system.aggregate_gradients(gradients, algorithm="krum")
        assert result.shape == gradient_shape

        # Result should not be influenced by Byzantine gradients
        # (should be small magnitude like honest ones)
        assert np.abs(np.mean(result)) < 1.0  # Not influenced by outliers

    async def test_statistics_reporting(self, abr_system):
        """Test comprehensive statistics"""
        # Create diverse node population
        for node_id in range(1, 11):
            gradient = np.random.randn(50)
            success_rate = 0.5 + (node_id / 20)  # Varying success rates

            for _ in range(20):
                validation_passed = np.random.random() < success_rate
                abr_system.update_reputation(
                    node_id=node_id,
                    gradient=gradient,
                    validation_passed=validation_passed,
                    pogq_score=0.7 if validation_passed else 0.3,
                    anomaly_score=0.1 if validation_passed else 0.8
                )

        stats = abr_system.get_statistics()

        assert stats["total_nodes"] == 10
        assert 0.0 <= stats["avg_reputation"] <= 1.0
        assert 0.0 <= stats["median_reputation"] <= 1.0
        assert 0.0 <= stats["byzantine_detection_rate"] <= 1.0
        assert "reputation_distribution" in stats
        assert sum(stats["reputation_distribution"].values()) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
