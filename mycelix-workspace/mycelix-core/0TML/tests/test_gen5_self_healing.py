# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Gen 5 Layer 7: Self-Healing Mechanism

Test Coverage:
- BFT estimation (4 tests)
- Gradient quarantine (5 tests)
- Healing activation/deactivation (4 tests)
- Adaptive thresholds (2 tests)
- Integration scenarios (5 tests)
- Statistics tracking (2 tests)

Total: 22 tests
"""

import pytest
import numpy as np
from gen5.self_healing import (
    BFTEstimator,
    GradientQuarantine,
    SelfHealingMechanism,
    HealingConfig,
)


class TestBFTEstimator:
    """Test suite for BFT estimation."""

    def test_bft_estimation_all_honest(self):
        """Test BFT = 0% when all scores are honest."""
        estimator = BFTEstimator(threshold=0.5)

        # All honest scores (> 0.5)
        scores = [0.80, 0.85, 0.90, 0.75, 0.88]
        estimator.update(scores)

        bft = estimator.estimate_bft()
        assert bft == 0.0

    def test_bft_estimation_all_byzantine(self):
        """Test BFT = 100% when all scores are Byzantine."""
        estimator = BFTEstimator(threshold=0.5)

        # All Byzantine scores (< 0.5)
        scores = [0.10, 0.15, 0.20, 0.25, 0.12]
        estimator.update(scores)

        bft = estimator.estimate_bft()
        assert bft == 1.0

    def test_bft_estimation_mixed(self):
        """Test correct BFT estimation for mixed scores."""
        estimator = BFTEstimator(threshold=0.5)

        # 40% Byzantine: 2 Byzantine, 3 honest
        scores = [0.15, 0.20, 0.80, 0.85, 0.90]
        estimator.update(scores)

        bft = estimator.estimate_bft()
        assert abs(bft - 0.40) < 0.01

    def test_is_under_attack(self):
        """Test attack detection."""
        estimator = BFTEstimator(threshold=0.5)

        # Below tolerance (40%)
        scores = [0.15, 0.20, 0.80, 0.85, 0.90]
        estimator.update(scores)
        assert not estimator.is_under_attack(tolerance=0.45)

        # Above tolerance (60%)
        scores = [0.15, 0.20, 0.10, 0.80, 0.85]
        estimator.update(scores)
        assert estimator.is_under_attack(tolerance=0.45)


class TestGradientQuarantine:
    """Test suite for gradient quarantine."""

    def test_quarantine_basic(self):
        """Test basic quarantine functionality."""
        quarantine = GradientQuarantine(quarantine_weight=0.1)

        # Quarantine gradient 0
        quarantine.quarantine_gradient(idx=0, round_num=0, original_weight=1.0)

        assert 0 in quarantine.quarantined_indices
        assert quarantine.get_weight(0) == 0.1

    def test_quarantine_recovery(self):
        """Test gradual weight recovery."""
        quarantine = GradientQuarantine(
            quarantine_weight=0.1, recovery_rate=0.05
        )

        # Quarantine gradient
        quarantine.quarantine_gradient(idx=0, round_num=0, original_weight=1.0)
        assert quarantine.get_weight(0) == 0.1

        # Apply recovery (5% increase per round)
        quarantine.apply_recovery(round_num=1)
        assert abs(quarantine.get_weight(0) - 0.105) < 0.001

        quarantine.apply_recovery(round_num=2)
        assert abs(quarantine.get_weight(0) - 0.11025) < 0.001

    def test_quarantine_release(self):
        """Test full release after reaching original weight."""
        quarantine = GradientQuarantine(
            quarantine_weight=0.9, recovery_rate=0.05
        )

        # Quarantine at high weight (near original)
        quarantine.quarantine_gradient(idx=0, round_num=0, original_weight=1.0)

        # Recovery should reach 1.0 quickly
        quarantine.apply_recovery(round_num=1)
        quarantine.apply_recovery(round_num=2)
        quarantine.apply_recovery(round_num=3)

        # Should be released
        assert 0 not in quarantine.quarantined_indices
        assert quarantine.get_weight(0) == 1.0

    def test_quarantine_weight_floor(self):
        """Test minimum weight is maintained."""
        quarantine = GradientQuarantine(quarantine_weight=0.1)

        quarantine.quarantine_gradient(idx=0, round_num=0, original_weight=1.0)

        # Weight should not go below quarantine_weight
        assert quarantine.get_weight(0) == 0.1
        assert quarantine.get_weight(0) >= 0.1

    def test_quarantine_duration(self):
        """Test quarantine duration tracking."""
        quarantine = GradientQuarantine()

        quarantine.quarantine_gradient(idx=0, round_num=10)

        duration = quarantine.get_quarantine_duration(0, current_round=15)
        assert duration == 5


class TestHealingActivation:
    """Test suite for healing activation and deactivation."""

    def test_healing_activation(self):
        """Test healing activates when BFT > tolerance."""
        healer = SelfHealingMechanism(
            HealingConfig(bft_tolerance=0.45, healing_threshold=0.40)
        )

        # Normal scores (40% Byzantine - below tolerance)
        scores = [0.15, 0.20, 0.80, 0.85, 0.90]
        _, details = healer.process_batch(scores, round_num=0)
        assert not details["is_healing"]

        # Attack scores (80% Byzantine - above tolerance)
        scores = [0.10, 0.15, 0.12, 0.18, 0.85]
        _, details = healer.process_batch(scores, round_num=1)
        assert details["is_healing"]

    def test_healing_deactivation(self):
        """Test healing deactivates when BFT < threshold for sustained period."""
        healer = SelfHealingMechanism(
            HealingConfig(
                bft_tolerance=0.45,
                healing_threshold=0.40,
                sustained_rounds=3,
            )
        )

        # Activate healing
        scores_attack = [0.10, 0.15, 0.12, 0.18, 0.20]  # 100% Byzantine
        healer.process_batch(scores_attack, round_num=0)
        assert healer.is_healing

        # Recovery (but need sustained low BFT)
        # Note: Window dilution means we need more rounds for BFT to drop
        scores_recovery = [0.80, 0.85, 0.90, 0.88, 0.82]  # 0% Byzantine

        healer.process_batch(scores_recovery, round_num=1)
        assert healer.is_healing  # Still healing (BFT ~50% due to window)

        healer.process_batch(scores_recovery, round_num=2)
        assert healer.is_healing  # Still healing (BFT ~33%, first low round)

        healer.process_batch(scores_recovery, round_num=3)
        assert healer.is_healing  # Still healing (BFT ~25%, second low round)

        healer.process_batch(scores_recovery, round_num=4)
        assert not healer.is_healing  # Deactivated (3 sustained low rounds: 2,3,4)

    def test_healing_statistics(self):
        """Test healing statistics tracking."""
        healer = SelfHealingMechanism()

        # Normal operation
        scores = [0.80, 0.85, 0.90, 0.75, 0.88]
        healer.process_batch(scores, round_num=0)

        stats = healer.get_stats()
        assert stats["healing_activations"] == 0
        assert stats["total_healing_rounds"] == 0

        # Trigger healing
        scores_attack = [0.10, 0.15, 0.12, 0.18, 0.20]
        healer.process_batch(scores_attack, round_num=1)
        healer.process_batch(scores_attack, round_num=2)

        stats = healer.get_stats()
        assert stats["healing_activations"] == 1
        assert stats["total_healing_rounds"] == 2

    def test_multiple_healing_cycles(self):
        """Test system can activate/deactivate healing multiple times."""
        healer = SelfHealingMechanism(
            HealingConfig(sustained_rounds=2)
        )

        scores_attack = [0.10, 0.15, 0.12]  # Byzantine
        scores_normal = [0.80, 0.85, 0.90]  # Honest

        # Cycle 1: attack → recovery (need extra rounds for window clearance)
        healer.process_batch(scores_attack, round_num=0)
        assert healer.is_healing

        healer.process_batch(scores_normal, round_num=1)
        healer.process_batch(scores_normal, round_num=2)
        healer.process_batch(scores_normal, round_num=3)
        assert not healer.is_healing

        # Cycle 2: attack → recovery (need more attack rounds to overcome window dilution)
        healer.process_batch(scores_attack, round_num=4)
        healer.process_batch(scores_attack, round_num=5)
        healer.process_batch(scores_attack, round_num=6)
        assert healer.is_healing

        healer.process_batch(scores_normal, round_num=7)
        healer.process_batch(scores_normal, round_num=8)
        healer.process_batch(scores_normal, round_num=9)
        healer.process_batch(scores_normal, round_num=10)
        healer.process_batch(scores_normal, round_num=11)
        assert not healer.is_healing

        stats = healer.get_stats()
        assert stats["healing_activations"] == 2


class TestAdaptiveThresholds:
    """Test suite for adaptive threshold adjustment."""

    def test_adaptive_threshold_adjustment(self):
        """Test threshold adjusts during healing."""
        healer = SelfHealingMechanism(
            HealingConfig(bft_tolerance=0.45, score_threshold=0.5)
        )

        # Normal operation: threshold = 0.5
        assert healer.adaptive_threshold == 0.5

        # Trigger healing with high BFT
        scores_attack = [0.10, 0.15, 0.12, 0.18, 0.20]  # 100% Byzantine
        _, details = healer.process_batch(scores_attack, round_num=0)

        # Threshold should decrease (more aggressive)
        assert details["adaptive_threshold"] < 0.5
        assert healer.adaptive_threshold < 0.5

    def test_threshold_resets_after_healing(self):
        """Test threshold returns to normal after healing."""
        healer = SelfHealingMechanism(
            HealingConfig(sustained_rounds=2)
        )

        original_threshold = healer.adaptive_threshold

        # Activate healing
        scores_attack = [0.10, 0.15, 0.12]
        healer.process_batch(scores_attack, round_num=0)
        assert healer.adaptive_threshold != original_threshold

        # Deactivate healing (need enough rounds for window clearance)
        scores_normal = [0.80, 0.85, 0.90]
        healer.process_batch(scores_normal, round_num=1)
        healer.process_batch(scores_normal, round_num=2)
        healer.process_batch(scores_normal, round_num=3)

        # Threshold should reset
        assert healer.adaptive_threshold == original_threshold


class TestIntegrationScenarios:
    """Integration tests with realistic attack scenarios."""

    def test_end_to_end_normal_operation(self):
        """Test normal operation without healing."""
        healer = SelfHealingMechanism()

        # Process 10 rounds of normal scores
        for round_num in range(10):
            scores = [np.random.uniform(0.7, 0.95) for _ in range(20)]
            decisions, details = healer.process_batch(scores, round_num)

            assert not details["is_healing"]
            assert all(d == "honest" for d in decisions)

        stats = healer.get_stats()
        assert stats["healing_activations"] == 0

    def test_end_to_end_healing_activation(self):
        """Test full healing protocol activation."""
        healer = SelfHealingMechanism()

        # Round 0-5: Normal (40% Byzantine)
        for round_num in range(5):
            scores = [0.15, 0.20, 0.80, 0.85, 0.90]
            _, details = healer.process_batch(scores, round_num)
            assert not details["is_healing"]

        # Round 6-10: Attack (80% Byzantine)
        for round_num in range(5, 10):
            scores = [0.10, 0.15, 0.12, 0.18, 0.85]
            _, details = healer.process_batch(scores, round_num)
            assert details["is_healing"]

        stats = healer.get_stats()
        assert stats["healing_activations"] == 1
        assert stats["total_healing_rounds"] == 5

    def test_gradual_attack_recovery(self):
        """Test attack → heal → return to normal cycle."""
        healer = SelfHealingMechanism(
            HealingConfig(sustained_rounds=3)
        )

        # Phase 1: Normal (rounds 0-2) - keep short to avoid window dilution
        for round_num in range(3):
            scores = [0.80, 0.85, 0.90, 0.75, 0.88]
            healer.process_batch(scores, round_num)

        assert not healer.is_healing

        # Phase 2: Attack (rounds 3-7) - sustained attack to exceed tolerance
        for round_num in range(3, 8):
            scores = [0.10, 0.15, 0.12, 0.18, 0.20]  # 100% Byzantine
            healer.process_batch(scores, round_num)

        assert healer.is_healing

        # Phase 3: Recovery (rounds 8-14) - need one more round for deactivation
        for round_num in range(8, 15):
            scores = [0.80, 0.85, 0.90, 0.75, 0.88]
            healer.process_batch(scores, round_num)

        # Should deactivate after 3 sustained low-BFT rounds
        assert not healer.is_healing

    def test_sustained_high_bft(self):
        """Test continuous healing under persistent attack."""
        healer = SelfHealingMechanism()

        # Sustained attack for 20 rounds
        for round_num in range(20):
            scores = [0.10, 0.15, 0.12, 0.18, 0.20]  # 100% Byzantine
            _, details = healer.process_batch(scores, round_num)

            if round_num > 0:  # After first round
                assert details["is_healing"]

        stats = healer.get_stats()
        assert stats["healing_activations"] == 1
        assert stats["total_healing_rounds"] == 20  # All rounds (healing activates on round 0)

    def test_quarantine_effectiveness(self):
        """Test that quarantine actually quarantines Byzantine gradients."""
        healer = SelfHealingMechanism()

        # Trigger healing
        scores = [0.10, 0.15, 0.12, 0.18, 0.20]  # All Byzantine
        decisions, details = healer.process_batch(scores, round_num=0)

        # All should be classified as Byzantine
        assert all(d == "byzantine" for d in decisions)

        # All should be quarantined
        quarantine_stats = details["quarantine_stats"]
        assert quarantine_stats["quarantined_count"] == 5


class TestStatistics:
    """Test suite for statistics tracking."""

    def test_stats_tracking(self):
        """Test all statistics are tracked correctly."""
        healer = SelfHealingMechanism()

        # Process attack rounds (start with attack to get high BFT)
        scores_attack = [0.10, 0.15, 0.12]

        healer.process_batch(scores_attack, round_num=0)
        healer.process_batch(scores_attack, round_num=1)
        healer.process_batch(scores_attack, round_num=2)

        stats = healer.get_stats()

        assert stats["total_rounds"] == 3
        assert stats["max_bft_observed"] == 1.0  # 100% Byzantine in all rounds
        assert stats["is_currently_healing"] is True
        assert stats["healing_activations"] >= 1
        assert stats["gradients_quarantined"] >= 3

    def test_reset(self):
        """Test reset functionality."""
        healer = SelfHealingMechanism()

        # Process some rounds
        scores = [0.10, 0.15, 0.12]
        healer.process_batch(scores, round_num=0)
        healer.process_batch(scores, round_num=1)

        # Reset
        healer.reset()

        stats = healer.get_stats()
        assert stats["total_rounds"] == 0
        assert stats["healing_activations"] == 0
        assert stats["max_bft_observed"] == 0.0
        assert not healer.is_healing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
