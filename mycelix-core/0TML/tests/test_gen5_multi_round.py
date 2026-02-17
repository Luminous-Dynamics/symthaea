"""
Tests for Gen 5 Layer 6: Multi-Round Temporal Detection
========================================================

Tests:
1. Sleeper Detection (4 tests)
   - Honest → Byzantine transition
   - No false positive on consistent agent
   - Byzantine → honest transition
   - CUSUM accuracy

2. Coordination Detection (3 tests)
   - Synchronized attacks
   - Non-coordinated agents
   - Lag tolerance

3. Reputation Tracking (3 tests)
   - Bayesian update correctness
   - Reputation convergence
   - New agent initialization

4. Temporal Anomaly (2 tests)
   - Z-score anomaly detection
   - No false positive on normal variance

5. Integration (8 tests)
   - History tracking
   - Multi-round pipeline
   - Real attack scenarios

Author: Luminous Dynamics
Date: November 12, 2025
"""

import pytest
import numpy as np

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gen5.multi_round import (
    MultiRoundDetector,
    MultiRoundConfig,
    ReputationTracker,
    TemporalAnomaly,
)


class TestSleeperDetection:
    """Test suite for sleeper agent detection."""

    def test_detect_honest_to_byzantine_transition(self):
        """Test detection of honest → Byzantine sleeper agent."""
        detector = MultiRoundDetector(
            MultiRoundConfig(
                window_size=100,
                sleeper_threshold=0.3,
                cusum_threshold=5.0,
            )
        )

        agent_id = "agent_sleeper"

        # Rounds 1-50: Honest (score ≈ 0.85)
        for round_idx in range(50):
            score = np.random.uniform(0.80, 0.90)
            detector.update_agent_history(agent_id, round_idx, score)

        # Rounds 51-100: Byzantine (score ≈ 0.15)
        for round_idx in range(50, 100):
            score = np.random.uniform(0.10, 0.20)
            detector.update_agent_history(agent_id, round_idx, score)

        # Detect sleeper
        is_sleeper, magnitude, change_point = detector.detect_sleeper_agent(
            agent_id
        )

        # Should detect sleeper
        assert is_sleeper
        assert magnitude > 0.5  # Large change
        assert 40 <= change_point <= 60  # Change around round 50

    def test_no_false_positive_consistent_agent(self):
        """Test no false positive on consistently honest agent."""
        detector = MultiRoundDetector()

        agent_id = "agent_honest"

        # 100 rounds: Consistently honest (score ≈ 0.85 ± 0.05)
        for round_idx in range(100):
            score = np.random.uniform(0.80, 0.90)
            detector.update_agent_history(agent_id, round_idx, score)

        # Should NOT detect sleeper
        is_sleeper, magnitude, change_point = detector.detect_sleeper_agent(
            agent_id
        )

        assert not is_sleeper
        assert magnitude < 0.2  # Small variance

    def test_detect_byzantine_to_honest_transition(self):
        """Test detection of Byzantine → honest transition."""
        detector = MultiRoundDetector(
            MultiRoundConfig(
                window_size=100,  # Need full history for 100-round test
                sleeper_threshold=0.3,
                cusum_threshold=5.0,
            )
        )

        agent_id = "agent_reformed"

        # Rounds 1-50: Byzantine
        for round_idx in range(50):
            score = np.random.uniform(0.10, 0.20)
            detector.update_agent_history(agent_id, round_idx, score)

        # Rounds 51-100: Honest (reformed)
        for round_idx in range(50, 100):
            score = np.random.uniform(0.80, 0.90)
            detector.update_agent_history(agent_id, round_idx, score)

        # Should detect transition
        is_sleeper, magnitude, change_point = detector.detect_sleeper_agent(
            agent_id
        )

        assert is_sleeper
        assert magnitude > 0.5

    def test_cusum_state_tracking(self):
        """Test CUSUM state is correctly tracked."""
        detector = MultiRoundDetector()

        agent_id = "agent_test"

        # Initially, CUSUM should be 0
        assert detector.cusum_state[agent_id] == 0.0

        # Add some consistent scores
        for round_idx in range(20):
            score = 0.85
            detector.update_agent_history(agent_id, round_idx, score)

        # CUSUM should remain low for consistent agent
        assert detector.cusum_state[agent_id] < 2.0


class TestCoordinationDetection:
    """Test suite for coordination detection."""

    def test_detect_synchronized_attacks(self):
        """Test detection of synchronized Byzantine attacks."""
        detector = MultiRoundDetector(
            MultiRoundConfig(coordination_threshold=0.7)
        )

        # Two agents with synchronized attacks
        agent1 = "agent_coord1"
        agent2 = "agent_coord2"

        # Rounds 1-50: Both honest (correlated behavior)
        for round_idx in range(50):
            # Generate correlated scores (same base + small noise)
            base_score = np.random.uniform(0.80, 0.90)
            score1 = base_score + np.random.normal(0, 0.01)
            score2 = base_score + np.random.normal(0, 0.01)
            detector.update_agent_history(agent1, round_idx, score1)
            detector.update_agent_history(agent2, round_idx, score2)

        # Rounds 51-100: Both Byzantine (synchronized, correlated)
        for round_idx in range(50, 100):
            # Generate correlated Byzantine scores
            base_score = np.random.uniform(0.10, 0.20)
            score1 = base_score + np.random.normal(0, 0.01)
            score2 = base_score + np.random.normal(0, 0.01)
            detector.update_agent_history(agent1, round_idx, score1)
            detector.update_agent_history(agent2, round_idx, score2)

        # Detect coordination
        coordinated = detector.detect_coordination([agent1, agent2])

        # Should detect coordination
        assert len(coordinated) == 1
        pair_agent1, pair_agent2, correlation = coordinated[0]
        assert {pair_agent1, pair_agent2} == {agent1, agent2}
        assert correlation > 0.7  # High correlation

    def test_ignore_non_coordinated_agents(self):
        """Test that independent agents are not flagged as coordinated."""
        detector = MultiRoundDetector(
            MultiRoundConfig(coordination_threshold=0.7)
        )

        agent1 = "agent_independent1"
        agent2 = "agent_independent2"

        # Independent random behavior
        np.random.seed(42)
        for round_idx in range(50):
            score1 = np.random.uniform(0.0, 1.0)
            score2 = np.random.uniform(0.0, 1.0)
            detector.update_agent_history(agent1, round_idx, score1)
            detector.update_agent_history(agent2, round_idx, score2)

        # Should NOT detect coordination
        coordinated = detector.detect_coordination([agent1, agent2])

        assert len(coordinated) == 0

    def test_handle_lag_tolerance(self):
        """Test coordination detection with time lag."""
        detector = MultiRoundDetector(
            MultiRoundConfig(
                coordination_threshold=0.7, lag_tolerance=2
            )
        )

        agent1 = "agent_lag1"
        agent2 = "agent_lag2"

        # Similar pattern but slightly offset
        # (Note: Current implementation doesn't support lag,
        #  so this tests that we still detect high correlation)
        for round_idx in range(50):
            score = np.sin(round_idx / 10)  # Sinusoidal pattern
            detector.update_agent_history(agent1, round_idx, score)
            detector.update_agent_history(
                agent2, round_idx, score
            )  # Same for now

        coordinated = detector.detect_coordination([agent1, agent2])

        # Should detect high correlation
        assert len(coordinated) > 0


class TestReputationTracking:
    """Test suite for Bayesian reputation tracking."""

    def test_bayesian_update_correctness(self):
        """Test Bayesian reputation updates are mathematically correct."""
        tracker = ReputationTracker(alpha_prior=1.0, beta_prior=1.0)

        agent_id = "agent_test"

        # Initial reputation (Beta(1,1) mean = 0.5)
        assert tracker.get_reputation(agent_id) == 0.5

        # Add honest evidence (score >= 0.5)
        tracker.update_reputation(agent_id, 0.8)

        # Reputation should increase
        # Beta(2, 1) mean = 2/3 ≈ 0.667
        reputation1 = tracker.get_reputation(agent_id)
        assert 0.6 < reputation1 < 0.7

        # Add Byzantine evidence (score < 0.5)
        tracker.update_reputation(agent_id, 0.2)

        # Reputation should decrease slightly
        # Beta(2, 2) mean = 2/4 = 0.5
        reputation2 = tracker.get_reputation(agent_id)
        assert reputation2 == 0.5

    def test_reputation_convergence(self):
        """Test reputation converges for consistent agent."""
        tracker = ReputationTracker()

        agent_id = "agent_consistent"

        # 100 rounds of consistent honest behavior
        for _ in range(100):
            tracker.update_reputation(agent_id, 0.9)

        # Reputation should converge to ~1.0
        reputation = tracker.get_reputation(agent_id)
        assert reputation > 0.98

        # Variance should be low (high confidence)
        variance = tracker.get_reputation_variance(agent_id)
        assert variance < 0.001

    def test_new_agent_initialization(self):
        """Test new agents start with neutral reputation."""
        tracker = ReputationTracker()

        # Unknown agent should have neutral reputation
        reputation = tracker.get_reputation("unknown_agent")
        assert reputation == 0.5

        # Variance should be maximum for unknown
        variance = tracker.get_reputation_variance("unknown_agent")
        assert variance == 0.25  # Beta(1, 1) variance


class TestTemporalAnomaly:
    """Test suite for temporal anomaly detection."""

    def test_z_score_anomaly_detection(self):
        """Test z-score based anomaly detection."""
        detector = MultiRoundDetector(
            MultiRoundConfig(z_threshold=3.0)
        )

        agent_id = "agent_anomaly"

        # Establish baseline: mean=0.85, std≈0.05
        for round_idx in range(50):
            score = np.random.uniform(0.80, 0.90)
            detector.update_agent_history(agent_id, round_idx, score)

        # Add anomalous score (way outside normal range)
        detector.update_agent_history(agent_id, 50, 0.15)

        # Should detect anomaly
        is_anomaly, z_score = detector.detect_temporal_anomaly(agent_id)

        assert is_anomaly
        assert z_score > 3.0  # More than 3 standard deviations

    def test_no_false_positive_normal_variance(self):
        """Test no false positive on normal variance."""
        detector = MultiRoundDetector()

        agent_id = "agent_normal"

        # Normal variance around mean
        for round_idx in range(50):
            score = np.random.normal(0.85, 0.05)
            detector.update_agent_history(agent_id, round_idx, score)

        # Add another normal score
        detector.update_agent_history(agent_id, 50, 0.87)

        # Should NOT detect anomaly
        is_anomaly, z_score = detector.detect_temporal_anomaly(agent_id)

        assert not is_anomaly
        assert z_score < 3.0


class TestIntegration:
    """Integration tests for multi-round detection."""

    def test_history_tracking_over_many_rounds(self):
        """Test history tracking maintains correct window."""
        config = MultiRoundConfig(window_size=50)
        detector = MultiRoundDetector(config)

        agent_id = "agent_long"

        # Track for 100 rounds (more than window)
        for round_idx in range(100):
            score = 0.85
            detector.update_agent_history(agent_id, round_idx, score)

        # History should be limited to window_size
        history_len = len(detector.agent_histories[agent_id])
        assert history_len == 50

        # Should have most recent rounds
        rounds = list(detector.agent_rounds[agent_id])
        assert min(rounds) >= 50  # Oldest round >= 50
        assert max(rounds) == 99  # Newest round = 99

    def test_complete_multi_round_pipeline(self):
        """Test complete multi-round detection pipeline."""
        detector = MultiRoundDetector()

        # Simulate 10 agents over 100 rounds
        n_agents = 10
        n_rounds = 100

        for round_idx in range(n_rounds):
            for agent_idx in range(n_agents):
                agent_id = f"agent_{agent_idx}"

                # Most agents honest
                if agent_idx < 7:
                    score = np.random.uniform(0.80, 0.90)
                else:
                    # 3 Byzantine agents
                    score = np.random.uniform(0.10, 0.20)

                detector.update_agent_history(agent_id, round_idx, score)

        # Get summary statistics
        stats = detector.get_stats()

        assert stats["total_agents"] == 10
        assert stats["total_rounds"] == 99  # 0-indexed
        assert stats["total_updates"] == 1000  # 10 agents × 100 rounds

        # Get reputations
        reputations = detector.get_all_reputations()
        assert len(reputations) == 10

        # Honest agents should have high reputation
        for agent_idx in range(7):
            agent_id = f"agent_{agent_idx}"
            assert reputations[agent_id] > 0.9

        # Byzantine agents should have low reputation
        for agent_idx in range(7, 10):
            agent_id = f"agent_{agent_idx}"
            assert reputations[agent_id] < 0.2

    def test_sleeper_agent_attack_scenario(self):
        """Test realistic sleeper agent attack."""
        detector = MultiRoundDetector(
            MultiRoundConfig(
                window_size=100,  # Need full history for 100-round test
                sleeper_threshold=0.3,
                cusum_threshold=5.0,
            )
        )

        agent_id = "sleeper"

        # Phase 1: Build reputation (rounds 0-49, honest)
        for round_idx in range(50):
            score = np.random.uniform(0.80, 0.90)
            detector.update_agent_history(agent_id, round_idx, score)

        # Reputation after Phase 1
        rep_phase1 = detector.reputation_tracker.get_reputation(agent_id)
        assert rep_phase1 > 0.9  # High reputation built

        # Phase 2: Attack (rounds 50-99, Byzantine)
        for round_idx in range(50, 100):
            score = np.random.uniform(0.10, 0.20)
            detector.update_agent_history(agent_id, round_idx, score)

        # Should detect sleeper
        sleepers = detector.get_sleeper_agents()
        assert len(sleepers) == 1
        assert sleepers[0]["agent_id"] == agent_id

        # Reputation after Phase 2 should drop significantly
        rep_phase2 = detector.reputation_tracker.get_reputation(agent_id)
        assert rep_phase2 <= 0.5  # Reputation damaged (50 honest + 50 Byzantine = 0.5)
        assert rep_phase2 < rep_phase1  # Clear drop from initial reputation

    def test_coordinated_attack_scenario(self):
        """Test realistic coordinated attack."""
        detector = MultiRoundDetector()

        # 5 coordinated attackers + 5 honest agents
        coordinated_agents = [f"attacker_{i}" for i in range(5)]
        honest_agents = [f"honest_{i}" for i in range(5)]

        # All agents honest initially (rounds 0-49)
        for round_idx in range(50):
            for agent_id in coordinated_agents + honest_agents:
                score = np.random.uniform(0.80, 0.90)
                detector.update_agent_history(agent_id, round_idx, score)

        # Coordinated attack (rounds 50-99)
        for round_idx in range(50, 100):
            # Attackers coordinate (same base score + small noise)
            base_attack_score = np.random.uniform(0.10, 0.20)
            for agent_id in coordinated_agents:
                score = base_attack_score + np.random.normal(0, 0.01)
                detector.update_agent_history(agent_id, round_idx, score)

            # Honest agents remain honest (independent)
            for agent_id in honest_agents:
                score = np.random.uniform(0.80, 0.90)
                detector.update_agent_history(agent_id, round_idx, score)

        # Detect coordination among attackers
        coordinated = detector.detect_coordination(coordinated_agents)

        # Should detect multiple coordinated pairs
        # With 5 agents: C(5,2) = 10 possible pairs
        assert len(coordinated) > 0

        # All pairs should have high correlation
        for agent1, agent2, correlation in coordinated:
            assert correlation > 0.7

    def test_gradual_poisoning_scenario(self):
        """Test gradual behavior change (slow drift)."""
        detector = MultiRoundDetector()

        agent_id = "gradual"

        # Gradual drift from honest to Byzantine
        for round_idx in range(100):
            # Score drifts from 0.9 to 0.1
            score = 0.9 - (0.8 * round_idx / 100)
            detector.update_agent_history(agent_id, round_idx, score)

        # May or may not detect as sleeper (gradual change)
        # But reputation should reflect the trend
        reputation = detector.reputation_tracker.get_reputation(agent_id)

        # Should be somewhere in middle (mixed evidence)
        assert 0.3 < reputation < 0.7

    def test_reputation_exploitation_scenario(self):
        """Test reputation exploitation attack."""
        detector = MultiRoundDetector()

        agent_id = "exploiter"

        # Build trust for 80 rounds
        for round_idx in range(80):
            score = np.random.uniform(0.85, 0.95)
            detector.update_agent_history(agent_id, round_idx, score)

        # Reputation should be very high
        rep_before = detector.reputation_tracker.get_reputation(agent_id)
        assert rep_before > 0.98

        # Exploit trust in rounds 80-100
        for round_idx in range(80, 100):
            score = np.random.uniform(0.0, 0.1)
            detector.update_agent_history(agent_id, round_idx, score)

        # Should detect as sleeper
        is_sleeper, _, _ = detector.detect_sleeper_agent(agent_id)
        assert is_sleeper

        # Reputation should drop (but slowly due to prior)
        rep_after = detector.reputation_tracker.get_reputation(agent_id)
        assert rep_after < rep_before
        assert rep_after < 0.9  # Noticeable drop

    def test_mixed_attack_scenario(self):
        """Test mixed scenario: sleepers + coordination."""
        detector = MultiRoundDetector(
            MultiRoundConfig(window_size=100)  # Need full history for 100-round test
        )

        # 2 sleeper agents that coordinate
        sleeper1 = "sleeper1"
        sleeper2 = "sleeper2"
        honest = "honest"

        # Rounds 0-49: All honest
        for round_idx in range(50):
            for agent_id in [sleeper1, sleeper2, honest]:
                score = np.random.uniform(0.80, 0.90)
                detector.update_agent_history(agent_id, round_idx, score)

        # Rounds 50-99: Sleepers attack together (coordinated)
        for round_idx in range(50, 100):
            # Coordinated Byzantine behavior (same base + small noise)
            base_attack_score = np.random.uniform(0.10, 0.20)
            for agent_id in [sleeper1, sleeper2]:
                score = base_attack_score + np.random.normal(0, 0.01)
                detector.update_agent_history(agent_id, round_idx, score)

            # Honest remains honest
            score = np.random.uniform(0.80, 0.90)
            detector.update_agent_history(honest, round_idx, score)

        # Should detect both sleepers
        sleepers = detector.get_sleeper_agents()
        sleeper_ids = [s["agent_id"] for s in sleepers]
        assert sleeper1 in sleeper_ids
        assert sleeper2 in sleeper_ids

        # Should detect coordination
        coordinated = detector.detect_coordination([sleeper1, sleeper2])
        assert len(coordinated) == 1

    def test_get_agent_summary(self):
        """Test comprehensive agent summary."""
        detector = MultiRoundDetector()

        agent_id = "test_agent"

        # Add some history
        for round_idx in range(50):
            score = np.random.uniform(0.80, 0.90)
            detector.update_agent_history(agent_id, round_idx, score)

        # Get summary
        summary = detector.get_agent_summary(agent_id)

        # Verify summary contains all fields
        assert summary["agent_id"] == agent_id
        assert summary["history_length"] == 50
        assert len(summary["rounds"]) == 50
        assert len(summary["scores"]) == 50
        assert 0.8 < summary["mean_score"] < 0.9
        assert summary["std_score"] < 0.1
        assert 0.9 < summary["reputation"] < 1.0
        assert "reputation_variance" in summary
        assert "reputation_ci" in summary
        assert "is_sleeper" in summary
        assert "is_temporal_anomaly" in summary


class TestStatistics:
    """Test statistics tracking."""

    def test_stats_tracking(self):
        """Test that statistics are correctly tracked."""
        detector = MultiRoundDetector()

        # Add some data
        for round_idx in range(10):
            for agent_idx in range(5):
                agent_id = f"agent_{agent_idx}"
                score = 0.85
                detector.update_agent_history(agent_id, round_idx, score)

        stats = detector.get_stats()

        assert stats["total_rounds"] == 9  # 0-indexed
        assert stats["total_updates"] == 50  # 5 agents × 10 rounds
        assert stats["total_agents"] == 5

    def test_stats_reset(self):
        """Test statistics can be reset."""
        detector = MultiRoundDetector()

        # Add data
        detector.update_agent_history("agent1", 0, 0.85)
        detector.stats["sleepers_detected"] = 5

        # Reset
        detector.reset_stats()

        stats = detector.get_stats()
        assert stats["total_rounds"] == 0
        assert stats["sleepers_detected"] == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
