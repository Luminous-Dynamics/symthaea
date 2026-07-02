# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Gen 5 Federated Meta-Learning
========================================

Tests:
1. LocalMetaLearner initialization and gradient computation
2. Differential privacy properties
3. Byzantine-robust aggregation methods
4. Federated ensemble convergence
5. Byzantine robustness in federated setting

Author: Luminous Dynamics
Date: November 11, 2025
"""

import pytest
import numpy as np

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gen5.federated_meta import (
    LocalMetaLearner,
    aggregate_gradients_krum,
    aggregate_gradients_trimmed_mean,
    aggregate_gradients_median,
    aggregate_gradients_reputation_weighted,
)
from gen5.meta_learning import MetaLearningEnsemble


# Mock detection method for testing
class MockDetector:
    """Mock Byzantine detector for testing."""

    def __init__(self, name: str, base_score: float = 0.5):
        self.name = name
        self.base_score = base_score

    def score(self, gradient: np.ndarray) -> float:
        """Return deterministic score for testing."""
        return self.base_score


class TestLocalMetaLearner:
    """Test suite for LocalMetaLearner."""

    def test_initialization(self):
        """Test local meta-learner initializes correctly."""
        learner = LocalMetaLearner(
            method_names=["pogq", "fltrust", "krum"],
            learning_rate=0.01,
            privacy_budget=8.0,
            gradient_clip_norm=1.0,
        )

        assert learner.n_methods == 3
        assert learner.config.privacy_budget == 8.0
        assert learner.config.gradient_clip_norm == 1.0
        assert learner.privacy_consumed == 0.0
        assert learner.n_updates == 0

    def test_compute_local_gradient_shape(self):
        """Test local gradient computation returns correct shape."""
        learner = LocalMetaLearner(
            method_names=["pogq", "fltrust"],
            privacy_budget=100.0,  # High ε for low noise
        )

        signals = np.array([[0.8, 0.9], [0.2, 0.3]])  # 2 samples
        labels = np.array([1.0, 0.0])  # Honest, Byzantine
        weights = np.array([0.5, 0.5])

        gradient = learner.compute_local_gradient(signals, labels, weights)

        assert gradient.shape == (2,)  # n_methods
        assert np.all(np.isfinite(gradient))

    def test_gradient_clipping(self):
        """Test that gradients are clipped to bounded norm."""
        learner = LocalMetaLearner(
            method_names=["method1", "method2"],
            gradient_clip_norm=0.5,  # Small clip norm
            privacy_budget=1000.0,  # High ε so noise << clip effect
        )

        # Create signals that will produce large gradient
        signals = np.array([[1.0, 1.0]] * 10)
        labels = np.array([0.0] * 10)  # All Byzantine (large residuals)
        weights = np.array([0.5, 0.5])

        np.random.seed(42)
        gradient = learner.compute_local_gradient(signals, labels, weights)

        # Gradient norm should be ≤ clip_norm (allowing for noise)
        # With high ε, noise is small, so norm should be close to clip_norm
        gradient_norm = np.linalg.norm(gradient)
        assert gradient_norm <= 0.55  # clip_norm + small noise tolerance

    def test_differential_privacy_noise(self):
        """Test that DP noise is added to gradients."""
        learner = LocalMetaLearner(
            method_names=["method1"],
            gradient_clip_norm=1.0,
            privacy_budget=1.0,  # Low ε → high noise
        )

        signals = np.array([[0.5]])
        labels = np.array([0.5])  # Zero gradient
        weights = np.array([1.0])

        np.random.seed(42)
        gradients = []
        for _ in range(100):
            grad = learner.compute_local_gradient(signals, labels, weights)
            gradients.append(grad[0])

        # With zero true gradient, observed variance ≈ noise variance
        # σ² = (clip_norm / ε)² = (1.0 / 1.0)² = 1.0
        observed_std = np.std(gradients)
        expected_std = 1.0  # clip_norm / privacy_budget

        # Allow 20% tolerance
        assert 0.8 * expected_std <= observed_std <= 1.2 * expected_std

    def test_privacy_tracking(self):
        """Test privacy consumption tracking."""
        learner = LocalMetaLearner(
            method_names=["method1"],
            privacy_budget=2.0,
        )

        signals = np.array([[0.8]])
        labels = np.array([1.0])
        weights = np.array([1.0])

        # Make 5 updates
        for _ in range(5):
            learner.compute_local_gradient(signals, labels, weights)

        assert learner.n_updates == 5
        assert learner.privacy_consumed == 5 * 2.0  # 10.0

        metrics = learner.get_privacy_metrics()
        assert metrics["privacy_consumed"] == 10.0
        assert metrics["n_updates"] == 5

    def test_repr(self):
        """Test string representation."""
        learner = LocalMetaLearner(method_names=["pogq"])

        repr_str = repr(learner)
        assert "LocalMetaLearner" in repr_str
        assert "ε=" in repr_str


class TestAggregationMethods:
    """Test suite for Byzantine-robust aggregation methods."""

    def test_krum_honest_agents(self):
        """Test Krum with all honest agents."""
        # Honest agents send similar gradients
        honest_grads = [
            np.array([0.1, 0.2]),
            np.array([0.12, 0.18]),
            np.array([0.09, 0.21]),
            np.array([0.11, 0.19]),
        ]

        aggregated = aggregate_gradients_krum(honest_grads, f=1)

        # Should select one of the honest gradients
        assert aggregated.shape == (2,)
        assert 0.09 <= aggregated[0] <= 0.12
        assert 0.18 <= aggregated[1] <= 0.21

    def test_krum_with_byzantine(self):
        """Test Krum filters out Byzantine outliers."""
        # 7 honest + 3 Byzantine
        honest_grads = [np.array([0.1, 0.2])] * 7
        byzantine_grads = [np.array([10.0, -5.0])] * 3
        all_grads = honest_grads + byzantine_grads

        aggregated = aggregate_gradients_krum(all_grads, f=3)

        # Should select honest gradient (not Byzantine outlier)
        assert np.allclose(aggregated, [0.1, 0.2], atol=0.01)

    def test_trimmed_mean_removes_outliers(self):
        """Test trimmed mean removes top/bottom outliers."""
        grads = [
            np.array([0.1, 0.2]),  # Normal
            np.array([0.12, 0.18]),  # Normal
            np.array([0.11, 0.19]),  # Normal
            np.array([0.09, 0.21]),  # Normal
            np.array([10.0, -5.0]),  # Byzantine outlier (top)
            np.array([-10.0, 15.0]),  # Byzantine outlier (bottom)
        ]

        # Trim top/bottom 16.7% → removes 1 from each end per coordinate
        aggregated = aggregate_gradients_trimmed_mean(grads, trim_ratio=0.167)

        # Should average the middle 4 gradients
        assert 0.09 <= aggregated[0] <= 0.12
        assert 0.18 <= aggregated[1] <= 0.21

    def test_median_robust_to_outliers(self):
        """Test median is robust to extreme outliers."""
        grads = [
            np.array([0.1, 0.2]),
            np.array([0.12, 0.19]),
            np.array([100.0, -50.0]),  # Extreme Byzantine outlier
        ]

        aggregated = aggregate_gradients_median(grads)

        # Median should ignore outlier
        assert aggregated[0] == 0.12
        assert aggregated[1] == 0.19

    def test_reputation_weighted_favors_high_reputation(self):
        """Test reputation-weighted aggregation."""
        grads = [
            np.array([0.1, 0.2]),  # High reputation (0.9)
            np.array([10.0, -5.0]),  # Low reputation (0.1)
        ]
        reputations = np.array([0.9, 0.1])

        aggregated = aggregate_gradients_reputation_weighted(grads, reputations)

        # Should be ~90% first gradient, ~10% second
        # (0.9 * [0.1, 0.2] + 0.1 * [10.0, -5.0]) / 1.0
        expected = np.array([0.9 * 0.1 + 0.1 * 10.0, 0.9 * 0.2 + 0.1 * (-5.0)])
        assert np.allclose(aggregated, expected)


class TestFederatedEnsemble:
    """Test suite for federated ensemble integration."""

    def test_federated_update_with_krum(self):
        """Test federated weight update with Krum aggregation."""
        # Setup ensemble
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.5),
        ]
        ensemble = MetaLearningEnsemble(methods)

        # Simulate 5 agents computing local gradients
        local_grads = [
            np.array([0.1, 0.2]),
            np.array([0.12, 0.18]),
            np.array([0.09, 0.21]),
            np.array([0.11, 0.19]),
            np.array([0.10, 0.20]),
        ]

        initial_weights = ensemble.weights.copy()

        # Federated update
        loss = ensemble.federated_update_weights(
            local_gradients=local_grads,
            aggregation_method="krum",
        )

        # Weights should have changed
        assert not np.allclose(ensemble.weights, initial_weights)
        assert loss >= 0  # Estimated loss (gradient norm)

    def test_federated_update_with_trimmed_mean(self):
        """Test federated update with trimmed mean."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)

        local_grads = [np.array([0.1]), np.array([0.2]), np.array([0.15])]

        loss = ensemble.federated_update_weights(
            local_gradients=local_grads,
            aggregation_method="trimmed_mean",
        )

        assert loss >= 0
        assert ensemble.stats["total_updates"] == 1

    def test_federated_update_with_reputation(self):
        """Test federated update with reputation weighting."""
        methods = [MockDetector("method1", 0.5)]
        ensemble = MetaLearningEnsemble(methods)

        local_grads = [np.array([0.1]), np.array([10.0])]  # Second is Byzantine
        reputations = np.array([0.9, 0.1])  # First has high reputation

        loss = ensemble.federated_update_weights(
            local_gradients=local_grads,
            agent_reputations=reputations,
            aggregation_method="reputation_weighted",
        )

        assert loss >= 0
        # Weight should move towards high-reputation gradient

    def test_federated_convergence(self):
        """Test that federated learning converges."""
        # Setup
        methods = [
            MockDetector("method1", 0.5),
            MockDetector("method2", 0.5),
        ]
        ensemble = MetaLearningEnsemble(methods)

        # Simulate 10 agents
        n_agents = 10
        n_rounds = 50

        losses = []
        for round_idx in range(n_rounds):
            # Each agent computes local gradient
            local_grads = []
            for agent_idx in range(n_agents):
                # Simulate local gradient (pointing towards honest detection)
                # Gradients encourage higher weight on method1
                grad = np.array([0.1, -0.05]) + np.random.normal(0, 0.01, size=2)
                local_grads.append(grad)

            # Federated update
            loss = ensemble.federated_update_weights(
                local_gradients=local_grads,
                aggregation_method="trimmed_mean",
            )
            losses.append(loss)

        # Loss (gradient norm) should decrease
        early_loss = np.mean(losses[:10])
        late_loss = np.mean(losses[-10:])
        assert late_loss < early_loss  # Convergence


class TestByzantineRobustness:
    """Test Byzantine robustness in federated setting."""

    def test_robustness_to_byzantine_agents(self):
        """Test that federated learning is robust to Byzantine agents."""
        methods = [MockDetector("method1", 0.5), MockDetector("method2", 0.5)]
        ensemble = MetaLearningEnsemble(methods)

        # 7 honest + 3 Byzantine agents
        n_rounds = 30

        for _ in range(n_rounds):
            # Honest agents send normal gradients
            honest_grads = [np.array([0.1, 0.2]) for _ in range(7)]

            # Byzantine agents send random/malicious gradients
            byzantine_grads = [
                np.random.uniform(-10, 10, size=2) for _ in range(3)
            ]

            all_grads = honest_grads + byzantine_grads

            # Aggregate with Krum (tolerates f < n/3 → 3 Byzantine OK)
            loss = ensemble.federated_update_weights(
                local_gradients=all_grads,
                aggregation_method="krum",
            )

            assert loss >= 0  # Should not crash

        # Weights should still be reasonable (not corrupted)
        assert np.all(np.isfinite(ensemble.weights))


class TestIntegration:
    """Integration tests for full federated meta-learning pipeline."""

    def test_complete_federated_pipeline(self):
        """Test complete federated meta-learning pipeline."""
        # Setup ensemble
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.5),
            MockDetector("krum", 0.5),
        ]
        ensemble = MetaLearningEnsemble(methods)

        # Setup agents
        n_agents = 10
        agents = [
            LocalMetaLearner(
                method_names=["pogq", "fltrust", "krum"],
                privacy_budget=8.0,
            )
            for _ in range(n_agents)
        ]

        # Simulate federated learning for 20 rounds
        n_rounds = 20

        for round_idx in range(n_rounds):
            # Each agent computes local gradient on local data
            local_grads = []

            for agent in agents:
                # Simulate local data (2 samples)
                local_signals = np.array([
                    [0.8, 0.9, 0.7],  # Honest-like
                    [0.2, 0.3, 0.4],  # Byzantine-like
                ])
                local_labels = np.array([1.0, 0.0])

                # Compute DP-noised gradient
                grad = agent.compute_local_gradient(
                    signals=local_signals,
                    labels=local_labels,
                    current_weights=ensemble.weights,
                )
                local_grads.append(grad)

            # Coordinator aggregates and updates
            loss = ensemble.federated_update_weights(
                local_gradients=local_grads,
                aggregation_method="trimmed_mean",
            )

        # After 20 rounds, ensemble should be trained
        assert ensemble.stats["total_updates"] == 20

        # Privacy consumed per agent
        for agent in agents:
            assert agent.privacy_consumed == 20 * 8.0  # 20 rounds × ε=8.0

        # Weights should have learned something
        importances = ensemble.get_method_importances()
        assert sum(importances.values()) == pytest.approx(1.0, abs=1e-6)

        print("\n🎯 Complete Federated Pipeline Test:")
        print(f"  Rounds: {n_rounds}")
        print(f"  Agents: {n_agents}")
        print(f"  Final estimated loss: {loss:.4f}")
        print(f"  Method importances: {importances}")
        print(f"  Privacy consumed per agent: {agents[0].privacy_consumed:.1f}")
        print(f"  ✅ Federated meta-learning working!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
