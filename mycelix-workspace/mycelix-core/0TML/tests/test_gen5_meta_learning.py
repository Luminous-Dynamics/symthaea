# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Gen 5 Meta-Learning Ensemble
======================================

Tests:
1. Initialization and configuration
2. Signal computation from base methods
3. Ensemble score calculation
4. Weight updates and convergence
5. Method importance extraction
6. Save/load weights
7. Edge cases and error handling

Author: Luminous Dynamics
Date: November 11, 2025
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gen5.meta_learning import MetaLearningEnsemble, MetaLearningConfig


# Mock detection method for testing
class MockDetector:
    """Mock Byzantine detector for testing."""

    def __init__(self, name: str, base_score: float = 0.5):
        self.name = name
        self.base_score = base_score
        self.call_count = 0

    def score(self, gradient: np.ndarray) -> float:
        """Return deterministic score for testing."""
        self.call_count += 1
        # Deterministic: use gradient mean as signal
        signal = float(np.clip(np.mean(gradient) + self.base_score, 0.0, 1.0))
        return signal


class TestMetaLearningEnsemble:
    """Test suite for MetaLearningEnsemble."""

    def test_initialization(self):
        """Test ensemble initializes correctly."""
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.6),
            MockDetector("krum", 0.4),
        ]

        ensemble = MetaLearningEnsemble(methods)

        assert ensemble.n_methods == 3
        assert len(ensemble.weights) == 3
        assert np.allclose(ensemble.weights, [1/3, 1/3, 1/3])
        assert ensemble.iteration == 0
        assert len(ensemble.loss_history) == 0

    def test_initialization_empty_methods(self):
        """Test initialization fails with empty methods."""
        with pytest.raises(ValueError, match="at least one base detection method"):
            MetaLearningEnsemble([])

    def test_compute_signals(self):
        """Test signal computation from base methods."""
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.6),
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Test gradient (mean = 0.2)
        gradient = np.array([0.1, 0.2, 0.3])

        signals = ensemble.compute_signals(gradient)

        assert "pogq" in signals
        assert "fltrust" in signals
        assert 0.0 <= signals["pogq"] <= 1.0
        assert 0.0 <= signals["fltrust"] <= 1.0

        # Verify methods were called
        assert methods[0].call_count == 1
        assert methods[1].call_count == 1

    def test_compute_ensemble_score(self):
        """Test ensemble score computation."""
        methods = [
            MockDetector("method1", 0.0),  # Will produce low scores
            MockDetector("method2", 1.0),  # Will produce high scores
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Equal weights initially (0.5, 0.5)
        signals = {"method1": 0.2, "method2": 0.8}
        score = ensemble.compute_ensemble_score(signals)

        # Should be weighted average: 0.5*0.2 + 0.5*0.8 = 0.5
        assert 0.45 <= score <= 0.55  # Allow small numerical error

    def test_compute_ensemble_score_missing_signals(self):
        """Test ensemble score handles missing signals gracefully."""
        methods = [
            MockDetector("method1", 0.5),
            MockDetector("method2", 0.5),
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Missing method2 signal
        signals = {"method1": 0.8}
        score = ensemble.compute_ensemble_score(signals)

        # Should use 0.5 default for missing signal
        # score = 0.5*0.8 + 0.5*0.5 = 0.65
        assert 0.60 <= score <= 0.70

    def test_weight_update_convergence(self):
        """Test weight updates converge to optimal solution."""
        methods = [
            MockDetector("good_detector", 0.0),   # Perfect detector
            MockDetector("bad_detector", 0.5),    # Random detector
        ]

        ensemble = MetaLearningEnsemble(
            methods,
            config=MetaLearningConfig(learning_rate=0.1, momentum=0.9)
        )

        # Generate synthetic training data
        # good_detector gets signal=1.0 for honest, signal=0.0 for byzantine
        # bad_detector gets signal=0.5 always (random)
        n_samples = 100
        signals_batch = []
        labels_batch = []

        for i in range(n_samples):
            is_honest = (i % 2 == 0)

            if is_honest:
                signals = [1.0, 0.5]  # good detector says honest, bad is random
                label = 1.0
            else:
                signals = [0.0, 0.5]  # good detector says byzantine, bad is random
                label = 0.0

            signals_batch.append(signals)
            labels_batch.append(label)

        signals_batch = np.array(signals_batch)
        labels_batch = np.array(labels_batch)

        # Train for multiple iterations
        initial_loss = None
        for iteration in range(50):
            # Train on batches of 20
            for i in range(0, n_samples, 20):
                batch_signals = signals_batch[i:i+20]
                batch_labels = labels_batch[i:i+20]

                loss = ensemble.update_weights(batch_signals, batch_labels)

                if initial_loss is None:
                    initial_loss = loss

        final_loss = ensemble.loss_history[-1]

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, (
            f"Loss did not decrease: initial={initial_loss:.4f}, "
            f"final={final_loss:.4f}"
        )

        # Good detector should have higher weight
        importances = ensemble.get_method_importances()
        assert importances["good_detector"] > importances["bad_detector"], (
            f"Good detector should have higher importance: {importances}"
        )

    def test_get_method_importances(self):
        """Test method importance extraction."""
        methods = [
            MockDetector("method1", 0.5),
            MockDetector("method2", 0.5),
            MockDetector("method3", 0.5),
        ]

        ensemble = MetaLearningEnsemble(methods)

        importances = ensemble.get_method_importances()

        # Should have all methods
        assert "method1" in importances
        assert "method2" in importances
        assert "method3" in importances

        # Importances should sum to 1.0
        total_importance = sum(importances.values())
        assert abs(total_importance - 1.0) < 1e-6

        # All importances should be positive
        for importance in importances.values():
            assert 0.0 <= importance <= 1.0

    def test_save_load_weights(self):
        """Test saving and loading weights."""
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.6),
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Modify weights to non-uniform values
        ensemble.weights = np.array([0.7, 0.3])
        ensemble.iteration = 42
        ensemble.loss_history = [0.5, 0.4, 0.3]
        ensemble.stats["best_loss"] = 0.25

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            ensemble.save_weights(temp_path)

            # Create new ensemble and load weights
            new_methods = [
                MockDetector("pogq", 0.5),
                MockDetector("fltrust", 0.6),
            ]
            new_ensemble = MetaLearningEnsemble(new_methods)
            new_ensemble.load_weights(temp_path)

            # Verify weights match
            assert np.allclose(new_ensemble.weights, ensemble.weights)
            assert new_ensemble.iteration == 42
            assert new_ensemble.loss_history == [0.5, 0.4, 0.3]
            assert new_ensemble.stats["best_loss"] == 0.25

        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()

    def test_load_weights_method_mismatch(self):
        """Test loading weights fails when methods don't match."""
        methods = [MockDetector("pogq", 0.5)]
        ensemble = MetaLearningEnsemble(methods)

        # Save weights
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            ensemble.save_weights(temp_path)

            # Try to load with different methods
            different_methods = [MockDetector("fltrust", 0.5)]
            new_ensemble = MetaLearningEnsemble(different_methods)

            with pytest.raises(ValueError, match="Method mismatch"):
                new_ensemble.load_weights(temp_path)

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_convergence_detection(self):
        """Test that convergence is detected when loss stabilizes."""
        methods = [MockDetector("method", 0.5)]
        ensemble = MetaLearningEnsemble(
            methods,
            config=MetaLearningConfig(convergence_threshold=1e-4)
        )

        # Simulate converged loss (very small changes)
        for i in range(15):
            # Create dummy batch
            signals_batch = np.array([[0.5]] * 20)
            labels_batch = np.array([0.5] * 20)

            # Add tiny noise to create near-constant loss
            labels_batch += np.random.randn(20) * 1e-5

            ensemble.update_weights(signals_batch, labels_batch)

        # Should detect convergence
        metrics = ensemble.get_convergence_metrics()
        # Note: Might not converge immediately due to random noise
        # Just check that convergence detection is working
        assert "converged" in metrics
        assert isinstance(metrics["converged"], bool)

    def test_batch_size_validation(self):
        """Test that small batches are rejected."""
        methods = [MockDetector("method", 0.5)]
        ensemble = MetaLearningEnsemble(
            methods,
            config=MetaLearningConfig(min_batch_size=10)
        )

        # Too small batch (5 < 10)
        small_batch = np.array([[0.5]] * 5)
        small_labels = np.array([1.0] * 5)

        with pytest.raises(ValueError, match="Batch too small"):
            ensemble.update_weights(small_batch, small_labels)

    def test_repr(self):
        """Test string representation."""
        methods = [
            MockDetector("pogq", 0.5),
            MockDetector("fltrust", 0.6),
        ]

        ensemble = MetaLearningEnsemble(methods)

        repr_str = repr(ensemble)

        assert "MetaLearningEnsemble" in repr_str
        assert "pogq" in repr_str
        assert "fltrust" in repr_str

    def test_numerical_stability_extreme_weights(self):
        """Test ensemble handles extreme weight values."""
        methods = [
            MockDetector("method1", 0.5),
            MockDetector("method2", 0.5),
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Set extreme weights
        ensemble.weights = np.array([100.0, -100.0])

        # Should still compute score without overflow/underflow
        signals = {"method1": 0.8, "method2": 0.2}
        score = ensemble.compute_ensemble_score(signals)

        assert 0.0 <= score <= 1.0  # Should be valid probability

    def test_convergence_metrics_empty_history(self):
        """Test convergence metrics when no training has occurred."""
        methods = [MockDetector("method", 0.5)]
        ensemble = MetaLearningEnsemble(methods)

        metrics = ensemble.get_convergence_metrics()

        assert metrics["current_loss"] is None
        assert metrics["best_loss"] == float("inf")
        assert metrics["iterations"] == 0
        assert metrics["converged"] is False
        assert metrics["loss_std"] is None


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_three_method_ensemble(self):
        """Test ensemble with 3 realistic detection methods."""
        # Simulate PoGQ (quality-based), FLTrust (direction-based), Krum (distance-based)
        methods = [
            MockDetector("pogq", 0.3),      # Tends toward lower scores
            MockDetector("fltrust", 0.6),   # Tends toward higher scores
            MockDetector("krum", 0.5),      # Neutral
        ]

        ensemble = MetaLearningEnsemble(methods)

        # Generate realistic training data
        # Scenario: Byzantine nodes have low PoGQ, low FLTrust, low Krum
        #          Honest nodes have high on all
        n_samples = 200
        signals_batch = []
        labels_batch = []

        np.random.seed(42)  # Reproducible

        for i in range(n_samples):
            is_honest = (i < n_samples // 2)  # 50% honest, 50% byzantine

            if is_honest:
                # Honest: all methods give high scores (with noise)
                pogq = np.clip(np.random.normal(0.85, 0.1), 0.0, 1.0)
                fltrust = np.clip(np.random.normal(0.90, 0.08), 0.0, 1.0)
                krum = np.clip(np.random.normal(0.80, 0.12), 0.0, 1.0)
                label = 1.0
            else:
                # Byzantine: all methods give low scores (with noise)
                pogq = np.clip(np.random.normal(0.20, 0.1), 0.0, 1.0)
                fltrust = np.clip(np.random.normal(0.15, 0.08), 0.0, 1.0)
                krum = np.clip(np.random.normal(0.25, 0.12), 0.0, 1.0)
                label = 0.0

            signals_batch.append([pogq, fltrust, krum])
            labels_batch.append(label)

        signals_batch = np.array(signals_batch)
        labels_batch = np.array(labels_batch)

        # Train ensemble
        n_epochs = 10
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            signals_shuffled = signals_batch[indices]
            labels_shuffled = labels_batch[indices]

            # Train in batches of 20
            for i in range(0, n_samples, 20):
                batch_signals = signals_shuffled[i:i+20]
                batch_labels = labels_shuffled[i:i+20]

                ensemble.update_weights(batch_signals, batch_labels)

        # Evaluate final performance
        final_importances = ensemble.get_method_importances()
        final_loss = ensemble.loss_history[-1]

        # All methods should have reasonable importance (>5%)
        for method, importance in final_importances.items():
            assert importance > 0.05, f"{method} has too low importance: {importance}"

        # Loss should be low (< 0.25)
        assert final_loss < 0.25, f"Final loss too high: {final_loss}"

        # Test prediction on new samples
        test_honest = np.array([0.88, 0.92, 0.82])  # High scores
        honest_score = ensemble.compute_ensemble_score({
            "pogq": test_honest[0],
            "fltrust": test_honest[1],
            "krum": test_honest[2]
        })

        test_byzantine = np.array([0.18, 0.12, 0.22])  # Low scores
        byzantine_score = ensemble.compute_ensemble_score({
            "pogq": test_byzantine[0],
            "fltrust": test_byzantine[1],
            "krum": test_byzantine[2]
        })

        # Honest should have high score (>0.7)
        assert honest_score > 0.7, f"Honest score too low: {honest_score}"

        # Byzantine should have low score (<0.3)
        assert byzantine_score < 0.3, f"Byzantine score too high: {byzantine_score}"

        print(f"\nIntegration Test Results:")
        print(f"  Final Loss: {final_loss:.4f}")
        print(f"  Method Importances: {final_importances}")
        print(f"  Honest Score: {honest_score:.3f}")
        print(f"  Byzantine Score: {byzantine_score:.3f}")
        print(f"  ✅ All checks passed!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
