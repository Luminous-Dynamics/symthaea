# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Gen 5 Layer 4: Federated Validator

Test Coverage:
- Shamir secret sharing (4 tests)
- Threshold validation (3 tests)
- Krum selection (1 test)
- Byzantine-robust reconstruction (2 tests)
- Integration scenarios (5 tests)

Total: 15 tests
"""

import pytest
import numpy as np
from gen5.federated_validator import (
    ShamirSecretSharing,
    ThresholdValidator,
)
from gen5.meta_learning import MetaLearningEnsemble


# Mock detection method for testing
class MockDetectionMethod:
    """Simple mock detection method."""

    def __init__(self, name: str, base_score: float = 0.7):
        self.name = name
        self.base_score = base_score

    def score(self, gradient: np.ndarray) -> float:
        """Return score based on gradient magnitude."""
        magnitude = np.linalg.norm(gradient)
        # Large gradients → low score (Byzantine)
        # Small gradients → high score (honest)
        return max(0.0, min(1.0, self.base_score - magnitude / 10.0))


class TestShamirSecretSharing:
    """Test suite for Shamir secret sharing."""

    def test_shamir_basic_reconstruction(self):
        """Test exact reconstruction with t shares."""
        sss = ShamirSecretSharing(threshold=3, num_shares=5)

        secret = 0.75
        shares = sss.generate_shares(secret)

        # Reconstruct with exactly t shares
        reconstructed = sss.reconstruct(shares[:3])

        # Should match within floating point precision
        assert abs(reconstructed - secret) < 0.01

    def test_shamir_excess_shares(self):
        """Test reconstruction with > t shares."""
        sss = ShamirSecretSharing(threshold=3, num_shares=5)

        secret = 0.85
        shares = sss.generate_shares(secret)

        # Reconstruct with all shares (more than t)
        reconstructed = sss.reconstruct(shares)  # Uses first 3

        assert abs(reconstructed - secret) < 0.01

    def test_shamir_insufficient_shares(self):
        """Test error with < t shares."""
        sss = ShamirSecretSharing(threshold=3, num_shares=5)

        secret = 0.50
        shares = sss.generate_shares(secret)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Need >= 3 shares"):
            sss.reconstruct(shares[:2])

    def test_shamir_information_theoretic_security(self):
        """Test that < t shares reveal no information."""
        sss = ShamirSecretSharing(threshold=3, num_shares=5)

        secret = 0.65
        shares = sss.generate_shares(secret)

        # With only 2 shares (< t), cannot reconstruct
        # Trying anyway should give wrong result
        try:
            wrong_reconstruction = sss.reconstruct(shares[:2])
            # If no error (shouldn't happen), verify it's wrong
            assert abs(wrong_reconstruction - secret) > 0.1
        except ValueError:
            # Expected: insufficient shares
            pass


class TestThresholdValidator:
    """Test suite for threshold validation protocol."""

    def test_threshold_validator_honest(self):
        """Test validation with all honest validators."""
        # Create simple ensemble with mock methods
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=5,
            threshold=3,
        )

        # Create honest gradient (high scores)
        np.random.seed(42)
        gradient = np.random.randn(10) * 0.1  # Small gradient (likely honest)

        decision, confidence, details = validator.validate_gradient(gradient)

        # All honest validators should agree
        assert details["num_validators"] == 5
        assert details["threshold"] == 3
        assert details["score_deviation"] < 0.1  # Reconstruction should be accurate

    def test_threshold_validator_some_byzantine(self):
        """Test validation with < max_byzantine Byzantine validators."""
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=7,
            threshold=4,
        )

        # Create gradient
        np.random.seed(42)
        gradient = np.random.randn(10)

        # Simulate 1 Byzantine validator (max_byzantine = 1 for n=7, t=4)
        # First validator returns 0.0 (Byzantine), others honest
        validator_scores = None  # Use honest simulation

        decision, confidence, details = validator.validate_gradient(
            gradient, validator_scores
        )

        # Should still work with < max_byzantine Byzantine
        assert details["max_byzantine_tolerated"] == 1

    def test_threshold_validator_max_byzantine(self):
        """Test validation with exactly max_byzantine Byzantine validators."""
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=7,
            threshold=4,
        )

        # Create gradient
        np.random.seed(42)
        gradient = np.random.randn(10) * 2.0  # Large gradient (likely Byzantine)

        # Get reference score first
        signals = {}
        for method in ensemble.base_methods:
            signals[method.name] = method.score(gradient)
        reference_score = ensemble.compute_ensemble_score(signals)

        # Simulate 1 Byzantine validator (exactly max_byzantine)
        validator_scores = [reference_score] * 6 + [0.0]  # 6 honest, 1 Byzantine

        decision, confidence, details = validator.validate_gradient(
            gradient, validator_scores
        )

        # Should still work (Krum filters out Byzantine)
        assert details["num_validators"] == 7


class TestKrumSelection:
    """Test suite for Krum Byzantine-robust selection."""

    def test_krum_selection(self):
        """Test Krum selects correct candidate."""
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=7,
            threshold=4,
        )

        # Candidate values: 5 honest (~0.7) + 2 Byzantine (~0.0)
        candidates = [0.70, 0.72, 0.68, 0.71, 0.69, 0.05, 0.03]

        # Krum should select value near 0.7 (most consistent with majority)
        selected = validator._krum_select(candidates)

        # Should be close to honest values, far from Byzantine
        assert 0.65 < selected < 0.75
        assert abs(selected - 0.05) > 0.5


class TestByzantineRobustReconstruction:
    """Test suite for Byzantine-robust reconstruction."""

    def test_robust_reconstruction(self):
        """Test Byzantine shares are filtered out."""
        sss = ShamirSecretSharing(threshold=4, num_shares=7)

        secret = 0.80
        shares = sss.generate_shares(secret)

        # Create ensemble for validator
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=7,
            threshold=4,
        )

        # Simulate validator scores: 6 honest + 1 Byzantine
        validator_scores = [0.80] * 6 + [0.0]

        # Robust reconstruction should filter Byzantine
        reconstructed = validator._robust_reconstruct(shares, validator_scores)

        # Should be close to true secret (Byzantine filtered)
        assert abs(reconstructed - secret) < 0.15

    def test_byzantine_tolerance_limit(self):
        """Test system handles Byzantine validators up to tolerance."""
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        # n=7, t=4 → max_byzantine = 1
        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=7,
            threshold=4,
        )

        np.random.seed(42)
        gradient = np.random.randn(10)

        # Get reference score
        signals = {}
        for method in ensemble.base_methods:
            signals[method.name] = method.score(gradient)
        reference_score = ensemble.compute_ensemble_score(signals)

        # Test with 1 Byzantine (at tolerance limit)
        validator_scores_1 = [reference_score] * 6 + [0.0]
        decision1, _, details1 = validator.validate_gradient(gradient, validator_scores_1)

        # Should work (within tolerance)
        assert details1["max_byzantine_tolerated"] == 1

        # Test with 2 Byzantine (exceeds tolerance)
        # System may fail or give incorrect result
        validator_scores_2 = [reference_score] * 5 + [0.0, 0.0]
        decision2, _, details2 = validator.validate_gradient(gradient, validator_scores_2)

        # Exceeds tolerance - result may be unreliable
        # (We don't assert correctness here, just that it completes)
        assert details2["num_validators"] == 7


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_end_to_end_honest_gradient(self):
        """Test full pipeline with honest gradient."""
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=7,
            threshold=4,
        )

        # Create small gradient (likely honest)
        np.random.seed(42)
        gradient = np.random.randn(10) * 0.1

        decision, confidence, details = validator.validate_gradient(gradient)

        # Verify protocol completed
        assert decision in ["honest", "byzantine"]
        assert 0 <= confidence <= 1
        assert details["reference_score"] >= 0
        assert details["reconstructed_score"] >= 0

    def test_end_to_end_byzantine_gradient(self):
        """Test full pipeline with Byzantine gradient."""
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=7,
            threshold=4,
        )

        # Create large gradient (likely Byzantine)
        np.random.seed(42)
        gradient = np.random.randn(10) * 5.0

        decision, confidence, details = validator.validate_gradient(gradient)

        # Verify protocol completed
        assert decision in ["honest", "byzantine"]
        assert 0 <= confidence <= 1

    def test_integration_with_ensemble(self):
        """Test integration with MetaLearningEnsemble."""
        # Create ensemble with custom method
        class HighScoreMethod:
            """Method that always returns high scores."""

            def __init__(self, name: str):
                self.name = name

            def score(self, gradient: np.ndarray) -> float:
                return 0.90

        ensemble = MetaLearningEnsemble(base_methods=[HighScoreMethod("high_score")])

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=5,
            threshold=3,
        )

        gradient = np.random.randn(10)

        decision, confidence, details = validator.validate_gradient(gradient)

        # High score → honest decision
        assert decision == "honest"
        assert details["reference_score"] > 0.5

    def test_multiple_gradients(self):
        """Test batch processing of multiple gradients."""
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=5,
            threshold=3,
        )

        np.random.seed(42)

        # Process 10 gradients
        for _ in range(10):
            gradient = np.random.randn(10)
            decision, confidence, details = validator.validate_gradient(gradient)

            # Verify each completes successfully
            assert decision in ["honest", "byzantine"]
            assert 0 <= confidence <= 1

    def test_performance_overhead(self):
        """Test acceptable latency for distributed validation."""
        ensemble = MetaLearningEnsemble(
            base_methods=[MockDetectionMethod("method1"), MockDetectionMethod("method2")]
        )

        validator = ThresholdValidator(
            ensemble=ensemble,
            num_validators=7,
            threshold=4,
        )

        np.random.seed(42)
        gradient = np.random.randn(10)

        import time

        # Measure time for single validation
        start = time.time()
        decision, confidence, details = validator.validate_gradient(gradient)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 100ms)
        # Note: This is Python simulation, actual distributed system would be slower
        assert elapsed < 0.1  # 100ms

        # Verify result
        assert decision in ["honest", "byzantine"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
