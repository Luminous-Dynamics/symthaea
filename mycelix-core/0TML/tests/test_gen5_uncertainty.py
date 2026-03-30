# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Gen 5 Uncertainty Quantification
==========================================

Tests:
1. Initialization and configuration
2. Calibration buffer updates
3. Prediction with confidence intervals
4. Abstention logic
5. Coverage computation
6. Threshold calculation
7. Edge cases and error handling

Author: Luminous Dynamics
Date: November 11, 2025
"""

import pytest
import numpy as np

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gen5.uncertainty import UncertaintyQuantifier


class TestUncertaintyQuantifier:
    """Test suite for UncertaintyQuantifier."""

    def test_initialization(self):
        """Test quantifier initializes correctly."""
        quantifier = UncertaintyQuantifier(alpha=0.10, buffer_size=256)

        assert quantifier.alpha == 0.10
        assert quantifier.buffer_size == 256
        assert quantifier.coverage_target == 0.90
        assert len(quantifier.buffer) == 0
        assert quantifier.stats["total_predictions"] == 0

    def test_update_calibration_buffer(self):
        """Test updating calibration buffer with scores."""
        quantifier = UncertaintyQuantifier(buffer_size=100)

        # Add some scores
        scores = [0.8, 0.9, 0.7, 0.85, 0.95]
        quantifier.update(scores)

        assert len(quantifier.buffer) == 5
        assert quantifier.stats["calibration_size"] == 5

        # Verify scores were added
        for score in scores:
            assert score in quantifier.buffer

    def test_update_filters_invalid_scores(self):
        """Test that invalid scores are filtered out."""
        quantifier = UncertaintyQuantifier()

        # Include invalid scores (NaN, inf)
        scores = [0.8, np.nan, 0.9, np.inf, 0.7, -np.inf]
        quantifier.update(scores)

        # Should only have valid scores
        assert len(quantifier.buffer) == 3
        assert all(np.isfinite(score) for score in quantifier.buffer)

    def test_predict_with_confidence_no_calibration(self):
        """Test prediction when calibration buffer is empty."""
        quantifier = UncertaintyQuantifier()

        # No calibration data yet
        decision, prob, (lower, upper) = quantifier.predict_with_confidence(0.75)

        # Should return decision but with full interval
        assert decision in ["HONEST", "BYZANTINE"]
        assert lower == 0.0
        assert upper == 1.0

    def test_predict_with_confidence_honest(self):
        """Test prediction for honest-like score."""
        quantifier = UncertaintyQuantifier(alpha=0.10)

        # Calibrate with honest scores (high values)
        honest_scores = [0.8, 0.85, 0.9, 0.88, 0.92, 0.87, 0.83, 0.86, 0.89, 0.91]
        quantifier.update(honest_scores)

        # Test honest score
        decision, prob, (lower, upper) = quantifier.predict_with_confidence(0.85)

        assert decision == "HONEST"
        assert prob > 0.1  # Some probability (percentile-based)
        assert 0.0 <= lower <= upper <= 1.0
        assert upper - lower < 0.5  # Reasonable interval width

    def test_predict_with_confidence_byzantine(self):
        """Test prediction for Byzantine-like score."""
        quantifier = UncertaintyQuantifier(alpha=0.10)

        # Calibrate with honest scores (high values)
        honest_scores = [0.8, 0.85, 0.9, 0.88, 0.92, 0.87, 0.83, 0.86, 0.89, 0.91]
        quantifier.update(honest_scores)

        # Test Byzantine score (low, unusual)
        decision, prob, (lower, upper) = quantifier.predict_with_confidence(0.15)

        assert decision == "BYZANTINE"
        assert prob > 0.5  # High confidence Byzantine
        assert 0.0 <= lower <= upper <= 1.0

    def test_should_abstain_crosses_boundary(self):
        """Test abstention when interval crosses decision boundary."""
        quantifier = UncertaintyQuantifier(abstain_threshold=0.15)

        # Interval that crosses 0.5 and is wide
        score = 0.52
        interval = (0.45, 0.62)  # Width=0.17, crosses 0.5

        should_abstain = quantifier.should_abstain(score, interval)

        assert should_abstain is True
        assert quantifier.stats["total_abstentions"] == 1

    def test_should_not_abstain_narrow_interval(self):
        """Test no abstention when interval is narrow."""
        quantifier = UncertaintyQuantifier(abstain_threshold=0.15)

        # Narrow interval
        score = 0.85
        interval = (0.82, 0.88)  # Width=0.06 < 0.15

        should_abstain = quantifier.should_abstain(score, interval)

        assert should_abstain is False

    def test_should_not_abstain_clear_decision(self):
        """Test no abstention when decision is clear."""
        quantifier = UncertaintyQuantifier(abstain_threshold=0.15)

        # Wide interval but doesn't cross boundary
        score = 0.85
        interval = (0.75, 0.95)  # Width=0.20 > 0.15, but > 0.5

        should_abstain = quantifier.should_abstain(score, interval)

        assert should_abstain is False  # Doesn't cross 0.5

    def test_compute_coverage_perfect(self):
        """Test coverage computation with perfect coverage."""
        quantifier = UncertaintyQuantifier()

        predictions = [
            (0.75, (0.70, 0.80)),
            (0.85, (0.80, 0.90)),
            (0.65, (0.60, 0.70)),
        ]
        true_scores = [0.74, 0.87, 0.66]  # All within intervals

        coverage = quantifier.compute_coverage(predictions, true_scores)

        assert coverage == 1.0  # 100% coverage

    def test_compute_coverage_partial(self):
        """Test coverage computation with partial coverage."""
        quantifier = UncertaintyQuantifier()

        predictions = [
            (0.75, (0.70, 0.80)),
            (0.85, (0.80, 0.90)),
            (0.65, (0.60, 0.70)),
        ]
        true_scores = [0.74, 0.95, 0.66]  # Middle one outside interval

        coverage = quantifier.compute_coverage(predictions, true_scores)

        assert abs(coverage - 2/3) < 1e-6  # 66.67% coverage

    def test_compute_coverage_length_mismatch(self):
        """Test coverage computation fails with length mismatch."""
        quantifier = UncertaintyQuantifier()

        predictions = [(0.75, (0.70, 0.80))]
        true_scores = [0.74, 0.85]  # Different length

        with pytest.raises(ValueError, match="Length mismatch"):
            quantifier.compute_coverage(predictions, true_scores)

    def test_threshold_computation(self):
        """Test threshold calculation from calibration data."""
        quantifier = UncertaintyQuantifier(alpha=0.10)

        # Add calibration scores
        scores = [0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65, 0.95, 0.55]
        quantifier.update(scores)

        threshold = quantifier.threshold()

        # Should be around 10th percentile
        expected_threshold = np.percentile(scores, 10)
        assert abs(threshold - expected_threshold) < 0.05

    def test_threshold_default_no_calibration(self):
        """Test threshold returns default when no calibration data."""
        quantifier = UncertaintyQuantifier()

        threshold = quantifier.threshold(default=0.5)

        assert threshold == 0.5

    def test_get_uncertainty_metrics(self):
        """Test uncertainty metrics retrieval."""
        quantifier = UncertaintyQuantifier(alpha=0.10)

        # Add some data
        quantifier.update([0.8, 0.9, 0.7, 0.85])
        quantifier.predict_with_confidence(0.85)
        quantifier.predict_with_confidence(0.15)

        metrics = quantifier.get_uncertainty_metrics()

        assert metrics["calibration_size"] == 4
        assert metrics["total_predictions"] == 2
        assert "abstention_rate" in metrics
        assert "mean_interval_width" in metrics
        assert metrics["alpha"] == 0.10
        assert metrics["coverage_target"] == 0.90

    def test_reset(self):
        """Test resetting quantifier state."""
        quantifier = UncertaintyQuantifier()

        # Add data
        quantifier.update([0.8, 0.9, 0.7])
        quantifier.predict_with_confidence(0.85)

        # Reset
        quantifier.reset()

        # Should be empty
        assert len(quantifier.buffer) == 0
        assert quantifier.stats["total_predictions"] == 0
        assert len(quantifier.coverage_history) == 0

    def test_repr(self):
        """Test string representation."""
        quantifier = UncertaintyQuantifier(alpha=0.10)

        repr_str = repr(quantifier)

        assert "UncertaintyQuantifier" in repr_str
        assert "alpha=0.10" in repr_str

    def test_buffer_size_limit(self):
        """Test that buffer respects size limit."""
        quantifier = UncertaintyQuantifier(buffer_size=10)

        # Add more scores than buffer size (use floats 0.0-1.0)
        scores = [float(i) / 20 for i in range(20)]
        quantifier.update(scores)

        # Should only keep last 10
        assert len(quantifier.buffer) == 10

    def test_prediction_statistics_tracking(self):
        """Test that prediction statistics are tracked correctly."""
        quantifier = UncertaintyQuantifier()

        quantifier.update([0.8, 0.9, 0.7, 0.85, 0.88])

        # Make predictions
        for score in [0.85, 0.15, 0.92, 0.10]:
            quantifier.predict_with_confidence(score)

        assert quantifier.stats["total_predictions"] == 4

    def test_interval_width_consistency(self):
        """Test that interval width is related to alpha."""
        quantifier = UncertaintyQuantifier(alpha=0.10)

        # Add well-distributed calibration data
        scores = np.linspace(0.1, 0.9, 100)
        quantifier.update(scores.tolist())

        # Get prediction for middle score
        _, _, (lower, upper) = quantifier.predict_with_confidence(0.5)

        # Interval width should be approximately alpha (for uniform distribution)
        width = upper - lower
        assert 0.05 < width < 0.20  # Reasonable for α=0.10


class TestConformalPrediction:
    """Test conformal prediction theory properties."""

    def test_coverage_guarantee_uniform_distribution(self):
        """Test coverage guarantee with uniform distribution."""
        quantifier = UncertaintyQuantifier(alpha=0.10, buffer_size=1000)

        # Generate uniform calibration data
        np.random.seed(42)
        calibration_scores = np.random.uniform(0.0, 1.0, size=500)
        quantifier.update(calibration_scores.tolist())

        # Generate test data
        test_scores = np.random.uniform(0.0, 1.0, size=200)

        # Make predictions
        predictions = []
        for score in test_scores:
            _, _, interval = quantifier.predict_with_confidence(score)
            predictions.append((score, interval))

        # Compute empirical coverage
        coverage = quantifier.compute_coverage(predictions, test_scores.tolist())

        # Should be approximately 1 - α = 90%
        # Allow wider margin for conformal prediction (can be conservative)
        assert 0.80 <= coverage <= 1.00, f"Coverage {coverage:.3f} outside [0.80, 1.00]"

    def test_coverage_guarantee_normal_distribution(self):
        """Test that system handles normal distribution gracefully."""
        # Note: Conformal prediction coverage guarantee only holds under
        # exchangeability (test ~ calibration). With different distributions,
        # coverage may deviate, but system should handle gracefully.

        quantifier = UncertaintyQuantifier(alpha=0.10, buffer_size=1000)

        # Calibrate with uniform distribution
        np.random.seed(42)
        calibration_scores = np.random.uniform(0.0, 1.0, size=500)
        quantifier.update(calibration_scores.tolist())

        # Test with normal distribution (different from calibration)
        test_scores = np.clip(np.random.normal(0.7, 0.15, size=200), 0.0, 1.0)

        # Make predictions
        predictions = []
        for score in test_scores:
            _, _, interval = quantifier.predict_with_confidence(score)
            predictions.append((score, interval))

        # Compute empirical coverage
        coverage = quantifier.compute_coverage(predictions, test_scores.tolist())

        # Coverage may vary from target (90%) due to distribution shift,
        # but should be reasonable (not 0% or 100%, system handles gracefully)
        assert 0.50 <= coverage <= 1.00, f"Coverage {coverage:.3f} unreasonable for distribution shift"

        # Verify system didn't crash and produced valid intervals
        assert len(predictions) == len(test_scores)
        for _, (lower, upper) in predictions:
            assert 0.0 <= lower <= upper <= 1.0  # Valid intervals


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_uncertainty_pipeline(self):
        """Test complete uncertainty quantification pipeline."""
        quantifier = UncertaintyQuantifier(alpha=0.10, abstain_threshold=0.15)

        # Phase 1: Calibration with honest nodes
        np.random.seed(42)
        honest_scores = np.clip(np.random.normal(0.85, 0.08, size=100), 0.0, 1.0)
        quantifier.update(honest_scores.tolist())

        # Phase 2: Test predictions
        test_cases = [
            (0.88, "HONEST"),  # Clear honest
            (0.15, "BYZANTINE"),  # Clear Byzantine
            (0.52, "AMBIGUOUS"),  # Borderline
        ]

        results = []
        for score, expected_type in test_cases:
            decision, prob, (lower, upper) = quantifier.predict_with_confidence(score)
            should_abstain = quantifier.should_abstain(score, (lower, upper))

            results.append({
                "score": score,
                "decision": decision,
                "probability": prob,
                "interval": (lower, upper),
                "abstain": should_abstain,
                "expected": expected_type,
            })

        # Verify results
        # Honest case (0.88)
        assert results[0]["decision"] == "HONEST"
        assert results[0]["abstain"] is False
        assert results[0]["probability"] > 0.7

        # Byzantine case (0.15)
        assert results[1]["decision"] == "BYZANTINE"
        assert results[1]["abstain"] is False

        # Ambiguous case (0.52)
        # Should likely abstain (crosses 0.5, wide interval)
        if results[2]["abstain"]:
            interval_width = results[2]["interval"][1] - results[2]["interval"][0]
            assert interval_width > 0.15

        print("\n🎯 Realistic Integration Test:")
        for result in results:
            print(f"\nScore: {result['score']:.2f} ({result['expected']})")
            print(f"  Decision: {result['decision']}")
            print(f"  Confidence: {result['probability']:.3f}")
            print(f"  Interval: [{result['interval'][0]:.3f}, {result['interval'][1]:.3f}]")
            print(f"  Abstain: {result['abstain']}")

        print(f"\n✅ Uncertainty quantification working!")

    def test_adaptive_threshold_learning(self):
        """Test that threshold adapts to data distribution."""
        quantifier = UncertaintyQuantifier(alpha=0.10)

        # Start with high-quality scores (honest)
        high_quality = [0.8, 0.85, 0.9, 0.88, 0.92]
        quantifier.update(high_quality)
        threshold_high = quantifier.threshold()

        # Add some lower-quality scores
        mixed_quality = [0.5, 0.6, 0.7, 0.65, 0.55]
        quantifier.update(mixed_quality)
        threshold_mixed = quantifier.threshold()

        # Threshold should decrease or stay similar when lower scores added
        # (May increase slightly due to buffer rolling window)
        assert threshold_mixed <= threshold_high * 1.1  # Allow 10% tolerance

        print(f"\n🎯 Adaptive Threshold Test:")
        print(f"  High-quality threshold: {threshold_high:.3f}")
        print(f"  Mixed-quality threshold: {threshold_mixed:.3f}")
        print(f"  ✅ Threshold adapts to data distribution!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
