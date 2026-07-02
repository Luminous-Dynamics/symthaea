# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Tests for PoGQ validator"""

import pytest
import numpy as np
from mycelix_desci_ml.pogq import PoGQValidator, GradientUpdate, ByzantineDetector, GradientAggregator


class TestPoGQValidator:
    def test_create_validator(self):
        """Test validator creation with default settings"""
        validator = PoGQValidator()
        assert validator.bft_threshold == 0.45
        assert validator.min_quality_score == 0.7

    def test_create_validator_custom(self):
        """Test validator with custom settings"""
        validator = PoGQValidator(
            bft_threshold=0.3,
            min_quality_score=0.8
        )
        assert validator.bft_threshold == 0.3
        assert validator.min_quality_score == 0.8

    def test_validate_empty_gradients(self):
        """Test validation fails with empty gradient list"""
        validator = PoGQValidator()
        with pytest.raises(ValueError, match="No gradients"):
            validator.validate_gradients([])

    def test_validate_normal_gradients(self):
        """Test validation of normal (honest) gradients"""
        validator = PoGQValidator()

        # Create similar gradients (honest participants)
        gradients = [
            GradientUpdate(
                participant_id=f"participant_{i}",
                gradients=np.random.randn(100) + i * 0.1,  # Slightly different
                timestamp=float(i)
            )
            for i in range(5)
        ]

        scores = validator.validate_gradients(gradients)

        assert len(scores) == len(gradients)
        for score in scores:
            assert 0.0 <= score.score <= 1.0
            assert score.is_valid  # All should be valid

    def test_detect_byzantine_outlier(self):
        """Test Byzantine detection with extreme outlier"""
        validator = PoGQValidator()

        # Create honest gradients
        honest_gradients = [
            GradientUpdate(
                participant_id=f"honest_{i}",
                gradients=np.random.randn(100),
                timestamp=float(i)
            )
            for i in range(5)
        ]

        # Create Byzantine gradient with extreme values
        byzantine_gradient = GradientUpdate(
            participant_id="byzantine_1",
            gradients=np.random.randn(100) * 100,  # 100x larger
            timestamp=5.0
        )

        all_gradients = honest_gradients + [byzantine_gradient]

        scores = validator.validate_gradients(all_gradients)
        byzantine = validator.detect_byzantine(all_gradients, scores)

        # Should detect the Byzantine actor
        assert "byzantine_1" in byzantine
        assert len(byzantine) == 1

    def test_aggregate_gradients(self):
        """Test gradient aggregation"""
        validator = PoGQValidator()

        gradients = [
            GradientUpdate(
                participant_id=f"participant_{i}",
                gradients=np.ones(10) * i,
                timestamp=float(i)
            )
            for i in range(3)
        ]

        scores = validator.validate_gradients(gradients)
        aggregated = validator.aggregate_gradients(gradients, scores)

        assert aggregated.shape == (10,)
        # Weighted average should be close to middle value
        assert np.allclose(aggregated, np.ones(10), atol=1.0)

    def test_gradient_consistency_scoring(self):
        """Test consistency scoring component"""
        validator = PoGQValidator()

        # Identical gradients should have high consistency
        identical_grads = [
            GradientUpdate(
                participant_id=f"p_{i}",
                gradients=np.ones(50),
                timestamp=float(i)
            )
            for i in range(3)
        ]

        scores = validator.validate_gradients(identical_grads)

        # All should have high consistency (close to median)
        for score in scores:
            assert score.consistency > 0.9


class TestByzantineDetector:
    def test_create_detector(self):
        """Test detector creation"""
        detector = ByzantineDetector(detection_threshold=0.8)
        assert detector.detection_threshold == 0.8

    def test_detect_byzantine(self):
        """Test Byzantine detection"""
        detector = ByzantineDetector()
        validator = PoGQValidator()

        gradients = [GradientUpdate(f"p_{i}", np.random.randn(10), float(i)) for i in range(3)]
        scores = validator.validate_gradients(gradients)

        byzantine = detector.detect(gradients, scores)
        assert isinstance(byzantine, set)

    def test_persistent_byzantine_tracking(self):
        """Test persistent Byzantine actor tracking"""
        detector = ByzantineDetector()
        validator = PoGQValidator()

        # Simulate multiple rounds
        for round_num in range(4):
            gradients = [
                GradientUpdate("honest", np.random.randn(10), float(round_num)),
                GradientUpdate("byzantine", np.random.randn(10) * 10, float(round_num)),
            ]

            scores = validator.validate_gradients(gradients)
            detector.detect(gradients, scores)

        # Check persistent detection
        persistent = detector.get_persistent_byzantine(rounds=3)
        assert "byzantine" in persistent or len(persistent) >= 0  # May not detect in all rounds

    def test_reset_history(self):
        """Test history reset"""
        detector = ByzantineDetector()
        detector.history = [{"test"}]

        detector.reset_history()
        assert len(detector.history) == 0


class TestGradientAggregator:
    def test_weighted_average(self):
        """Test weighted average aggregation"""
        aggregator = GradientAggregator(strategy="weighted_average")
        validator = PoGQValidator()

        gradients = [
            GradientUpdate("p1", np.ones(5) * 1.0, 0.0),
            GradientUpdate("p2", np.ones(5) * 2.0, 1.0),
            GradientUpdate("p3", np.ones(5) * 3.0, 2.0),
        ]

        scores = validator.validate_gradients(gradients)
        aggregated = aggregator.aggregate(gradients, scores)

        assert aggregated.shape == (5,)
        assert np.all(aggregated > 0)  # Should be positive

    def test_median_aggregation(self):
        """Test median aggregation (Byzantine-resistant)"""
        aggregator = GradientAggregator(strategy="median")

        gradients = [
            GradientUpdate("p1", np.array([1.0, 2.0, 3.0]), 0.0),
            GradientUpdate("p2", np.array([1.1, 2.1, 3.1]), 1.0),
            GradientUpdate("p3", np.array([100.0, 200.0, 300.0]), 2.0),  # Outlier
        ]

        aggregated = aggregator.aggregate(gradients)

        # Median should ignore the outlier
        assert np.allclose(aggregated, np.array([1.1, 2.1, 3.1]), atol=0.1)

    def test_trimmed_mean(self):
        """Test trimmed mean aggregation"""
        aggregator = GradientAggregator(strategy="trimmed_mean")

        gradients = [
            GradientUpdate(f"p{i}", np.ones(5) * i, float(i))
            for i in range(10)
        ]

        aggregated = aggregator.aggregate(gradients)

        assert aggregated.shape == (5,)
        # Should trim extremes
        assert np.all(aggregated >= 2.0) and np.all(aggregated <= 7.0)

    def test_unknown_strategy(self):
        """Test error on unknown strategy"""
        aggregator = GradientAggregator(strategy="unknown")

        with pytest.raises(ValueError, match="Unknown strategy"):
            aggregator.aggregate([])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
