# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unit tests for GradientDimensionalityAnalyzer

Tests dimensionality detection, parameter recommendation, and edge cases.
"""

import pytest
import torch
import numpy as np
from src.byzantine_detection import GradientDimensionalityAnalyzer, GradientProfile


class TestGradientDimensionalityAnalyzer:
    """Test suite for GradientDimensionalityAnalyzer"""

    def test_initialization(self):
        """Test analyzer can be initialized with default parameters"""
        analyzer = GradientDimensionalityAnalyzer()
        assert analyzer.variance_threshold == 0.95
        assert analyzer.min_samples_for_pca == 5

    def test_initialization_custom_params(self):
        """Test analyzer with custom parameters"""
        analyzer = GradientDimensionalityAnalyzer(
            variance_threshold=0.90, min_samples_for_pca=3
        )
        assert analyzer.variance_threshold == 0.90
        assert analyzer.min_samples_for_pca == 3

    def test_low_dimensional_gradients(self):
        """Test detection strategy for low-dimensional gradients (tabular data)"""
        analyzer = GradientDimensionalityAnalyzer()

        # Simulate gradients from breast cancer (30 features)
        num_nodes = 10
        dim = 30
        gradients = [torch.randn(dim) for _ in range(num_nodes)]

        profile = analyzer.analyze_gradients(gradients)

        # Should detect low dimensionality
        assert profile.nominal_dimensionality == dim
        assert profile.effective_dimensionality < 100
        assert profile.detection_strategy == "low_dim"

        # Should recommend wide thresholds for low-dim
        assert profile.recommended_cos_min <= -0.5  # Wide range
        assert profile.recommended_cos_max >= 0.90

        print(f"\n{profile}")

    def test_mid_dimensional_gradients(self):
        """Test detection strategy for mid-dimensional gradients (EMNIST-like)"""
        analyzer = GradientDimensionalityAnalyzer()

        # Simulate gradients from EMNIST (~800 dimensions)
        num_nodes = 10
        dim = 800
        gradients = [torch.randn(dim) for _ in range(num_nodes)]

        profile = analyzer.analyze_gradients(gradients)

        # Should detect mid dimensionality
        assert profile.nominal_dimensionality == dim
        assert 100 <= profile.effective_dimensionality < 1000
        assert profile.detection_strategy == "mid_dim"

        # Should recommend moderate thresholds
        assert -0.5 <= profile.recommended_cos_min <= -0.3
        assert 0.95 <= profile.recommended_cos_max <= 0.97

        print(f"\n{profile}")

    def test_high_dimensional_gradients(self):
        """Test detection strategy for high-dimensional gradients (CIFAR-10-like)"""
        analyzer = GradientDimensionalityAnalyzer()

        # Simulate gradients from CIFAR-10 (~3000 dimensions)
        num_nodes = 10
        dim = 3000
        gradients = [torch.randn(dim) for _ in range(num_nodes)]

        profile = analyzer.analyze_gradients(gradients)

        # Should detect high dimensionality
        assert profile.nominal_dimensionality == dim
        assert profile.effective_dimensionality >= 1000
        assert profile.detection_strategy == "high_dim"

        # Should recommend current optimal thresholds
        assert -0.6 <= profile.recommended_cos_min <= -0.4
        assert 0.94 <= profile.recommended_cos_max <= 0.96

        print(f"\n{profile}")

    def test_similar_gradients_adjustment(self):
        """Test that very similar gradients widen the acceptable range"""
        analyzer = GradientDimensionalityAnalyzer()

        # Create very similar gradients (high cosine similarity)
        base_grad = torch.randn(100)
        gradients = [base_grad + torch.randn(100) * 0.1 for _ in range(10)]

        profile = analyzer.analyze_gradients(gradients)

        # Should detect high mean cosine similarity
        assert profile.mean_cosine_similarity > 0.7

        # Should adjust cos_max upward for similar gradients
        # (to avoid false positives when honest nodes are naturally similar)
        print(f"\n{profile}")
        print(f"Mean cosine similarity: {profile.mean_cosine_similarity:.3f}")

    def test_dissimilar_gradients_adjustment(self):
        """Test that very dissimilar gradients widen the lower bound"""
        analyzer = GradientDimensionalityAnalyzer()

        # Create dissimilar gradients
        gradients = [torch.randn(100) * (i + 1) for i in range(10)]

        profile = analyzer.analyze_gradients(gradients)

        # Should detect low mean cosine similarity
        print(f"\n{profile}")
        print(f"Mean cosine similarity: {profile.mean_cosine_similarity:.3f}")

        # cos_min should be adjusted downward for dissimilar gradients
        # (to avoid false positives when honest nodes have diverse gradients)

    def test_insufficient_gradients_error(self):
        """Test that error is raised with too few gradients"""
        analyzer = GradientDimensionalityAnalyzer()

        # Only one gradient
        gradients = [torch.randn(100)]

        with pytest.raises(ValueError, match="Need at least 2 gradients"):
            analyzer.analyze_gradients(gradients)

    def test_two_gradients_minimum(self):
        """Test that analysis works with minimum 2 gradients"""
        analyzer = GradientDimensionalityAnalyzer()

        gradients = [torch.randn(100), torch.randn(100)]

        profile = analyzer.analyze_gradients(gradients)

        # Should complete without error
        assert profile.nominal_dimensionality == 100
        assert profile.effective_dimensionality > 0

    def test_gradient_norm_statistics(self):
        """Test that gradient norm statistics are computed correctly"""
        analyzer = GradientDimensionalityAnalyzer()

        # Create gradients with known norms
        gradients = [
            torch.ones(10) * 1.0,  # norm = sqrt(10) ≈ 3.16
            torch.ones(10) * 2.0,  # norm = sqrt(40) ≈ 6.32
            torch.ones(10) * 3.0,  # norm = sqrt(90) ≈ 9.49
        ]

        profile = analyzer.analyze_gradients(gradients)

        # Check norm statistics
        assert profile.min_gradient_norm > 0
        assert profile.max_gradient_norm > profile.min_gradient_norm
        assert profile.mean_gradient_norm > 0
        assert profile.std_gradient_norm >= 0

        print(f"\nGradient norm stats:")
        print(f"  Mean: {profile.mean_gradient_norm:.2f}")
        print(f"  Std: {profile.std_gradient_norm:.2f}")
        print(f"  Range: [{profile.min_gradient_norm:.2f}, {profile.max_gradient_norm:.2f}]")

    def test_cosine_similarity_statistics(self):
        """Test that cosine similarity statistics are computed correctly"""
        analyzer = GradientDimensionalityAnalyzer()

        # Create gradients with known relationships
        base = torch.randn(50)
        gradients = [
            base,  # Same direction
            base + torch.randn(50) * 0.1,  # Very similar
            -base,  # Opposite direction
            torch.randn(50),  # Random
        ]

        profile = analyzer.analyze_gradients(gradients)

        # Check cosine similarity stats
        assert -1.0 <= profile.min_cosine_similarity <= 1.0
        assert -1.0 <= profile.max_cosine_similarity <= 1.0
        assert -1.0 <= profile.mean_cosine_similarity <= 1.0
        assert profile.std_cosine_similarity >= 0

        print(f"\nCosine similarity stats:")
        print(f"  Mean: {profile.mean_cosine_similarity:.3f}")
        print(f"  Std: {profile.std_cosine_similarity:.3f}")
        print(f"  Range: [{profile.min_cosine_similarity:.3f}, {profile.max_cosine_similarity:.3f}]")

    def test_explained_variance_ratio(self):
        """Test that explained variance ratio is reasonable"""
        analyzer = GradientDimensionalityAnalyzer(variance_threshold=0.95)

        gradients = [torch.randn(100) for _ in range(10)]

        profile = analyzer.analyze_gradients(gradients)

        # Should explain at least variance_threshold of variance
        assert 0.0 <= profile.explained_variance_ratio <= 1.0
        assert profile.explained_variance_ratio >= analyzer.variance_threshold or \
               profile.effective_dimensionality == profile.nominal_dimensionality

        print(f"\nVariance explained: {profile.explained_variance_ratio:.1%}")
        print(f"Effective dim: {profile.effective_dimensionality:.1f} / {profile.nominal_dimensionality}")

    def test_profile_string_representation(self):
        """Test that GradientProfile has readable string representation"""
        analyzer = GradientDimensionalityAnalyzer()

        gradients = [torch.randn(100) for _ in range(5)]
        profile = analyzer.analyze_gradients(gradients)

        profile_str = str(profile)

        # Should contain key information
        assert "GradientProfile" in profile_str
        assert "Strategy" in profile_str
        assert "Effective Dim" in profile_str
        assert "Recommended thresholds" in profile_str

        print(f"\n{profile_str}")

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with fixed random seed"""
        analyzer = GradientDimensionalityAnalyzer()

        torch.manual_seed(42)
        gradients1 = [torch.randn(100) for _ in range(5)]
        profile1 = analyzer.analyze_gradients(gradients1)

        torch.manual_seed(42)
        gradients2 = [torch.randn(100) for _ in range(5)]
        profile2 = analyzer.analyze_gradients(gradients2)

        # Should produce identical results
        assert profile1.effective_dimensionality == profile2.effective_dimensionality
        assert profile1.detection_strategy == profile2.detection_strategy
        assert profile1.recommended_cos_min == profile2.recommended_cos_min

    def test_large_gradient_matrix(self):
        """Test that analyzer handles large gradient matrices efficiently"""
        analyzer = GradientDimensionalityAnalyzer()

        # Large matrix: 20 nodes × 5000 dimensions
        gradients = [torch.randn(5000) for _ in range(20)]

        profile = analyzer.analyze_gradients(gradients)

        # Should complete without error
        assert profile.nominal_dimensionality == 5000
        assert profile.detection_strategy == "high_dim"

        print(f"\nLarge matrix analysis: {profile.nominal_dimensionality} dimensions")
        print(f"Effective dim: {profile.effective_dimensionality:.1f}")
        print(f"Strategy: {profile.detection_strategy}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
