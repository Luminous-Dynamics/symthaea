# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Simple test script for GradientDimensionalityAnalyzer
"""

import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from byzantine_detection import GradientDimensionalityAnalyzer


def test_low_dim():
    """Test low-dimensional gradients (like breast cancer)"""
    print("\n" + "=" * 60)
    print("TEST 1: Low-Dimensional Gradients (Breast Cancer-like)")
    print("=" * 60)

    analyzer = GradientDimensionalityAnalyzer()

    # Simulate gradients from breast cancer (30 features)
    num_nodes = 10
    dim = 30
    gradients = [torch.randn(dim) for _ in range(num_nodes)]

    profile = analyzer.analyze_gradients(gradients)

    print(profile)

    assert profile.nominal_dimensionality == dim
    assert profile.detection_strategy == "low_dim", f"Expected low_dim, got {profile.detection_strategy}"
    assert profile.recommended_cos_min <= -0.5, f"Expected wide threshold, got {profile.recommended_cos_min}"

    print("✅ PASSED")


def test_mid_dim():
    """Test mid-dimensional gradients (like EMNIST)"""
    print("\n" + "=" * 60)
    print("TEST 2: Mid-Dimensional Gradients (EMNIST-like)")
    print("=" * 60)

    analyzer = GradientDimensionalityAnalyzer()

    # Simulate gradients from EMNIST (~800 dimensions)
    num_nodes = 10
    dim = 800
    gradients = [torch.randn(dim) for _ in range(num_nodes)]

    profile = analyzer.analyze_gradients(gradients)

    print(profile)

    assert profile.nominal_dimensionality == dim
    assert profile.detection_strategy == "mid_dim", f"Expected mid_dim, got {profile.detection_strategy}"

    print("✅ PASSED")


def test_high_dim():
    """Test high-dimensional gradients (like CIFAR-10)"""
    print("\n" + "=" * 60)
    print("TEST 3: High-Dimensional Gradients (CIFAR-10-like)")
    print("=" * 60)

    analyzer = GradientDimensionalityAnalyzer()

    # Simulate gradients from CIFAR-10 (~3000 dimensions)
    num_nodes = 10
    dim = 3000
    gradients = [torch.randn(dim) for _ in range(num_nodes)]

    profile = analyzer.analyze_gradients(gradients)

    print(profile)

    assert profile.nominal_dimensionality == dim
    assert profile.detection_strategy == "high_dim", f"Expected high_dim, got {profile.detection_strategy}"
    assert -0.6 <= profile.recommended_cos_min <= -0.4

    print("✅ PASSED")


def test_similar_gradients():
    """Test very similar gradients (high cosine similarity)"""
    print("\n" + "=" * 60)
    print("TEST 4: Very Similar Gradients")
    print("=" * 60)

    analyzer = GradientDimensionalityAnalyzer()

    # Create very similar gradients
    base_grad = torch.randn(100)
    gradients = [base_grad + torch.randn(100) * 0.05 for _ in range(10)]

    profile = analyzer.analyze_gradients(gradients)

    print(profile)
    print(f"Mean cosine similarity: {profile.mean_cosine_similarity:.3f}")

    assert profile.mean_cosine_similarity > 0.7, "Expected high cosine similarity"

    print("✅ PASSED")


def main():
    print("\n🧪 Testing GradientDimensionalityAnalyzer\n")

    try:
        test_low_dim()
        test_mid_dim()
        test_high_dim()
        test_similar_gradients()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
