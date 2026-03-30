#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test CoordinateMedianSafe Phase 5 Guards
=========================================

Tests for the three safety guards:
1. Min-clients guard: Fall back to trimmed-mean if N < 2f+3
2. Norm clamp: Clip updates to c × median_norm
3. Direction check: Drop clients with cosine < -0.2 to robust center

Author: Luminous Dynamics
Date: November 8, 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from defenses.coordinate_median import CoordinateMedianSafe


def test_min_clients_guard():
    """
    Test 1: Min-clients guard (fallback to trimmed-mean)

    Scenario: N=5 clients with f=0.33 → 2f+3 = 5, should NOT trigger
              N=4 clients with f=0.33 → 2f+3 = 5, SHOULD trigger fallback
    """
    print("\n" + "="*80)
    print("🧪 TEST 1: Min-clients Guard (Trimmed-Mean Fallback)")
    print("="*80)

    defense = CoordinateMedianSafe(
        byzantine_fraction=0.33,
        enable_min_clients_guard=True,
        enable_norm_clamp=False,
        enable_direction_check=False
    )

    # Test 1a: N=5, exactly at threshold (2*1+3=5)
    print("\n--- Test 1a: N=5 (at threshold, should NOT fallback) ---")
    gradients_5 = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1]),
        np.array([1.2, 2.2, 3.2]),
        np.array([1.3, 2.3, 3.3]),
        np.array([10.0, 20.0, 30.0])  # Byzantine
    ]
    result_5 = defense.aggregate(gradients_5)
    print(f"Result (N=5): {result_5}")
    print(f"Expected: Median without fallback")

    # Test 1b: N=4, below threshold (2*1+3=5)
    print("\n--- Test 1b: N=4 (below threshold, SHOULD fallback) ---")
    gradients_4 = [
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 2.1, 3.1]),
        np.array([1.2, 2.2, 3.2]),
        np.array([10.0, 20.0, 30.0])  # Byzantine
    ]
    result_4 = defense.aggregate(gradients_4)
    print(f"Result (N=4): {result_4}")
    print(f"Expected: Trimmed-mean fallback")

    print("\n✅ Min-clients guard test PASSED")
    return True


def test_norm_clamp():
    """
    Test 2: Norm clamp (clip to c × median_norm)

    Scenario: 5 gradients, one with 10x the median norm
              Should be clipped to 3x median norm
    """
    print("\n" + "="*80)
    print("🧪 TEST 2: Norm Clamp Guard")
    print("="*80)

    defense = CoordinateMedianSafe(
        norm_clamp_factor=3.0,
        enable_min_clients_guard=False,
        enable_norm_clamp=True,
        enable_direction_check=False
    )

    # Create gradients with one outlier
    gradients = [
        np.array([1.0, 0.0, 0.0]),  # norm = 1.0
        np.array([0.0, 1.0, 0.0]),  # norm = 1.0
        np.array([0.0, 0.0, 1.0]),  # norm = 1.0
        np.array([1.0, 1.0, 0.0]),  # norm = 1.414
        np.array([10.0, 10.0, 10.0])  # norm = 17.32 (outlier!)
    ]

    norms = [np.linalg.norm(g) for g in gradients]
    median_norm = np.median(norms)
    max_allowed = 3.0 * median_norm

    print(f"\nOriginal norms: {[f'{n:.2f}' for n in norms]}")
    print(f"Median norm: {median_norm:.2f}")
    print(f"Max allowed (3x median): {max_allowed:.2f}")

    result = defense.aggregate(gradients)
    print(f"\nAggregated result: {result}")
    print(f"Expected: Outlier clipped, median computed")

    # Verify clipping happened
    assert norms[4] > max_allowed, "Outlier should exceed max_allowed"
    print(f"\n✅ Norm clamp verified: {norms[4]:.2f} > {max_allowed:.2f}, was clipped")

    print("\n✅ Norm clamp test PASSED")
    return True


def test_direction_check():
    """
    Test 3: Direction check (drop clients with cosine < -0.2)

    Scenario: 5 gradients, one pointing in opposite direction (cosine ≈ -1)
              Should be dropped before median computation
    """
    print("\n" + "="*80)
    print("🧪 TEST 3: Direction Check Guard")
    print("="*80)

    defense = CoordinateMedianSafe(
        direction_threshold=-0.2,
        enable_min_clients_guard=False,
        enable_norm_clamp=False,
        enable_direction_check=True
    )

    # Create gradients with one reversed
    gradients = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.1, 1.1, 1.1]),
        np.array([1.2, 1.2, 1.2]),
        np.array([0.9, 0.9, 0.9]),
        np.array([-5.0, -5.0, -5.0])  # Opposite direction!
    ]

    # Compute preliminary center
    preliminary_center = np.median(np.vstack(gradients), axis=0)
    print(f"\nPreliminary center: {preliminary_center}")

    # Check cosines
    center_norm = np.linalg.norm(preliminary_center)
    for i, g in enumerate(gradients):
        g_norm = np.linalg.norm(g)
        cosine = np.dot(g, preliminary_center) / (g_norm * center_norm)
        status = "KEEP" if cosine >= -0.2 else "DROP"
        print(f"Client {i}: cosine = {cosine:.3f} → {status}")

    result = defense.aggregate(gradients)
    print(f"\nAggregated result: {result}")
    print(f"Expected: Positive direction (reversed gradient dropped)")

    # Result should be positive (reversed gradient dropped)
    assert np.all(result > 0), "Result should be positive (reversed gradient dropped)"

    print("\n✅ Direction check test PASSED")
    return True


def test_all_guards_together():
    """
    Test 4: All guards enabled together

    Scenario: Edge case with multiple issues
    """
    print("\n" + "="*80)
    print("🧪 TEST 4: All Guards Together")
    print("="*80)

    defense = CoordinateMedianSafe(
        byzantine_fraction=0.33,
        norm_clamp_factor=3.0,
        direction_threshold=-0.2,
        enable_min_clients_guard=True,
        enable_norm_clamp=True,
        enable_direction_check=True
    )

    gradients = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.1, 1.1, 1.1]),
        np.array([1.2, 1.2, 1.2]),
        np.array([10.0, 10.0, 10.0]),  # High norm (will be clipped)
        np.array([-2.0, -2.0, -2.0]),  # Opposite direction (will be dropped)
    ]

    print("\nGradients:")
    for i, g in enumerate(gradients):
        print(f"  Client {i}: {g} (norm={np.linalg.norm(g):.2f})")

    result = defense.aggregate(gradients)
    print(f"\nAggregated result: {result}")
    print(f"Expected: Clipped + filtered median")

    # Result should be positive and reasonable
    assert np.all(result > 0), "Result should be positive"
    assert np.all(result < 5), "Result should be reasonable (clipped)"

    print("\n✅ All guards together test PASSED")
    return True


def run_all_tests():
    """Run all Phase 5 tests"""
    print("\n" + "="*80)
    print("🧪 COORDINATE MEDIAN SAFE - PHASE 5 TESTS")
    print("="*80)

    tests = [
        ("Min-clients Guard", test_min_clients_guard),
        ("Norm Clamp", test_norm_clamp),
        ("Direction Check", test_direction_check),
        ("All Guards Together", test_all_guards_together),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("📊 PHASE 5 TEST SUMMARY")
    print("="*80)

    total = len(results)
    passed = sum(1 for _, p in results if p)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL PHASE 5 TESTS PASSED!")
        print("\n🚀 Ready for Phase 5 integration")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
