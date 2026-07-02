#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Defense Registry and All Baseline Defenses
================================================

Verification test for Week 2 implementation:
- All 12 defenses can be instantiated
- Registry factory function works
- Basic aggregation works on dummy data
- Explain() returns provenance

Author: Luminous Dynamics
Date: November 8, 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from defenses import (
    get_defense,
    list_defenses,
    DEFENSE_REGISTRY,
    PoGQv41Enhanced,
    PoGQv41Config,
    FLTrust,
    RobustFederatedAveraging,
    CoordinateMedian,
    TrimmedMean,
    Krum,
    MultiKrum,
    Bulyan,
    FedGuardStrict,
    FoolsGold,
    BOBA,
    FedAvg
)


def test_registry_completeness():
    """Test that registry has all expected defenses"""
    print("\n" + "="*80)
    print("🔍 TEST 1: Registry Completeness")
    print("="*80)

    expected_defenses = {
        "pogq_v4.1", "fltrust", "rfa", "coord_median", "trimmed_mean",
        "krum", "multi_krum", "bulyan", "fedguard_strict",
        "foolsgold", "boba", "fedavg"
    }

    actual_defenses = set(DEFENSE_REGISTRY.keys())

    print(f"Expected: {len(expected_defenses)} defenses")
    print(f"Found: {len(actual_defenses)} defenses")

    missing = expected_defenses - actual_defenses
    extra = actual_defenses - expected_defenses

    if missing:
        print(f"❌ Missing: {missing}")
        return False

    if extra:
        print(f"⚠️  Extra: {extra}")

    print("✅ Registry complete!")
    return True


def test_factory_function():
    """Test that factory function works for all defenses"""
    print("\n" + "="*80)
    print("🏭 TEST 2: Factory Function")
    print("="*80)

    defenses_to_test = [
        ("pogq_v4.1", {}),
        ("fltrust", {}),
        ("rfa", {}),
        ("coord_median", {}),
        ("trimmed_mean", {"trim_ratio": 0.1}),
        ("krum", {"f": 2}),
        ("multi_krum", {"f": 2, "k": 3}),
        ("bulyan", {"f": 2}),
        ("fedguard_strict", {}),
        ("foolsgold", {}),
        ("boba", {}),
        ("fedavg", {}),
    ]

    success_count = 0

    for name, config in defenses_to_test:
        try:
            defense = get_defense(name, **config)
            print(f"✅ {name:20s} - Instantiated")
            success_count += 1
        except Exception as e:
            print(f"❌ {name:20s} - FAILED: {e}")

    print(f"\n{success_count}/{len(defenses_to_test)} defenses instantiated successfully")
    return success_count == len(defenses_to_test)


def test_basic_aggregation():
    """Test that all defenses can aggregate dummy gradients"""
    print("\n" + "="*80)
    print("🔄 TEST 3: Basic Aggregation")
    print("="*80)

    # Generate dummy gradients
    np.random.seed(42)
    n_clients = 10
    gradient_dim = 100

    client_updates = [
        np.random.randn(gradient_dim).astype(np.float32)
        for _ in range(n_clients)
    ]
    client_ids = [f"client_{i}" for i in range(n_clients)]

    # Test each defense
    defenses_to_test = [
        ("rfa", {}, {}),
        ("coord_median", {}, {}),
        ("trimmed_mean", {"trim_ratio": 0.1}, {}),
        ("krum", {"f": 2}, {}),
        ("multi_krum", {"f": 2, "k": 5}, {}),
        ("bulyan", {"f": 2}, {}),
        ("foolsgold", {}, {"client_ids": client_ids}),
        ("fedavg", {}, {}),
    ]

    success_count = 0

    for name, config, context in defenses_to_test:
        try:
            defense = get_defense(name, **config)
            aggregated = defense.aggregate(client_updates, **context)

            # Verify shape
            assert aggregated.shape == (gradient_dim,), f"Wrong shape: {aggregated.shape}"
            print(f"✅ {name:20s} - Aggregated (norm={np.linalg.norm(aggregated):.3f})")
            success_count += 1

        except Exception as e:
            print(f"❌ {name:20s} - FAILED: {e}")

    print(f"\n{success_count}/{len(defenses_to_test)} defenses aggregated successfully")
    return success_count == len(defenses_to_test)


def test_fltrust_with_server_gradient():
    """Test FLTrust with server gradient"""
    print("\n" + "="*80)
    print("🎯 TEST 4: FLTrust with Server Gradient")
    print("="*80)

    np.random.seed(42)
    gradient_dim = 100

    # Server gradient
    server_gradient = np.random.randn(gradient_dim).astype(np.float32)

    # Client gradients
    client_updates = [
        server_gradient + 0.1 * np.random.randn(gradient_dim).astype(np.float32)
        for _ in range(5)
    ]

    try:
        fltrust = get_defense("fltrust")
        aggregated = fltrust.aggregate(client_updates, server_gradient=server_gradient)

        assert aggregated.shape == (gradient_dim,)
        print(f"✅ FLTrust aggregated with server gradient (norm={np.linalg.norm(aggregated):.3f})")
        return True

    except Exception as e:
        print(f"❌ FLTrust FAILED: {e}")
        return False


def test_boba_with_class_histograms():
    """Test BOBA with class histograms"""
    print("\n" + "="*80)
    print("📊 TEST 5: BOBA with Class Histograms")
    print("="*80)

    np.random.seed(42)
    gradient_dim = 100
    n_classes = 10

    # Client gradients
    client_updates = [
        np.random.randn(gradient_dim).astype(np.float32)
        for _ in range(5)
    ]

    # Class histograms (probability distributions)
    client_class_histograms = [
        np.random.dirichlet([1.0] * n_classes)
        for _ in range(5)
    ]

    try:
        boba = get_defense("boba")
        aggregated = boba.aggregate(
            client_updates,
            client_class_histograms=client_class_histograms
        )

        assert aggregated.shape == (gradient_dim,)
        print(f"✅ BOBA aggregated with class histograms (norm={np.linalg.norm(aggregated):.3f})")
        return True

    except Exception as e:
        print(f"❌ BOBA FAILED: {e}")
        return False


def test_fedguard_strict_calibration():
    """Test FedGuard-strict calibration and provenance"""
    print("\n" + "="*80)
    print("🛡️  TEST 6: FedGuard-strict Calibration & Provenance")
    print("="*80)

    np.random.seed(42)
    gradient_dim = 100

    # Clean calibration set
    clean_gradients = [
        np.random.randn(gradient_dim).astype(np.float32)
        for _ in range(50)
    ]

    try:
        fedguard = get_defense("fedguard_strict")

        # Calibrate
        fedguard.calibrate_on_clean_server_set(clean_gradients)

        # Check provenance
        provenance = fedguard.explain()
        print(f"  Extractor hash: {provenance['provenance']['extractor_hash']}")
        print(f"  Training set hash: {provenance['provenance']['training_set_hash']}")

        # Test aggregation after calibration
        client_updates = [
            np.random.randn(gradient_dim).astype(np.float32)
            for _ in range(5)
        ]
        aggregated = fedguard.aggregate(client_updates)

        assert aggregated.shape == (gradient_dim,)
        print(f"✅ FedGuard-strict calibrated and aggregated (norm={np.linalg.norm(aggregated):.3f})")
        return True

    except Exception as e:
        print(f"❌ FedGuard-strict FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pogq_v41_demonstration():
    """Test PoGQ-v4.1 basic functionality"""
    print("\n" + "="*80)
    print("🚀 TEST 7: PoGQ-v4.1 Enhanced")
    print("="*80)

    try:
        # Configuration
        config = PoGQv41Config(
            pca_components=16,  # Reduced for test
            lambda_adaptive=True,
            mondrian_conformal=True,
            conformal_alpha=0.10
        )

        pogq = PoGQv41Enhanced(config)

        # Reference gradient
        gradient_dim = 100
        reference_gradient = np.random.randn(gradient_dim).astype(np.float32)

        # Fit PCA
        reference_gradients = [
            np.random.randn(gradient_dim).astype(np.float32)
            for _ in range(20)
        ]
        pogq.fit_pca(reference_gradients)

        # Test aggregation (simplified without validation data)
        client_updates = [
            reference_gradient + 0.1 * np.random.randn(gradient_dim)
            for _ in range(5)
        ]
        client_ids = [f"client_{i}" for i in range(5)]
        client_classes = [{0, 1}] * 5

        aggregated, diagnostics = pogq.aggregate(
            client_updates,
            client_ids,
            client_classes,
            reference_gradient,
            validation_data=None,
            round_number=1
        )

        print(f"  Honest clients: {diagnostics['n_honest']}/{diagnostics['n_clients']}")
        print(f"  Byzantine: {diagnostics['n_byzantine']}")
        print(f"  Lambda stats: {diagnostics['lambda_stats']}")

        assert aggregated.shape == (gradient_dim,)
        print(f"✅ PoGQ-v4.1 aggregated (norm={np.linalg.norm(aggregated):.3f})")
        return True

    except Exception as e:
        print(f"❌ PoGQ-v4.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 DEFENSE REGISTRY VERIFICATION TESTS")
    print("="*80)

    tests = [
        ("Registry Completeness", test_registry_completeness),
        ("Factory Function", test_factory_function),
        ("Basic Aggregation", test_basic_aggregation),
        ("FLTrust Server Gradient", test_fltrust_with_server_gradient),
        ("BOBA Class Histograms", test_boba_with_class_histograms),
        ("FedGuard-strict Calibration", test_fedguard_strict_calibration),
        ("PoGQ-v4.1 Enhanced", test_pogq_v41_demonstration),
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
    print("📊 TEST SUMMARY")
    print("="*80)

    total = len(results)
    passed = sum(1 for _, p in results if p)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
