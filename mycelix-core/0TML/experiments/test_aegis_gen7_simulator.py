#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Quick Test: AEGIS+Gen7 Integration in Simulator
================================================

Verifies that the aegis_gen7 aggregator works correctly in the simulator.

Expected Results:
- Gen7 Layer 0 achieves perfect Byzantine detection (AUC 1.00)
- All Byzantine clients rejected by cryptographic verification
- System maintains >80% accuracy even at 36% Byzantine (should work where AEGIS fails)
"""

import sys
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.simulator import run_fl, FLScenario
from experiments.datasets.emnist import load_emnist_subset

# Try importing Gen7
try:
    import gen7_zkstark
    print("✅ Gen7 module loaded successfully")
except ImportError as e:
    print(f"❌ Gen7 not available: {e}")
    print("Please install Gen7 wheel first.")
    sys.exit(1)


def test_aegis_gen7_36_percent():
    """Test aegis_gen7 at 36% Byzantine (where vanilla AEGIS fails)."""
    print("\n" + "="*80)
    print("TEST: AEGIS+Gen7 at 36% Byzantine (vanilla AEGIS fails here)")
    print("="*80)

    # Load EMNIST dataset
    print("\n📊 Loading EMNIST subset...")
    fl_data = load_emnist_subset(n_train=6000, n_test=1000)

    # Create scenario with 36% Byzantine (catastrophic for vanilla AEGIS)
    scenario = FLScenario(
        n_clients=50,
        byz_frac=0.36,  # 18/50 Byzantine clients
        noniid_alpha=1.0,  # IID for simplicity
        attack="sign_flip",
        seed=101,
    )

    # Run with aegis_gen7
    print(f"\n🚀 Running FL with aegis_gen7 aggregator...")
    print(f"   Byzantine: {int(scenario.byz_frac * scenario.n_clients)}/50 (36%)")
    print(f"   Attack: {scenario.attack}")

    result = run_fl(
        scenario=scenario,
        fl_data=fl_data,
        aggregator="aegis_gen7",
        n_rounds=10,
    )

    # Print results
    print(f"\n📈 Results:")
    print(f"   Final Accuracy (AEGIS+Gen7): {result['clean_acc']:.1%}")
    print(f"   AUC (Gen7 Layer 0): {result.get('auc', 0.0):.4f}")
    print(f"   Wall Time: {result['wall_time']:.2f}s")

    # Validation
    if result['clean_acc'] > 0.80:
        print(f"\n✅ TEST PASSED: {result['clean_acc']:.1%} > 80% (system works at 36%!)")
        return True
    else:
        print(f"\n❌ TEST FAILED: {result['clean_acc']:.1%} ≤ 80%")
        return False


def test_aegis_gen7_45_percent():
    """Test aegis_gen7 at 45% Byzantine (target claim)."""
    print("\n" + "="*80)
    print("TEST: AEGIS+Gen7 at 45% Byzantine (target claim)")
    print("="*80)

    # Load EMNIST dataset
    print("\n📊 Loading EMNIST subset...")
    fl_data = load_emnist_subset(n_train=6000, n_test=1000)

    # Create scenario with 45% Byzantine
    scenario = FLScenario(
        n_clients=50,
        byz_frac=0.45,  # 22/50 Byzantine clients (rounding down = 22)
        noniid_alpha=1.0,  # IID
        attack="sign_flip",
        seed=101,
    )

    # Run with aegis_gen7
    print(f"\n🚀 Running FL with aegis_gen7 aggregator...")
    print(f"   Byzantine: {int(scenario.byz_frac * scenario.n_clients)}/50 (45%)")
    print(f"   Attack: {scenario.attack}")

    result = run_fl(
        scenario=scenario,
        fl_data=fl_data,
        aggregator="aegis_gen7",
        n_rounds=10,
    )

    # Print results
    print(f"\n📈 Results:")
    print(f"   Final Accuracy (AEGIS+Gen7): {result['clean_acc']:.1%}")
    print(f"   AUC (Gen7 Layer 0): {result.get('auc', 0.0):.4f}")
    print(f"   Wall Time: {result['wall_time']:.2f}s")

    # Validation (target: >80% accuracy at 45%)
    if result['clean_acc'] > 0.80:
        print(f"\n✅ TEST PASSED: {result['clean_acc']:.1%} > 80% (45% BFT achieved!)")
        return True
    else:
        print(f"\n⚠️  TEST RESULT: {result['clean_acc']:.1%} ≤ 80% at 45%")
        print(f"    (May need zkSTARK proofs, not just Dilithium signatures)")
        return False


def main():
    """Run quick Gen7+AEGIS integration tests."""
    print("\n" + "="*80)
    print("AEGIS+Gen7 Simulator Integration Test Suite")
    print("="*80)

    passed = []

    # Test 1: 36% Byzantine (where vanilla AEGIS fails)
    try:
        passed.append(test_aegis_gen7_36_percent())
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        passed.append(False)

    # Test 2: 45% Byzantine (target claim)
    try:
        passed.append(test_aegis_gen7_45_percent())
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        passed.append(False)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Tests Passed: {sum(passed)}/{len(passed)}")

    if all(passed):
        print("\n🎉 ALL TESTS PASSED! Gen7 integration working correctly.")
        print("\nNext Steps:")
        print("  1. Run full BFT sweep (35-45% in 1% steps)")
        print("  2. Compare AUC metrics: AEGIS vs AEGIS+Gen7")
        print("  3. Measure performance overhead (target <100ms/client)")
        print("  4. Update paper with empirical 45% BFT validation")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
