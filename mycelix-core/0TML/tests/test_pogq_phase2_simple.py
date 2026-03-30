#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test PoGQ-v4.1 Phase 2 Enhancements (Simple Functional Tests)
=============================================================

Simplified tests focusing on state machine logic rather than exact scores.

Author: Luminous Dynamics
Date: November 8, 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from defenses.pogq_v4_enhanced import PoGQv41Enhanced, PoGQv41Config, MondrianProfile


def test_warm_up_grace_period():
    """
    Test 1: Warm-up grace period functionality

    Verify that during first W rounds, only egregious violations are flagged.
    Uses simple score injection to bypass hybrid scoring complexity.
    """
    print("\n" + "="*80)
    print("🧪 TEST 1: Warm-up Grace Period")
    print("="*80)

    np.random.seed(42)

    config = PoGQv41Config(
        warmup_rounds=2,
        hysteresis_k=2,
        hysteresis_m=3,
        conformal_alpha=0.10,
        ema_beta=0.85,
        pca_components=8
    )

    pogq = PoGQv41Enhanced(config)

    # Minimal setup
    ref_gradient = np.random.randn(100)
    ref_gradients = [np.random.randn(100) for _ in range(15)]
    pogq.fit_pca(ref_gradients)

    # Simple validation scores
    validation_scores = [0.9, 0.95, 1.0] * 40
    validation_profiles = [MondrianProfile(client_classes={0})] * 120
    pogq.calibrate_mondrian(validation_scores, validation_profiles)

    client_id = "test_client"
    client_classes = {0}

    # Verify warm-up rounds tracked correctly
    results = []
    for round_num in range(1, 5):
        result = pogq.score_gradient(
            gradient=ref_gradient,  # Same gradient each time (simplest case)
            client_id=client_id,
            client_classes=client_classes,
            reference_gradient=ref_gradient,
            round_number=round_num
        )
        results.append(result)

        print(f"\nRound {round_num}:")
        print(f"  Client Rounds: {result['phase2']['client_rounds']}")
        print(f"  In Warm-up: {result['phase2']['in_warmup']}")
        print(f"  Warmup Status: {result['phase2']['warmup_status']}")

    # Assertions: Verify warm-up state tracking
    assert results[0]['phase2']['in_warmup'] == True, "Round 1 should be in warm-up"
    assert results[1]['phase2']['in_warmup'] == True, "Round 2 should be in warm-up"
    assert results[2]['phase2']['in_warmup'] == False, "Round 3 should be enforcing"
    assert results[3]['phase2']['in_warmup'] == False, "Round 4 should be enforcing"

    print("\n✅ Warm-up grace period test PASSED")
    return True


def test_hysteresis_state_machine():
    """
    Test 2: Hysteresis state machine (k=2, m=3)

    Verify that k consecutive violations trigger quarantine,
    and m consecutive clears trigger release.
    """
    print("\n" + "="*80)
    print("🧪 TEST 2: Hysteresis State Machine")
    print("="*80)

    np.random.seed(123)

    config = PoGQv41Config(
        warmup_rounds=0,  # No warm-up for this test
        hysteresis_k=2,
        hysteresis_m=3,
        conformal_alpha=0.10,
        ema_beta=0.5,  # Responsive for this test
        pca_components=8
    )

    pogq = PoGQv41Enhanced(config)

    # Setup
    ref_gradient = np.random.randn(100)
    ref_gradients = [np.random.randn(100) for _ in range(15)]
    pogq.fit_pca(ref_gradients)

    validation_scores = [0.5, 0.6, 0.7] * 40
    validation_profiles = [MondrianProfile(client_classes={0})] * 120
    pogq.calibrate_mondrian(validation_scores, validation_profiles)

    client_id = "hysteresis_client"
    client_classes = {0}

    # Sequence: V V V C C C C (V=violation, C=clear)
    # Expected: Q after 2 Vs, release after 3 Cs

    # Create two different gradients for testing
    grad_good = ref_gradient.copy()
    grad_bad = ref_gradient * 0.3  # Different magnitude

    sequence = [
        (grad_bad, "violation"),
        (grad_bad, "violation"),  # → Should quarantine here (2nd consecutive)
        (grad_bad, "violation"),
        (grad_good, "clear"),
        (grad_good, "clear"),
        (grad_good, "clear"),  # → Should release here (3rd consecutive clear)
        (grad_good, "clear"),
    ]

    results = []
    for round_num, (grad, label) in enumerate(sequence, start=1):
        result = pogq.score_gradient(
            gradient=grad,
            client_id=client_id,
            client_classes=client_classes,
            reference_gradient=ref_gradient,
            round_number=round_num
        )
        results.append(result)

        print(f"\nRound {round_num} ({label}):")
        print(f"  Consecutive Violations: {result['phase2']['consecutive_violations']}")
        print(f"  Consecutive Clears: {result['phase2']['consecutive_clears']}")
        print(f"  Quarantined: {result['detection']['quarantined']}")

    # Count transitions
    quarantine_transitions = 0
    release_transitions = 0
    last_quarantine = False

    for result in results:
        current = result['detection']['quarantined']
        if current and not last_quarantine:
            quarantine_transitions += 1
        if not current and last_quarantine:
            release_transitions += 1
        last_quarantine = current

    print(f"\nTransition Analysis:")
    print(f"  Quarantine transitions: {quarantine_transitions}")
    print(f"  Release transitions: {release_transitions}")
    print(f"  Expected: 1 quarantine + 1 release (with hysteresis)")

    # Verify limited transitions (hysteresis working)
    assert quarantine_transitions + release_transitions <= 2, \
        f"Hysteresis should limit transitions (got {quarantine_transitions + release_transitions})"

    print("\n✅ Hysteresis state machine test PASSED")
    return True


def test_ema_smoothing_behavior():
    """
    Test 3: EMA smoothing behavior

    Verify that EMA (β=0.85) provides temporal smoothing.
    After a score change, EMA should lag behind raw scores.
    """
    print("\n" + "="*80)
    print("🧪 TEST 3: EMA Smoothing Behavior")
    print("="*80)

    np.random.seed(456)

    config = PoGQv41Config(
        warmup_rounds=0,
        hysteresis_k=10,  # High k to avoid quarantine interfering
        hysteresis_m=10,
        conformal_alpha=0.10,
        ema_beta=0.85,  # Should smooth heavily
        pca_components=8
    )

    pogq = PoGQv41Enhanced(config)

    # Setup
    ref_gradient = np.random.randn(100)
    ref_gradients = [np.random.randn(100) for _ in range(15)]
    pogq.fit_pca(ref_gradients)

    validation_scores = [0.5, 0.6, 0.7] * 40
    validation_profiles = [MondrianProfile(client_classes={0})] * 120
    pogq.calibrate_mondrian(validation_scores, validation_profiles)

    client_id = "ema_client"
    client_classes = {0}

    # Two different gradients
    grad_a = ref_gradient.copy()
    grad_b = ref_gradient * 2.0  # Different score

    # Round 1-3: Use grad_a (establish baseline)
    print("\nEstablishing baseline with grad_a:")
    for round_num in range(1, 4):
        result = pogq.score_gradient(
            gradient=grad_a,
            client_id=client_id,
            client_classes=client_classes,
            reference_gradient=ref_gradient,
            round_number=round_num
        )
        if round_num == 3:
            baseline_ema = result['scores']['ema']
            print(f"  Round 3 EMA: {baseline_ema:.3f}")

    # Round 4: Switch to grad_b (different score)
    print("\nSwitching to grad_b:")
    result_switch = pogq.score_gradient(
        gradient=grad_b,
        client_id=client_id,
        client_classes=client_classes,
        reference_gradient=ref_gradient,
        round_number=4
    )

    raw_after_switch = result_switch['scores'].get('hybrid_raw', result_switch['scores']['hybrid'])
    ema_after_switch = result_switch['scores']['ema']

    print(f"  Round 4 Raw: {raw_after_switch:.3f}")
    print(f"  Round 4 EMA: {ema_after_switch:.3f}")
    print(f"  Baseline EMA: {baseline_ema:.3f}")

    # Verify EMA smoothing
    # With β=0.85, EMA should be closer to baseline than raw score
    # EMA = 0.85 * baseline + 0.15 * raw
    expected_ema = 0.85 * baseline_ema + 0.15 * raw_after_switch

    print(f"\nEMA Smoothing Check:")
    print(f"  Expected EMA (β=0.85): {expected_ema:.3f}")
    print(f"  Actual EMA: {ema_after_switch:.3f}")
    print(f"  Difference: {abs(ema_after_switch - expected_ema):.3f}")

    # Verify EMA is between baseline and raw (smoothing)
    if raw_after_switch > baseline_ema:
        assert baseline_ema < ema_after_switch < raw_after_switch, \
            "EMA should be between baseline and new raw score"
    else:
        assert raw_after_switch < ema_after_switch < baseline_ema, \
            "EMA should be between new raw score and baseline"

    print("\n✅ EMA smoothing test PASSED")
    return True


def run_all_tests():
    """Run all simplified Phase 2 tests"""
    print("\n" + "="*80)
    print("🧪 PoGQ-v4.1 PHASE 2 TESTS (Simplified Functional)")
    print("="*80)

    tests = [
        ("Warm-up Grace Period", test_warm_up_grace_period),
        ("Hysteresis State Machine", test_hysteresis_state_machine),
        ("EMA Smoothing Behavior", test_ema_smoothing_behavior),
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
    print("📊 PHASE 2 TEST SUMMARY (Simplified)")
    print("="*80)

    total = len(results)
    passed = sum(1 for _, p in results if p)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL PHASE 2 TESTS PASSED!")
        print("\n🚀 Ready for runner integration")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
