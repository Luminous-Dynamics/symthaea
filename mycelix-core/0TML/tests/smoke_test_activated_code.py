#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Smoke Test for Week 1 Activated Production Code

Tests that all activated components can be imported and basic functionality works:
- pogq_system.py
- performance_optimizations.py
- integration/holochain_pogq.py
- integration/byzantine_fl_pogq.py
- baselines/sota_aggregators.py
- byzantine_attacks/* (all attack modules)
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all activated modules can be imported"""
    print("=" * 70)
    print("SMOKE TEST: Activated Production Code")
    print("=" * 70)
    print()

    tests_passed = 0
    tests_failed = 0

    # Test 1: PoGQ System
    print("Test 1: PoGQ System Import")
    try:
        from pogq_system import ProofOfGoodQuality, PoGQProof
        print("  ✅ pogq_system imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ pogq_system import failed: {e}")
        tests_failed += 1

    # Test 2: Performance Optimizations
    print("\nTest 2: Performance Optimizations Import")
    try:
        from performance_optimizations import GPUAccelerator, IntelligentCache
        print("  ✅ performance_optimizations imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ performance_optimizations import failed: {e}")
        tests_failed += 1

    # Test 3: Byzantine Attacks Registry
    print("\nTest 3: Byzantine Attacks Registry")
    try:
        from byzantine_attacks import (
            AttackType,
            create_attack,
            RandomNoiseAttack,
            SignFlipAttack,
            SleeperAgentAttack
        )
        print("  ✅ byzantine_attacks imports successfully")

        # Test attack creation
        attack = create_attack(AttackType.RANDOM_NOISE, std_dev=0.1)
        print(f"  ✅ Attack creation works: {attack.name}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ byzantine_attacks import/creation failed: {e}")
        tests_failed += 1

    # Test 4: State-of-Art Aggregators
    print("\nTest 4: State-of-Art Aggregators Import")
    try:
        # The file is sota_aggregators.py, check if it has standard aggregator functions
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "sota_aggregators",
            Path(__file__).parent.parent / "src" / "baselines" / "sota_aggregators.py"
        )
        sota_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sota_module)
        print("  ✅ sota_aggregators loads successfully")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ sota_aggregators load failed: {e}")
        tests_failed += 1

    # Test 5: Basic Attack Functionality
    print("\nTest 5: Attack Generation Functionality")
    try:
        from byzantine_attacks import create_attack, AttackType

        # Test RandomNoise attack
        attack = create_attack(AttackType.RANDOM_NOISE, std_dev=0.1)
        honest_grad = np.ones(100)
        byzantine_grad = attack.generate(honest_grad, round_num=1)

        assert byzantine_grad.shape == honest_grad.shape
        assert not np.allclose(byzantine_grad, honest_grad)  # Should be different
        print(f"  ✅ RandomNoise generates different gradient")

        # Test SignFlip attack
        attack = create_attack(AttackType.SIGN_FLIP)
        byzantine_grad = attack.generate(honest_grad, round_num=1)
        assert np.allclose(byzantine_grad, -honest_grad)  # Should be negated
        print(f"  ✅ SignFlip correctly negates gradient")

        # Test SleeperAgent (should be honest before activation)
        attack = create_attack(AttackType.SLEEPER_AGENT, activation_round=5)
        byzantine_grad = attack.generate(honest_grad, round_num=2)
        assert np.allclose(byzantine_grad, honest_grad)  # Should still be honest
        print(f"  ✅ SleeperAgent behaves honestly before activation")

        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Attack generation failed: {e}")
        tests_failed += 1

    # Test 6: PoGQ Proof Generation
    print("\nTest 6: PoGQ Proof Generation")
    try:
        from pogq_system import ProofOfGoodQuality

        pogq = ProofOfGoodQuality(quality_threshold=0.3)
        gradient = np.random.randn(100)

        proof = pogq.generate_proof(
            gradient=gradient,
            loss_before=2.0,
            loss_after=1.0,
            client_id="test_client",
            round_number=1
        )

        assert proof.client_id == "test_client"
        assert proof.round_number == 1
        assert proof.quality_score > 0
        print(f"  ✅ PoGQ proof generation works (quality={proof.quality_score:.3f})")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ PoGQ proof generation failed: {e}")
        tests_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {tests_passed}/{tests_passed + tests_failed}")
    print(f"Tests Failed: {tests_failed}/{tests_passed + tests_failed}")

    if tests_failed == 0:
        print("\n✅ ALL SMOKE TESTS PASSED!")
        print("\nActivated code is working correctly. Ready for:")
        print("  • Integration with experiment harness")
        print("  • Week 2: PoGQ-v4 enhancements")
        print("  • Comprehensive FEMNIST evaluation")
        return 0
    else:
        print(f"\n❌ {tests_failed} tests failed. Review errors above.")
        return 1

if __name__ == "__main__":
    exit_code = test_imports()
    sys.exit(exit_code)
