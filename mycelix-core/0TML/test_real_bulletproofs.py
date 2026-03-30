#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Real Bulletproofs Integration

Tests the RealBulletproofs class with pybulletproofs library.
"""

import sys
import os

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zkpoc import ZKPoC, RealBulletproofs


def test_real_bulletproofs():
    """Test real Bulletproofs library integration"""
    print("🔐 Testing Real Bulletproofs Integration\n")
    print("=" * 60)

    # Test 1: Check if pybulletproofs is available
    print("\n1. Checking pybulletproofs availability...")
    try:
        from pybulletproofs import zkrp_prove, zkrp_verify
        print("   ✅ pybulletproofs installed")
    except ImportError:
        pytest.skip("pybulletproofs not installed")

    # Test 2: Initialize RealBulletproofs
    print("\n2. Initializing RealBulletproofs...")
    bp = RealBulletproofs()
    assert bp.available, "RealBulletproofs not available"

    # Test 3: Generate and verify a simple proof
    print("\n3. Testing basic proof generation and verification...")
    try:
        # Test with a simple integer value
        test_value = 950  # Represents PoGQ score 0.95
        print(f"   Generating proof for value {test_value} (32-bit range)...")

        proof, commitment, _ = bp.zkrp_prove(test_value, 32)

        print(f"   ✅ Proof generated")
        print(f"      Proof size: {len(proof)} bytes")
        print(f"      Commitment size: {len(commitment)} bytes")

        # Verify the proof
        print(f"\n   Verifying proof...")
        is_valid = bp.zkrp_verify(proof, commitment)

        assert is_valid, "Bulletproof verification failed"

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(str(e))

    # Test 4: Test ZKPoC with real Bulletproofs
    print("\n4. Testing ZKPoC with real Bulletproofs...")
    zkpoc = ZKPoC(pogq_threshold=0.7, use_real_bulletproofs=True)

    # Test case 1: High-quality gradient
    print("\n   Test Case 1: High-quality gradient (PoGQ = 0.95)")
    try:
        proof = zkpoc.generate_proof(0.95)
        print(f"      ✅ Proof generated")

        is_valid = zkpoc.verify_proof(proof)
        assert is_valid, "ZKPoC verification failed" 

    except Exception as e:
        print(f"      ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Unexpected error: {e}")

    # Test case 2: Marginal gradient
    print("\n   Test Case 2: Marginal gradient (PoGQ = 0.71)")
    try:
        proof = zkpoc.generate_proof(0.71)
        print(f"      ✅ Proof generated")

        is_valid = zkpoc.verify_proof(proof)
        assert is_valid, "ZKPoC verification failed for marginal gradient"

    except Exception as e:
        print(f"      ❌ Error: {e}")
        pytest.fail(str(e))

    # Test case 3: Low-quality gradient (should fail)
    print("\n   Test Case 3: Low-quality gradient (PoGQ = 0.45)")
    try:
        proof = zkpoc.generate_proof(0.45)
        pytest.fail("generate_proof should reject score below threshold")
    except ValueError as e:
        print(f"      ✅ Correctly rejected: {e}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("\nReal Bulletproofs integration working:")
    print("  • Proof generation: ✅")
    print("  • Proof verification: ✅")
    print("  • ZKPoC integration: ✅")
print("  • Threshold enforcement: ✅")


if __name__ == "__main__":
    success = test_real_bulletproofs()
    sys.exit(0 if success else 1)
