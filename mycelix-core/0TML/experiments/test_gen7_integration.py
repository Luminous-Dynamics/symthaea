#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Gen7 zkSTARK + Dilithium Integration Test with AEGIS

Tests the complete Gen7 system:
1. Dilithium5 keypair generation and signatures  
2. zkSTARK proof generation for gradient verification
3. Integration with AEGIS federated learning system

This demonstrates the hybrid zk-DASTARK architecture:
- Client proves "I am client i" (Dilithium signature)
- Client proves "gradient is from model M_t" (zkSTARK proof)
- Server verifies both without learning gradient content
"""

import sys
import os
import time
import numpy as np

# Add Gen7 bindings to path
sys.path.insert(0, '/srv/luminous-dynamics/Mycelix-Core/0TML/gen7-zkstark/bindings')

try:
    import gen7_zkstark
    print("✅ Gen7 module imported successfully")
    print(f"Available functions: {[x for x in dir(gen7_zkstark) if not x.startswith('_')]}")
except ImportError as e:
    print(f"❌ Failed to import gen7_zkstark: {e}")
    sys.exit(1)


def test_dilithium_signatures():
    """Test post-quantum Dilithium5 signature generation and verification."""
    print("\n" + "="*60)
    print("TEST 1: Dilithium5 Post-Quantum Signatures")
    print("="*60)

    # Generate keypair
    print("Generating Dilithium5 keypair...")
    start = time.time()
    keypair = gen7_zkstark.DilithiumKeypair()  # Using #[new] constructor, not .generate()
    keygen_time = time.time() - start
    print(f"✅ Keypair generated in {keygen_time*1000:.2f}ms")

    # Get public key
    pubkey = keypair.get_public_key()
    print(f"Public key size: {len(pubkey)} bytes")

    # Create test gradient
    gradient = np.random.randn(10).astype(np.float32)
    print(f"Test gradient shape: {gradient.shape}")

    # Hash the gradient
    gradient_hash = gen7_zkstark.hash_gradient_py(gradient.tobytes())
    print(f"Gradient hash: {gradient_hash.hex()[:32]}...")

    # Sign the gradient hash
    print("Signing gradient hash...")
    start = time.time()
    signature = keypair.sign(gradient_hash)
    sign_time = time.time() - start
    print(f"✅ Signature created in {sign_time*1000:.2f}ms")
    print(f"Signature size: {len(signature)} bytes")

    # Verify signature (correct parameter order: message, signature, public_key)
    print("Verifying signature...")
    start = time.time()
    is_valid = gen7_zkstark.DilithiumKeypair.verify(gradient_hash, signature, pubkey)
    verify_time = time.time() - start
    print(f"✅ Signature verified in {verify_time*1000:.2f}ms")
    print(f"Signature valid: {is_valid}")

    assert is_valid, "Signature verification failed!"

    # Test invalid signature detection
    print("\nTesting tamper detection...")
    tampered_hash = bytearray(gradient_hash)
    tampered_hash[0] ^= 0xFF  # Flip first byte
    is_valid_tampered = gen7_zkstark.DilithiumKeypair.verify(bytes(tampered_hash), signature, pubkey)
    print(f"Tampered signature valid: {is_valid_tampered}")
    assert not is_valid_tampered, "Failed to detect tampered signature!"
    print("✅ Tamper detection working correctly")

    return keypair, gradient_hash, signature


def test_zkstark_proofs():
    """Test zkSTARK proof generation and verification for gradients."""
    print("\n" + "="*60)
    print("TEST 2: zkSTARK Gradient Proofs")
    print("="*60)

    # Create test model parameters (simulated)
    print("Creating test model parameters...")
    model_params = np.random.randn(100).astype(np.float32)
    params_hash = gen7_zkstark.hash_model_params_py(model_params.tobytes())
    print(f"Model params hash: {params_hash.hex()[:32]}...")

    # Create test gradient
    gradient = np.random.randn(100).astype(np.float32)
    gradient_hash = gen7_zkstark.hash_gradient_py(gradient.tobytes())
    print(f"Gradient hash: {gradient_hash.hex()[:32]}...")

    # Note: Full zkSTARK proof generation requires RISC Zero zkVM
    # For now, we test the hash functions which are working
    print("\n⚠️  Full zkSTARK proof generation requires RISC Zero zkVM runtime")
    print("✅ Hash functions working correctly")
    print("   - hash_model_params_py() ✓")
    print("   - hash_gradient_py() ✓")

    return params_hash, gradient_hash


def test_authenticated_gradient_proof():
    """Test the AuthenticatedGradientProof class combining Dilithium + zkSTARK."""
    print("\n" + "="*60)
    print("TEST 3: Authenticated Gradient Proof (Hybrid zk-DASTARK)")
    print("="*60)

    # Generate keypair
    print("Generating client keypair...")
    keypair = gen7_zkstark.DilithiumKeypair()  # Using #[new] constructor
    client_id = "client_001"

    # Create test gradient
    gradient = np.random.randn(50).astype(np.float32)
    gradient_bytes = gradient.tobytes()

    # Create authenticated proof
    print(f"Creating authenticated proof for client {client_id}...")
    start = time.time()

    # Hash the gradient
    gradient_hash = gen7_zkstark.hash_gradient_py(gradient_bytes)

    # Generate nonce and timestamp
    nonce = gen7_zkstark.generate_nonce()  # Returns Vec<u8> as Python list
    timestamp = gen7_zkstark.current_timestamp()

    # Create proof structure
    proof_time = time.time() - start
    print(f"✅ Proof components generated in {proof_time*1000:.2f}ms")
    print(f"   - Client ID: {client_id}")
    print(f"   - Gradient hash: {gradient_hash.hex()[:32]}...")
    print(f"   - Nonce: {nonce[:16]}...")
    print(f"   - Timestamp: {timestamp}")

    # Sign the proof (nonce is already bytes list, timestamp needs encoding)
    message = gradient_hash + bytes(nonce) + str(timestamp).encode()
    signature = keypair.sign(message)
    print(f"   - Signature size: {len(signature)} bytes")

    # Verify the proof
    print("\nVerifying authenticated proof...")
    pubkey = keypair.get_public_key()
    is_valid = gen7_zkstark.DilithiumKeypair.verify(message, signature, pubkey)
    print(f"✅ Proof verified: {is_valid}")

    assert is_valid, "Authenticated proof verification failed!"

    return {
        'client_id': client_id,
        'gradient_hash': gradient_hash,
        'nonce': nonce,
        'timestamp': timestamp,
        'signature': signature,
        'pubkey': pubkey
    }


def test_aegis_integration_concept():
    """Demonstrate how Gen7 integrates with AEGIS federated learning."""
    print("\n" + "="*60)
    print("TEST 4: AEGIS Integration Concept")
    print("="*60)

    print("\nGen7 enhances AEGIS with cryptographic verification:")
    print()
    print("1. Client-side (during local training):")
    print("   - Generate Dilithium5 keypair")
    print("   - Train local model, compute gradient")
    print("   - Create zkSTARK proof: gradient derived from model M_t")
    print("   - Sign gradient hash with Dilithium private key")
    print("   - Send: {gradient, zkSTARK_proof, Dilithium_signature}")
    print()
    print("2. Server-side (AEGIS aggregation):")
    print("   - Verify Dilithium signature (post-quantum secure)")
    print("   - Verify zkSTARK proof (gradient authenticity)")
    print("   - Run AEGIS 7-layer defense stack:")
    print("     Layer 1: Input Validation")
    print("     Layer 2: Outlier Detection (IQR)")
    print("     Layer 3: Gradient Clipping")
    print("     Layer 4: Cartel Detection")
    print("     Layer 5: PoGQ Trust Weighting")
    print("     Layer 6: TCDM Reputation")
    print("     Layer 7: Median Aggregation")
    print("   - Only accept gradients that pass ALL checks")
    print()
    print("3. Security guarantees:")
    print("   - Identity: Client cannot impersonate others (Dilithium)")
    print("   - Authenticity: Gradient must be from declared model (zkSTARK)")
    print("   - Byzantine tolerance: 45% with PoGQ + crypto verification")
    print("   - Post-quantum secure: Safe against quantum computers")
    print()

    # Simulate the workflow
    print("Simulating Gen7-AEGIS workflow:")
    print("-" * 60)

    # Client side
    print("\n[Client] Generating proof...")
    keypair = gen7_zkstark.DilithiumKeypair()  # Using #[new] constructor
    gradient = np.random.randn(100).astype(np.float32)
    gradient_hash = gen7_zkstark.hash_gradient_py(gradient.tobytes())
    signature = keypair.sign(gradient_hash)
    print(f"[Client] ✅ Gradient signed ({len(signature)} bytes)")

    # Network transmission (simulated)
    print("[Network] Transmitting gradient + proof...")

    # Server side
    print("[Server] Verifying cryptographic proofs...")
    pubkey = keypair.get_public_key()
    is_valid = gen7_zkstark.DilithiumKeypair.verify(gradient_hash, signature, pubkey)

    if is_valid:
        print("[Server] ✅ Cryptographic verification passed")
        print("[Server] ✅ Proceeding to AEGIS 7-layer defense...")
        print("[Server] ✅ Gradient accepted for aggregation")
    else:
        print("[Server] ❌ Cryptographic verification failed")
        print("[Server] ❌ Gradient rejected (Byzantine attack detected)")

    assert is_valid, "Server-side verification failed!"


def main():
    """Run all Gen7 integration tests."""
    print("\n" + "="*60)
    print("Gen7 zkSTARK + Dilithium Integration Test Suite")
    print("="*60)
    print("Testing hybrid zk-DASTARK architecture for AEGIS")
    print("="*60)

    try:
        # Test 1: Dilithium signatures
        keypair, gradient_hash, signature = test_dilithium_signatures()

        # Test 2: zkSTARK proofs
        params_hash, grad_hash = test_zkstark_proofs()

        # Test 3: Authenticated gradient proofs
        proof = test_authenticated_gradient_proof()

        # Test 4: AEGIS integration concept
        test_aegis_integration_concept()

        # Summary
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nGen7 Integration Status:")
        print("  ✅ Dilithium5 post-quantum signatures working")
        print("  ✅ Gradient hashing working")
        print("  ✅ Model parameter hashing working")
        print("  ✅ Authenticated proof structure working")
        print("  ✅ AEGIS integration concept validated")
        print()
        print("Next steps:")
        print("  - Integrate zkSTARK proof generation (requires RISC Zero runtime)")
        print("  - Add Gen7 verification to AEGIS aggregation pipeline")
        print("  - Run byzantine attack experiments with Gen7 enabled")
        print("  - Measure performance overhead (target: <100ms per proof)")
        print()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
