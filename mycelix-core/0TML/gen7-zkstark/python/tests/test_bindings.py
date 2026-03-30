# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Gen-7 zkSTARK Python bindings
"""
import pytest

def test_import():
    """Test that the module can be imported"""
    try:
        import gen7_zkstark
        assert hasattr(gen7_zkstark, 'prove_gradient_zkstark')
        assert hasattr(gen7_zkstark, 'verify_gradient_zkstark')
        assert hasattr(gen7_zkstark, 'hash_model_params_py')
        assert hasattr(gen7_zkstark, 'hash_gradient_py')
        print("✅ All functions exported correctly")
    except ImportError as e:
        pytest.skip(f"Module not built yet: {e}")

def test_hash_functions():
    """Test hashing helper functions"""
    try:
        import gen7_zkstark

        # Test model param hashing
        model_params = [0.5, -0.3, 0.8, 0.2]
        hash1 = gen7_zkstark.hash_model_params_py(model_params)
        assert len(hash1) == 32, "Hash should be 32 bytes"

        # Same params should give same hash
        hash2 = gen7_zkstark.hash_model_params_py(model_params)
        assert hash1 == hash2, "Same params should give same hash"

        # Different params should give different hash
        different_params = [0.5, -0.3, 0.8, 0.3]
        hash3 = gen7_zkstark.hash_model_params_py(different_params)
        assert hash1 != hash3, "Different params should give different hash"

        print(f"✅ Model hash (first 8 bytes): {hash1[:8].hex()}")

        # Test gradient hashing
        gradient = [0.01, -0.02, 0.03, 0.01]
        grad_hash = gen7_zkstark.hash_gradient_py(gradient)
        assert len(grad_hash) == 32, "Gradient hash should be 32 bytes"

        print(f"✅ Gradient hash (first 8 bytes): {grad_hash[:8].hex()}")

    except ImportError:
        pytest.skip("Module not built yet")

def test_proof_generation_simple():
    """Test zkSTARK proof generation with toy example"""
    try:
        import gen7_zkstark

        # Toy model: 4 parameters
        model_params = [0.5, -0.3, 0.8, 0.2]

        # Toy dataset: 3 samples, 4 features each
        local_data = [
            # Sample 1
            1.0, 0.5, 0.3, 0.1,
            # Sample 2
            0.8, 0.6, 0.4, 0.2,
            # Sample 3
            0.9, 0.7, 0.5, 0.3,
        ]

        # Toy labels: 3 samples, 2 classes (one-hot)
        local_labels = [
            # Sample 1: class 0
            1, 0,
            # Sample 2: class 1
            0, 1,
            # Sample 3: class 0
            1, 0,
        ]

        # Simplified gradient (matching host demo)
        gradient = [p / 100 for p in model_params]

        # Generate proof
        print("🔄 Generating zkSTARK proof (this may take 30-60 seconds)...")
        proof_bytes = gen7_zkstark.prove_gradient_zkstark(
            model_params=model_params,
            gradient=gradient,
            local_data=local_data,
            local_labels=local_labels,
            num_samples=3,
            input_dim=4,
            num_classes=2,
            epochs=1,
            learning_rate=0.01,
        )

        assert isinstance(proof_bytes, bytes), "Proof should be bytes"
        assert len(proof_bytes) > 0, "Proof should not be empty"
        print(f"✅ Proof generated: {len(proof_bytes)} bytes")

        # Verify proof
        print("🔄 Verifying zkSTARK proof...")
        result = gen7_zkstark.verify_gradient_zkstark(proof_bytes)

        assert result["verified"] == True, "Proof should verify"
        assert result["epochs"] == 1, "Epochs should match"
        assert result["num_samples"] == 3, "Num samples should match"
        assert len(result["gradient_hash"]) == 32, "Gradient hash should be 32 bytes"

        print(f"✅ Proof verified successfully!")
        print(f"   Gradient hash: {result['gradient_hash'][:8].hex()}")
        print(f"   Epochs: {result['epochs']}")
        print(f"   Samples: {result['num_samples']}")

        return True

    except ImportError:
        pytest.skip("Module not built yet")
    except Exception as e:
        print(f"❌ Proof test failed: {e}")
        pytest.fail(str(e))

if __name__ == "__main__":
    print("=" * 60)
    print("Gen-7 zkSTARK Python Bindings Test")
    print("=" * 60)

    print("\n1. Testing module import...")
    test_import()

    print("\n2. Testing hash functions...")
    test_hash_functions()

    print("\n3. Testing proof generation and verification...")
    test_proof_generation_simple()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
