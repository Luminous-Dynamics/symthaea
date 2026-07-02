#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Verify Rust backend integration for mycelix_fl.

Usage:
    python -m mycelix_fl.verify_rust
"""

import sys


def verify() -> bool:
    """Verify Rust backend is working correctly."""
    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy not available")
        return False

    from mycelix_fl.rust_bridge import (
        RustHypervectorEncoder,
        RustPhiMeasurer,
        RustShapleyComputer,
        RustUnifiedDetector,
        RUST_AVAILABLE,
        get_version,
        get_dimension,
    )

    print("=" * 60)
    print("Mycelix FL - Rust Backend Verification")
    print("=" * 60)

    print(f"\nRust available: {RUST_AVAILABLE}")
    print(f"Rust version: {get_version()}")
    print(f"HV dimension: {get_dimension()}")

    if not RUST_AVAILABLE:
        print("\n⚠️  Rust backend not available - using Python fallback")
        print("   Install zerotrustml-core wheel for 100-1000x speedup")
        return True  # Not an error, just fallback mode

    all_ok = True

    # Test 1: HypervectorEncoder
    print("\n1. HypervectorEncoder...", end=" ")
    try:
        encoder = RustHypervectorEncoder(dimension=16384)
        hv = encoder.encode(np.random.randn(500).astype(np.float32))
        assert encoder.is_rust, "Not using Rust"
        assert hv.dimension == 16384, "Wrong dimension"
        print("✓ OK")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        all_ok = False

    # Test 2: PhiMeasurer
    print("2. PhiMeasurer...", end=" ")
    try:
        phi = RustPhiMeasurer(dimension=16384)
        hvs = [encoder.encode(np.random.randn(500).astype(np.float32)) for _ in range(5)]
        result = phi.measure_phi(hvs)
        assert 0 <= result.phi_value <= 2, f"Phi out of range: {result.phi_value}"
        assert phi.is_rust, "Not using Rust"
        print(f"✓ OK (Φ={result.phi_value:.4f})")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        all_ok = False

    # Test 3: ShapleyComputer
    print("3. ShapleyComputer...", end=" ")
    try:
        shapley = RustShapleyComputer(dimension=16384)
        contributions = shapley.compute_exact(hvs)
        weights = shapley.aggregation_weights(hvs)
        assert len(contributions) == 5, "Wrong contribution count"
        assert abs(sum(weights) - 1.0) < 0.01, f"Weights don't sum to 1: {sum(weights)}"
        assert shapley.is_rust, "Not using Rust"
        print("✓ OK")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        all_ok = False

    # Test 4: UnifiedDetector
    print("4. UnifiedDetector...", end=" ")
    try:
        detector = RustUnifiedDetector(dimension=16384)
        gradients = {f"n{i}": np.random.randn(500).astype(np.float32) for i in range(5)}
        gradients["byz"] = np.random.randn(500).astype(np.float32) * 100
        result = detector.detect(gradients)
        assert "byz" in result.byzantine_nodes, "Failed to detect Byzantine"
        assert detector.is_rust, "Not using Rust"
        print(f"✓ OK (detected: {result.byzantine_nodes})")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        all_ok = False

    # Test 5: Full FL System
    print("5. MycelixFL System...", end=" ")
    try:
        from mycelix_fl import MycelixFL, FLConfig, has_rust_backend

        config = FLConfig(
            num_rounds=1,
            byzantine_threshold=0.45,
            use_compression=True,
            use_detection=True,
        )
        fl = MycelixFL(config=config)

        gradients = {f"node-{i}": np.random.randn(500).astype(np.float32) for i in range(5)}
        gradients["byzantine-0"] = np.random.randn(500).astype(np.float32) * 100

        result = fl.execute_round(gradients, round_num=0)
        assert result.aggregated_gradient is not None, "No aggregated gradient"
        print(f"✓ OK (detected {len(result.byzantine_nodes)} Byzantine)")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ ALL VERIFICATIONS PASSED")
    else:
        print("✗ SOME VERIFICATIONS FAILED")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
