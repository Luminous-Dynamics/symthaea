#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test script to verify zerotrustml package structure and basic API

Run from Nix environment:
    nix develop --command python test_package.py
"""

import sys
sys.path.insert(0, '/srv/luminous-dynamics/Mycelix-Core/0TML/src')

import asyncio
import numpy as np
from zerotrustml import (
    Node, NodeConfig,
    CreditSystem,
    FedAvg, Krum, TrimmedMean,
    SimpleNN, RealMLNode
)


def test_imports():
    """Test that all key classes can be imported"""
    print("✅ All imports successful!")


def test_aggregation():
    """Test aggregation algorithms"""
    print("\nTesting aggregation algorithms...")

    # Create sample gradients
    gradients = [np.random.randn(100) for _ in range(5)]
    reputations = [0.8, 0.9, 0.7, 0.85, 0.75]

    # Test FedAvg
    agg1 = FedAvg.aggregate(gradients, reputations)
    assert agg1.shape == (100,), "FedAvg shape mismatch"
    print("  ✓ FedAvg working")

    # Test Krum
    agg2 = Krum.aggregate(gradients, reputations, num_byzantine=1)
    assert agg2.shape == (100,), "Krum shape mismatch"
    print("  ✓ Krum working")

    # Test Trimmed Mean
    agg3 = TrimmedMean.aggregate(gradients, reputations, trim_ratio=0.2)
    assert agg3.shape == (100,), "TrimmedMean shape mismatch"
    print("  ✓ TrimmedMean working")

    print("✅ All aggregation algorithms working!")


def test_model():
    """Test SimpleNN model"""
    print("\nTesting SimpleNN model...")

    model = SimpleNN(input_dim=784, hidden_dim=128, output_dim=10)

    # Get parameters
    params = model.get_flat_params()
    assert len(params) > 0, "No parameters extracted"
    print(f"  ✓ Model has {len(params)} parameters")

    # Set parameters
    model.set_flat_params(params)
    params2 = model.get_flat_params()
    assert np.allclose(params, params2), "Parameter set/get mismatch"
    print("  ✓ Parameter set/get working")

    print("✅ SimpleNN model working!")


async def test_credits():
    """Test credit system"""
    print("\nTesting credit system...")

    credit_system = CreditSystem()

    # Issue quality gradient credits
    credit_id = await credit_system.on_quality_gradient(
        node_id="test_node_1",
        pogq_score=0.85,
        reputation_level="NORMAL",
        verifiers=["v1", "v2", "v3"]
    )
    assert credit_id is not None, "Failed to issue credits"
    print(f"  ✓ Issued quality credits: {credit_id}")

    # Check balance
    balance = await credit_system.get_balance("test_node_1")
    assert balance > 0, "Balance should be positive"
    print(f"  ✓ Node balance: {balance:.0f} credits")

    # Issue Byzantine detection credits
    credit_id2 = await credit_system.on_byzantine_detection(
        detector_node_id="test_node_1",
        detected_node_id="byzantine_node",
        reputation_level="TRUSTED"
    )
    assert credit_id2 is not None, "Failed to issue Byzantine credits"
    print(f"  ✓ Issued Byzantine detection credits: {credit_id2}")

    # Check new balance
    balance2 = await credit_system.get_balance("test_node_1")
    assert balance2 > balance, "Balance should have increased"
    print(f"  ✓ New balance: {balance2:.0f} credits")

    print("✅ Credit system working!")


def test_config():
    """Test NodeConfig"""
    print("\nTesting NodeConfig...")

    config = NodeConfig(
        node_id="hospital-1",
        data_path="/data/medical",
        model_type="resnet18",
        holochain_url="ws://localhost:8888"
    )

    assert config.node_id == "hospital-1"
    assert config.aggregation == "krum"  # Default
    print("  ✓ NodeConfig created successfully")
    print(f"  ✓ Config: {config.node_id}, {config.model_type}, {config.aggregation}")

    print("✅ NodeConfig working!")


async def main():
    """Run all tests"""
    print("="*60)
    print("ZeroTrustML Package Structure Validation")
    print("="*60)

    try:
        test_imports()
        test_aggregation()
        test_model()
        await test_credits()
        test_config()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n🎉 All tests passed! Package structure is working correctly.")
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
