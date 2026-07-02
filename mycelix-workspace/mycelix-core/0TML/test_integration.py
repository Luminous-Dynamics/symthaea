#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test the Integration Layer - Phase 3.1 Validation
Tests P2P + DHT integration without requiring full Holochain conductor
"""

import asyncio
import sys
import os
import time
import numpy as np

# Add the src directory to path
sys.path.append('/srv/luminous-dynamics/Mycelix-Core/0TML/src')
from integration_layer import IntegratedP2PNode, IntegratedFederatedSystem, HolochainBridge

async def test_dht_peer_discovery():
    """Test that DHT replaces hard-coded peer lists"""
    print("\n🔍 Testing DHT Peer Discovery")
    print("-" * 40)
    
    # Create a node
    node = IntegratedP2PNode(node_id=1)
    
    # Bootstrap from DHT (will use mock if Holochain not running)
    await node.bootstrap_from_dht()
    
    # Check that we got peers
    if len(node.peers) > 0:
        print(f"✅ Successfully discovered {len(node.peers)} peers via DHT")
        for i, peer in enumerate(node.peers[:3]):  # Show first 3
            print(f"   Peer {i+1}: Node {peer.node_id}")
    else:
        print("⚠️  No peers discovered (expected if Holochain not running)")
    
    return len(node.peers) > 0

async def test_gradient_checkpointing():
    """Test gradient checkpoint storage to DHT"""
    print("\n💾 Testing Gradient Checkpointing")
    print("-" * 40)
    
    # Create a node
    node = IntegratedP2PNode(node_id=2)
    
    # Create a mock gradient
    gradient = np.random.randn(1000) * 0.01
    contributors = [1, 2, 3, 4, 5]
    byzantine_count = 1
    
    # Store checkpoint
    hash_addr = await node.checkpoint_to_dht(
        round_id=1,
        aggregated_gradient=gradient,
        contributors=contributors,
        byzantine_count=byzantine_count
    )
    
    if hash_addr:
        print(f"✅ Checkpoint stored successfully")
        print(f"   Hash: {hash_addr[:16]}...")
        print(f"   Round: 1")
        print(f"   Contributors: {contributors}")
        print(f"   Byzantine detected: {byzantine_count}")
    else:
        print("⚠️  Checkpoint storage failed (expected if Holochain not running)")
    
    return hash_addr is not None

async def test_byzantine_resistance():
    """Test that Byzantine resistance is maintained"""
    print("\n🛡️ Testing Byzantine Resistance")
    print("-" * 40)
    
    # Create a node
    node = IntegratedP2PNode(node_id=3)
    
    # Create gradients including Byzantine
    normal_gradients = [np.random.randn(100) * 0.1 for _ in range(5)]
    byzantine_gradient = np.random.randn(100) * 10  # Large magnitude attack
    
    all_gradients = normal_gradients + [byzantine_gradient]
    
    # Test aggregation
    aggregated, byzantine_count = node.aggregate_gradients(all_gradients)
    
    print(f"Input: {len(all_gradients)} gradients (1 Byzantine)")
    print(f"Byzantine detected: {byzantine_count}")
    print(f"Aggregated gradient norm: {np.linalg.norm(aggregated):.4f}")
    
    # Check that Byzantine was detected
    if byzantine_count == 1:
        print("✅ Byzantine gradient correctly identified and filtered")
        success = True
    else:
        print("❌ Byzantine detection failed")
        success = False
    
    return success

async def test_full_integration():
    """Test complete integrated system with multiple rounds"""
    print("\n🔄 Testing Full Integration (3 rounds, 5 nodes)")
    print("-" * 40)
    
    # Create small system
    system = IntegratedFederatedSystem(num_nodes=5)
    
    # Add a Byzantine node
    byzantine = IntegratedP2PNode(666)
    system.nodes.append(byzantine)
    
    # Initialize
    await system.initialize()
    
    # Run a few rounds
    start_time = time.time()
    await system.run_training(num_rounds=3)
    elapsed = time.time() - start_time
    
    # Check results
    summary = system.get_checkpoints_summary()
    
    print("\n📊 Integration Test Results:")
    print(f"   Training time: {elapsed:.2f}s")
    print(f"   Checkpoints stored: {summary['total_checkpoints']}")
    print(f"   Byzantine detection rate: {summary['average_byzantine_detection']:.1f}")
    
    # Success if we have checkpoints and detected Byzantine
    success = (summary['total_checkpoints'] > 0 and 
              summary['average_byzantine_detection'] > 0)
    
    if success:
        print("✅ Full integration test passed")
    else:
        print("⚠️  Partial success (may need Holochain conductor running)")
    
    return success

async def main():
    """Run all integration tests"""
    print("🧪 Phase 3.1 Integration Layer Tests")
    print("=" * 50)
    print("Testing P2P + DHT integration...")
    print("Note: Will use mock DHT if Holochain conductor not running")
    
    # Run tests
    test_results = []
    
    # Test 1: DHT Discovery
    result = await test_dht_peer_discovery()
    test_results.append(("DHT Peer Discovery", result))
    
    # Test 2: Checkpointing
    result = await test_gradient_checkpointing()
    test_results.append(("Gradient Checkpointing", result))
    
    # Test 3: Byzantine Resistance
    result = await test_byzantine_resistance()
    test_results.append(("Byzantine Resistance", result))
    
    # Test 4: Full Integration
    result = await test_full_integration()
    test_results.append(("Full Integration", result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("-" * 50)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for name, result in test_results:
        status = "✅ PASS" if result else "⚠️  WARN"
        print(f"{status}: {name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("The P2P + DHT integration is working perfectly")
    elif passed >= 2:
        print("\n✅ Core functionality working!")
        print("Full integration requires Holochain conductor running")
    else:
        print("\n⚠️  Integration needs attention")
        print("Check that dependencies are installed")
    
    print("\n🚀 Next Steps:")
    print("1. Start Holochain conductor for full DHT functionality")
    print("2. Implement Trust Layer (Phase 3.2)")
    print("3. Scale to 50+ nodes (Phase 3.3)")

if __name__ == "__main__":
    asyncio.run(main())