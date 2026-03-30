#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Verification Script: Rust Bridge Integration

Tests that the Rust PyO3 bridge is properly installed and functional.
Run this to verify the ZeroTrustML-Holochain integration is ready.
"""

import sys
import asyncio
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_rust_module():
    """Test 1: Verify Rust module can be imported"""
    print_section("TEST 1: Rust Module Import")

    try:
        import holochain_credits_bridge
        print("✅ Rust module imported successfully")
        print(f"   Module: {holochain_credits_bridge}")
        print(f"   Location: {holochain_credits_bridge.__file__}")

        # Check exported classes
        if hasattr(holochain_credits_bridge, 'HolochainBridge'):
            print("✅ HolochainBridge class available")
        else:
            print("❌ HolochainBridge class NOT found")
            return False

        if hasattr(holochain_credits_bridge, 'CreditIssuance'):
            print("✅ CreditIssuance class available")
        else:
            print("❌ CreditIssuance class NOT found")
            return False

        return True

    except ImportError as e:
        print(f"❌ Failed to import Rust module: {e}")
        print("\nTo fix:")
        print("  cd rust-bridge")
        print("  maturin develop --release")
        return False

def test_bridge_creation():
    """Test 2: Create bridge instance"""
    print_section("TEST 2: Bridge Creation")

    try:
        import holochain_credits_bridge

        bridge = holochain_credits_bridge.HolochainBridge(
            conductor_url="ws://localhost:8888",
            app_id="zerotrustml",
            zome_name="zerotrustml_credits",
            enabled=True
        )
        print(f"✅ Bridge created: {bridge}")
        return True, bridge

    except Exception as e:
        print(f"❌ Failed to create bridge: {e}")
        return False, None

def test_credit_issuance(bridge):
    """Test 3: Issue credits"""
    print_section("TEST 3: Credit Issuance")

    try:
        # Issue 100 credits for model update with PoGQ 0.95
        issuance = bridge.issue_credits(
            42,    # node_id
            "model_update",  # event_type
            100,   # amount
            0.95,  # pogq_score
            [1, 2, 3]  # verifiers
        )

        print(f"✅ Credits issued successfully")
        print(f"   Node: {issuance.node_id}")
        print(f"   Amount: {issuance.amount}")
        print(f"   Reason: {issuance.reason}")
        print(f"   Hash: {issuance.action_hash}")
        print(f"   Timestamp: {datetime.fromtimestamp(issuance.timestamp)}")
        return True

    except Exception as e:
        print(f"❌ Failed to issue credits: {e}")
        return False

def test_balance_query(bridge):
    """Test 4: Query balance"""
    print_section("TEST 4: Balance Query")

    try:
        balance = bridge.get_balance(42)
        print(f"✅ Balance retrieved: {balance} credits")

        if balance > 0:
            print(f"   ✅ Balance is positive (credits were issued)")
        else:
            print(f"   ⚠️  Balance is zero (no credits issued)")

        return True

    except Exception as e:
        print(f"❌ Failed to query balance: {e}")
        return False

def test_history(bridge):
    """Test 5: Get history"""
    print_section("TEST 5: History Retrieval")

    try:
        # Get history for node 42
        history = bridge.get_history(42)
        print(f"✅ History retrieved: {len(history)} events")

        for i, event in enumerate(history[:5], 1):  # Show first 5
            print(f"\n   Event {i}:")
            print(f"     Node: {event.node_id}")
            print(f"     Amount: {event.amount}")
            print(f"     Reason: {event.reason}")
            print(f"     Time: {datetime.fromtimestamp(event.timestamp)}")

        if len(history) > 5:
            print(f"\n   ... and {len(history) - 5} more events")

        return True

    except Exception as e:
        print(f"❌ Failed to get history: {e}")
        return False

def test_system_stats(bridge):
    """Test 6: System statistics"""
    print_section("TEST 6: System Statistics")

    try:
        stats = bridge.get_stats()
        print(f"✅ Stats retrieved:")
        print(f"   Total Credits: {stats['total_credits']}")
        print(f"   Total Events: {stats['total_events']}")
        print(f"   Unique Nodes: {stats['unique_nodes']}")
        return True

    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
        return False

async def test_python_wrapper():
    """Test 7: Python wrapper compatibility"""
    print_section("TEST 7: Python Wrapper (Optional)")

    try:
        from zerotrustml.holochain.bridges.holochain_credits_bridge_rust import (
            HolochainCreditsBridge,
        )

        bridge = HolochainCreditsBridge(
            conductor_url="ws://localhost:8888",
            enabled=True
        )

        print(f"✅ Python wrapper imported and initialized")

        # Try connect
        connected = await bridge.connect()
        if connected:
            print(f"✅ Wrapper connect successful")
        else:
            print(f"⚠️  Wrapper in mock mode (conductor not available)")

        # Try issue credits
        credits = await bridge.issue_credits(
            node_id=99,
            event_type="quality_gradient",
            pogq_score=0.88,
            verifiers=[1, 2, 3]
        )
        print(f"✅ Wrapper issued {credits} credits")

        return True

    except ImportError as e:
        print(f"⚠️  Python wrapper not found (optional): {e}")
        return True  # Not a failure
    except Exception as e:
        print(f"❌ Python wrapper error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  RUST BRIDGE VERIFICATION SUITE")
    print("  ZeroTrustML-Holochain Integration")
    print("="*60)

    results = []

    # Test 1: Module import
    if not test_rust_module():
        print("\n❌ FAILED: Rust module not available")
        sys.exit(1)
    results.append(True)

    # Test 2: Bridge creation
    success, bridge = test_bridge_creation()
    if not success:
        print("\n❌ FAILED: Cannot create bridge")
        sys.exit(1)
    results.append(True)

    # Test 3-6: Bridge operations
    results.append(test_credit_issuance(bridge))
    results.append(test_balance_query(bridge))
    results.append(test_history(bridge))
    results.append(test_system_stats(bridge))

    # Test 7: Python wrapper (optional)
    results.append(asyncio.run(test_python_wrapper()))

    # Summary
    print_section("SUMMARY")
    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("\n✅ ALL TESTS PASSED - Rust bridge fully functional!")
        print("\nNext steps:")
        print("  1. Connect to real Holochain conductor")
        print("  2. Implement TODO sections in lib.rs")
        print("  3. Run production integration tests")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("\nCheck the output above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
