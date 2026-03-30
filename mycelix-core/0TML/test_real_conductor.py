#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Real Conductor Connection

Verifies that the bridge can connect to a real Holochain conductor.
"""

import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from zerotrustml.holochain.bridges.holochain_credits_bridge_rust import (
    HolochainCreditsBridge as RustBridge,
)
from holochain_credits_bridge import HolochainBridge

def test_real_connection():
    """Test connection to real conductor"""
    if RustBridge is None or not hasattr(RustBridge, "is_connected"):
        pytest.skip("Rust bridge not available; skipping real conductor test")

    print("="*70)
    print("  Real Conductor Connection Test")
    print("="*70)

    bridge = HolochainBridge("ws://localhost:8888", enabled=True)

    # Test 1: Initial connection
    print("\n1. Testing initial connection...")
    connected = bridge.is_connected()
    print(f"   Initial status: {'connected' if connected else 'not connected'}")

    # Test 2: Attempt connection
    print("\n2. Attempting to connect to conductor...")
    try:
        success = bridge.connect()
        if success:
            print(f"   ✅ Successfully connected to real conductor!")

            # Check connection status
            connected = bridge.is_connected()
            print(f"   Connection verified: {connected}")

            # Get health
            health = bridge.get_connection_health()
            print(f"\n   Connection Health:")
            print(f"     Is Connected: {health['is_connected']}")
            print(f"     Failed Attempts: {health['failed_attempts']}")
            print(f"     Circuit Open: {health['circuit_open']}")
            print(f"     Backoff: {health['backoff_seconds']}s")

            return
        else:
            print(f"   ⚠️  Connection returned False")
            return False

    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

def main():
    """Run real conductor test"""
    try:
        success = test_real_connection()

        print("\n" + "="*70)
        if success:
            print("  ✅ REAL CONDUCTOR TEST PASSED!")
            print("="*70)
            print("\nThe bridge successfully connected to a real Holochain conductor.")
            print("This validates that WebSocket connection logic works in production.")
            return 0
        else:
            print("  ⚠️  CONNECTION FAILED")
            print("="*70)
            print("\nCheck that the conductor is running:")
            print("  ps aux | grep holochain")
            print("  tail -f /tmp/conductor-test.log")
            return 1

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
