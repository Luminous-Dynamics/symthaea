#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Rust Bridge Connection with Fixed URL
"""

import asyncio
import holochain_credits_bridge

def test_connection():
    """Test connection to Holochain conductor via rust bridge"""
    print("=" * 70)
    print("  Rust Bridge Connection Test (After ws:// Fix)")
    print("=" * 70)
    print()

    # Create bridge with default URL (now includes ws://)
    print("1. Creating HolochainBridge...")
    bridge = holochain_credits_bridge.HolochainBridge()
    print(f"   ✓ Bridge created: {bridge}")
    print()

    # Test connection
    print("2. Testing connection to conductor...")
    print("   URL: ws://localhost:8888")
    print("   Attempting AdminWebsocket.connect()...")
    print()

    try:
        # This is a synchronous call - bridge uses py.allow_threads internally
        # PyO3 automatically provides the Python context - no arguments needed
        success = bridge.connect()

        if success:
            print("✅ CONNECTION SUCCESSFUL!")
            print()
            print("   The rust bridge connected successfully to Holochain!")
            print("   AdminWebsocket connection established.")
            print()
            return True
        else:
            print("❌ Connection returned False")
            print("   Bridge enabled but connection failed")
            return False

    except Exception as e:
        print(f"❌ Connection failed with exception:")
        print(f"   Error: {e}")
        print(f"   Type: {type(e).__name__}")
        print()

        # Check if conductor is running
        print("Diagnostic checks:")
        import subprocess
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=holochain", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            print(f"   ✓ Conductor container running: {result.stdout.strip()}")
        else:
            print("   ✗ No Holochain conductor container found")
            print("   Start it with: docker-compose -f holochain/docker-compose.holochain.yml up -d")

        return False

if __name__ == "__main__":
    print()
    success = test_connection()
    print()
    print("=" * 70)
    if success:
        print("✅ TEST PASSED - WebSocket connection working!")
        print()
        print("Next step: Run the full gen7-zkSTARK + Holochain integration test")
        print("  python test_gen7_holochain_integration.py")
    else:
        print("❌ TEST FAILED - Connection issue persists")
        print()
        print("Check:")
        print("  1. Is the conductor running? docker ps")
        print("  2. Check logs: docker logs holochain-zerotrustml")
        print("  3. Check conductor config: holochain/conductor-config-minimal.yaml")
    print("=" * 70)
    print()

    exit(0 if success else 1)
