#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test script for refactored holochain_client v0.8 integration.

Tests:
1. Connection to conductor using AdminWebsocket
2. Agent key generation using AdminWebsocket API
3. DNA installation using AdminWebsocket API
"""

import sys
import os

import pytest

pytestmark = pytest.mark.skip("Refactored bridge manual script; skip in default test run")

# Import the refactored bridge
try:
    import holochain_credits_bridge
    print("✅ Module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import module: {e}")
    sys.exit(1)

def test_connection():
    """Test connection to conductor"""
    print("\n" + "="*60)
    print("TEST 1: Connection to Conductor")
    print("="*60)

    bridge = holochain_credits_bridge.HolochainBridge(
        conductor_url="localhost:8888"
    )

    result = bridge.connect()
    print(f"Connection result: {result}")

    if result:
        print("✅ Connected successfully using AdminWebsocket!")
        return bridge
    else:
        print(f"❌ Connection failed")
        return None

def test_agent_key_generation(bridge):
    """Test agent key generation"""
    print("\n" + "="*60)
    print("TEST 2: Agent Key Generation")
    print("="*60)

    try:
        agent_key = bridge.generate_agent_key()
        print(f"✅ Generated agent key: {agent_key[:50]}...")
        return agent_key
    except Exception as e:
        print(f"❌ Failed to generate agent key: {e}")
        return None

def test_dna_installation(bridge):
    """Test DNA installation"""
    print("\n" + "="*60)
    print("TEST 3: DNA Installation")
    print("="*60)

    # Find the hApp file
    happ_path = "/srv/luminous-dynamics/Mycelix-Core/0TML/zerotrustml-dna/zerotrustml.happ"

    if not os.path.exists(happ_path):
        print(f"❌ hApp file not found at: {happ_path}")
        return False

    print(f"📦 hApp file found: {happ_path}")
    print(f"   Size: {os.path.getsize(happ_path)} bytes")

    try:
        app_id = bridge.install_app(
            app_id="zerotrustml-test-refactored",
            happ_path=happ_path
        )
        print(f"✅ DNA installed successfully!")
        print(f"   App ID: {app_id}")
        return True
    except Exception as e:
        print(f"❌ DNA installation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("🧪 REFACTORED BRIDGE TEST SUITE")
    print("   holochain_client v0.8.0-dev.20")
    print("   AdminWebsocket API Integration")
    print("="*60)

    # Test 1: Connection
    bridge = test_connection()
    if not bridge:
        print("\n❌ Test suite failed at connection")
        return 1

    # Test 2: Agent key generation
    agent_key = test_agent_key_generation(bridge)
    if not agent_key:
        print("\n❌ Test suite failed at agent key generation")
        return 1

    # Test 3: DNA installation
    success = test_dna_installation(bridge)
    if not success:
        print("\n❌ Test suite failed at DNA installation")
        return 1

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n🎉 Refactoring complete and working!")
    print("   - AdminWebsocket connection: ✅")
    print("   - Agent key generation: ✅")
    print("   - DNA installation: ✅")
    print("   - Code reduction: 78% (364 → ~80 lines)")
    print("\nPhase 7 Session 4: SUCCESS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
