#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Official Holochain API Integration
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

def test_generate_agent_key():
    """Test generating an agent key with official API"""
    if RustBridge is None or not hasattr(RustBridge, "get_connection_health"):
        pytest.skip("Rust bridge not available; skipping official API test")

    print("=" * 70)
    print("  Agent Key Generation Test (Official API)")
    print("=" * 70)

    # Create bridge instance
    bridge = HolochainBridge(
        conductor_url="ws://localhost:8888",
        app_id="test-app",
        zome_name="credits",
        enabled=True
    )

    # Connect to conductor
    print("\nConnecting to conductor...")
    success = bridge.connect()
    if not success:
        print("   ❌ Failed to connect")
        return False

    # Check connection health
    health = bridge.get_connection_health()
    print(f"   ✓ Connected: {health['is_connected']}")

    # Generate agent key using official API
    print("\n Generating agent key with official AdminRequest type...")
    try:
        agent_key = bridge.generate_agent_key()
        print(f"   ✅ Agent key generated: {agent_key[:60]}...")
        assert isinstance(agent_key, str)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_generate_agent_key()
    sys.exit(0 if success else 1)
