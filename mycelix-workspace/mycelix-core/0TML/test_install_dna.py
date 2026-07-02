#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test DNA Installation with Admin API

Tests the newly implemented admin API methods for installing the ZeroTrustML DNA.
"""

import sys
import os
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_ROOT))

from zerotrustml.holochain.bridges.holochain_credits_bridge_rust import (
    HolochainCreditsBridge as RustBridge,
)
from holochain_credits_bridge import HolochainBridge

def test_install_dna():
    """Test installing the ZeroTrustML DNA via admin API"""
    if RustBridge is None or not hasattr(RustBridge, "install_app"):
        pytest.skip("Rust bridge not available; skipping DNA install test")

    print("=" * 70)
    print("  DNA Installation Test (Admin API)")
    print("=" * 70)

    # Create bridge instance
    print("\n1. Creating bridge instance...")
    bridge = HolochainBridge(
        conductor_url="ws://localhost:8888",
        app_id="zerotrustml-app",
        zome_name="credits",
        enabled=True
    )
    print("   ✅ Bridge created")

    # Connect to conductor
    print("\n2. Connecting to conductor...")
    try:
        success = bridge.connect()
        if not success:
            print("   ❌ Failed to connect")
            return False
        print("   ✅ Connected successfully")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
        return False

    # Check connection health
    print("\n3. Checking connection health...")
    try:
        health = bridge.get_connection_health()
        print(f"   Connected: {health['is_connected']}")
        print(f"   Failed attempts: {health['failed_attempts']}")
        print(f"   Circuit open: {health['circuit_open']}")
    except Exception as e:
        print(f"   ⚠️  Could not get health: {e}")

    # Install the DNA
    print("\n4. Installing ZeroTrustML DNA...")
    try:
        happ_path = os.path.join(
            os.path.dirname(__file__),
            "zerotrustml-dna",
            "zerotrustml.happ"
        )

        if not os.path.exists(happ_path):
            print(f"   ❌ hApp bundle not found: {happ_path}")
            return False

        print(f"   Installing from: {happ_path}")

        # Call the install_app method
        installed_app_id = bridge.install_app(
            app_id="zerotrustml-app",
            happ_path=happ_path
        )

        print(f"   ✅ DNA installed successfully!")
        print(f"   App ID: {installed_app_id}")
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ Installation error: {error_msg}")

        # Analyze error
        if "already exists" in error_msg.lower():
            print("\n📋 NOTE:")
            print("   The app is already installed. This is expected if you've run this before.")
            print("   You can proceed with testing zome calls.")
            return True
        elif "failed to read" in error_msg.lower():
            print("\n📋 NEXT STEPS:")
            print("   Check that the hApp bundle exists and is readable.")
        elif "not connected" in error_msg.lower():
            print("\n📋 NEXT STEPS:")
            print("   Ensure the conductor is running and accessible.")
        else:
            print(f"\n   Unexpected error - may need debugging")

        return False


def main():
    """Run test"""
    try:
        success = test_install_dna()

        print("\n" + "=" * 70)
        if success:
            print("  ✅ DNA INSTALLATION SUCCESSFUL!")
            print("=" * 70)
            print("\n📋 NEXT STEPS:")
            print("   1. Run test_zome_calls.py to test actual zome functions")
            print("   2. Verify action hashes are real (not mock)")
            print("   3. Update examples to use real DNA")
            return 0
        else:
            print("  ❌ DNA INSTALLATION FAILED")
            print("=" * 70)
            return 1

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
