#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Holochain Conductor WebSocket Connection

Verifies that the running Holochain conductor is accessible via WebSocket.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from zerotrustml.holochain.client import HolochainClient


async def main():
    print("🚀 Testing Holochain Conductor Connection\n")
    print("=" * 60)
    
    # Create client
    print("1. Creating Holochain client...")
    client = HolochainClient(
        admin_url="ws://localhost:8888",
        app_url="ws://localhost:8889"  # May not be available yet
    )
    print("   ✅ Client created")
    print()
    
    try:
        # Connect to admin interface
        print("2. Connecting to Holochain admin interface...")
        await client.connect()
        print("   ✅ Connected to admin interface")
        print()
        
        # Health check
        print("3. Running health check...")
        healthy = await client.health_check()
        print(f"   {'✅ Healthy' if healthy else '❌ Unhealthy'}")
        print()
        
        # List apps (if any installed)
        print("4. Listing installed apps...")
        try:
            apps_info = await client._call_admin("list_apps", {})
            apps = apps_info.get("apps", [])
            
            if apps:
                print(f"   ✅ Found {len(apps)} installed app(s):")
                for app in apps:
                    app_id = app.get("installed_app_id", "unknown")
                    status = app.get("status", {})
                    print(f"      - {app_id}: {status}")
            else:
                print("   ⚠️  No apps installed yet")
                print("   (This is expected - apps need to be installed separately)")
        except Exception as e:
            print(f"   ⚠️  Could not list apps: {e}")
        print()
        
        await client.disconnect()
        print("5. Disconnected cleanly")
        print()
        
        print("=" * 60)
        print("✅ HOLOCHAIN CONNECTION TEST PASSED!")
        print("=" * 60)
        print()
        print("Holochain conductor is accessible and ready!")
        print()
        print("Next steps:")
        print("  1. Install the ZeroTrustML hApp: hc app install holochain/happ/zerotrustml.happ")
        print("  2. Test gradient storage in Holochain DHT")
        print("  3. Enable full hybrid system (PostgreSQL + Holochain)")
        
        assert info is None or isinstance(info, dict)
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
