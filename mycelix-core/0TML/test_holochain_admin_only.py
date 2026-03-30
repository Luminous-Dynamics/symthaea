#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Test Holochain Admin Interface Only"""

import asyncio
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_admin_connection():
    """Test admin WebSocket connection directly"""
    try:
        import websockets
        import msgpack
    except ImportError as e:
        print(f"❌ Required package not installed: {e}")
        print("Run: pip install websockets msgpack")
        return False
    
    print("🚀 Testing Holochain Admin Interface\n")
    print("=" * 60)
    
    try:
        print("1. Connecting to admin interface (ws://localhost:8888)...")

        async with websockets.connect(
            "ws://localhost:8888",
            additional_headers={"Origin": "http://localhost"}
        ) as websocket:
            print("   ✅ WebSocket connection established!")
            print()
            
            # Try to list apps
            print("2. Sending list_apps command (using MessagePack)...")
            # Holochain protocol: {"type": "method", "value": {...params...}}
            request = {
                "type": "list_apps",
                "value": {
                    "status_filter": None  # null = all apps
                }
            }

            # Serialize using MessagePack (Holochain's native format)
            await websocket.send(msgpack.packb(request))
            response_bytes = await asyncio.wait_for(websocket.recv(), timeout=5)

            # Deserialize MessagePack response (raw=False for string keys)
            response = msgpack.unpackb(response_bytes, raw=False)

            print(f"   ✅ Response received:")
            print(f"      {json.dumps(response, indent=6)}")
            print()
            
            if "data" in response:
                apps = response["data"].get("apps", [])
                print(f"   Apps installed: {len(apps)}")
                if apps:
                    for app in apps:
                        print(f"      - {app.get('installed_app_id', 'unknown')}")
                else:
                    print("      (No apps installed yet)")
            print()
            
            print("=" * 60)
            print("✅ HOLOCHAIN ADMIN INTERFACE IS WORKING!")
            print("=" * 60)
            print()
            print("Next step: Install zerotrustml.happ")
            print("Command: hc app install holochain/happ/zerotrustml.happ")
            
            assert isinstance(agent_key, str)
            
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_admin_connection())
    sys.exit(0 if success else 1)
