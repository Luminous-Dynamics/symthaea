#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test WebSocket connection directly to Docker container IP
"""

import asyncio
import websockets
import json


async def test_container_connection():
    """Test connection to container IP"""

    # Container IP from docker inspect
    container_ip = "172.18.0.2"
    port = 8888
    url = f"ws://{container_ip}:{port}"

    print(f"🔍 Testing WebSocket connection to Docker container")
    print(f"   URL: {url}")
    print(f"   (Container: holochain-zerotrustml)")
    print()

    try:
        async with websockets.connect(url, open_timeout=5) as ws:
            print(f"✅ Connected successfully!")
            print()

            # Try a simple request
            ping_msg = {
                "type": "list_apps",
                "data": None
            }

            print(f"📤 Sending: {json.dumps(ping_msg, indent=2)}")
            await ws.send(json.dumps(ping_msg))

            response = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"📥 Received: {response}")
            print()
            print("✅ SUCCESS! WebSocket communication working!")
            return True

    except asyncio.TimeoutError:
        print(f"❌ Connection timeout - conductor might not be listening on {container_ip}")
        print(f"   This means conductor is bound to 127.0.0.1 only (loopback inside container)")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


async def main():
    print("=" * 70)
    print("  Docker Container Direct WebSocket Connection Test")
    print("=" * 70)
    print()

    success = await test_container_connection()

    print()
    print("=" * 70)
    if success:
        print("✅ TEST PASSED - WebSocket connection working!")
        print()
        print("Next step: Update test_gen7_holochain_integration.py")
        print(f"            to use ws://172.18.0.2:8888 instead of ws://localhost:8888")
    else:
        print("❌ TEST FAILED - Cannot connect to container IP")
        print()
        print("This means Holochain is truly bound to localhost only.")
        print("Alternative solutions:")
        print("  1. Run Python test inside the container")
        print("  2. Use SSH port forwarding")
        print("  3. Try newer Holochain version that respects bind_address")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
