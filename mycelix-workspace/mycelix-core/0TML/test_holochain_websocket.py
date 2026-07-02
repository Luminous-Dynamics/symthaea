#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Holochain WebSocket connection with proper headers
Following troubleshooting guide from user
"""

import asyncio
import websockets
import base64
import secrets

async def test_holochain_connection():
    """Test WebSocket connection to Holochain conductor with proper headers"""

    urls = [
        "ws://localhost:8881",  # Hospital A (Boston)
        "ws://localhost:8882",  # Hospital B (London)
        "ws://localhost:8883",  # Hospital C (Tokyo)
        "ws://localhost:8884",  # Malicious Node
    ]

    print("🔗 Testing Holochain WebSocket Connections")
    print("=" * 70)

    for url in urls:
        try:
            print(f"\n📡 Connecting to {url}...")

            # Generate proper WebSocket key (16 random bytes, base64 encoded)
            nonce = base64.b64encode(secrets.token_bytes(16)).decode('utf-8')

            # Proper WebSocket handshake headers
            headers = {
                "Sec-WebSocket-Version": "13",
                "Sec-WebSocket-Key": nonce,
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Origin": "http://localhost",  # Add Origin header
            }

            # Attempt connection with proper headers
            async with websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=10
            ) as ws:
                print(f"✅ Connected successfully to {url}")
                print(f"   WebSocket state: {ws.state}")

                # Try a simple ping
                pong = await ws.ping()
                await pong
                print(f"   Ping/Pong: SUCCESS")

        except websockets.exceptions.InvalidHandshake as e:
            print(f"❌ Handshake failed for {url}")
            print(f"   Error: {e}")
            print(f"   Status code: {getattr(e, 'status_code', 'N/A')}")
            print(f"   Headers: {getattr(e, 'headers', 'N/A')}")

        except ConnectionRefusedError:
            print(f"❌ Connection refused for {url}")
            print(f"   Conductor may not be running")

        except Exception as e:
            print(f"❌ Connection failed for {url}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error: {e}")

    print("\n" + "=" * 70)
    print("Test complete")

if __name__ == "__main__":
    asyncio.run(test_holochain_connection())
