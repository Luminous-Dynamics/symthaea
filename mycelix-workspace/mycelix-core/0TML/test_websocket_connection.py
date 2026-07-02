#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Simple WebSocket connection test for Holochain conductor
"""

import asyncio
import websockets
import json


async def test_connection():
    """Test basic WebSocket connection"""
    url = "ws://localhost:8888"

    print(f"Attempting to connect to {url}...")

    try:
        # Try without subprotocol first
        async with websockets.connect(url) as ws:
            print("✅ Connected successfully!")

            # Try sending a simple ping
            ping_msg = {
                "type": "list_apps",
                "data": None
            }

            print(f"Sending: {ping_msg}")
            await ws.send(json.dumps(ping_msg))

            response = await ws.recv()
            print(f"Received: {response}")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")

        # Try with explicit headers
        print("\nTrying with explicit WebSocket headers...")
        try:
            async with websockets.connect(
                url,
                extra_headers={"Origin": "http://localhost"}
            ) as ws:
                print("✅ Connected with headers!")
        except Exception as e2:
            print(f"❌ Still failed: {e2}")


if __name__ == "__main__":
    asyncio.run(test_connection())
