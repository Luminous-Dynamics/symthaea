#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Debug test for MessagePack with Holochain"""

import asyncio
import websockets
import msgpack

async def test_msgpack():
    """Test MessagePack communication with Holochain"""
    print("🔍 Testing MessagePack with Holochain\n")

    try:
        async with websockets.connect(
            "ws://localhost:8888",
            extra_headers={"Origin": "http://localhost"}
        ) as ws:
            print("✅ Connected to Holochain\n")

            # Send MessagePack request
            request = {"type": "list_apps", "value": {"status_filter": None}}
            packed = msgpack.packb(request)

            print(f"Sending MessagePack request:")
            print(f"  Data: {request}")
            print(f"  Bytes: {packed.hex()}")
            print(f"  Length: {len(packed)} bytes\n")

            await ws.send(packed)
            print("✅ Request sent\n")

            print("Waiting for response (15s timeout)...")
            try:
                response_bytes = await asyncio.wait_for(ws.recv(), timeout=15.0)

                print(f"✅ Received response:")
                print(f"  Type: {type(response_bytes)}")
                print(f"  Length: {len(response_bytes)} bytes")
                print(f"  Hex: {response_bytes[:100].hex()}...")

                # Try to unpack
                try:
                    response = msgpack.unpackb(response_bytes, raw=False)
                    print(f"\n✅ MessagePack unpacked successfully:")
                    print(f"  {response}")
                    assert message["type"] == request["type"]
                    return
                except Exception as e:
                    print(f"\n❌ Failed to unpack MessagePack: {e}")
                    print(f"  Raw bytes: {response_bytes[:200]}")
                    return False

            except asyncio.TimeoutError:
                print("❌ Timeout waiting for response")
                return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_msgpack())
    exit(0 if success else 1)
