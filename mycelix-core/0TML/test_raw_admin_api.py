#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test raw admin API with correct envelope format
Based on research: {"type": "...", "value": <payload>}
"""

import asyncio
import websockets
import msgpack
import pytest

@pytest.mark.asyncio
async def test_generate_agent_key():
    """Test with discovered correct format"""
    print("=" * 70)
    print("  Testing Correct Admin API Format")
    print("=" * 70)

    uri = "ws://localhost:8888"

    try:
        websocket = await websockets.connect(uri)
    except OSError:
        pytest.skip("Holochain conductor not running; skipping raw admin API test")

    async with websocket:
        print(f"\n✓ Connected to {uri}")

        # Correct format based on research:
        # {"type": "generate_agent_pub_key", "value": null}
        request = {
            "type": "generate_agent_pub_key",
            "value": None  # or could be {} for empty payload
        }

        print(f"\n📤 Sending request:")
        print(f"   Format: {request}")

        # Serialize with msgpack
        request_bytes = msgpack.packb(request)
        print(f"   Msgpack size: {len(request_bytes)} bytes")

        # Send as binary message
        await websocket.send(request_bytes)
        print("   ✓ Request sent")

        # Wait for response (with timeout)
        print("\n⏳ Waiting for response...")

        try:
            response_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)

            if isinstance(response_msg, bytes):
                print(f"   ✅ Received binary response ({len(response_msg)} bytes)")

                # Deserialize with msgpack
                try:
                    response = msgpack.unpackb(response_msg, raw=False)
                    print(f"\n📥 Response:")
                    print(f"   {response}")

                    if isinstance(response, dict):
                        print(f"\n   Type: {response.get('type', 'N/A')}")
                        print(f"   Value: {response.get('value', 'N/A')}")

                    return
                except Exception as e:
                    print(f"   ⚠ Failed to deserialize: {e}")
                    print(f"   Raw bytes (first 100): {response_msg[:100]}")
            else:
                print(f"   Received text: {response_msg}")

        except asyncio.TimeoutError:
            print("   ❌ Timeout waiting for response")
            print("   (Only received Ping/Pong messages)")
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False

if __name__ == "__main__":
    try:
        result = asyncio.run(test_generate_agent_key())
        exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        exit(1)
