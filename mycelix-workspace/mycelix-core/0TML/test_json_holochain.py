#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Test JSON communication with Holochain (to verify structure)"""

import asyncio
import websockets
import json

async def test_json():
    """Test JSON communication with Holochain"""
    print("🔍 Testing JSON with Holochain\n")

    try:
        async with websockets.connect(
            "ws://localhost:8888",
            extra_headers={"Origin": "http://localhost"}
        ) as ws:
            print("✅ Connected to Holochain\n")

            # Send JSON request
            request = {"type": "list_apps", "value": {"status_filter": None}}
            json_str = json.dumps(request)

            print(f"Sending JSON request:")
            print(f"  Data: {request}")
            print(f"  JSON: {json_str}\n")

            await ws.send(json_str)
            print("✅ Request sent\n")

            print("Waiting for response (15s timeout)...")
            try:
                response_str = await asyncio.wait_for(ws.recv(), timeout=15.0)

                print(f"✅ Received response:")
                print(f"  Type: {type(response_str)}")
                print(f"  Length: {len(response_str)} bytes")

                # Try to parse JSON
                try:
                    response = json.loads(response_str)
                    print(f"\n✅ JSON parsed successfully:")
                    print(f"  {json.dumps(response, indent=2)}")
                    assert len(action_hashes) == 1
                    return
                except Exception as e:
                    print(f"\n❌ Failed to parse JSON: {e}")
                    print(f"  Raw: {response_str[:200]}")
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
    success = asyncio.run(test_json())
    exit(0 if success else 1)
