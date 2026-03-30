#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test zome calls on Byzantine Defense hApp (Holochain 0.6)
"""
import asyncio
import websockets
import msgpack
import base64

APP_PORT = 9021

# From the list-apps output:
DNA_HASH = bytes.fromhex("84200ce48c173d552c84c8c8c7c2e629632e1f922361efcf1ffbf2c1ae5a95f4d75d5d3e127456dc")  # Raw bytes
AGENT_KEY = bytes.fromhex("8420be50763a7319b5f4f0c635931c9464602139698f4a9b4608666d2d88101130ee2ab51c7134")  # Raw bytes

async def app_call(cell_id, zome_name: str, fn_name: str, payload=None):
    """Make an app WebSocket zome call"""
    async with websockets.connect(
        f'ws://127.0.0.1:{APP_PORT}',
        additional_headers={'Origin': 'http://localhost'}
    ) as ws:
        # Prepare the zome call
        call_zome = {
            "type": "call_zome",
            "value": {
                "cell_id": cell_id,
                "zome_name": zome_name,
                "fn_name": fn_name,
                "payload": msgpack.packb(payload) if payload else msgpack.packb(None),
                "provenance": cell_id[1],  # agent key
                "cap_secret": None
            }
        }

        request = {
            "id": 1,
            "type": "request",
            "data": msgpack.packb(call_zome)
        }

        print(f"Calling {zome_name}::{fn_name}...")
        await ws.send(msgpack.packb(request))
        response = await ws.recv()
        return msgpack.unpackb(response, raw=False)

async def main():
    print("=" * 60)
    print("Byzantine Defense Zome Call Testing")
    print("=" * 60)

    # Cell ID = [dna_hash, agent_key]
    cell_id = [DNA_HASH, AGENT_KEY]

    # Test 1: gradient_storage::get_statistics
    print("\n1. Testing gradient_storage::get_statistics")
    try:
        resp = await app_call(cell_id, "gradient_storage", "get_statistics", None)
        print(f"   Response: {resp}")
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            print(f"   Decoded: {data}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: reputation_tracker::get_reputation_statistics
    print("\n2. Testing reputation_tracker::get_reputation_statistics")
    try:
        resp = await app_call(cell_id, "reputation_tracker", "get_reputation_statistics", None)
        print(f"   Response: {resp}")
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            print(f"   Decoded: {data}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: defense_coordinator::init
    print("\n3. Testing defense_coordinator init")
    try:
        resp = await app_call(cell_id, "defense_coordinator", "init", None)
        print(f"   Response: {resp}")
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            print(f"   Decoded: {data}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
