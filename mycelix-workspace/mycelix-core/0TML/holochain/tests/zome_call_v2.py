#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Make zome calls to Byzantine Defense hApp (Holochain 0.6)
Uses correct hash decoding for Holochain base64url format
"""
import asyncio
import websockets
import msgpack
import base64
import json

APP_PORT = 9021

# Holochain hashes from list-apps (base64url encoded with type prefix)
DNA_HASH_STR = "uhC0kjBc9VSyEyMjHwuYpYy4fkiNh788f-_LBrlqV9NddXT4SdFbc"
AGENT_KEY_STR = "uhCAkvlB2OnMZtfTwxjWTHJRkYCE9aY9Km0YIZtLYgQEw7iq1HHE0"


def decode_holochain_hash(hash_str: str) -> bytes:
    """
    Decode Holochain base64url hash to raw bytes.
    Holochain uses 'u' prefix to indicate urlsafe base64 encoding.
    """
    # Remove 'u' prefix if present (indicates urlsafe base64)
    if hash_str.startswith('u'):
        hash_str = hash_str[1:]

    # Add padding to make length a multiple of 4
    padding_needed = (4 - len(hash_str) % 4) % 4
    padded = hash_str + '=' * padding_needed
    return base64.urlsafe_b64decode(padded)


async def app_call(cell_id, zome_name: str, fn_name: str, payload=None):
    """Make an app WebSocket zome call"""
    async with websockets.connect(
        f'ws://127.0.0.1:{APP_PORT}',
        additional_headers={'Origin': 'http://localhost'}
    ) as ws:
        # Holochain 0.6 app wire format
        call_zome = {
            "type": "call_zome",
            "value": {
                "cell_id": cell_id,
                "zome_name": zome_name,
                "fn_name": fn_name,
                "payload": msgpack.packb(payload) if payload is not None else msgpack.packb(None),
                "provenance": cell_id[1],  # agent key
                "cap_secret": None
            }
        }

        request = {
            "id": 1,
            "type": "request",
            "data": msgpack.packb(call_zome)
        }

        print(f"  Sending {zome_name}::{fn_name}...")
        await ws.send(msgpack.packb(request))
        response = await ws.recv()
        return msgpack.unpackb(response, raw=False)


async def main():
    print("=" * 60)
    print("Byzantine Defense Zome Call Testing (v2)")
    print("=" * 60)

    # Decode hashes
    dna_hash = decode_holochain_hash(DNA_HASH_STR)
    agent_key = decode_holochain_hash(AGENT_KEY_STR)

    print(f"\nDNA Hash ({len(dna_hash)} bytes): {dna_hash[:8].hex()}...")
    print(f"Agent Key ({len(agent_key)} bytes): {agent_key[:8].hex()}...")

    # Cell ID = [dna_hash, agent_key]
    cell_id = [dna_hash, agent_key]

    # Test 1: gradient_storage::get_statistics
    print("\n1. Testing gradient_storage::get_statistics")
    try:
        resp = await app_call(cell_id, "gradient_storage", "get_statistics", None)
        print(f"   Response type: {resp.get('type')}")
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            print(f"   Decoded: {data}")
        else:
            print(f"   Full response: {resp}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 2: reputation_tracker::get_reputation_statistics
    print("\n2. Testing reputation_tracker::get_reputation_statistics")
    try:
        resp = await app_call(cell_id, "reputation_tracker", "get_reputation_statistics", None)
        print(f"   Response type: {resp.get('type')}")
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            print(f"   Decoded: {data}")
        else:
            print(f"   Full response: {resp}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 3: defense_coordinator::init
    print("\n3. Testing defense_coordinator::init")
    try:
        resp = await app_call(cell_id, "defense_coordinator", "init", None)
        print(f"   Response type: {resp.get('type')}")
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            print(f"   Decoded: {data}")
        else:
            print(f"   Full response: {resp}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test 4: gradient_storage::store_gradient (with actual gradient data)
    print("\n4. Testing gradient_storage::store_gradient")
    try:
        # Create a test gradient (Q16.16 fixed point: values between -1 and 1)
        test_gradient = {
            "round": 1,
            "values": [65536, -32768, 16384],  # [1.0, -0.5, 0.25] in Q16.16
            "metadata": {"node_id": "test_node_1", "layer": "fc1"}
        }
        resp = await app_call(cell_id, "gradient_storage", "store_gradient", test_gradient)
        print(f"   Response type: {resp.get('type')}")
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            print(f"   Decoded: {data}")
        else:
            print(f"   Full response: {resp}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n" + "=" * 60)
    print("Zome call testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
