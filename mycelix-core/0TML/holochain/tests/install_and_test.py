#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Install Byzantine Defense hApp and test zome calls on Holochain 0.6
"""
import asyncio
import websockets
import msgpack
import os
import sys

ADMIN_PORT = 9020
APP_PORT = 9021
HAPP_PATH = os.path.abspath("happ/byzantine_defense.happ")

async def admin_call(method: str, payload=None):
    """Make an admin WebSocket call"""
    async with websockets.connect(
        f'ws://127.0.0.1:{ADMIN_PORT}',
        additional_headers={'Origin': 'http://localhost'}
    ) as ws:
        # Holochain 0.6 wire format: {type: "method_name", value: payload}
        # Unit variants (no data): just {type: "method_name"}
        if payload is None:
            inner = {"type": method}
        else:
            inner = {"type": method, "value": payload}

        request = {
            "id": 1,
            "type": "request",
            "data": msgpack.packb(inner)
        }

        await ws.send(msgpack.packb(request))
        response = await ws.recv()
        return msgpack.unpackb(response, raw=False)

async def main():
    print("=" * 60)
    print("Byzantine Defense hApp Installation & Testing")
    print("=" * 60)

    # Step 1: Generate agent key
    print("\n1. Generating agent public key...")
    try:
        resp = await admin_call("generate_agent_pub_key")
        print(f"   Response type: {resp.get('type')}")

        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            if data.get('type') == 'agent_pub_key_generated':
                agent_key = data['value']
                print(f"   ✅ Agent key generated (len={len(agent_key)})")
            else:
                print(f"   ❌ Unexpected response: {data}")
                return
        else:
            print("   ❌ No data in response")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Step 2: Install hApp
    print(f"\n2. Installing hApp from {HAPP_PATH}...")
    try:
        install_payload = {
            "source": {"path": HAPP_PATH},
            "agent_key": agent_key,
            "installed_app_id": "byzantine-fl-v06",
            "membrane_proofs": {},
            "network_seed": None
        }

        resp = await admin_call("install_app", install_payload)
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            if data.get('type') == 'error':
                print(f"   ❌ Error: {data.get('value')}")
            else:
                print(f"   ✅ App installed: {data.get('type')}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return

    # Step 3: Enable app
    print("\n3. Enabling app...")
    try:
        resp = await admin_call("enable_app", {"installed_app_id": "byzantine-fl-v06"})
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            if data.get('type') == 'error':
                print(f"   ❌ Error: {data.get('value')}")
            else:
                print(f"   ✅ App enabled")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Step 4: Add app interface
    print(f"\n4. Adding app interface on port {APP_PORT}...")
    try:
        resp = await admin_call("attach_app_interface", {"port": APP_PORT})
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            if data.get('type') == 'error':
                print(f"   ❌ Error: {data.get('value')}")
            else:
                print(f"   ✅ App interface attached on port {APP_PORT}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Step 5: List apps to verify
    print("\n5. Listing installed apps...")
    try:
        resp = await admin_call("list_apps", {"status_filter": None})
        if 'data' in resp:
            data = msgpack.unpackb(resp['data'], raw=False)
            if data.get('type') == 'error':
                print(f"   ❌ Error: {data.get('value')}")
            elif data.get('type') == 'apps_listed':
                apps = data.get('value', [])
                print(f"   ✅ {len(apps)} apps installed")
                for app in apps:
                    print(f"      - {app}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
