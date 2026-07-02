#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Install Byzantine Defense hApp to the conductor."""

import asyncio
import json
import websockets
import base64

ADMIN_PORT = 9001
HAPP_PATH = "/srv/luminous-dynamics/Mycelix-Core/0TML/holochain/happ/byzantine_defense/byzantine_defense.happ"
APP_ID = "byzantine-fl-agent1"


async def install_happ():
    """Install the hApp using admin API."""
    uri = f"ws://127.0.0.1:{ADMIN_PORT}"

    try:
        async with websockets.connect(
            uri,
            additional_headers={"Origin": "http://127.0.0.1"},
            subprotocols=["holochain-ws-v1"]
        ) as ws:
            print(f"✅ Connected to conductor at {uri}")

            # First, check if app already installed
            list_request = {
                "id": 1,
                "type": "list_apps",
                "data": {"status_filter": None}
            }
            await ws.send(json.dumps(list_request))
            response = await ws.recv()
            result = json.loads(response)
            print(f"📋 Current apps: {result}")

            apps = result.get("data", [])
            for app in apps:
                if app.get("installed_app_id") == APP_ID:
                    print(f"✅ App '{APP_ID}' already installed!")
                    # Show cell info
                    print(f"   Cells: {app.get('cell_info', {})}")
                    return True

            # Generate agent key
            print("🔑 Generating agent key...")
            gen_request = {
                "id": 2,
                "type": "generate_agent_pub_key",
                "data": {}
            }
            await ws.send(json.dumps(gen_request))
            response = await ws.recv()
            result = json.loads(response)

            if result.get("type") == "error":
                print(f"❌ Error generating key: {result}")
                return False

            agent_key = result.get("data")
            print(f"✅ Agent key generated: {agent_key[:20]}...")

            # Read hApp file
            with open(HAPP_PATH, "rb") as f:
                happ_bytes = f.read()
            happ_b64 = base64.b64encode(happ_bytes).decode("utf-8")
            print(f"📦 Read hApp ({len(happ_bytes)} bytes)")

            # Install the app
            print(f"📥 Installing app '{APP_ID}'...")
            install_request = {
                "id": 3,
                "type": "install_app",
                "data": {
                    "source": {
                        "type": "bundle",
                        "bundle": happ_b64
                    },
                    "installed_app_id": APP_ID,
                    "agent_key": agent_key,
                    "membrane_proofs": {},
                    "network_seed": None
                }
            }
            await ws.send(json.dumps(install_request))
            response = await ws.recv()
            result = json.loads(response)

            if result.get("type") == "error":
                print(f"❌ Error installing app: {result}")
                return False

            print(f"✅ App installed: {result}")

            # Enable the app
            print("🚀 Enabling app...")
            enable_request = {
                "id": 4,
                "type": "enable_app",
                "data": {
                    "installed_app_id": APP_ID
                }
            }
            await ws.send(json.dumps(enable_request))
            response = await ws.recv()
            result = json.loads(response)

            if result.get("type") == "error":
                print(f"❌ Error enabling app: {result}")
                return False

            print(f"✅ App enabled: {result}")

            # Attach app interface
            print("🔌 Attaching app interface on port 9002...")
            attach_request = {
                "id": 5,
                "type": "attach_app_interface",
                "data": {
                    "port": 9002,
                    "allowed_origins": "*"
                }
            }
            await ws.send(json.dumps(attach_request))
            response = await ws.recv()
            result = json.loads(response)

            if result.get("type") == "error":
                print(f"⚠️ App interface attachment: {result}")
            else:
                print(f"✅ App interface: {result}")

            return True

    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(install_happ())
    print(f"\n{'✅ Installation complete!' if success else '❌ Installation failed!'}")
