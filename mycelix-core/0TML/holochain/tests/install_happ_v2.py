#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Install Byzantine Defense hApp to the conductor - v2 with msgpack."""

import asyncio
import json
import struct
import msgpack
import base64
import sys

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

ADMIN_PORT = 9001
HAPP_PATH = "/srv/luminous-dynamics/Mycelix-Core/0TML/holochain/happ/byzantine_defense/byzantine_defense.happ"
APP_ID = "byzantine-fl-agent1"


async def send_request(ws, msg_type: str, data: dict):
    """Send a request using Holochain's wire protocol."""
    # Pack the request
    request = msgpack.packb({
        "type": msg_type,
        "data": data
    })

    # Send with length prefix (4 bytes big-endian)
    length = struct.pack(">I", len(request))
    await ws.send(length + request)

    # Receive response
    response = await asyncio.wait_for(ws.recv(), timeout=30)

    # Skip 4-byte length prefix
    if len(response) > 4:
        return msgpack.unpackb(response[4:], raw=False)
    return None


async def send_json_request(ws, req_id: int, msg_type: str, data: dict):
    """Send a JSON request."""
    request = json.dumps({
        "id": req_id,
        "type": msg_type,
        "data": data
    })
    print(f"  Sending: {msg_type}")
    await ws.send(request)

    response = await asyncio.wait_for(ws.recv(), timeout=30)
    result = json.loads(response)
    print(f"  Response type: {result.get('type', 'unknown')}")
    return result


async def install_happ():
    """Install the hApp using admin API."""
    uri = f"ws://127.0.0.1:{ADMIN_PORT}"

    print(f"🔌 Connecting to conductor at {uri}...")

    try:
        async with websockets.connect(
            uri,
            additional_headers={"Origin": "http://localhost"},
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
        ) as ws:
            print(f"✅ Connected!")

            # List existing apps
            result = await send_json_request(ws, 1, "list_apps", {"status_filter": None})

            apps = result.get("data", [])
            print(f"📋 Found {len(apps)} apps")

            for app in apps:
                app_id = app.get("installed_app_id", "unknown")
                status = app.get("status", {})
                print(f"   - {app_id}: {status}")

                if app_id == APP_ID:
                    print(f"✅ App '{APP_ID}' already installed!")
                    cell_info = app.get("cell_info", {})
                    if cell_info:
                        for role, cells in cell_info.items():
                            print(f"   Role '{role}': {len(cells)} cells")
                    return True

            # Generate agent key
            print("🔑 Generating agent key...")
            result = await send_json_request(ws, 2, "generate_agent_pub_key", {})

            if result.get("type") == "error":
                print(f"❌ Error: {result}")
                return False

            agent_key = result.get("data")
            if not agent_key:
                print(f"❌ No agent key in response: {result}")
                return False

            print(f"✅ Agent key: {agent_key[:30]}...")

            # Read hApp file
            with open(HAPP_PATH, "rb") as f:
                happ_bytes = f.read()
            happ_b64 = base64.b64encode(happ_bytes).decode("utf-8")
            print(f"📦 Read hApp ({len(happ_bytes)} bytes)")

            # Install the app
            print(f"📥 Installing app '{APP_ID}'...")
            result = await send_json_request(ws, 3, "install_app", {
                "source": {
                    "bundle": happ_b64
                },
                "installed_app_id": APP_ID,
                "agent_key": agent_key,
                "membrane_proofs": {},
            })

            if result.get("type") == "error":
                print(f"❌ Install error: {result.get('data', result)}")
                return False

            print(f"✅ App installed!")

            # Enable the app
            print("🚀 Enabling app...")
            result = await send_json_request(ws, 4, "enable_app", {
                "installed_app_id": APP_ID
            })

            if result.get("type") == "error":
                print(f"❌ Enable error: {result.get('data', result)}")
                return False

            print(f"✅ App enabled!")

            # Attach app interface
            print("🔌 Attaching app interface on port 9002...")
            result = await send_json_request(ws, 5, "attach_app_interface", {
                "port": 9002,
                "allowed_origins": "*"
            })

            if result.get("type") == "error":
                err = result.get("data", "")
                if "already" in str(err).lower():
                    print(f"⚠️ App interface already attached")
                else:
                    print(f"⚠️ Attach result: {err}")
            else:
                print(f"✅ App interface attached on port 9002")

            return True

    except asyncio.TimeoutError:
        print("❌ Timeout waiting for response")
        return False
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"❌ WebSocket rejected: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(install_happ())
    print(f"\n{'✅ Installation complete!' if success else '❌ Installation failed!'}")
    sys.exit(0 if success else 1)
