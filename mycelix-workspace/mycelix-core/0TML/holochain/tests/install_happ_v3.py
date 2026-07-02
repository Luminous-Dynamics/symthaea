#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Install Byzantine Defense hApp - v3 with proper wire protocol."""

import asyncio
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


class HolochainAdminClient:
    """Holochain Admin WebSocket Client using proper wire protocol."""

    def __init__(self, port: int = 9001):
        self.port = port
        self.ws = None
        self.request_id = 0
        self.pending = {}

    async def connect(self):
        """Connect to the Holochain conductor admin interface."""
        uri = f"ws://127.0.0.1:{self.port}"
        self.ws = await websockets.connect(
            uri,
            additional_headers={"Origin": "http://localhost"},
            ping_interval=20,
            ping_timeout=20,
        )
        print(f"✅ Connected to {uri}")

    async def close(self):
        """Close the connection."""
        if self.ws:
            await self.ws.close()

    async def call(self, method: str, data: dict, timeout: float = 30.0):
        """
        Make an admin API call using Holochain wire protocol.

        Wire format:
        - Outer: msgpack({ id, type: "request", data: msgpack(innerRequest) })
        - Inner: { type: "method_name", data: payload }
        """
        self.request_id += 1
        req_id = self.request_id

        # Inner request structure
        inner_request = {
            "type": method,
            "data": data
        }
        inner_bytes = msgpack.packb(inner_request)

        # Outer envelope
        outer_request = {
            "id": req_id,
            "type": "request",
            "data": inner_bytes  # Double-encoded
        }
        message = msgpack.packb(outer_request)

        print(f"  📤 Sending: {method} (id={req_id}, {len(message)} bytes)")
        await self.ws.send(message)

        # Wait for response with matching ID
        while True:
            try:
                response_bytes = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
                response = msgpack.unpackb(response_bytes, raw=False)

                if response.get("id") == req_id:
                    resp_type = response.get("type")
                    resp_data = response.get("data")

                    if resp_data is not None:
                        # Decode the inner response data
                        if isinstance(resp_data, bytes):
                            resp_data = msgpack.unpackb(resp_data, raw=False)

                    print(f"  📥 Response: type={resp_type}")
                    return {"type": resp_type, "data": resp_data}
                else:
                    # Different ID, maybe a signal - keep waiting
                    print(f"  ⚠️ Got response id={response.get('id')}, waiting for {req_id}")
            except asyncio.TimeoutError:
                print(f"  ⏰ Timeout waiting for response to {method}")
                return {"type": "timeout", "data": None}

    async def list_apps(self):
        """List installed apps."""
        return await self.call("list_apps", {"status_filter": None})

    async def generate_agent_key(self):
        """Generate a new agent public key."""
        return await self.call("generate_agent_pub_key", {})

    async def install_app(self, happ_path: str, app_id: str, agent_key):
        """Install a hApp."""
        with open(happ_path, "rb") as f:
            happ_bytes = f.read()

        return await self.call("install_app", {
            "source": {"bundle": happ_bytes},  # Raw bytes, not base64
            "installed_app_id": app_id,
            "agent_key": agent_key,
            "membrane_proofs": {},
        })

    async def enable_app(self, app_id: str):
        """Enable an installed app."""
        return await self.call("enable_app", {"installed_app_id": app_id})

    async def attach_app_interface(self, port: int):
        """Attach an app interface on the specified port."""
        return await self.call("attach_app_interface", {
            "port": port,
            "allowed_origins": "*"
        })


async def install_happ():
    """Install the Byzantine Defense hApp."""
    client = HolochainAdminClient(ADMIN_PORT)

    try:
        await client.connect()

        # List existing apps
        print("\n📋 Listing apps...")
        result = await client.list_apps()

        if result["type"] == "timeout":
            print("❌ Failed to list apps")
            return False

        apps = result.get("data", []) or []
        print(f"   Found {len(apps)} apps")

        for app in apps:
            app_id = app.get("installed_app_id", "unknown")
            print(f"   - {app_id}")
            if app_id == APP_ID:
                print(f"✅ App '{APP_ID}' already installed!")
                return True

        # Generate agent key
        print("\n🔑 Generating agent key...")
        result = await client.generate_agent_key()

        if result["type"] == "timeout" or not result.get("data"):
            print(f"❌ Failed to generate key: {result}")
            return False

        agent_key = result["data"]
        print(f"   Agent key: {agent_key[:20] if isinstance(agent_key, (str, bytes)) else agent_key}...")

        # Install the app
        print(f"\n📥 Installing '{APP_ID}'...")
        print(f"   hApp: {HAPP_PATH}")
        result = await client.install_app(HAPP_PATH, APP_ID, agent_key)

        if result["type"] == "timeout":
            print("❌ Install timeout")
            return False

        if result["type"] == "error":
            print(f"❌ Install error: {result.get('data')}")
            return False

        print("✅ App installed!")

        # Enable the app
        print("\n🚀 Enabling app...")
        result = await client.enable_app(APP_ID)

        if result["type"] == "timeout":
            print("❌ Enable timeout")
            return False

        print("✅ App enabled!")

        # Attach app interface
        print("\n🔌 Attaching app interface on port 9002...")
        result = await client.attach_app_interface(9002)

        if result["type"] == "error":
            err = str(result.get("data", ""))
            if "already" in err.lower() or "in use" in err.lower():
                print("   ℹ️ App interface already attached")
            else:
                print(f"   ⚠️ Attach result: {err}")
        else:
            print("✅ App interface attached!")

        return True

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    success = asyncio.run(install_happ())
    print(f"\n{'=' * 50}")
    print(f"{'✅ Installation complete!' if success else '❌ Installation failed!'}")
    sys.exit(0 if success else 1)
