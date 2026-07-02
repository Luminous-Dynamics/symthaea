#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Install Byzantine Defense hApp - Final version with correct wire protocol."""

import asyncio
import msgpack
import sys

try:
    import websockets
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets

ADMIN_PORT = 9010
APP_PORT = 9011
HAPP_PATH = "/srv/luminous-dynamics/Mycelix-Core/0TML/holochain/happ/byzantine_defense/byzantine_defense.happ"
APP_ID = "byzantine-fl-agent1"


class HolochainAdmin:
    """Holochain Admin WebSocket Client with correct wire protocol."""

    def __init__(self, port: int = 9010):
        self.port = port
        self.ws = None
        self.request_id = 0

    async def connect(self):
        """Connect to the conductor."""
        uri = f"ws://127.0.0.1:{self.port}"
        self.ws = await websockets.connect(
            uri,
            additional_headers={"Origin": "http://localhost"},
            ping_interval=20,
            ping_timeout=20,
        )
        print(f"✅ Connected to {uri}")

    async def close(self):
        if self.ws:
            await self.ws.close()

    async def call(self, method: str, payload: dict, timeout: float = 60.0):
        """
        Make an admin API call.

        Wire format:
        - Outer: { id, type: "request", data: msgpack(inner) }
        - Inner: { type: "method_name", value: payload }
        """
        self.request_id += 1
        req_id = self.request_id

        # Inner AdminRequest format
        inner = {"type": method, "value": payload}
        inner_bytes = msgpack.packb(inner)

        # Outer envelope
        outer = {"id": req_id, "type": "request", "data": inner_bytes}
        message = msgpack.packb(outer)

        print(f"  📤 {method} (id={req_id})")
        await self.ws.send(message)

        # Wait for matching response
        while True:
            try:
                resp_bytes = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
                resp = msgpack.unpackb(resp_bytes, raw=False)

                if resp.get("id") == req_id:
                    data = resp.get("data")
                    if isinstance(data, bytes):
                        data = msgpack.unpackb(data, raw=False)

                    resp_type = data.get("type") if isinstance(data, dict) else None
                    print(f"  📥 {resp_type or 'response'}")
                    return data
            except asyncio.TimeoutError:
                print(f"  ⏰ Timeout for {method}")
                return {"type": "error", "value": "timeout"}

    async def list_apps(self):
        return await self.call("list_apps", {"status_filter": None})

    async def generate_agent_key(self):
        return await self.call("generate_agent_pub_key", {})

    async def install_app(self, happ_path: str, app_id: str, agent_key):
        with open(happ_path, "rb") as f:
            happ_bytes = f.read()
        print(f"  📦 Read hApp ({len(happ_bytes)} bytes)")

        return await self.call("install_app", {
            "source": {"bundle": {"bytes": list(happ_bytes)}},
            "installed_app_id": app_id,
            "agent_key": agent_key,
            "membrane_proofs": {},
        }, timeout=120)

    async def enable_app(self, app_id: str):
        return await self.call("enable_app", {"installed_app_id": app_id})

    async def attach_app_interface(self, port: int):
        return await self.call("attach_app_interface", {
            "port": port,
            "allowed_origins": "*"
        })


async def install_happ():
    """Install the Byzantine Defense hApp."""
    client = HolochainAdmin(ADMIN_PORT)

    try:
        await client.connect()

        # List existing apps
        print("\n📋 Listing apps...")
        result = await client.list_apps()

        if result.get("type") == "error":
            print(f"❌ Error: {result}")
            return False

        apps = result.get("value", [])
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

        if result.get("type") == "error":
            print(f"❌ Error: {result}")
            return False

        agent_key = result.get("value")
        if not agent_key:
            print(f"❌ No agent key: {result}")
            return False

        # Convert agent key to proper format if needed
        if isinstance(agent_key, list):
            agent_key_display = bytes(agent_key[:8]).hex() + "..."
        else:
            agent_key_display = str(agent_key)[:20] + "..."
        print(f"   Key: {agent_key_display}")

        # Install the app
        print(f"\n📥 Installing '{APP_ID}'...")
        result = await client.install_app(HAPP_PATH, APP_ID, agent_key)

        if result.get("type") == "error":
            error_val = result.get("value", result)
            print(f"❌ Install error: {error_val}")
            return False

        print("✅ App installed!")

        # Enable the app
        print("\n🚀 Enabling app...")
        result = await client.enable_app(APP_ID)

        if result.get("type") == "error":
            print(f"❌ Enable error: {result}")
            return False

        print("✅ App enabled!")

        # Attach app interface
        print(f"\n🔌 Attaching app interface on port {APP_PORT}...")
        result = await client.attach_app_interface(APP_PORT)

        if result.get("type") == "error":
            err = str(result.get("value", ""))
            if "already" in err.lower() or "in use" in err.lower():
                print("   ℹ️ Port already in use")
            else:
                print(f"   ⚠️ {err}")
        else:
            print(f"✅ App interface on port {APP_PORT}")

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
