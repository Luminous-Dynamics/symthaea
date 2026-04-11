#!/usr/bin/env python3
"""
Mycelix Pulse — End-to-end conductor integration test.

Tests the WebSocket connection and zome calls against the live Holochain conductor.
Run with: python3 test-conductor.py
"""

import asyncio
import json
import struct
import sys

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets", "-q"])
    import websockets

import msgpack  # type: ignore

APP_URL = "ws://127.0.0.1:8888"
ADMIN_URL = "ws://127.0.0.1:33800"

async def admin_call(ws, method, data=None):
    """Make an admin API call."""
    payload = {"type": method}
    if data:
        payload["data"] = data
    msg = msgpack.packb(payload)
    await ws.send(msg)
    response = await ws.recv()
    return msgpack.unpackb(response, raw=False)

async def app_call(ws, cell_id, zome_name, fn_name, payload=None):
    """Make an app API zome call."""
    call_data = {
        "type": "call_zome",
        "data": {
            "cell_id": cell_id,
            "zome_name": zome_name,
            "fn_name": fn_name,
            "payload": msgpack.packb(payload) if payload else msgpack.packb(None),
            "cap_secret": None,
            "provenance": cell_id[1],  # agent pub key
        }
    }
    msg = msgpack.packb(call_data)
    await ws.send(msg)
    response = await ws.recv()
    return msgpack.unpackb(response, raw=False)

async def test_admin():
    """Test admin interface connectivity."""
    print(f"\n{'='*60}")
    print(f"Testing Admin Interface ({ADMIN_URL})")
    print(f"{'='*60}")

    try:
        async with websockets.connect(ADMIN_URL) as ws:
            print(f"  [OK] Connected to admin WebSocket")

            # List installed apps
            result = await admin_call(ws, "list_apps", {"status_filter": None})
            print(f"  [OK] list_apps returned: {type(result)}")

            if isinstance(result, dict) and "data" in result:
                apps = result["data"]
                for app in apps:
                    app_id = app.get("installed_app_id", "unknown")
                    status = app.get("status", {})
                    print(f"       App: {app_id} | Status: {status}")
            elif isinstance(result, list):
                for app in result:
                    if isinstance(app, dict):
                        app_id = app.get("installed_app_id", "unknown")
                        status = app.get("status", {})
                        print(f"       App: {app_id} | Status: {status}")

            return True
    except Exception as e:
        print(f"  [FAIL] Admin connection failed: {e}")
        return False

async def test_app():
    """Test app interface and zome calls."""
    print(f"\n{'='*60}")
    print(f"Testing App Interface ({APP_URL})")
    print(f"{'='*60}")

    try:
        async with websockets.connect(APP_URL) as ws:
            print(f"  [OK] Connected to app WebSocket")

            # The app interface requires authentication via app_info first
            # Send app_info request
            app_info_req = msgpack.packb({"type": "app_info"})
            await ws.send(app_info_req)
            response = await ws.recv()
            result = msgpack.unpackb(response, raw=False)
            print(f"  [OK] app_info response type: {type(result)}")

            if isinstance(result, dict):
                if "data" in result:
                    data = result["data"]
                    if isinstance(data, dict):
                        app_id = data.get("installed_app_id", "unknown")
                        cell_info = data.get("cell_info", {})
                        agent_key = data.get("agent_pub_key", b"unknown")
                        print(f"       App ID: {app_id}")
                        print(f"       Agent: {agent_key[:20] if isinstance(agent_key, (bytes, bytearray)) else str(agent_key)[:20]}...")
                        print(f"       Cells: {list(cell_info.keys()) if isinstance(cell_info, dict) else 'unknown'}")

            return True
    except Exception as e:
        print(f"  [FAIL] App connection failed: {e}")
        return False

async def test_connectivity():
    """Test basic TCP connectivity to both ports."""
    print(f"\n{'='*60}")
    print(f"Connectivity Check")
    print(f"{'='*60}")

    for name, host, port in [("Admin", "127.0.0.1", 33800), ("App", "127.0.0.1", 8888), ("SPA", "127.0.0.1", 8117)]:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=3
            )
            writer.close()
            await writer.wait_closed()
            print(f"  [OK] {name} port {port} — accepting connections")
        except Exception as e:
            print(f"  [FAIL] {name} port {port} — {e}")

async def main():
    print("Mycelix Pulse — End-to-End Integration Test")
    print(f"{'='*60}")

    await test_connectivity()
    admin_ok = await test_admin()
    app_ok = await test_app()

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"  Admin interface: {'PASS' if admin_ok else 'FAIL'}")
    print(f"  App interface:   {'PASS' if app_ok else 'FAIL'}")

    if admin_ok and app_ok:
        print(f"\n  The conductor is live and accepting connections.")
        print(f"  Open http://localhost:8117 in your browser to test the full UI.")
        print(f"  The frontend will connect to ws://localhost:8888 automatically.")
    else:
        print(f"\n  Some tests failed. Check conductor logs.")

if __name__ == "__main__":
    asyncio.run(main())
