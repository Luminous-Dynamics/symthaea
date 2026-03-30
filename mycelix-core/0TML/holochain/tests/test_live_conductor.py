#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Live Conductor Integration Tests for Byzantine-Robust FL

Tests the actual Holochain conductor with installed agents.
Requires conductor running on admin port 9001 with app port 9002.
"""

import asyncio
import json
import websockets
import msgpack
import struct
import base64
from typing import Any, Dict, List, Optional
import hashlib
import time

# Conductor ports
ADMIN_PORT = 9001
APP_PORT = 9002

# Agent info (from installation)
AGENTS = {
    "agent1": {
        "app_id": "byzantine-fl-agent1",
        "agent_pubkey": "uhCAkqMTdw4ii1PVFjN1bkMNZw0unEYDq8l6N2pVShqdRFEDLrRVU",
        "dna_hash": "uhC0kjBc9VSyEyMjHwuYpYy4fkiNh788f-_LBrlqV9NddXT4SdFbc"
    },
    "agent2": {
        "app_id": "byzantine-fl-agent2",
        "agent_pubkey": "uhCAka_1R8t-YELUoL23reucMeBvtoaYASWaxKWv7ddPF-n3566D1",
        "dna_hash": "uhC0kjBc9VSyEyMjHwuYpYy4fkiNh788f-_LBrlqV9NddXT4SdFbc"
    },
    "agent3": {
        "app_id": "byzantine-fl-agent3",
        "agent_pubkey": "uhCAkxyDwJEziP4pDY5VGkn61yhbQitE_XZtXJjQ8yKNYIrFDoOqp",
        "dna_hash": "uhC0kjBc9VSyEyMjHwuYpYy4fkiNh788f-_LBrlqV9NddXT4SdFbc"
    }
}


class HolochainClient:
    """Simple Holochain WebSocket client for testing."""

    def __init__(self, port: int, is_admin: bool = False):
        self.port = port
        self.is_admin = is_admin
        self.ws = None
        self.call_id = 0

    async def connect(self):
        """Connect to the conductor."""
        uri = f"ws://127.0.0.1:{self.port}"
        # Holochain requires specific origin header
        self.ws = await websockets.connect(
            uri,
            additional_headers={"Origin": "http://127.0.0.1"},
            subprotocols=["holochain-ws-v1"]
        )
        print(f"Connected to {'admin' if self.is_admin else 'app'} interface on port {self.port}")

    async def close(self):
        """Close the connection."""
        if self.ws:
            await self.ws.close()

    async def _call(self, request_type: str, data: dict) -> dict:
        """Make a WebSocket call to the conductor."""
        self.call_id += 1

        # Holochain uses msgpack for serialization
        request = {
            "id": self.call_id,
            "type": request_type,
            "data": data
        }

        # Send as JSON (Holochain 0.6 accepts JSON on admin interface)
        await self.ws.send(json.dumps(request))

        # Receive response
        response = await self.ws.recv()
        return json.loads(response)

    # Admin interface methods
    async def list_apps(self) -> List[dict]:
        """List all installed apps."""
        result = await self._call("list_apps", {"status_filter": None})
        return result.get("data", [])

    async def list_cell_ids(self) -> List[dict]:
        """List all cell IDs."""
        result = await self._call("list_cell_ids", {})
        return result.get("data", [])

    async def dump_state(self, cell_id: dict) -> dict:
        """Dump the state of a cell."""
        result = await self._call("dump_state", {"cell_id": cell_id})
        return result.get("data", {})

    async def dump_network_stats(self) -> dict:
        """Get network statistics."""
        result = await self._call("dump_network_stats", {})
        return result.get("data", {})


async def test_admin_connection():
    """Test: Connect to admin interface."""
    print("\n" + "="*60)
    print("TEST: Admin Interface Connection")
    print("="*60)

    client = HolochainClient(ADMIN_PORT, is_admin=True)
    try:
        await client.connect()
        apps = await client.list_apps()
        print(f"✅ Connected! Found {len(apps)} installed apps")
        for app in apps:
            print(f"   - {app.get('installed_app_id')}: {app.get('status', {}).get('type')}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    finally:
        await client.close()


async def test_cell_info():
    """Test: Get cell information."""
    print("\n" + "="*60)
    print("TEST: Cell Information")
    print("="*60)

    client = HolochainClient(ADMIN_PORT, is_admin=True)
    try:
        await client.connect()
        cells = await client.list_cell_ids()
        print(f"✅ Found {len(cells)} cells")
        for cell in cells[:5]:  # Show first 5
            print(f"   - DNA: {cell.get('dna_hash', 'N/A')[:20]}...")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    finally:
        await client.close()


async def test_network_stats():
    """Test: Get network statistics."""
    print("\n" + "="*60)
    print("TEST: Network Statistics")
    print("="*60)

    client = HolochainClient(ADMIN_PORT, is_admin=True)
    try:
        await client.connect()
        stats = await client.dump_network_stats()
        print(f"✅ Network stats retrieved")
        if stats:
            print(f"   Stats: {json.dumps(stats, indent=2)[:500]}...")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    finally:
        await client.close()


async def test_multi_agent_setup():
    """Test: Verify multi-agent setup."""
    print("\n" + "="*60)
    print("TEST: Multi-Agent Setup Verification")
    print("="*60)

    client = HolochainClient(ADMIN_PORT, is_admin=True)
    try:
        await client.connect()
        apps = await client.list_apps()

        # Check all 3 agents are present and enabled
        agent_apps = {app['installed_app_id']: app for app in apps}

        all_ok = True
        for agent_name, agent_info in AGENTS.items():
            app_id = agent_info['app_id']
            if app_id in agent_apps:
                status = agent_apps[app_id].get('status', {}).get('type', 'unknown')
                if status == 'enabled':
                    print(f"✅ {app_id}: enabled")
                else:
                    print(f"⚠️ {app_id}: {status}")
                    all_ok = False
            else:
                print(f"❌ {app_id}: not found")
                all_ok = False

        return all_ok
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    finally:
        await client.close()


async def run_all_tests():
    """Run all conductor integration tests."""
    print("\n" + "="*60)
    print("BYZANTINE-ROBUST FL: LIVE CONDUCTOR TESTS")
    print("="*60)
    print(f"Admin Port: {ADMIN_PORT}")
    print(f"App Port: {APP_PORT}")
    print(f"Agents: {len(AGENTS)}")

    results = {}

    # Run tests
    results['admin_connection'] = await test_admin_connection()
    results['cell_info'] = await test_cell_info()
    results['network_stats'] = await test_network_stats()
    results['multi_agent'] = await test_multi_agent_setup()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
