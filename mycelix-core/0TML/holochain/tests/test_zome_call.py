#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Test zome calls to the defense_coordinator."""

import asyncio
import msgpack
import websockets
import numpy as np

ADMIN_PORT = 9010
APP_PORT = 9011
APP_ID = "byzantine-fl-agent1"


async def get_cell_id():
    """Get the cell ID from the conductor."""
    ws = await websockets.connect(
        f'ws://127.0.0.1:{ADMIN_PORT}',
        additional_headers={'Origin': 'http://localhost'}
    )

    # List apps
    inner = {"type": "list_apps", "value": {"status_filter": None}}
    outer = {"id": 1, "type": "request", "data": msgpack.packb(inner)}
    await ws.send(msgpack.packb(outer))

    resp = await asyncio.wait_for(ws.recv(), timeout=10)
    decoded = msgpack.unpackb(resp, raw=False)
    data = decoded.get("data")
    if isinstance(data, bytes):
        data = msgpack.unpackb(data, raw=False)

    apps = data.get("value", [])
    await ws.close()

    for app in apps:
        if app.get("installed_app_id") == APP_ID:
            cell_info = app.get("cell_info", {})
            for role_name, cells in cell_info.items():
                if cells:
                    cell = cells[0]
                    if cell.get("type") == "provisioned":
                        cell_value = cell.get("value", {})
                        cell_id = cell_value.get("cell_id")
                        if cell_id:
                            # cell_id can be a list [dna_hash, agent_key] or dict
                            if isinstance(cell_id, list):
                                return tuple(cell_id)
                            else:
                                return (cell_id["dna_hash"], cell_id["agent_pub_key"])
    return None


async def call_zome(cell_id, zome_name: str, fn_name: str, payload: dict):
    """Call a zome function."""
    ws = await websockets.connect(
        f'ws://127.0.0.1:{APP_PORT}',
        additional_headers={'Origin': 'http://localhost'}
    )

    dna_hash, agent_key = cell_id

    # App request format
    # { id, type: "request", data: msgpack({ type: "call_zome", value: { ... } }) }
    zome_call = {
        "type": "call_zome",
        "value": {
            "cell_id": [dna_hash, agent_key],
            "zome_name": zome_name,
            "fn_name": fn_name,
            "payload": msgpack.packb(payload),  # Zome payloads are msgpack
            "provenance": agent_key,
            "cap_secret": None
        }
    }

    outer = {"id": 1, "type": "request", "data": msgpack.packb(zome_call)}
    await ws.send(msgpack.packb(outer))

    resp = await asyncio.wait_for(ws.recv(), timeout=30)
    decoded = msgpack.unpackb(resp, raw=False)
    data = decoded.get("data")
    if isinstance(data, bytes):
        data = msgpack.unpackb(data, raw=False)

    await ws.close()
    return data


def float_to_fixed(x: float, scale: int = 65536) -> int:
    """Convert float to Q16.16 fixed-point."""
    return int(x * scale)


async def test_defense_coordinator():
    """Test the defense_coordinator zome."""
    print("=" * 60)
    print("BYZANTINE DEFENSE HOLOCHAIN INTEGRATION TEST")
    print("=" * 60)

    # Get cell ID
    print("\n1. Getting cell ID...")
    cell_id = await get_cell_id()
    if not cell_id:
        print("❌ Failed to get cell ID")
        return False

    dna_hash, agent_key = cell_id
    print(f"   DNA hash: {dna_hash[:20]}...")
    print(f"   Agent key: {agent_key[:20]}...")

    # Test gradient processing
    print("\n2. Testing gradient processing...")

    # Create test gradients (5 nodes, 20 dimensions)
    np.random.seed(42)
    base_grad = np.random.randn(20) * 0.1

    gradients = []
    for i in range(5):
        grad = base_grad + np.random.randn(20) * 0.01
        grad_fp = [float_to_fixed(x) for x in grad]
        gradients.append({
            "node_id": f"node{i}",
            "values": grad_fp,
            "round": 1
        })

    # Call process_round
    print("   Calling process_round...")
    payload = {
        "round": 1,
        "gradients": gradients,
        "config": None,
        "reputation_scores": [(f"node{i}", int(0.5 * 65536)) for i in range(5)]
    }

    try:
        result = await call_zome(cell_id, "defense_coordinator", "process_round", payload)
        print(f"   Result type: {result.get('type') if isinstance(result, dict) else type(result)}")

        if isinstance(result, dict) and result.get("type") == "error":
            print(f"   ❌ Error: {result.get('value')}")
        else:
            print(f"   ✅ Defense processing complete!")
            if isinstance(result, dict):
                print(f"   Aggregated gradient length: {len(result.get('aggregated_gradient', []))}")
                print(f"   Flagged nodes: {result.get('flagged_nodes', [])}")

    except Exception as e:
        print(f"   ❌ Zome call failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test complete!")
    return True


if __name__ == "__main__":
    asyncio.run(test_defense_coordinator())
