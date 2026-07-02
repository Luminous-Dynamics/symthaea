#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Install ZeroTrustML DNA on Holochain Conductor

This script installs the ZeroTrustML hApp on a running Holochain conductor
and returns the installed_app_id that can be used for zome calls.
"""

import asyncio
import websockets
import json
import sys
from pathlib import Path

# Conductor admin WebSocket URL
CONDUCTOR_URL = "ws://localhost:8888"

# Path to hApp bundle
HAPP_PATH = Path(__file__).parent / "zerotrustml-dna" / "zerotrustml.happ"


async def install_zerotrustml_happ():
    """Install ZeroTrustML hApp on conductor"""
    print("=" * 70)
    print("  ZeroTrustML DNA Installation")
    print("=" * 70)

    # Verify hApp file exists
    if not HAPP_PATH.exists():
        print(f"\n❌ ERROR: hApp file not found at {HAPP_PATH}")
        return None

    print(f"\n📦 hApp bundle: {HAPP_PATH}")
    print(f"   Size: {HAPP_PATH.stat().st_size / 1024:.1f} KB")

    try:
        # Connect to conductor admin interface with proper subprotocol
        print(f"\n🔌 Connecting to conductor at {CONDUCTOR_URL}...")
        # Holochain admin interface needs no subprotocol, just plain WebSocket
        # but we need to use the right message format (msgpack, not JSON)
        async with websockets.connect(
            CONDUCTOR_URL,
            additional_headers={
                "Sec-WebSocket-Protocol": ""
            }
        ) as websocket:
            print("   ✅ Connected to conductor")

            # Step 1: Generate agent key
            print("\n1️⃣  Generating agent key...")
            generate_key_request = {
                "id": 1,
                "type": "generate_agent_pub_key",
                "data": None
            }

            await websocket.send(json.dumps(generate_key_request))
            response = await websocket.recv()
            response_data = json.loads(response)

            if "data" not in response_data or response_data["data"] is None:
                print(f"   ❌ Failed to generate agent key: {response_data}")
                return None

            agent_key = response_data["data"]
            print(f"   ✅ Agent key: {agent_key[:16]}...{agent_key[-16:]}")

            # Step 2: Read hApp bundle
            print("\n2️⃣  Reading hApp bundle...")
            with open(HAPP_PATH, "rb") as f:
                happ_bytes = f.read()

            # Convert to base64-like representation (JSON-safe)
            import base64
            happ_b64 = base64.b64encode(happ_bytes).decode('utf-8')
            print(f"   ✅ hApp bundle loaded ({len(happ_bytes)} bytes)")

            # Step 3: Install hApp
            print("\n3️⃣  Installing hApp on conductor...")
            installed_app_id = "zerotrustml-app"

            install_request = {
                "id": 2,
                "type": "install_app",
                "data": {
                    "installed_app_id": installed_app_id,
                    "agent_key": agent_key,
                    "bundle": {
                        "bundled": happ_b64
                    },
                    "membrane_proofs": {},
                    "network_seed": None
                }
            }

            await websocket.send(json.dumps(install_request))
            response = await websocket.recv()
            response_data = json.loads(response)

            if "data" not in response_data:
                print(f"   ❌ Failed to install hApp: {response_data}")
                return None

            print(f"   ✅ hApp installed: {installed_app_id}")

            # Step 4: Enable app
            print("\n4️⃣  Enabling hApp...")
            enable_request = {
                "id": 3,
                "type": "enable_app",
                "data": {
                    "installed_app_id": installed_app_id
                }
            }

            await websocket.send(json.dumps(enable_request))
            response = await websocket.recv()
            response_data = json.loads(response)

            if "data" not in response_data:
                print(f"   ❌ Failed to enable app: {response_data}")
                return None

            print(f"   ✅ hApp enabled and running")

            # Step 5: Get app info
            print("\n5️⃣  Retrieving app info...")
            info_request = {
                "id": 4,
                "type": "get_app_info",
                "data": {
                    "installed_app_id": installed_app_id
                }
            }

            await websocket.send(json.dumps(info_request))
            response = await websocket.recv()
            response_data = json.loads(response)

            if "data" in response_data and response_data["data"]:
                app_info = response_data["data"]
                print(f"   ✅ App info retrieved")
                print(f"\n   App Details:")
                print(f"     ID: {installed_app_id}")
                print(f"     Status: {app_info.get('status', 'unknown')}")

                # Get cell info
                if "cell_info" in app_info:
                    cell_info = app_info["cell_info"]
                    print(f"     Cells: {len(cell_info)}")

                    for role_name, cells in cell_info.items():
                        print(f"\n     Role: {role_name}")
                        for cell in cells:
                            if "provisioned" in cell:
                                cell_data = cell["provisioned"]
                                cell_id = cell_data.get("cell_id")
                                if cell_id:
                                    dna_hash, agent = cell_id
                                    print(f"       DNA Hash: {dna_hash[:16]}...{dna_hash[-16:]}")
                                    print(f"       Agent: {agent[:16]}...{agent[-16:]}")

            print("\n" + "=" * 70)
            print("  ✅ INSTALLATION COMPLETE!")
            print("=" * 70)
            print(f"\nTo use in examples:")
            print(f'  installed_app_id = "{installed_app_id}"')
            print(f'  agent_key = "{agent_key}"')

            return {
                "installed_app_id": installed_app_id,
                "agent_key": agent_key,
                "app_info": app_info if "data" in response_data else None
            }

    except ConnectionRefusedError:
        print(f"\n❌ ERROR: Could not connect to conductor at {CONDUCTOR_URL}")
        print("   Make sure the conductor is running:")
        print("   ps aux | grep holochain")
        return None
    except websockets.exceptions.InvalidStatus as e:
        print(f"\n❌ ERROR: WebSocket connection rejected: {e}")
        print("   The conductor may not support the WebSocket protocol used")
        print("   Try using holochain-client library instead")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run installation"""
    result = asyncio.run(install_zerotrustml_happ())

    if result:
        # Write installation info to file for later use
        info_file = Path(__file__).parent / "zerotrustml_installation_info.json"
        with open(info_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n📄 Installation info saved to: {info_file}")
        return 0
    else:
        print("\n❌ Installation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
