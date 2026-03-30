#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Install ZeroTrustML hApp to running Holochain conductor via admin API
"""

import asyncio
import websockets
import json
import base64
from pathlib import Path

async def install_happ(conductor_url: str, happ_path: str):
    """Install hApp bundle to conductor via admin API"""

    # Read the hApp bundle
    print(f"📦 Reading hApp bundle: {happ_path}")
    with open(happ_path, 'rb') as f:
        happ_bundle = f.read()

    # Encode as base64
    happ_b64 = base64.b64encode(happ_bundle).decode('utf-8')
    print(f"✓ Bundle loaded: {len(happ_bundle)} bytes")

    # Connect to admin interface
    print(f"\n🔌 Connecting to conductor at {conductor_url}...")
    # Add Origin header to satisfy conductor requirements
    extra_headers = {"Origin": "http://localhost"}
    async with websockets.connect(conductor_url, extra_headers=extra_headers) as websocket:
        print("✓ Connected to conductor")

        # Step 1: Generate agent key
        print("\n👤 Generating agent key...")
        generate_agent_request = {
            "type": "generate_agent_pub_key",
            "data": None
        }

        await websocket.send(json.dumps(generate_agent_request))
        response = await websocket.recv()
        response_data = json.loads(response)

        if response_data.get('type') == 'error':
            print(f"❌ Error generating agent: {response_data}")
            return False

        agent_key = response_data.get('data')
        print(f"✓ Agent key generated: {agent_key[:20]}...")

        # Step 2: Install hApp
        print(f"\n📥 Installing hApp bundle...")
        install_request = {
            "type": "install_app",
            "data": {
                "installed_app_id": "zerotrustml",
                "agent_key": agent_key,
                "bundle": {
                    "bundle": happ_b64,
                    "resources": {}
                },
                "membrane_proofs": {},
                "network_seed": None
            }
        }

        await websocket.send(json.dumps(install_request))
        print("⏳ Waiting for installation response...")
        response = await websocket.recv()
        response_data = json.loads(response)

        print(f"\n📨 Response received:")
        print(json.dumps(response_data, indent=2))

        if response_data.get('type') == 'error':
            print(f"\n❌ Installation failed: {response_data}")
            return False

        print(f"\n✅ hApp installed successfully!")

        # Step 3: Enable the app
        print(f"\n🚀 Enabling app...")
        enable_request = {
            "type": "enable_app",
            "data": {
                "installed_app_id": "zerotrustml"
            }
        }

        await websocket.send(json.dumps(enable_request))
        response = await websocket.recv()
        response_data = json.loads(response)

        print(f"📨 Enable response:")
        print(json.dumps(response_data, indent=2))

        if response_data.get('type') == 'error':
            print(f"\n⚠️  App installed but failed to enable: {response_data}")
            return True  # Still consider it a success

        print(f"\n✅ App enabled successfully!")
        return True

async def main():
    """Main entry point"""
    conductor_url = "ws://localhost:8888"
    happ_path = "zerotrustml-dna/zerotrustml.happ"

    print("=" * 60)
    print("  ZeroTrustML hApp Installation")
    print("=" * 60)

    try:
        success = await install_happ(conductor_url, happ_path)

        if success:
            print("\n" + "=" * 60)
            print("  ✅ Installation Complete!")
            print("=" * 60)
            print("\nNext steps:")
            print("  1. Test zome calls via Rust bridge")
            print("  2. Verify real action hashes are returned")
            print("  3. Check DHT storage with get_balance()")
        else:
            print("\n" + "=" * 60)
            print("  ❌ Installation Failed")
            print("=" * 60)

    except FileNotFoundError:
        print(f"\n❌ hApp bundle not found: {happ_path}")
        print("   Run: cd zerotrustml-dna && hc app pack .")
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
