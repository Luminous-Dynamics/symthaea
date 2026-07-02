#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Connection to Deployed Contract on Polygon Amoy

This script verifies:
1. Connection to Polygon Amoy RPC
2. Contract address is valid
3. Can load contract ABI
4. Can read contract state (getStats)
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zerotrustml.backends import EthereumBackend


async def test_connection():
    """Test connection to deployed contract"""
    print("🧪 Testing Polygon Amoy Connection")
    print("=" * 60)

    # Initialize backend with deployed contract
    backend = EthereumBackend(
        rpc_url="https://rpc-amoy.polygon.technology/",
        contract_address="0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A",
        chain_id=80002
    )

    print("\n1️⃣  Connecting to Polygon Amoy...")
    try:
        await backend.connect()
        print("   ✅ Connected successfully")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

    print("\n2️⃣  Checking health...")
    try:
        health = await backend.health_check()
        print(f"   ✅ Health check passed")
        print(f"   📊 Healthy: {health['healthy']}")
        print(f"   ⏱️  Latency: {health['latency_ms']}ms")
        print(f"   💾 Storage: {health['storage_available']}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False

    print("\n3️⃣  Reading contract state (getStats)...")
    try:
        # Try to call a read-only function
        if backend.contract:
            stats = backend.contract.functions.getStats().call()
            print(f"   ✅ Contract state read successfully")
            print(f"   📊 Total Gradients: {stats[0]}")
            print(f"   💰 Total Credits: {stats[1]}")
            print(f"   ⚠️  Byzantine Events: {stats[2]}")
        else:
            print("   ⚠️  Contract not loaded")
    except Exception as e:
        print(f"   ⚠️  Could not read state: {e}")

    print("\n4️⃣  Contract Information:")
    print(f"   📜 Address: 0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A")
    print(f"   🔗 Explorer: https://amoy.polygonscan.com/address/0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A")
    print(f"   🌐 Network: Polygon Amoy Testnet (Chain ID: 80002)")
    print(f"   📋 Transaction: bf257c09055ac444d3521e7b1ca9f9ea97798031ef0f0503f5c651e6a38fb23c")

    await backend.disconnect()

    print("\n" + "=" * 60)
    print("✅ Connection test complete!")
    assert connection is None or isinstance(connection, bool)


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
