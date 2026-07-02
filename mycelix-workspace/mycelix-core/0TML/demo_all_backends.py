#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Multi-Backend Demo
Demonstrates all 5 storage backends working together with different strategies

Backends Demonstrated:
1. PostgreSQL - Enterprise database
2. LocalFile - Development/testing
3. Holochain - Decentralized DHT
4. Ethereum - Blockchain (Polygon L2)
5. Cosmos - Blockchain (Cosmos SDK)

Strategies Demonstrated:
- "primary" - Write to first backend only
- "all" - Write to all backends in parallel
- "quorum" - Write to majority of backends
"""

import asyncio
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, '/srv/luminous-dynamics/Mycelix-Core/0TML/src')

from zerotrustml.backends import (
    StorageBackend,
    BackendType,
    PostgreSQLBackend,
    LocalFileBackend,
    HolochainBackend,
    EthereumBackend,
    CosmosBackend
)


# ============================================
# TEST DATA
# ============================================

def create_test_gradient(gradient_id: str, node_id: str, round_num: int) -> Dict[str, Any]:
    """Create test gradient data"""
    import hashlib
    gradient_data = [0.1, 0.2, 0.3, 0.4, 0.5]
    gradient_hash = hashlib.sha256(str(gradient_data).encode()).hexdigest()

    return {
        "id": gradient_id,
        "node_id": node_id,
        "round_num": round_num,
        "gradient": gradient_data,
        "gradient_hash": gradient_hash,
        "pogq_score": 0.95,
        "zkpoc_verified": True,
        "validation_passed": True,
        "reputation_score": 0.85,
        "timestamp": time.time()
    }


# ============================================
# BACKEND INITIALIZATION
# ============================================

async def setup_backends() -> List[StorageBackend]:
    """
    Initialize all available backends

    Returns:
        List of connected backends
    """
    backends = []
    print("=" * 60)
    print("🚀 INITIALIZING BACKENDS")
    print("=" * 60)
    print()

    # 1. LocalFile Backend (always available)
    print("1️⃣  LocalFile Backend")
    print("-" * 60)
    try:
        localfile = LocalFileBackend(data_dir="/tmp/zerotrustml_demo")
        await localfile.connect()
        health = await localfile.health_check()
        print(f"   ✅ Connected")
        print(f"   📊 Health: {health['healthy']}")
        print(f"   ⏱️  Latency: {health['latency_ms']}ms")
        backends.append(localfile)
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    print()

    # 2. PostgreSQL Backend (if available)
    print("2️⃣  PostgreSQL Backend")
    print("-" * 60)
    try:
        postgresql = PostgreSQLBackend(
            host="localhost",
            port=5432,
            database="zerotrustml_demo",
            user="zerotrustml",
            password="test123"
        )
        await postgresql.connect()
        health = await postgresql.health_check()
        print(f"   ✅ Connected")
        print(f"   📊 Health: {health['healthy']}")
        print(f"   ⏱️  Latency: {health['latency_ms']}ms")
        backends.append(postgresql)
    except Exception as e:
        print(f"   ⚠️  Not available: {e}")
        print(f"   💡 Run: docker run -d -p 5432:5432 -e POSTGRES_DB=zerotrustml_demo \\")
        print(f"              -e POSTGRES_USER=zerotrustml -e POSTGRES_PASSWORD=test123 postgres")
    print()

    # 3. Holochain Backend (if conductor running)
    print("3️⃣  Holochain Backend")
    print("-" * 60)
    try:
        holochain = HolochainBackend(
            admin_url="ws://localhost:8888",
            app_url="ws://localhost:8889",
            app_id="zerotrustml_demo"
        )
        await holochain.connect()
        health = await holochain.health_check()
        print(f"   ✅ Connected")
        print(f"   📊 Health: {health['healthy']}")
        print(f"   ⏱️  Latency: {health['latency_ms']}ms")
        backends.append(holochain)
    except Exception as e:
        print(f"   ⚠️  Not available: {e}")
        print(f"   💡 Run: holochain -c conductor.yaml")
    print()

    # 4. Ethereum Backend - Polygon Amoy Testnet (DEPLOYED)
    print("4️⃣  Ethereum Backend (Polygon Amoy Testnet)")
    print("-" * 60)
    try:
        ethereum = EthereumBackend(
            rpc_url="https://rpc-amoy.polygon.technology/",
            contract_address="0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A",  # ✅ DEPLOYED!
            chain_id=80002,  # Polygon Amoy
            # Optional: Add private key from build/.ethereum_key for write operations
            # private_key=open("build/.ethereum_key").read().strip()
        )
        await ethereum.connect()
        health = await ethereum.health_check()
        print(f"   ✅ Connected to Polygon Amoy")
        print(f"   📊 Health: {health['healthy']}")
        print(f"   ⏱️  Latency: {health['latency_ms']}ms")
        print(f"   📜 Contract: 0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A")
        print(f"   🔗 Explorer: https://amoy.polygonscan.com/address/0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A")
        print(f"   ⛽ Gas Price: {health.get('gas_price', 'N/A')}")
        backends.append(ethereum)
    except Exception as e:
        print(f"   ⚠️  Not available: {e}")
        print(f"   💡 Check network connection to Polygon Amoy")
    print()

    # 5. Cosmos Backend (if local chain running)
    print("5️⃣  Cosmos Backend")
    print("-" * 60)
    try:
        cosmos = CosmosBackend(
            chain="local",
            mnemonic="test test test test test test test test test test test junk",  # Test mnemonic
            contract_address=None  # Will deploy in production
        )
        await cosmos.connect()
        health = await cosmos.health_check()
        print(f"   ✅ Connected")
        print(f"   📊 Health: {health['healthy']}")
        print(f"   ⏱️  Latency: {health['latency_ms']}ms")
        print(f"   🔗 Chain: {health.get('chain_id', 'N/A')}")
        backends.append(cosmos)
    except Exception as e:
        print(f"   ⚠️  Not available: {e}")
        print(f"   💡 Run: wasmd start (local Cosmos chain)")
    print()

    print(f"✅ {len(backends)} backend(s) initialized successfully")
    print()

    return backends


# ============================================
# STRATEGY DEMONSTRATIONS
# ============================================

async def demo_primary_strategy(backends: List[StorageBackend]):
    """Demonstrate 'primary' strategy (write to first backend only)"""
    print("=" * 60)
    print("📝 STRATEGY: PRIMARY (Write to first backend only)")
    print("=" * 60)
    print()

    if not backends:
        print("❌ No backends available")
        return

    # Use first backend
    backend = backends[0]
    print(f"Using: {backend.backend_type.value}")
    print()

    # Test gradient storage
    gradient = create_test_gradient("primary-gradient-001", "node-1", 1)

    start = time.time()
    gradient_id = await backend.store_gradient(gradient)
    latency = (time.time() - start) * 1000

    print(f"✅ Gradient stored: {gradient_id[:16]}...")
    print(f"⏱️  Latency: {latency:.2f}ms")
    print()

    # Verify retrieval
    retrieved = await backend.get_gradient(gradient_id)
    if retrieved:
        print(f"✅ Gradient retrieved successfully")
        print(f"   Node: {retrieved.node_id}")
        print(f"   Round: {retrieved.round_num}")
        print(f"   PoGQ: {retrieved.pogq_score}")
    else:
        print(f"❌ Gradient retrieval failed")
    print()


async def demo_all_strategy(backends: List[StorageBackend]):
    """Demonstrate 'all' strategy (write to all backends)"""
    print("=" * 60)
    print("🌐 STRATEGY: ALL (Write to all backends in parallel)")
    print("=" * 60)
    print()

    if len(backends) < 2:
        print("⚠️  Need at least 2 backends for 'all' strategy")
        return

    gradient = create_test_gradient("all-gradient-001", "node-2", 2)

    # Write to all backends in parallel
    print(f"Writing to {len(backends)} backends simultaneously...")
    print()

    start = time.time()
    tasks = [backend.store_gradient(gradient) for backend in backends]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_latency = (time.time() - start) * 1000

    # Report results
    successful = 0
    for i, (backend, result) in enumerate(zip(backends, results)):
        if isinstance(result, Exception):
            print(f"   {i+1}. {backend.backend_type.value}: ❌ {result}")
        else:
            print(f"   {i+1}. {backend.backend_type.value}: ✅ {result[:16]}...")
            successful += 1

    print()
    print(f"✅ {successful}/{len(backends)} backends succeeded")
    print(f"⏱️  Total latency: {total_latency:.2f}ms")
    print()


async def demo_quorum_strategy(backends: List[StorageBackend]):
    """Demonstrate 'quorum' strategy (write to majority)"""
    print("=" * 60)
    print("⚖️  STRATEGY: QUORUM (Write to majority of backends)")
    print("=" * 60)
    print()

    if len(backends) < 3:
        print("⚠️  Need at least 3 backends for 'quorum' strategy")
        return

    gradient = create_test_gradient("quorum-gradient-001", "node-3", 3)

    # Calculate quorum
    quorum_size = (len(backends) // 2) + 1
    quorum_backends = backends[:quorum_size]

    print(f"Quorum: {quorum_size}/{len(backends)} backends")
    print(f"Selected: {', '.join(b.backend_type.value for b in quorum_backends)}")
    print()

    # Write to quorum
    start = time.time()
    tasks = [backend.store_gradient(gradient) for backend in quorum_backends]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_latency = (time.time() - start) * 1000

    # Report results
    successful = sum(1 for r in results if not isinstance(r, Exception))

    print(f"✅ {successful}/{quorum_size} quorum backends succeeded")
    print(f"⏱️  Total latency: {total_latency:.2f}ms")

    if successful >= quorum_size:
        print("✅ Quorum achieved - data is consistent")
    else:
        print("❌ Quorum failed - data may be inconsistent")
    print()


# ============================================
# ADDITIONAL FEATURE DEMOS
# ============================================

async def demo_credits(backends: List[StorageBackend]):
    """Demonstrate credit issuance across backends"""
    print("=" * 60)
    print("💰 CREDIT ISSUANCE DEMO")
    print("=" * 60)
    print()

    if not backends:
        print("❌ No backends available")
        return

    backend = backends[0]
    node_id = "node-demo-1"

    # Issue credits
    print(f"Issuing 100 credits to {node_id}...")
    credit_id = await backend.issue_credit(
        holder=node_id,
        amount=100,
        earned_from="gradient_quality"
    )
    print(f"✅ Credit issued: {credit_id[:16]}...")
    print()

    # Check balance
    balance = await backend.get_credit_balance(node_id)
    print(f"💰 Balance: {balance} credits")
    print()


async def demo_byzantine_events(backends: List[StorageBackend]):
    """Demonstrate Byzantine event logging"""
    print("=" * 60)
    print("🚨 BYZANTINE EVENT LOGGING DEMO")
    print("=" * 60)
    print()

    if not backends:
        print("❌ No backends available")
        return

    backend = backends[0]

    # Log Byzantine event
    event = {
        "node_id": "malicious-node-1",
        "round_num": 5,
        "detection_method": "gradient_comparison",
        "severity": "high",
        "details": {
            "expected_hash": "abc123",
            "received_hash": "def456",
            "divergence": 0.95
        }
    }

    print(f"Logging Byzantine event...")
    print(f"   Node: {event['node_id']}")
    print(f"   Severity: {event['severity']}")
    print()

    event_id = await backend.log_byzantine_event(event)
    print(f"✅ Event logged: {event_id[:16]}...")
    print()

    # Retrieve events
    events = await backend.get_byzantine_events(node_id="malicious-node-1")
    print(f"📋 Retrieved {len(events)} event(s) for node")
    print()


async def demo_statistics(backends: List[StorageBackend]):
    """Show statistics from all backends"""
    print("=" * 60)
    print("📊 BACKEND STATISTICS")
    print("=" * 60)
    print()

    for i, backend in enumerate(backends, 1):
        stats = await backend.get_stats()

        print(f"{i}. {backend.backend_type.value.upper()}")
        print(f"   Total Gradients: {stats.get('total_gradients', 0)}")
        print(f"   Total Credits: {stats.get('total_credits_issued', 0)}")
        print(f"   Byzantine Events: {stats.get('total_byzantine_events', 0)}")
        print(f"   Storage Size: {stats.get('storage_size_bytes', 0)} bytes")
        print()


# ============================================
# MAIN DEMO
# ============================================

async def main():
    """Run complete multi-backend demo"""
    print()
    print("🎯" * 30)
    print("   ZeroTrustML MULTI-BACKEND DEMO")
    print("   Demonstrating 5 Storage Backends")
    print("🎯" * 30)
    print()

    # Setup all backends
    backends = await setup_backends()

    if not backends:
        print("❌ No backends available. Please start at least one backend.")
        print()
        print("Quick start options:")
        print("1. LocalFile: No setup needed (always works)")
        print("2. PostgreSQL: docker run -d -p 5432:5432 -e POSTGRES_DB=zerotrustml_demo \\")
        print("               -e POSTGRES_USER=zerotrustml -e POSTGRES_PASSWORD=test123 postgres")
        print("3. Ethereum: ganache-cli --deterministic")
        return

    # Demonstrate strategies
    await demo_primary_strategy(backends)
    await demo_all_strategy(backends)
    await demo_quorum_strategy(backends)

    # Demonstrate additional features
    await demo_credits(backends)
    await demo_byzantine_events(backends)
    await demo_statistics(backends)

    # Cleanup
    print("=" * 60)
    print("🧹 CLEANUP")
    print("=" * 60)
    print()

    for backend in backends:
        await backend.disconnect()
        print(f"✅ Disconnected: {backend.backend_type.value}")

    print()
    print("🎉" * 30)
    print("   DEMO COMPLETE!")
    print("🎉" * 30)
    print()


if __name__ == "__main__":
    asyncio.run(main())
