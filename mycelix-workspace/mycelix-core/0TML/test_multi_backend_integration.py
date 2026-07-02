#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Multi-Backend Integration Tests for ZeroTrustML Phase 10

Tests all 5 storage backends (PostgreSQL, LocalFile, Ethereum, Holochain, Cosmos)
for consistency, functionality, and interoperability.

Usage:
    python test_multi_backend_integration.py
    python test_multi_backend_integration.py --backend ethereum
    python test_multi_backend_integration.py --skip-blockchain
"""

import asyncio
import sys
import time
import hashlib
from pathlib import Path
from dataclasses import asdict
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zerotrustml.backends import (
    StorageBackend,
    PostgreSQLBackend,
    LocalFileBackend,
    EthereumBackend,
    HolochainBackend,
    CosmosBackend,
    GradientRecord,
    CreditRecord,
    ByzantineEvent,
)


class BackendTestSuite:
    """Test suite for a single backend"""

    def __init__(self, backend: StorageBackend, name: str):
        self.backend = backend
        self.name = name
        self.results = {
            "connect": False,
            "store_gradient": False,
            "retrieve_gradient": False,
            "issue_credit": False,
            "get_balance": False,
            "log_byzantine": False,
            "disconnect": False,
        }
        self.errors = []

    async def run_tests(self) -> bool:
        """Run all tests for this backend"""
        print(f"\n🧪 Testing {self.name} Backend")
        print("=" * 70)

        try:
            # Test 1: Connect
            print(f"\n1️⃣  Testing connect...")
            success = await self.backend.connect()
            self.results["connect"] = success
            if not success:
                self.errors.append("Failed to connect")
                print(f"   ❌ Connection failed")
                return False
            print(f"   ✅ Connected successfully")

            # Test 2: Store Gradient
            print(f"\n2️⃣  Testing store_gradient...")
            test_gradient = self._create_test_gradient()
            gradient_dict = asdict(test_gradient)
            success = await self.backend.store_gradient(gradient_dict)
            self.results["store_gradient"] = success
            if not success:
                self.errors.append("Failed to store gradient")
                print(f"   ❌ Store failed")
            else:
                print(f"   ✅ Gradient stored: {test_gradient.id}")

            # Test 3: Retrieve Gradient
            print(f"\n3️⃣  Testing get_gradient...")
            retrieved = await self.backend.get_gradient(test_gradient.id)
            self.results["retrieve_gradient"] = retrieved is not None
            if retrieved:
                print(f"   ✅ Gradient retrieved")
                # Handle both dict and dataclass
                if hasattr(retrieved, 'id'):
                    print(f"      • ID: {retrieved.id}")
                    print(f"      • Round: {retrieved.round_num}")
                    print(f"      • POGQ: {retrieved.pogq_score}")
                else:
                    print(f"      • ID: {retrieved.get('id')}")
                    print(f"      • Round: {retrieved.get('round_num')}")
                    print(f"      • POGQ: {retrieved.get('pogq_score')}")
            else:
                self.errors.append("Failed to retrieve gradient")
                print(f"   ❌ Retrieval failed")

            # Test 4: Issue Credit
            print(f"\n4️⃣  Testing issue_credit...")
            holder = "test_holder_123"
            amount = 100
            earned_from = "test_gradient_contribution"
            success = await self.backend.issue_credit(holder, amount, earned_from)
            self.results["issue_credit"] = success
            if not success:
                self.errors.append("Failed to issue credit")
                print(f"   ❌ Credit issuance failed")
            else:
                print(f"   ✅ Credit issued: {amount}")

            # Test 5: Get Credit Balance
            print(f"\n5️⃣  Testing get_credit_balance...")
            balance = await self.backend.get_credit_balance(holder)
            self.results["get_balance"] = balance >= amount
            print(f"   ✅ Balance: {balance}")

            # Test 6: Log Byzantine Event
            print(f"\n6️⃣  Testing log_byzantine_event...")
            byzantine_event = {
                "node_id": "malicious_node",
                "round_num": 1,
                "detection_method": "gradient_verification",
                "severity": "high"
            }
            success = await self.backend.log_byzantine_event(byzantine_event)
            self.results["log_byzantine"] = success
            if not success:
                self.errors.append("Failed to log Byzantine event")
                print(f"   ❌ Byzantine logging failed")
            else:
                print(f"   ✅ Byzantine event logged")

            # Test 7: Disconnect
            print(f"\n7️⃣  Testing disconnect...")
            success = await self.backend.disconnect()
            self.results["disconnect"] = success
            print(f"   ✅ Disconnected successfully")

            return all(self.results.values())

        except Exception as e:
            self.errors.append(f"Exception: {str(e)}")
            print(f"   ❌ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_test_gradient(self) -> GradientRecord:
        """Create a test gradient record"""
        gradient_id = f"integration_test_{self.name.lower()}_{int(time.time())}"

        return GradientRecord(
            id=gradient_id,
            node_id="test_node_integration",
            round_num=1,
            gradient=[0.1, 0.2, 0.3, 0.4, 0.5],
            gradient_hash=hashlib.sha256(b"test_gradient_data").hexdigest(),
            pogq_score=0.95,
            zkpoc_verified=True,
            reputation_score=0.85,
            timestamp=time.time(),
            backend_metadata={}
        )

    def print_summary(self):
        """Print test summary"""
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)

        print(f"\n📊 {self.name} Backend Summary:")
        print(f"   Tests passed: {passed}/{total}")

        if self.errors:
            print(f"   Errors:")
            for error in self.errors:
                print(f"      • {error}")


async def test_localfile_backend():
    """Test LocalFile backend"""
    backend = LocalFileBackend(data_dir="data/integration_test")
    suite = BackendTestSuite(backend, "LocalFile")
    success = await suite.run_tests()
    suite.print_summary()
    return success


async def test_ethereum_backend():
    """Test Ethereum backend (Polygon Amoy)"""
    # Read private key
    key_file = Path("build/.ethereum_key")
    if not key_file.exists():
        print("⚠️  Skipping Ethereum: No private key found")
        return None

    private_key = key_file.read_text().strip()

    backend = EthereumBackend(
        rpc_url="https://rpc-amoy.polygon.technology/",
        contract_address="0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A",
        private_key=private_key,
        chain_id=80002
    )

    suite = BackendTestSuite(backend, "Ethereum")
    success = await suite.run_tests()
    suite.print_summary()
    return success


async def test_postgresql_backend():
    """Test PostgreSQL backend"""
    # Check if PostgreSQL is available
    backend = PostgreSQLBackend(
        host="localhost",
        port=5432,
        database="zerotrustml",
        user="postgres",
        password="zerotrustml_secret"
    )

    suite = BackendTestSuite(backend, "PostgreSQL")

    try:
        success = await suite.run_tests()
        suite.print_summary()
        return success
    except Exception as e:
        print(f"⚠️  Skipping PostgreSQL: {e}")
        return None


async def test_holochain_backend():
    """Test Holochain backend"""
    print("\n⚠️  Holochain backend requires conductor running")
    print("   Skipping for now - will test when conductor is deployed")
    return None


async def test_cosmos_backend():
    """Test Cosmos backend"""
    print("\n⚠️  Cosmos backend requires deployed contract")
    print("   Skipping for now - contract needs WASM build and deployment")
    return None


async def run_all_tests(skip_blockchain: bool = False):
    """Run tests for all backends"""
    print("\n" + "=" * 70)
    print("🧪 ZeroTrustML Multi-Backend Integration Tests")
    print("=" * 70)

    results = {}

    # Always test LocalFile (no dependencies)
    print("\n" + "=" * 70)
    results["LocalFile"] = await test_localfile_backend()

    # Test PostgreSQL if available
    print("\n" + "=" * 70)
    results["PostgreSQL"] = await test_postgresql_backend()

    # Test Ethereum if not skipping blockchain
    if not skip_blockchain:
        print("\n" + "=" * 70)
        results["Ethereum"] = await test_ethereum_backend()

    # Holochain and Cosmos require deployment
    print("\n" + "=" * 70)
    results["Holochain"] = await test_holochain_backend()

    print("\n" + "=" * 70)
    results["Cosmos"] = await test_cosmos_backend()

    # Print final summary
    print("\n" + "=" * 70)
    print("📊 Final Integration Test Summary")
    print("=" * 70)

    for backend, result in results.items():
        if result is True:
            status = "✅ PASSING"
        elif result is False:
            status = "❌ FAILING"
        else:
            status = "⏭️  SKIPPED"

        print(f"   {backend:15} {status}")

    # Count results
    tested = sum(1 for r in results.values() if r is not None)
    passed = sum(1 for r in results.values() if r is True)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\n   Tested: {tested}/5")
    print(f"   Passed: {passed}/{tested}")
    print(f"   Skipped: {skipped}/5")

    print("\n" + "=" * 70)

    # Return overall success
    return passed == tested if tested > 0 else False


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="ZeroTrustML Multi-Backend Integration Tests")
    parser.add_argument("--backend", choices=["localfile", "postgresql", "ethereum", "holochain", "cosmos"],
                       help="Test only a specific backend")
    parser.add_argument("--skip-blockchain", action="store_true",
                       help="Skip blockchain backends (Ethereum, Cosmos)")

    args = parser.parse_args()

    if args.backend:
        # Test single backend
        backend_tests = {
            "localfile": test_localfile_backend,
            "postgresql": test_postgresql_backend,
            "ethereum": test_ethereum_backend,
            "holochain": test_holochain_backend,
            "cosmos": test_cosmos_backend,
        }

        print(f"\n🎯 Testing {args.backend} backend only")
        success = await backend_tests[args.backend]()
        sys.exit(0 if success else 1)

    else:
        # Test all backends
        success = await run_all_tests(skip_blockchain=args.skip_blockchain)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
