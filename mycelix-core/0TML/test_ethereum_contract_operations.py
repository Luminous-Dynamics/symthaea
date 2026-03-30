#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Ethereum Contract Operations on Polygon Amoy

This script tests all contract functions:
1. Read-only operations (getStats, getVersion)
2. Write operations (storeGradient, issueCredit)
3. Query operations (getGradient, getCreditBalance)
"""

import asyncio
import sys
from pathlib import Path
import hashlib
import time
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zerotrustml.backends import EthereumBackend, GradientRecord, CreditRecord


async def test_contract_operations():
    """Test all contract operations"""
    print("🧪 Testing ZeroTrustML Contract on Polygon Amoy")
    print("=" * 70)

    # Read wallet private key
    key_file = Path("build/.ethereum_key")
    if not key_file.exists():
        print("❌ Private key not found at build/.ethereum_key")
        return False

    private_key = key_file.read_text().strip()

    # Initialize backend with deployed contract
    backend = EthereumBackend(
        rpc_url="https://rpc-amoy.polygon.technology/",
        contract_address="0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A",
        private_key=private_key,
        chain_id=80002
    )

    print("\n1️⃣  Connecting to Polygon Amoy...")
    try:
        await backend.connect()
        print(f"   ✅ Connected")
        print(f"   📍 Wallet: {backend.account.address}")
        print(f"   📜 Contract: {backend.contract_address}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False

    # Test 1: Read contract version (read-only)
    print("\n2️⃣  Reading contract version...")
    try:
        version = backend.contract.functions.getVersion().call()
        print(f"   ✅ Contract version: {version}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Test 2: Read contract stats (read-only)
    print("\n3️⃣  Reading contract stats...")
    try:
        stats = backend.contract.functions.getStats().call()
        print(f"   ✅ Contract stats:")
        print(f"      • Total Gradients: {stats[0]}")
        print(f"      • Total Credits: {stats[1]}")
        print(f"      • Byzantine Events: {stats[2]}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Test 3: Check wallet balance
    print("\n4️⃣  Checking wallet balance...")
    try:
        balance = backend.w3.eth.get_balance(backend.account.address)
        balance_pol = backend.w3.from_wei(balance, 'ether')
        print(f"   ✅ Balance: {balance_pol:.4f} POL")

        if balance == 0:
            print("   ⚠️  No POL for gas fees - skipping write operations")
            print("   💡 Get testnet POL: https://faucet.polygon.technology/")
            await backend.disconnect()
            assert len(tx_hash) > 0
            return
    except Exception as e:
        print(f"   ⚠️  Could not check balance: {e}")

    # Test 4: Store gradient (write operation)
    print("\n5️⃣  Testing storeGradient (write operation)...")
    try:
        # Create test gradient
        gradient_id = f"test_gradient_{int(time.time())}"
        node_hash = hashlib.sha256(b"test_node").digest()
        round_num = 1
        gradient_hash = hashlib.sha256(b"test_gradient_data").hexdigest()
        pogq_score = 950  # 0-1000 scale
        zkpoc_verified = True

        gradient = GradientRecord(
            id=gradient_id,
            node_id="test_node",
            round_num=round_num,
            gradient=[0.1, 0.2, 0.3, 0.4, 0.5],  # Example gradient data
            gradient_hash=gradient_hash,
            pogq_score=pogq_score / 1000.0,
            zkpoc_verified=zkpoc_verified,
            reputation_score=0.85,
            timestamp=float(time.time()),
            backend_metadata={
                "node_hash": node_hash.hex()
            }
        )

        print(f"   📦 Storing gradient: {gradient_id}")
        print(f"      • Round: {round_num}")
        print(f"      • Quality: {pogq_score/1000.0}")

        # Store gradient (convert to dict)
        gradient_dict = asdict(gradient)
        success = await backend.store_gradient(gradient_dict)

        if success:
            print(f"   ✅ Gradient stored successfully")

            # Verify storage
            print(f"\n6️⃣  Verifying gradient storage...")
            exists = backend.contract.functions.gradientExists(gradient_id).call()
            if exists:
                print(f"   ✅ Gradient verified on-chain")

                # Get gradient data
                data = backend.contract.functions.getGradient(gradient_id).call()
                print(f"   📊 Retrieved data:")
                print(f"      • ID: {data[0]}")
                print(f"      • Round: {data[2]}")
                print(f"      • Hash: {data[3]}")
                print(f"      • POGQ Score: {data[4]}")
                print(f"      • ZKPoC: {data[5]}")
            else:
                print(f"   ⚠️  Gradient not found on-chain")
        else:
            print(f"   ⚠️  Storage may have failed (check transaction)")

    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 5: Issue credit (write operation)
    print("\n7️⃣  Testing issueCredit (write operation)...")
    try:
        holder_hash = hashlib.sha256(b"test_holder").digest()
        credit_amount = 100
        earned_from = "test_gradient_contribution"

        credit = CreditRecord(
            transaction_id=f"credit_tx_{int(time.time())}",
            holder="test_holder",
            amount=credit_amount,
            earned_from=earned_from,
            timestamp=float(time.time()),
            backend_metadata={"holder_hash": holder_hash.hex()}
        )

        print(f"   💰 Issuing credit: {credit_amount}")
        print(f"      • Earned from: {earned_from}")

        # Issue credit with individual parameters
        success = await backend.issue_credit(
            holder="test_holder",
            amount=credit_amount,
            earned_from=earned_from
        )

        if success:
            print(f"   ✅ Credit issued successfully")

            # Check balance
            print(f"\n8️⃣  Checking credit balance...")
            balance = backend.contract.functions.getCreditBalance(holder_hash).call()
            print(f"   ✅ Credit balance: {balance}")
        else:
            print(f"   ⚠️  Credit issuance may have failed")

    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test 6: Read updated stats
    print("\n9️⃣  Reading updated contract stats...")
    try:
        stats = backend.contract.functions.getStats().call()
        print(f"   ✅ Updated stats:")
        print(f"      • Total Gradients: {stats[0]}")
        print(f"      • Total Credits: {stats[1]}")
        print(f"      • Byzantine Events: {stats[2]}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    await backend.disconnect()

    print("\n" + "=" * 70)
    print("✅ Contract operations test complete!")
    print("\n📊 Contract Explorer:")
    print("   https://amoy.polygonscan.com/address/0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A")

    assert isinstance(status, dict)


if __name__ == "__main__":
    success = asyncio.run(test_contract_operations())
    sys.exit(0 if success else 1)
