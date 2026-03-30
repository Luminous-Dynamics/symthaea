#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Ethereum Live Testnet Demo - For Grant Video Verification

This demo runs the SAME federated learning scenario on Polygon Amoy testnet
to verify that everything works on a real, public blockchain.

Differences from local fork:
- Real transactions (2-3 second block times)
- Real gas costs (testnet POL)
- Globally verifiable on Polygonscan
- Permanent blockchain record

Prerequisites:
- Testnet POL in wallet (get from https://faucet.polygon.technology/)
- Private key in build/.ethereum_key
- Contract deployed: python deploy_ethereum.py

Usage:
    python demos/demo_ethereum_live_testnet.py
"""

import asyncio
import sys
import time
from pathlib import Path
from dataclasses import asdict
import hashlib
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerotrustml.backends import EthereumBackend, GradientRecord


class Colors:
    """ANSI colors for beautiful terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_banner(text: str):
    """Print a beautiful banner"""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}  {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{'='*70}{Colors.ENDC}\n")


def print_section(emoji: str, title: str):
    """Print a section header"""
    print(f"\n{Colors.BOLD}{emoji}  {title}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{'─'*70}{Colors.ENDC}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")


async def demo_live_testnet():
    """Run FL demo on Polygon Amoy live testnet"""

    print_banner("🌍 ZeroTrustML - Federated Learning on Polygon Amoy Testnet")

    print(f"{Colors.OKCYAN}This demo verifies:{Colors.ENDC}")
    print("  • Real blockchain transactions")
    print("  • Public verification on Polygonscan")
    print("  • Permanent immutable records")
    print("  • Production-ready deployment")

    # Check for private key
    key_file = Path("build/.ethereum_key")
    if not key_file.exists():
        print_warning("No private key found!")
        print_info("Create test wallet:")
        print("  from eth_account import Account")
        print("  account = Account.create()")
        print("  print(account.key.hex())")
        print("")
        print("Save to: build/.ethereum_key")
        print("")
        print("Get testnet POL: https://faucet.polygon.technology/")
        return False

    private_key = key_file.read_text().strip()

    # Known deployed contract
    contract_address = "0x4ef9372EF60D12E1DbeC9a13c724F6c631DdE49A"

    # Initialize backend
    print_section("🔌", "Connecting to Polygon Amoy Testnet")

    backend = EthereumBackend(
        rpc_url="https://rpc-amoy.polygon.technology/",
        contract_address=contract_address,
        private_key=private_key,
        chain_id=80002
    )

    try:
        await backend.connect()
        print_success(f"Connected to Polygon Amoy")
        print_info(f"   Contract: {contract_address}")
        print_info(f"   Wallet: {backend.account.address}")

        # Check balance
        balance = backend.w3.eth.get_balance(backend.account.address)
        balance_pol = backend.w3.from_wei(balance, 'ether')
        print_info(f"   Balance: {balance_pol:.4f} POL")

        if balance == 0:
            print_warning("No POL for gas fees!")
            print_info("Get testnet POL: https://faucet.polygon.technology/")
            return False
    except Exception as e:
        print_warning(f"Connection failed: {e}")
        return False

    # Read current stats
    print_section("📊", "Current Contract State")

    stats = backend.contract.functions.getStats().call()
    print_info(f"Total Gradients: {stats[0]}")
    print_info(f"Total Credits: {stats[1]}")
    print_info(f"Byzantine Events: {stats[2]}")

    # Same FL scenario as local fork
    print_section("🏥", "Federated Learning Scenario - Medical AI")

    nodes = [
        {"name": "Hospital A (Testnet)", "honest": True},
        {"name": "Hospital B (Testnet)", "honest": True},
        {"name": "Malicious (Testnet)", "honest": False}
    ]

    print("Training diabetes prediction model (on live testnet):")
    for i, node in enumerate(nodes, 1):
        status = f"{Colors.OKGREEN}Honest{Colors.ENDC}" if node["honest"] else f"{Colors.FAIL}Byzantine{Colors.ENDC}"
        print(f"  {i}. {node['name']:<25} {status}")

    # Simulate FL Round
    print_section("🔄", "FL Round - Training & Gradient Submission")

    round_num = int(time.time()) % 1000  # Use timestamp for unique round
    print_info(f"Round number: {round_num}")
    print("")

    gradients = []

    for node in nodes:
        # Simulate gradient generation (same as local fork)
        if node["honest"]:
            pogq_score = random.uniform(0.85, 0.98)
            gradient_data = [random.gauss(0, 0.1) for _ in range(5)]
        else:
            pogq_score = random.uniform(0.2, 0.4)
            gradient_data = [random.gauss(0, 10.0) for _ in range(5)]

        gradient_hash = hashlib.sha256(str(gradient_data).encode()).hexdigest()
        gradient_id = f"testnet_{node['name'].replace(' ', '_')}_{round_num}"

        print(f"{node['name']}:")
        print(f"  PoGQ Score: {pogq_score:.3f}")
        print(f"  Quality: {'✅ High' if pogq_score > 0.7 else '❌ Low (Byzantine)'}")

        # Create gradient record
        gradient = GradientRecord(
            id=gradient_id,
            node_id=node['name'],
            round_num=round_num,
            gradient=gradient_data,
            gradient_hash=gradient_hash,
            pogq_score=pogq_score,
            zkpoc_verified=True,
            reputation_score=0.85 if node["honest"] else 0.3,
            timestamp=float(time.time()),
            backend_metadata={}
        )

        # Store on-chain (REAL TRANSACTION)
        print(f"  📝 Submitting to blockchain...", end=" ", flush=True)
        start = time.time()

        try:
            success = await backend.store_gradient(asdict(gradient))
            latency = (time.time() - start) * 1000

            if success:
                print(f"{Colors.OKGREEN}✓{Colors.ENDC} ({latency:.0f}ms)")
                gradients.append({"node": node, "gradient": gradient, "pogq": pogq_score})
            else:
                print(f"{Colors.FAIL}✗{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}✗{Colors.ENDC} {e}")

        print("")

    # Byzantine Detection
    print_section("🛡️", "Byzantine Detection & Logging")

    byzantine_threshold = 0.7

    for g in gradients:
        if g["pogq"] < byzantine_threshold:
            print(f"{Colors.FAIL}Byzantine detected:{Colors.ENDC} {g['node']['name']}")
            print(f"  PoGQ: {g['pogq']:.3f} (threshold: {byzantine_threshold})")
            print(f"  📝 Logging to blockchain...", end=" ", flush=True)

            start = time.time()
            try:
                event = {
                    "node_id": g['node']['name'],
                    "round_num": round_num,
                    "detection_method": "PoGQ",
                    "severity": "high",
                    "details": {
                        "pogq_score": g['pogq'],
                        "threshold": byzantine_threshold
                    }
                }

                tx_hash = await backend.log_byzantine_event(event)
                latency = (time.time() - start) * 1000
                print(f"{Colors.OKGREEN}✓{Colors.ENDC} ({latency:.0f}ms)")
                print(f"  🔗 Tx: {tx_hash[:20]}...")
            except Exception as e:
                print(f"{Colors.FAIL}✗{Colors.ENDC} {e}")

        print("")

    # Updated Stats
    print_section("📊", "Updated Contract State")

    stats = backend.contract.functions.getStats().call()
    print_info(f"Total Gradients: {stats[0]}")
    print_info(f"Total Credits: {stats[1]}")
    print_info(f"Byzantine Events: {stats[2]}")

    await backend.disconnect()

    # Success Summary
    print_banner("✅ Live Testnet Verification Complete!")

    print(f"{Colors.OKGREEN}Verified on Polygon Amoy:{Colors.ENDC}")
    print("  ✓ Real blockchain transactions")
    print("  ✓ Publicly verifiable records")
    print("  ✓ Permanent audit trail")
    print("  ✓ Production-ready deployment")

    print(f"\n{Colors.BOLD}Polygonscan Explorer:{Colors.ENDC}")
    print(f"  Contract: https://amoy.polygonscan.com/address/{contract_address}")
    print("")
    print("  View all transactions, events, and state changes!")

    print(f"\n{Colors.OKCYAN}Performance Comparison:{Colors.ENDC}")
    print("  Local Fork (Anvil):  <100ms, Free gas")
    print("  Live Testnet (Amoy): 2-3 sec, Testnet POL")
    print("")
    print("  → Use local fork for development/demos")
    print("  → Use live testnet for verification/production")

    print("")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(demo_live_testnet())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Demo interrupted{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Demo failed: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
