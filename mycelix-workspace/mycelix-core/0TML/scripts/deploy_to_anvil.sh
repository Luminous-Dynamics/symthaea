#!/usr/bin/env bash
# Deploy ZeroTrustML contract to Anvil local fork
#
# This script deploys the ZeroTrustMLGradientStorage contract to a running
# Anvil instance for local testing and demo recording.
#
# Prerequisites:
#   - Anvil must be running (./scripts/start_anvil_fork.sh)
#   - Python environment with web3 installed
#
# Usage:
#   ./scripts/deploy_to_anvil.sh

set -e

echo "🚀 Deploy ZeroTrustML Contract to Anvil"
echo "===================================="
echo ""

# Check if Anvil is running
if ! curl -s -X POST -H "Content-Type: application/json" \
    --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
    http://localhost:8545 > /dev/null 2>&1; then
    echo "❌ Anvil not running on http://localhost:8545"
    echo ""
    echo "Start Anvil first:"
    echo "  ./scripts/start_anvil_fork.sh"
    echo ""
    exit 1
fi

echo "✅ Anvil running on http://localhost:8545"
echo ""

# Anvil's first pre-funded account (same every time)
ANVIL_ACCOUNT_0="0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
ANVIL_PRIVATE_KEY_0="0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

echo "📋 Deployment Configuration:"
echo "  RPC: http://localhost:8545"
echo "  Chain ID: 31337 (Anvil default, not 80002 for fork)"
echo "  Deployer: $ANVIL_ACCOUNT_0"
echo "  Gas: Free (Anvil)"
echo ""

# Save private key temporarily for deployment
mkdir -p build
echo "$ANVIL_PRIVATE_KEY_0" > build/.anvil_key
trap "rm -f build/.anvil_key" EXIT

echo "🔨 Deploying contract..."
echo ""

# Run Python deployment script with Anvil configuration
ETHEREUM_PRIVATE_KEY="$ANVIL_PRIVATE_KEY_0" python3 - <<'PYTHON'
import sys
import json
from pathlib import Path
from web3 import Web3
from eth_account import Account

# Configuration
RPC_URL = "http://localhost:8545"
CHAIN_ID = 31337  # Anvil default

print("📦 Loading contract artifacts...")

# Load ABI
abi_file = Path("build/ZeroTrustMLGradientStorage.abi.json")
if not abi_file.exists():
    print("❌ ABI not found. Run: python generate_abi.py")
    sys.exit(1)

with open(abi_file) as f:
    abi = json.load(f)

# Load bytecode
bin_file = Path("build/ZeroTrustMLGradientStorage.bin")
if not bin_file.exists():
    print("❌ Bytecode not found. Run: python compile_contract.py")
    sys.exit(1)

with open(bin_file) as f:
    bytecode = f.read().strip()

print(f"✅ ABI: {len([x for x in abi if x['type'] == 'function'])} functions")
print(f"✅ Bytecode: {len(bytecode)} bytes")

# Connect to Anvil
print("\n🔌 Connecting to Anvil...")
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    print("❌ Failed to connect to Anvil")
    sys.exit(1)

print(f"✅ Connected (block: {w3.eth.block_number})")

# Load account
import os
private_key = os.environ.get('ETHEREUM_PRIVATE_KEY')
account = Account.from_key(private_key)

print(f"\n💰 Deployer: {account.address}")
balance = w3.eth.get_balance(account.address)
print(f"   Balance: {w3.from_wei(balance, 'ether')} ETH")

# Create contract
Contract = w3.eth.contract(abi=abi, bytecode=bytecode)

# Build transaction
print("\n📝 Building deployment transaction...")
nonce = w3.eth.get_transaction_count(account.address)
gas_estimate = Contract.constructor().estimate_gas({'from': account.address})

tx = Contract.constructor().build_transaction({
    'from': account.address,
    'gas': gas_estimate,
    'gasPrice': w3.eth.gas_price,
    'nonce': nonce,
    'chainId': CHAIN_ID
})

print(f"   Gas estimate: {gas_estimate}")

# Sign and send
print("\n🚀 Deploying contract...")
signed_tx = w3.eth.account.sign_transaction(tx, private_key)
tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

print(f"   Transaction: {tx_hash.hex()}")

# Wait for receipt
print("   Waiting for confirmation...")
receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

if receipt['status'] == 1:
    contract_address = receipt['contractAddress']
    print(f"\n✅ Contract deployed!")
    print(f"   Address: {contract_address}")
    print(f"   Block: {receipt['blockNumber']}")
    print(f"   Gas used: {receipt['gasUsed']}")

    # Save address
    address_file = Path("build/anvil_contract_address.txt")
    with open(address_file, 'w') as f:
        f.write(f"{contract_address}\n")

    # Save deployment info
    deployment_info = {
        'contract_address': contract_address,
        'deployer': account.address,
        'chain_id': CHAIN_ID,
        'chain_name': 'Anvil Local Fork',
        'rpc_url': RPC_URL,
        'block_number': receipt['blockNumber'],
        'transaction_hash': tx_hash.hex(),
        'gas_used': receipt['gasUsed']
    }

    info_file = Path("build/anvil_deployment.json")
    with open(info_file, 'w') as f:
        json.dump(deployment_info, f, indent=2)

    print(f"\n💾 Saved to:")
    print(f"   {address_file}")
    print(f"   {info_file}")
else:
    print("\n❌ Deployment failed!")
    sys.exit(1)
PYTHON

if [ $? -eq 0 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎉 SUCCESS!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "Contract deployed to Anvil local fork!"
    echo ""
    echo "Address saved to: build/anvil_contract_address.txt"
    echo ""
    echo "Next steps:"
    echo "  1. Run FL demo: python demos/demo_ethereum_local_fork.py"
    echo "  2. Record demo video with OBS"
    echo "  3. Deploy to live testnet: python deploy_ethereum.py"
    echo ""
else
    echo ""
    echo "❌ Deployment failed"
    exit 1
fi
