#!/usr/bin/env bash
# Start Anvil local fork of Polygon Amoy for ZeroTrustML demos
#
# This script starts a local Ethereum node that forks Polygon Amoy testnet,
# allowing for free, instant transactions perfect for demo recording.
#
# Usage:
#   ./scripts/start_anvil_fork.sh
#
# Requirements:
#   - Foundry installed (https://book.getfoundry.sh/getting-started/installation)
#   - anvil command available

set -e

echo "🔧 ZeroTrustML Anvil Local Fork Setup"
echo "=================================="
echo ""

# Check if anvil is installed
if ! command -v anvil &> /dev/null; then
    echo "❌ Anvil not found!"
    echo ""
    echo "Install Foundry (includes Anvil):"
    echo "  curl -L https://foundry.paradigm.xyz | bash"
    echo "  foundryup"
    echo ""
    exit 1
fi

echo "✅ Anvil found: $(which anvil)"
echo ""

# Configuration
FORK_URL="https://rpc-amoy.polygon.technology/"
CHAIN_ID=80002
PORT=8545
BLOCK_TIME=1  # 1 second blocks for realistic feel

# Alternative RPC endpoints (if primary fails)
ALT_RPC_1="https://polygon-amoy.drpc.org"
ALT_RPC_2="https://polygon-amoy-bor-rpc.publicnode.com"

echo "📋 Configuration:"
echo "  Fork URL: $FORK_URL"
echo "  Chain ID: $CHAIN_ID"
echo "  Port: $PORT"
echo "  Block Time: ${BLOCK_TIME}s"
echo ""

echo "🔍 Testing RPC connection..."
if curl -s --max-time 5 "$FORK_URL" > /dev/null 2>&1; then
    echo "✅ RPC accessible"
else
    echo "⚠️  Primary RPC slow/unavailable, trying alternative..."
    FORK_URL="$ALT_RPC_1"
    if curl -s --max-time 5 "$FORK_URL" > /dev/null 2>&1; then
        echo "✅ Using alternative RPC: $FORK_URL"
    else
        echo "❌ All RPC endpoints unavailable"
        echo "   Check your internet connection or try again later"
        exit 1
    fi
fi

echo ""
echo "🚀 Starting Anvil fork..."
echo "   Press Ctrl+C to stop"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start Anvil with:
# - Fork of Polygon Amoy
# - Block time of 1 second (realistic for demos)
# - 10 pre-funded accounts (each with 10000 ETH/POL)
# - No rate limiting
# - Mining mode: auto (mines on each transaction)
anvil \
    --fork-url "$FORK_URL" \
    --chain-id "$CHAIN_ID" \
    --port "$PORT" \
    --block-time "$BLOCK_TIME" \
    --accounts 10 \
    --balance 10000 \
    --no-rate-limit \
    --order fifo

# Note: Anvil will print available accounts and private keys
# Example output:
#   Available Accounts
#   ==================
#   (0) 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266 (10000 ETH)
#   (1) 0x70997970C51812dc3A010C7d01b50e0d17dc79C8 (10000 ETH)
#   ...
#
#   Private Keys
#   ==================
#   (0) 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
#   (1) 0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d
#   ...
