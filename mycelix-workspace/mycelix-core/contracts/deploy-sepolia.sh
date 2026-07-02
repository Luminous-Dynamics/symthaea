#!/usr/bin/env bash
# Mycelix Contracts Sepolia Deployment Script
# Deploys: MycelixRegistry, ReputationAnchor, PaymentRouter to Sepolia testnet
#
# Prerequisites:
#   - Foundry installed (via nix-shell or foundryup)
#   - .env file configured with PRIVATE_KEY
#   - Wallet funded with at least 0.001 SepETH (recommended: 0.005+)
#
# Usage:
#   ./deploy-sepolia.sh [--verify]   # Deploy (optionally verify on Etherscan)
#   ./deploy-sepolia.sh --dry-run    # Simulate without broadcasting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Mycelix Contracts Sepolia Deployment${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo -e "${RED}Error: .env file not found${NC}"
    echo "Please copy .env.example to .env and configure it"
    exit 1
fi

# Validate required environment variables
if [ -z "$PRIVATE_KEY" ]; then
    echo -e "${RED}Error: PRIVATE_KEY not set in .env${NC}"
    exit 1
fi

# Set RPC URL (use environment or fallback)
RPC_URL="${SEPOLIA_RPC_URL:-https://sepolia.drpc.org}"

# Parse arguments
VERIFY_FLAG=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --verify)
            if [ -z "$ETHERSCAN_API_KEY" ]; then
                echo -e "${YELLOW}Warning: ETHERSCAN_API_KEY not set, skipping verification${NC}"
            else
                VERIFY_FLAG="--verify --etherscan-api-key $ETHERSCAN_API_KEY"
            fi
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
    esac
done

# Check forge is available
if ! command -v forge &> /dev/null; then
    echo -e "${YELLOW}Forge not found, running in nix-shell...${NC}"
    FORGE_CMD="nix-shell --run"
    RUN_IN_NIX=true
else
    FORGE_CMD=""
    RUN_IN_NIX=false
fi

# Get deployer address
DEPLOYER_ADDRESS=$(cast wallet address --private-key "$PRIVATE_KEY" 2>/dev/null || echo "unknown")
echo -e "${BLUE}Deployer:${NC} $DEPLOYER_ADDRESS"
echo -e "${BLUE}Network:${NC} Sepolia (Chain ID: 11155111)"
echo -e "${BLUE}RPC URL:${NC} $RPC_URL"
echo ""

# Check balance
echo -e "${YELLOW}Checking wallet balance...${NC}"
if [ "$RUN_IN_NIX" = true ]; then
    BALANCE=$(nix-shell --run "cast balance $DEPLOYER_ADDRESS --rpc-url $RPC_URL 2>/dev/null" 2>&1 | tail -1 || echo "0")
else
    BALANCE=$(cast balance "$DEPLOYER_ADDRESS" --rpc-url "$RPC_URL" 2>/dev/null || echo "0")
fi

# Convert balance to ETH for display
if [ "$BALANCE" != "0" ] && [ -n "$BALANCE" ]; then
    echo -e "${GREEN}Balance: $BALANCE wei${NC}"
else
    echo -e "${RED}Warning: Could not fetch balance or balance is 0${NC}"
    echo -e "${YELLOW}Make sure wallet is funded with SepETH from a faucet:${NC}"
    echo "  - https://sepolia-faucet.pk910.de/"
    echo "  - https://cloud.google.com/application/web3/faucet/ethereum/sepolia"
fi
echo ""

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - Simulating deployment...${NC}"
    echo ""

    if [ "$RUN_IN_NIX" = true ]; then
        nix-shell --run "source .env && forge script script/Deploy.s.sol:DeploySepolia --rpc-url $RPC_URL -vvv"
    else
        forge script script/Deploy.s.sol:DeploySepolia --rpc-url "$RPC_URL" -vvv
    fi

    echo ""
    echo -e "${GREEN}Dry run complete!${NC}"
    echo "To deploy for real, run without --dry-run flag"
    exit 0
fi

# Confirm deployment
echo -e "${YELLOW}Ready to deploy to Sepolia testnet${NC}"
echo "Contracts to deploy:"
echo "  1. MycelixRegistry"
echo "  2. ReputationAnchor"
echo "  3. PaymentRouter"
echo ""
read -p "Continue with deployment? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Run deployment
echo ""
echo -e "${BLUE}Starting deployment...${NC}"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="deployments/sepolia_$TIMESTAMP.json"

mkdir -p deployments

if [ "$RUN_IN_NIX" = true ]; then
    nix-shell --run "source .env && forge script script/Deploy.s.sol:DeploySepolia --rpc-url $RPC_URL --broadcast $VERIFY_FLAG -vvv" 2>&1 | tee "deployments/sepolia_$TIMESTAMP.log"
else
    forge script script/Deploy.s.sol:DeploySepolia \
        --rpc-url "$RPC_URL" \
        --broadcast \
        $VERIFY_FLAG \
        -vvv 2>&1 | tee "deployments/sepolia_$TIMESTAMP.log"
fi

# Extract addresses from broadcast
BROADCAST_FILE="broadcast/Deploy.s.sol/11155111/run-latest.json"
if [ -f "$BROADCAST_FILE" ]; then
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}  Deployment Complete!${NC}"
    echo -e "${GREEN}======================================${NC}"
    echo ""
    echo -e "${BLUE}Deployment log saved to:${NC} deployments/sepolia_$TIMESTAMP.log"
    echo -e "${BLUE}Broadcast data:${NC} $BROADCAST_FILE"
    echo ""
    echo "View on Etherscan:"
    echo "  https://sepolia.etherscan.io/address/$DEPLOYER_ADDRESS"
else
    echo -e "${RED}Warning: Broadcast file not found${NC}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
