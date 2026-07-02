#!/usr/bin/env bash
# Mycelix Contracts Setup Script
# Run this after cloning to initialize the project

set -e

echo "Setting up Mycelix Contracts..."

# Check if forge is available
if ! command -v forge &> /dev/null; then
    echo "Error: Foundry (forge) not found. Please run inside nix shell:"
    echo "  nix develop"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
forge install foundry-rs/forge-std --no-commit
forge install OpenZeppelin/openzeppelin-contracts@v5.0.2 --no-commit
forge install dmfxyz/murky --no-commit

# Build contracts
echo "Building contracts..."
forge build

# Run tests
echo "Running tests..."
forge test

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and configure"
echo "  2. Deploy: forge script script/Deploy.s.sol:DeployMumbai --rpc-url \$RPC_URL --broadcast"
echo ""
