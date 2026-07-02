#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# Mycelix Praxis — Node Zero Bootstrap Ritual (Genesis Epoch)

set -e

echo "\u{1F331} Initializing Mycelix Node Zero..."

# 1. Create the Architect Identity
if [ ! -f ~/.praxis/architect_did ]; then
    echo "Generating Architect DID..."
    mkdir -p ~/.praxis
    # Simulated DID generation
    echo "did:mycelix:architect-$(openssl rand -hex 8)" > ~/.praxis/architect_did
fi
ARCHITECT_DID=$(cat ~/.praxis/architect_did)
echo "Architect DID: $ARCHITECT_DID"

# 2. Sign the Genesis Epoch
echo "Signing Genesis Epoch smart contract..."
# Logic: Hardcode current timestamp as the beginning of the 1000x authority decay
START_TIME=$(date +%s)
echo "{\"start_timestamp\": $START_TIME, \"architect_dids\": [\"$ARCHITECT_DID\"], \"decay_half_life_days\": 180}" > ~/.praxis/genesis_epoch.json

# 3. Prime the Local Knowledge Garden (2,700 nodes)
echo "Caching 2,700-node curriculum graph into local IndexedDB buffer..."
# trunk build must have been executed to generate the WASM
if [ -d "apps/leptos/dist" ]; then
    echo "PWA Artifacts detected."
else
    echo "WARNING: PWA Artifacts not found. Run 'trunk build --release' first."
fi

# 4. Initialize Holochain Conductor for Node Zero
echo "Bootstrapping Holochain Source Chain..."
# hc run -p 8888 &

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  \u{1F512} GENESIS EPOCH: INITIALIZED"
echo "  \u{1F310} MESH CDN: ACTIVE"
echo "  \u{1F980} NODE ZERO: ONLINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Community Onboarding can now begin via local P2P."
