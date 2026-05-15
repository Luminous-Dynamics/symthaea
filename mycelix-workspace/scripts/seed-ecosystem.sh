#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Seed Data Constellation — Live validator for decoupled hApps.
#
# 1. Starts a Holochain sandbox.
# 2. Installs standalone Substrate hApps (Identity, Finance).
# 3. Installs standalone Satellite hApps (Civic, Knowledge).
# 4. Seeds data and grants cross-hApp Capability access.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Bypass interactive passphrase prompt for sandbox keys
export HC_SANDBOX_PASSPHRASE=""

echo "🍄 Mycelix: Seeding Constellation..."

# Ensure we're in a nix shell with tools
if ! command -v hc &>/dev/null; then
  echo "Error: 'hc' CLI not found. Run this inside 'nix develop'."
  exit 1
fi

# 1. Setup Sandbox
echo "--- Initializing Sandbox ---"
hc sandbox clean
# Create a new sandbox
echo "" | hc sandbox --piped create --num-sandboxes 1
# Run the sandbox with forced admin port 4444
echo "" | hc sandbox --piped -f 4444 run 0 &
SANDBOX_PID=$!
sleep 10 # Wait for conductor to start

ROOT_DIR="/srv/luminous-dynamics"

# 2. Install Substrates
echo "--- Installing Substrates ---"
echo "" | hc sandbox --piped call -r 4444 install-app \
  --app-id "substrate-identity" \
  "$ROOT_DIR/mycelix-identity/mycelix-identity.happ"

echo "" | hc sandbox --piped call -r 4444 install-app \
  --app-id "substrate-finance" \
  "$ROOT_DIR/mycelix-finance/mycelix-finance.happ"

# 3. Install Satellites
echo "--- Installing Satellites ---"
echo "" | hc sandbox --piped call -r 4444 install-app \
  --app-id "satellite-civic" \
  "$ROOT_DIR/mycelix-civic/mycelix-civic.happ"

echo "" | hc sandbox --piped call -r 4444 install-app \
  --app-id "satellite-knowledge" \
  "$ROOT_DIR/mycelix-knowledge/knowledge.happ"

echo "Waiting for apps to settle..."
sleep 10

# 4. Get Agent Key
echo "--- Fetching Agent Key ---"
# We list agents from the running conductor
AGENT_KEY=$(echo "" | hc sandbox --piped call -r 4444 list-agents | grep -o "uhCAk[^ ]*" | head -1)
echo "  Agent: $AGENT_KEY"
AGENT_DID="did:mycelix:$AGENT_KEY"

# 5. Seed Identity Data
echo "--- Seeding Identity ---"
# Report a 0.8 score from 'finance' cluster
PAYLOAD="{\"agent_pubkey_b64\":\"$AGENT_KEY\", \"cluster\":\"finance\", \"score\":0.8}"
echo "" | hc sandbox --piped call -r 4444 zome-call \
  --app-id "substrate-identity" \
  --zome "reputation_aggregator" \
  --fn "report_domain_score" \
  "$PAYLOAD"

# 6. Verify Cross-hApp Connectivity
echo "--- Verifying Constellation Connectivity ---"
# Call 'verify_tier_remote' on Civic, which calls Identity
# This verifies the Constellation Protocol end-to-end
echo "" | hc sandbox --piped call -r 4444 zome-call \
  --app-id "satellite-civic" \
  --zome "civic_bridge" \
  --fn "verify_tier_remote" \
  "\"$AGENT_KEY\""

echo "✅ Constellation seeded and verified!"
echo "Killing sandbox ($SANDBOX_PID)..."
kill $SANDBOX_PID
