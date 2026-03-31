#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Seed the Terra Atlas Holochain DNA with static data.
# Prerequisites:
#   1. Holochain conductor running with atlas DNA installed
#   2. hc CLI available (from holochain dev tools)
#   3. App WebSocket on port 8300 (default)
#
# Usage: ./scripts/seed-data.sh [--conductor-url ws://localhost:8300]

set -euo pipefail

CONDUCTOR_URL="${1:-ws://localhost:8300}"
DATA_DIR="$(dirname "$0")/../../terra-atlas-leptos/assets/data"

echo "=== Terra Atlas DHT Seed ==="
echo "Conductor: $CONDUCTOR_URL"
echo "Data dir: $DATA_DIR"
echo ""

# Check data files exist
for f in sites-clustered.json maglev-network.json resontia-vaults.json \
         terra-lumina-sites.json fossil-deposits.json nuclear-sites.json; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "ERROR: Missing $DATA_DIR/$f"
        exit 1
    fi
done

echo "All data files found."
echo ""
echo "To seed, run the Rust seed binary:"
echo "  cargo run -p atlas-seed -- --conductor-url $CONDUCTOR_URL"
echo ""
echo "Or manually via hc CLI:"
echo "  cat $DATA_DIR/fossil-deposits.json | jq -c '.[]' | while read -r deposit; do"
echo "    hc call atlas_infrastructure register_fossil_deposit \"\$deposit\""
echo "  done"
echo ""
echo "Data counts:"
echo "  Sites: $(jq length "$DATA_DIR/sites-clustered.json")"
echo "  Geothermal: $(jq '.geothermal_nodes | length' "$DATA_DIR/maglev-network.json")"
echo "  Corridors: $(jq '.maglev_corridors | length' "$DATA_DIR/maglev-network.json")"
echo "  Vaults: $(jq length "$DATA_DIR/resontia-vaults.json")"
echo "  Terra Lumina: $(jq length "$DATA_DIR/terra-lumina-sites.json")"
echo "  Fossil Deposits: $(jq length "$DATA_DIR/fossil-deposits.json")"
echo "  Nuclear Sites: $(jq length "$DATA_DIR/nuclear-sites.json")"
echo ""
echo "Total: $(( \
    $(jq length "$DATA_DIR/sites-clustered.json") + \
    $(jq '.geothermal_nodes | length' "$DATA_DIR/maglev-network.json") + \
    $(jq '.maglev_corridors | length' "$DATA_DIR/maglev-network.json") + \
    $(jq length "$DATA_DIR/resontia-vaults.json") + \
    $(jq length "$DATA_DIR/terra-lumina-sites.json") + \
    $(jq length "$DATA_DIR/fossil-deposits.json") + \
    $(jq length "$DATA_DIR/nuclear-sites.json") \
)) entries ready to seed"
