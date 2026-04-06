#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Start the EduNet Holochain conductor in the foreground.
# Run in tmux/screen: nix develop --command bash scripts/start-conductor.sh
set -euo pipefail

echo "=== EduNet Conductor ==="

# Build hApp if needed
if [ ! -f "happ/mycelix-edunet.happ" ]; then
    echo "Building hApp..."
    cargo build --workspace --target wasm32-unknown-unknown --release 2>&1 | tail -1
    hc dna pack dna/ -o dna/edunet.dna
    cp dna/edunet.dna happ/edunet.dna
    hc app pack happ/ -o happ/mycelix-edunet.happ
fi

echo "Starting sandbox conductor (foreground)..."
echo "Press Ctrl+C to stop."
echo ""
echo "passphrase" | hc sandbox --piped generate happ/mycelix-edunet.happ --run=8888
