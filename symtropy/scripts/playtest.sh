#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Symtropy playtest script — run game with governance systems + 4D rendering
#
# Usage:
#   ./scripts/playtest.sh              # Normal playtest
#   ./scripts/playtest.sh --headless   # Headless validation (300 ticks)
#   ./scripts/playtest.sh --ai         # AI player mode (Symthaea plays)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-}"

echo "=== Symtropy Playtest ==="
echo "Date: $(date -I)"
echo ""

if [[ "$MODE" == "--headless" ]]; then
    echo "Running headless validation (300 ticks)..."
    cargo run --release -p symtropy-sim-bridge --bin headless_test -- --ticks 300 2>&1
    echo ""
    echo "Headless validation complete."
    exit 0
fi

FEATURES=""
ARGS=""

# Check if mycelix feature compiles (governance/economy/faction systems)
if cargo check --features mycelix 2>/dev/null; then
    FEATURES="--features mycelix"
    echo "Mycelix governance: ENABLED"
else
    echo "Mycelix governance: DISABLED (compilation issue)"
fi

if [[ "$MODE" == "--ai" ]]; then
    ARGS="--ai-player"
    echo "Mode: AI player (Symthaea plays)"
else
    echo "Mode: Human player"
fi

echo ""
echo "=== Playtest Checklist ==="
echo "[ ] Player spawns and can move (WASD)"
echo "[ ] Flashlight flickers with stress"
echo "[ ] 3 NPC crew members visible (green)"
echo "[ ] Fusion Core visible (yellow)"
echo "[ ] Leviathan awakens on noise"
echo "[ ] F4 toggles 4D mode"
echo "[ ] [ and ] keys slide W hyperplane in 4D"
echo "[ ] Dimensional secrets visible only in 4D"
echo "[ ] HUD shows consciousness/energy info"
if [[ -n "$FEATURES" ]]; then
    echo "[ ] NPC governance proposals appear (FEP surprise > 0.4)"
    echo "[ ] TEND currency visible"
    echo "[ ] DKG ceremony triggers on core extraction"
fi
echo ""
echo "Starting game..."
cargo run --release $FEATURES -- $ARGS
