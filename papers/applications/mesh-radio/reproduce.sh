#!/usr/bin/env bash
# Reproduce the code-metrics in mesh_radio.tex.
#
# The paper is at present SIMULATOR-VALIDATED, not hardware-deployed
# (see "Scope of Validation" section). This script runs the unit and
# integration tests that exercise the Radio Dispatcher, Spectrum Manager,
# and Mesh subsystem against simulated link characteristics.
#
# NOT RUN: LoRa hardware deployment (planned as hardware revision work),
# end-to-end power-consumption measurement, over-the-air packet traces.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

echo "=== Mesh Radio reproduction (simulator) ==="
cd "$REPO"

echo
echo "[1/3] Radio Dispatcher (6,309 LOC, 151 tests) — 7-file module"
cargo test --release --lib cognitive_loop::managers::radio_dispatcher --features mesh

echo
echo "[2/3] Mesh subsystem (10,120 LOC, 222 tests)"
cargo test --release --lib swarm::mesh --features mesh

echo
echo "[3/3] Spectrum Manager + tier selection"
cargo test --release --lib spectrum_manager --features mesh

echo
echo "=== Done. Simulator-only validation. ==="
echo "For hardware deployment see Scope of Validation section in the paper."
