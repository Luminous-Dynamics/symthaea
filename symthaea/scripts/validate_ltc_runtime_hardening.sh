#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

# Define a clean isolated target directory to avoid stale process locks
export CARGO_TARGET_DIR="/tmp/symthaea-ltc-hardening-target"
echo "=== Starting LTC Runtime Hardening Validation (Target Dir: $CARGO_TARGET_DIR) ==="

echo "1. Running HDC-LTC Crate Unit and Property Tests..."
cargo test -p symthaea-hdc-ltc

echo "2. Running Probe Stream Integration Tests..."
cargo test -p symthaea-probe-stream

echo "3. Running Irregular Timestep Replay Example..."
cargo run -p symthaea-hdc-ltc --example irregular_timestep_replay

echo "4. Running Z3 Formal Stability Verification Test..."
cargo test -p symthaea-core --test fol_ext_stability_verification

echo "=== LTC Runtime Hardening Validation: SUCCESS ==="
