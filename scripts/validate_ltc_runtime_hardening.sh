#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# validate_ltc_runtime_hardening.sh
#
# Clean isolated validation of the entire Symthaea hardening arc:
#   Phase 2.7  — LTC timing / proptest / Z3 stability / epistemic modulation
#   Phase 2.8  — Causal graph topology + drift detection
#   Phase 2.9  — World model transitions + end-to-end integration harness
#
# Uses CARGO_TARGET_DIR=/tmp/symthaea-ltc-hardening-target to avoid
# stale build-lock contention with the main workspace target.

set -euo pipefail

export CARGO_TARGET_DIR="/tmp/symthaea-ltc-hardening-target"
echo "=== Starting LTC Runtime Hardening Validation (Target Dir: $CARGO_TARGET_DIR) ==="

echo ""
echo "1. Running HDC-LTC Crate Unit and Property Tests..."
cargo test -p symthaea-hdc-ltc

echo ""
echo "2. Running Probe Stream Integration Tests..."
cargo test -p symthaea-probe-stream

echo ""
echo "3. Running Irregular Timestep Replay Example..."
cargo run -p symthaea-hdc-ltc --example irregular_timestep_replay

echo ""
echo "4. Running Z3 Formal Stability Verification Test..."
cargo test -p symthaea-core --test fol_ext_stability_verification

echo ""
echo "5. Running Phase 2.7 — Active Inference / Epistemic Modulation Tests..."
cargo test -p nixward --lib "mind::active_inference"

echo ""
echo "6. Running Phase 2.8 — Causal Graph Hardening Tests..."
cargo test -p nixward --lib "mind::causal_graph"

echo ""
echo "7. Running Phase 2.8 — Drift Detection Hardening Tests..."
cargo test -p nixward --lib "mind::hdc_world_model"

echo ""
echo "8. Running Phase 2.9-A — World Model Transition Tests..."
cargo test -p nixward --lib "mind::world_model"

echo ""
echo "9. Running Phase 2.9-C — End-to-End Cross-Layer Integration Harness..."
cargo test -p nixward --test sensor_drift_causal_action_integration

echo ""
echo "=== LTC Runtime Hardening Validation: SUCCESS ==="
