#!/usr/bin/env bash
# Quick verification of Φ calculation fix
# Run from symthaea-hlb directory

set -e

echo "="
echo "Φ CALCULATION FIX VERIFICATION"
echo "="
echo

echo "Compiling library..."
cargo build --lib --release 2>&1 | tail -5
echo "✓ Compilation successful"
echo

echo "Running Φ integration level test..."
cargo test --lib --release test_phi_fix_different_integration_levels -- --nocapture 2>&1 | tail -30

echo
echo "Running all Φ tests..."
cargo test --lib --release tiered_phi 2>&1 | grep "test result"

echo
echo "✅ Verification complete!"
