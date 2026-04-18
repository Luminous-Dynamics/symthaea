#!/usr/bin/env bash
# Reproduce all benchmark results from:
# "Binary-Field STARKs for Hyperdimensional Computing"
#
# Requirements: Rust (stable), ~32GB RAM for full-scale Winterfell
# Usage: cd papers/binius-hdc && bash reproduce.sh
#
# This script runs ALL benchmarks and outputs paper-ready tables.
# Expected runtime: ~3-5 minutes (including 16K Winterfell ~24s)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ZKP_CORE="$REPO_ROOT/crates/mycelix-zkp-core"
HDC_BENCH="$REPO_ROOT/crates/hdc-zkp-bench"
HEALTH_ZKP="$REPO_ROOT/mycelix-health/crates/health-zkp"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Reproducing: Binary-Field STARKs for HDC                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Table 2: XOR Binding Comparison (both backends, same scale) ──
echo "=== Table 2: XOR Binding (Binius vs Winterfell) ==="
echo "Running Binius 16,384-bit XOR + Winterfell 256-bit + 16,384-bit..."
cd "$HDC_BENCH"
RUSTFLAGS="-C target-cpu=native" cargo run --release 2>&1 | grep -E "AND constr|Proof size|Prover time|Verifier time|Verification|COMPARISON" | head -20
echo ""

echo "Running Winterfell 16,384-bit XOR (this takes ~24 seconds)..."
cd "$ZKP_CORE"
cargo test circuits::winterfell_xor::tests::test_benchmark_16384bit_xor \
  --lib --features backend-winterfell -- --ignored --nocapture 2>&1 | \
  grep -E "MEASURED|constraints|Prove|Verify|Proof"
echo ""

# ── Table 3: E2E Health Pipeline ──
echo "=== Table 3: E2E Health Attestation Pipeline ==="
cd "$HEALTH_ZKP"
cargo test --test e2e_health_attestation --features verify-full \
  --target x86_64-unknown-linux-gnu -- --nocapture 2>&1 | \
  grep -E "PAPER METRICS|STARK|Dilithium|Total|Proof size|Signature|PASSED" | head -10
echo ""

# ── Table 4: All HDC Operations ──
echo "=== Table 4: HDC Operation Summary ==="
cd "$HDC_BENCH"
RUSTFLAGS="-C target-cpu=native" cargo run --release 2>&1 | \
  grep -E "AND constr|Proof size|Prover time|BUNDLING|HAMMING|CfC|SIGMOID|PASSED" | head -25
echo ""

# ── Unit test validation ──
echo "=== Validation: All tests pass ==="
cd "$ZKP_CORE"
cargo test --lib --features full 2>&1 | grep "test result"
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All benchmarks complete. See output above for paper tables. ║"
echo "╚══════════════════════════════════════════════════════════════╝"
