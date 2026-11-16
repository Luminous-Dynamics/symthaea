#!/usr/bin/env bash
# Reproduce the public Binius proof-circuit tests and benchmark demonstrations.
#
# Scope:
# - Runs the standalone Binius crate included in this public repository.
# - Does NOT reproduce a Winterfell XOR proof or the health-attestation pipeline.
# - Winterfell values printed by main.rs are analytic estimates, not measurements.
#
# Run from any directory:
#   bash papers/theory-foundations/binius-hdc/reproduce.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
HDC_BENCH="$REPO_ROOT/crates/hdc-zkp-bench"

echo "=== Public Binius HDC proof-circuit tests ==="
RUSTFLAGS="-C target-cpu=native" cargo test --manifest-path "$HDC_BENCH/Cargo.toml"

echo
echo "=== Public Binius proof demonstrations ==="
RUSTFLAGS="-C target-cpu=native" cargo run --release --manifest-path "$HDC_BENCH/Cargo.toml"

echo
echo "Complete. Treat benchmark outputs as Binius measurements only; Winterfell figures are explicitly modelled estimates."
