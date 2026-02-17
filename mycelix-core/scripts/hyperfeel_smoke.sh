#!/usr/bin/env bash
#
# HyperFeel end-to-end smoke test
#
# This script:
#   1. Enters the Mycelix-Core Nix dev shell
#   2. Builds & tests the Rust `mycelix-hyperfeel` crate
#   3. Runs the Python HyperFeel tests (bridge + encoder)
#
# Usage (from the Mycelix-Core directory):
#   ./scripts/hyperfeel_smoke.sh
#

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

echo "🧪 Running HyperFeel smoke test via Nix dev shell..."
echo "   Root: $ROOT_DIR"
echo

nix develop .#default --command bash -lc '
  set -euo pipefail

  echo "📦 Step 1: Build & test Rust mycelix-hyperfeel crate"
  cd 0TML/mycelix-hyperfeel
  cargo test
  cargo build --release

  echo
  echo "📦 Step 2: Run Python HyperFeel bridge + encoder tests"
  export PATH="$PWD/target/release:$PATH"
  cd ..
  pytest src/tests/test_hyperfeel_bridge.py src/tests/test_hyperfeel.py

  echo
  echo "✅ HyperFeel smoke test completed successfully."
'

