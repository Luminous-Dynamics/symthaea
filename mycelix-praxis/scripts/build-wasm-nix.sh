#!/usr/bin/env bash
set -euo pipefail

# Build all zomes to WASM inside nix develop environment
# This script should be run from the project root

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "🔨 Building all 20 WASM zomes..."
echo "Using nix develop environment for proper tooling..."

# Run the build
nix develop --command cargo build --target wasm32-unknown-unknown --release \
  -p learning_integrity \
  -p fl_integrity \
  -p credential_integrity \
  -p dao_integrity \
  -p srs_integrity \
  -p gamification_integrity \
  -p adaptive_integrity \
  -p integration_integrity \
  -p pods_integrity \
  -p knowledge_integrity \
  -p learning_coordinator \
  -p fl_coordinator \
  -p credential_coordinator \
  -p dao_coordinator \
  -p srs_coordinator \
  -p gamification_coordinator \
  -p adaptive_coordinator \
  -p integration_coordinator \
  -p pods_coordinator \
  -p knowledge_coordinator

echo ""
echo "✅ All 20 WASM zomes built successfully!"
echo ""

# List the built WASM files
echo "📦 Built WASM files:"
find target/wasm32-unknown-unknown/release -name "*.wasm" -type f 2>/dev/null | sort
