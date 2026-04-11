#!/usr/bin/env bash
# Run platform integration tests sequentially with minimal resource usage.
# Use when no other cargo builds are active.
#
# Usage: ./scripts/run-platform-tests.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export CARGO_BUILD_JOBS=2

echo "=== Platform Unit Tests ==="
cargo test -p symthaea-exoskeleton --lib 2>&1 | grep "test result"
cargo test -p symthaea-surgical --lib 2>&1 | grep "test result"
cargo test -p symthaea-orbital --lib 2>&1 | grep "test result"
cargo test -p symthaea-quadruped --lib 2>&1 | grep "test result"

echo ""
echo "=== Integration Tests (requires humanoid feature) ==="
echo "Running exoskeleton in cognitive loop..."
cargo test --features humanoid,exoskeleton --test platform_integration \
  -- test_exoskeleton_in_cognitive_loop --nocapture 2>&1 | grep -E "EXOSKELETON|test result"

echo "Running surgical in cognitive loop..."
cargo test --features humanoid,surgical --test platform_integration \
  -- test_surgical_in_cognitive_loop --nocapture 2>&1 | grep -E "SURGICAL|test result"

echo "Running orbital in cognitive loop..."
cargo test --features humanoid,orbital --test platform_integration \
  -- test_orbital_in_cognitive_loop --nocapture 2>&1 | grep -E "ORBITAL|test result"

echo "Running quadruped in cognitive loop..."
cargo test --features humanoid,quadruped --test platform_integration \
  -- test_quadruped_in_cognitive_loop --nocapture 2>&1 | grep -E "QUADRUPED|test result"

echo ""
echo "=== Head-to-Head Comparison ==="
cargo test --features humanoid,exoskeleton,surgical,orbital,quadruped \
  --test platform_integration -- test_head_to_head --nocapture 2>&1 | \
  grep -E "HEAD-TO-HEAD|Platform|Humanoid|Exoskeleton|Surgical|Orbital|Quadruped|Disembodied|test result"

echo ""
echo "Done."
