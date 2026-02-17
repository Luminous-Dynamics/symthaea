#!/usr/bin/env bash
set -euo pipefail

# Build all coordinator zomes to wasm32 with helpful checks and local cargo home.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARGO_HOME="${ROOT_DIR}/.cargo"

missing=0
for bin in rust-lld ld.lld; do
  if command -v "$bin" >/dev/null 2>&1; then
    export RUSTFLAGS="${RUSTFLAGS:-} -Clinker=${bin}"
    missing=0
    break
  else
    missing=1
  fi
done

if [[ $missing -eq 1 ]]; then
  echo "error: rust-lld/ld.lld not found; install a linker (e.g., llvm lld) and retry." >&2
  exit 1
fi

echo "Using linker flags: ${RUSTFLAGS:-<none>}"
echo "Building zome coordinators for wasm32-unknown-unknown..."

pushd "${ROOT_DIR}" >/dev/null

# Build all integrity zomes
echo "Building integrity zomes..."
cargo build \
  --target wasm32-unknown-unknown \
  --release \
  -p learning_integrity \
  -p fl_integrity \
  -p credential_integrity \
  -p dao_integrity \
  -p srs_integrity \
  -p gamification_integrity \
  -p adaptive_integrity \
  -p integration_integrity \
  -p pods_integrity \
  -p knowledge_integrity

# Build all coordinator zomes
echo "Building coordinator zomes..."
cargo build \
  --target wasm32-unknown-unknown \
  --release \
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

popd >/dev/null

echo "✅ All 20 WASM zomes built successfully"
