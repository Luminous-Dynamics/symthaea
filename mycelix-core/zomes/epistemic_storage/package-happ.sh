#!/usr/bin/env bash
# Package epistemic_storage DNA and hApp
# Must be run from within nix develop .#holochain environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${SCRIPT_DIR}/workdir"

echo "=== Epistemic Storage Packager ==="
echo ""

# Check for hc tool
if ! command -v hc &> /dev/null; then
    echo "Error: 'hc' tool not found."
    echo ""
    echo "Please enter the Holochain nix environment first:"
    echo "  cd /srv/luminous-dynamics/mycelix-workspace"
    echo "  nix develop .#holochain"
    echo ""
    exit 1
fi

echo "hc version: $(hc --version)"
echo ""

# Ensure workdir exists
mkdir -p "${WORKDIR}"

# Check WASM files exist
INTEGRITY_WASM="${SCRIPT_DIR}/target/wasm32-unknown-unknown/release/epistemic_storage_integrity.wasm"
COORDINATOR_WASM="${SCRIPT_DIR}/target/wasm32-unknown-unknown/release/epistemic_storage_coordinator.wasm"

if [[ ! -f "$INTEGRITY_WASM" ]] || [[ ! -f "$COORDINATOR_WASM" ]]; then
    echo "Error: WASM files not found. Build first:"
    echo "  cargo build --release --target wasm32-unknown-unknown"
    exit 1
fi

echo "✓ Found WASM files:"
echo "  - $(basename $INTEGRITY_WASM) ($(du -h $INTEGRITY_WASM | cut -f1))"
echo "  - $(basename $COORDINATOR_WASM) ($(du -h $COORDINATOR_WASM | cut -f1))"
echo ""

# Copy WASM to workdir (for reference)
cp "$INTEGRITY_WASM" "${WORKDIR}/"
cp "$COORDINATOR_WASM" "${WORKDIR}/"

# Package DNA
echo "Packaging DNA..."
cd "${SCRIPT_DIR}"
hc dna pack . -o "${WORKDIR}/epistemic_storage.dna"

if [[ -f "${WORKDIR}/epistemic_storage.dna" ]]; then
    echo "✓ DNA packaged: $(du -h ${WORKDIR}/epistemic_storage.dna | cut -f1)"
else
    echo "Error: DNA packaging failed"
    exit 1
fi

# Package hApp
echo ""
echo "Packaging hApp..."
hc app pack . -o "${WORKDIR}/epistemic_storage.happ"

if [[ -f "${WORKDIR}/epistemic_storage.happ" ]]; then
    echo "✓ hApp packaged: $(du -h ${WORKDIR}/epistemic_storage.happ | cut -f1)"
else
    echo "Error: hApp packaging failed"
    exit 1
fi

echo ""
echo "=== Packaging Complete ==="
echo ""
echo "Files in ${WORKDIR}/:"
ls -lh "${WORKDIR}/"
echo ""
echo "Next steps:"
echo "  # Run with sandbox conductor:"
echo "  hc sandbox generate ${WORKDIR}/epistemic_storage.happ --run 8888"
echo ""
echo "  # Run SDK integration tests:"
echo "  cd /srv/luminous-dynamics/mycelix-workspace/sdk-ts"
echo "  CONDUCTOR_AVAILABLE=true npm test tests/storage/dht.test.ts"
