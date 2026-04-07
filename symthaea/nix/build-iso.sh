#!/usr/bin/env bash
# Build the Sovereign Inoculation NixOS installer ISO.
#
# Prerequisites: NixOS or Nix installed, the relay binary built.
#
# Usage:
#   # First, build the relay:
#   cargo build --bin ssh-relay --features server -p symthaea-spore --release
#
#   # Then build the ISO:
#   ./nix/build-iso.sh
#
# Output: result/iso/sovereign-inoculation-*.iso

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "╔══════════════════════════════════════════╗"
echo "║   Building Sovereign Inoculation ISO     ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check relay binary exists
if [ ! -f target/release/ssh-relay ]; then
    echo "Relay binary not found. Building..."
    cargo build --bin ssh-relay --features server -p symthaea-spore --release
fi

echo "Relay binary: $(du -h target/release/ssh-relay | cut -f1)"
echo ""

# Build ISO
echo "Building ISO (this takes 5-15 minutes)..."
nix-build nix/installer-iso.nix -o result-iso

ISO=$(ls result-iso/iso/*.iso 2>/dev/null | head -1)
if [ -z "$ISO" ]; then
    echo "ERROR: ISO not found in result-iso/iso/"
    exit 1
fi

SIZE=$(du -h "$ISO" | cut -f1)
echo ""
echo "════════════════════════════════════════════"
echo "  ISO built successfully!"
echo "  File: $ISO"
echo "  Size: $SIZE"
echo ""
echo "  Flash to USB:"
echo "    sudo dd if=$ISO of=/dev/sdX bs=4M status=progress"
echo "  Or use Etcher: https://etcher.balena.io"
echo ""
echo "  Upload to GitHub release:"
echo "    gh release create v0.1.0 '$ISO' --title 'Sovereign Inoculation v0.1.0'"
echo "════════════════════════════════════════════"
