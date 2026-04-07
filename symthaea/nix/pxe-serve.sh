#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Serve NixForHumanity installer over the network for PXE boot.
# Run this on any NixOS machine to let other machines boot from the network.
#
# Usage:
#   ./pxe-serve.sh [port]       (default: 8080)
#
# Prerequisites:
#   - NixOS kernel and initrd must exist in /nix/store (build the ISO first)
#   - python3 must be available
set -euo pipefail

PORT="${1:-8080}"
KERNEL=$(ls /nix/store/*/bzImage 2>/dev/null | head -1 || true)
INITRD=$(ls /nix/store/*/initrd 2>/dev/null | head -1 || true)
IP=$(ip -4 route get 1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' | head -1 || true)

if [ -z "$KERNEL" ] || [ -z "$INITRD" ]; then
    echo "ERROR: NixOS kernel/initrd not found in nix store."
    echo "Build the ISO first:"
    echo "  nix-build nix/installer-iso.nix"
    exit 1
fi

if [ -z "$IP" ]; then
    IP="127.0.0.1"
fi

echo ""
echo "  NixForHumanity PXE Server"
echo "  ========================="
echo ""
echo "  Serving on http://$IP:$PORT"
echo ""
echo "  Kernel: $KERNEL"
echo "  Initrd: $INITRD"
echo ""
echo "  --- Setup Instructions ---"
echo ""
echo "  For dnsmasq PXE, add to dnsmasq.conf:"
echo "    dhcp-boot=pxelinux.0,,$IP"
echo "    dhcp-option=66,$IP"
echo ""
echo "  For pixiecore (simplest):"
echo "    sudo pixiecore boot $KERNEL $INITRD \\"
echo "      --cmdline='init=\$(ls /nix/store/*/init | head -1)'"
echo ""
echo "  For iPXE chain loading:"
echo "    kernel http://$IP:$PORT/bzImage"
echo "    initrd http://$IP:$PORT/initrd"
echo "    boot"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

ln -s "$KERNEL" "$TMPDIR/bzImage"
ln -s "$INITRD" "$TMPDIR/initrd"

# Also create a simple iPXE script
cat > "$TMPDIR/boot.ipxe" <<IPXE
#!ipxe
kernel http://$IP:$PORT/bzImage
initrd http://$IP:$PORT/initrd
boot
IPXE

cd "$TMPDIR"
python3 -m http.server "$PORT" --bind 0.0.0.0
