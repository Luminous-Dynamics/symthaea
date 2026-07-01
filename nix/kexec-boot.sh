#!/usr/bin/env bash
# Boot into NixForHumanity installer without a USB drive.
# Run this on any Linux machine:
#   curl -sL https://install.nixforhumanity.org/boot | sudo bash
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║  NixForHumanity — Network Boot           ║"
echo "║  No USB drive needed.                    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check we're root
if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: Run with sudo"
    exit 1
fi

# Check kexec is available
if ! command -v kexec >/dev/null 2>&1; then
    echo "Installing kexec-tools..."
    apt-get install -y kexec-tools 2>/dev/null || \
    dnf install -y kexec-tools 2>/dev/null || \
    pacman -S --noconfirm kexec-tools 2>/dev/null || \
    { echo "ERROR: Cannot install kexec-tools. Install manually."; exit 1; }
fi

# Download NixOS minimal netboot kernel + initrd
MIRROR="https://channels.nixos.org/nixos-unstable"
echo "Downloading NixOS kernel..."
curl -L -o /tmp/nixos-kernel "${MIRROR}/latest-nixos-minimal-x86_64-linux.iso" 2>&1 | tail -1

# For now, just print instructions — full kexec requires extracting kernel/initrd from ISO
echo ""
echo "Downloaded NixOS ISO to /tmp/nixos-kernel"
echo ""
echo "To boot without USB:"
echo "  1. Mount the ISO: mount -o loop /tmp/nixos-kernel /mnt"
echo "  2. kexec -l /mnt/boot/bzImage --initrd=/mnt/boot/initrd"
echo "  3. kexec -e"
echo ""
echo "Or flash to USB with:"
echo "  dd if=/tmp/nixos-kernel of=/dev/sdX bs=4M status=progress"
