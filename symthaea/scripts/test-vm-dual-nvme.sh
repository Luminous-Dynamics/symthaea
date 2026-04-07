#!/usr/bin/env bash
# test-vm-dual-nvme.sh — E2E test: dual-NVMe VM for installer validation
#
# Creates a QEMU VM with 2 emulated NVMe drives, boots from NixOS minimal ISO,
# and exposes SSH on port 2222 for the Symthaea portal to connect.
#
# Prerequisites:
#   - NixOS minimal ISO (auto-downloads if missing)
#   - OVMF UEFI firmware
#   - eval-api running on :8090
#   - ssh-relay running on :8091
#
# Usage:
#   ./scripts/test-vm-dual-nvme.sh              # Create + boot VM
#   ./scripts/test-vm-dual-nvme.sh --clean      # Delete VM disk images
#   ./scripts/test-vm-dual-nvme.sh --headless   # No display (SSH only)

set -euo pipefail

VM_DIR="/var/cache/qemu/symthaea-test"
FAST_DISK="$VM_DIR/nvme-fast.qcow2"
STD_DISK="$VM_DIR/nvme-standard.qcow2"
ISO_DIR="$VM_DIR"
SSH_PORT=2222
RAM=4096
CPUS=4
DISPLAY_OPT="-display gtk"

# Parse args
for arg in "$@"; do
  case "$arg" in
    --clean)
      echo "Cleaning VM disk images..."
      rm -rf "$VM_DIR"
      echo "Done."
      exit 0
      ;;
    --headless)
      DISPLAY_OPT="-display none -daemonize"
      ;;
  esac
done

mkdir -p "$VM_DIR"

# Create disk images if they don't exist
if [ ! -f "$FAST_DISK" ]; then
  echo "Creating fast NVMe disk (256G)..."
  qemu-img create -f qcow2 "$FAST_DISK" 256G
fi

if [ ! -f "$STD_DISK" ]; then
  echo "Creating standard NVMe disk (256G)..."
  qemu-img create -f qcow2 "$STD_DISK" 256G
fi

# Find NixOS ISO
ISO=$(find "$ISO_DIR" -name "nixos-minimal*.iso" -o -name "nixos-minimal.iso" -type f 2>/dev/null | head -1)
if [ -z "$ISO" ]; then
  echo "No NixOS minimal ISO found in $ISO_DIR"
  echo "Downloading..."
  curl -L -o "$ISO_DIR/nixos-minimal.iso" \
    "https://channels.nixos.org/nixos-unstable/latest-nixos-minimal-x86_64-linux.iso"
  ISO="$ISO_DIR/nixos-minimal.iso"
fi

# Find OVMF firmware
OVMF=$(find /nix/store -name "OVMF.fd" -path "*/FV/*" 2>/dev/null | head -1)
if [ -z "$OVMF" ]; then
  # Fallback paths
  for path in \
    /run/current-system/sw/share/qemu/OVMF.fd \
    /run/current-system/sw/share/OVMF/OVMF.fd; do
    [ -f "$path" ] && OVMF="$path" && break
  done
fi

if [ -z "$OVMF" ]; then
  echo "OVMF UEFI firmware not found."
  echo "On NixOS: nix-env -iA nixpkgs.OVMF"
  exit 1
fi

echo "=== Symthaea E2E Test VM ==="
echo "Fast NVMe:     $FAST_DISK"
echo "Standard NVMe: $STD_DISK"
echo "ISO:           $ISO"
echo "OVMF:          $OVMF"
echo "SSH:           localhost:$SSH_PORT"
echo ""
echo "After VM boots:"
echo "  1. Set root password:  passwd root"
echo "  2. Start SSH:          systemctl start sshd"
echo "  3. Open portal, go to Inoculate tab"
echo "  4. Connect via SSH: host=localhost, port=$SSH_PORT, user=root"
echo "  5. Select drive → Deploy"
echo ""
echo "Starting VM..."

qemu-system-x86_64 \
  -enable-kvm \
  -m "$RAM" \
  -smp "$CPUS" \
  -bios "$OVMF" \
  -device nvme,id=nvme0,serial=SAMSUNG-FAST \
  -drive file="$FAST_DISK",if=none,id=drv0,format=qcow2 \
  -device nvme-ns,drive=drv0,bus=nvme0 \
  -device nvme,id=nvme1,serial=INTEL-STD \
  -drive file="$STD_DISK",if=none,id=drv1,format=qcow2 \
  -device nvme-ns,drive=drv1,bus=nvme1 \
  -cdrom "$ISO" \
  -boot d \
  -net nic,model=virtio \
  -net user,hostfwd=tcp::${SSH_PORT}-:22 \
  $DISPLAY_OPT
