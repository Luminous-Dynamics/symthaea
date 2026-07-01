#!/usr/bin/env bash
# test-vm-sata.sh — SATA disk VM for testing SATA disko profile
set -euo pipefail

VM_DIR="/var/cache/qemu/symthaea-test"
DISK="$VM_DIR/sata-disk.qcow2"
SSH_PORT=2224

mkdir -p "$VM_DIR"
[ -f "$DISK" ] || qemu-img create -f qcow2 "$DISK" 256G

ISO=$(find "$VM_DIR" -name "nixos-minimal-*.iso" -type f 2>/dev/null | head -1)
[ -z "$ISO" ] && { echo "No NixOS ISO in $VM_DIR"; exit 1; }

OVMF=$(find /run/current-system/sw/share -name "OVMF.fd" 2>/dev/null | head -1)
[ -z "$OVMF" ] && { echo "OVMF not found"; exit 1; }

echo "SATA disk VM — SSH on localhost:$SSH_PORT"

qemu-system-x86_64 \
  -enable-kvm -m 4096 -smp 4 -bios "$OVMF" \
  -drive file="$DISK",if=none,id=sata0,format=qcow2 \
  -device ahci,id=ahci -device ide-hd,drive=sata0,bus=ahci.0 \
  -cdrom "$ISO" -boot d \
  -net nic,model=virtio -net user,hostfwd=tcp::${SSH_PORT}-:22 \
  -display gtk
