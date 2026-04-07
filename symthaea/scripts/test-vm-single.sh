#!/usr/bin/env bash
# test-vm-single.sh — Single-disk NVMe VM for basic install testing
set -euo pipefail

VM_DIR="/var/cache/qemu/symthaea-test"
DISK="$VM_DIR/nvme-single.qcow2"
SSH_PORT=2223

mkdir -p "$VM_DIR"
[ -f "$DISK" ] || qemu-img create -f qcow2 "$DISK" 256G

ISO=$(find "$VM_DIR" -name "nixos-minimal-*.iso" -type f 2>/dev/null | head -1)
[ -z "$ISO" ] && { echo "No NixOS ISO in $VM_DIR"; exit 1; }

OVMF=$(find /run/current-system/sw/share -name "OVMF.fd" 2>/dev/null | head -1)
[ -z "$OVMF" ] && { echo "OVMF not found"; exit 1; }

echo "Single NVMe VM — SSH on localhost:$SSH_PORT"

qemu-system-x86_64 \
  -enable-kvm -m 4096 -smp 4 -bios "$OVMF" \
  -device nvme,id=nvme0,serial=SINGLE001 \
  -drive file="$DISK",if=none,id=drv0,format=qcow2 \
  -device nvme-ns,drive=drv0,bus=nvme0 \
  -cdrom "$ISO" -boot d \
  -net nic,model=virtio -net user,hostfwd=tcp::${SSH_PORT}-:22 \
  -display gtk
