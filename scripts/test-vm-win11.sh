#!/usr/bin/env bash
# test-vm-win11.sh — Windows 11 QEMU VM for dual-boot testing
#
# Creates a Win11 VM with UEFI, TPM 2.0, and Secure Boot for testing
# the Sovereign Inoculation alongside-Windows install flow.
#
# Prerequisites:
#   - Windows 11 Enterprise evaluation ISO (free, 90 days)
#     https://www.microsoft.com/en-us/evalcenter/evaluate-windows-11-enterprise
#   - VirtIO drivers ISO
#     https://fedorapeople.org/groups/virt/virtio-win/direct-downloads/stable-virtio/virtio-win.iso
#   - swtpm (software TPM) — `nix-shell -p swtpm`
#
# Usage:
#   WIN11_ISO=/path/to/Win11.iso ./scripts/test-vm-win11.sh          # Install
#   WIN11_ISO=/path/to/Win11.iso ./scripts/test-vm-win11.sh --boot   # Boot installed
#   ./scripts/test-vm-win11.sh --dual-boot                           # Boot with NixOS ISO too
#   ./scripts/test-vm-win11.sh --clean                               # Delete VM
#
# After Windows is installed:
#   1. Shrink C: by 40GB in Disk Management (leave as unallocated)
#   2. Enable BitLocker: manage-bde -on C: -RecoveryPassword
#   3. Reboot, then test NixOS alongside install

set -euo pipefail

VM_DIR="/var/cache/qemu/symthaea-test/win11"
DISK="$VM_DIR/disk.qcow2"
TPM_DIR="$VM_DIR/tpm"
VARS_FILE="$VM_DIR/OVMF_VARS.fd"
SSH_PORT=2223  # Different from NixOS VM (2222)
RAM=8192
CPUS=4

MODE="install"
for arg in "$@"; do
  case "$arg" in
    --boot) MODE="boot" ;;
    --dual-boot) MODE="dual-boot" ;;
    --clean)
      echo "Cleaning Win11 VM..."
      rm -rf "$VM_DIR"
      echo "Done."
      exit 0
      ;;
  esac
done

mkdir -p "$VM_DIR" "$TPM_DIR"

# Find OVMF firmware
OVMF_DIR=""
for candidate in \
  "$(nix-build '<nixpkgs>' -A OVMF.fd --no-out-link 2>/dev/null)/FV" \
  "/run/current-system/sw/share/qemu" \
  "/nix/store/$(ls /nix/store | grep OVMF | head -1)/FV"; do
  [ -f "$candidate/OVMF_CODE.fd" ] && OVMF_DIR="$candidate" && break
done

if [ -z "$OVMF_DIR" ]; then
  echo "ERROR: OVMF firmware not found. Try: nix-shell -p OVMF"
  exit 1
fi

# Create disk if needed
if [ ! -f "$DISK" ]; then
  echo "Creating 100G disk..."
  qemu-img create -f qcow2 "$DISK" 100G
fi

# Copy writable OVMF vars if needed
if [ ! -f "$VARS_FILE" ]; then
  cp "$OVMF_DIR/OVMF_VARS.fd" "$VARS_FILE"
fi

# Start software TPM
echo "Starting software TPM 2.0..."
swtpm socket \
  --tpmstate dir="$TPM_DIR" \
  --ctrl type=unixio,path="$TPM_DIR/sock" \
  --tpm2 \
  --log level=5 &
SWTPM_PID=$!
trap "kill $SWTPM_PID 2>/dev/null || true" EXIT
sleep 1

# Build QEMU args
QEMU_ARGS=(
  -name "Win11-DualBoot-Test"
  -machine q35,accel=kvm,smm=on
  -cpu host
  -smp "$CPUS"
  -m "$RAM"
  # UEFI firmware
  -drive "if=pflash,format=raw,readonly=on,file=$OVMF_DIR/OVMF_CODE.fd"
  -drive "if=pflash,format=raw,file=$VARS_FILE"
  # TPM 2.0
  -chardev "socket,id=chrtpm,path=$TPM_DIR/sock"
  -tpmdev emulator,id=tpm0,chardev=chrtpm
  -device tpm-tis,tpmdev=tpm0
  # Secure Boot
  -global driver=cfi.pflash01,property=secure,value=on
  -global ICH9-LPC.disable_s3=1
  # Disk
  -drive "file=$DISK,format=qcow2,if=virtio"
  # Network
  -device virtio-net-pci,netdev=net0
  -netdev "user,id=net0,hostfwd=tcp::${SSH_PORT}-:22"
  # Display
  -vga qxl
  -display gtk
)

case "$MODE" in
  install)
    WIN_ISO="${WIN11_ISO:-}"
    VIRTIO_ISO="${VIRTIO_WIN_ISO:-}"

    if [ -z "$WIN_ISO" ]; then
      echo "Set WIN11_ISO env var to your Windows 11 evaluation ISO path"
      echo "Download: https://www.microsoft.com/en-us/evalcenter/evaluate-windows-11-enterprise"
      exit 1
    fi

    QEMU_ARGS+=(-drive "file=$WIN_ISO,media=cdrom,index=0")
    [ -n "$VIRTIO_ISO" ] && QEMU_ARGS+=(-drive "file=$VIRTIO_ISO,media=cdrom,index=1")
    QEMU_ARGS+=(-boot d)

    echo "=== Win11 Install Mode ==="
    echo "During install, if no drives are found:"
    echo "  Click 'Load driver' → Browse CD → viostor/w11/amd64/"
    echo ""
    echo "To bypass TPM/Secure Boot check (simpler testing):"
    echo "  Press Shift+F10 at install screen, then:"
    echo "  reg add HKLM\\SYSTEM\\Setup\\LabConfig /v BypassTPMCheck /t REG_DWORD /d 1"
    echo "  reg add HKLM\\SYSTEM\\Setup\\LabConfig /v BypassSecureBootCheck /t REG_DWORD /d 1"
    ;;

  boot)
    echo "=== Win11 Boot Mode ==="
    ;;

  dual-boot)
    # Add NixOS ISO as second CD for dual-boot testing
    NIXOS_ISO=$(find /var/cache/qemu/symthaea-test -name "nixos-*.iso" -type f 2>/dev/null | head -1)
    if [ -n "$NIXOS_ISO" ]; then
      QEMU_ARGS+=(-drive "file=$NIXOS_ISO,media=cdrom,index=0")
      echo "=== Dual-Boot Test Mode ==="
      echo "NixOS ISO: $NIXOS_ISO"
      echo "Press F12 at boot to select boot device"
    else
      echo "No NixOS ISO found in /var/cache/qemu/symthaea-test/"
      echo "Run ./scripts/test-vm-dual-nvme.sh first to download it"
      exit 1
    fi
    ;;
esac

echo "Disk: $DISK"
echo "TPM: $TPM_DIR"
echo ""
echo "Starting VM..."

qemu-system-x86_64 "${QEMU_ARGS[@]}"
