#!/usr/bin/env bash
# Test the symthaea-quicken-fb boot animation.
#
# Modes:
#   1. Binary verification (default, always runs)
#   2. Direct DRM test   — requires a free TTY and /dev/dri/card*
#   3. Full NixOS VM     — requires `nix build .#vm-test` (see nix/vm-test.nix)
#
# Usage:
#   ./scripts/test-quicken-vm.sh                          # verify binary
#   ./scripts/test-quicken-vm.sh --run                    # run on local DRM
#   ./scripts/test-quicken-vm.sh --genesis "my phrase"    # custom seed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$REPO_ROOT/target/debug/quicken-fb"

GENESIS="consciousness awakens"
RUN_DIRECT=false

# ── Argument parsing ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)       RUN_DIRECT=true; shift ;;
    --genesis)   GENESIS="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--run] [--genesis <phrase>]"
      echo ""
      echo "  --run        Run on local DRM device (needs root + free TTY)"
      echo "  --genesis    Genesis phrase for pattern seeding"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ── Build if needed ────────────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
  echo "Building symthaea-quicken-fb..."
  (cd "$REPO_ROOT" && cargo build -p symthaea-quicken-fb)
fi

# ── Binary verification ───────────────────────────────────────────────
echo "=== Binary verification ==="
"$BINARY" --help
echo ""

echo "=== Binary info ==="
ls -lh "$BINARY"
file "$BINARY"
echo ""

# ── DRM device check ──────────────────────────────────────────────────
echo "=== DRM device check ==="
if [ -d /dev/dri ]; then
  ls -la /dev/dri/
  echo ""
  # Show which cards support KMS modesetting
  for card in /dev/dri/card*; do
    if [ -c "$card" ]; then
      echo "  $card: $(stat -c '%G' "$card") group, $(stat -c '%a' "$card") perms"
    fi
  done
else
  echo "No /dev/dri/ directory — headless server or missing kernel modules."
fi
echo ""

# ── Direct run ─────────────────────────────────────────────────────────
if $RUN_DIRECT; then
  # Find a usable DRM card
  DEVICE=""
  for card in /dev/dri/card0 /dev/dri/card1 /dev/dri/card2; do
    if [ -c "$card" ]; then
      DEVICE="$card"
      break
    fi
  done

  if [ -z "$DEVICE" ]; then
    echo "ERROR: No DRM card found in /dev/dri/"
    exit 1
  fi

  echo "Running quicken-fb on $DEVICE with genesis: \"$GENESIS\""
  echo "Press Ctrl+C to stop."
  echo ""
  sudo "$BINARY" --genesis-phrase "$GENESIS" --device "$DEVICE"
  exit $?
fi

# ── Instructions ───────────────────────────────────────────────────────
echo "=== How to test ==="
echo ""
echo "Option 1: Run directly on this machine (needs root + DRM):"
echo "  $0 --run --genesis '$GENESIS'"
echo ""
echo "Option 2: Full NixOS VM (recommended for isolated testing):"
echo "  cd $REPO_ROOT"
echo "  nix build .#vm-test"
echo "  QEMU_OPTS='-device virtio-gpu-pci -display gtk' ./result/bin/run-*-vm"
echo ""
echo "Option 3: Manual QEMU with pre-built binary:"
echo "  qemu-system-x86_64 \\"
echo "    -enable-kvm -m 1G \\"
echo "    -device virtio-gpu-pci \\"
echo "    -display gtk \\"
echo "    -kernel /boot/vmlinuz-* \\"
echo "    -append 'console=tty0' \\"
echo "    -virtfs local,path=$REPO_ROOT/target,mount_tag=host,security_model=none"
echo ""
