#!/usr/bin/env bash
# automated-demo.sh — Fully automated E2E demo of Sovereign Inoculation
#
# This script:
#   1. Resets QEMU VM disk images (fresh state)
#   2. Boots the VM headless
#   3. Waits for SSH readiness
#   4. Starts eval-api + ssh-relay (if not already running)
#   5. Drives the full install via WebSocket to ssh-relay
#   6. Logs all output to a timestamped file
#   7. Optionally records screen (if --record flag passed)
#
# Usage:
#   ./scripts/automated-demo.sh              # Run headless demo
#   ./scripts/automated-demo.sh --record     # Run with screen recording
#   ./scripts/automated-demo.sh --visible    # Show QEMU window
#
# Requirements: qemu, python3, websocat (or python websocket-client)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VM_DIR="/var/cache/qemu/symthaea-test"
SSH_PORT=2222
RELAY_PORT="${RELAY_PORT:-8094}"
LOG_DIR="$ROOT_DIR/demo-logs"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$LOG_DIR/demo-$TIMESTAMP.log"
RECORD=false
VISIBLE=false

for arg in "$@"; do
  case "$arg" in
    --record) RECORD=true ;;
    --visible) VISIBLE=true ;;
  esac
done

mkdir -p "$LOG_DIR"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }

cleanup() {
  log "Cleaning up..."
  kill "$VM_PID" 2>/dev/null || true
  kill "$RELAY_PID" 2>/dev/null || true
  kill "$EVAL_PID" 2>/dev/null || true
  kill "$RECORD_PID" 2>/dev/null || true
  log "Demo log: $LOG_FILE"
}
trap cleanup EXIT

# ── Step 1: Fresh disk images ──
log "=== Sovereign Inoculation Automated Demo ==="
log "Step 1: Creating fresh disk images..."
rm -f "$VM_DIR/nvme-fast.qcow2" "$VM_DIR/nvme-standard.qcow2"
qemu-img create -f qcow2 "$VM_DIR/nvme-fast.qcow2" 256G 2>&1 | tee -a "$LOG_FILE"
qemu-img create -f qcow2 "$VM_DIR/nvme-standard.qcow2" 256G 2>&1 | tee -a "$LOG_FILE"

# ── Step 2: Find ISO and OVMF ──
ISO=$(find "$VM_DIR" -name "nixos-*.iso" -type f 2>/dev/null | head -1)
if [ -z "$ISO" ]; then
  log "ERROR: No NixOS ISO found in $VM_DIR"
  exit 1
fi

OVMF=$(find /nix/store -name "OVMF.fd" -path "*/FV/*" 2>/dev/null | head -1)
if [ -z "$OVMF" ]; then
  for path in /run/current-system/sw/share/qemu/OVMF.fd /run/current-system/sw/share/OVMF/OVMF.fd; do
    [ -f "$path" ] && OVMF="$path" && break
  done
fi
[ -z "$OVMF" ] && log "ERROR: OVMF not found" && exit 1

log "  ISO: $ISO"
log "  OVMF: $OVMF"

# ── Step 3: Boot VM ──
log "Step 2: Booting QEMU VM..."
DISPLAY_OPT="-display none -daemonize"
$VISIBLE && DISPLAY_OPT="-display gtk"

qemu-system-x86_64 \
  -enable-kvm \
  -m 4096 \
  -smp 4 \
  -bios "$OVMF" \
  -device nvme,id=nvme0,serial=SAMSUNG-FAST \
  -drive file="$VM_DIR/nvme-fast.qcow2",if=none,id=drv0,format=qcow2 \
  -device nvme-ns,drive=drv0,bus=nvme0 \
  -device nvme,id=nvme1,serial=INTEL-STD \
  -drive file="$VM_DIR/nvme-standard.qcow2",if=none,id=drv1,format=qcow2 \
  -device nvme-ns,drive=drv1,bus=nvme1 \
  -cdrom "$ISO" \
  -boot d \
  -net nic,model=virtio \
  -net user,hostfwd=tcp::${SSH_PORT}-:22 \
  -serial mon:stdio \
  $DISPLAY_OPT \
  -pidfile /tmp/symthaea-demo-vm.pid \
  2>&1 | tee -a "$LOG_FILE" &

VM_PID=$!
log "  VM PID: $VM_PID"

# ── Step 4: Wait for SSH ──
log "Step 3: Waiting for SSH (this takes 30-90s for NixOS boot)..."
MAX_WAIT=120
WAITED=0
while ! ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
          -p $SSH_PORT root@localhost echo "SSH ready" 2>/dev/null; do
  sleep 5
  WAITED=$((WAITED + 5))
  if [ $WAITED -ge $MAX_WAIT ]; then
    log "ERROR: SSH not available after ${MAX_WAIT}s"
    exit 1
  fi
  printf "." | tee -a "$LOG_FILE"
done
log ""
log "  SSH ready after ${WAITED}s"

# ── Step 5: Verify VM disk layout ──
log "Step 4: Probing hardware..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT root@localhost \
  "lsblk -o NAME,SIZE,TYPE,FSTYPE; echo '---'; uname -a" 2>/dev/null | tee -a "$LOG_FILE"

# ── Step 6: Start servers (if not already running) ──
RELAY_PID=""
EVAL_PID=""

if ! curl -s http://localhost:8090/health >/dev/null 2>&1; then
  log "Step 5a: Starting eval-api..."
  eval-api >> "$LOG_FILE" 2>&1 &
  EVAL_PID=$!
  sleep 2
fi

if ! curl -s http://localhost:$RELAY_PORT >/dev/null 2>&1; then
  log "Step 5b: Starting ssh-relay..."
  ssh-relay --port $RELAY_PORT >> "$LOG_FILE" 2>&1 &
  RELAY_PID=$!
  sleep 2
fi

# ── Step 7: Optional screen recording ──
RECORD_PID=""
if $RECORD; then
  RECORDING="$HOME/Videos/sovereign-inoculation-demo-$TIMESTAMP.mp4"
  log "Step 6: Starting screen recording → $RECORDING"
  if command -v wf-recorder &>/dev/null; then
    wf-recorder -f "$RECORDING" -c libx264 -p crf=23 &
  else
    nix-shell -p wf-recorder --run "wf-recorder -f '$RECORDING' -c libx264 -p crf=23" &
  fi
  RECORD_PID=$!
  sleep 2
fi

# ── Step 8: Drive install via WebSocket ──
log "Step 7: Driving install via WebSocket..."

# Use python3 to send WebSocket messages (available on NixOS)
python3 << 'PYEOF' 2>&1 | tee -a "$LOG_FILE"
import asyncio
import json
import os
import sys

async def run_demo():
    try:
        import websockets
    except ImportError:
        # Fall back to raw socket if websockets not installed
        print("ERROR: python3 websockets module not available")
        print("Install with: nix-shell -p python3Packages.websockets")
        sys.exit(1)

    uri = f"ws://localhost:{os.environ.get('RELAY_PORT', '8094')}"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        # Step 1: Connect to VM via SSH
        print(">> Sending connect...")
        await ws.send(json.dumps({
            "action": "connect",
            "host": "localhost",
            "port": 2222,
            "username": "root",
            "password": "test"
        }))

        # Wait for connected response
        resp = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(resp)
        print(f"<< {msg.get('type', 'unknown')}: {msg.get('message', msg.get('data', ''))}")

        if msg.get("type") != "connected":
            print(f"ERROR: Expected 'connected', got '{msg.get('type')}'")
            return

        # Step 2: Discover disks
        print(">> Sending discover_disks...")
        await ws.send(json.dumps({
            "action": "discover_disks",
            "host": "localhost",
            "port": 2222,
            "username": "root",
            "password": "test"
        }))

        resp = await asyncio.wait_for(ws.recv(), timeout=30)
        msg = json.loads(resp)
        print(f"<< {msg.get('type', 'unknown')}: disk data received")
        if msg.get("type") == "disks":
            disks = json.loads(msg.get("data", "{}"))
            for dev in disks.get("blockdevices", []):
                if dev.get("type") == "disk" and "nvme" in dev.get("name", ""):
                    print(f"   Found: /dev/{dev['name']} ({dev.get('size', '?')})")

        # Step 3: Run single-disk install on nvme0n1
        print(">> Sending install (single layout on /dev/nvme0n1)...")
        await ws.send(json.dumps({
            "action": "install",
            "host": "localhost",
            "port": 2222,
            "username": "root",
            "password": "test",
            "disk": "/dev/nvme0n1",
            "layout": "single",
            "hostname": "symthaea-demo",
            "flake_nix": "",
            "disko_nix": "",
            "hardware_nix": ""
        }))

        # Stream progress
        complete = False
        while not complete:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=600)
                msg = json.loads(resp)
                msg_type = msg.get("type", "")

                if msg_type == "progress":
                    stage = msg.get("stage", "")
                    pct = msg.get("percentage", 0)
                    phase = msg.get("phase", "")
                    print(f"   [{pct:3d}%] {stage} ({phase})")

                elif msg_type == "output":
                    data = msg.get("data", "").strip()
                    if data:
                        # Only print significant lines
                        if data.startswith("STAGE:") or data.startswith("===") or \
                           "installed" in data or "OK" in data or "COMPLETE" in data or \
                           "Error" in data.lower():
                            print(f"   >> {data}")

                elif msg_type == "error":
                    print(f"   ERROR: {msg.get('message', msg.get('data', ''))}")
                    complete = True

                elif msg_type == "exit":
                    code = msg.get("code", -1)
                    print(f"   Exit code: {code}")
                    complete = True

                if "COMPLETE" in msg.get("data", ""):
                    complete = True

            except asyncio.TimeoutError:
                print("   TIMEOUT: No response for 10 minutes")
                complete = True

        print("\n=== Demo Complete ===")

asyncio.run(run_demo())
PYEOF

log "=== Automated demo finished ==="
log "Log saved to: $LOG_FILE"

if $RECORD && [ -n "$RECORD_PID" ]; then
  kill "$RECORD_PID" 2>/dev/null
  log "Recording saved to: $RECORDING"
fi
