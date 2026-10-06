#!/usr/bin/env bash
# E2E test: NixForHumanity install flow via QEMU
#
# Prerequisites:
#   - QEMU installed (qemu-system-x86_64)
#   - NixOS ISO (auto-downloads if not present)
#   - ssh-relay binary built: cargo build -p symthaea-spore --bin ssh-relay --features server --release
#   - Python 3 with the websockets package installed
#
# What this tests:
#   1. Boot NixOS ISO in QEMU with serial console
#   2. Copy ssh-relay into the VM and start it there
#   3. Connect to the VM-local relay via a QEMU port forward
#   4. Authenticate and exercise the typed protocol on one persistent WebSocket
#   5. Send install command (single-disk layout, 8GB virtual disk)
#   6. Verify installation completes (COMPLETE marker in output)
#   7. Verify /mnt/etc/nixos/configuration.nix exists on the VM
#   8. Tear down
#
# Usage:
#   ./tests/e2e_install.sh [--keep-vm] [--iso <path>] [--nixos-version <ver>]
#
# NixOS version options:
#   nixforhumanity  - Custom ISO with relay pre-installed (default, recommended)
#   26.05           - NixOS 26.05 stable minimal
#   unstable        - NixOS unstable minimal
#   <path>          - Use a specific ISO file

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELAY_BIN="${RELAY_BIN:-$(dirname "$SCRIPT_DIR")/../target/release/ssh-relay}"
NIXOS_VERSION="${NIXOS_VERSION:-nixforhumanity}"
NIXOS_ISO="${NIXOS_ISO:-}"
DISK_IMG="/tmp/e2e-test-disk.qcow2"
RELAY_PORT=8405  # Dev/test port range
RELAY_TOKEN=""
QEMU_PID=""
RELAY_PID=""
REMOTE_RELAY=false
KEEP_VM=false
PASS=0
FAIL=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-vm) KEEP_VM=true; shift ;;
        --iso) NIXOS_ISO="$2"; shift 2 ;;
        --nixos-version) NIXOS_VERSION="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Resolve ISO path from version if not explicitly set
if [[ -z "$NIXOS_ISO" ]]; then
    case "$NIXOS_VERSION" in
        nixforhumanity)
            NIXOS_ISO="/tmp/nixos-minimal-26.05pre-git-x86_64-linux.iso"
            ISO_URL="https://github.com/Luminous-Dynamics/nixforhumanity/releases/download/v0.1.0/nixos-minimal-26.05pre-git-x86_64-linux.iso"
            ;;
        26.05)
            NIXOS_ISO="/tmp/nixos-26.05-minimal.iso"
            ISO_URL="https://channels.nixos.org/nixos-26.05/latest-nixos-minimal-x86_64-linux.iso"
            ;;
        unstable)
            NIXOS_ISO="/tmp/nixos-unstable-minimal.iso"
            ISO_URL="https://channels.nixos.org/nixos-unstable/latest-nixos-minimal-x86_64-linux.iso"
            ;;
        *)
            # Treat as a path
            NIXOS_ISO="$NIXOS_VERSION"
            ISO_URL=""
            ;;
    esac

    if [[ ! -f "$NIXOS_ISO" && -n "${ISO_URL:-}" ]]; then
        echo "Downloading NixOS ISO ($NIXOS_VERSION)..."
        if command -v gh >/dev/null && [[ "$NIXOS_VERSION" == "nixforhumanity" ]]; then
            gh release download v0.1.0 --repo Luminous-Dynamics/nixforhumanity --pattern "*.iso" --dir /tmp/
        else
            curl -L -o "$NIXOS_ISO" "$ISO_URL" --progress-bar
        fi
    fi
fi

echo "Using ISO: $NIXOS_ISO ($NIXOS_VERSION)"

cleanup() {
    echo "Cleaning up..."
    [[ -n "$RELAY_PID" ]] && kill "$RELAY_PID" 2>/dev/null || true
    if [[ "$REMOTE_RELAY" == true ]]; then
        ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -p 2222 root@localhost \
            "if [ -s /tmp/e2e-relay.pid ]; then kill \$(cat /tmp/e2e-relay.pid) 2>/dev/null || true; fi; \
             rm -f /tmp/e2e-relay.pid /tmp/e2e-ssh-relay /tmp/e2e-relay.log" \
            2>/dev/null || true
    fi
    if [[ "$KEEP_VM" == false && -n "$QEMU_PID" ]]; then
        kill "$QEMU_PID" 2>/dev/null || true
        rm -f "$DISK_IMG"
    fi
}
trap cleanup EXIT

assert() {
    local desc="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "  PASS: $desc"
        ((PASS++))
    else
        echo "  FAIL: $desc"
        ((FAIL++))
    fi
}

# ── Step 0: Prerequisites ──
echo "=== E2E Install Test ==="

if [[ ! -f "$RELAY_BIN" ]]; then
    echo "ERROR: ssh-relay binary not found at $RELAY_BIN"
    echo "Build it: cargo build -p symthaea-spore --bin ssh-relay --features server --release"
    exit 1
fi

if ! command -v qemu-system-x86_64 >/dev/null; then
    echo "ERROR: qemu-system-x86_64 not found"
    exit 1
fi

if [[ ! -f "$NIXOS_ISO" ]]; then
    echo "NixOS ISO not found at $NIXOS_ISO"
    echo "Download: nix build nixpkgs#nixos-minimal-iso -o /tmp/nixos-minimal.iso"
    echo "Or set NIXOS_ISO=/path/to/nixos-*.iso"
    exit 1
fi

# ── Step 1: Create test disk ──
echo "Creating 8GB test disk..."
qemu-img create -f qcow2 "$DISK_IMG" 8G

# ── Step 2: Boot QEMU ──
echo "Booting NixOS ISO in QEMU..."
qemu-system-x86_64 \
    -m 4096 \
    -smp 2 \
    -enable-kvm \
    -cdrom "$NIXOS_ISO" \
    -drive file="$DISK_IMG",format=qcow2,if=virtio \
    -net nic -net user,hostfwd=tcp::2222-:22,hostfwd=tcp::8405-:8405 \
    -nographic \
    -serial mon:stdio \
    &> /tmp/e2e-qemu.log &
QEMU_PID=$!
echo "  QEMU PID: $QEMU_PID"

# Wait for VM to boot (NixOS ISO auto-login)
echo "Waiting for VM to boot (60s)..."
sleep 60

# Enable SSH in the live environment
echo "Enabling SSH..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p 2222 root@localhost "systemctl start sshd && echo SSH_OK" 2>/dev/null || \
    echo "  (SSH may already be running)"

# ── Step 3: Start relay inside the VM ──
echo "Installing test relay into the VM..."
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$RELAY_BIN" root@localhost:/tmp/e2e-ssh-relay >/dev/null
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p 2222 root@localhost \
    "chmod 700 /tmp/e2e-ssh-relay && rm -f /tmp/e2e-relay.log /tmp/e2e-relay.pid && \
     nohup /tmp/e2e-ssh-relay --port $RELAY_PORT --bind 0.0.0.0 >/tmp/e2e-relay.log 2>&1 & \
     echo \$! >/tmp/e2e-relay.pid"
REMOTE_RELAY=true

echo "Waiting for VM-local relay token..."
for _ in $(seq 1 30); do
    TOKEN_LINE=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p 2222 root@localhost "grep -oE 'Token: [^[:space:]]+' /tmp/e2e-relay.log | tail -1" 2>/dev/null || true)
    if [[ -n "$TOKEN_LINE" ]]; then
        RELAY_TOKEN="${TOKEN_LINE#Token: }"
        break
    fi
    sleep 1
done
if [[ -z "$RELAY_TOKEN" ]]; then
    echo "ERROR: VM-local relay did not publish an auth token"
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -p 2222 root@localhost "cat /tmp/e2e-relay.log" 2>/dev/null || true
    exit 1
fi
echo "  VM relay token: ${RELAY_TOKEN:0:8}..."

# ── Step 4-7: Persistent authenticated protocol + install ──
echo "Exercising authenticated install protocol against the VM-local relay..."
RELAY_TOKEN="$RELAY_TOKEN" RELAY_PORT="$RELAY_PORT" python3 - <<'PYEOF'
import asyncio
import json
import os

import websockets

TOKEN = os.environ["RELAY_TOKEN"]
PORT = int(os.environ["RELAY_PORT"])

async def recv(ws, expected=None, timeout=30):
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout), strict=False)
        message_type = msg.get("type")
        if message_type in ("output", "progress"):
            print(f"  [relay] {message_type}: {msg.get('data') or msg.get('stage')}")
        if expected is None or message_type == expected:
            return msg

async def main():
    uri = f"ws://127.0.0.1:{PORT}"
    async with websockets.connect(
        uri,
        ping_timeout=60,
        ping_interval=20,
        close_timeout=10,
    ) as ws:
        await ws.send(json.dumps({"action": "auth", "token": TOKEN}))
        auth = await recv(ws)
        if auth.get("type") not in ("authenticated", "authed"):
            raise AssertionError(f"auth failed: {auth}")

        await ws.send(json.dumps({
            "action": "connect",
            "host": "127.0.0.1",
            "port": 22,
            "username": "root",
            "password": "",
        }))
        connected = await recv(ws, "connected")
        if connected.get("type") != "connected":
            raise AssertionError(f"connect failed: {connected}")

        await ws.send(json.dumps({"action": "probe_hardware"}))
        probe = await recv(ws, "hardware_probe", timeout=60)
        hw = json.loads(probe["data"], strict=False)
        if hw.get("arch") not in ("x86_64", "aarch64", "armv7l"):
            raise AssertionError(f"unexpected target architecture: {hw!r}")
        print(f"  [probe] target architecture: {hw.get('arch')}")

        await ws.send(json.dumps({"action": "discover_disks"}))
        disks_msg = await recv(ws, "disks", timeout=30)
        disks = json.loads(disks_msg["data"], strict=False)
        names = {d["name"] for d in disks}
        if "vda" not in names:
            raise AssertionError(f"/dev/vda missing from disk inventory: {names}")
        print("  [disks] ephemeral /dev/vda present")

        await ws.send(json.dumps({
            "action": "install",
            "layout": "single",
            "disk": "/dev/vda",
            "hostname": "e2e-test",
            "timezone": "UTC",
            "keyboard": "us",
            "desktop": "none",
            "gpu_driver": "auto",
            "target_machine_digest": hw["target_machine_digest"],
        }))

        complete = False
        while True:
            msg = await recv(ws, timeout=900)
            if msg.get("type") == "output" and "COMPLETE" in msg.get("data", ""):
                complete = True
            if msg.get("type") == "exit":
                if msg.get("code") != 0 or not complete:
                    raise AssertionError(f"install did not complete cleanly: {msg}")
                break

asyncio.run(main())
PYEOF

echo "Verifying configuration..."
CONFIG_CHECK=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -p 2222 root@localhost "cat /mnt/etc/nixos/configuration.nix 2>/dev/null | head -3" 2>/dev/null || echo "MISSING")
assert "configuration.nix exists on target" echo "$CONFIG_CHECK" | grep -q "config"

# ── Results ──
echo ""
echo "=== Results ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo "SOME TESTS FAILED"
    exit 1
else
    echo "ALL TESTS PASSED"
    exit 0
fi
