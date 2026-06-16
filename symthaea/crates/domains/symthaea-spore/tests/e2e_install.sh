#!/usr/bin/env bash
# E2E test: NixForHumanity install flow via QEMU
#
# Prerequisites:
#   - QEMU installed (qemu-system-x86_64)
#   - NixOS ISO (auto-downloads if not present)
#   - ssh-relay binary built: cargo build -p symthaea-spore --bin ssh-relay --features server --release
#   - websocat installed (for WebSocket testing)
#
# What this tests:
#   1. Boot NixOS ISO in QEMU with serial console
#   2. Start ssh-relay on host
#   3. Connect to relay via WebSocket
#   4. Authenticate
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
#   25.05           - NixOS 25.05 stable minimal
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
        25.05)
            NIXOS_ISO="/tmp/nixos-25.05-minimal.iso"
            ISO_URL="https://channels.nixos.org/nixos-25.05/latest-nixos-minimal-x86_64-linux.iso"
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
    -net nic -net user,hostfwd=tcp::2222-:22 \
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

# ── Step 3: Start relay ──
echo "Starting ssh-relay..."
$RELAY_BIN --port $RELAY_PORT --bind 127.0.0.1 &> /tmp/e2e-relay.log &
RELAY_PID=$!
sleep 2

# Extract token from relay log
RELAY_TOKEN=$(grep -oP 'Token: \K\S+' /tmp/e2e-relay.log || echo "")
if [[ -z "$RELAY_TOKEN" ]]; then
    echo "ERROR: Could not extract relay token from log"
    cat /tmp/e2e-relay.log
    exit 1
fi
echo "  Relay token: ${RELAY_TOKEN:0:8}..."

# ── Step 4: Test authentication ──
echo "Testing relay authentication..."
AUTH_RESP=$(echo "{\"action\":\"auth\",\"token\":\"$RELAY_TOKEN\",\"ssh_host\":\"127.0.0.1\",\"ssh_port\":2222,\"ssh_user\":\"root\",\"ssh_password\":\"\"}" | \
    timeout 10 websocat -n1 "ws://127.0.0.1:$RELAY_PORT" 2>/dev/null || echo "TIMEOUT")
assert "Relay accepts valid token" echo "$AUTH_RESP" | grep -q '"ok"'

# ── Step 5: Test hardware probe ──
echo "Testing hardware probe..."
PROBE_RESP=$(echo "{\"action\":\"probe_hardware\"}" | \
    timeout 15 websocat -n1 "ws://127.0.0.1:$RELAY_PORT" 2>/dev/null || echo "TIMEOUT")
assert "Hardware probe returns disks" echo "$PROBE_RESP" | grep -q "disk\|vd\|sd"

# ── Step 6: Test install (dry run) ──
echo "Testing install command (single layout on /dev/vda)..."
# NOTE: This is a real install on the virtual disk — it will partition and format it.
# Only safe because it targets the ephemeral QEMU disk.
INSTALL_CMD="{\"action\":\"install\",\"layout\":\"single\",\"disk\":\"/dev/vda\",\"hostname\":\"e2e-test\",\"timezone\":\"UTC\",\"keyboard\":\"us\",\"desktop\":\"none\",\"gpu_driver\":\"auto\"}"
# Stream install output for up to 10 minutes
echo "$INSTALL_CMD" | timeout 600 websocat "ws://127.0.0.1:$RELAY_PORT" 2>/dev/null | tee /tmp/e2e-install.log &
INSTALL_PID=$!

# Wait for completion
echo "  Waiting for install to complete (up to 10 min)..."
COMPLETE=false
for i in $(seq 1 120); do
    sleep 5
    if grep -q "COMPLETE" /tmp/e2e-install.log 2>/dev/null; then
        COMPLETE=true
        break
    fi
    if ! kill -0 $INSTALL_PID 2>/dev/null; then
        break
    fi
done
kill $INSTALL_PID 2>/dev/null || true

assert "Install completed successfully" [[ "$COMPLETE" == "true" ]]

# ── Step 7: Verify config exists ──
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
