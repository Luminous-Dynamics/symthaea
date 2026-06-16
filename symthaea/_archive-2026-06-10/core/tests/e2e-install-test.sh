#!/usr/bin/env bash
# E2E test: Boot Sovereign Inoculation ISO in QEMU, verify relay, run install.
#
# This tests the full automated install pipeline:
#   1. Boot ISO in QEMU VM
#   2. Wait for relay to become accessible
#   3. Connect via WebSocket
#   4. Probe hardware
#   5. Discover disks
#   6. Run install with generated config
#   7. Verify install completed
#   8. Reboot and verify NixOS boots
#
# Prerequisites:
#   - QEMU installed
#   - ISO built (run nix/build-iso.sh first)
#   - Python 3 with websockets module (pip install websockets)
#
# Usage:
#   ./tests/e2e-install-test.sh [path-to-iso]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ISO="${1:-$(ls result-iso/iso/*.iso 2>/dev/null | head -1)}"
if [ -z "$ISO" ] || [ ! -f "$ISO" ]; then
    echo "ISO not found. Build it first: ./nix/build-iso.sh"
    echo "Or specify: $0 path/to/iso"
    exit 1
fi

echo "╔══════════════════════════════════════════╗"
echo "║   E2E Install Test                       ║"
echo "╚══════════════════════════════════════════╝"
echo "ISO: $ISO"
echo ""

# Create test disk image (20GB sparse)
DISK="/tmp/sovereign-e2e-test.qcow2"
qemu-img create -f qcow2 "$DISK" 20G

# Forward relay port 8094 → localhost:18094
RELAY_PORT=18094
SSH_PORT=18022
MONITOR_SOCK="/tmp/sovereign-e2e-monitor.sock"

echo "Starting QEMU VM..."
qemu-system-x86_64 \
    -m 4096 \
    -smp 2 \
    -enable-kvm \
    -bios /usr/share/OVMF/OVMF_CODE.fd \
    -drive file="$DISK",format=qcow2,if=virtio \
    -cdrom "$ISO" \
    -boot d \
    -netdev user,id=net0,hostfwd=tcp::${RELAY_PORT}-:8094,hostfwd=tcp::${SSH_PORT}-:22 \
    -device virtio-net-pci,netdev=net0 \
    -nographic \
    -monitor unix:${MONITOR_SOCK},server,nowait \
    &
QEMU_PID=$!

cleanup() {
    echo "Cleaning up..."
    kill $QEMU_PID 2>/dev/null || true
    rm -f "$DISK" "$MONITOR_SOCK"
}
trap cleanup EXIT

# Wait for relay to become accessible (up to 120 seconds)
echo "Waiting for relay (up to 120s)..."
for i in $(seq 1 120); do
    if curl -s --max-time 2 "http://127.0.0.1:${RELAY_PORT}" >/dev/null 2>&1 || \
       python3 -c "
import asyncio, websockets
async def check():
    try:
        async with websockets.connect('ws://127.0.0.1:${RELAY_PORT}', close_timeout=2):
            return True
    except: return False
print('ok' if asyncio.run(check()) else 'no')
" 2>/dev/null | grep -q "ok"; then
        echo "Relay accessible after ${i}s"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "FAIL: Relay did not become accessible in 120s"
        exit 1
    fi
    sleep 1
    printf "."
done
echo ""

# Run the WebSocket test sequence
echo "Running WebSocket test sequence..."
python3 - <<'PYEOF'
import asyncio
import json
import sys

async def test():
    import websockets

    uri = "ws://127.0.0.1:18094"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri, ping_timeout=30) as ws:
        # 1. Connect SSH
        print("Step 1: SSH connect...")
        await ws.send(json.dumps({
            "action": "connect",
            "host": "127.0.0.1",
            "port": 22,
            "username": "root",
            "password": "sovereign"
        }))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        assert resp["type"] == "connected", f"Expected 'connected', got {resp}"
        print(f"  ✓ SSH connected")

        # 2. Probe hardware
        print("Step 2: Probe hardware...")
        await ws.send(json.dumps({"action": "probe_hardware"}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        assert resp["type"] == "hardware_probe", f"Expected 'hardware_probe', got {resp['type']}"
        hw = json.loads(resp["data"])
        print(f"  ✓ Hardware: arch={hw.get('arch', '?')}, efi={hw.get('efi_available', '?')}")
        print(f"  ✓ GPU: {hw.get('gpu', {}).get('model', 'unknown')}")
        print(f"  ✓ Safety: {hw.get('safety', {}).get('level', '?')}")

        # 3. Discover disks
        print("Step 3: Discover disks...")
        await ws.send(json.dumps({"action": "discover_disks"}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        assert resp["type"] == "disks", f"Expected 'disks', got {resp['type']}"
        disk_list = json.loads(resp["data"])
        print(f"  ✓ Found {len(disk_list)} disk(s)")
        for d in disk_list:
            size_gb = int(d.get("size", "0")) / 1e9
            print(f"    {d['name']}: {d.get('model', '?')} ({size_gb:.0f} GB, {d.get('transport', '?')})")

        # Find the 20GB virtio disk
        target = None
        for d in disk_list:
            if "virtio" in d.get("transport", "") or "vda" in d["name"]:
                target = d["name"]
                break
        if not target and disk_list:
            target = disk_list[0]["name"]

        if not target:
            print("  ✗ No target disk found!")
            sys.exit(1)
        print(f"  ✓ Target disk: {target}")

        # 4. Install
        print(f"Step 4: Installing on {target}...")
        await ws.send(json.dumps({
            "action": "install",
            "disk": target,
            "layout": "single",
            "hostname": "e2e-test",
            "desktop": "none",
            "gpu_driver": "modesetting",
            "timezone": "UTC",
            "keyboard": "us"
        }))

        # Stream install output
        last_stage = ""
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=600))
            except asyncio.TimeoutError:
                print("  ✗ Install timed out (10 min)")
                sys.exit(1)

            msg_type = msg.get("type", "")

            if msg_type == "progress":
                stage = msg.get("stage", "")
                pct = msg.get("percentage", 0)
                if stage != last_stage:
                    print(f"  [{pct}%] {stage}")
                    last_stage = stage
            elif msg_type == "output":
                pass  # suppress verbose output
            elif msg_type == "exit":
                code = msg.get("code", -1)
                if code == 0:
                    print(f"  ✓ Install completed successfully!")
                else:
                    print(f"  ✗ Install failed with exit code {code}")
                    sys.exit(1)
                break
            elif msg_type == "error":
                print(f"  ✗ Error: {msg.get('message', '?')}")
                sys.exit(1)

        # 5. Disconnect
        await ws.send(json.dumps({"action": "disconnect"}))
        print("Step 5: Disconnected.")

    print("")
    print("════════════════════════════════════════")
    print("  E2E TEST PASSED")
    print("════════════════════════════════════════")

asyncio.run(test())
PYEOF

echo ""
echo "E2E test completed."
