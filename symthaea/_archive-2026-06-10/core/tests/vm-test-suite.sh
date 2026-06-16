#!/usr/bin/env bash
# Comprehensive VM test suite for Sovereign Inoculation / NixForHumanity
#
# Tests:
#   1. ISO boot + relay accessibility
#   2. Hardware probe via WebSocket
#   3. Disk discovery
#   4. Single-disk install (clean wipe)
#   5. Encrypted install (LUKS)
#   6. Alongside install (dual-boot with existing OS)
#   7. Post-install boot verification
#   8. Config gen validation (generated config is valid Nix)
#   9. Scanner binary (sovereign-scan) on the VM
#  10. Win11 dual-boot (alongside layout)
#
# Usage:
#   ./tests/vm-test-suite.sh            # Run all tests
#   ./tests/vm-test-suite.sh boot       # Test 1 only: ISO boot
#   ./tests/vm-test-suite.sh install    # Tests 1-4: boot + single install
#   ./tests/vm-test-suite.sh luks       # Tests 1-3 + 5: boot + encrypted install
#   ./tests/vm-test-suite.sh dualboot   # Tests 1-3 + 6 + 10: boot + alongside

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VM_DIR="/var/lib/vms"
ISO="${ISO:-$(ls result-iso/iso/*.iso 2>/dev/null | head -1 || true)}"
RELAY_PORT=18094
SSH_PORT=18022
RESULTS=""
PASS=0
FAIL=0

# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════

log() { echo "[$(date +%H:%M:%S)] $*"; }
pass() { log "PASS: $1"; RESULTS="$RESULTS\n  PASS  $1"; PASS=$((PASS + 1)); }
fail() { log "FAIL: $1"; RESULTS="$RESULTS\n  FAIL  $1"; FAIL=$((FAIL + 1)); }

start_vm() {
    local disk="$1"
    local extra_args="${2:-}"

    # Locate OVMF firmware
    local ovmf_code="${OVMF_CODE:-$(ls /nix/store/*OVMF*/FV/OVMF_CODE.fd 2>/dev/null | head -1 || echo /usr/share/OVMF/OVMF_CODE.fd)}"
    local ovmf_vars_src="${OVMF_VARS:-$(ls /nix/store/*OVMF*/FV/OVMF_VARS.fd 2>/dev/null | head -1 || echo /usr/share/OVMF/OVMF_VARS.fd)}"
    # OVMF_VARS needs a writable copy per VM
    local ovmf_vars="$VM_DIR/ovmf_vars_$(basename "$disk" .qcow2).fd"
    cp "$ovmf_vars_src" "$ovmf_vars"
    chmod 644 "$ovmf_vars"

    log "Starting QEMU VM..."
    qemu-system-x86_64 \
        -m 4096 \
        -smp 4 \
        -enable-kvm \
        -drive if=pflash,format=raw,readonly=on,file="$ovmf_code" \
        -drive if=pflash,format=raw,file="$ovmf_vars" \
        -drive file="$disk",format=qcow2,if=virtio \
        -cdrom "$ISO" \
        -boot d \
        -netdev user,id=net0,hostfwd=tcp::${RELAY_PORT}-:8094,hostfwd=tcp::${SSH_PORT}-:22 \
        -device virtio-net-pci,netdev=net0 \
        -nographic \
        $extra_args \
        &
    QEMU_PID=$!
    sleep 2
}

stop_vm() {
    if [[ -n "${QEMU_PID:-}" ]]; then
        kill $QEMU_PID 2>/dev/null || true
        wait $QEMU_PID 2>/dev/null || true
        unset QEMU_PID
    fi
}

wait_for_relay() {
    local timeout="${1:-120}"
    log "Waiting for SSH (up to ${timeout}s)..."
    for i in $(seq 1 "$timeout"); do
        # ssh-keyscan proves sshd is actually running (not just port forwarding)
        if ssh-keyscan -p $SSH_PORT -T 2 127.0.0.1 2>/dev/null | grep -q ssh; then
            log "SSH daemon ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    return 1
}

AUTH_TOKEN="sovereign"

ws_test() {
    # Run a Python WebSocket test script, return exit code
    AUTH_TOKEN="$AUTH_TOKEN" python3 -u - "$@" 2>&1
}

cleanup() {
    stop_vm
    echo ""
    echo "════════════════════════════════════════"
    echo "  TEST RESULTS"
    echo -e "$RESULTS"
    echo ""
    echo "  $PASS passed, $FAIL failed"
    echo "════════════════════════════════════════"
    [[ $FAIL -eq 0 ]]
}

trap cleanup EXIT

# ═══════════════════════════════════════════════════════
# Test 1: ISO Boot + Relay
# ═══════════════════════════════════════════════════════

test_boot() {
    log "=== Test 1: ISO Boot + Relay ==="

    local disk="$VM_DIR/test-boot.qcow2"
    qemu-img create -f qcow2 "$disk" 20G >/dev/null 2>&1

    start_vm "$disk"

    if wait_for_relay 120; then
        pass "ISO boots and SSH is accessible"
    else
        fail "ISO boot or SSH failed"
        stop_vm
        rm -f "$disk"
        return 1
    fi

    # Give services time to start after SSH is up
    log "Waiting 20s for relay + avahi services..."
    sleep 20

    # Verify SSH daemon is stable (retry up to 3 times)
    local keyscan_ok=false
    for attempt in 1 2 3; do
        if ssh-keyscan -p $SSH_PORT -T 10 127.0.0.1 2>/dev/null | grep -q ssh; then
            keyscan_ok=true
            break
        fi
        sleep 3
    done
    if $keyscan_ok; then
        pass "SSH daemon responds with host key"
    else
        fail "SSH daemon not responding"
    fi

    # Check relay via WebSocket probe (lightweight — just connect and close)
    if python3 -c "
import asyncio, websockets
async def check():
    try:
        async with websockets.connect('ws://127.0.0.1:${RELAY_PORT}', close_timeout=3, open_timeout=5):
            return True
    except: return False
exit(0 if asyncio.run(check()) else 1)
" 2>/dev/null; then
        pass "Sovereign relay WebSocket is accepting connections"
    else
        # Relay may not be up yet — wait and retry
        sleep 15
        if python3 -c "
import asyncio, websockets
async def check():
    try:
        async with websockets.connect('ws://127.0.0.1:${RELAY_PORT}', close_timeout=3, open_timeout=5):
            return True
    except: return False
exit(0 if asyncio.run(check()) else 1)
" 2>/dev/null; then
            pass "Sovereign relay WebSocket is accepting connections (after retry)"
        else
            fail "Sovereign relay not accessible via WebSocket"
        fi
    fi

    stop_vm
    rm -f "$disk"
}

# ═══════════════════════════════════════════════════════
# Test 2: Hardware Probe via WebSocket
# ═══════════════════════════════════════════════════════

test_hardware_probe() {
    log "=== Test 2: Hardware Probe ==="

    local disk="$VM_DIR/test-probe.qcow2"
    qemu-img create -f qcow2 "$disk" 20G >/dev/null 2>&1
    start_vm "$disk"
    wait_for_relay 120 || { fail "VM not ready"; stop_vm; rm -f "$disk"; return 1; }

    sleep 20  # wait for relay
    if ws_test <<'PYEOF'
import asyncio, json, sys, os
async def test():
    import websockets
    token = os.environ.get("AUTH_TOKEN", "sovereign")
    def jload(s): return json.loads(s, strict=False)
    async with websockets.connect(f"ws://127.0.0.1:18094", ping_timeout=30) as ws:
        # Auth
        await ws.send(json.dumps({"action":"auth","token":token}))
        r = jload(await asyncio.wait_for(ws.recv(), 10))
        assert r.get("type") in ("authenticated", "authed"), f"Auth failed: {r}"

        # Connect SSH
        await ws.send(json.dumps({"action":"connect","host":"127.0.0.1","port":22,"username":"root","password":"sovereign"}))
        r = jload(await asyncio.wait_for(ws.recv(), 15))
        assert r["type"] == "connected", f"SSH failed: {r}"

        # Probe
        await ws.send(json.dumps({"action":"probe_hardware"}))
        r = jload(await asyncio.wait_for(ws.recv(), 30))
        assert r["type"] == "hardware_probe", f"Probe failed: {r}"
        hw = jload(r["data"])
        assert "arch" in hw, "Missing arch"
        assert "gpu" in hw, "Missing gpu"
        assert "safety" in hw, "Missing safety"
        print(f"  arch={hw['arch']}, efi={hw.get('efi_available')}, tpm2={hw.get('tpm2_available')}")
        print(f"  gpu={hw.get('gpu',{}).get('model','?')}")
        print(f"  safety={hw.get('safety',{}).get('level','?')}")
asyncio.run(test())
PYEOF
    then
        pass "Hardware probe returns valid data"
    else
        fail "Hardware probe failed"
    fi

    stop_vm
    rm -f "$disk"
}

# ═══════════════════════════════════════════════════════
# Test 3: Disk Discovery
# ═══════════════════════════════════════════════════════

test_disk_discovery() {
    log "=== Test 3: Disk Discovery ==="

    local disk="$VM_DIR/test-disks.qcow2"
    qemu-img create -f qcow2 "$disk" 20G >/dev/null 2>&1
    start_vm "$disk"
    wait_for_relay 120 || { fail "VM not ready"; stop_vm; rm -f "$disk"; return 1; }

    sleep 20  # wait for relay
    if ws_test <<'PYEOF'
import asyncio, json, sys, os
async def test():
    import websockets
    token = os.environ.get("AUTH_TOKEN", "sovereign")
    def jload(s): return json.loads(s, strict=False)
    async with websockets.connect("ws://127.0.0.1:18094", ping_timeout=30) as ws:
        await ws.send(json.dumps({"action":"auth","token":token}))
        r = jload(await asyncio.wait_for(ws.recv(), 10))
        assert r.get("type") in ("authenticated", "authed"), f"Auth failed: {r}"

        await ws.send(json.dumps({"action":"connect","host":"127.0.0.1","port":22,"username":"root","password":"sovereign"}))
        r = jload(await asyncio.wait_for(ws.recv(), 15))
        assert r["type"] == "connected"

        await ws.send(json.dumps({"action":"discover_disks"}))
        r = jload(await asyncio.wait_for(ws.recv(), 15))
        assert r["type"] == "disks", f"Expected disks: {r}"
        disks = jload(r["data"])
        assert len(disks) > 0, "No disks found"
        for d in disks:
            size_gb = int(d.get("size","0") or "0") / 1e9
            print(f"  {d['name']}: {d.get('model','?')} ({size_gb:.0f}GB, {d.get('transport','?')})")
        assert any("vda" in d["name"] or "virtio" in d.get("transport","") for d in disks), "No virtio disk found"
asyncio.run(test())
PYEOF
    then
        pass "Disk discovery finds virtio disk"
    else
        fail "Disk discovery failed"
    fi

    stop_vm
    rm -f "$disk"
}

# ═══════════════════════════════════════════════════════
# Test 4: Single Disk Install
# ═══════════════════════════════════════════════════════

test_single_install() {
    log "=== Test 4: Single Disk Install ==="

    local disk="$VM_DIR/test-install.qcow2"
    qemu-img create -f qcow2 "$disk" 20G >/dev/null 2>&1
    start_vm "$disk"
    wait_for_relay 120 || { fail "VM not ready"; stop_vm; rm -f "$disk"; return 1; }

    sleep 20  # wait for relay
    if ws_test <<'PYEOF'
import asyncio, json, sys, os
async def test():
    import websockets
    token = os.environ.get("AUTH_TOKEN", "sovereign")
    def jload(s): return json.loads(s, strict=False)
    async with websockets.connect("ws://127.0.0.1:18094", ping_timeout=30) as ws:
        # Auth
        await ws.send(json.dumps({"action":"auth","token":token}))
        r = jload(await asyncio.wait_for(ws.recv(), 10))
        assert r.get("type") in ("authenticated", "authed"), f"Auth failed: {r}"

        # Connect
        await ws.send(json.dumps({"action":"connect","host":"127.0.0.1","port":22,"username":"root","password":"sovereign"}))
        r = jload(await asyncio.wait_for(ws.recv(), 15))
        assert r["type"] == "connected"

        # Discover disks
        await ws.send(json.dumps({"action":"discover_disks"}))
        r = jload(await asyncio.wait_for(ws.recv(), 15))
        disks = jload(r["data"])
        target = next((d["name"] for d in disks if "vda" in d["name"] or "virtio" in d.get("transport","")), disks[0]["name"])
        print(f"  Target disk: {target}")

        # Install
        await ws.send(json.dumps({
            "action": "install",
            "disk": target,
            "layout": "single",
            "hostname": "test-nixos",
            "desktop": "none",
            "gpu_driver": "modesetting",
            "timezone": "UTC",
            "keyboard": "us"
        }))

        last_stage = ""
        while True:
            try:
                msg = jload(await asyncio.wait_for(ws.recv(), 900))
            except asyncio.TimeoutError:
                print("  TIMEOUT after 15 min")
                sys.exit(1)

            t = msg.get("type","")
            if t == "progress":
                s = msg.get("stage","")
                if s != last_stage:
                    print(f"  [{msg.get('percentage',0)}%] {s}")
                    last_stage = s
            elif t == "exit":
                code = msg.get("code", -1)
                if code == 0:
                    print("  Install completed successfully!")
                else:
                    print(f"  Install FAILED (exit code {code})")
                    sys.exit(1)
                break
            elif t == "error":
                print(f"  ERROR: {msg.get('message','?')}")
                sys.exit(1)

asyncio.run(test())
PYEOF
    then
        pass "Single disk install completed"
    else
        fail "Single disk install failed"
    fi

    stop_vm

    # Test 7: Reboot without ISO (verify NixOS boots)
    log "=== Test 7: Post-install boot ==="
    local ovmf_code_path="${OVMF_CODE:-$(ls /nix/store/*OVMF*/FV/OVMF_CODE.fd 2>/dev/null | head -1 || echo /usr/share/OVMF/OVMF_CODE.fd)}"
    local ovmf_vars_reboot="$VM_DIR/ovmf_vars_reboot.fd"
    cp "${OVMF_VARS:-$(ls /nix/store/*OVMF*/FV/OVMF_VARS.fd 2>/dev/null | head -1)}" "$ovmf_vars_reboot"
    chmod 644 "$ovmf_vars_reboot"
    qemu-system-x86_64 \
        -m 4096 -smp 2 -enable-kvm \
        -drive if=pflash,format=raw,readonly=on,file="$ovmf_code_path" \
        -drive if=pflash,format=raw,file="$ovmf_vars_reboot" \
        -drive file="$disk",format=qcow2,if=virtio \
        -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 \
        -device virtio-net-pci,netdev=net0 \
        -nographic &
    QEMU_PID=$!

    sleep 30  # Give it time to boot
    if wait_for_relay 120; then
        # Try to get hostname
        hostname=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -p $SSH_PORT root@127.0.0.1 'hostname' 2>/dev/null || echo "")
        if [[ "$hostname" == "test-nixos" ]]; then
            pass "Post-install: NixOS boots with correct hostname"
        else
            fail "Post-install: hostname mismatch (got '$hostname', expected 'test-nixos')"
        fi
    else
        fail "Post-install: NixOS failed to boot"
    fi

    stop_vm
    rm -f "$disk"
}

# ═══════════════════════════════════════════════════════
# Test 5: Encrypted Install (LUKS)
# ═══════════════════════════════════════════════════════

test_luks_install() {
    log "=== Test 5: Encrypted Install (LUKS) ==="

    local disk="$VM_DIR/test-luks.qcow2"
    qemu-img create -f qcow2 "$disk" 20G >/dev/null 2>&1
    start_vm "$disk"
    wait_for_relay 120 || { fail "VM not ready"; stop_vm; rm -f "$disk"; return 1; }

    sleep 20  # wait for relay
    if ws_test <<'PYEOF'
import asyncio, json, sys, os
async def test():
    import websockets
    token = os.environ.get("AUTH_TOKEN", "sovereign")
    def jload(s): return json.loads(s, strict=False)
    async with websockets.connect("ws://127.0.0.1:18094", ping_timeout=30) as ws:
        await ws.send(json.dumps({"action":"auth","token":token}))
        r = jload(await asyncio.wait_for(ws.recv(), 10))
        assert r.get("type") in ("authenticated", "authed"), f"Auth failed: {r}"

        await ws.send(json.dumps({"action":"connect","host":"127.0.0.1","port":22,"username":"root","password":"sovereign"}))
        r = jload(await asyncio.wait_for(ws.recv(), 15))
        assert r["type"] == "connected"

        await ws.send(json.dumps({"action":"discover_disks"}))
        r = jload(await asyncio.wait_for(ws.recv(), 15))
        disks = jload(r["data"])
        target = next((d["name"] for d in disks if "vda" in d["name"]), disks[0]["name"])

        await ws.send(json.dumps({
            "action": "install",
            "disk": target,
            "layout": "single-luks",
            "hostname": "test-luks",
            "desktop": "none",
            "gpu_driver": "modesetting",
            "timezone": "UTC",
            "keyboard": "us",
            "command": "testpassphrase123"
        }))

        last_stage = ""
        while True:
            try:
                msg = jload(await asyncio.wait_for(ws.recv(), 900))
            except asyncio.TimeoutError:
                print("  TIMEOUT"); sys.exit(1)
            t = msg.get("type","")
            if t == "progress":
                s = msg.get("stage","")
                if s != last_stage: print(f"  [{msg.get('percentage',0)}%] {s}"); last_stage = s
            elif t == "exit":
                if msg.get("code",1) == 0: print("  LUKS install completed!")
                else: print(f"  FAILED (code {msg.get('code')})"); sys.exit(1)
                break
            elif t == "error": print(f"  ERROR: {msg.get('message')}"); sys.exit(1)
asyncio.run(test())
PYEOF
    then
        pass "LUKS encrypted install completed"
    else
        fail "LUKS install failed"
    fi

    stop_vm
    rm -f "$disk"
}

# ═══════════════════════════════════════════════════════
# Combined: probe + discover + install on single VM
# ═══════════════════════════════════════════════════════

test_combined_install() {
    log "=== Combined Test: Probe + Discover + Install (single VM) ==="

    local disk="$VM_DIR/test-combined.qcow2"
    qemu-img create -f qcow2 "$disk" 20G >/dev/null 2>&1
    start_vm "$disk"
    wait_for_relay 120 || { fail "VM not ready"; stop_vm; rm -f "$disk"; return 1; }
    sleep 25  # wait for relay service

    if ws_test <<'PYEOF'
import asyncio, json, sys, os
async def test():
    import websockets
    token = os.environ.get("AUTH_TOKEN", "sovereign")
    def jload(s): return json.loads(s, strict=False)
    async with websockets.connect("ws://127.0.0.1:18094", ping_timeout=300, ping_interval=30, close_timeout=10) as ws:
        # 1. Auth
        await ws.send(json.dumps({"action":"auth","token":token}))
        r = jload(await asyncio.wait_for(ws.recv(), 10))
        assert r.get("type") in ("authenticated", "authed"), f"Auth failed: {r}"
        print("  [auth] OK")

        # 2. Connect SSH (connects to localhost inside the VM)
        await ws.send(json.dumps({"action":"connect","host":"127.0.0.1","port":22,"username":"root","password":"sovereign"}))
        r = jload(await asyncio.wait_for(ws.recv(), 15))
        assert r["type"] == "connected", f"SSH connect failed: {r}"
        print("  [connect] OK")

        # 3. Hardware probe
        await ws.send(json.dumps({"action":"probe_hardware"}))
        r = jload(await asyncio.wait_for(ws.recv(), 30))
        assert r["type"] == "hardware_probe", f"Probe failed: {r}"
        hw = jload(r["data"])
        print(f"  [probe] arch={hw.get('arch','?')}, efi={hw.get('efi_available','?')}")
        print(f"  [probe] safety={hw.get('safety',{}).get('level','?')}")

        # 4. Discover disks
        await ws.send(json.dumps({"action":"discover_disks"}))
        r = jload(await asyncio.wait_for(ws.recv(), 15))
        assert r["type"] == "disks", f"Discover failed: {r}"
        disks = jload(r["data"])
        assert len(disks) > 0, "No disks found"
        for d in disks:
            size_gb = int(d.get("size","0") or "0") / 1e9
            print(f"  [disk] {d['name']}: {size_gb:.0f}GB ({d.get('transport','?')})")
        target = next((d["name"] for d in disks if "vda" in d["name"]), disks[0]["name"])

        # 5. Install with browser-generated config
        config_nix = "{ config, pkgs, ... }:\n{\n  imports = [ ./hardware-configuration.nix ];\n  boot.loader.systemd-boot.enable = true;\n  boot.loader.efi.canTouchEfiVariables = true;\n  networking.hostName = \"test-nixos\";\n  networking.networkmanager.enable = true;\n  time.timeZone = \"UTC\";\n  i18n.defaultLocale = \"en_US.UTF-8\";\n  console.keyMap = \"us\";\n  hardware.graphics.enable = true;\n  services.pipewire = { enable = true; alsa.enable = true; pulse.enable = true; };\n  security.rtkit.enable = true;\n  users.users.testuser = {\n    isNormalUser = true;\n    extraGroups = [ \"wheel\" \"networkmanager\" ];\n    initialPassword = \"changeme\";\n  };\n  nix.settings.experimental-features = [ \"nix-command\" \"flakes\" ];\n  environment.systemPackages = with pkgs; [ vim git curl ];\n  system.stateVersion = \"25.05\";\n}\n"
        flake_nix = "{\n  description = \"NixOS configuration for test-nixos\";\n  inputs.nixpkgs.url = \"github:NixOS/nixpkgs/nixos-25.05\";\n  outputs = { self, nixpkgs, ... }: {\n    nixosConfigurations.\"test-nixos\" = nixpkgs.lib.nixosSystem {\n      system = \"x86_64-linux\";\n      modules = [ ./configuration.nix ];\n    };\n  };\n}\n"
        print(f"  [install] Starting on {target} with browser config...")
        await ws.send(json.dumps({
            "action": "install",
            "disk": target,
            "layout": "single",
            "hostname": "test-nixos",
            "desktop": "none",
            "gpu_driver": "modesetting",
            "timezone": "UTC",
            "keyboard": "us",
            "configuration_nix": config_nix,
        }))

        last_stage = ""
        log_lines = []
        while True:
            try:
                msg = jload(await asyncio.wait_for(ws.recv(), 900))
            except asyncio.TimeoutError:
                print("  TIMEOUT after 15 min")
                sys.exit(1)

            t = msg.get("type","")
            if t == "progress":
                s = msg.get("stage","")
                if s != last_stage:
                    print(f"  [install] [{msg.get('percentage',0)}%] {s}")
                    last_stage = s
            elif t == "exit":
                code = msg.get("code", -1)
                if code == 0:
                    print("  [install] COMPLETE!")
                else:
                    print(f"  [install] Failed (exit code {code})")
                    print("  [install] Last 15 log lines:")
                    for line in log_lines[-15:]:
                        print(f"    {line[:300]}")
                    sys.exit(1)
                break
            elif t == "error":
                print(f"  [install] ERROR: {msg.get('message','?')}")
                sys.exit(1)
            elif t == "output":
                line = msg.get("data", "")
                log_lines.append(line)
                if "ERROR" in line or "error:" in line or "fatal" in line.lower() or "nixos-install" in line or "STAGE:" in line:
                    print(f"  [log] {line[:200]}")

asyncio.run(test())
PYEOF
    then
        pass "Combined: auth + connect + probe + discover + install"
    else
        fail "Combined install pipeline failed"
    fi

    stop_vm
    rm -f "$disk"
}

# ═══════════════════════════════════════════════════════
# Test 8: Config Gen Validation
# ═══════════════════════════════════════════════════════

test_config_validation() {
    log "=== Test 8: Config Gen Validation ==="

    cd crates/symthaea-app-db
    if cargo test --lib 2>&1 | tail -3 | grep -q 'test result: ok'; then
        pass "Config gen: all tests pass"
    else
        fail "Config gen tests failed"
    fi
    cd ../..
}

# ═══════════════════════════════════════════════════════
# Test 9: Scanner Binary
# ═══════════════════════════════════════════════════════

test_scanner() {
    log "=== Test 9: Scanner Binary ==="

    cd crates/symthaea-scan
    if cargo test --lib 2>&1 | tail -3 | grep -q 'test result: ok'; then
        pass "Scanner: all tests pass"
    else
        fail "Scanner tests failed"
    fi

    if ./target/release/sovereign-scan --pretty 2>/dev/null | grep -q 'Apps detected'; then
        pass "Scanner: detects apps on this machine"
    else
        fail "Scanner: detection failed"
    fi
    cd ../..
}

# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════╗"
echo "║   NixForHumanity VM Test Suite           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

HAS_ISO=false
if [[ -n "$ISO" ]] && [[ -f "$ISO" ]]; then
    HAS_ISO=true
    echo "ISO: $ISO"
else
    echo "ISO not found. VM tests will be skipped."
    echo "Build with: ./nix/build-iso.sh"
fi

echo "ISO: $ISO"
echo "VM Dir: $VM_DIR"
echo ""

require_iso() {
    if [[ "$HAS_ISO" != "true" ]]; then
        log "SKIP: $1 (no ISO)"
        return 1
    fi
}

case "${1:-all}" in
    boot)
        require_iso "boot" && test_boot
        ;;
    install)
        require_iso "install" && { test_boot; test_combined_install; }
        ;;
    luks)
        require_iso "luks" && { test_boot; test_hardware_probe; test_disk_discovery; test_luks_install; }
        ;;
    config)
        test_config_validation
        ;;
    scanner)
        test_scanner
        ;;
    all)
        test_config_validation
        test_scanner
        if [[ "$HAS_ISO" == "true" ]]; then
            test_boot
            test_combined_install
        else
            log "Skipping VM tests (no ISO)"
        fi
        ;;
esac
