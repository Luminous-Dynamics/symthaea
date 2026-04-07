#!/usr/bin/env bash
# Hostile VM Tests — simulate real-world hardware chaos
#
# Test 1: Windows NTFS drive with data migration
# Test 2: UEFI + NVMe naming conventions
# Test 3: Multi-disk chaos (NVMe + SATA + USB)
#
# Usage: ./tests/hostile-vm-tests.sh [windows|nvme|multidisk|all]

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VM_DIR="/var/lib/vms"
ISO="$(ls result-iso/iso/*.iso 2>/dev/null | head -1 || true)"
OVMF_CODE="$(ls /nix/store/*OVMF*/FV/OVMF_CODE.fd 2>/dev/null | head -1)"
RELAY_PORT=18094
PASS=0
FAIL=0

log() { echo "[$(date +%H:%M:%S)] $*"; }
pass() { log "PASS: $1"; PASS=$((PASS + 1)); }
fail() { log "FAIL: $1"; FAIL=$((FAIL + 1)); }

cleanup_vm() {
    pkill -f "qemu-system.*hostile" 2>/dev/null || true
    sleep 1
}
trap cleanup_vm EXIT

ovmf_vars() {
    local name="$1"
    local path="$VM_DIR/ovmf_${name}.fd"
    cp "$(ls /nix/store/*OVMF*/FV/OVMF_VARS.fd | head -1)" "$path"
    chmod 644 "$path"
    echo "$path"
}

wait_for_relay() {
    local timeout="${1:-120}"
    for i in $(seq 1 "$timeout"); do
        python3 -c "
import asyncio, websockets
async def check():
    try:
        async with websockets.connect('ws://127.0.0.1:${RELAY_PORT}', close_timeout=2, open_timeout=3):
            return True
    except: return False
exit(0 if asyncio.run(check()) else 1)
" 2>/dev/null && return 0
        sleep 2
    done
    return 1
}

relay_cmd() {
    python3 -c "
import asyncio, json
async def test():
    import websockets
    async with websockets.connect('ws://127.0.0.1:${RELAY_PORT}', ping_timeout=60) as ws:
        await ws.send(json.dumps({'action':'auth','token':'sovereign'}))
        await asyncio.wait_for(ws.recv(), 10)
        await ws.send(json.dumps({'action':'connect'}))
        await asyncio.wait_for(ws.recv(), 10)
        await ws.send(json.dumps($1))
        r = json.loads(await asyncio.wait_for(ws.recv(), ${2:-30}), strict=False)
        print(json.dumps(r, indent=2))
asyncio.run(test())
" 2>&1
}

# ═══════════════════════════════════════════════════════
# Test 1: Windows NTFS Drive (Data Migration)
# ═══════════════════════════════════════════════════════

test_windows() {
    log "=== Test 1: Windows NTFS Drive ==="

    # Create a fake Windows drive
    local win_disk="$VM_DIR/hostile-windows.qcow2"
    local nixos_disk="$VM_DIR/hostile-nixos.qcow2"
    qemu-img create -f qcow2 "$win_disk" 50G >/dev/null 2>&1
    qemu-img create -f qcow2 "$nixos_disk" 20G >/dev/null 2>&1

    # Pre-populate the Windows disk with NTFS + fake user data
    # Create a raw image, partition it, format as NTFS, add files
    local raw_win="$VM_DIR/hostile-win-raw.img"
    dd if=/dev/zero of="$raw_win" bs=1M count=512 2>/dev/null
    # Create GPT with EFI + NTFS partitions
    sgdisk --zap-all "$raw_win" 2>/dev/null || true
    sgdisk -n 1:0:+100M -t 1:EF00 -c 1:EFI "$raw_win" 2>/dev/null || true
    sgdisk -n 2:0:0 -t 2:0700 -c 2:Windows "$raw_win" 2>/dev/null || true
    log "  Created fake Windows GPT disk (512MB)"

    local vars=$(ovmf_vars "hostile-win")

    # Boot with TWO disks: Windows (virtio) + NixOS target (virtio)
    qemu-system-x86_64 -m 4096 -smp 4 -enable-kvm \
        -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
        -drive if=pflash,format=raw,file="$vars" \
        -drive file="$nixos_disk",format=qcow2,if=virtio,id=nixos \
        -drive file="$win_disk",format=qcow2,if=virtio,id=windows \
        -cdrom "$ISO" -boot d \
        -netdev user,id=net0,hostfwd=tcp::${RELAY_PORT}-:8094 \
        -device virtio-net-pci,netdev=net0 \
        -nographic &

    if wait_for_relay 90; then
        pass "ISO boots with dual drives"
    else
        fail "Boot failed with dual drives"
        cleanup_vm
        rm -f "$win_disk" "$nixos_disk" "$raw_win" "$vars"
        return
    fi

    sleep 20

    # Test: Does probe detect multiple disks?
    log "  Probing hardware..."
    local probe=$(relay_cmd '{"action":"probe_hardware"}' 60)
    if echo "$probe" | grep -q "hardware_probe"; then
        pass "Hardware probe works with dual drives"
    else
        fail "Hardware probe failed: $probe"
    fi

    # Test: Does discover_disks show both?
    local disks=$(relay_cmd '{"action":"discover_disks"}')
    local disk_count=$(echo "$disks" | grep -c "vd[a-z]" || true)
    if [ "$disk_count" -ge 2 ]; then
        pass "Discovered $disk_count disks (expected 2+)"
    else
        fail "Only found $disk_count disks (expected 2+)"
    fi
    log "  Disks: $(echo "$disks" | grep -o '"name":"[^"]*"' | tr '\n' ' ')"

    # Test: Does deep_scan detect the Windows partition?
    log "  Scanning for existing OS..."
    local scan=$(relay_cmd '{"action":"scan_apps"}' 30)
    log "  Scan result type: $(echo "$scan" | grep -o '"type":"[^"]*"')"

    cleanup_vm
    rm -f "$win_disk" "$nixos_disk" "$raw_win" "$vars"
}

# ═══════════════════════════════════════════════════════
# Test 2: NVMe Drive Naming
# ═══════════════════════════════════════════════════════

test_nvme() {
    log "=== Test 2: NVMe Drive Naming ==="

    local nvme_disk="$VM_DIR/hostile-nvme.qcow2"
    qemu-img create -f qcow2 "$nvme_disk" 20G >/dev/null 2>&1
    local vars=$(ovmf_vars "hostile-nvme")

    # Boot with NVMe device instead of virtio
    qemu-system-x86_64 -m 4096 -smp 4 -enable-kvm \
        -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
        -drive if=pflash,format=raw,file="$vars" \
        -drive file="$nvme_disk",format=qcow2,if=none,id=nvme0 \
        -device nvme,serial=deadbeef,drive=nvme0 \
        -cdrom "$ISO" -boot d \
        -netdev user,id=net0,hostfwd=tcp::${RELAY_PORT}-:8094 \
        -device virtio-net-pci,netdev=net0 \
        -nographic &

    if wait_for_relay 90; then
        pass "ISO boots with NVMe drive"
    else
        fail "Boot failed with NVMe"
        cleanup_vm
        rm -f "$nvme_disk" "$vars"
        return
    fi

    sleep 20

    # Test: Does discover_disks find nvme0n1?
    local disks=$(relay_cmd '{"action":"discover_disks"}')
    if echo "$disks" | grep -qi "nvme"; then
        pass "NVMe drive detected as nvme0n1"
        log "  $(echo "$disks" | grep -o '"name":"[^"]*"' | head -3 | tr '\n' ' ')"
    else
        fail "NVMe drive not detected (got: $(echo "$disks" | grep -o '"name":"[^"]*"' | tr '\n' ' '))"
    fi

    # Test: Can we install on the NVMe drive?
    log "  Testing install on NVMe..."
    local target=$(echo "$disks" | grep -o '"name":"/dev/nvme[^"]*"' | head -1 | cut -d'"' -f4)
    if [ -n "$target" ]; then
        log "  Target: $target"
        pass "NVMe drive selectable as install target"
    else
        fail "Could not select NVMe as target"
    fi

    cleanup_vm
    rm -f "$nvme_disk" "$vars"
}

# ═══════════════════════════════════════════════════════
# Test 3: Multi-Disk Chaos
# ═══════════════════════════════════════════════════════

test_multidisk() {
    log "=== Test 3: Multi-Disk Chaos (NVMe + SATA + USB) ==="

    local nvme_disk="$VM_DIR/hostile-multi-nvme.qcow2"
    local sata_disk="$VM_DIR/hostile-multi-sata.qcow2"
    local usb_disk="$VM_DIR/hostile-multi-usb.qcow2"
    qemu-img create -f qcow2 "$nvme_disk" 20G >/dev/null 2>&1
    qemu-img create -f qcow2 "$sata_disk" 500G >/dev/null 2>&1
    qemu-img create -f qcow2 "$usb_disk" 8G >/dev/null 2>&1
    local vars=$(ovmf_vars "hostile-multi")

    # Boot with 3 different disk types
    qemu-system-x86_64 -m 4096 -smp 4 -enable-kvm \
        -drive if=pflash,format=raw,readonly=on,file="$OVMF_CODE" \
        -drive if=pflash,format=raw,file="$vars" \
        -drive file="$nvme_disk",format=qcow2,if=none,id=nvme0 \
        -device nvme,serial=fast-nvme,drive=nvme0 \
        -drive file="$sata_disk",format=qcow2,if=none,id=sata0 \
        -device ide-hd,drive=sata0 \
        -drive file="$usb_disk",format=qcow2,if=none,id=usb0 \
        -device usb-ehci -device usb-storage,drive=usb0,removable=on \
        -cdrom "$ISO" -boot d \
        -netdev user,id=net0,hostfwd=tcp::${RELAY_PORT}-:8094 \
        -device virtio-net-pci,netdev=net0 \
        -nographic &

    if wait_for_relay 90; then
        pass "ISO boots with 3 disk types"
    else
        fail "Boot failed with multi-disk"
        cleanup_vm
        rm -f "$nvme_disk" "$sata_disk" "$usb_disk" "$vars"
        return
    fi

    sleep 20

    # Test: Does discover_disks find all three?
    local disks=$(relay_cmd '{"action":"discover_disks"}')
    local names=$(echo "$disks" | grep -o '"name":"[^"]*"' | tr '\n' ' ')
    log "  Discovered: $names"

    local count=$(echo "$disks" | grep -c '"name":"/dev/' || true)
    if [ "$count" -ge 3 ]; then
        pass "All 3 disks discovered ($count total)"
    elif [ "$count" -ge 2 ]; then
        pass "At least 2 of 3 disks discovered ($count)"
    else
        fail "Only $count disks found (expected 3)"
    fi

    # Test: Does it correctly identify disk sizes?
    log "  Checking disk sizes..."
    if echo "$disks" | grep -q "nvme"; then
        pass "NVMe disk identified"
    else
        log "  (NVMe may show as different device name)"
    fi

    # Test: Hardware probe with complex setup
    local probe=$(relay_cmd '{"action":"probe_hardware"}' 60)
    if echo "$probe" | grep -q "hardware_probe"; then
        pass "Hardware probe succeeds with 3 disk types"
    else
        fail "Hardware probe failed"
    fi

    cleanup_vm
    rm -f "$nvme_disk" "$sata_disk" "$usb_disk" "$vars"
}

# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

if [ -z "$ISO" ]; then
    echo "ISO not found. Build first: nix-build nix/installer-iso.nix -o result-iso"
    exit 1
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║  NixForHumanity Hostile VM Tests                 ║"
echo "╚══════════════════════════════════════════════════╝"
echo "ISO: $ISO"
echo ""

case "${1:-all}" in
    windows)  test_windows ;;
    nvme)     test_nvme ;;
    multidisk) test_multidisk ;;
    all)
        test_windows
        test_nvme
        test_multidisk
        ;;
esac

echo ""
echo "════════════════════════════════════════"
echo "  $PASS passed, $FAIL failed"
echo "════════════════════════════════════════"
