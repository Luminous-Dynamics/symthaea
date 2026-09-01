#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Capture a privacy-minimized boot performance evidence directory.
# This script intentionally does not collect journal contents, hostname, SSIDs,
# serial numbers, process lists, environment variables, or command lines.

set -euo pipefail

OUT="${1:-spore-boot-measurement-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"

capture() {
    local target="$1"
    shift
    if "$@" >"$OUT/$target" 2>&1; then
        return 0
    fi
    printf 'command unavailable or failed\n' >"$OUT/$target"
}

capture systemd-analyze-time.txt systemd-analyze time
capture systemd-critical-chain.txt systemd-analyze critical-chain
capture systemd-blame.txt systemd-analyze blame

capture spore-unit.properties systemctl show symthaea-boot-animation.service \
    --property=LoadState \
    --property=ActiveState \
    --property=SubState \
    --property=ExecMainStartTimestampMonotonic \
    --property=ExecMainExitTimestampMonotonic \
    --property=ActiveEnterTimestampMonotonic \
    --property=InactiveEnterTimestampMonotonic

capture display-manager.properties systemctl show display-manager.service \
    --property=LoadState \
    --property=ActiveState \
    --property=SubState \
    --property=ExecMainStartTimestampMonotonic \
    --property=ActiveEnterTimestampMonotonic

capture graphical-target.properties systemctl show graphical.target \
    --property=ActiveState \
    --property=ActiveEnterTimestampMonotonic

capture kernel-release.txt uname -r
if command -v nixos-version >/dev/null 2>&1; then
    capture nixos-version.txt nixos-version
fi

for receipt in \
    /run/symthaea/boot-performance-v1.json \
    /run/symthaea/boot-display-released-v1.json
 do
    if [[ -f "$receipt" ]]; then
        cp -- "$receipt" "$OUT/$(basename "$receipt")"
    fi
 done

cat >"$OUT/README.txt" <<'EOF'
Spore boot performance evidence v1

Collected:
- systemd aggregate boot timing;
- systemd critical chain and blame views;
- selected monotonic timing properties for Spore, display manager, graphical target;
- kernel/NixOS version when available;
- opt-in Spore renderer and handoff receipts when present.

Deliberately not collected:
- journal contents;
- hostname;
- user names;
- SSIDs/network identifiers;
- hardware serials;
- process lists;
- environment variables;
- command lines.

Use repeated alternating Spore ON/OFF boots for comparative claims. One boot is
an observation, not a performance conclusion.
EOF

printf '%s\n' "$OUT"
