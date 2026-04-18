#!/usr/bin/env bash
# verify-demos.sh — headless smoke-test of all symtropy platform demos.
#
# Each demo is launched with SYMTROPY_DEMO_CAPTURE_DIR pointing to a
# per-demo tmpdir. The capture plugin schedules screenshots at
# t = 1.5, 4.0, 7.0 s and then exits cleanly at 8.5 s. We also wrap
# in `timeout 45s` as a belt-and-braces fallback in case a demo hangs.
#
# NixOS runtime note: Bevy 0.18 needs libX11/libXi/vulkan-loader/libGL at
# runtime. The monorepo flake at /srv/luminous-dynamics/flake.nix already
# sets LD_LIBRARY_PATH for these, so this script auto-re-execs inside
# `nix develop` when it isn't already.
#
# After each run we check:
#   - exit code was 0 (or 124 which `timeout` reports on forced kill)
#   - 3 PNG files landed in the capture dir
# and report GREEN/RED per demo.
#
# Captured PNGs stay in /tmp/symtropy-verify/<demo>/ for eyeball inspection.
# This script does NOT commit or push anything.

# Ensure we're inside the nix develop shell (needed for libX11/vulkan on NixOS).
if [ -z "${IN_NIX_SHELL:-}" ]; then
    exec nix develop /srv/luminous-dynamics --command bash "$0" "$@"
fi

# Provide a virtual X display if DISPLAY isn't already set. Running against
# Xvfb avoids popping windows on the user's real desktop.
if [ -z "${DISPLAY:-}" ] || [ "${SYMTROPY_VERIFY_USE_XVFB:-0}" = "1" ]; then
    Xvfb :99 -screen 0 1600x900x24 > /dev/null 2>&1 &
    XVFB_PID=$!
    trap 'kill $XVFB_PID 2>/dev/null || true' EXIT
    export DISPLAY=:99
    sleep 1
fi

set -uo pipefail

DEMOS=(
    manipulator
    flight
    vehicle
    auv
    helicopter
    exoskeleton
    orbital
    surgical
    humanoid
    quadruped
)

REPO="/srv/luminous-dynamics"
BASE_CAPTURE="/tmp/symtropy-verify"
HARD_TIMEOUT=45
SCRIPT_START="$(date +%s)"

mkdir -p "$BASE_CAPTURE"
rm -rf "$BASE_CAPTURE"/*

printf "%-15s | %-6s | %-4s | %s\n" "demo" "exit" "pngs" "status"
printf -- "--------------- | ------ | ---- | -----------\n"

fail_count=0
for demo in "${DEMOS[@]}"; do
    crate_dir="$REPO/symtropy/crates/symtropy-${demo}-demo"
    capture_dir="$BASE_CAPTURE/$demo"
    mkdir -p "$capture_dir"

    if [ ! -f "$crate_dir/Cargo.toml" ]; then
        printf "%-15s | %-6s | %-4s | MISSING (no Cargo.toml)\n" "$demo" "--" "0"
        fail_count=$((fail_count + 1))
        continue
    fi

    SYMTROPY_DEMO_CAPTURE_DIR="$capture_dir" \
    timeout "${HARD_TIMEOUT}s" \
        cargo run --release --manifest-path "$crate_dir/Cargo.toml" \
            --bin "${demo}-demo" \
            > "$capture_dir/stdout.log" 2> "$capture_dir/stderr.log"
    exit_code=$?

    png_count=$(find "$capture_dir" -maxdepth 1 -name "${demo}_t*.png" 2>/dev/null | wc -l)

    if [ "$exit_code" -eq 0 ] && [ "$png_count" -ge 3 ]; then
        status="GREEN"
    elif [ "$exit_code" -eq 124 ]; then
        status="TIMEOUT (captures: $png_count)"
        fail_count=$((fail_count + 1))
    else
        status="RED (see $capture_dir/stderr.log)"
        fail_count=$((fail_count + 1))
    fi
    printf "%-15s | %-6s | %-4s | %s\n" "$demo" "$exit_code" "$png_count" "$status"
done

elapsed=$(($(date +%s) - SCRIPT_START))
echo ""
echo "total wall-clock: ${elapsed}s"
if [ "$fail_count" -eq 0 ]; then
    echo "all ${#DEMOS[@]}/${#DEMOS[@]} demos GREEN"
    exit 0
fi
echo "${fail_count} of ${#DEMOS[@]} demos failed"
exit 1
