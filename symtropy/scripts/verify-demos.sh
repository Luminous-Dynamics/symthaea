#!/usr/bin/env bash
# verify-demos.sh — headless smoke-test of all symtropy platform demos.
#
# Each demo is launched with SYMTROPY_DEMO_CAPTURE_DIR pointing to a
# per-demo tmpdir. The capture plugin schedules screenshots at
# t = 1.5, 4.0, 7.0 s and then exits cleanly at 8.5 s. We also wrap
# in `timeout 45s` as a belt-and-braces fallback in case a demo hangs.
#
# After each run we check:
#   - exit code was 0 (or 124 which `timeout` reports on forced kill)
#   - 3 PNG files landed in the capture dir
# and report GREEN/RED per demo.
#
# Captured PNGs stay in /tmp/symtropy-verify/<demo>/ for eyeball inspection.
# This script does NOT commit or push anything.

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
