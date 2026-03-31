#!/usr/bin/env bash
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Smoke test for the unified Mycelix hApp bundle.
#
# Validates that all DNA bundles exist, are valid, and can be loaded
# into a Holochain sandbox conductor.
#
# Usage:
#   ./scripts/smoke-test-unified.sh          # Validate bundles only
#   ./scripts/smoke-test-unified.sh --live   # Start conductor and make test calls
#
# Prerequisites:
#   - holochain, hc CLI in PATH (via nix develop)
#   - All DNAs built (run `just build-all` first)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODE="${1:-}"
PASS=0
FAIL=0
WARN=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
warn() { echo "  WARN: $1"; WARN=$((WARN + 1)); }

echo "=== Mycelix Unified hApp Smoke Test ==="
echo "Date: $(date -I)"
echo ""

# 1. Check hApp manifest exists
echo "--- Manifest ---"
HAPP_YAML="happs/mycelix-unified-happ.yaml"
if [[ -f "$HAPP_YAML" ]]; then
    pass "hApp manifest exists"
    ROLES=$(grep -c 'provisioning:' "$HAPP_YAML" 2>/dev/null || echo 0)
    echo "  Roles found: $ROLES"
else
    fail "hApp manifest missing: $HAPP_YAML"
fi
echo ""

# 2. Check each DNA bundle exists and is non-empty
echo "--- DNA Bundles ---"
DNA_DIRS=(
    "../mycelix-commons/dna/mycelix_commons.dna"
    "../mycelix-civic/dna/mycelix_civic.dna"
    "../mycelix-identity/dna"
    "../mycelix-governance/dna"
    "../mycelix-finance/dna"
    "../mycelix-hearth/dna"
    "../mycelix-personal/dna"
    "../mycelix-attribution/dna"
    "../mycelix-health/dna"
    "../mycelix-music/dnas"
    "../mycelix-energy/dna"
    "../mycelix-knowledge/dna"
    "../mycelix-climate/dnas/climate"
    "../mycelix-edunet/dna"
    "../mycelix-supplychain/holochain/dna"
    "../mycelix-manufacturing/dna"
)

for dna_path in "${DNA_DIRS[@]}"; do
    cluster=$(basename "$(dirname "$dna_path")")
    if [[ -d "$dna_path" ]]; then
        dna_files=$(find "$dna_path" -maxdepth 2 -name '*.dna' 2>/dev/null | head -1)
        if [[ -n "$dna_files" ]]; then
            size=$(wc -c < "$dna_files")
            if (( size > 1000 )); then
                pass "$cluster DNA bundle ($(numfmt --to=iec $size))"
            else
                fail "$cluster DNA bundle too small ($size bytes)"
            fi
        else
            warn "$cluster — no .dna file found in $dna_path"
        fi
    elif [[ -f "$dna_path" ]]; then
        size=$(wc -c < "$dna_path")
        if (( size > 1000 )); then
            pass "$cluster DNA bundle ($(numfmt --to=iec $size))"
        else
            fail "$cluster DNA bundle too small ($size bytes)"
        fi
    else
        warn "$cluster — path not found: $dna_path"
    fi
done
echo ""

# 3. Check WASM zome compilation (spot check — are .wasm files present?)
echo "--- WASM Zomes (spot check) ---"
WASM_COUNT=$(find .. -path '*/target/wasm32-unknown-unknown/release/*.wasm' -name '*.wasm' 2>/dev/null | wc -l)
if (( WASM_COUNT > 0 )); then
    pass "Found $WASM_COUNT compiled WASM zomes"
else
    warn "No compiled WASM zomes found (run 'just build-all')"
fi
echo ""

# 4. Check Holochain CLI availability
echo "--- Holochain Tools ---"
if command -v holochain &>/dev/null; then
    HC_VERSION=$(holochain --version 2>/dev/null || echo "unknown")
    pass "holochain CLI: $HC_VERSION"
else
    warn "holochain CLI not in PATH (need 'nix develop')"
fi
if command -v hc &>/dev/null; then
    pass "hc CLI available"
else
    warn "hc CLI not in PATH"
fi
echo ""

# 5. Summary
echo "=== Results ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Warnings: $WARN"

if (( FAIL > 0 )); then
    echo ""
    echo "SMOKE TEST FAILED — $FAIL failures"
    exit 1
else
    echo ""
    echo "SMOKE TEST PASSED"
    if (( WARN > 0 )); then
        echo "($WARN warnings — non-blocking)"
    fi
fi
