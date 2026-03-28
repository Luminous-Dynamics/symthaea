#!/usr/bin/env bash
set -euo pipefail

# Mycelix Crate Publishing Script
# Publishes standalone crates to crates.io in dependency order.
#
# Prerequisites:
#   cargo login <your-crates-io-token>
#
# Usage:
#   ./scripts/publish-crates.sh          # Dry run (default)
#   ./scripts/publish-crates.sh --publish # Actually publish

DRY_RUN=true
if [[ "${1:-}" == "--publish" ]]; then
    DRY_RUN=false
fi

PUBLISH_CMD="cargo publish"
if $DRY_RUN; then
    PUBLISH_CMD="cargo publish --dry-run"
    echo "=== DRY RUN MODE (pass --publish to actually publish) ==="
fi

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║  Mycelix Crate Publishing Pipeline                ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# Phase 1: No dependencies (publish in parallel conceptually, but sequential for safety)
echo "── Phase 1: Independent crates ──"

echo "[1/7] Publishing mycelix-core-types..."
(cd mycelix-core/libs/mycelix-core-types && $PUBLISH_CMD --allow-dirty)
if ! $DRY_RUN; then sleep 30; fi  # Wait for crates.io index

echo "[2/7] Publishing mycelix-differential-privacy..."
(cd mycelix-core/libs/differential-privacy && $PUBLISH_CMD --allow-dirty)

echo "[3/7] Publishing feldman-dkg..."
(cd mycelix-core/libs/feldman-dkg && $PUBLISH_CMD --allow-dirty)

echo "[4/7] Publishing winterfell-pogq..."
(cd mycelix-core/0TML/vsv-stark/winterfell-pogq && $PUBLISH_CMD --allow-dirty)

echo "[5/7] Publishing symthaea-hdc-crypto..."
(cd symthaea-hdc-crypto && $PUBLISH_CMD --allow-dirty)

# Phase 2: Depends on mycelix-core-types
if ! $DRY_RUN; then
    echo ""
    echo "── Waiting 60s for crates.io index to update ──"
    sleep 60
fi
echo ""
echo "── Phase 2: Dependent crates ──"

echo "[6/7] Publishing rb-bft-consensus..."
(cd mycelix-core/libs/rb-bft-consensus && $PUBLISH_CMD --allow-dirty)

echo "[7/7] Publishing kvector-zkp..."
(cd mycelix-core/libs/kvector-zkp && $PUBLISH_CMD --allow-dirty)

echo ""
echo "═══════════════════════════════════════════════════"
if $DRY_RUN; then
    echo "  DRY RUN COMPLETE — all crates would publish successfully"
    echo "  Run with --publish to actually publish to crates.io"
else
    echo "  ALL 7 CRATES PUBLISHED TO CRATES.IO"
fi
echo "═══════════════════════════════════════════════════"
