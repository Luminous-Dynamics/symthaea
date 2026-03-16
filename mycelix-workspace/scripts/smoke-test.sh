#!/usr/bin/env bash
# Mycelix Resilience Kit — Smoke Test
#
# Starts Observatory in preview mode, checks all routes return 200,
# then shuts down. Catches broken imports before field deployment.
#
# Usage:
#   ./scripts/smoke-test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OBS_DIR="$WORKSPACE_DIR/observatory"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PREVIEW_PID=""
PORT=4176  # Different from dev port to avoid conflicts

cleanup() {
    if [ -n "$PREVIEW_PID" ]; then
        kill "$PREVIEW_PID" 2>/dev/null || true
        wait "$PREVIEW_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo ""
echo "============================================"
echo "  RESILIENCE KIT — SMOKE TEST"
echo "============================================"
echo ""

# Build
echo -e "${YELLOW}[1/3]${NC} Building Observatory..."
cd "$OBS_DIR"
npx vite build 2>&1 | tail -3
echo ""

# Start preview server
echo -e "${YELLOW}[2/3]${NC} Starting preview server on port $PORT..."
npx vite preview --port "$PORT" &>/dev/null &
PREVIEW_PID=$!
sleep 3

# Verify the server started
if ! kill -0 "$PREVIEW_PID" 2>/dev/null; then
    echo -e "${RED}FAIL${NC}: Preview server failed to start"
    exit 1
fi

# Check routes
echo -e "${YELLOW}[3/3]${NC} Checking routes..."
ROUTES=(
    "/"
    "/resilience"
    "/tend"
    "/food"
    "/mutual-aid"
    "/emergency"
    "/value-anchor"
    "/water"
    "/household"
    "/knowledge"
    "/care-circles"
    "/shelter"
    "/supplies"
    "/admin"
    "/governance"
    "/network"
    "/analytics"
    "/attribution"
    "/welcome"
    "/print"
)

PASSED=0
FAILED=0

for route in "${ROUTES[@]}"; do
    status=$(curl -sf -o /dev/null -w '%{http_code}' --connect-timeout 5 "http://localhost:${PORT}${route}" 2>/dev/null || echo "000")
    if [ "$status" = "200" ]; then
        echo -e "  ${GREEN}OK${NC}  $route"
        PASSED=$((PASSED + 1))
    else
        echo -e "  ${RED}FAIL${NC} $route (HTTP $status)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
if [ "$FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}ALL $PASSED ROUTES PASSED${NC}"
else
    echo -e "  ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC}"
fi
echo "============================================"
echo ""

exit "$FAILED"
