#!/bin/bash
#
# Automated script to run the simple flow example
#

set -e

API_URL="${API_URL:-http://localhost:8080}"

echo "╔════════════════════════════════════════╗"
echo "║  Example 1: Simple Flow                ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Check if service is running
echo "→ Checking if service is running..."
if ! curl -s "${API_URL}/health" > /dev/null 2>&1; then
    echo "✗ Service not running at ${API_URL}"
    echo "  Start it with: cd ../../rust/service && cargo run"
    exit 1
fi
echo "✓ Service is running"
echo ""

# Post event
echo "→ Posting PRODUCED event..."
RESPONSE=$(curl -s -X POST "${API_URL}/v1/events" \
    -H 'Content-Type: application/json' \
    -d @event.json)

CLAIM_ID=$(echo "$RESPONSE" | grep -o '"claim_id":"[^"]*"' | cut -d'"' -f4)
VC_JWT=$(echo "$RESPONSE" | grep -o '"vc_jwt":"[^"]*"' | cut -d'"' -f4)

if [ -z "$CLAIM_ID" ]; then
    echo "✗ Failed to create claim"
    echo "$RESPONSE"
    exit 1
fi

echo "✓ Event created successfully"
echo "  Claim ID: $CLAIM_ID"
echo "  VC JWT: ${VC_JWT:0:50}..."
echo ""

# Retrieve claim
echo "→ Retrieving claim..."
CLAIM=$(curl -s "${API_URL}/v1/claims/${CLAIM_ID}")

if echo "$CLAIM" | grep -q "\"id\":\"${CLAIM_ID}\""; then
    echo "✓ Claim retrieved successfully"
    echo ""
    echo "Claim Details:"
    echo "$CLAIM" | jq '.'
else
    echo "✗ Failed to retrieve claim"
    echo "$CLAIM"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════╗"
echo "║  Example Complete! ✓                   ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  - Try modifying event.json"
echo "  - Check out Example 2: Full Supply Chain"
echo "  - View claim ID: $CLAIM_ID"
