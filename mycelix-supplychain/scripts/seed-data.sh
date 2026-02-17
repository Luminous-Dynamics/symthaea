#!/bin/bash
#
# Seed Database with Sample Supply Chain Data
# Creates a complete supply chain scenario for testing
#

set -e

API_URL="${API_URL:-http://localhost:8080}"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}  Seeding Supply Chain Test Data${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Check if service is running
if ! curl -s "${API_URL}/health" > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠ Service not running at ${API_URL}${NC}"
    echo "Start it with: make run"
    exit 1
fi

echo -e "${GREEN}✓ Service is running${NC}"
echo ""

# Function to post event and extract claim ID
post_event() {
    local file=$1
    local desc=$2

    echo -e "${BLUE}→ ${desc}${NC}"

    response=$(curl -s -X POST "${API_URL}/v1/events" \
        -H 'Content-Type: application/json' \
        -d @"${file}")

    claim_id=$(echo "$response" | grep -o '"claim_id":"[^"]*"' | cut -d'"' -f4)

    if [ -n "$claim_id" ]; then
        echo -e "${GREEN}  ✓ Claim ID: ${claim_id}${NC}"
        echo "$claim_id"
    else
        echo -e "${YELLOW}  ⚠ Failed: $response${NC}"
        return 1
    fi
}

echo -e "${BLUE}Creating sample events...${NC}"
echo ""

# 1. Raw Material Production
claim1=$(post_event "specs/examples/batch_produced.json" "1. Producing batch of widgets at Plant A")
echo ""

# 2. Shipment
claim2=$(post_event "specs/examples/shipment_departed.json" "2. Shipping batch to warehouse")
echo ""

# 3. Certification
claim3=$(post_event "specs/examples/certificate_issued.json" "3. Issuing ISO 9001 certification")
echo ""

# Summary
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Seeding Complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""
echo "Created 3 claims with full lineage."
echo ""
echo -e "${BLUE}Try these queries:${NC}"
echo ""
echo "  # Get first claim:"
echo -e "  ${GREEN}curl ${API_URL}/v1/claims/${claim1}${NC}"
echo ""
echo "  # Verify last claim's VC:"
echo -e "  ${GREEN}curl -X POST ${API_URL}/v1/verify \\${NC}"
echo -e "    ${GREEN}-H 'Content-Type: application/json' \\${NC}"
echo -e "    ${GREEN}-d '{\"vc_jwt\":\"...\",\"check_lineage\":true}'${NC}"
echo ""
echo "  # Health check:"
echo -e "  ${GREEN}curl ${API_URL}/health${NC}"
echo ""
