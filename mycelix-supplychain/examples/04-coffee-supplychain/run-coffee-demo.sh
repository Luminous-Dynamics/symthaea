#!/bin/bash
#
# Coffee Supply Chain Demo Script
# Demonstrates complete farm-to-cup traceability
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

API_URL="${SUPPLYCHAIN_URL:-http://localhost:8080}"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Coffee Supply Chain - Farm to Cup Traceability Demo   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if service is running
echo -e "${YELLOW}→${NC} Checking if provenance service is running..."
if ! curl -s -f "$API_URL/health" > /dev/null; then
    echo -e "${RED}✗${NC} Service is not running at $API_URL"
    echo "  Start the service with: cargo run --release"
    exit 1
fi
echo -e "${GREEN}✓${NC} Service is healthy"
echo ""

# Function to ingest an event
ingest_event() {
    local step=$1
    local file=$2
    local description=$3

    echo -e "${BLUE}═══ Step $step: $description ═══${NC}"

    response=$(curl -s -X POST "$API_URL/v1/events" \
        -H 'Content-Type: application/json' \
        -d @"events/$file")

    claim_id=$(echo "$response" | jq -r '.claim_id')
    lineage_hash=$(echo "$response" | jq -r '.lineage_hash' | cut -c1-16)

    echo -e "${GREEN}✓${NC} Claim created: ${YELLOW}$claim_id${NC}"
    echo -e "  Lineage hash: $lineage_hash..."
    echo ""
}

# Execute the supply chain journey
echo -e "${BLUE}Starting coffee journey from Ethiopia to your cup...${NC}"
echo ""

sleep 1

ingest_event "1" "01-farm-produced.json" \
    "☕ Farm produces 5000kg coffee cherries (Ethiopia)"

sleep 0.5

ingest_event "2" "02-farm-certified-organic.json" \
    "🌿 Organic certification issued"

sleep 0.5

ingest_event "3" "03-processor-transformed.json" \
    "⚙️  Processing mill transforms cherries → 1000kg green beans"

sleep 0.5

ingest_event "4" "04-exporter-certified-fairtrade.json" \
    "🤝 Fair Trade certification issued"

sleep 0.5

ingest_event "5" "05-exporter-shipped.json" \
    "🚢 Export shipment to USA (33-day journey)"

sleep 0.5

ingest_event "6" "06-roaster-received.json" \
    "📦 Roaster receives green beans in Oakland, CA"

sleep 0.5

ingest_event "7" "07-roaster-transformed.json" \
    "🔥 Roasting transforms green → 850kg roasted beans"

sleep 0.5

ingest_event "8" "08-cafe-received.json" \
    "☕ Cafe receives roasted beans in San Francisco"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓${NC} All 8 events ingested successfully!"
echo ""

# Retrieve lineage
echo -e "${YELLOW}→${NC} Retrieving complete lineage for final batch..."
lineage=$(curl -s "$API_URL/v1/batches/BATCH-2025-ROASTED-001/lineage")
claim_count=$(echo "$lineage" | jq '.claims | length')

echo -e "${GREEN}✓${NC} Lineage retrieved: $claim_count events tracked"
echo ""

# Display lineage tree
echo -e "${BLUE}Complete Supply Chain Lineage:${NC}"
echo ""
echo "$lineage" | jq -r '.claims[] | "\(.assertion.event_type | ljust(15)) | \(.subject.batch_id | ljust(30)) | \(.evidence.facility.name)"'
echo ""

# Summary
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     Journey Summary                      ║${NC}"
echo -e "${BLUE}╠══════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║${NC} Organizations:  5 (Farm, Processor, Exporter, Roaster, Cafe) "
echo -e "${BLUE}║${NC} Countries:      2 (Ethiopia → USA)                     "
echo -e "${BLUE}║${NC} Events:         8 total                                "
echo -e "${BLUE}║${NC} Certifications: 2 (Organic, Fair Trade)                "
echo -e "${BLUE}║${NC} Transformations: 2 (Cherries→Green, Green→Roasted)     "
echo -e "${BLUE}║${NC} Final Yield:    17% of original cherry weight          "
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}Demo completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  • View lineage: supplychain lineage BATCH-2025-ROASTED-001"
echo "  • Get claim: supplychain get <claim-id>"
echo "  • Check metrics: curl $API_URL/metrics"
echo ""
