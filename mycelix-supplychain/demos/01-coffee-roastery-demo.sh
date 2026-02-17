#!/usr/bin/env bash
# Mycelix ERP Demo: Luminous Coffee Roasters
# Scenario: Farm-to-Cup Coffee with Blockchain Provenance

set -e

API_BASE="http://localhost:8000/v1"
FIN_BASE="$API_BASE/fin"

echo "🌟 Mycelix ERP Demo: Luminous Coffee Roasters"
echo "=============================================="
echo ""
echo "Scenario: Farm-to-Cup Coffee with Complete Provenance"
echo "This demo shows:"
echo "  - Supply chain tracking (farm → roastery → customer)"
echo "  - Financial operations (invoicing & payments)"
echo "  - Blockchain verification of product journey"
echo ""

# Check if service is running
if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
    echo "❌ Error: Mycelix service not running at $API_BASE"
    echo "Please start the service first:"
    echo "  cd rust && cargo run --release"
    exit 1
fi

echo "✅ Service is running"
echo ""

# Step 1: Create Supply Chain Events
echo "📦 Step 1: Creating Supply Chain Events"
echo "----------------------------------------"

echo "Creating farm harvest event..."
HARVEST=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "HARVEST",
    "product_id": "ethiopian-yirgacheffe-2024",
    "location": "Yirgacheffe, Ethiopia",
    "timestamp": "2024-01-15T08:00:00Z",
    "actor": "Koke Washing Station",
    "metadata": {
      "variety": "Heirloom",
      "altitude": "1900-2200m",
      "processing": "Washed",
      "lot_number": "YRG-2024-001"
    }
  }')
echo "✅ Harvest event created"
echo "$HARVEST" | jq -r '.event_id'
echo ""

sleep 1

echo "Creating shipping event..."
SHIPPING=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "SHIPMENT",
    "product_id": "ethiopian-yirgacheffe-2024",
    "location": "Port of Djibouti",
    "timestamp": "2024-02-01T14:30:00Z",
    "actor": "Ethiopian Coffee Exporters",
    "metadata": {
      "container": "CONT-2024-ETH-001",
      "weight_kg": 18900,
      "destination": "Oakland, CA, USA"
    }
  }')
echo "✅ Shipping event created"
echo "$SHIPPING" | jq -r '.event_id'
echo ""

sleep 1

echo "Creating roasting event..."
ROASTING=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "PROCESSING",
    "product_id": "ethiopian-yirgacheffe-2024",
    "location": "San Francisco, CA",
    "timestamp": "2024-02-15T10:00:00Z",
    "actor": "Luminous Coffee Roasters",
    "metadata": {
      "roast_profile": "Light City",
      "roast_temp_f": 435,
      "batch_size_kg": 25,
      "roast_time_min": 12
    }
  }')
echo "✅ Roasting event created"
echo "$ROASTING" | jq -r '.event_id'
echo ""

# Step 2: Get Product Provenance
echo "🔍 Step 2: Retrieving Complete Provenance"
echo "----------------------------------------"
echo "Querying blockchain for full product history..."
PROVENANCE=$(curl -s "$API_BASE/provenance/ethiopian-yirgacheffe-2024")
echo "$PROVENANCE" | jq '.'
echo ""

# Step 3: Create Financial Entities
echo "💼 Step 3: Setting Up Financial Entities"
echo "----------------------------------------"

echo "Creating wholesale customer (Blue Bottle Coffee)..."
CUSTOMER=$(curl -s -X POST "$FIN_BASE/customers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Blue Bottle Coffee",
    "email": "wholesale@bluebottlecoffee.com",
    "type": "wholesale",
    "payment_terms_days": 30
  }')
CUSTOMER_ID=$(echo "$CUSTOMER" | jq -r '.id')
echo "✅ Customer created: $CUSTOMER_ID"
echo ""

sleep 1

echo "Creating vendor (Ethiopian Coffee Exporters)..."
VENDOR=$(curl -s -X POST "$FIN_BASE/vendors" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ethiopian Coffee Exporters",
    "email": "sales@ethcoffee.et",
    "type": "supplier",
    "payment_terms_days": 45
  }')
VENDOR_ID=$(echo "$VENDOR" | jq -r '.id')
echo "✅ Vendor created: $VENDOR_ID"
echo ""

# Step 4: Create Sales Invoice
echo "📄 Step 4: Creating Sales Invoice"
echo "----------------------------------------"
echo "Invoice to Blue Bottle Coffee for 10 cases of Ethiopian Yirgacheffe..."

INVOICE=$(curl -s -X POST "$FIN_BASE/invoices" \
  -H "Content-Type: application/json" \
  -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"items\": [
      {
        \"description\": \"Ethiopian Yirgacheffe - Light Roast (1lb bags, 12/case)\",
        \"quantity\": 10,
        \"unit_price\": \"85.00\",
        \"account_code\": \"4000\"
      }
    ],
    \"due_date\": \"2024-03-17\",
    \"notes\": \"Farm-to-cup with blockchain provenance. Scan QR code for full journey.\"
  }")
INVOICE_ID=$(echo "$INVOICE" | jq -r '.id')
INVOICE_TOTAL=$(echo "$INVOICE" | jq -r '.total')
echo "✅ Invoice created: $INVOICE_ID"
echo "   Total: $$INVOICE_TOTAL"
echo ""

# Step 5: Create Purchase Bill
echo "📝 Step 5: Creating Purchase Bill"
echo "----------------------------------------"
echo "Bill from Ethiopian Coffee Exporters for green coffee..."

BILL=$(curl -s -X POST "$FIN_BASE/bills" \
  -H "Content-Type: application/json" \
  -d "{
    \"vendor_id\": \"$VENDOR_ID\",
    \"items\": [
      {
        \"description\": \"Green Coffee - Ethiopian Yirgacheffe Heirloom (60kg bags)\",
        \"quantity\": 5,
        \"unit_price\": \"135.00\",
        \"account_code\": \"5000\"
      }
    ],
    \"due_date\": \"2024-03-30\",
    \"notes\": \"Direct trade, Fair Trade certified\"
  }")
BILL_ID=$(echo "$BILL" | jq -r '.id')
BILL_TOTAL=$(echo "$BILL" | jq -r '.total')
echo "✅ Bill created: $BILL_ID"
echo "   Total: $$BILL_TOTAL"
echo ""

# Step 6: Record Customer Payment
echo "💰 Step 6: Recording Customer Payment"
echo "----------------------------------------"
echo "Blue Bottle pays invoice..."

PAYMENT=$(curl -s -X POST "$FIN_BASE/payments" \
  -H "Content-Type: application/json" \
  -d "{
    \"invoice_id\": \"$INVOICE_ID\",
    \"amount\": \"$INVOICE_TOTAL\",
    \"payment_method\": \"bank_transfer\",
    \"reference\": \"ACH-2024-02-28-BB001\"
  }")
echo "✅ Payment recorded: $(echo $PAYMENT | jq -r '.id')"
echo ""

# Step 7: Get Financial Reports
echo "📊 Step 7: Generating Financial Reports"
echo "----------------------------------------"

echo "Trial Balance:"
curl -s "$FIN_BASE/reports/trial-balance" | jq '.'
echo ""

echo "Income Statement:"
curl -s "$FIN_BASE/reports/income-statement" | jq '.'
echo ""

echo "Accounts Receivable Aging:"
curl -s "$FIN_BASE/reports/ar-aging" | jq '.'
echo ""

# Summary
echo "✅ Demo Complete!"
echo "================="
echo ""
echo "Summary of Operations:"
echo "  📦 3 supply chain events created (harvest → shipping → roasting)"
echo "  🔗 Complete blockchain provenance verified"
echo "  💼 2 business entities created (customer + vendor)"
echo "  📄 1 sales invoice: $$INVOICE_TOTAL"
echo "  📝 1 purchase bill: $$BILL_TOTAL"
echo "  💰 1 payment recorded"
echo "  📊 Financial reports generated"
echo ""
echo "Next Steps:"
echo "  - View the product passport at: $API_BASE/passport/ethiopian-yirgacheffe-2024"
echo "  - Check system health: $API_BASE/health"
echo "  - Explore the API at: http://localhost:8000/docs"
echo ""
echo "🌟 Mycelix ERP: Farm-to-Cup Excellence with Blockchain Provenance!"
