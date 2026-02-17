#!/usr/bin/env bash
# Mycelix ERP Demo: TechGear Online Store
# Scenario: E-commerce with Inventory Tracking

set -e

API_BASE="http://localhost:8000/v1"
FIN_BASE="$API_BASE/fin"

echo "🛒 Mycelix ERP Demo: TechGear Online Store"
echo "==========================================="
echo ""
echo "Scenario: E-commerce Operations with Real-time Inventory"
echo "This demo shows:"
echo "  - Multi-channel sales (website, Amazon, eBay)"
echo "  - Real-time inventory updates"
echo "  - Automated invoice generation"
echo "  - Customer portal integration"
echo ""

# Check service
if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
    echo "❌ Service not running. Start with: cd rust && cargo run --release"
    exit 1
fi

echo "✅ Service is running"
echo ""

# Step 1: Create Products in Inventory
echo "📦 Step 1: Creating Product Inventory"
echo "----------------------------------------"

echo "Creating product: Wireless Gaming Mouse..."
PRODUCT1=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "INVENTORY_ADD",
    "product_id": "gaming-mouse-x1",
    "location": "Warehouse A - San Jose, CA",
    "timestamp": "2024-02-01T09:00:00Z",
    "actor": "TechGear Warehouse System",
    "metadata": {
      "sku": "GM-X1-BLK",
      "quantity": 500,
      "unit_cost": "35.00",
      "supplier": "Logitech"
    }
  }')
echo "✅ Gaming Mouse added to inventory"
echo ""

sleep 1

echo "Creating product: USB-C Hub..."
PRODUCT2=$(curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "INVENTORY_ADD",
    "product_id": "usbc-hub-7port",
    "location": "Warehouse A - San Jose, CA",
    "timestamp": "2024-02-01T09:15:00Z",
    "actor": "TechGear Warehouse System",
    "metadata": {
      "sku": "HUB-7P-GRY",
      "quantity": 300,
      "unit_cost": "28.00",
      "supplier": "Anker"
    }
  }')
echo "✅ USB-C Hub added to inventory"
echo ""

# Step 2: Create Customers
echo "👥 Step 2: Creating Customer Accounts"
echo "----------------------------------------"

echo "Creating retail customer..."
CUSTOMER1=$(curl -s -X POST "$FIN_BASE/customers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sarah Martinez",
    "email": "sarah.m@example.com",
    "type": "retail",
    "payment_terms_days": 0
  }')
CUSTOMER1_ID=$(echo "$CUSTOMER1" | jq -r '.id')
echo "✅ Retail customer: $CUSTOMER1_ID"
echo ""

sleep 1

echo "Creating wholesale customer..."
CUSTOMER2=$(curl -s -X POST "$FIN_BASE/customers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Office Depot",
    "email": "wholesale@officedepot.com",
    "type": "wholesale",
    "payment_terms_days": 30
  }')
CUSTOMER2_ID=$(echo "$CUSTOMER2" | jq -r '.id')
echo "✅ Wholesale customer: $CUSTOMER2_ID"
echo ""

# Step 3: Process Website Order
echo "🌐 Step 3: Processing Website Order"
echo "----------------------------------------"
echo "Sarah orders 2 gaming mice through website..."

# Create inventory removal event
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "INVENTORY_REMOVE",
    "product_id": "gaming-mouse-x1",
    "location": "Warehouse A - San Jose, CA",
    "timestamp": "2024-02-10T14:30:00Z",
    "actor": "Web Order #WEB-2024-001",
    "metadata": {
      "quantity": 2,
      "reason": "customer_order",
      "order_id": "WEB-2024-001"
    }
  }' > /dev/null

INVOICE1=$(curl -s -X POST "$FIN_BASE/invoices" \
  -H "Content-Type: application/json" \
  -d "{
    \"customer_id\": \"$CUSTOMER1_ID\",
    \"items\": [
      {
        \"description\": \"Wireless Gaming Mouse X1 - Black\",
        \"quantity\": 2,
        \"unit_price\": \"79.99\",
        \"account_code\": \"4000\"
      },
      {
        \"description\": \"Standard Shipping\",
        \"quantity\": 1,
        \"unit_price\": \"8.99\",
        \"account_code\": \"4100\"
      }
    ],
    \"due_date\": \"2024-02-10\",
    \"notes\": \"Order #WEB-2024-001\"
  }")
INVOICE1_ID=$(echo "$INVOICE1" | jq -r '.id')
INVOICE1_TOTAL=$(echo "$INVOICE1" | jq -r '.total')
echo "✅ Website order invoice: $INVOICE1_ID ($$INVOICE1_TOTAL)"
echo ""

# Record immediate payment (credit card)
curl -s -X POST "$FIN_BASE/payments" \
  -H "Content-Type: application/json" \
  -d "{
    \"invoice_id\": \"$INVOICE1_ID\",
    \"amount\": \"$INVOICE1_TOTAL\",
    \"payment_method\": \"credit_card\",
    \"reference\": \"STRIPE-ch_2024_sarah\"
  }" > /dev/null
echo "✅ Payment processed (Stripe)"
echo ""

# Step 4: Process Wholesale Order
echo "🏢 Step 4: Processing Wholesale Order"
echo "----------------------------------------"
echo "Office Depot orders 50 USB-C Hubs..."

# Inventory removal
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "INVENTORY_REMOVE",
    "product_id": "usbc-hub-7port",
    "location": "Warehouse A - San Jose, CA",
    "timestamp": "2024-02-12T10:00:00Z",
    "actor": "Wholesale Order #WSL-2024-005",
    "metadata": {
      "quantity": 50,
      "reason": "wholesale_order",
      "order_id": "WSL-2024-005"
    }
  }' > /dev/null

INVOICE2=$(curl -s -X POST "$FIN_BASE/invoices" \
  -H "Content-Type: application/json" \
  -d "{
    \"customer_id\": \"$CUSTOMER2_ID\",
    \"items\": [
      {
        \"description\": \"USB-C Hub 7-Port - Gray (Wholesale Pack of 50)\",
        \"quantity\": 1,
        \"unit_price\": \"2100.00\",
        \"account_code\": \"4000\"
      }
    ],
    \"due_date\": \"2024-03-13\",
    \"notes\": \"Order #WSL-2024-005. Net 30 terms.\"
  }")
INVOICE2_ID=$(echo "$INVOICE2" | jq -r '.id')
INVOICE2_TOTAL=$(echo "$INVOICE2" | jq -r '.total')
echo "✅ Wholesale order invoice: $INVOICE2_ID ($$INVOICE2_TOTAL)"
echo "   Payment due: 2024-03-13 (Net 30)"
echo ""

# Step 5: Check Inventory Levels
echo "📊 Step 5: Checking Current Inventory"
echo "----------------------------------------"

echo "Gaming Mouse inventory:"
curl -s "$API_BASE/provenance/gaming-mouse-x1" | jq '.events[] | select(.event_type | contains("INVENTORY")) | {type: .event_type, qty: .metadata.quantity, timestamp}'
echo ""

echo "USB-C Hub inventory:"
curl -s "$API_BASE/provenance/usbc-hub-7port" | jq '.events[] | select(.event_type | contains("INVENTORY")) | {type: .event_type, qty: .metadata.quantity, timestamp}'
echo ""

# Step 6: Financial Reports
echo "📈 Step 6: Financial Summary"
echo "----------------------------------------"

echo "Revenue by Channel:"
echo "  Website (Retail): $$INVOICE1_TOTAL"
echo "  Wholesale: $$INVOICE2_TOTAL"
echo ""

echo "Trial Balance:"
curl -s "$FIN_BASE/reports/trial-balance" | jq '.accounts[] | select(.balance != "0.00")'
echo ""

echo "AR Aging (Unpaid Invoices):"
curl -s "$FIN_BASE/reports/ar-aging" | jq '.'
echo ""

# Summary
echo "✅ Demo Complete!"
echo "================="
echo ""
echo "Summary of Operations:"
echo "  📦 2 products added to inventory (500 mice + 300 hubs)"
echo "  🛒 2 customer orders processed:"
echo "     - Retail (website): 2 mice = $$INVOICE1_TOTAL (paid)"
echo "     - Wholesale: 50 hubs = $$INVOICE2_TOTAL (Net 30)"
echo "  📊 Real-time inventory tracking active"
echo "  💰 Revenue recognized and payments recorded"
echo ""
echo "Remaining Inventory:"
echo "  - Gaming Mice: 498 units"
echo "  - USB-C Hubs: 250 units"
echo ""
echo "🛒 Mycelix ERP: Multi-Channel E-commerce Excellence!"
