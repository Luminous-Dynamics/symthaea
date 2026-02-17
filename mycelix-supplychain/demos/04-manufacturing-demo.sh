#!/usr/bin/env bash
# Mycelix ERP Demo: Precision Parts Manufacturing
# Scenario: Job Shop with Work Orders and Material Tracking

set -e

API_BASE="http://localhost:8000/v1"
FIN_BASE="$API_BASE/fin"

echo "🏭 Mycelix ERP Demo: Precision Parts Manufacturing"
echo "===================================================="
echo ""
echo "Scenario: Custom CNC Job Shop"
echo "This demo shows:"
echo "  - Work order tracking (raw materials → finished goods)"
echo "  - Bill of materials (BOM) management"
echo "  - Job costing and profitability"
echo "  - Quality control checkpoints"
echo ""

if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
    echo "❌ Service not running"
    exit 1
fi

echo "✅ Service is running"
echo ""

# Step 1: Create Customer Order
echo "📞 Step 1: Receiving Customer Order"
echo "----------------------------------------"

echo "Creating customer (Tesla Manufacturing)..."
CUSTOMER=$(curl -s -X POST "$FIN_BASE/customers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Tesla Manufacturing",
    "email": "procurement@tesla.com",
    "type": "oem",
    "payment_terms_days": 45
  }')
CUSTOMER_ID=$(echo "$CUSTOMER" | jq -r '.id')
echo "✅ Customer: Tesla Manufacturing ($CUSTOMER_ID)"
echo ""

# Step 2: Material Receipt
echo "📦 Step 2: Receiving Raw Materials"
echo "----------------------------------------"

echo "Receiving aluminum stock from supplier..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "MATERIAL_RECEIPT",
    "product_id": "aluminum-6061-bar-stock",
    "location": "Warehouse - Raw Materials",
    "timestamp": "2024-02-01T08:00:00Z",
    "actor": "Receiving Department",
    "metadata": {
      "po_number": "PO-2024-0125",
      "quantity": 50,
      "unit": "bars (12ft x 1in diameter)",
      "supplier": "Metal Supermarkets",
      "lot_number": "AL6061-2024-Q1-001",
      "cert_number": "MTR-2024-001"
    }
  }' > /dev/null
echo "✅ 50 aluminum bars received (12ft x 1in)"
echo ""

sleep 1

# Step 3: Create Work Order
echo "🔧 Step 3: Creating Work Order"
echo "----------------------------------------"

echo "Work Order: Custom Mounting Brackets (500 units)..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "WORK_ORDER_START",
    "product_id": "mounting-bracket-tesla-rev3",
    "location": "Production Floor - CNC Area",
    "timestamp": "2024-02-05T07:00:00Z",
    "actor": "Production Scheduler",
    "metadata": {
      "wo_number": "WO-2024-0042",
      "customer": "Tesla Manufacturing",
      "quantity": 500,
      "due_date": "2024-02-20",
      "priority": "high",
      "material": "aluminum-6061-bar-stock",
      "material_qty_per_unit": 0.15,
      "estimated_hours": 120
    }
  }' > /dev/null
echo "✅ Work Order WO-2024-0042 created"
echo "   Part: Mounting Bracket (Tesla Rev 3)"
echo "   Quantity: 500 units"
echo "   Material: 75 bars of aluminum (500 × 0.15)"
echo ""

sleep 1

# Step 4: Production Stages
echo "⚙️  Step 4: Recording Production Stages"
echo "----------------------------------------"

echo "Stage 1: CNC Machining..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "PROCESSING",
    "product_id": "mounting-bracket-tesla-rev3",
    "location": "CNC Machine #3",
    "timestamp": "2024-02-06T09:00:00Z",
    "actor": "CNC Operator - Mike Johnson",
    "metadata": {
      "wo_number": "WO-2024-0042",
      "operation": "CNC Milling",
      "units_completed": 250,
      "machine_hours": 60,
      "operator_hours": 60,
      "status": "in_progress"
    }
  }' > /dev/null
echo "✅ 250 units machined (50% complete)"
echo ""

sleep 1

echo "Stage 2: Quality Control Inspection..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "QUALITY_CHECK",
    "product_id": "mounting-bracket-tesla-rev3",
    "location": "QC Station #1",
    "timestamp": "2024-02-07T14:00:00Z",
    "actor": "QC Inspector - Sarah Lee",
    "metadata": {
      "wo_number": "WO-2024-0042",
      "inspection_type": "dimensional",
      "sample_size": 25,
      "passed": 25,
      "failed": 0,
      "measurements": {
        "tolerance_spec": "±0.001in",
        "actual_variance": "±0.0005in"
      },
      "status": "approved"
    }
  }' > /dev/null
echo "✅ QC inspection passed (25/25 samples within spec)"
echo ""

sleep 1

echo "Stage 3: Final Machining..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "PROCESSING",
    "product_id": "mounting-bracket-tesla-rev3",
    "location": "CNC Machine #3",
    "timestamp": "2024-02-08T09:00:00Z",
    "actor": "CNC Operator - Mike Johnson",
    "metadata": {
      "wo_number": "WO-2024-0042",
      "operation": "CNC Milling",
      "units_completed": 500,
      "machine_hours": 60,
      "operator_hours": 60,
      "status": "completed"
    }
  }' > /dev/null
echo "✅ All 500 units machined (100% complete)"
echo ""

sleep 1

echo "Stage 4: Final QC and Packaging..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "QUALITY_CHECK",
    "product_id": "mounting-bracket-tesla-rev3",
    "location": "Final Inspection",
    "timestamp": "2024-02-09T11:00:00Z",
    "actor": "QC Inspector - Sarah Lee",
    "metadata": {
      "wo_number": "WO-2024-0042",
      "inspection_type": "final",
      "quantity_inspected": 500,
      "passed": 498,
      "failed": 2,
      "rework": 0,
      "yield_rate": "99.6%",
      "status": "approved_for_shipment"
    }
  }' > /dev/null
echo "✅ Final QC complete: 498/500 passed (99.6% yield)"
echo ""

# Step 5: Complete Work Order
echo "✅ Step 5: Completing Work Order"
echo "----------------------------------------"

curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "WORK_ORDER_COMPLETE",
    "product_id": "mounting-bracket-tesla-rev3",
    "location": "Finished Goods Warehouse",
    "timestamp": "2024-02-09T16:00:00Z",
    "actor": "Production Manager",
    "metadata": {
      "wo_number": "WO-2024-0042",
      "units_produced": 498,
      "units_scrapped": 2,
      "actual_hours": 125,
      "estimated_hours": 120,
      "variance_hours": 5,
      "material_used_bars": 75,
      "status": "closed"
    }
  }' > /dev/null
echo "✅ Work Order WO-2024-0042 closed"
echo "   Completed: 498 units (target: 500)"
echo "   Labor: 125 hours (est: 120 hours)"
echo "   Material: 75 aluminum bars"
echo ""

# Step 6: Create Material Vendor Bill
echo "📝 Step 6: Recording Material Costs"
echo "----------------------------------------"

echo "Creating vendor (Metal Supermarkets)..."
VENDOR=$(curl -s -X POST "$FIN_BASE/vendors" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Metal Supermarkets",
    "email": "invoices@metalsupermarkets.com",
    "type": "supplier",
    "payment_terms_days": 30
  }')
VENDOR_ID=$(echo "$VENDOR" | jq -r '.id')
echo "✅ Vendor: Metal Supermarkets ($VENDOR_ID)"
echo ""

sleep 1

echo "Recording material bill..."
MATERIAL_BILL=$(curl -s -X POST "$FIN_BASE/bills" \
  -H "Content-Type: application/json" \
  -d "{
    \"vendor_id\": \"$VENDOR_ID\",
    \"items\": [
      {
        \"description\": \"Aluminum 6061 Bar Stock (12ft x 1in) - 50 bars\",
        \"quantity\": 1,
        \"unit_price\": \"3500.00\",
        \"account_code\": \"5100\"
      }
    ],
    \"due_date\": \"2024-03-02\",
    \"notes\": \"PO-2024-0125. Material for WO-2024-0042\"
  }")
MATERIAL_COST=$(echo "$MATERIAL_BILL" | jq -r '.total')
echo "✅ Material cost: $$MATERIAL_COST"
echo ""

# Step 7: Create Customer Invoice
echo "💰 Step 7: Invoicing Customer"
echo "----------------------------------------"

echo "Creating invoice for completed work order..."
INVOICE=$(curl -s -X POST "$FIN_BASE/invoices" \
  -H "Content-Type: application/json" \
  -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"items\": [
      {
        \"description\": \"Custom Mounting Brackets - Tesla Rev 3 (500 units @ \$45/unit)\",
        \"quantity\": 1,
        \"unit_price\": \"22500.00\",
        \"account_code\": \"4000\"
      },
      {
        \"description\": \"Engineering/Setup Fee\",
        \"quantity\": 1,
        \"unit_price\": \"1500.00\",
        \"account_code\": \"4100\"
      },
      {
        \"description\": \"Expedite Fee (Priority Production)\",
        \"quantity\": 1,
        \"unit_price\": \"750.00\",
        \"account_code\": \"4100\"
      }
    ],
    \"due_date\": \"2024-03-25\",
    \"notes\": \"Work Order #WO-2024-0042. 498 units delivered (2 units credit). Net 45 terms.\"
  }")
INVOICE_ID=$(echo "$INVOICE" | jq -r '.id')
INVOICE_TOTAL=$(echo "$INVOICE" | jq -r '.total')
echo "✅ Invoice created: $INVOICE_ID"
echo "   Total: $$INVOICE_TOTAL"
echo ""

# Step 8: Job Costing Analysis
echo "📊 Step 8: Job Profitability Analysis"
echo "----------------------------------------"

LABOR_COST=$(echo "125 * 65" | bc)  # 125 hours × $65/hr shop rate
MATERIAL_PER_UNIT=$(echo "scale=2; $MATERIAL_COST / 500" | bc)
TOTAL_COGS=$(echo "$MATERIAL_COST + $LABOR_COST" | bc)
GROSS_PROFIT=$(echo "$INVOICE_TOTAL - $TOTAL_COGS" | bc)
MARGIN=$(echo "scale=1; ($GROSS_PROFIT / $INVOICE_TOTAL) * 100" | bc)

echo "Job Costing Summary for WO-2024-0042:"
echo "  Revenue: $$INVOICE_TOTAL"
echo "  Material Cost: $$MATERIAL_COST ($$MATERIAL_PER_UNIT per unit)"
echo "  Labor Cost: $$LABOR_COST (125 hrs @ \$65/hr)"
echo "  Total COGS: $$TOTAL_COGS"
echo "  Gross Profit: $$GROSS_PROFIT"
echo "  Gross Margin: ${MARGIN}%"
echo ""

# Step 9: Get Production History
echo "🔍 Step 9: Production Audit Trail"
echo "----------------------------------------"

echo "Complete production history:"
curl -s "$API_BASE/provenance/mounting-bracket-tesla-rev3" | \
  jq -r '.events[] | "\(.timestamp) | \(.event_type) | \(.actor) | \(.metadata.status // "N/A")"'
echo ""

# Summary
echo "✅ Demo Complete!"
echo "================="
echo ""
echo "Summary of Operations:"
echo "  📦 50 aluminum bars received from supplier"
echo "  🔧 Work Order created for 500 mounting brackets"
echo "  ⚙️  Production stages tracked (machining → QC → finishing)"
echo "  ✅ 498 units completed (99.6% yield)"
echo "  💰 Job costing calculated:"
echo "     - Revenue: $$INVOICE_TOTAL"
echo "     - COGS: $$TOTAL_COGS"
echo "     - Profit: $$GROSS_PROFIT (${MARGIN}% margin)"
echo "  📊 Complete audit trail maintained"
echo ""
echo "Production Metrics:"
echo "  - Yield: 99.6% (498/500)"
echo "  - Labor efficiency: 104% (125/120 hours)"
echo "  - Quality: 100% of samples passed inspection"
echo "  - On-time delivery: YES (completed 2/9, due 2/20)"
echo ""
echo "🏭 Mycelix ERP: Manufacturing Excellence with Complete Traceability!"
