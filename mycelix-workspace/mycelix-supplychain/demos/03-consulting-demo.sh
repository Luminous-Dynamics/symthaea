#!/usr/bin/env bash
# Mycelix ERP Demo: Luminous Consulting Group
# Scenario: Professional Services with Time Tracking

set -e

API_BASE="http://localhost:8000/v1"
FIN_BASE="$API_BASE/fin"

echo "💼 Mycelix ERP Demo: Luminous Consulting Group"
echo "==============================================="
echo ""
echo "Scenario: Professional Services Firm"
echo "This demo shows:"
echo "  - Project-based billing"
echo "  - Time & expense tracking"
echo "  - Retainer management"
echo "  - Multi-consultant allocation"
echo ""

if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
    echo "❌ Service not running"
    exit 1
fi

echo "✅ Service is running"
echo ""

# Step 1: Create Clients
echo "👥 Step 1: Creating Client Accounts"
echo "----------------------------------------"

echo "Creating retainer client (Acme Corp)..."
CLIENT1=$(curl -s -X POST "$FIN_BASE/customers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Acme Corporation",
    "email": "billing@acmecorp.com",
    "type": "corporate",
    "payment_terms_days": 15
  }')
CLIENT1_ID=$(echo "$CLIENT1" | jq -r '.id')
echo "✅ Client created: Acme Corp ($CLIENT1_ID)"
echo ""

sleep 1

echo "Creating project client (StartupXYZ)..."
CLIENT2=$(curl -s -X POST "$FIN_BASE/customers" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "StartupXYZ Inc",
    "email": "finance@startupxyz.io",
    "type": "startup",
    "payment_terms_days": 30
  }')
CLIENT2_ID=$(echo "$CLIENT2" | jq -r '.id')
echo "✅ Client created: StartupXYZ ($CLIENT2_ID)"
echo ""

# Step 2: Create Consultants as "Resources"
echo "👨‍💼 Step 2: Recording Consultant Resources"
echo "----------------------------------------"

echo "Recording Senior Consultant hours..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "SERVICE_DELIVERY",
    "product_id": "consulting-hours-senior",
    "location": "Client Site - Acme Corp",
    "timestamp": "2024-02-05T09:00:00Z",
    "actor": "Dr. Elena Rodriguez (Senior Consultant)",
    "metadata": {
      "project": "Digital Transformation Strategy",
      "hours": 8,
      "rate_per_hour": "250.00",
      "billable": true,
      "client": "Acme Corporation"
    }
  }' > /dev/null
echo "✅ 8 hours logged - Dr. Elena Rodriguez"
echo ""

sleep 1

echo "Recording Junior Consultant hours..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "SERVICE_DELIVERY",
    "product_id": "consulting-hours-junior",
    "location": "Remote",
    "timestamp": "2024-02-05T09:00:00Z",
    "actor": "Marcus Chen (Junior Consultant)",
    "metadata": {
      "project": "API Integration - StartupXYZ",
      "hours": 16,
      "rate_per_hour": "150.00",
      "billable": true,
      "client": "StartupXYZ Inc"
    }
  }' > /dev/null
echo "✅ 16 hours logged - Marcus Chen"
echo ""

# Step 3: Create Retainer Invoice
echo "📄 Step 3: Creating Retainer Invoice"
echo "----------------------------------------"
echo "Monthly retainer for Acme Corp..."

RETAINER=$(curl -s -X POST "$FIN_BASE/invoices" \
  -H "Content-Type: application/json" \
  -d "{
    \"customer_id\": \"$CLIENT1_ID\",
    \"items\": [
      {
        \"description\": \"Monthly Retainer - Strategic Advisory (40 hours)\",
        \"quantity\": 1,
        \"unit_price\": \"9000.00\",
        \"account_code\": \"4200\"
      },
      {
        \"description\": \"Additional Hours - Digital Transformation (8 hours @ \$250/hr)\",
        \"quantity\": 1,
        \"unit_price\": \"2000.00\",
        \"account_code\": \"4200\"
      }
    ],
    \"due_date\": \"2024-02-20\",
    \"notes\": \"Invoice #2024-02. Net 15 terms. Retainer covers standard advisory services.\"
  }")
RETAINER_ID=$(echo "$RETAINER" | jq -r '.id')
RETAINER_TOTAL=$(echo "$RETAINER" | jq -r '.total')
echo "✅ Retainer invoice: $RETAINER_ID"
echo "   Total: $$RETAINER_TOTAL"
echo ""

# Step 4: Create Project-Based Invoice
echo "📋 Step 4: Creating Project Invoice"
echo "----------------------------------------"
echo "Time & materials invoice for StartupXYZ..."

PROJECT_INV=$(curl -s -X POST "$FIN_BASE/invoices" \
  -H "Content-Type: application/json" \
  -d "{
    \"customer_id\": \"$CLIENT2_ID\",
    \"items\": [
      {
        \"description\": \"API Integration Consulting (16 hours @ \$150/hr)\",
        \"quantity\": 1,
        \"unit_price\": \"2400.00\",
        \"account_code\": \"4200\"
      },
      {
        \"description\": \"Project Management (4 hours @ \$175/hr)\",
        \"quantity\": 1,
        \"unit_price\": \"700.00\",
        \"account_code\": \"4200\"
      },
      {
        \"description\": \"Cloud Infrastructure Costs (AWS)\",
        \"quantity\": 1,
        \"unit_price\": \"150.00\",
        \"account_code\": \"4300\"
      }
    ],
    \"due_date\": \"2024-03-07\",
    \"notes\": \"Project: API Integration Phase 1. Time & Materials. Net 30 terms.\"
  }")
PROJECT_ID=$(echo "$PROJECT_INV" | jq -r '.id')
PROJECT_TOTAL=$(echo "$PROJECT_INV" | jq -r '.total')
echo "✅ Project invoice: $PROJECT_ID"
echo "   Total: $$PROJECT_TOTAL"
echo ""

# Step 5: Record Expenses
echo "💰 Step 5: Recording Billable Expenses"
echo "----------------------------------------"

echo "Creating vendor for expense reimbursement..."
VENDOR=$(curl -s -X POST "$FIN_BASE/vendors" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Dr. Elena Rodriguez",
    "email": "elena@luminousconsulting.com",
    "type": "consultant",
    "payment_terms_days": 7
  }')
VENDOR_ID=$(echo "$VENDOR" | jq -r '.id')
echo "✅ Consultant vendor account: $VENDOR_ID"
echo ""

sleep 1

echo "Recording expense bill (travel reimbursement)..."
EXPENSE=$(curl -s -X POST "$FIN_BASE/bills" \
  -H "Content-Type: application/json" \
  -d "{
    \"vendor_id\": \"$VENDOR_ID\",
    \"items\": [
      {
        \"description\": \"Travel to Acme Corp - Airfare\",
        \"quantity\": 1,
        \"unit_price\": \"450.00\",
        \"account_code\": \"6100\"
      },
      {
        \"description\": \"Travel to Acme Corp - Hotel (2 nights)\",
        \"quantity\": 1,
        \"unit_price\": \"350.00\",
        \"account_code\": \"6100\"
      }
    ],
    \"due_date\": \"2024-02-12\",
    \"notes\": \"Billable to Acme Corp - Digital Transformation project\"
  }")
EXPENSE_ID=$(echo "$EXPENSE" | jq -r '.id')
EXPENSE_TOTAL=$(echo "$EXPENSE" | jq -r '.total')
echo "✅ Expense bill: $EXPENSE_ID ($$EXPENSE_TOTAL)"
echo ""

# Step 6: Record Retainer Payment
echo "💳 Step 6: Recording Client Payment"
echo "----------------------------------------"
echo "Acme Corp pays retainer invoice..."

PAYMENT=$(curl -s -X POST "$FIN_BASE/payments" \
  -H "Content-Type: application/json" \
  -d "{
    \"invoice_id\": \"$RETAINER_ID\",
    \"amount\": \"$RETAINER_TOTAL\",
    \"payment_method\": \"wire_transfer\",
    \"reference\": \"WIRE-ACME-2024-02-15\"
  }")
echo "✅ Payment received: $(echo $PAYMENT | jq -r '.id')"
echo "   Amount: $$RETAINER_TOTAL"
echo ""

# Step 7: Utilization Report
echo "📊 Step 7: Consultant Utilization Report"
echo "----------------------------------------"

echo "Billable Hours Summary:"
curl -s "$API_BASE/provenance/consulting-hours-senior" | \
  jq -r '.events[] | select(.event_type == "SERVICE_DELIVERY") |
  "  \(.actor): \(.metadata.hours) hours @ $\(.metadata.rate_per_hour)/hr = $\((.metadata.hours | tonumber) * (.metadata.rate_per_hour | tonumber))"'

curl -s "$API_BASE/provenance/consulting-hours-junior" | \
  jq -r '.events[] | select(.event_type == "SERVICE_DELIVERY") |
  "  \(.actor): \(.metadata.hours) hours @ $\(.metadata.rate_per_hour)/hr = $\((.metadata.hours | tonumber) * (.metadata.rate_per_hour | tonumber))"'
echo ""

# Step 8: Financial Reports
echo "📈 Step 8: Financial Performance"
echo "----------------------------------------"

echo "Income Statement (Service Revenue):"
curl -s "$FIN_BASE/reports/income-statement" | jq '.'
echo ""

echo "AR Aging (Outstanding Invoices):"
curl -s "$FIN_BASE/reports/ar-aging" | jq '.'
echo ""

# Summary
echo "✅ Demo Complete!"
echo "================="
echo ""
echo "Summary of Operations:"
echo "  👥 2 clients created (Acme Corp + StartupXYZ)"
echo "  ⏱️  24 billable hours tracked (8 senior + 16 junior)"
echo "  📄 2 invoices generated:"
echo "     - Retainer + Overage (Acme): $$RETAINER_TOTAL (PAID)"
echo "     - Time & Materials (StartupXYZ): $$PROJECT_TOTAL (Net 30)"
echo "  💰 Expense reimbursement: $$EXPENSE_TOTAL (billable)"
echo "  📊 Utilization tracking active"
echo ""
echo "Consultant Performance:"
echo "  - Dr. Elena Rodriguez: 8 hrs @ \$250/hr = \$2,000 (billable)"
echo "  - Marcus Chen: 16 hrs @ \$150/hr = \$2,400 (billable)"
echo "  - Total billable revenue: \$$(echo "$RETAINER_TOTAL + $PROJECT_TOTAL" | bc)"
echo ""
echo "💼 Mycelix ERP: Professional Services Excellence!"
