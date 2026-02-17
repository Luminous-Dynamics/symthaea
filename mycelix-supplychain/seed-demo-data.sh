#!/usr/bin/env bash
# Seed Demo Data for Mycelix ERP
#
# Creates a complete demo scenario for a coffee roastery business

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

API_BASE="${API_BASE:-http://localhost:8080}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-mycelix_erp}"
DB_USER="${DB_USER:-postgres}"

export DATABASE_URL="postgresql://${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

echo -e "${GREEN}☕ Seeding Demo Data for Luminous Coffee Roasters${NC}"
echo "=============================================="
echo ""

# Function to make API requests
api_post() {
    local endpoint="$1"
    local data="$2"
    curl -s -X POST "${API_BASE}${endpoint}" \
        -H "Content-Type: application/json" \
        -d "$data"
}

# Create demo GL accounts (beyond the default chart of accounts)
echo -e "${YELLOW}Creating custom GL accounts...${NC}"

# Revenue - Coffee Sales
api_post "/v1/fin/accounts" '{
  "code": "4010",
  "name": "Coffee Sales - Wholesale",
  "account_type": "Revenue",
  "description": "Revenue from wholesale coffee sales"
}'

# Cost of Goods Sold
api_post "/v1/fin/accounts" '{
  "code": "5010",
  "name": "Cost of Goods Sold - Green Coffee Beans",
  "account_type": "Expense",
  "description": "Cost of purchasing green coffee beans"
}'

# Create demo customer
echo -e "${YELLOW}Creating demo customer (Artisan Cafe)...${NC}"
CUSTOMER_ID=$(psql "$DATABASE_URL" -t -c "
INSERT INTO customers (id, name, email, phone, address)
VALUES (
    gen_random_uuid(),
    'Artisan Cafe',
    'orders@artisancafe.example.com',
    '+1-555-0123',
    '123 Main St, Portland, OR 97201'
)
RETURNING id;
" | tr -d ' ')

echo "Customer ID: $CUSTOMER_ID"

# Create demo vendor
echo -e "${YELLOW}Creating demo vendor (Colombian Coffee Co)...${NC}"
VENDOR_ID=$(psql "$DATABASE_URL" -t -c "
INSERT INTO vendors (id, name, email, phone, address)
VALUES (
    gen_random_uuid(),
    'Colombian Coffee Co.',
    'sales@colombiancoffee.example.com',
    '+57-1-555-9999',
    'Bogotá, Colombia'
)
RETURNING id;
" | tr -d ' ')

echo "Vendor ID: $VENDOR_ID"

# Create demo invoice
echo -e "${YELLOW}Creating demo invoice...${NC}"
INVOICE_RESPONSE=$(api_post "/v1/fin/invoices" "{
  \"customer_id\": \"$CUSTOMER_ID\",
  \"invoice_date\": \"2025-01-15\",
  \"due_date\": \"2025-02-14\",
  \"currency\": \"USD\",
  \"line_items\": [
    {
      \"description\": \"Medium Roast Colombian Blend - 50 lbs\",
      \"quantity\": 50,
      \"unit_price\": \"12.50\",
      \"account_code\": \"4010\"
    },
    {
      \"description\": \"Dark Roast Ethiopian - 30 lbs\",
      \"quantity\": 30,
      \"unit_price\": \"15.00\",
      \"account_code\": \"4010\"
    }
  ]
}")

INVOICE_ID=$(echo "$INVOICE_RESPONSE" | jq -r '.id')
INVOICE_NUMBER=$(echo "$INVOICE_RESPONSE" | jq -r '.invoice_number')
INVOICE_TOTAL=$(echo "$INVOICE_RESPONSE" | jq -r '.total_amount')

echo "Invoice created: $INVOICE_NUMBER (Total: \$${INVOICE_TOTAL})"

# Send the invoice
echo -e "${YELLOW}Sending invoice to customer...${NC}"
api_post "/v1/fin/invoices/${INVOICE_ID}/send" "{}"
echo "✅ Invoice sent!"

# Create demo bill from vendor
echo -e "${YELLOW}Creating demo bill from vendor...${NC}"
BILL_RESPONSE=$(api_post "/v1/fin/bills" "{
  \"vendor_id\": \"$VENDOR_ID\",
  \"bill_date\": \"2025-01-10\",
  \"due_date\": \"2025-02-10\",
  \"currency\": \"USD\",
  \"line_items\": [
    {
      \"description\": \"Green Coffee Beans - Colombian Supremo - 100 kg\",
      \"quantity\": 100,
      \"unit_price\": \"8.00\",
      \"account_code\": \"5010\"
    }
  ]
}")

BILL_ID=$(echo "$BILL_RESPONSE" | jq -r '.id')
BILL_NUMBER=$(echo "$BILL_RESPONSE" | jq -r '.bill_number')
BILL_TOTAL=$(echo "$BILL_RESPONSE" | jq -r '.total_amount')

echo "Bill created: $BILL_NUMBER (Total: \$${BILL_TOTAL})"

# Approve the bill
echo -e "${YELLOW}Approving bill for payment...${NC}"
api_post "/v1/fin/bills/${BILL_ID}/approve" "{}"
echo "✅ Bill approved!"

# Record payment received from customer
echo -e "${YELLOW}Recording customer payment...${NC}"
PAYMENT_RESPONSE=$(api_post "/v1/fin/payments" "{
  \"payment_type\": \"Receivable\",
  \"payment_date\": \"2025-01-20\",
  \"amount\": \"${INVOICE_TOTAL}\",
  \"payment_method\": \"BankTransfer\",
  \"reference_number\": \"WIRE-2025-0120-001\",
  \"invoice_id\": \"$INVOICE_ID\",
  \"description\": \"Payment received via wire transfer\"
}")

echo "✅ Payment recorded: \$${INVOICE_TOTAL}"

# Record payment sent to vendor
echo -e "${YELLOW}Recording vendor payment...${NC}"
PAYMENT_RESPONSE=$(api_post "/v1/fin/payments" "{
  \"payment_type\": \"Payable\",
  \"payment_date\": \"2025-01-22\",
  \"amount\": \"${BILL_TOTAL}\",
  \"payment_method\": \"BankTransfer\",
  \"reference_number\": \"WIRE-2025-0122-001\",
  \"bill_id\": \"$BILL_ID\",
  \"description\": \"Payment to supplier via wire transfer\"
}")

echo "✅ Payment recorded: \$${BILL_TOTAL}"

echo ""
echo -e "${GREEN}🎉 Demo data seeded successfully!${NC}"
echo ""
echo "Demo Scenario Summary:"
echo "======================"
echo "  Company: Luminous Coffee Roasters"
echo "  Customer: Artisan Cafe"
echo "  Vendor: Colombian Coffee Co."
echo ""
echo "  Invoice: $INVOICE_NUMBER (\$$INVOICE_TOTAL) - PAID"
echo "  Bill: $BILL_NUMBER (\$$BILL_TOTAL) - PAID"
echo ""
echo "Test the API:"
echo "  curl ${API_BASE}/v1/fin/invoices"
echo "  curl ${API_BASE}/v1/fin/bills"
echo "  curl ${API_BASE}/v1/fin/reports/trial-balance"
echo ""
echo "View in database:"
echo "  psql \"$DATABASE_URL\" -c 'SELECT * FROM invoices;'"
echo "  psql \"$DATABASE_URL\" -c 'SELECT * FROM journal_entries;'"
echo ""
