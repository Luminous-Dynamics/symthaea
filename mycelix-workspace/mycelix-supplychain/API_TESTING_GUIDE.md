# 🧪 Mycelix ERP - Complete API Testing Guide

**Comprehensive testing scenarios with real curl commands**

---

## 📋 Table of Contents

1. [Quick Start Testing](#quick-start-testing)
2. [GL Accounts API](#gl-accounts-api)
3. [Journal Entries API](#journal-entries-api)
4. [Invoices API](#invoices-api)
5. [Bills API](#bills-api)
6. [Payments API](#payments-api)
7. [Financial Reports API](#financial-reports-api)
8. [Complete Business Scenarios](#complete-business-scenarios)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start Testing

### Prerequisites
```bash
# 1. Service must be running
cd rust && nix develop ../ --command cargo run

# 2. Database initialized
export FIN_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/mycelix_erp"
./init-database.sh

# 3. Set API base URL
export API_BASE="http://localhost:8080"
```

### Verify Service Health
```bash
# Basic health check
curl $API_BASE/health | jq

# Expected output:
# {
#   "status": "healthy",
#   "timestamp": "2025-12-30T18:00:00Z"
# }

# Detailed readiness check
curl $API_BASE/health/ready | jq

# Liveness probe
curl $API_BASE/health/live | jq
```

---

## GL Accounts API

### List All GL Accounts
```bash
curl $API_BASE/v1/fin/accounts | jq

# Pretty print with filtering
curl $API_BASE/v1/fin/accounts | jq '.[] | {code, name, type: .account_type}'

# Count accounts by type
curl $API_BASE/v1/fin/accounts | jq 'group_by(.account_type) | map({type: .[0].account_type, count: length})'
```

### Get Specific GL Account
```bash
# First, get an account ID
ACCOUNT_ID=$(curl -s $API_BASE/v1/fin/accounts | jq -r '.[0].id')

# Then fetch it
curl $API_BASE/v1/fin/accounts/$ACCOUNT_ID | jq
```

### Create Custom GL Account
```bash
# Revenue account
curl -X POST $API_BASE/v1/fin/accounts \
  -H "Content-Type: application/json" \
  -d '{
  "code": "4100",
  "name": "Service Revenue - Consulting",
  "account_type": "Revenue",
  "description": "Revenue from consulting services"
}' | jq

# Asset account
curl -X POST $API_BASE/v1/fin/accounts \
  -H "Content-Type: application/json" \
  -d '{
  "code": "1500",
  "name": "Office Equipment",
  "account_type": "Asset",
  "description": "Computers, furniture, and office equipment"
}' | jq

# Expense account
curl -X POST $API_BASE/v1/fin/accounts \
  -H "Content-Type: application/json" \
  -d '{
  "code": "6100",
  "name": "Marketing Expenses",
  "account_type": "Expense",
  "description": "Advertising, promotions, and marketing costs"
}' | jq
```

### Get Chart of Accounts Summary
```bash
# Group accounts by type with totals
curl $API_BASE/v1/fin/accounts | jq '
  group_by(.account_type) |
  map({
    type: .[0].account_type,
    count: length,
    accounts: map({code, name})
  })
'
```

---

## Journal Entries API

### Create Manual Journal Entry
```bash
# Get account IDs first
CASH_ID=$(curl -s $API_BASE/v1/fin/accounts | jq -r '.[] | select(.code == "1000") | .id')
EQUITY_ID=$(curl -s $API_BASE/v1/fin/accounts | jq -r '.[] | select(.code == "3000") | .id')

# Create entry (Owner investment)
curl -X POST $API_BASE/v1/fin/journal-entries \
  -H "Content-Type: application/json" \
  -d "{
  \"entry_date\": \"2025-01-01\",
  \"description\": \"Owner initial investment\",
  \"line_items\": [
    {
      \"account_id\": \"$CASH_ID\",
      \"debit_amount\": \"50000.00\",
      \"description\": \"Cash investment\"
    },
    {
      \"account_id\": \"$EQUITY_ID\",
      \"credit_amount\": \"50000.00\",
      \"description\": \"Owner equity\"
    }
  ]
}" | jq
```

### List Journal Entries
```bash
# All entries
curl $API_BASE/v1/fin/journal-entries | jq

# Only posted entries
curl $API_BASE/v1/fin/journal-entries | jq '.[] | select(.status == "Posted")'

# Entries from specific date
curl $API_BASE/v1/fin/journal-entries | jq '.[] | select(.entry_date >= "2025-01-01")'
```

### Get Journal Entry Details
```bash
# Get first entry ID
ENTRY_ID=$(curl -s $API_BASE/v1/fin/journal-entries | jq -r '.[0].id')

# Fetch full details
curl $API_BASE/v1/fin/journal-entries/$ENTRY_ID | jq

# Verify debits = credits
curl $API_BASE/v1/fin/journal-entries/$ENTRY_ID | jq '
  .line_items |
  {
    total_debits: (map(.debit_amount // 0) | add),
    total_credits: (map(.credit_amount // 0) | add),
    balanced: ((map(.debit_amount // 0) | add) == (map(.credit_amount // 0) | add))
  }
'
```

### Post Journal Entry
```bash
# Post (finalize) an entry
DRAFT_ID=$(curl -s $API_BASE/v1/fin/journal-entries | jq -r '.[] | select(.status == "Draft") | .id | select(. != null)' | head -1)

curl -X POST $API_BASE/v1/fin/journal-entries/$DRAFT_ID/post | jq
```

---

## Invoices API

### Create Customer Invoice
```bash
# First, get or create a customer (via seed script or manually in DB)
# For demo, we'll use the customer from seed data

# Get customer ID
CUSTOMER_ID=$(psql $FIN_DATABASE_URL -t -c "SELECT id FROM customers LIMIT 1;" | tr -d ' ')

# Create invoice
curl -X POST $API_BASE/v1/fin/invoices \
  -H "Content-Type: application/json" \
  -d "{
  \"customer_id\": \"$CUSTOMER_ID\",
  \"invoice_date\": \"2025-01-15\",
  \"due_date\": \"2025-02-14\",
  \"currency\": \"USD\",
  \"line_items\": [
    {
      \"description\": \"Web Development - 20 hours\",
      \"quantity\": 20,
      \"unit_price\": \"150.00\",
      \"account_code\": \"4000\"
    },
    {
      \"description\": \"Server Hosting - January\",
      \"quantity\": 1,
      \"unit_price\": \"99.00\",
      \"account_code\": \"4000\"
    }
  ]
}" | jq
```

### List All Invoices
```bash
# All invoices
curl $API_BASE/v1/fin/invoices | jq

# Pretty summary
curl $API_BASE/v1/fin/invoices | jq '.[] | {
  number: .invoice_number,
  date: .invoice_date,
  total: .total_amount,
  status: .status
}'

# Only unpaid invoices
curl $API_BASE/v1/fin/invoices | jq '.[] | select(.status != "Paid")'

# Total outstanding receivables
curl $API_BASE/v1/fin/invoices | jq '[.[] | select(.status != "Paid") | .total_amount] | add'
```

### Get Invoice Details
```bash
# Get specific invoice
INVOICE_ID=$(curl -s $API_BASE/v1/fin/invoices | jq -r '.[0].id')
curl $API_BASE/v1/fin/invoices/$INVOICE_ID | jq

# Calculate tax percentage
curl $API_BASE/v1/fin/invoices/$INVOICE_ID | jq '
  {
    subtotal,
    tax_amount,
    total_amount,
    tax_rate: ((.tax_amount / .subtotal) * 100 | round)
  }
'
```

### Send Invoice to Customer
```bash
# Mark invoice as sent
INVOICE_ID=$(curl -s $API_BASE/v1/fin/invoices | jq -r '.[] | select(.status == "Draft") | .id | select(. != null)' | head -1)

curl -X POST $API_BASE/v1/fin/invoices/$INVOICE_ID/send | jq
```

---

## Bills API

### Create Vendor Bill
```bash
# Get vendor ID (from seed data or create manually)
VENDOR_ID=$(psql $FIN_DATABASE_URL -t -c "SELECT id FROM vendors LIMIT 1;" | tr -d ' ')

# Create bill
curl -X POST $API_BASE/v1/fin/bills \
  -H "Content-Type: application/json" \
  -d "{
  \"vendor_id\": \"$VENDOR_ID\",
  \"bill_date\": \"2025-01-10\",
  \"due_date\": \"2025-02-10\",
  \"currency\": \"USD\",
  \"line_items\": [
    {
      \"description\": \"Office Supplies - Bulk Order\",
      \"quantity\": 1,
      \"unit_price\": \"450.00\",
      \"account_code\": \"5000\"
    },
    {
      \"description\": \"Software Licenses - Annual\",
      \"quantity\": 5,
      \"unit_price\": \"199.00\",
      \"account_code\": \"5000\"
    }
  ]
}" | jq
```

### List All Bills
```bash
# All bills
curl $API_BASE/v1/fin/bills | jq

# Unpaid bills
curl $API_BASE/v1/fin/bills | jq '.[] | select(.status != "Paid")'

# Bills due soon (next 7 days)
curl $API_BASE/v1/fin/bills | jq --arg future_date "$(date -d '+7 days' +%Y-%m-%d)" '
  .[] | select(.due_date <= $future_date and .status != "Paid")
'

# Total accounts payable
curl $API_BASE/v1/fin/bills | jq '[.[] | select(.status != "Paid") | .total_amount] | add'
```

### Get Bill Details
```bash
BILL_ID=$(curl -s $API_BASE/v1/fin/bills | jq -r '.[0].id')
curl $API_BASE/v1/fin/bills/$BILL_ID | jq
```

### Approve Bill for Payment
```bash
# Approve a draft bill
BILL_ID=$(curl -s $API_BASE/v1/fin/bills | jq -r '.[] | select(.status == "Draft") | .id | select(. != null)' | head -1)

curl -X POST $API_BASE/v1/fin/bills/$BILL_ID/approve | jq
```

---

## Payments API

### Record Customer Payment (Receivable)
```bash
# Get an unpaid invoice
INVOICE_ID=$(curl -s $API_BASE/v1/fin/invoices | jq -r '.[] | select(.status == "Sent") | .id | select(. != null)' | head -1)
INVOICE_TOTAL=$(curl -s $API_BASE/v1/fin/invoices/$INVOICE_ID | jq -r '.total_amount')

# Record payment
curl -X POST $API_BASE/v1/fin/payments \
  -H "Content-Type: application/json" \
  -d "{
  \"payment_type\": \"Receivable\",
  \"payment_date\": \"$(date +%Y-%m-%d)\",
  \"amount\": \"$INVOICE_TOTAL\",
  \"payment_method\": \"BankTransfer\",
  \"reference_number\": \"WIRE-$(date +%Y%m%d)-001\",
  \"invoice_id\": \"$INVOICE_ID\",
  \"description\": \"Payment received via wire transfer\"
}" | jq
```

### Record Vendor Payment (Payable)
```bash
# Get an approved bill
BILL_ID=$(curl -s $API_BASE/v1/fin/bills | jq -r '.[] | select(.status == "Approved") | .id | select(. != null)' | head -1)
BILL_TOTAL=$(curl -s $API_BASE/v1/fin/bills/$BILL_ID | jq -r '.total_amount')

# Record payment
curl -X POST $API_BASE/v1/fin/payments \
  -H "Content-Type: application/json" \
  -d "{
  \"payment_type\": \"Payable\",
  \"payment_date\": \"$(date +%Y-%m-%d)\",
  \"amount\": \"$BILL_TOTAL\",
  \"payment_method\": \"Check\",
  \"reference_number\": \"CHK-1001\",
  \"bill_id\": \"$BILL_ID\",
  \"description\": \"Payment to vendor via check #1001\"
}" | jq
```

### List All Payments
```bash
# All payments
curl $API_BASE/v1/fin/payments | jq

# Payments by type
curl $API_BASE/v1/fin/payments | jq 'group_by(.payment_type) | map({type: .[0].payment_type, count: length, total: (map(.amount) | add)})'

# Payments by method
curl $API_BASE/v1/fin/payments | jq 'group_by(.payment_method) | map({method: .[0].payment_method, count: length, total: (map(.amount) | add)})'

# Recent payments (last 7 days)
curl $API_BASE/v1/fin/payments | jq --arg week_ago "$(date -d '-7 days' +%Y-%m-%d)" '
  .[] | select(.payment_date >= $week_ago)
'
```

---

## Financial Reports API

### Trial Balance
```bash
# Get trial balance
curl $API_BASE/v1/fin/reports/trial-balance | jq

# Verify balance (debits should equal credits)
curl $API_BASE/v1/fin/reports/trial-balance | jq '
  {
    total_debits: (map(.debit_balance // 0) | add),
    total_credits: (map(.credit_balance // 0) | add),
    balanced: ((map(.debit_balance // 0) | add) == (map(.credit_balance // 0) | add))
  }
'

# Summary by account type
curl $API_BASE/v1/fin/reports/trial-balance | jq '
  group_by(.account_type) |
  map({
    type: .[0].account_type,
    debit_total: (map(.debit_balance // 0) | add),
    credit_total: (map(.credit_balance // 0) | add)
  })
'
```

### Income Statement
```bash
# Get income statement
curl $API_BASE/v1/fin/reports/income-statement | jq

# Calculate key metrics
curl $API_BASE/v1/fin/reports/income-statement | jq '
  {
    total_revenue: .total_revenue,
    total_expenses: .total_expenses,
    net_income: .net_income,
    profit_margin: ((.net_income / .total_revenue) * 100 | round)
  }
'
```

### Balance Sheet
```bash
# Get balance sheet
curl $API_BASE/v1/fin/reports/balance-sheet | jq

# Verify accounting equation (Assets = Liabilities + Equity)
curl $API_BASE/v1/fin/reports/balance-sheet | jq '
  {
    total_assets: .total_assets,
    total_liabilities: .total_liabilities,
    total_equity: .total_equity,
    sum_liabilities_equity: (.total_liabilities + .total_equity),
    balanced: (.total_assets == (.total_liabilities + .total_equity))
  }
'
```

---

## Complete Business Scenarios

### Scenario 1: Freelance Consulting Business

```bash
# 1. Create customer
CUSTOMER_ID=$(psql $FIN_DATABASE_URL -t -c "
INSERT INTO customers (id, name, email, phone, address)
VALUES (gen_random_uuid(), 'Tech Startup Inc.', 'finance@techstartup.example.com', '+1-555-0199', '456 Innovation Way, San Francisco, CA 94105')
RETURNING id;
" | tr -d ' ')

# 2. Create invoice for consulting services
INVOICE_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/invoices \
  -H "Content-Type: application/json" \
  -d "{
  \"customer_id\": \"$CUSTOMER_ID\",
  \"invoice_date\": \"2025-01-20\",
  \"due_date\": \"2025-02-20\",
  \"currency\": \"USD\",
  \"line_items\": [
    {\"description\": \"Software Architecture Consulting - 40 hours\", \"quantity\": 40, \"unit_price\": \"200.00\", \"account_code\": \"4000\"},
    {\"description\": \"Code Review Services - 10 hours\", \"quantity\": 10, \"unit_price\": \"150.00\", \"account_code\": \"4000\"}
  ]
}")

INVOICE_ID=$(echo "$INVOICE_RESPONSE" | jq -r '.id')
INVOICE_TOTAL=$(echo "$INVOICE_RESPONSE" | jq -r '.total_amount')
echo "Invoice created: $INVOICE_TOTAL"

# 3. Send invoice
curl -s -X POST $API_BASE/v1/fin/invoices/$INVOICE_ID/send | jq -r '.status'

# 4. Record payment (30 days later)
curl -s -X POST $API_BASE/v1/fin/payments \
  -H "Content-Type: application/json" \
  -d "{
  \"payment_type\": \"Receivable\",
  \"payment_date\": \"2025-02-19\",
  \"amount\": \"$INVOICE_TOTAL\",
  \"payment_method\": \"BankTransfer\",
  \"reference_number\": \"ACH-20250219-001\",
  \"invoice_id\": \"$INVOICE_ID\",
  \"description\": \"Payment via ACH transfer\"
}" | jq -r '.id'

echo "✅ Consulting scenario complete!"
```

### Scenario 2: E-commerce Business

```bash
# 1. Record cost of goods sold (purchased inventory)
VENDOR_ID=$(psql $FIN_DATABASE_URL -t -c "
INSERT INTO vendors (id, name, email, phone, address)
VALUES (gen_random_uuid(), 'Wholesale Supplier Co.', 'sales@supplier.example.com', '+1-555-0288', 'Los Angeles, CA')
RETURNING id;
" | tr -d ' ')

# Create bill for inventory purchase
BILL_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/bills \
  -H "Content-Type: application/json" \
  -d "{
  \"vendor_id\": \"$VENDOR_ID\",
  \"bill_date\": \"2025-01-05\",
  \"due_date\": \"2025-02-05\",
  \"currency\": \"USD\",
  \"line_items\": [
    {\"description\": \"Product Inventory - 100 units\", \"quantity\": 100, \"unit_price\": \"25.00\", \"account_code\": \"5010\"}
  ]
}")

BILL_ID=$(echo "$BILL_RESPONSE" | jq -r '.id')

# 2. Approve and pay bill
curl -s -X POST $API_BASE/v1/fin/bills/$BILL_ID/approve > /dev/null
curl -s -X POST $API_BASE/v1/fin/payments \
  -H "Content-Type: application/json" \
  -d "{
  \"payment_type\": \"Payable\",
  \"payment_date\": \"2025-01-15\",
  \"amount\": \"2500.00\",
  \"payment_method\": \"BankTransfer\",
  \"reference_number\": \"WIRE-20250115\",
  \"bill_id\": \"$BILL_ID\",
  \"description\": \"Inventory payment\"
}" > /dev/null

# 3. Create sales invoices (sell inventory)
for i in {1..5}; do
  CUSTOMER_ID=$(psql $FIN_DATABASE_URL -t -c "INSERT INTO customers (id, name, email) VALUES (gen_random_uuid(), 'Customer $i', 'customer$i@example.com') RETURNING id;" | tr -d ' ')

  curl -s -X POST $API_BASE/v1/fin/invoices \
    -H "Content-Type: application/json" \
    -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"invoice_date\": \"2025-01-$(printf '%02d' $((10 + i)))\",
    \"due_date\": \"2025-02-$(printf '%02d' $((10 + i)))\",
    \"currency\": \"USD\",
    \"line_items\": [{\"description\": \"Product Sale\", \"quantity\": 20, \"unit_price\": \"49.99\", \"account_code\": \"4000\"}]
  }" > /dev/null
done

echo "✅ E-commerce scenario complete! 5 customer invoices created"

# 4. Check profitability
REVENUE=$(curl -s $API_BASE/v1/fin/reports/income-statement | jq -r '.total_revenue')
EXPENSES=$(curl -s $API_BASE/v1/fin/reports/income-statement | jq -r '.total_expenses')
echo "Revenue: \$$REVENUE | Expenses: \$$EXPENSES | Profit: \$$(echo "$REVENUE - $EXPENSES" | bc)"
```

### Scenario 3: Monthly Recurring Revenue (SaaS)

```bash
# Create monthly subscription invoices
for month in {1..3}; do
  MONTH_STR=$(printf '%02d' $month)

  curl -s -X POST $API_BASE/v1/fin/invoices \
    -H "Content-Type: application/json" \
    -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"invoice_date\": \"2025-${MONTH_STR}-01\",
    \"due_date\": \"2025-${MONTH_STR}-15\",
    \"currency\": \"USD\",
    \"line_items\": [
      {\"description\": \"Pro Plan Subscription - Month ${month}\", \"quantity\": 1, \"unit_price\": \"99.00\", \"account_code\": \"4000\"},
      {\"description\": \"Additional User Seats (5)\", \"quantity\": 5, \"unit_price\": \"19.00\", \"account_code\": \"4000\"}
    ]
  }" | jq -r '.invoice_number'
done

echo "✅ Created 3 months of recurring invoices"

# Calculate MRR
MRR=$(echo "99 + (5 * 19)" | bc)
ARR=$(echo "$MRR * 12" | bc)
echo "MRR: \$$MRR | ARR: \$$ARR"
```

---

## Troubleshooting

### Common Issues

**Issue**: `curl: (7) Failed to connect to localhost port 8080`
```bash
# Solution: Start the service first
cd rust && nix develop ../ --command cargo run
```

**Issue**: `"error": "Database connection failed"`
```bash
# Solution: Initialize database
export FIN_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/mycelix_erp"
./init-database.sh
```

**Issue**: `"error": "Customer not found"`
```bash
# Solution: Create customer first or use seed data
./seed-demo-data.sh
```

**Issue**: `jq: command not found`
```bash
# Solution: Install jq
nix-shell -p jq  # On NixOS
# OR
sudo apt install jq  # On Ubuntu
# OR
brew install jq  # On macOS
```

### Debugging Tips

**Enable verbose output**:
```bash
curl -v $API_BASE/v1/fin/accounts
```

**Save response to file**:
```bash
curl $API_BASE/v1/fin/invoices > invoices.json
```

**Test with different formats**:
```bash
# Pretty print
curl $API_BASE/v1/fin/accounts | jq '.'

# Compact
curl $API_BASE/v1/fin/accounts | jq -c '.'

# Raw output (no JSON parsing)
curl $API_BASE/v1/fin/accounts
```

**Check server logs**:
```bash
# In the terminal where service is running, look for errors
# Logs show request/response details
```

---

## Advanced Testing

### Load Testing
```bash
# Create 100 invoices quickly
for i in {1..100}; do
  curl -s -X POST $API_BASE/v1/fin/invoices \
    -H "Content-Type: application/json" \
    -d "{\"customer_id\":\"$CUSTOMER_ID\",\"invoice_date\":\"2025-01-01\",\"due_date\":\"2025-02-01\",\"currency\":\"USD\",\"line_items\":[{\"description\":\"Test\",\"quantity\":1,\"unit_price\":\"10.00\",\"account_code\":\"4000\"}]}" &
done
wait
echo "✅ Created 100 invoices"
```

### Performance Testing
```bash
# Measure response time
time curl $API_BASE/v1/fin/reports/trial-balance > /dev/null

# Benchmark
ab -n 100 -c 10 $API_BASE/health  # Apache Bench
```

### Data Validation
```bash
# Verify all invoices have GL entries
INVOICE_COUNT=$(curl -s $API_BASE/v1/fin/invoices | jq 'length')
ENTRY_COUNT=$(curl -s $API_BASE/v1/fin/journal-entries | jq 'length')
echo "Invoices: $INVOICE_COUNT | Journal Entries: $ENTRY_COUNT"

# Verify double-entry bookkeeping
curl $API_BASE/v1/fin/journal-entries | jq '.[] |
  .line_items |
  {
    entry: .id,
    debits: (map(.debit_amount // 0) | add),
    credits: (map(.credit_amount // 0) | add),
    balanced: ((map(.debit_amount // 0) | add) == (map(.credit_amount // 0) | add))
  } |
  select(.balanced == false)
'  # Should return empty (all balanced)
```

---

**Last Updated**: December 30, 2025
**API Version**: v1
**Service**: Mycelix ERP - FIN Module

🧪 **Happy Testing!**
