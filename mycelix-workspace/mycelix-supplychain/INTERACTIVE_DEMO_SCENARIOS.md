# 🎬 Mycelix ERP - Interactive Demo Scenarios

**Ready-to-run business scenarios for testing and demonstrations**

---

## 📋 Available Scenarios

1. [Coffee Roastery](#scenario-1-coffee-roastery) (Default) ☕
2. [E-commerce Store](#scenario-2-e-commerce-store) 🛒
3. [Consulting Firm](#scenario-3-consulting-firm) 💼
4. [Manufacturing Plant](#scenario-4-manufacturing-plant) 🏭
5. [Restaurant Chain](#scenario-5-restaurant-chain) 🍕
6. [Pharmaceutical Company](#scenario-6-pharmaceutical-company) 💊

---

## 🚀 Quick Start

### Prerequisites
```bash
# 1. Start the service
cd rust && nix develop ../ --command cargo run

# 2. In another terminal, set environment
export API_BASE="http://localhost:8080"
export FIN_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/mycelix_erp"
```

### Run a Scenario
```bash
# Choose one:
./run-scenario-coffee.sh
./run-scenario-ecommerce.sh
./run-scenario-consulting.sh
./run-scenario-manufacturing.sh
./run-scenario-restaurant.sh
./run-scenario-pharma.sh
```

---

## ☕ Scenario 1: Coffee Roastery

**Business**: Luminous Coffee Roasters
**Model**: Buy green beans → Roast → Sell to cafes
**Demo**: Supply chain provenance + financial tracking

### Business Flow

```
┌──────────────────────────────────────────────────────────┐
│  SUPPLY CHAIN FLOW                                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ① Colombian Farm → ② Importer → ③ Our Warehouse        │
│     (Green beans)     (Shipping)    (Storage)           │
│                                                          │
│  ④ Roasting → ⑤ Quality Control → ⑥ Customer Delivery   │
│     (Processing)  (Testing)         (Cafe)              │
│                                                          │
│  Each step: Blockchain event + Financial impact          │
└──────────────────────────────────────────────────────────┘
```

### Financial Impact

| Event | GL Entry | Amount |
|-------|----------|--------|
| Buy green beans | Debit: Inventory / Credit: A/P | $800 |
| Pay supplier | Debit: A/P / Credit: Cash | $800 |
| Sell roasted coffee | Debit: A/R / Credit: Revenue | $1,075 |
| Record COGS | Debit: COGS / Credit: Inventory | $800 |
| Receive payment | Debit: Cash / Credit: A/R | $1,075 |

**Gross Profit**: $1,075 - $800 = **$275** (34% margin)

### Run the Scenario

Create `run-scenario-coffee.sh`:
```bash
#!/usr/bin/env bash
# Coffee Roastery Complete Scenario

set -e
API_BASE="${API_BASE:-http://localhost:8080}"

echo "☕ Coffee Roastery Demo - Starting..."

# 1. Create vendor (Colombian supplier)
echo "📦 Creating vendor: Colombian Coffee Co."
VENDOR_ID=$(psql $FIN_DATABASE_URL -t -c "
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

# 2. Create customer (Cafe)
echo "🏪 Creating customer: Artisan Cafe"
CUSTOMER_ID=$(psql $FIN_DATABASE_URL -t -c "
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

# 3. Purchase green beans (create bill)
echo "📄 Creating bill for green coffee beans..."
BILL_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/bills \
  -H "Content-Type: application/json" \
  -d "{
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
echo "✅ Bill created: $BILL_NUMBER ($800)"

# 4. Approve and pay bill
echo "💰 Approving and paying bill..."
curl -s -X POST $API_BASE/v1/fin/bills/$BILL_ID/approve > /dev/null

curl -s -X POST $API_BASE/v1/fin/payments \
  -H "Content-Type: application/json" \
  -d "{
  \"payment_type\": \"Payable\",
  \"payment_date\": \"2025-01-15\",
  \"amount\": \"800.00\",
  \"payment_method\": \"BankTransfer\",
  \"reference_number\": \"WIRE-20250115-001\",
  \"bill_id\": \"$BILL_ID\",
  \"description\": \"Payment to Colombian supplier\"
}" > /dev/null
echo "✅ Paid $800 to supplier"

# 5. Roast coffee (supply chain event - would create blockchain entry)
echo "🔥 Roasting coffee beans..."
# In full system, this would be:
# curl -X POST $API_BASE/v1/events -d '{"type":"roasted","batch_id":"...","temp":420}'
echo "✅ Roasting complete (batch #2025-001)"

# 6. Create customer invoice
echo "📄 Creating invoice for Artisan Cafe..."
INVOICE_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/invoices \
  -H "Content-Type: application/json" \
  -d "{
  \"customer_id\": \"$CUSTOMER_ID\",
  \"invoice_date\": \"2025-01-20\",
  \"due_date\": \"2025-02-20\",
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
echo "✅ Invoice created: $INVOICE_NUMBER (\$$INVOICE_TOTAL)"

# 7. Send invoice
echo "📧 Sending invoice to customer..."
curl -s -X POST $API_BASE/v1/fin/invoices/$INVOICE_ID/send > /dev/null
echo "✅ Invoice sent"

# 8. Receive payment
echo "💵 Recording customer payment..."
curl -s -X POST $API_BASE/v1/fin/payments \
  -H "Content-Type: application/json" \
  -d "{
  \"payment_type\": \"Receivable\",
  \"payment_date\": \"2025-02-15\",
  \"amount\": \"$INVOICE_TOTAL\",
  \"payment_method\": \"BankTransfer\",
  \"reference_number\": \"ACH-20250215-001\",
  \"invoice_id\": \"$INVOICE_ID\",
  \"description\": \"Payment from Artisan Cafe\"
}" > /dev/null
echo "✅ Received \$$INVOICE_TOTAL from customer"

# 9. Show results
echo ""
echo "☕ ═══════════════════════════════════════"
echo "   COFFEE ROASTERY SCENARIO COMPLETE"
echo "═══════════════════════════════════════"
echo ""
echo "📊 Financial Summary:"
echo "  Revenue:    \$$INVOICE_TOTAL"
echo "  COGS:       \$800.00"
echo "  Gross Profit: \$$(echo "$INVOICE_TOTAL - 800" | bc)"
echo "  Margin:     $(echo "scale=1; ($INVOICE_TOTAL - 800) / $INVOICE_TOTAL * 100" | bc)%"
echo ""
echo "🔗 Supply Chain Events:"
echo "  ✅ Green beans purchased from Colombia"
echo "  ✅ Beans roasted (batch #2025-001)"
echo "  ✅ Coffee delivered to Artisan Cafe"
echo ""
echo "💼 Financial Transactions:"
echo "  ✅ Bill #$BILL_NUMBER: \$800 (PAID)"
echo "  ✅ Invoice #$INVOICE_NUMBER: \$$INVOICE_TOTAL (PAID)"
echo ""
echo "📈 View Reports:"
echo "  curl $API_BASE/v1/fin/reports/trial-balance | jq"
echo "  curl $API_BASE/v1/fin/reports/income-statement | jq"
echo ""
```

Make executable and run:
```bash
chmod +x run-scenario-coffee.sh
./run-scenario-coffee.sh
```

---

## 🛒 Scenario 2: E-commerce Store

**Business**: Tech Gadgets Online
**Model**: Buy wholesale → Sell online → Ship direct
**Demo**: High volume transactions + inventory tracking

### Business Metrics
- Orders per day: 50-100
- Average order value: $75
- Monthly revenue: $225,000
- Gross margin: 40%

### Run Script

Create `run-scenario-ecommerce.sh`:
```bash
#!/usr/bin/env bash
# E-commerce Store Scenario

set -e
API_BASE="${API_BASE:-http://localhost:8080}"

echo "🛒 E-commerce Store Demo - Starting..."

# 1. Create wholesale vendor
VENDOR_ID=$(psql $FIN_DATABASE_URL -t -c "
INSERT INTO vendors (id, name, email)
VALUES (gen_random_uuid(), 'Wholesale Electronics Inc.', 'sales@wholesale.example.com')
RETURNING id;
" | tr -d ' ')

# 2. Purchase inventory (1000 units @ $30 each)
echo "📦 Purchasing wholesale inventory..."
BILL_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/bills \
  -H "Content-Type: application/json" \
  -d "{
  \"vendor_id\": \"$VENDOR_ID\",
  \"bill_date\": \"$(date +%Y-%m-01)\",
  \"due_date\": \"$(date -d '+30 days' +%Y-%m-%d)\",
  \"currency\": \"USD\",
  \"line_items\": [
    {\"description\": \"Wireless Earbuds - 500 units\", \"quantity\": 500, \"unit_price\": \"20.00\", \"account_code\": \"5010\"},
    {\"description\": \"Phone Cases - 300 units\", \"quantity\": 300, \"unit_price\": \"5.00\", \"account_code\": \"5010\"},
    {\"description\": \"USB Cables - 200 units\", \"quantity\": 200, \"unit_price\": \"2.50\", \"account_code\": \"5010\"}
  ]
}")

BILL_ID=$(echo "$BILL_RESPONSE" | jq -r '.id')
BILL_TOTAL=$(echo "$BILL_RESPONSE" | jq -r '.total_amount')
echo "✅ Inventory purchased: \$$BILL_TOTAL"

# 3. Pay vendor
curl -s -X POST $API_BASE/v1/fin/bills/$BILL_ID/approve > /dev/null
curl -s -X POST $API_BASE/v1/fin/payments \
  -H "Content-Type: application/json" \
  -d "{
  \"payment_type\": \"Payable\",
  \"payment_date\": \"$(date +%Y-%m-%d)\",
  \"amount\": \"$BILL_TOTAL\",
  \"payment_method\": \"BankTransfer\",
  \"reference_number\": \"ACH-$(date +%Y%m%d)\",
  \"bill_id\": \"$BILL_ID\",
  \"description\": \"Wholesale inventory payment\"
}" > /dev/null

# 4. Create 10 customer orders
echo "🎁 Creating customer orders..."
TOTAL_REVENUE=0

for i in {1..10}; do
  CUSTOMER_ID=$(psql $FIN_DATABASE_URL -t -c "
  INSERT INTO customers (id, name, email)
  VALUES (gen_random_uuid(), 'Customer $i', 'customer$i@example.com')
  RETURNING id;
  " | tr -d ' ')

  INVOICE_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/invoices \
    -H "Content-Type: application/json" \
    -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"invoice_date\": \"$(date +%Y-%m-%d)\",
    \"due_date\": \"$(date -d '+14 days' +%Y-%m-%d)\",
    \"currency\": \"USD\",
    \"line_items\": [
      {\"description\": \"Wireless Earbuds\", \"quantity\": 1, \"unit_price\": \"49.99\", \"account_code\": \"4000\"},
      {\"description\": \"Phone Case\", \"quantity\": 1, \"unit_price\": \"14.99\", \"account_code\": \"4000\"}
    ]
  }")

  AMOUNT=$(echo "$INVOICE_RESPONSE" | jq -r '.total_amount')
  TOTAL_REVENUE=$(echo "$TOTAL_REVENUE + $AMOUNT" | bc)
  echo "  Order $i: \$$AMOUNT"
done

echo "✅ Created 10 orders totaling \$$TOTAL_REVENUE"

# 5. Summary
echo ""
echo "🛒 ═══════════════════════════════════════"
echo "   E-COMMERCE SCENARIO COMPLETE"
echo "═══════════════════════════════════════"
echo ""
echo "📊 Financial Summary:"
echo "  Inventory Cost:  \$$BILL_TOTAL"
echo "  Revenue (10 orders): \$$TOTAL_REVENUE"
echo "  Gross Profit: \$$(echo "$TOTAL_REVENUE - ($BILL_TOTAL / 100)" | bc)"
echo ""
echo "📦 Inventory Purchased:"
echo "  • 500 Wireless Earbuds @ \$20"
echo "  • 300 Phone Cases @ \$5"
echo "  • 200 USB Cables @ \$2.50"
echo ""
echo "🎁 Customer Orders: 10 completed"
echo ""
```

---

## 💼 Scenario 3: Consulting Firm

**Business**: Digital Strategy Consultants
**Model**: Time & materials billing
**Demo**: Service invoicing + project tracking

### Business Metrics
- Hourly rate: $200/hour
- Avg project: 40 hours ($8,000)
- Monthly projects: 5-8
- Monthly revenue: $40,000-$64,000

### Run Script

Create `run-scenario-consulting.sh`:
```bash
#!/usr/bin/env bash
# Consulting Firm Scenario

set -e
API_BASE="${API_BASE:-http://localhost:8080}"

echo "💼 Consulting Firm Demo - Starting..."

# Create 3 clients with different project types
CLIENTS=(
  "Tech Startup Inc.:Software Architecture Review:80:175"
  "Manufacturing Co.:ERP Implementation Consulting:120:150"
  "Retail Chain LLC:Digital Transformation Strategy:60:200"
)

TOTAL_REVENUE=0

for client_data in "${CLIENTS[@]}"; do
  IFS=':' read -r client_name project hours rate <<< "$client_data"

  # Create customer
  CUSTOMER_ID=$(psql $FIN_DATABASE_URL -t -c "
  INSERT INTO customers (id, name, email)
  VALUES (gen_random_uuid(), '$client_name', '$(echo $client_name | tr ' ' '.')@example.com')
  RETURNING id;
  " | tr -d ' ')

  # Create invoice
  PROJECT_TOTAL=$(echo "$hours * $rate" | bc)

  INVOICE_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/invoices \
    -H "Content-Type: application/json" \
    -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"invoice_date\": \"$(date +%Y-%m-%d)\",
    \"due_date\": \"$(date -d '+30 days' +%Y-%m-%d)\",
    \"currency\": \"USD\",
    \"line_items\": [
      {
        \"description\": \"$project - $hours hours @ \$$rate/hr\",
        \"quantity\": $hours,
        \"unit_price\": \"$rate.00\",
        \"account_code\": \"4100\"
      }
    ]
  }")

  INVOICE_NUMBER=$(echo "$INVOICE_RESPONSE" | jq -r '.invoice_number')
  TOTAL_REVENUE=$(echo "$TOTAL_REVENUE + $PROJECT_TOTAL" | bc)

  echo "✅ $client_name: $INVOICE_NUMBER - \$$PROJECT_TOTAL ($hours hrs)"
done

echo ""
echo "💼 ═══════════════════════════════════════"
echo "   CONSULTING FIRM SCENARIO COMPLETE"
echo "═══════════════════════════════════════"
echo ""
echo "📊 Monthly Performance:"
echo "  Total Revenue:  \$$TOTAL_REVENUE"
echo "  Projects:       3"
echo "  Total Hours:    260"
echo "  Avg Rate:       \$$(echo "$TOTAL_REVENUE / 260" | bc)/hr"
echo ""
```

---

## 🏭 Scenario 4: Manufacturing Plant

**Business**: Custom Metal Parts Manufacturing
**Model**: Job shop - custom orders
**Demo**: Work-in-progress tracking + job costing

### Run Script

Create `run-scenario-manufacturing.sh`:
```bash
#!/usr/bin/env bash
# Manufacturing Plant Scenario

set -e
API_BASE="${API_BASE:-http://localhost:8080}"

echo "🏭 Manufacturing Plant Demo - Starting..."

# 1. Purchase raw materials
VENDOR_ID=$(psql $FIN_DATABASE_URL -t -c "
INSERT INTO vendors (id, name, email)
VALUES (gen_random_uuid(), 'Steel Supplier Inc.', 'sales@steel.example.com')
RETURNING id;
" | tr -d ' ')

echo "🔧 Purchasing raw materials..."
BILL_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/bills \
  -H "Content-Type: application/json" \
  -d "{
  \"vendor_id\": \"$VENDOR_ID\",
  \"bill_date\": \"$(date +%Y-%m-01)\",
  \"due_date\": \"$(date -d '+30 days' +%Y-%m-%d)\",
  \"currency\": \"USD\",
  \"line_items\": [
    {\"description\": \"Stainless Steel Sheets - 1000 lbs\", \"quantity\": 1000, \"unit_price\": \"3.50\", \"account_code\": \"5010\"},
    {\"description\": \"Aluminum Rods - 500 units\", \"quantity\": 500, \"unit_price\": \"8.00\", \"account_code\": \"5010\"}
  ]
}")

MATERIALS_COST=$(echo "$BILL_RESPONSE" | jq -r '.total_amount')
echo "✅ Materials purchased: \$$MATERIALS_COST"

# 2. Create customer orders (3 jobs)
JOBS=(
  "Aerospace Components Ltd:Precision Wing Brackets:50:450"
  "Automotive Parts Co:Custom Engine Mounts:100:180"
  "Medical Devices Inc:Surgical Instrument Housings:25:800"
)

TOTAL_REVENUE=0

for job_data in "${JOBS[@]}"; do
  IFS=':' read -r customer job_name quantity price <<< "$job_data"

  CUSTOMER_ID=$(psql $FIN_DATABASE_URL -t -c "
  INSERT INTO customers (id, name, email)
  VALUES (gen_random_uuid(), '$customer', 'orders@$(echo $customer | tr ' ' '.')@example.com')
  RETURNING id;
  " | tr -d ' ')

  JOB_TOTAL=$(echo "$quantity * $price" | bc)

  curl -s -X POST $API_BASE/v1/fin/invoices \
    -H "Content-Type: application/json" \
    -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"invoice_date\": \"$(date +%Y-%m-%d)\",
    \"due_date\": \"$(date -d '+45 days' +%Y-%m-%d)\",
    \"currency\": \"USD\",
    \"line_items\": [
      {\"description\": \"$job_name - Qty: $quantity\", \"quantity\": $quantity, \"unit_price\": \"$price.00\", \"account_code\": \"4000\"}
    ]
  }" > /dev/null

  TOTAL_REVENUE=$(echo "$TOTAL_REVENUE + $JOB_TOTAL" | bc)
  echo "✅ Job: $customer - \$$JOB_TOTAL"
done

GROSS_PROFIT=$(echo "$TOTAL_REVENUE - $MATERIALS_COST" | bc)
MARGIN=$(echo "scale=1; $GROSS_PROFIT / $TOTAL_REVENUE * 100" | bc)

echo ""
echo "🏭 ═══════════════════════════════════════"
echo "   MANUFACTURING SCENARIO COMPLETE"
echo "═══════════════════════════════════════"
echo ""
echo "📊 Job Costing Summary:"
echo "  Total Revenue:     \$$TOTAL_REVENUE"
echo "  Materials Cost:    \$$MATERIALS_COST"
echo "  Gross Profit:      \$$GROSS_PROFIT"
echo "  Margin:            $MARGIN%"
echo ""
echo "🔧 Jobs Completed: 3"
echo ""
```

---

## 🍕 Scenario 5: Restaurant Chain

**Business**: Pizza Restaurant (3 locations)
**Model**: Food service + delivery
**Demo**: Multi-location accounting + food cost tracking

### Run Script

Create `run-scenario-restaurant.sh`:
```bash
#!/usr/bin/env bash
# Restaurant Chain Scenario

set -e
API_BASE="${API_BASE:-http://localhost:8080}"

echo "🍕 Restaurant Chain Demo - Starting..."

# 1. Purchase food supplies
VENDOR_ID=$(psql $FIN_DATABASE_URL -t -c "
INSERT INTO vendors (id, name, email)
VALUES (gen_random_uuid(), 'Restaurant Supply Co.', 'orders@supply.example.com')
RETURNING id;
" | tr -d ' ')

echo "🥗 Purchasing weekly food supplies..."
BILL_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/bills \
  -H "Content-Type: application/json" \
  -d "{
  \"vendor_id\": \"$VENDOR_ID\",
  \"bill_date\": \"$(date +%Y-%m-%d)\",
  \"due_date\": \"$(date -d '+7 days' +%Y-%m-%d)\",
  \"currency\": \"USD\",
  \"line_items\": [
    {\"description\": \"Pizza Dough - 200 lbs\", \"quantity\": 200, \"unit_price\": \"2.50\", \"account_code\": \"5010\"},
    {\"description\": \"Mozzarella Cheese - 100 lbs\", \"quantity\": 100, \"unit_price\": \"4.50\", \"account_code\": \"5010\"},
    {\"description\": \"Pizza Sauce - 50 gallons\", \"quantity\": 50, \"unit_price\": \"8.00\", \"account_code\": \"5010\"},
    {\"description\": \"Toppings (Various) - Bulk\", \"quantity\": 1, \"unit_price\": \"1200.00\", \"account_code\": \"5010\"}
  ]
}")

FOOD_COST=$(echo "$BILL_RESPONSE" | jq -r '.total_amount')
echo "✅ Food supplies: \$$FOOD_COST"

# 2. Record daily sales for 3 locations
LOCATIONS=("Downtown" "Suburbs" "Beachfront")
TOTAL_REVENUE=0

for location in "${LOCATIONS[@]}"; do
  # Simulate day's sales
  DAILY_SALES=$(echo "1500 + $RANDOM % 1000" | bc)  # $1,500-$2,500 per day

  CUSTOMER_ID=$(psql $FIN_DATABASE_URL -t -c "
  INSERT INTO customers (id, name, email)
  VALUES (gen_random_uuid(), '$location Location - Daily Sales', 'pos@$location.pizzeria.example.com')
  RETURNING id;
  " | tr -d ' ')

  curl -s -X POST $API_BASE/v1/fin/invoices \
    -H "Content-Type: application/json" \
    -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"invoice_date\": \"$(date +%Y-%m-%d)\",
    \"due_date\": \"$(date +%Y-%m-%d)\",
    \"currency\": \"USD\",
    \"line_items\": [
      {\"description\": \"Daily Sales - $location\", \"quantity\": 1, \"unit_price\": \"$DAILY_SALES.00\", \"account_code\": \"4000\"}
    ]
  }" > /dev/null

  TOTAL_REVENUE=$(echo "$TOTAL_REVENUE + $DAILY_SALES" | bc)
  echo "✅ $location: \$$DAILY_SALES"
done

FOOD_COST_PCT=$(echo "scale=1; $FOOD_COST / ($TOTAL_REVENUE * 7) * 100" | bc)  # Weekly cost as % of daily sales

echo ""
echo "🍕 ═══════════════════════════════════════"
echo "   RESTAURANT CHAIN SCENARIO COMPLETE"
echo "═══════════════════════════════════════"
echo ""
echo "📊 Daily Performance (All Locations):"
echo "  Total Revenue:     \$$TOTAL_REVENUE"
echo "  Food Cost (week):  \$$FOOD_COST"
echo "  Food Cost %:       $FOOD_COST_PCT%"
echo ""
echo "🏪 Locations: 3"
echo "  • Downtown: High volume"
echo "  • Suburbs: Family-friendly"
echo "  • Beachfront: Premium prices"
echo ""
```

---

## 💊 Scenario 6: Pharmaceutical Company

**Business**: Generic Drug Manufacturer
**Model**: Regulated production + distribution
**Demo**: Compliance tracking + batch verification

### Run Script

Create `run-scenario-pharma.sh`:
```bash
#!/usr/bin/env bash
# Pharmaceutical Company Scenario

set -e
API_BASE="${API_BASE:-http://localhost:8080}"

echo "💊 Pharmaceutical Company Demo - Starting..."

# 1. Purchase raw chemical ingredients (regulated)
VENDOR_ID=$(psql $FIN_DATABASE_URL -t -c "
INSERT INTO vendors (id, name, email)
VALUES (gen_random_uuid(), 'Certified Chemical Supply LLC', 'compliance@chemicals.example.com')
RETURNING id;
" | tr -d ' ')

echo "🧪 Purchasing raw ingredients (FDA regulated)..."
BILL_RESPONSE=$(curl -s -X POST $API_BASE/v1/fin/bills \
  -H "Content-Type: application/json" \
  -d "{
  \"vendor_id\": \"$VENDOR_ID\",
  \"bill_date\": \"$(date +%Y-%m-%d)\",
  \"due_date\": \"$(date -d '+30 days' +%Y-%m-%d)\",
  \"currency\": \"USD\",
  \"line_items\": [
    {\"description\": \"Active Pharmaceutical Ingredient (API) - Batch #A-2025-001 - 50kg\", \"quantity\": 50, \"unit_price\": \"850.00\", \"account_code\": \"5010\"},
    {\"description\": \"Excipients - Batch #E-2025-001 - 200kg\", \"quantity\": 200, \"unit_price\": \"45.00\", \"account_code\": \"5010\"}
  ]
}")

MATERIALS_COST=$(echo "$BILL_RESPONSE" | jq -r '.total_amount')
echo "✅ Raw materials: \$$MATERIALS_COST (COA verified)"

# 2. Production (blockchain event for audit trail)
echo "🏭 Manufacturing batch..."
# In full system:
# curl -X POST $API_BASE/v1/events -d '{
#   "type":"production_started",
#   "batch_id":"DRUG-2025-Q1-001",
#   "temperature_log":"...FDA required data...",
#   "blockchain_hash":"..."
# }'
echo "✅ Batch #DRUG-2025-Q1-001 produced (10,000 units)"

# 3. Create sales invoices (to pharmacies/distributors)
DISTRIBUTORS=(
  "National Pharmacy Chain:5000:8.50"
  "Regional Distributor Co:3000:8.75"
  "Hospital Group LLC:2000:9.00"
)

TOTAL_REVENUE=0

for dist_data in "${DISTRIBUTORS[@]}"; do
  IFS=':' read -r distributor quantity price <<< "$dist_data"

  CUSTOMER_ID=$(psql $FIN_DATABASE_URL -t -c "
  INSERT INTO customers (id, name, email)
  VALUES (gen_random_uuid(), '$distributor', 'purchasing@$(echo $distributor | tr ' ' '.')@example.com')
  RETURNING id;
  " | tr -d ' ')

  ORDER_TOTAL=$(echo "$quantity * $price" | bc)

  curl -s -X POST $API_BASE/v1/fin/invoices \
    -H "Content-Type: application/json" \
    -d "{
    \"customer_id\": \"$CUSTOMER_ID\",
    \"invoice_date\": \"$(date +%Y-%m-%d)\",
    \"due_date\": \"$(date -d '+60 days' +%Y-%m-%d)\",
    \"currency\": \"USD\",
    \"line_items\": [
      {
        \"description\": \"Generic Drug XYZ - 100mg - Batch #DRUG-2025-Q1-001 - Qty: $quantity units\",
        \"quantity\": $quantity,
        \"unit_price\": \"$price\",
        \"account_code\": \"4000\"
      }
    ]
  }" > /dev/null

  TOTAL_REVENUE=$(echo "$TOTAL_REVENUE + $ORDER_TOTAL" | bc)
  echo "✅ $distributor: \$$ORDER_TOTAL ($quantity units)"
done

GROSS_PROFIT=$(echo "$TOTAL_REVENUE - $MATERIALS_COST" | bc)
MARGIN=$(echo "scale=1; $GROSS_PROFIT / $TOTAL_REVENUE * 100" | bc)

echo ""
echo "💊 ═══════════════════════════════════════"
echo "   PHARMACEUTICAL SCENARIO COMPLETE"
echo "═══════════════════════════════════════"
echo ""
echo "📊 Batch Economics:"
echo "  Total Revenue:     \$$TOTAL_REVENUE"
echo "  Materials Cost:    \$$MATERIALS_COST"
echo "  Gross Profit:      \$$GROSS_PROFIT"
echo "  Margin:            $MARGIN%"
echo ""
echo "🔬 Compliance:"
echo "  ✅ Batch #DRUG-2025-Q1-001"
echo "  ✅ COA (Certificate of Analysis) verified"
echo "  ✅ Blockchain audit trail active"
echo "  ✅ 10,000 units produced & distributed"
echo ""
echo "📋 Traceability:"
echo "  Every unit traceable to raw material batch"
echo "  Instant recall capability via blockchain"
echo ""
```

---

## 🎯 Using the Scenarios

### For Demos
```bash
# Pick the most relevant scenario for your audience:
./run-scenario-coffee.sh      # Food & beverage companies
./run-scenario-pharma.sh       # Regulated industries
./run-scenario-manufacturing.sh # Job shops, make-to-order
./run-scenario-ecommerce.sh    # Online retailers
./run-scenario-consulting.sh   # Service businesses
./run-scenario-restaurant.sh   # Multi-location retail
```

### For Testing
```bash
# Run all scenarios to test system at scale
for scenario in coffee ecommerce consulting manufacturing restaurant pharma; do
  ./run-scenario-$scenario.sh
  sleep 5  # Pause between scenarios
done

# Check final state
curl $API_BASE/v1/fin/reports/trial-balance | jq
```

### For Training
```bash
# Show new users how different businesses use the system
# Start with simplest (coffee) and progress to most complex (pharma)
```

---

## 📚 Customizing Scenarios

### Create Your Own

Template for new scenario:
```bash
#!/usr/bin/env bash
# YOUR_BUSINESS_TYPE Scenario

set -e
API_BASE="${API_BASE:-http://localhost:8080}"

echo "🏢 YOUR_BUSINESS Demo - Starting..."

# 1. Create vendors
# 2. Purchase materials/inventory
# 3. Create customers
# 4. Generate sales
# 5. Record payments
# 6. Show summary

echo "🏢 ═══════════════════════════════════════"
echo "   YOUR_BUSINESS SCENARIO COMPLETE"
echo "═══════════════════════════════════════"
```

---

**Last Updated**: December 30, 2025
**Scenarios Available**: 6
**Industries Covered**: Food, E-commerce, Services, Manufacturing, Hospitality, Pharma

🎬 **Ready to demo!**
