#!/usr/bin/env bash
# Mycelix ERP Demo: Bella Vista Restaurant Group
# Scenario: Multi-Location Restaurant with Food Cost Tracking

set -e

API_BASE="http://localhost:8000/v1"
FIN_BASE="$API_BASE/fin"

echo "🍽️  Mycelix ERP Demo: Bella Vista Restaurant Group"
echo "======================================================"
echo ""
echo "Scenario: Italian Restaurant Chain (3 locations)"
echo "This demo shows:"
echo "  - Food cost tracking and recipe costing"
echo "  - Multi-location inventory management"
echo "  - Vendor management (food suppliers)"
echo "  - Daily sales and COGS calculation"
echo ""

if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
    echo "❌ Service not running"
    exit 1
fi

echo "✅ Service is running"
echo ""

# Step 1: Create Food Suppliers
echo "🚚 Step 1: Creating Food Suppliers"
echo "----------------------------------------"

echo "Creating produce supplier..."
PRODUCE=$(curl -s -X POST "$FIN_BASE/vendors" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Fresh Valley Farms",
    "email": "orders@freshvalley.com",
    "type": "food_supplier",
    "payment_terms_days": 7
  }')
PRODUCE_ID=$(echo "$PRODUCE" | jq -r '.id')
echo "✅ Produce supplier: Fresh Valley Farms ($PRODUCE_ID)"
echo ""

sleep 1

echo "Creating protein supplier..."
PROTEIN=$(curl -s -X POST "$FIN_BASE/vendors" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prime Meats & Seafood",
    "email": "wholesale@primemeats.com",
    "type": "food_supplier",
    "payment_terms_days": 14
  }')
PROTEIN_ID=$(echo "$PROTEIN" | jq -r '.id')
echo "✅ Protein supplier: Prime Meats ($PROTEIN_ID)"
echo ""

sleep 1

echo "Creating dry goods supplier..."
DRYGOODS=$(curl -s -X POST "$FIN_BASE/vendors" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Restaurant Depot",
    "email": "invoicing@restaurantdepot.com",
    "type": "food_supplier",
    "payment_terms_days": 30
  }')
DRYGOODS_ID=$(echo "$DRYGOODS" | jq -r '.id')
echo "✅ Dry goods supplier: Restaurant Depot ($DRYGOODS_ID)"
echo ""

# Step 2: Receive Inventory
echo "📦 Step 2: Receiving Inventory (Downtown Location)"
echo "----------------------------------------"

echo "Receiving produce delivery..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "INVENTORY_ADD",
    "product_id": "roma-tomatoes",
    "location": "Bella Vista Downtown - Walk-in Cooler",
    "timestamp": "2024-02-05T06:00:00Z",
    "actor": "Fresh Valley Farms Delivery",
    "metadata": {
      "quantity": 50,
      "unit": "lbs",
      "unit_cost": "1.85",
      "expiration": "2024-02-12",
      "temp_check_f": 38,
      "quality": "Grade A"
    }
  }' > /dev/null
echo "✅ 50 lbs Roma tomatoes received"
echo ""

sleep 1

echo "Receiving protein delivery..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "INVENTORY_ADD",
    "product_id": "salmon-filet",
    "location": "Bella Vista Downtown - Walk-in Freezer",
    "timestamp": "2024-02-05T07:00:00Z",
    "actor": "Prime Meats & Seafood",
    "metadata": {
      "quantity": 30,
      "unit": "lbs",
      "unit_cost": "18.50",
      "origin": "Norwegian Atlantic",
      "temp_check_f": -5,
      "quality": "Premium"
    }
  }' > /dev/null
echo "✅ 30 lbs salmon filet received"
echo ""

sleep 1

echo "Receiving dry goods..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "INVENTORY_ADD",
    "product_id": "pasta-linguine",
    "location": "Bella Vista Downtown - Dry Storage",
    "timestamp": "2024-02-05T08:00:00Z",
    "actor": "Restaurant Depot Delivery",
    "metadata": {
      "quantity": 40,
      "unit": "lbs",
      "unit_cost": "2.25",
      "brand": "De Cecco",
      "best_by": "2025-02-01"
    }
  }' > /dev/null
echo "✅ 40 lbs pasta linguine received"
echo ""

# Step 3: Recipe/Dish Preparation Tracking
echo "👨‍🍳 Step 3: Recipe Costing (Signature Dishes)"
echo "----------------------------------------"

echo "Recording prep for Salmon Piccata (signature dish)..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "RECIPE_PREP",
    "product_id": "salmon-piccata-dish",
    "location": "Bella Vista Downtown - Kitchen",
    "timestamp": "2024-02-05T15:00:00Z",
    "actor": "Chef Marco Russo",
    "metadata": {
      "recipe": "Salmon Piccata",
      "portions": 20,
      "ingredients": {
        "salmon_filet_lbs": 10,
        "roma_tomatoes_lbs": 3,
        "pasta_linguine_lbs": 5,
        "capers_oz": 4,
        "butter_oz": 8,
        "lemon_each": 6
      },
      "cost_per_portion": "8.75",
      "menu_price": "32.00",
      "food_cost_percent": "27.3%"
    }
  }' > /dev/null
echo "✅ 20 portions of Salmon Piccata prepped"
echo "   Cost per portion: $8.75"
echo "   Menu price: $32.00"
echo "   Food cost: 27.3%"
echo ""

# Step 4: Daily Sales
echo "💵 Step 4: Recording Daily Sales"
echo "----------------------------------------"

echo "Downtown location - Monday dinner service..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "SALES_TRANSACTION",
    "product_id": "daily-sales-downtown",
    "location": "Bella Vista Downtown",
    "timestamp": "2024-02-05T22:00:00Z",
    "actor": "POS System",
    "metadata": {
      "service": "dinner",
      "covers": 85,
      "total_sales": "4250.00",
      "payment_breakdown": {
        "cash": "450.00",
        "credit_card": "3200.00",
        "gift_card": "600.00"
      },
      "top_sellers": [
        {"dish": "Salmon Piccata", "qty": 12},
        {"dish": "Margherita Pizza", "qty": 18},
        {"dish": "Tiramisu", "qty": 22}
      ]
    }
  }' > /dev/null
echo "✅ Dinner service recorded: 85 covers, $4,250 sales"
echo ""

sleep 1

echo "Midtown location - Monday dinner service..."
curl -s -X POST "$API_BASE/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "SALES_TRANSACTION",
    "product_id": "daily-sales-midtown",
    "location": "Bella Vista Midtown",
    "timestamp": "2024-02-05T22:00:00Z",
    "actor": "POS System",
    "metadata": {
      "service": "dinner",
      "covers": 110,
      "total_sales": "5800.00",
      "payment_breakdown": {
        "cash": "580.00",
        "credit_card": "4620.00",
        "gift_card": "600.00"
      }
    }
  }' > /dev/null
echo "✅ Midtown location: 110 covers, $5,800 sales"
echo ""

# Step 5: Create Supplier Bills
echo "📝 Step 5: Recording Supplier Invoices"
echo "----------------------------------------"

echo "Produce invoice..."
PRODUCE_BILL=$(curl -s -X POST "$FIN_BASE/bills" \
  -H "Content-Type: application/json" \
  -d "{
    \"vendor_id\": \"$PRODUCE_ID\",
    \"items\": [
      {
        \"description\": \"Roma Tomatoes (50 lbs @ \$1.85/lb)\",
        \"quantity\": 1,
        \"unit_price\": \"92.50\",
        \"account_code\": \"5200\"
      },
      {
        \"description\": \"Fresh Basil (5 lbs @ \$12/lb)\",
        \"quantity\": 1,
        \"unit_price\": \"60.00\",
        \"account_code\": \"5200\"
      },
      {
        \"description\": \"Arugula (10 lbs @ \$4.50/lb)\",
        \"quantity\": 1,
        \"unit_price\": \"45.00\",
        \"account_code\": \"5200\"
      }
    ],
    \"due_date\": \"2024-02-12\",
    \"notes\": \"Weekly produce delivery - Downtown location\"
  }")
PRODUCE_TOTAL=$(echo "$PRODUCE_BILL" | jq -r '.total')
echo "✅ Produce bill: $$PRODUCE_TOTAL (due 2/12)"
echo ""

sleep 1

echo "Protein invoice..."
PROTEIN_BILL=$(curl -s -X POST "$FIN_BASE/bills" \
  -H "Content-Type: application/json" \
  -d "{
    \"vendor_id\": \"$PROTEIN_ID\",
    \"items\": [
      {
        \"description\": \"Norwegian Salmon Filet (30 lbs @ \$18.50/lb)\",
        \"quantity\": 1,
        \"unit_price\": \"555.00\",
        \"account_code\": \"5200\"
      },
      {
        \"description\": \"Chicken Breast (40 lbs @ \$5.50/lb)\",
        \"quantity\": 1,
        \"unit_price\": \"220.00\",
        \"account_code\": \"5200\"
      }
    ],
    \"due_date\": \"2024-02-19\",
    \"notes\": \"Weekly protein delivery\"
  }")
PROTEIN_TOTAL=$(echo "$PROTEIN_BILL" | jq -r '.total')
echo "✅ Protein bill: $$PROTEIN_TOTAL (due 2/19)"
echo ""

# Step 6: Generate Daily P&L Summary
echo "📊 Step 6: Daily P&L Summary"
echo "----------------------------------------"

TOTAL_SALES=$(echo "4250.00 + 5800.00" | bc)
TOTAL_COGS=$(echo "scale=2; $TOTAL_SALES * 0.28" | bc)  # 28% food cost target
GROSS_PROFIT=$(echo "$TOTAL_SALES - $TOTAL_COGS" | bc)
MARGIN=$(echo "scale=1; ($GROSS_PROFIT / $TOTAL_SALES) * 100" | bc)

echo "Bella Vista Restaurant Group - Feb 5, 2024"
echo "  Total Sales: $$TOTAL_SALES"
echo "    Downtown: $4,250 (85 covers)"
echo "    Midtown: $5,800 (110 covers)"
echo ""
echo "  Food Cost (COGS): $$TOTAL_COGS (28.0% target)"
echo "  Gross Profit: $$GROSS_PROFIT"
echo "  Gross Margin: ${MARGIN}%"
echo ""
echo "  Average Check:"
echo "    Downtown: $(echo "scale=2; 4250 / 85" | bc) per person"
echo "    Midtown: $(echo "scale=2; 5800 / 110" | bc) per person"
echo ""

# Step 7: Inventory Status
echo "📦 Step 7: Current Inventory Status"
echo "----------------------------------------"

echo "Downtown location inventory:"
echo "  Roma Tomatoes: 50 lbs (expires 2/12)"
echo "  Salmon Filet: 30 lbs → 20 lbs (used 10 lbs for Salmon Piccata)"
echo "  Pasta Linguine: 40 lbs → 35 lbs (used 5 lbs)"
echo ""

# Step 8: Financial Reports
echo "💰 Step 8: Financial Reports"
echo "----------------------------------------"

echo "Accounts Payable (Food Suppliers):"
echo "  Fresh Valley Farms: $$PRODUCE_TOTAL (due 2/12)"
echo "  Prime Meats: $$PROTEIN_TOTAL (due 2/19)"
echo "  Total AP: $(echo "$PRODUCE_TOTAL + $PROTEIN_TOTAL" | bc)"
echo ""

echo "Trial Balance:"
curl -s "$FIN_BASE/reports/trial-balance" | jq '.accounts[] | select(.balance != "0.00")'
echo ""

# Summary
echo "✅ Demo Complete!"
echo "================="
echo ""
echo "Summary of Operations:"
echo "  🚚 3 food suppliers configured"
echo "  📦 Inventory received (produce + protein + dry goods)"
echo "  👨‍🍳 Recipe costing: Salmon Piccata (27.3% food cost)"
echo "  💵 Daily sales recorded:"
echo "     - Downtown: $4,250 (85 covers)"
echo "     - Midtown: $5,800 (110 covers)"
echo "     - Total: $$TOTAL_SALES"
echo "  📝 Supplier bills recorded: $(echo "$PRODUCE_TOTAL + $PROTEIN_TOTAL" | bc)"
echo "  📊 P&L tracking: ${MARGIN}% gross margin"
echo ""
echo "Key Metrics:"
echo "  - Food cost target: 28%"
echo "  - Average check: \$$(echo "scale=2; $TOTAL_SALES / 195" | bc) per person (combined)"
echo "  - Sales per location: Downtown \$50/cover, Midtown \$53/cover"
echo "  - Inventory turnover: Tracking freshness and expiration"
echo ""
echo "🍽️  Mycelix ERP: Restaurant Management Excellence!"
