# 🚀 Mycelix ERP - Quick Start Guide

**Get the FIN module running in 5 minutes**

---

## Prerequisites

- NixOS or Linux with Nix installed
- PostgreSQL running (or Docker)
- ~2GB free disk space

---

## Step 1: Enter Development Environment (1 min)

```bash
cd /srv/luminous-dynamics/mycelix-supplychain

# This downloads dependencies if first time (~5 min first run)
nix develop
```

**You should see**:
```
🚀 Mycelix ERP Development Environment
======================================
Rust version: rustc 1.92.0
Cargo version: cargo 1.92.0
PostgreSQL: psql (PostgreSQL) 17.7
```

---

## Step 2: Start PostgreSQL (1 min)

### Option A: System PostgreSQL
```bash
sudo systemctl start postgresql
```

### Option B: Docker PostgreSQL
```bash
docker run -d \
  --name mycelix-postgres \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=postgres \
  postgres:15
```

**Verify it's running**:
```bash
psql -h localhost -U postgres -c "SELECT version();"
```

---

## Step 3: Initialize Database (1 min)

```bash
# Set database connection URL
export FIN_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/mycelix_erp"

# Run initialization script
./init-database.sh
```

**You should see**:
```
🗄️  Mycelix ERP Database Setup
======================================
✅ PostgreSQL is running
✅ Database 'mycelix_erp' created
✅ Migrations applied successfully
🎉 Database setup complete!
```

---

## Step 4: Build and Run Service (2 min)

```bash
cd rust

# First time: This takes 5-10 minutes to compile
# Subsequent runs: ~30 seconds
nix develop ../ --command cargo run
```

**You should see**:
```
2025-12-30T18:00:00.000Z INFO provenance_service - Service DID: did:key:z6Mk...
2025-12-30T18:00:00.000Z INFO provenance_service - ✅ Finance database connected
2025-12-30T18:00:00.000Z INFO provenance_service - ✅ Finance module endpoints enabled at /v1/fin/*
2025-12-30T18:00:00.000Z INFO provenance_service - Starting server on 0.0.0.0:8080
```

---

## Step 5: Test It Works (1 min)

Open a new terminal:

### Test Health Endpoint
```bash
curl http://localhost:8080/health | jq
```

**Expected**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-30T18:00:00Z"
}
```

### Test FIN Module
```bash
curl http://localhost:8080/v1/fin/accounts | jq
```

**Expected**: List of 23 default GL accounts
```json
[
  {
    "id": "...",
    "code": "1000",
    "name": "Cash",
    "account_type": "Asset",
    ...
  },
  ...
]
```

---

## Step 6: Load Demo Data (1 min)

```bash
./seed-demo-data.sh
```

**You should see**:
```
☕ Seeding Demo Data for Luminous Coffee Roasters
==============================================
✅ Creating custom GL accounts...
✅ Creating demo customer (Artisan Cafe)...
✅ Creating demo invoice...
✅ Invoice sent!
🎉 Demo data seeded successfully!
```

---

## Step 7: Explore the Demo (5 min)

### View All Invoices
```bash
curl http://localhost:8080/v1/fin/invoices | jq
```

**Expected**: Invoice for $1,075 to Artisan Cafe

### View Trial Balance
```bash
curl http://localhost:8080/v1/fin/reports/trial-balance | jq
```

**Expected**: Debits = Credits (double-entry bookkeeping verified!)

### View Journal Entries
```bash
curl http://localhost:8080/v1/fin/journal-entries | jq
```

**Expected**: 4 journal entries created from invoice and bill

### Check Payment Status
```bash
curl http://localhost:8080/v1/fin/payments | jq
```

**Expected**: 2 payments (one received, one sent)

---

## 📊 Demo Scenario Explanation

**Company**: Luminous Coffee Roasters

**What happened**:

1. **Purchased green coffee beans** from Colombian Coffee Co.
   - Bill #BILL-001: $800
   - Status: Paid

2. **Roasted and sold coffee** to Artisan Cafe
   - Invoice #INV-001: $1,075 (50 lbs @ $12.50 + 30 lbs @ $15.00)
   - Status: Paid

3. **Automatic GL Entries Created**:
   - Debit: Accounts Receivable ($1,075)
   - Credit: Revenue - Coffee Sales ($1,075)
   - Debit: Cost of Goods Sold ($800)
   - Credit: Accounts Payable ($800)

4. **Payments Recorded**:
   - Customer payment received via wire transfer
   - Vendor payment sent via wire transfer

**Result**: $275 gross profit ($1,075 revenue - $800 COGS)

---

## 🧪 Testing Other Features

### Create Your Own Invoice
```bash
curl -X POST http://localhost:8080/v1/fin/invoices \
  -H "Content-Type: application/json" \
  -d '{
  "customer_id": "YOUR_CUSTOMER_ID",
  "invoice_date": "2025-01-15",
  "due_date": "2025-02-14",
  "currency": "USD",
  "line_items": [
    {
      "description": "Consulting Services",
      "quantity": 10,
      "unit_price": "150.00",
      "account_code": "4000"
    }
  ]
}' | jq
```

### Create a GL Account
```bash
curl -X POST http://localhost:8080/v1/fin/accounts \
  -H "Content-Type: application/json" \
  -d '{
  "code": "2100",
  "name": "Notes Payable",
  "account_type": "Liability",
  "description": "Long-term debt obligations"
}' | jq
```

### View Income Statement
```bash
curl http://localhost:8080/v1/fin/reports/income-statement | jq
```

### View Balance Sheet
```bash
curl http://localhost:8080/v1/fin/reports/balance-sheet | jq
```

---

## 🐛 Troubleshooting

### "Can't connect to PostgreSQL"
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Or for Docker:
docker ps | grep postgres

# Verify connection manually:
psql -h localhost -U postgres -c "SELECT 1;"
```

### "Table 'invoices' doesn't exist"
```bash
# Re-run migrations
export FIN_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/mycelix_erp"
./init-database.sh
```

### "Service won't start"
```bash
# Check if port 8080 is available
netstat -tuln | grep 8080

# If occupied, change port in main.rs:
# let addr = "0.0.0.0:8081";  // Change from 8080
```

### "Compilation errors"
```bash
# Clean and rebuild
cd rust
cargo clean
cargo build
```

### "Demo data fails with 'customer_id' error"
```bash
# Re-run the seed script (it creates customers first)
./seed-demo-data.sh

# Or check database:
psql "$FIN_DATABASE_URL" -c "SELECT * FROM customers;"
```

---

## 🔄 Reset Everything

```bash
# Stop service (Ctrl+C)

# Drop database
psql -h localhost -U postgres -c "DROP DATABASE IF EXISTS mycelix_erp;"

# Re-initialize
./init-database.sh

# Re-seed demo data
./seed-demo-data.sh

# Restart service
cd rust && nix develop ../ --command cargo run
```

---

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FIN_DATABASE_URL` | None | PostgreSQL connection string for FIN module |
| `DATABASE_URL` | `sqlite://data/claims.db` | SQLite DB for supply chain claims |
| `ALLOWED_ORIGINS` | `http://localhost:3000,...` | CORS allowed origins |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `mycelix_erp` | PostgreSQL database name |
| `DB_USER` | `postgres` | PostgreSQL username |

---

## 🎯 Next Steps

1. **Explore the API** - Try different endpoints
2. **Read the docs** - See DEMO_WALKTHROUGH.md for full demo script
3. **Check the code** - Browse rust/service/src/fin/
4. **Run tests** - `cargo test` (when implemented)
5. **Build features** - See IMPROVEMENT_PLAN.md for ideas

---

## 📚 Additional Resources

- **Full Demo Script**: DEMO_WALKTHROUGH.md
- **API Documentation**: (Generate with OpenAPI - future task)
- **Architecture Design**: AUTH_MULTITENANCY_DESIGN.md
- **Improvement Ideas**: IMPROVEMENT_PLAN.md
- **Pitch Deck**: PITCH_DECK.md
- **Customer Outreach**: FIRST_CUSTOMER_OUTREACH.md

---

## ✅ Success Checklist

After following this guide, you should have:

- [x] Development environment working
- [x] PostgreSQL database initialized
- [x] 23 default GL accounts created
- [x] Service running on port 8080
- [x] Demo customer and vendor created
- [x] Sample invoice and bill generated
- [x] Payments recorded
- [x] Financial reports available

**If all checked**: Congratulations! You have a working Mycelix ERP demo! 🎉

---

**Last Updated**: December 30, 2025
**Version**: v0.1.0-alpha
**Support**: See TROUBLESHOOTING.md or open GitHub issue

🚀 **Happy hacking!**
