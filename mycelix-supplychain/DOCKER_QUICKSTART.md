# Docker Quick Start - Mycelix ERP Demo

Get Mycelix ERP running in under 2 minutes with Docker.

## Prerequisites

- Docker & Docker Compose installed
- Git

## One-Command Setup

```bash
# Clone the repository
git clone https://github.com/luminous-dynamics/mycelix-supplychain
cd mycelix-supplychain

# Start everything
docker compose -f docker-compose.demo.yml up -d

# Wait for services to initialize (~30 seconds)
sleep 30

# Open the dashboard
open http://localhost:3000
```

## Demo Credentials

| Role | Email | Password |
|------|-------|----------|
| **Admin** | admin@acme-demo.com | demo123 |
| **Accountant** | accountant@acme-demo.com | demo123 |
| **Sales** | sales@acme-demo.com | demo123 |

## What's Included

### Pre-loaded Demo Data

The demo comes with realistic data for a fictional company "Acme Supply Co":

**Chart of Accounts** (35 accounts)
- Full US GAAP structure
- Assets, Liabilities, Equity, Revenue, Expenses

**Customers & Vendors**
- 5 customers (Contoso, Fabrikam, etc.)
- 5 vendors (Office Depot, AWS, etc.)

**Financial Documents**
- 6 invoices (draft, sent, paid, overdue)
- 5 bills with approval workflow
- 3 payments (receipts and payouts)
- Journal entries (opening balances, rent, payroll)

### Dashboard Pages

1. **Dashboard Home** - Overview with KPIs
   - Cash balance, receivables, payables
   - Overdue invoices count
   - Recent invoices and bills

2. **Chart of Accounts** - Browse all GL accounts
   - Grouped by type (Asset, Liability, etc.)
   - Search functionality
   - Active/inactive indicators

3. **Invoices** - Manage customer invoices
   - Status filtering (Draft, Sent, Paid, Overdue)
   - Outstanding and overdue totals
   - Create new invoice (coming soon)

4. **Bills** - Manage vendor bills
   - Approval workflow
   - Payables tracking
   - Due date management

5. **Payments** - Track all transactions
   - Receipts (incoming) and payments (outgoing)
   - Multiple payment methods
   - Status tracking

## API Access

While the dashboard is running, you can also access the API directly:

```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@acme-demo.com", "password": "demo123"}' \
  | jq -r '.token')

# Get accounts
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/fin/accounts

# Get invoices
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/fin/invoices

# Get trial balance
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/fin/reports/trial-balance
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Dashboard | 3000 | Next.js web application |
| API | 8080 | Rust backend service |
| PostgreSQL | 5432 | Database |

## Stopping the Demo

```bash
docker compose -f docker-compose.demo.yml down

# To also remove data volumes:
docker compose -f docker-compose.demo.yml down -v
```

## Troubleshooting

### Dashboard won't load
Wait longer for services to initialize. Check logs:
```bash
docker compose -f docker-compose.demo.yml logs dashboard
```

### API returns 500 errors
Check if database migrations ran:
```bash
docker compose -f docker-compose.demo.yml logs api
```

### Database connection errors
Ensure PostgreSQL is running:
```bash
docker compose -f docker-compose.demo.yml ps
```

## Next Steps

1. **Explore the Dashboard** - Click through all pages
2. **Try the API** - Use the curl examples above
3. **Read the Docs** - See [README.md](README.md) for full documentation
4. **Join the Community** - Discord, GitHub discussions

## Development Setup

To run without Docker for development:

```bash
# Start PostgreSQL separately
createdb mycelix_demo

# Run migrations
cd rust
sqlx migrate run

# Start API
cargo run --release --bin mycelix-erp

# In another terminal, start dashboard
cd dashboard
npm install
npm run dev
```

---

**Questions?** Email support@mycelix.net or open a GitHub issue.
