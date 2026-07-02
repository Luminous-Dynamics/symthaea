# FIN Module Integration Guide

Quick guide to integrating the Finance module into your Mycelix deployment.

## 🚀 Quick Integration (5 Steps)

### Step 1: Run Database Migrations

```bash
cd /srv/luminous-dynamics/mycelix-supplychain

# Run the FIN module migration
sqlx migrate run --source migrations/
```

This creates:
- 8 finance tables
- 6 custom ENUM types
- 23 standard GL accounts (seed data)
- All necessary indexes and constraints

### Step 2: Update main.rs (Add FIN Router)

```rust
// src/main.rs
use provenance_service::fin;
use sqlx::PgPool;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ... existing setup ...

    // Initialize database pool
    let pool = PgPool::connect(&config.database_url).await?;

    // Create FIN services
    let fin_state = fin::api::FinState {
        ledger: fin::LedgerService::new(pool.clone()),
        invoicing: fin::InvoicingService::new(pool.clone()),
        payments: fin::PaymentService::new(pool.clone()),
    };

    // Build app with both SCM and FIN routes
    let app = Router::new()
        // Existing SCM routes
        .route("/v1/events", post(api::post_event))
        .route("/v1/claims", get(api::list_claims))
        // ... other existing routes ...

        // NEW: FIN module routes
        .merge(fin::api::router(fin_state))

        // Health check
        .route("/health", get(health::health_check))
        .with_state(app_state);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, app).await?;

    Ok(())
}
```

### Step 3: Update Cargo.toml Dependencies

Already done! ✅ The workspace Cargo.toml now includes:
```toml
rust_decimal = { version = "1.33", features = ["serde"] }
```

### Step 4: Build and Test

```bash
# Build the service (check for compilation errors)
cd rust/service
cargo build

# Run tests
cargo test

# Start the service
cargo run
```

### Step 5: Test the API

```bash
# Health check
curl http://localhost:8080/health

# Create a GL account
curl -X POST http://localhost:8080/v1/fin/accounts \
  -H 'Content-Type: application/json' \
  -d '{
    "account_number": "9999",
    "account_name": "Test Account",
    "account_type": "ASSET",
    "parent_account_id": null,
    "currency": "USD"
  }'

# List all GL accounts (including seed data)
curl http://localhost:8080/v1/fin/accounts

# Create an invoice (replace customer_id with valid UUID)
curl -X POST http://localhost:8080/v1/fin/invoices \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id": "650e8400-e29b-41d4-a716-446655440000",
    "invoice_date": "2025-12-30T10:00:00Z",
    "due_date": "2026-01-29T10:00:00Z",
    "currency": "USD",
    "lines": [{
      "description": "Professional Services",
      "quantity": 40,
      "unit_price": 150.00,
      "tax_rate": 0.075,
      "item_id": null
    }]
  }'
```

---

## 🔧 Advanced Configuration

### Environment Variables

```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost:5432/mycelix
FIN_DEFAULT_CURRENCY=USD
FIN_ENABLE_MULTI_CURRENCY=true
FIN_DKG_INTEGRATION=true
```

### Custom GL Account Setup

```bash
# Use the CLI to import your custom chart of accounts
cargo run --bin mycelix-cli -- fin accounts import chart_of_accounts.csv
```

---

## 🎯 Common Use Cases

### Use Case 1: Invoice-to-Payment Flow

```rust
// 1. Create customer invoice
let invoice = invoicing.create_invoice(CreateInvoiceRequest {
    customer_id: customer.id,
    invoice_date: Utc::now(),
    due_date: Utc::now() + Duration::days(30),
    currency: "USD".to_string(),
    lines: vec![...],
}).await?;

// 2. Send invoice to customer
let sent_invoice = invoicing.send_invoice(invoice.id).await?;

// 3. Record payment when received
let payment = payments.create_payment(CreatePaymentRequest {
    payment_type: PaymentType::Receivable,
    payment_date: Utc::now(),
    amount: invoice.total_amount,
    currency: "USD".to_string(),
    payment_method: PaymentMethod::BankTransfer,
    reference: Some("Wire Ref: 123456".to_string()),
    invoice_id: Some(invoice.id),
    bill_id: None,
}).await?;

// Payment service automatically updates invoice status to PAID
```

### Use Case 2: Bill Approval and Payment

```rust
// 1. Create vendor bill
let bill = invoicing.create_bill(CreateBillRequest {
    vendor_id: vendor.id,
    bill_date: Utc::now(),
    due_date: Utc::now() + Duration::days(15),
    currency: "USD".to_string(),
    lines: vec![...],
}).await?;

// 2. Approve bill for payment
let approved_bill = invoicing.approve_bill(bill.id).await?;

// 3. Make payment
let payment = payments.create_payment(CreatePaymentRequest {
    payment_type: PaymentType::Payable,
    payment_date: Utc::now(),
    amount: bill.total_amount,
    currency: "USD".to_string(),
    payment_method: PaymentMethod::Check,
    reference: Some("Check #12345".to_string()),
    invoice_id: None,
    bill_id: Some(bill.id),
}).await?;
```

### Use Case 3: Manual Journal Entry

```rust
// Create a manual adjustment entry
let entry = ledger.create_journal_entry(
    "Depreciation for December".to_string(),
    Some("DEP-2025-12".to_string()),
    current_user_id,
    vec![
        (equipment_account_id, None, Some(Decimal::new(50000, 2))),  // Credit Equipment
        (accum_deprec_id, Some(Decimal::new(50000, 2)), None),       // Debit Accumulated Depreciation
    ],
).await?;

// Post the entry to make it effective
let posted_entry = ledger.post_journal_entry(entry.id).await?;
```

---

## 🔍 Monitoring and Observability

### Key Metrics to Track

```rust
// Add Prometheus metrics
use prometheus::{Counter, Histogram};

lazy_static! {
    static ref INVOICES_CREATED: Counter =
        register_counter!("fin_invoices_created_total", "Total invoices created").unwrap();

    static ref PAYMENT_AMOUNT: Histogram =
        register_histogram!("fin_payment_amount_usd", "Payment amounts in USD").unwrap();
}

// In your code
INVOICES_CREATED.inc();
PAYMENT_AMOUNT.observe(payment.amount.to_f64().unwrap());
```

### Health Checks

```rust
// Check database connectivity
async fn fin_health_check(pool: &PgPool) -> Result<(), sqlx::Error> {
    sqlx::query("SELECT 1 FROM gl_accounts LIMIT 1")
        .fetch_one(pool)
        .await?;
    Ok(())
}
```

---

## 🐛 Troubleshooting

### Issue: Migration fails with "type account_type already exists"

**Solution**: Drop and recreate the database:
```bash
dropdb mycelix
createdb mycelix
sqlx migrate run
```

### Issue: Compilation error "cannot find rust_decimal"

**Solution**: Add to service Cargo.toml:
```toml
rust_decimal = { workspace = true }
```

### Issue: "Debits must equal credits" error

**Solution**: Ensure your journal entry lines balance:
```rust
// Total debits must equal total credits
let debit_total: Decimal = debits.iter().sum();
let credit_total: Decimal = credits.iter().sum();
assert_eq!(debit_total, credit_total);
```

### Issue: Invoice status not updating after payment

**Solution**: Check payment amount matches invoice total:
```rust
// Payment service automatically updates status
// But only if amounts are correct
assert!(payment.amount <= invoice.total_amount);
```

---

## 📊 Performance Optimization

### Database Indexes

Already created! ✅ The migration includes:
- `idx_gl_accounts_type` - Faster filtering by account type
- `idx_invoices_customer` - Faster customer invoice lookups
- `idx_payments_invoice` - Faster payment-to-invoice joins

### Query Optimization Tips

```rust
// Use eager loading for related data
let invoices_with_lines = sqlx::query_as!(
    InvoiceWithLines,
    r#"
    SELECT i.*, array_agg(il.*) as lines
    FROM invoices i
    JOIN invoice_lines il ON il.invoice_id = i.id
    WHERE i.customer_id = $1
    GROUP BY i.id
    "#,
    customer_id
).fetch_all(pool).await?;
```

---

## 🔒 Security Best Practices

### 1. Always Use Transactions

```rust
let mut tx = pool.begin().await?;
// ... multiple operations
tx.commit().await?;
```

### 2. Parameterized Queries Only

```rust
// ✅ SAFE
sqlx::query("SELECT * FROM invoices WHERE id = $1")
    .bind(invoice_id)

// ❌ DANGEROUS (SQL injection risk)
sqlx::query(&format!("SELECT * FROM invoices WHERE id = '{}'", invoice_id))
```

### 3. Validate Input

```rust
// Check amounts are positive
if req.amount <= Decimal::ZERO {
    return Err("Amount must be positive");
}

// Check currency is valid
if !["USD", "EUR", "GBP"].contains(&req.currency.as_str()) {
    return Err("Invalid currency code");
}
```

---

## 🎓 Learning Resources

- **Rust + SQLx**: https://docs.rs/sqlx/latest/sqlx/
- **Double-Entry Bookkeeping**: https://en.wikipedia.org/wiki/Double-entry_bookkeeping
- **Axum Framework**: https://docs.rs/axum/latest/axum/
- **Financial Reports**: https://www.investopedia.com/financial-statements-4689752

---

## 📞 Support

- **Issues**: https://github.com/Luminous-Dynamics/mycelix-supplychain/issues
- **Discussions**: https://github.com/Luminous-Dynamics/mycelix-supplychain/discussions
- **Email**: dev@luminous-dynamics.dev

---

**Quick Start Checklist**:
- [ ] Run migrations (`sqlx migrate run`)
- [ ] Update main.rs with FIN router
- [ ] Build and test (`cargo build && cargo test`)
- [ ] Start service (`cargo run`)
- [ ] Test API endpoints with curl
- [ ] Monitor logs for any errors

**Congratulations!** 🎉 Your Mycelix ERP now has a production-ready finance module!
