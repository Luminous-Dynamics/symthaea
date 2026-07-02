# Finance (FIN) Module - API Documentation

Comprehensive finance module for Mycelix ERP providing general ledger, accounts receivable/payable, invoicing, and financial reporting with cryptographic auditability.

## 🎯 Features

- **General Ledger** - Double-entry bookkeeping with cryptographic tamper detection
- **Invoicing** - Customer invoices with automatic GL integration
- **Bills** - Vendor bills and accounts payable
- **Payments** - Payment processing for both receivables and payables
- **Financial Reports** - Trial balance, income statement, balance sheet
- **Audit Trail** - Every transaction linked to DKG claims for immutability
- **Multi-Currency** - Support for multiple currencies per account
- **Dimensions** - Track by cost center, department, or custom dimensions

## 📊 Architecture

```
┌─────────────────────┐
│   FIN Module API    │
│  (REST Endpoints)   │
└──────────┬──────────┘
           │
    ┌──────┴──────┬──────────────┬──────────────┐
    │             │              │              │
┌───▼───┐   ┌────▼────┐   ┌─────▼────┐  ┌──────▼──────┐
│Ledger │   │Invoicing│   │ Payments │  │  Reporting  │
│Service│   │ Service │   │  Service │  │   Service   │
└───┬───┘   └────┬────┘   └─────┬────┘  └──────┬──────┘
    │            │              │              │
    └────────────┴──────────────┴──────────────┘
                      │
              ┌───────▼────────┐
              │  PostgreSQL    │
              │  + DKG Claims  │
              └────────────────┘
```

## 🚀 Quick Start

### 1. Run Database Migrations

```bash
# Create finance tables
sqlx migrate run --source migrations/
```

### 2. Initialize Services

```rust
use sqlx::PgPool;
use provenance_service::fin::{LedgerService, InvoicingService, PaymentService};

// Initialize services
let pool = PgPool::connect("postgresql://...").await?;
let ledger = LedgerService::new(pool.clone());
let invoicing = InvoicingService::new(pool.clone());
let payments = PaymentService::new(pool);
```

### 3. Create GL Accounts

```rust
use provenance_service::fin::models::*;

let account = ledger.create_account(CreateGlAccountRequest {
    account_number: "1100".to_string(),
    account_name: "Accounts Receivable".to_string(),
    account_type: AccountType::Asset,
    parent_account_id: None,
    currency: "USD".to_string(),
}).await?;
```

### 4. Create an Invoice

```rust
let invoice = invoicing.create_invoice(CreateInvoiceRequest {
    customer_id: customer_id,
    invoice_date: Utc::now(),
    due_date: Utc::now() + Duration::days(30),
    currency: "USD".to_string(),
    lines: vec![
        CreateInvoiceLineRequest {
            description: "Professional Services".to_string(),
            quantity: rust_decimal::Decimal::new(40, 0),  // 40 hours
            unit_price: rust_decimal::Decimal::new(15000, 2),  // $150.00
            tax_rate: Some(rust_decimal::Decimal::new(75, 3)),  // 7.5%
            item_id: None,
        }
    ],
}).await?;
```

### 5. Record a Payment

```rust
let payment = payments.create_payment(CreatePaymentRequest {
    payment_type: PaymentType::Receivable,
    payment_date: Utc::now(),
    amount: rust_decimal::Decimal::new(600000, 2),  // $6,000.00
    currency: "USD".to_string(),
    payment_method: PaymentMethod::BankTransfer,
    reference: Some("Wire Ref: 123456".to_string()),
    invoice_id: Some(invoice.id),
    bill_id: None,
}).await?;
```

## 📖 API Endpoints

### General Ledger Accounts

#### `POST /v1/fin/accounts`
Create a new GL account.

**Request:**
```json
{
  "account_number": "1000",
  "account_name": "Cash",
  "account_type": "ASSET",
  "parent_account_id": null,
  "currency": "USD"
}
```

**Response:** `201 Created`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "account_number": "1000",
  "account_name": "Cash",
  "account_type": "ASSET",
  "parent_account_id": null,
  "is_active": true,
  "currency": "USD",
  "created_at": "2025-12-30T10:00:00Z",
  "updated_at": "2025-12-30T10:00:00Z"
}
```

#### `GET /v1/fin/accounts`
List all active GL accounts.

**Response:** `200 OK`
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "account_number": "1000",
    "account_name": "Cash",
    "account_type": "ASSET",
    ...
  }
]
```

#### `GET /v1/fin/accounts/:id`
Get a specific GL account.

**Response:** `200 OK` or `404 Not Found`

### Invoices

#### `POST /v1/fin/invoices`
Create a new customer invoice.

**Request:**
```json
{
  "customer_id": "650e8400-e29b-41d4-a716-446655440000",
  "invoice_date": "2025-12-30T10:00:00Z",
  "due_date": "2026-01-29T10:00:00Z",
  "currency": "USD",
  "lines": [
    {
      "description": "Professional Services - Web Development",
      "quantity": 40,
      "unit_price": 150.00,
      "tax_rate": 0.075,
      "item_id": null
    }
  ]
}
```

**Response:** `201 Created`
```json
{
  "id": "750e8400-e29b-41d4-a716-446655440000",
  "invoice_number": "INV-750E8400",
  "customer_id": "650e8400-e29b-41d4-a716-446655440000",
  "invoice_date": "2025-12-30T10:00:00Z",
  "due_date": "2026-01-29T10:00:00Z",
  "currency": "USD",
  "subtotal": 6000.00,
  "tax_amount": 450.00,
  "total_amount": 6450.00,
  "status": "DRAFT",
  "journal_entry_id": null,
  "claim_id": null,
  "created_at": "2025-12-30T10:00:00Z",
  "updated_at": "2025-12-30T10:00:00Z"
}
```

#### `GET /v1/fin/invoices`
List all invoices (most recent first).

#### `GET /v1/fin/invoices/:id`
Get a specific invoice.

#### `POST /v1/fin/invoices/:id/send`
Mark invoice as sent to customer.

### Bills (Accounts Payable)

#### `POST /v1/fin/bills`
Create a new vendor bill (NOT YET IMPLEMENTED).

#### `GET /v1/fin/bills`
List all bills (NOT YET IMPLEMENTED).

#### `GET /v1/fin/bills/:id`
Get a specific bill (NOT YET IMPLEMENTED).

#### `POST /v1/fin/bills/:id/approve`
Approve a bill for payment (NOT YET IMPLEMENTED).

### Payments

#### `POST /v1/fin/payments`
Record a payment for an invoice or bill.

**Request:**
```json
{
  "payment_type": "RECEIVABLE",
  "payment_date": "2025-12-30T10:00:00Z",
  "amount": 6450.00,
  "currency": "USD",
  "payment_method": "BANK_TRANSFER",
  "reference": "Wire Ref: 123456",
  "invoice_id": "750e8400-e29b-41d4-a716-446655440000",
  "bill_id": null
}
```

**Response:** `201 Created`

#### `GET /v1/fin/payments`
List all payments.

#### `GET /v1/fin/payments/:id`
Get a specific payment.

### Financial Reports

#### `GET /v1/fin/reports/trial-balance?as_of_date=2025-12-31`
Generate trial balance report (NOT YET IMPLEMENTED).

#### `GET /v1/fin/reports/income-statement?start_date=2025-01-01&end_date=2025-12-31`
Generate income statement (NOT YET IMPLEMENTED).

#### `GET /v1/fin/reports/balance-sheet?as_of_date=2025-12-31`
Generate balance sheet (NOT YET IMPLEMENTED).

## 🔒 Security Features

### 1. Cryptographic Tamper Detection
Every journal entry has a SHA-256 hash of all line items:
```rust
lines_hash = SHA256(account_id || debit || credit || description ...)
```

### 2. DKG Claim Integration
All posted entries can be linked to DKG claims for immutable audit trail:
```rust
// When posting journal entry
let claim = create_dkg_claim(&entry)?;
entry.claim_id = Some(claim.id);
```

### 3. Double-Entry Validation
All journal entries enforce debits = credits:
```rust
if total_debits != total_credits {
    return Err("Debits must equal credits");
}
```

## 📊 Database Schema

### Key Tables

- `gl_accounts` - Chart of accounts
- `journal_entries` - Double-entry journal
- `journal_lines` - Individual debit/credit lines
- `invoices` - Customer invoices
- `invoice_lines` - Invoice line items
- `bills` - Vendor bills
- `bill_lines` - Bill line items
- `payments` - Payment records

See `migrations/001_create_fin_tables.sql` for complete schema.

## 🎯 Roadmap

### ✅ Implemented (v0.1)
- GL accounts CRUD
- Invoice creation and listing
- Payment recording and status updates
- Database schema with standard chart of accounts

### 🚧 In Progress
- Journal entry creation and posting
- Bill creation and approval
- Financial reports (trial balance, P&L, balance sheet)
- DKG claim integration

### 🔮 Planned (v0.2+)
- Multi-currency conversion
- Recurring invoices
- Payment terms and aging reports
- Budget tracking
- Tax calculation engine
- Bank reconciliation
- Expense claims
- Fixed asset management

## 💡 Best Practices

### 1. Always Use Transactions
```rust
let mut tx = pool.begin().await?;
// ... multiple database operations
tx.commit().await?;
```

### 2. Validate Before Creating
```rust
// Check customer exists
if !customer_exists(customer_id).await? {
    return Err("Customer not found");
}
```

### 3. Use rust_decimal for Money
```rust
// ✅ CORRECT
let amount = rust_decimal::Decimal::new(15000, 2);  // $150.00

// ❌ WRONG (floating point errors)
let amount = 150.00_f64;
```

### 4. Link to DKG for Auditability
```rust
// Create DKG claim for immutable record
let claim = dkg_client.create_claim(&entry_data).await?;
entry.claim_id = Some(claim.id);
```

## 🧪 Testing

```bash
# Run all finance module tests
cargo test fin::

# Test specific service
cargo test fin::ledger::tests

# Integration tests
cargo test --test integration_fin
```

## 📝 License

Apache-2.0 - See LICENSE file

## 🤝 Contributing

See main CONTRIBUTING.md for guidelines.

---

**Status**: Alpha - Core functionality implemented, extended features in progress.
**Module Owner**: FIN team
**Last Updated**: 2025-12-30
