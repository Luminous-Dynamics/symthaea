# Mycelix ERP: Technical Architecture Document

**Version**: 1.0
**Date**: December 30, 2025
**Status**: Draft
**Authors**: Tristan Stoltz, Claude Code

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architectural Principles](#3-architectural-principles)
4. [Core Architecture](#4-core-architecture)
5. [Module Architecture](#5-module-architecture)
6. [Shared Infrastructure](#6-shared-infrastructure)
7. [Data Architecture](#7-data-architecture)
8. [API Architecture](#8-api-architecture)
9. [Security Architecture](#9-security-architecture)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Integration Architecture](#11-integration-architecture)
12. [Scalability & Performance](#12-scalability--performance)
13. [Development Architecture](#13-development-architecture)
14. [Migration & Evolution](#14-migration--evolution)
15. [Appendices](#15-appendices)

---

## 1. Executive Summary

### 1.1 Vision

**Mycelix ERP** is the world's first decentralized Enterprise Resource Planning system with cryptographic provenance. It provides comprehensive business management capabilities (Supply Chain, Finance, CRM, Manufacturing, HR, Projects, Assets) while ensuring:

- **Data Sovereignty**: Organizations own their data, not the vendor
- **Cryptographic Proof**: Every transaction is verifiable via W3C Verifiable Credentials
- **Privacy by Design**: Selective disclosure using ZK-proofs (SD-JWT/BBS+)
- **Zero Vendor Lock-In**: Open protocols, standard formats, exportable data
- **Agent-Centric**: Built on Holochain DHT principles
- **Production-Grade**: Rust performance, comprehensive testing, enterprise security

### 1.2 Current State (v0.4.0)

**Production-Ready Components**:
- ✅ Supply Chain Management (SCM) module
- ✅ Cryptographic provenance infrastructure
- ✅ Verifiable Credentials system
- ✅ REST API with OpenAPI specification
- ✅ TypeScript SDK
- ✅ PostgreSQL persistence
- ✅ Docker/Kubernetes deployment
- ✅ 100% test pass rate
- ✅ Comprehensive documentation

**Statistics**:
- **Lines of Code**: ~15,000 (Rust + TypeScript)
- **Test Coverage**: >90%
- **Performance**: <30ms for complex lineage queries
- **Security**: 5/5 production readiness (OWASP headers, input validation, CORS)
- **API Endpoints**: 15 (SCM module)

### 1.3 Target State (v2.0 - 18 Months)

**Complete ERP Suite**:
- ✅ **SCM** - Supply Chain Management (complete)
- 🚧 **FIN** - Financial Management (planned)
- 🚧 **CRM** - Customer Relationship Management (planned)
- 🚧 **MRP** - Manufacturing Resource Planning (planned)
- 🚧 **HR** - Human Resources (planned)
- 🚧 **PM** - Project Management (planned)
- 🚧 **ASSET** - Asset Management (planned)

**Projected Statistics**:
- **Lines of Code**: ~100,000
- **Test Coverage**: >90%
- **API Endpoints**: 250+
- **Supported Industries**: Manufacturing, Healthcare, Finance, Food/Beverage, Defense
- **Deployment Options**: SaaS, Self-Hosted, Hybrid
- **Pricing**: $25-$75/user/month (vs SAP $150-$300)

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Web Dashboard│  │ Mobile Apps  │  │  CLI Tools   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────┬─────────────────┬───────────────┬─────────────────┘
             │                 │               │
             └─────────────────┴───────────────┘
                               │
                      ┌────────▼────────┐
                      │   TypeScript    │
                      │      SDK        │
                      └────────┬────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        │         API Gateway (Optional)               │
        │         Traefik / Envoy / Nginx             │
        └──────────────────────┬──────────────────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        │            Mycelix ERP Services              │
        ├──────────────────────────────────────────────┤
        │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐        │
        │  │ SCM │  │ FIN │  │ CRM │  │ MRP │  ...   │
        │  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘        │
        │     └────────┴────────┴────────┘            │
        │              │                               │
        │  ┌───────────▼──────────────────┐           │
        │  │   Shared Infrastructure      │           │
        │  │ ┌──────┐ ┌──────┐ ┌───────┐ │           │
        │  │ │ Auth │ │ Crypto│ │ DKG  │ │           │
        │  │ └──────┘ └──────┘ └───────┘ │           │
        │  └──────────────────────────────┘           │
        └──────────────────────┬──────────────────────┘
                               │
        ┌──────────────────────┴──────────────────────┐
        │            Data Layer                        │
        ├──────────────────────────────────────────────┤
        │  ┌─────────────┐     ┌──────────────┐       │
        │  │ PostgreSQL  │     │  Holochain   │       │
        │  │   (ACID)    │     │    DHT       │       │
        │  └─────────────┘     └──────────────┘       │
        │  ┌─────────────┐     ┌──────────────┐       │
        │  │ TimescaleDB │     │ Meilisearch  │       │
        │  │ (Analytics) │     │   (Search)   │       │
        │  └─────────────┘     └──────────────┘       │
        └──────────────────────────────────────────────┘
```

### 2.2 Technology Stack

#### Core Services (Rust)
```toml
[dependencies]
# Web Framework
axum = "0.7"              # Modern async web framework
tower = "0.4"             # Middleware
tower-http = "0.5"        # HTTP utilities

# Async Runtime
tokio = { version = "1", features = ["full"] }

# Database
sqlx = { version = "0.7", features = ["postgres", "sqlite", "runtime-tokio-rustls"] }
sea-orm = "0.12"          # Optional ORM for complex queries

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Cryptography
ed25519-dalek = "2"       # Ed25519 signatures
ring = "0.17"             # General cryptography
jsonwebtoken = "9"        # JWT handling
bbs = "0.4"               # BBS+ signatures

# Decimal Math (Finance)
rust_decimal = "1.33"
rust_decimal_macros = "1.33"

# Date/Time
chrono = { version = "0.4", features = ["serde"] }

# Error Handling
thiserror = "1"
anyhow = "1"

# Validation
validator = { version = "0.16", features = ["derive"] }

# Configuration
config = "0.13"
dotenvy = "0.15"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

# Metrics
prometheus = "0.13"

# Testing
mockito = "1"
wiremock = "0.6"
```

#### Client SDK (TypeScript)
```json
{
  "dependencies": {
    "axios": "^1.6.0",
    "zod": "^3.22.0",
    "@veramo/core": "^4.2.0",
    "decimal.js": "^10.4.0",
    "date-fns": "^3.0.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "vitest": "^1.0.0",
    "@types/node": "^20.10.0"
  }
}
```

#### Infrastructure
- **Database**: PostgreSQL 15+ (primary), SQLite (dev/testing)
- **Time-Series**: TimescaleDB (analytics, reporting)
- **Search**: Meilisearch or Typesense (full-text search)
- **Cache**: Redis (optional, for high-scale deployments)
- **Message Queue**: RabbitMQ or NATS (inter-service communication)
- **Container**: Docker, Docker Compose
- **Orchestration**: Kubernetes (production)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Logging**: Loki or ELK stack

### 2.3 Design Philosophy

**1. Modular Monolith First, Microservices Later**
- Single binary with multiple modules
- Extract to microservices only when scaling demands
- Shared transaction boundaries, easier development

**2. Cryptographic Provenance by Default**
- Every entity is a Verifiable Credential
- W3C VC standard compliance
- Selective disclosure (SD-JWT/BBS+)
- Tamper-evident audit trails

**3. Agent-Centric Architecture**
- Users own their data (Holochain principles)
- DIDs for identity
- Optional centralized deployment for convenience

**4. API-First Development**
- OpenAPI specification before implementation
- TypeScript SDK auto-generated
- Versioned APIs (/v1/, /v2/)

**5. Zero-Trust Security**
- Authenticate every request
- Validate all inputs
- Encrypt sensitive data at rest
- Audit all state changes

**6. Developer Experience Priority**
- Comprehensive documentation
- Example code for every feature
- Error messages with actionable suggestions
- Fast local development environment

---

## 3. Architectural Principles

### 3.1 SOLID Principles (Rust)

**Single Responsibility**
```rust
// ❌ BAD: Mixed responsibilities
struct InvoiceService {
    fn create_invoice() { }
    fn send_email() { }
    fn generate_pdf() { }
}

// ✅ GOOD: Separated concerns
struct InvoiceService {
    fn create_invoice() { }
}
struct EmailService {
    fn send_invoice_email() { }
}
struct PdfGenerator {
    fn generate_invoice_pdf() { }
}
```

**Open/Closed Principle**
```rust
// Extend via traits, not modification
trait EventHandler {
    fn handle(&self, event: &Event) -> Result<()>;
}

struct InvoiceCreatedHandler;
impl EventHandler for InvoiceCreatedHandler {
    fn handle(&self, event: &Event) -> Result<()> {
        // Handle invoice created
    }
}
```

**Dependency Inversion**
```rust
// Depend on abstractions, not concretions
trait InvoiceRepository {
    async fn create(&self, invoice: &Invoice) -> Result<String>;
    async fn get(&self, id: &str) -> Result<Option<Invoice>>;
}

struct PostgresInvoiceRepository { /* ... */ }
impl InvoiceRepository for PostgresInvoiceRepository { /* ... */ }

struct InvoiceService<R: InvoiceRepository> {
    repo: R,
}
```

### 3.2 Domain-Driven Design

**Bounded Contexts**
- Each module (SCM, FIN, CRM, etc.) is a bounded context
- Clear boundaries, well-defined interfaces
- No direct database access across modules

**Ubiquitous Language**
- Use business terminology in code
- `Invoice` not `FinancialDocument001`
- `WorkOrder` not `ManufacturingTask`

**Aggregates**
```rust
// Invoice is an aggregate root
pub struct Invoice {
    pub invoice_id: String,           // Aggregate ID
    pub line_items: Vec<LineItem>,    // Owned entities
    // ...
}

impl Invoice {
    // Business logic encapsulated
    pub fn add_line_item(&mut self, item: LineItem) -> Result<()> {
        self.validate_line_item(&item)?;
        self.line_items.push(item);
        self.recalculate_total();
        Ok(())
    }
}
```

**Domain Events**
```rust
pub enum DomainEvent {
    InvoiceCreated { invoice_id: String, customer_id: String },
    InvoicePaid { invoice_id: String, amount: Decimal },
    ShipmentArrived { shipment_id: String, facility_id: String },
}

// Events trigger cross-module communication
```

### 3.3 CQRS (Command Query Responsibility Segregation)

**Commands** (Write Operations)
```rust
pub struct CreateInvoiceCommand {
    pub customer_id: String,
    pub line_items: Vec<CreateLineItemDto>,
}

pub async fn handle_create_invoice(
    cmd: CreateInvoiceCommand,
    service: &InvoiceService,
) -> Result<String> {
    // Validate
    // Execute business logic
    // Persist
    // Emit event
}
```

**Queries** (Read Operations)
```rust
pub struct InvoiceQuery {
    pub customer_id: Option<String>,
    pub status: Option<InvoiceStatus>,
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
}

pub async fn query_invoices(
    query: InvoiceQuery,
    repo: &dyn InvoiceRepository,
) -> Result<Vec<Invoice>> {
    // Optimized read path
}
```

**Benefits**:
- Write path optimized for consistency
- Read path optimized for performance
- Can scale independently

### 3.4 Event-Driven Architecture

```rust
pub trait EventBus {
    async fn publish(&self, event: DomainEvent) -> Result<()>;
    async fn subscribe<F>(&self, handler: F)
    where
        F: Fn(DomainEvent) -> BoxFuture<'static, Result<()>> + Send + Sync;
}

// Example: Invoice created → Generate accounting entry
impl EventHandler for AccountingEventHandler {
    async fn handle(&self, event: DomainEvent) -> Result<()> {
        match event {
            DomainEvent::InvoiceCreated { invoice_id, .. } => {
                let invoice = self.invoice_repo.get(&invoice_id).await?;
                self.create_gl_entries(invoice).await?;
            }
            _ => {}
        }
        Ok(())
    }
}
```

---

## 4. Core Architecture

### 4.1 Repository Structure

```
mycelix-erp/
├── Cargo.toml                    # Workspace root
├── README.md                     # Project overview
├── LICENSE                       # Apache 2.0
├── ARCHITECTURE.md              # This document
├── ROADMAP.md                   # 18-month plan
│
├── services/                    # All ERP modules
│   ├── scm/                     # Supply Chain Management
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── main.rs          # Service entry point
│   │   │   ├── api/             # REST API handlers
│   │   │   ├── models/          # Domain models
│   │   │   ├── handlers/        # Business logic
│   │   │   ├── db/              # Database access
│   │   │   └── lib.rs
│   │   ├── migrations/          # SQL migrations
│   │   └── tests/               # Integration tests
│   │
│   ├── fin/                     # Financial Management
│   │   └── (same structure)
│   │
│   ├── crm/                     # Customer Relationship
│   ├── mrp/                     # Manufacturing
│   ├── hr/                      # Human Resources
│   ├── pm/                      # Project Management
│   └── asset/                   # Asset Management
│
├── shared/                      # Shared infrastructure
│   ├── crypto/                  # Cryptographic utilities
│   │   ├── src/
│   │   │   ├── signing.rs       # Ed25519, BBS+
│   │   │   ├── jwt.rs           # JWT handling
│   │   │   ├── vc.rs            # Verifiable Credentials
│   │   │   └── lib.rs
│   │   └── tests/
│   │
│   ├── auth/                    # Authentication/Authorization
│   │   ├── src/
│   │   │   ├── jwt.rs           # JWT validation
│   │   │   ├── rbac.rs          # Role-based access control
│   │   │   ├── did.rs           # DID support
│   │   │   └── lib.rs
│   │   └── tests/
│   │
│   ├── models/                  # Shared domain models
│   │   ├── src/
│   │   │   ├── organization.rs
│   │   │   ├── user.rs
│   │   │   ├── address.rs
│   │   │   ├── money.rs         # Decimal currency handling
│   │   │   └── lib.rs
│   │   └── tests/
│   │
│   ├── storage/                 # Database abstraction
│   │   ├── src/
│   │   │   ├── postgres.rs
│   │   │   ├── sqlite.rs
│   │   │   ├── migrations.rs
│   │   │   └── lib.rs
│   │   └── tests/
│   │
│   ├── events/                  # Event bus
│   │   ├── src/
│   │   │   ├── bus.rs           # In-memory event bus
│   │   │   ├── nats.rs          # NATS integration
│   │   │   └── lib.rs
│   │   └── tests/
│   │
│   └── dkg/                     # DKG integration
│       ├── src/
│       │   ├── claim.rs
│       │   ├── epistemic.rs
│       │   └── lib.rs
│       └── tests/
│
├── sdk/                         # Client SDKs
│   ├── typescript/
│   │   ├── package.json
│   │   ├── src/
│   │   │   ├── client.ts        # Main API client
│   │   │   ├── modules/         # Per-module SDKs
│   │   │   │   ├── scm.ts
│   │   │   │   ├── fin.ts
│   │   │   │   └── crm.ts
│   │   │   ├── types/           # TypeScript definitions
│   │   │   └── index.ts
│   │   ├── tests/
│   │   └── README.md
│   │
│   ├── python/                  # Future: Python SDK
│   └── rust/                    # Future: Rust SDK (for extensions)
│
├── gateway/                     # API Gateway (optional)
│   ├── traefik/
│   │   └── traefik.yml
│   └── envoy/
│       └── envoy.yaml
│
├── infrastructure/              # Deployment configs
│   ├── docker/
│   │   ├── Dockerfile           # Multi-stage build
│   │   ├── docker-compose.yml   # Local development
│   │   └── docker-compose.prod.yml
│   │
│   ├── kubernetes/
│   │   ├── namespace.yaml
│   │   ├── deployments/
│   │   ├── services/
│   │   ├── ingress.yaml
│   │   └── configmaps/
│   │
│   └── terraform/               # Infrastructure as Code
│       ├── aws/
│       ├── gcp/
│       └── azure/
│
├── docs/                        # Documentation
│   ├── index.md                 # Landing page
│   ├── architecture/
│   │   ├── overview.md
│   │   ├── modules/
│   │   └── deployment.md
│   ├── api/
│   │   ├── openapi.yaml         # Combined spec
│   │   └── modules/             # Per-module specs
│   ├── guides/
│   │   ├── quickstart.md
│   │   ├── deployment.md
│   │   └── integration.md
│   └── examples/                # Code examples
│
├── scripts/                     # Development scripts
│   ├── dev-setup.sh
│   ├── test-all.sh
│   ├── build-all.sh
│   └── deploy.sh
│
└── tests/                       # E2E tests
    ├── integration/
    ├── performance/
    └── security/
```

### 4.2 Module Communication Patterns

**Pattern 1: Synchronous API Calls** (Rare)
```rust
// ❌ Avoid tight coupling
// FIN module directly calling CRM module
let customer = crm_service.get_customer(&customer_id).await?;
```

**Pattern 2: Domain Events** (Preferred)
```rust
// ✅ Loose coupling via events
// FIN module publishes event
event_bus.publish(DomainEvent::InvoicePaid {
    invoice_id,
    customer_id,
    amount
}).await?;

// CRM module subscribes
impl EventHandler for CustomerHandler {
    async fn handle(&self, event: DomainEvent) -> Result<()> {
        match event {
            DomainEvent::InvoicePaid { customer_id, amount, .. } => {
                self.update_customer_lifetime_value(customer_id, amount).await?;
            }
            _ => {}
        }
        Ok(())
    }
}
```

**Pattern 3: Shared Database** (For reads)
```rust
// Read-only queries across modules (same database)
// FIN can read CRM customer table for lookups
let customer_name = sqlx::query_scalar!(
    "SELECT name FROM crm_customers WHERE customer_id = $1",
    customer_id
)
.fetch_one(&pool)
.await?;
```

**Pattern 4: API Gateway Aggregation**
```rust
// Client requests aggregated data
// Gateway calls multiple modules and combines
GET /v1/dashboard/customer/{id}

// Returns:
{
  "customer": { /* from CRM */ },
  "invoices": [ /* from FIN */ ],
  "shipments": [ /* from SCM */ ]
}
```

---

## 5. Module Architecture

### 5.1 Supply Chain Management (SCM) - Complete

**Status**: ✅ Production-ready (v0.4.0)

**Core Entities**:
```rust
pub struct SupplyEvent {
    pub event_id: String,
    pub event_type: EventType,        // PRODUCED, TRANSFORMED, SHIPPED, etc.
    pub product_id: String,
    pub batch_id: String,
    pub quantity: Decimal,
    pub unit: String,
    pub facility_id: String,
    pub timestamp: DateTime<Utc>,
    pub previous_batches: Vec<String>, // Lineage
    pub metadata: serde_json::Value,
}

pub struct VerifiableCredential {
    pub context: Vec<String>,
    pub type_: Vec<String>,
    pub issuer: String,               // DID
    pub issuance_date: DateTime<Utc>,
    pub credential_subject: serde_json::Value,
    pub proof: Proof,                 // Ed25519 signature
}

pub struct DKGClaim {
    pub claim_id: String,
    pub event_type: String,
    pub product_id: String,
    pub lineage_hash: String,         // Hash of previous claims
    pub vc_jwt: String,
    pub timestamp: DateTime<Utc>,
}
```

**API Endpoints**:
```
POST   /v1/scm/events               # Ingest supply chain event
GET    /v1/scm/events/:id           # Get event by ID
GET    /v1/scm/events?batch_id={id} # Query by batch
POST   /v1/scm/events/batch         # Batch ingestion (up to 100)

GET    /v1/scm/claims/:id           # Get DKG claim
GET    /v1/scm/lineage/upstream/:id # Trace upstream
GET    /v1/scm/lineage/downstream/:id # Trace downstream

POST   /v1/scm/verify               # Verify VC signature
GET    /v1/scm/passport/:batch_id   # Generate product passport
```

**Database Schema**:
```sql
-- See services/scm/migrations/001_initial.sql
-- See services/scm/migrations/002_performance_indexes.sql
```

**Integration Points**:
- **→ FIN**: Shipment received → Create bill
- **→ MRP**: Raw material received → Update inventory
- **← CRM**: Sales order → Create shipment

### 5.2 Financial Management (FIN) - Planned (Phase 1)

**Target**: Months 1-3

**Core Entities**:
```rust
// General Ledger Entry
pub struct GLEntry {
    pub entry_id: String,
    pub organization_id: String,
    pub account_id: String,          // Chart of accounts
    pub debit: Option<Decimal>,
    pub credit: Option<Decimal>,
    pub reference: String,            // Invoice/Bill/PO number
    pub description: String,
    pub timestamp: DateTime<Utc>,
    pub created_by: String,
    pub vc: VerifiableCredential,    // Cryptographic proof
}

// Chart of Accounts
pub struct Account {
    pub account_id: String,
    pub account_code: String,         // e.g., "1000"
    pub name: String,                 // e.g., "Cash"
    pub account_type: AccountType,    // Asset, Liability, Equity, Revenue, Expense
    pub parent_account: Option<String>,
    pub is_active: bool,
}

pub enum AccountType {
    Asset,
    Liability,
    Equity,
    Revenue,
    Expense,
}

// Accounts Receivable - Invoice
pub struct Invoice {
    pub invoice_id: String,
    pub organization_id: String,
    pub customer_id: String,
    pub invoice_number: String,       // Human-readable
    pub issue_date: DateTime<Utc>,
    pub due_date: DateTime<Utc>,
    pub line_items: Vec<InvoiceLineItem>,
    pub subtotal: Decimal,
    pub tax: Decimal,
    pub total: Decimal,
    pub amount_paid: Decimal,
    pub status: InvoiceStatus,
    pub payment_terms: PaymentTerms,
    pub notes: Option<String>,
    pub scm_claims: Vec<String>,      // Link to shipments
    pub vc: VerifiableCredential,
}

pub struct InvoiceLineItem {
    pub line_item_id: String,
    pub description: String,
    pub quantity: Decimal,
    pub unit_price: Decimal,
    pub total: Decimal,
    pub product_id: Option<String>,
    pub account_code: String,         // Revenue account
}

pub enum InvoiceStatus {
    Draft,
    Sent,
    Viewed,
    PartiallyPaid,
    Paid,
    Overdue,
    Cancelled,
    Refunded,
}

pub struct PaymentTerms {
    pub net_days: u32,                // e.g., 30 for "Net 30"
    pub discount_days: Option<u32>,   // e.g., 10 for "2/10 Net 30"
    pub discount_percent: Option<Decimal>,
}

// Accounts Payable - Bill
pub struct Bill {
    pub bill_id: String,
    pub organization_id: String,
    pub vendor_id: String,
    pub bill_number: String,
    pub bill_date: DateTime<Utc>,
    pub due_date: DateTime<Utc>,
    pub line_items: Vec<BillLineItem>,
    pub subtotal: Decimal,
    pub tax: Decimal,
    pub total: Decimal,
    pub amount_paid: Decimal,
    pub status: BillStatus,
    pub purchase_order_id: Option<String>,
    pub scm_claims: Vec<String>,      // Link to received shipments
    pub vc: VerifiableCredential,
}

pub struct BillLineItem {
    pub line_item_id: String,
    pub description: String,
    pub quantity: Decimal,
    pub unit_price: Decimal,
    pub total: Decimal,
    pub product_id: Option<String>,
    pub account_code: String,         // Expense/Asset account
}

pub enum BillStatus {
    Draft,
    Submitted,
    Approved,
    PartiallyPaid,
    Paid,
    Overdue,
    Cancelled,
}

// Payment
pub struct Payment {
    pub payment_id: String,
    pub organization_id: String,
    pub payment_type: PaymentType,    // Invoice payment, Bill payment
    pub reference_id: String,          // Invoice ID or Bill ID
    pub amount: Decimal,
    pub payment_date: DateTime<Utc>,
    pub payment_method: PaymentMethod,
    pub notes: Option<String>,
    pub vc: VerifiableCredential,
}

pub enum PaymentType {
    InvoicePayment,
    BillPayment,
}

pub enum PaymentMethod {
    Cash,
    Check { check_number: String },
    BankTransfer { reference: String },
    CreditCard { last_four: String },
    Other { description: String },
}
```

**API Endpoints**:
```
# General Ledger
POST   /v1/fin/gl/entries
GET    /v1/fin/gl/entries
GET    /v1/fin/gl/accounts
POST   /v1/fin/gl/accounts

# Accounts Receivable
POST   /v1/fin/ar/invoices
GET    /v1/fin/ar/invoices
GET    /v1/fin/ar/invoices/:id
PUT    /v1/fin/ar/invoices/:id
DELETE /v1/fin/ar/invoices/:id
POST   /v1/fin/ar/invoices/:id/send
POST   /v1/fin/ar/invoices/:id/payments

# Accounts Payable
POST   /v1/fin/ap/bills
GET    /v1/fin/ap/bills
GET    /v1/fin/ap/bills/:id
PUT    /v1/fin/ap/bills/:id
DELETE /v1/fin/ap/bills/:id
POST   /v1/fin/ap/bills/:id/payments

# Reports
GET    /v1/fin/reports/balance-sheet?as_of={date}
GET    /v1/fin/reports/income-statement?start={date}&end={date}
GET    /v1/fin/reports/cash-flow?start={date}&end={date}
GET    /v1/fin/reports/trial-balance?as_of={date}
GET    /v1/fin/reports/aging-receivables
GET    /v1/fin/reports/aging-payables
```

**Database Schema**:
```sql
-- Chart of Accounts
CREATE TABLE fin_accounts (
    account_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    account_code TEXT NOT NULL,
    name TEXT NOT NULL,
    account_type TEXT NOT NULL,
    parent_account TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization_id, account_code)
);

-- General Ledger
CREATE TABLE fin_gl_entries (
    entry_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    account_id TEXT NOT NULL,
    debit NUMERIC(19, 4),
    credit NUMERIC(19, 4),
    reference TEXT NOT NULL,
    description TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    created_by TEXT NOT NULL,
    vc_jwt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES fin_accounts(account_id),
    CHECK ((debit IS NOT NULL AND credit IS NULL) OR (debit IS NULL AND credit IS NOT NULL))
);

-- Invoices
CREATE TABLE fin_invoices (
    invoice_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    issue_date TIMESTAMP NOT NULL,
    due_date TIMESTAMP NOT NULL,
    subtotal NUMERIC(19, 4) NOT NULL,
    tax NUMERIC(19, 4) NOT NULL,
    total NUMERIC(19, 4) NOT NULL,
    amount_paid NUMERIC(19, 4) DEFAULT 0,
    status TEXT NOT NULL,
    payment_terms JSONB,
    notes TEXT,
    vc_jwt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization_id, invoice_number)
);

CREATE TABLE fin_invoice_line_items (
    line_item_id TEXT PRIMARY KEY,
    invoice_id TEXT NOT NULL,
    description TEXT NOT NULL,
    quantity NUMERIC(19, 4) NOT NULL,
    unit_price NUMERIC(19, 4) NOT NULL,
    total NUMERIC(19, 4) NOT NULL,
    product_id TEXT,
    account_code TEXT NOT NULL,
    FOREIGN KEY (invoice_id) REFERENCES fin_invoices(invoice_id) ON DELETE CASCADE
);

-- Bills (similar structure to Invoices)
CREATE TABLE fin_bills (
    bill_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    vendor_id TEXT NOT NULL,
    bill_number TEXT NOT NULL,
    bill_date TIMESTAMP NOT NULL,
    due_date TIMESTAMP NOT NULL,
    subtotal NUMERIC(19, 4) NOT NULL,
    tax NUMERIC(19, 4) NOT NULL,
    total NUMERIC(19, 4) NOT NULL,
    amount_paid NUMERIC(19, 4) DEFAULT 0,
    status TEXT NOT NULL,
    purchase_order_id TEXT,
    vc_jwt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization_id, bill_number)
);

CREATE TABLE fin_bill_line_items (
    line_item_id TEXT PRIMARY KEY,
    bill_id TEXT NOT NULL,
    description TEXT NOT NULL,
    quantity NUMERIC(19, 4) NOT NULL,
    unit_price NUMERIC(19, 4) NOT NULL,
    total NUMERIC(19, 4) NOT NULL,
    product_id TEXT,
    account_code TEXT NOT NULL,
    FOREIGN KEY (bill_id) REFERENCES fin_bills(bill_id) ON DELETE CASCADE
);

-- Payments
CREATE TABLE fin_payments (
    payment_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    payment_type TEXT NOT NULL,
    reference_id TEXT NOT NULL,
    amount NUMERIC(19, 4) NOT NULL,
    payment_date TIMESTAMP NOT NULL,
    payment_method JSONB NOT NULL,
    notes TEXT,
    vc_jwt TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_gl_entries_account ON fin_gl_entries(account_id, timestamp DESC);
CREATE INDEX idx_gl_entries_reference ON fin_gl_entries(reference);
CREATE INDEX idx_invoices_customer ON fin_invoices(customer_id);
CREATE INDEX idx_invoices_status ON fin_invoices(status);
CREATE INDEX idx_invoices_due_date ON fin_invoices(due_date);
CREATE INDEX idx_bills_vendor ON fin_bills(vendor_id);
CREATE INDEX idx_bills_status ON fin_bills(status);
CREATE INDEX idx_bills_due_date ON fin_bills(due_date);
CREATE INDEX idx_payments_reference ON fin_payments(reference_id);
```

**Integration Points**:
- **← SCM**: Shipment received → Auto-create bill
- **← CRM**: Quote accepted → Create invoice
- **→ CRM**: Invoice overdue → Update customer status

**Cryptographic Features**:
```rust
impl Invoice {
    // Generate verifiable credential for invoice
    pub fn to_verifiable_credential(&self) -> VerifiableCredential {
        VC {
            context: vec![
                "https://www.w3.org/2018/credentials/v1",
                "https://schema.mycelix.net/fin/v1"
            ],
            type_: vec!["VerifiableCredential", "InvoiceCredential"],
            issuer: self.organization_did.clone(),
            issuance_date: Utc::now(),
            credential_subject: json!({
                "invoiceId": self.invoice_id,
                "invoiceNumber": self.invoice_number,
                "customer": self.customer_id,
                "amount": self.total,
                "currency": "USD",
                "dueDate": self.due_date,
                "status": self.status,
                "lineageLinks": self.scm_claims,
            }),
            proof: self.sign_with_ed25519(),
        }
    }

    // Generate ZK-proof of payment without revealing amount
    pub fn generate_payment_proof(&self) -> ZKProof {
        // Prove "invoice was paid in full" without revealing actual amount
        // Useful for credit checks
    }
}
```

### 5.3 Customer Relationship Management (CRM) - Planned (Phase 2)

**Target**: Months 4-6

**Core Entities**:
```rust
pub struct Customer {
    pub customer_id: String,
    pub organization_id: String,
    pub name: String,
    pub contact_info: ContactInfo,
    pub billing_address: Address,
    pub shipping_address: Option<Address>,
    pub payment_terms: PaymentTerms,
    pub credit_limit: Option<Decimal>,
    pub lifetime_value: Decimal,      // Auto-calculated
    pub status: CustomerStatus,
    pub tags: Vec<String>,
    pub custom_fields: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub struct ContactInfo {
    pub primary_email: String,
    pub primary_phone: Option<String>,
    pub website: Option<String>,
    pub contacts: Vec<Contact>,       // Multiple contacts per customer
}

pub struct Contact {
    pub contact_id: String,
    pub name: String,
    pub title: Option<String>,
    pub email: String,
    pub phone: Option<String>,
    pub is_primary: bool,
}

pub enum CustomerStatus {
    Active,
    Inactive,
    OnHold,
    Blacklisted,
}

// Lead Management
pub struct Lead {
    pub lead_id: String,
    pub organization_id: String,
    pub source: LeadSource,
    pub contact_info: ContactInfo,
    pub company_name: Option<String>,
    pub status: LeadStatus,
    pub score: i32,                   // Lead scoring 0-100
    pub assigned_to: Option<String>,  // Sales rep
    pub notes: Vec<Note>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub enum LeadSource {
    Website,
    Referral { from: String },
    TradeShow { event: String },
    ColdCall,
    Advertisement,
    Other { description: String },
}

pub enum LeadStatus {
    New,
    Contacted,
    Qualified,
    Unqualified,
    Converted { customer_id: String },
    Lost { reason: String },
}

// Opportunity / Deal
pub struct Opportunity {
    pub opportunity_id: String,
    pub organization_id: String,
    pub customer_id: Option<String>,  // If converted from lead
    pub lead_id: Option<String>,
    pub name: String,                 // e.g., "Q4 Widget Purchase"
    pub amount: Decimal,
    pub probability: f32,             // 0.0 - 1.0
    pub expected_close_date: DateTime<Utc>,
    pub stage: OpportunityStage,
    pub assigned_to: String,          // Sales rep
    pub quote_id: Option<String>,
    pub notes: Vec<Note>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub enum OpportunityStage {
    Prospecting,
    Qualification,
    NeedsAnalysis,
    Proposal,
    Negotiation,
    ClosedWon { order_id: String },
    ClosedLost { reason: String },
}

// Quote / Proposal
pub struct Quote {
    pub quote_id: String,
    pub organization_id: String,
    pub customer_id: String,
    pub quote_number: String,
    pub opportunity_id: Option<String>,
    pub line_items: Vec<QuoteLineItem>,
    pub subtotal: Decimal,
    pub discount: Option<Discount>,
    pub tax: Decimal,
    pub total: Decimal,
    pub valid_until: DateTime<Utc>,
    pub status: QuoteStatus,
    pub terms_and_conditions: String,
    pub vc: VerifiableCredential,    // Customer can verify quote
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub struct QuoteLineItem {
    pub line_item_id: String,
    pub product_id: String,
    pub description: String,
    pub quantity: Decimal,
    pub unit_price: Decimal,
    pub total: Decimal,
}

pub enum QuoteStatus {
    Draft,
    Sent,
    Viewed,
    Accepted,
    Rejected,
    Expired,
    Revised { from: String },        // Revised from another quote
}

pub struct Discount {
    pub discount_type: DiscountType,
    pub amount: Decimal,
}

pub enum DiscountType {
    Percentage,
    FixedAmount,
}
```

**API Endpoints**:
```
# Customers
POST   /v1/crm/customers
GET    /v1/crm/customers
GET    /v1/crm/customers/:id
PUT    /v1/crm/customers/:id
DELETE /v1/crm/customers/:id

# Leads
POST   /v1/crm/leads
GET    /v1/crm/leads
GET    /v1/crm/leads/:id
PUT    /v1/crm/leads/:id
POST   /v1/crm/leads/:id/convert         # Convert to customer
DELETE /v1/crm/leads/:id

# Opportunities
POST   /v1/crm/opportunities
GET    /v1/crm/opportunities
GET    /v1/crm/opportunities/:id
PUT    /v1/crm/opportunities/:id
POST   /v1/crm/opportunities/:id/close-won
POST   /v1/crm/opportunities/:id/close-lost
DELETE /v1/crm/opportunities/:id

# Quotes
POST   /v1/crm/quotes
GET    /v1/crm/quotes
GET    /v1/crm/quotes/:id
PUT    /v1/crm/quotes/:id
POST   /v1/crm/quotes/:id/send
POST   /v1/crm/quotes/:id/accept
POST   /v1/crm/quotes/:id/reject
DELETE /v1/crm/quotes/:id

# Reports
GET    /v1/crm/reports/sales-pipeline
GET    /v1/crm/reports/lead-conversion
GET    /v1/crm/reports/customer-lifetime-value
```

**Integration Points**:
- **→ FIN**: Quote accepted → Create invoice
- **→ SCM**: Sales order → Create shipment
- **← FIN**: Invoice overdue → Update customer status

### 5.4 Manufacturing Resource Planning (MRP) - Planned (Phase 3)

**Target**: Months 7-9

**Core Entities** (abbreviated for space):
```rust
pub struct BillOfMaterials {
    pub bom_id: String,
    pub product_id: String,
    pub version: u32,
    pub components: Vec<Component>,
    pub manufacturing_steps: Vec<ManufacturingStep>,
    pub vc: VerifiableCredential,
}

pub struct WorkOrder {
    pub work_order_id: String,
    pub bom_id: String,
    pub quantity_to_produce: Decimal,
    pub scheduled_start: DateTime<Utc>,
    pub actual_start: Option<DateTime<Utc>>,
    pub actual_completion: Option<DateTime<Utc>>,
    pub status: WorkOrderStatus,
    pub scm_batch_id: Option<String>,  // Link to produced batch
    pub vc: VerifiableCredential,
}
```

### 5.5 Human Resources (HR) - Planned (Phase 5)

**Target**: Months 13-15

(Details omitted for brevity - similar structure)

### 5.6 Project Management (PM) - Planned (Phase 6)

**Target**: Months 16-18

(Details omitted for brevity)

### 5.7 Asset Management (ASSET) - Planned (Phase 7)

**Target**: Months 18+

(Details omitted for brevity)

---

## 6. Shared Infrastructure

### 6.1 Cryptography (`shared/crypto`)

**Purpose**: Centralize all cryptographic operations

**Key Components**:

```rust
// Ed25519 Signing
pub struct Signer {
    keypair: ed25519_dalek::Keypair,
}

impl Signer {
    pub fn new() -> Self {
        let mut csprng = OsRng;
        let keypair = Keypair::generate(&mut csprng);
        Self { keypair }
    }

    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let secret = SecretKey::from_bytes(seed).unwrap();
        let public = PublicKey::from(&secret);
        Self { keypair: Keypair { secret, public } }
    }

    pub fn sign(&self, message: &[u8]) -> Signature {
        self.keypair.sign(message)
    }

    pub fn public_key(&self) -> PublicKey {
        self.keypair.public
    }
}

pub fn verify(
    public_key: &PublicKey,
    message: &[u8],
    signature: &Signature,
) -> bool {
    public_key.verify(message, signature).is_ok()
}

// JWT Handling
pub struct JWTService {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
}

impl JWTService {
    pub fn create_vc_jwt(&self, vc: &VerifiableCredential) -> Result<String> {
        let claims = VCClaims {
            sub: vc.credential_subject.to_string(),
            iss: vc.issuer.clone(),
            iat: vc.issuance_date.timestamp(),
            exp: vc.expiration_date.map(|d| d.timestamp()),
            vc: vc.clone(),
        };

        encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|e| CryptoError::JWTCreation(e.to_string()))
    }

    pub fn verify_vc_jwt(&self, jwt: &str) -> Result<VerifiableCredential> {
        let token_data = decode::<VCClaims>(
            jwt,
            &self.decoding_key,
            &Validation::default(),
        )
        .map_err(|e| CryptoError::JWTVerification(e.to_string()))?;

        Ok(token_data.claims.vc)
    }
}

// Verifiable Credentials
pub struct VCBuilder {
    context: Vec<String>,
    type_: Vec<String>,
    issuer: String,
    credential_subject: serde_json::Value,
}

impl VCBuilder {
    pub fn new(issuer: String) -> Self {
        Self {
            context: vec!["https://www.w3.org/2018/credentials/v1".to_string()],
            type_: vec!["VerifiableCredential".to_string()],
            issuer,
            credential_subject: json!({}),
        }
    }

    pub fn add_type(mut self, type_: String) -> Self {
        self.type_.push(type_);
        self
    }

    pub fn add_context(mut self, context: String) -> Self {
        self.context.push(context);
        self
    }

    pub fn credential_subject(mut self, subject: serde_json::Value) -> Self {
        self.credential_subject = subject;
        self
    }

    pub fn build(self, signer: &Signer) -> VerifiableCredential {
        let vc = VerifiableCredential {
            context: self.context,
            type_: self.type_,
            issuer: self.issuer,
            issuance_date: Utc::now(),
            credential_subject: self.credential_subject,
            proof: Proof::default(), // Placeholder
        };

        // Sign the VC
        let payload = serde_json::to_vec(&vc).unwrap();
        let signature = signer.sign(&payload);

        VerifiableCredential {
            proof: Proof::Ed25519 {
                signature: base64::encode(signature.to_bytes()),
                public_key: base64::encode(signer.public_key().to_bytes()),
            },
            ..vc
        }
    }
}

// Hashing
pub fn sha256(data: &[u8]) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

pub fn compute_lineage_hash(previous_hashes: &[String]) -> String {
    let combined = previous_hashes.join("");
    sha256(combined.as_bytes())
}
```

**Tests**:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_and_verify() {
        let signer = Signer::new();
        let message = b"test message";
        let signature = signer.sign(message);

        assert!(verify(&signer.public_key(), message, &signature));
    }

    #[test]
    fn test_vc_creation() {
        let signer = Signer::new();
        let vc = VCBuilder::new("did:mycelix:org123".to_string())
            .add_type("InvoiceCredential".to_string())
            .credential_subject(json!({
                "invoiceId": "INV-001",
                "amount": 1000.00
            }))
            .build(&signer);

        assert_eq!(vc.issuer, "did:mycelix:org123");
        assert!(vc.type_.contains(&"InvoiceCredential".to_string()));
    }
}
```

### 6.2 Authentication & Authorization (`shared/auth`)

**JWT-based Authentication**:
```rust
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};

#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,              // User ID
    pub org: String,              // Organization ID
    pub roles: Vec<String>,       // User roles
    pub permissions: Vec<String>, // Explicit permissions
    pub exp: i64,                 // Expiration timestamp
    pub iat: i64,                 // Issued at
}

pub struct AuthService {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    token_duration: chrono::Duration,
}

impl AuthService {
    pub fn new(secret: &[u8]) -> Self {
        Self {
            encoding_key: EncodingKey::from_secret(secret),
            decoding_key: DecodingKey::from_secret(secret),
            token_duration: chrono::Duration::hours(24),
        }
    }

    pub fn create_token(&self, user: &User) -> Result<String, AuthError> {
        let expiration = Utc::now()
            .checked_add_signed(self.token_duration)
            .expect("valid timestamp")
            .timestamp();

        let claims = Claims {
            sub: user.user_id.clone(),
            org: user.organization_id.clone(),
            roles: user.roles.clone(),
            permissions: self.compute_permissions(&user.roles),
            exp: expiration,
            iat: Utc::now().timestamp(),
        };

        encode(&Header::default(), &claims, &self.encoding_key)
            .map_err(|_| AuthError::TokenCreation)
    }

    pub fn verify_token(&self, token: &str) -> Result<Claims, AuthError> {
        decode::<Claims>(token, &self.decoding_key, &Validation::default())
            .map(|data| data.claims)
            .map_err(|_| AuthError::InvalidToken)
    }

    fn compute_permissions(&self, roles: &[String]) -> Vec<String> {
        // Role-based permissions mapping
        let mut permissions = Vec::new();

        for role in roles {
            match role.as_str() {
                "admin" => {
                    permissions.extend_from_slice(&[
                        "invoices:read", "invoices:write", "invoices:delete",
                        "customers:read", "customers:write", "customers:delete",
                        // ... all permissions
                    ]);
                }
                "accountant" => {
                    permissions.extend_from_slice(&[
                        "invoices:read", "invoices:write",
                        "bills:read", "bills:write",
                        "reports:read",
                    ]);
                }
                "sales_rep" => {
                    permissions.extend_from_slice(&[
                        "customers:read", "customers:write",
                        "quotes:read", "quotes:write",
                        "opportunities:read", "opportunities:write",
                    ]);
                }
                _ => {}
            }
        }

        permissions.sort();
        permissions.dedup();
        permissions.iter().map(|s| s.to_string()).collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Failed to create token")]
    TokenCreation,
    #[error("Invalid token")]
    InvalidToken,
    #[error("Token expired")]
    TokenExpired,
    #[error("Insufficient permissions")]
    InsufficientPermissions,
}
```

**Middleware for Axum**:
```rust
use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
    http::StatusCode,
};

pub async fn auth_middleware(
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract token from Authorization header
    let token = req
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // Verify token
    let auth_service = /* get from app state */;
    let claims = auth_service
        .verify_token(token)
        .map_err(|_| StatusCode::UNAUTHORIZED)?;

    // Store claims in request extensions
    req.extensions_mut().insert(claims);

    Ok(next.run(req).await)
}

// Permission check
pub async fn require_permission(
    required: &str,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let claims = req
        .extensions()
        .get::<Claims>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    if !claims.permissions.contains(&required.to_string()) {
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(req).await)
}
```

### 6.3 Event Bus (`shared/events`)

**In-Memory Event Bus** (for single-instance deployment):
```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainEvent {
    // SCM Events
    BatchProduced { batch_id: String, product_id: String, quantity: Decimal },
    ShipmentDeparted { shipment_id: String, destination: String },
    ShipmentArrived { shipment_id: String, batch_id: String },

    // FIN Events
    InvoiceCreated { invoice_id: String, customer_id: String, total: Decimal },
    InvoicePaid { invoice_id: String, amount: Decimal },
    BillCreated { bill_id: String, vendor_id: String, total: Decimal },

    // CRM Events
    LeadCreated { lead_id: String, source: String },
    LeadConverted { lead_id: String, customer_id: String },
    OpportunityWon { opportunity_id: String, amount: Decimal },
    QuoteAccepted { quote_id: String, customer_id: String },
}

#[async_trait]
pub trait EventHandler: Send + Sync {
    async fn handle(&self, event: &DomainEvent) -> Result<(), Box<dyn std::error::Error>>;
}

pub struct EventBus {
    handlers: Arc<RwLock<Vec<Box<dyn EventHandler>>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn subscribe(&self, handler: Box<dyn EventHandler>) {
        let mut handlers = self.handlers.write().await;
        handlers.push(handler);
    }

    pub async fn publish(&self, event: DomainEvent) -> Result<(), Box<dyn std::error::Error>> {
        let handlers = self.handlers.read().await;

        for handler in handlers.iter() {
            if let Err(e) = handler.handle(&event).await {
                tracing::error!("Event handler failed: {}", e);
                // Continue processing other handlers
            }
        }

        Ok(())
    }
}
```

**Example Handler**:
```rust
// FIN module listens for SCM shipment arrivals to auto-create bills
pub struct ShipmentArrivedHandler {
    bill_service: Arc<BillService>,
}

#[async_trait]
impl EventHandler for ShipmentArrivedHandler {
    async fn handle(&self, event: &DomainEvent) -> Result<(), Box<dyn std::error::Error>> {
        if let DomainEvent::ShipmentArrived { shipment_id, batch_id } = event {
            // Fetch shipment details
            let shipment = self.get_shipment_details(shipment_id).await?;

            // Auto-create bill if purchase order exists
            if let Some(po_id) = shipment.purchase_order_id {
                let bill = Bill {
                    vendor_id: shipment.supplier_id,
                    line_items: self.extract_line_items(&shipment),
                    purchase_order_id: Some(po_id),
                    scm_claims: vec![batch_id.clone()],
                    // ...
                };

                self.bill_service.create_bill(bill).await?;
            }
        }

        Ok(())
    }
}
```

---

## 7. Data Architecture

### 7.1 Database Strategy

**Primary Database**: PostgreSQL 15+
- ACID compliance (critical for financial data)
- JSON/JSONB support (flexible metadata)
- Excellent performance with proper indexes
- Mature ecosystem

**Development Database**: SQLite
- Zero configuration
- Fast tests
- Same SQL syntax as PostgreSQL (mostly)

**Time-Series Data**: TimescaleDB (PostgreSQL extension)
- Optimized for analytics and reporting
- Automatic partitioning
- Downsampling for old data

**Search**: Meilisearch
- Full-text search
- Typo-tolerant
- Fast (<50ms searches)
- Easy to deploy

### 7.2 Schema Design Principles

**Multi-Tenancy**:
```sql
-- Every table has organization_id
CREATE TABLE invoices (
    invoice_id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,  -- Tenant isolation
    -- ...
    FOREIGN KEY (organization_id) REFERENCES organizations(organization_id)
);

-- Row-Level Security (RLS) in PostgreSQL
CREATE POLICY tenant_isolation ON invoices
    USING (organization_id = current_setting('app.current_organization')::text);

ALTER TABLE invoices ENABLE ROW LEVEL SECURITY;
```

**Temporal Data** (for audit trails):
```sql
-- Track all changes
CREATE TABLE invoice_history (
    history_id SERIAL PRIMARY KEY,
    invoice_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    data JSONB NOT NULL,           -- Full invoice snapshot
    changed_by TEXT NOT NULL,
    changed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    change_type TEXT NOT NULL      -- INSERT, UPDATE, DELETE
);

-- Trigger to auto-populate history
CREATE OR REPLACE FUNCTION record_invoice_history()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO invoice_history (invoice_id, version, data, changed_by, change_type)
    VALUES (
        COALESCE(NEW.invoice_id, OLD.invoice_id),
        (SELECT COALESCE(MAX(version), 0) + 1 FROM invoice_history WHERE invoice_id = COALESCE(NEW.invoice_id, OLD.invoice_id)),
        row_to_json(COALESCE(NEW, OLD)),
        current_setting('app.current_user'),
        TG_OP
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER invoice_history_trigger
    AFTER INSERT OR UPDATE OR DELETE ON invoices
    FOR EACH ROW EXECUTE FUNCTION record_invoice_history();
```

**Soft Deletes**:
```sql
-- Don't actually delete, mark as deleted
ALTER TABLE invoices ADD COLUMN deleted_at TIMESTAMP;

-- Query only non-deleted
CREATE VIEW active_invoices AS
SELECT * FROM invoices WHERE deleted_at IS NULL;

-- "Delete" function
UPDATE invoices SET deleted_at = NOW() WHERE invoice_id = $1;
```

### 7.3 Migration Strategy

**Tool**: SQLx migrations (Rust-native)

```bash
# Create migration
sqlx migrate add create_invoices_table

# Apply migrations
sqlx migrate run

# Revert last migration
sqlx migrate revert
```

**Migration File Naming**:
```
migrations/
├── 20250101000001_create_organizations.sql
├── 20250101000002_create_users.sql
├── 20250101000003_create_scm_tables.sql
├── 20250201000001_create_fin_tables.sql
├── 20250201000002_add_fin_indexes.sql
└── 20250301000001_create_crm_tables.sql
```

**Migration Best Practices**:
```sql
-- Always make migrations idempotent
CREATE TABLE IF NOT EXISTS invoices (...);

-- Add indexes concurrently (no locks)
CREATE INDEX CONCURRENTLY idx_invoices_customer ON invoices(customer_id);

-- Use transactions
BEGIN;
-- ... DDL statements
COMMIT;

-- Add NOT NULL constraints safely
-- 1. Add column as nullable
ALTER TABLE invoices ADD COLUMN new_field TEXT;

-- 2. Backfill data
UPDATE invoices SET new_field = 'default_value' WHERE new_field IS NULL;

-- 3. Add NOT NULL constraint
ALTER TABLE invoices ALTER COLUMN new_field SET NOT NULL;
```

---

## 8. API Architecture

### 8.1 REST API Design

**URL Structure**:
```
https://api.mycelix.net/v1/{module}/{resource}/{id}/{action}
```

**Examples**:
```
POST   /v1/fin/invoices
GET    /v1/fin/invoices/:id
PUT    /v1/fin/invoices/:id
DELETE /v1/fin/invoices/:id
POST   /v1/fin/invoices/:id/send
POST   /v1/fin/invoices/:id/payments

GET    /v1/scm/lineage/upstream/:batch_id
GET    /v1/crm/customers/:id/invoices
```

**Versioning**:
- `/v1/` for current stable API
- `/v2/` when breaking changes needed
- Old versions supported for 18 months minimum

**OpenAPI Specification**:
```yaml
openapi: 3.0.3
info:
  title: Mycelix ERP API
  version: 1.0.0
  description: |
    Complete ERP system with cryptographic provenance.

    Modules:
    - SCM: Supply Chain Management
    - FIN: Financial Management
    - CRM: Customer Relationship Management
    - MRP: Manufacturing Resource Planning
    - HR: Human Resources
    - PM: Project Management
    - ASSET: Asset Management

servers:
  - url: https://api.mycelix.net/v1
    description: Production
  - url: http://localhost:8080/v1
    description: Local development

security:
  - bearerAuth: []

paths:
  /fin/invoices:
    post:
      summary: Create invoice
      tags: [Financial]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateInvoiceRequest'
      responses:
        '201':
          description: Invoice created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Invoice'
        '400':
          $ref: '#/components/responses/BadRequest'
        '401':
          $ref: '#/components/responses/Unauthorized'

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

  schemas:
    Invoice:
      type: object
      required: [invoice_id, customer_id, total, status]
      properties:
        invoice_id:
          type: string
        customer_id:
          type: string
        total:
          type: number
          format: decimal
        status:
          type: string
          enum: [DRAFT, SENT, PAID, OVERDUE]

  responses:
    BadRequest:
      description: Invalid request
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                type: string
              details:
                type: array
                items:
                  type: object
                  properties:
                    field:
                      type: string
                    message:
                      type: string
```

### 8.2 Error Handling

**Standard Error Response**:
```json
{
  "error": "VALIDATION_ERROR",
  "message": "Invoice validation failed",
  "details": [
    {
      "field": "line_items",
      "message": "At least one line item is required"
    },
    {
      "field": "customer_id",
      "message": "Customer ID must not be empty"
    }
  ],
  "request_id": "req_abc123",
  "timestamp": "2025-12-30T10:30:00Z"
}
```

**Error Codes**:
```rust
pub enum ApiError {
    // 400 - Bad Request
    ValidationError { details: Vec<ValidationDetail> },
    InvalidInput { field: String, message: String },

    // 401 - Unauthorized
    Unauthorized { message: String },
    InvalidToken,
    ExpiredToken,

    // 403 - Forbidden
    InsufficientPermissions { required: String },

    // 404 - Not Found
    NotFound { resource: String, id: String },

    // 409 - Conflict
    Conflict { message: String },
    DuplicateResource { field: String, value: String },

    // 500 - Internal Server Error
    InternalError { message: String },
    DatabaseError { message: String },
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_code, message, details) = match self {
            ApiError::ValidationError { details } => (
                StatusCode::BAD_REQUEST,
                "VALIDATION_ERROR",
                "Validation failed",
                Some(details),
            ),
            ApiError::Unauthorized { message } => (
                StatusCode::UNAUTHORIZED,
                "UNAUTHORIZED",
                &message,
                None,
            ),
            // ... other cases
        };

        let body = json!({
            "error": error_code,
            "message": message,
            "details": details,
            "request_id": /* extract from request */,
            "timestamp": Utc::now(),
        });

        (status, Json(body)).into_response()
    }
}
```

### 8.3 Pagination

**Query Parameters**:
```
GET /v1/fin/invoices?limit=20&offset=40&sort=created_at:desc
```

**Response**:
```json
{
  "data": [ /* ... invoices ... */ ],
  "pagination": {
    "total": 150,
    "limit": 20,
    "offset": 40,
    "has_more": true
  }
}
```

**Implementation**:
```rust
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_limit")]
    pub limit: i64,

    #[serde(default)]
    pub offset: i64,

    pub sort: Option<String>,
}

fn default_limit() -> i64 { 20 }

pub async fn list_invoices(
    Query(params): Query<PaginationParams>,
    State(repo): State<Arc<dyn InvoiceRepository>>,
) -> Result<Json<PaginatedResponse<Invoice>>, ApiError> {
    let total = repo.count().await?;
    let invoices = repo.list(params.limit, params.offset, params.sort).await?;

    Ok(Json(PaginatedResponse {
        data: invoices,
        pagination: Pagination {
            total,
            limit: params.limit,
            offset: params.offset,
            has_more: params.offset + params.limit < total,
        },
    }))
}
```

---

## 9. Security Architecture

### 9.1 Authentication Flow

```
┌─────────┐                                ┌─────────────┐
│ Client  │                                │  API Server │
└────┬────┘                                └──────┬──────┘
     │                                            │
     │  POST /v1/auth/login                       │
     │  { email, password }                       │
     ├───────────────────────────────────────────>│
     │                                            │
     │                                      [Verify credentials]
     │                                      [Generate JWT]
     │                                            │
     │  { token: "eyJ...", expires_in: 86400 }   │
     │<───────────────────────────────────────────┤
     │                                            │
     │  GET /v1/fin/invoices                      │
     │  Authorization: Bearer eyJ...              │
     ├───────────────────────────────────────────>│
     │                                            │
     │                                      [Verify JWT]
     │                                      [Check permissions]
     │                                      [Execute query]
     │                                            │
     │  { data: [...] }                           │
     │<───────────────────────────────────────────┤
     │                                            │
```

### 9.2 Role-Based Access Control (RBAC)

**Roles**:
- `admin`: Full access to all modules
- `accountant`: Full access to FIN, read-only to SCM/CRM
- `sales_manager`: Full access to CRM, read-only to FIN
- `sales_rep`: Limited access to CRM (assigned customers only)
- `warehouse_manager`: Full access to SCM
- `warehouse_worker`: Limited access to SCM (receive/ship only)
- `viewer`: Read-only access to all modules

**Permission Mapping**:
```rust
pub fn get_role_permissions(role: &str) -> Vec<&'static str> {
    match role {
        "admin" => vec!["*"],
        "accountant" => vec![
            "fin:*",
            "scm:read",
            "crm:read",
        ],
        "sales_manager" => vec![
            "crm:*",
            "fin:invoices:read",
        ],
        "sales_rep" => vec![
            "crm:customers:read",
            "crm:customers:write:assigned",
            "crm:quotes:*:assigned",
        ],
        _ => vec![]
    }
}
```

**Permission Check Middleware**:
```rust
pub async fn require_permission(
    required: &str,
    claims: &Claims,
) -> Result<(), ApiError> {
    // Admin has all permissions
    if claims.permissions.contains(&"*".to_string()) {
        return Ok(());
    }

    // Check exact permission
    if claims.permissions.contains(&required.to_string()) {
        return Ok(());
    }

    // Check wildcard permissions
    let parts: Vec<&str> = required.split(':').collect();
    for i in 1..parts.len() {
        let wildcard = format!("{}:*", parts[..i].join(":"));
        if claims.permissions.contains(&wildcard) {
            return Ok(());
        }
    }

    Err(ApiError::InsufficientPermissions {
        required: required.to_string(),
    })
}
```

### 9.3 Data Encryption

**At Rest**:
- PostgreSQL with encryption enabled
- Disk-level encryption (LUKS on Linux)
- Backup encryption

**In Transit**:
- TLS 1.3 for all API communication
- mTLS for service-to-service communication (optional)

**Sensitive Fields** (e.g., SSN, credit card):
```rust
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};

pub struct FieldEncryption {
    key: LessSafeKey,
}

impl FieldEncryption {
    pub fn new(key_bytes: &[u8; 32]) -> Self {
        let unbound_key = UnboundKey::new(&AES_256_GCM, key_bytes).unwrap();
        let key = LessSafeKey::new(unbound_key);
        Self { key }
    }

    pub fn encrypt(&self, plaintext: &str) -> String {
        let nonce = /* generate random nonce */;
        let mut in_out = plaintext.as_bytes().to_vec();
        self.key.seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce),
            Aad::empty(),
            &mut in_out,
        ).unwrap();

        base64::encode(&in_out)
    }

    pub fn decrypt(&self, ciphertext: &str) -> Result<String, EncryptionError> {
        let mut in_out = base64::decode(ciphertext)?;
        let plaintext = self.key.open_in_place(
            Nonce::assume_unique_for_key(nonce),
            Aad::empty(),
            &mut in_out,
        )?;

        Ok(String::from_utf8(plaintext.to_vec())?)
    }
}
```

### 9.4 Audit Logging

**What to Log**:
- All authentication attempts (success/failure)
- All authorization failures
- All data modifications (CREATE, UPDATE, DELETE)
- All sensitive data access (financial reports, customer PII)

**Log Format** (JSON):
```json
{
  "timestamp": "2025-12-30T10:30:00Z",
  "event_type": "INVOICE_CREATED",
  "user_id": "user_123",
  "organization_id": "org_456",
  "resource_type": "invoice",
  "resource_id": "inv_789",
  "action": "CREATE",
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "request_id": "req_abc123",
  "changes": {
    "total": 1000.00,
    "customer_id": "cust_001"
  }
}
```

**Implementation**:
```rust
pub async fn log_audit_event(
    event_type: &str,
    user_id: &str,
    org_id: &str,
    resource_type: &str,
    resource_id: &str,
    action: &str,
    changes: Option<serde_json::Value>,
) -> Result<()> {
    let event = json!({
        "timestamp": Utc::now(),
        "event_type": event_type,
        "user_id": user_id,
        "organization_id": org_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "action": action,
        "changes": changes,
    });

    // Write to audit log table
    sqlx::query!(
        "INSERT INTO audit_log (event) VALUES ($1)",
        event
    )
    .execute(&pool)
    .await?;

    Ok(())
}
```

---

## 10. Deployment Architecture

### 10.1 Deployment Options

**Option 1: SaaS (Managed)**
- Mycelix hosts and manages
- Multi-tenant PostgreSQL
- Auto-scaling Kubernetes
- Pricing: $25-$75/user/month

**Option 2: Self-Hosted (On-Premise)**
- Customer installs on their infrastructure
- Docker Compose or Kubernetes
- Pricing: One-time license fee + annual support

**Option 3: Hybrid**
- Mycelix SaaS for most modules
- Sensitive data (e.g., payroll) self-hosted
- Encrypted sync between instances

### 10.2 Kubernetes Deployment

**Architecture**:
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mycelix-erp

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mycelix-erp
  namespace: mycelix-erp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mycelix-erp
  template:
    metadata:
      labels:
        app: mycelix-erp
    spec:
      containers:
      - name: mycelix-erp
        image: mycelix/erp:v2.0.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mycelix-secrets
              key: database-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: mycelix-secrets
              key: jwt-secret
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: mycelix-erp
  namespace: mycelix-erp
spec:
  selector:
    app: mycelix-erp
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mycelix-erp
  namespace: mycelix-erp
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.mycelix.net
    secretName: mycelix-tls
  rules:
  - host: api.mycelix.net
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mycelix-erp
            port:
              number: 80

---
# hpa.yaml (Horizontal Pod Autoscaler)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mycelix-erp
  namespace: mycelix-erp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mycelix-erp
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### 10.3 Docker Compose (Local Development)

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mycelix_erp
      POSTGRES_USER: mycelix
      POSTGRES_PASSWORD: development_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mycelix"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  meilisearch:
    image: getmeili/meilisearch:v1.5
    environment:
      MEILI_MASTER_KEY: development_master_key
    ports:
      - "7700:7700"
    volumes:
      - meilisearch_data:/meili_data

  mycelix-erp:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      DATABASE_URL: postgresql://mycelix:development_password@postgres:5432/mycelix_erp
      REDIS_URL: redis://redis:6379
      MEILISEARCH_URL: http://meilisearch:7700
      MEILISEARCH_API_KEY: development_master_key
      JWT_SECRET: development_jwt_secret_change_in_production
      RUST_LOG: info
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./config:/app/config

volumes:
  postgres_data:
  meilisearch_data:
```

### 10.4 Monitoring & Observability

**Prometheus Metrics**:
```rust
use prometheus::{IntCounterVec, HistogramVec, Registry};

lazy_static! {
    pub static ref HTTP_REQUESTS: IntCounterVec = IntCounterVec::new(
        opts!("http_requests_total", "Total HTTP requests"),
        &["method", "path", "status"]
    ).unwrap();

    pub static ref HTTP_DURATION: HistogramVec = HistogramVec::new(
        histogram_opts!("http_request_duration_seconds", "HTTP request duration"),
        &["method", "path"]
    ).unwrap();

    pub static ref INVOICES_CREATED: IntCounterVec = IntCounterVec::new(
        opts!("invoices_created_total", "Total invoices created"),
        &["organization_id"]
    ).unwrap();
}

// Middleware to record metrics
pub async fn metrics_middleware(
    req: Request,
    next: Next,
) -> Response {
    let method = req.method().to_string();
    let path = req.uri().path().to_string();

    let timer = HTTP_DURATION
        .with_label_values(&[&method, &path])
        .start_timer();

    let response = next.run(req).await;

    timer.observe_duration();

    HTTP_REQUESTS
        .with_label_values(&[&method, &path, response.status().as_str()])
        .inc();

    response
}
```

**Grafana Dashboard**:
- Request rate (req/s)
- Error rate (%)
- Response time (p50, p95, p99)
- Database connection pool usage
- Active users
- Invoices created per day
- Revenue trends

**Structured Logging**:
```rust
use tracing::{info, warn, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// Initialize logging
tracing_subscriber::registry()
    .with(tracing_subscriber::fmt::layer().json())
    .with(tracing_subscriber::EnvFilter::from_default_env())
    .init();

// Usage
info!(
    user_id = %user.user_id,
    organization_id = %user.organization_id,
    "User logged in"
);

error!(
    error = %e,
    invoice_id = %invoice_id,
    "Failed to create invoice"
);
```

---

## 11. Integration Architecture

### 11.1 External System Integration

**Supported Integration Patterns**:

**1. REST API** (Primary)
- JSON request/response
- OAuth 2.0 authentication
- Webhook notifications

**2. CSV Import/Export**
- Batch data migration
- Scheduled exports for analytics

**3. MQTT** (IoT devices for SCM)
- Real-time sensor data
- Equipment status updates

**4. Email Integration**
- Send invoices via email
- Parse incoming purchase orders

**5. Webhooks** (Outbound events)
- Notify external systems of events
- Retry with exponential backoff

### 11.2 Common Integrations

**Accounting Software**:
- QuickBooks (export GL entries)
- Xero (sync invoices)
- Sage (import bills)

**Payment Gateways**:
- Stripe (credit card payments)
- PayPal (invoice payments)
- Bank ACH (direct debit)

**Shipping**:
- FedEx (track shipments)
- UPS (generate labels)
- USPS (delivery confirmation)

**E-commerce**:
- Shopify (sync orders → create invoices)
- WooCommerce (inventory sync)
- Magento (customer sync)

**Implementation Example** (Stripe):
```rust
use stripe::{Client, CreatePaymentIntent};

pub async fn process_stripe_payment(
    invoice_id: &str,
    amount: Decimal,
    stripe_token: &str,
) -> Result<String, PaymentError> {
    let client = Client::new(stripe_api_key);

    let payment_intent = CreatePaymentIntent::new(
        (amount * 100).to_u64().unwrap(), // Convert to cents
        stripe::Currency::USD,
    )
    .description(format!("Payment for invoice {}", invoice_id))
    .payment_method(stripe_token);

    let intent = PaymentIntent::create(&client, payment_intent).await?;

    Ok(intent.id.to_string())
}
```

---

## 12. Scalability & Performance

### 12.1 Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **API Response Time (p95)** | <100ms | Fast user experience |
| **Database Query Time (p95)** | <50ms | Indexed queries |
| **Concurrent Users** | 10,000+ | Medium enterprise |
| **Transactions Per Second** | 1,000+ | High throughput |
| **Data Size** | 100GB+ per tenant | Multi-year history |

### 12.2 Scaling Strategies

**Horizontal Scaling** (Add more instances):
- Stateless API servers
- Load balancer (Nginx/HAProxy)
- Session stored in Redis (not server memory)

**Vertical Scaling** (Bigger machines):
- Database (PostgreSQL)
- Full-text search (Meilisearch)

**Database Optimization**:
```sql
-- Partitioning (for large tables)
CREATE TABLE invoices_2025 PARTITION OF invoices
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Read replicas for reports
-- Write: primary database
-- Read: replica database

-- Connection pooling
-- max_connections = 200 in PostgreSQL
-- Pool size = 20 per API instance
```

**Caching**:
```rust
use redis::AsyncCommands;

pub async fn get_customer_with_cache(
    customer_id: &str,
    redis: &redis::aio::Connection,
    db: &sqlx::PgPool,
) -> Result<Customer> {
    // Try cache first
    let cache_key = format!("customer:{}", customer_id);

    if let Ok(cached) = redis.get::<_, String>(&cache_key).await {
        return Ok(serde_json::from_str(&cached)?);
    }

    // Cache miss, query database
    let customer = sqlx::query_as!(Customer, "SELECT * FROM customers WHERE customer_id = $1", customer_id)
        .fetch_one(db)
        .await?;

    // Store in cache (TTL: 5 minutes)
    let _: () = redis.set_ex(&cache_key, serde_json::to_string(&customer)?, 300).await?;

    Ok(customer)
}
```

### 12.3 Load Testing

**Tool**: k6 (https://k6.io/)

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },   // Ramp up to 100 users
    { duration: '5m', target: 100 },   // Stay at 100 users
    { duration: '2m', target: 200 },   // Ramp up to 200 users
    { duration: '5m', target: 200 },   // Stay at 200 users
    { duration: '2m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<100'],  // 95% of requests < 100ms
    http_req_failed: ['rate<0.01'],    // Error rate < 1%
  },
};

export default function () {
  // Login
  let loginRes = http.post('http://api.mycelix.net/v1/auth/login', {
    email: 'test@example.com',
    password: 'password123',
  });

  check(loginRes, {
    'login successful': (r) => r.status === 200,
  });

  let token = loginRes.json('token');

  // List invoices
  let params = {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  };

  let invoicesRes = http.get('http://api.mycelix.net/v1/fin/invoices', params);

  check(invoicesRes, {
    'invoices loaded': (r) => r.status === 200,
    'response time OK': (r) => r.timings.duration < 100,
  });

  sleep(1);
}
```

---

## 13. Development Architecture

### 13.1 Development Workflow

**Branch Strategy** (Git Flow):
```
main              # Production-ready code
├─ develop        # Integration branch
│  ├─ feature/fin-invoice-creation
│  ├─ feature/crm-lead-management
│  └─ bugfix/scm-lineage-query
└─ hotfix/security-patch
```

**Commit Convention**:
```
<type>(<scope>): <subject>

Types: feat, fix, docs, refactor, test, chore
Scopes: scm, fin, crm, mrp, hr, pm, asset, shared

Examples:
feat(fin): add invoice creation API
fix(scm): correct lineage hash calculation
docs(crm): update API documentation
test(fin): add invoice validation tests
```

**Pull Request Template**:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No decrease in test coverage
- [ ] API spec updated (if applicable)
- [ ] Migration added (if schema changed)

## Related Issues
Fixes #123
```

### 13.2 CI/CD Pipeline

**GitHub Actions** (`.github/workflows/ci.yml`):
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  RUST_BACKTRACE: 1
  DATABASE_URL: postgresql://test:test@localhost:5432/mycelix_test

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          components: rustfmt, clippy
      - run: cargo fmt --check
      - run: cargo clippy -- -D warnings

  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: mycelix_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Cache cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}

      - name: Run migrations
        run: sqlx migrate run

      - name: Run tests
        run: cargo test --all-features

      - name: Generate coverage
        uses: actions-rs/tarpaulin@v0.1
        with:
          args: '--all-features --workspace --timeout 300 --out Xml'

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Build release
        run: cargo build --release

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: mycelix-erp
          path: target/release/mycelix-erp

  docker:
    runs-on: ubuntu-latest
    needs: [lint, test, build]
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: mycelix/erp:latest,mycelix/erp:${{ github.sha }}
```

### 13.3 Local Development Setup

**One-Command Setup**:
```bash
# scripts/dev-setup.sh
#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up Mycelix ERP development environment..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker required"; exit 1; }
command -v cargo >/dev/null 2>&1 || { echo "❌ Rust required"; exit 1; }

# Start PostgreSQL
docker-compose up -d postgres

# Wait for PostgreSQL
until docker-compose exec postgres pg_isready -U mycelix; do
  echo "Waiting for PostgreSQL..."
  sleep 1
done

# Run migrations
sqlx migrate run

# Install dependencies
cargo build

# Run tests
cargo test

echo "✅ Setup complete! Run 'cargo run' to start the server."
```

---

## 14. Migration & Evolution

### 14.1 Data Migration (From Legacy Systems)

**Common Scenarios**:
1. **From Excel/CSV** → Mycelix ERP
2. **From QuickBooks** → Mycelix ERP
3. **From SAP** → Mycelix ERP

**Migration Tool** (`scripts/migrate.sh`):
```bash
#!/usr/bin/env bash

# Example: Migrate from QuickBooks CSV export
cargo run --bin migrate-quickbooks \
  --input ./data/quickbooks_export.csv \
  --output ./data/mycelix_import.json \
  --organization-id org_123

# Import into Mycelix
curl -X POST http://localhost:8080/v1/admin/import \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @./data/mycelix_import.json
```

**Migration Validation**:
```rust
pub async fn validate_migration(
    source_totals: &MigrationTotals,
    target_totals: &MigrationTotals,
) -> MigrationReport {
    let mut report = MigrationReport::default();

    // Check record counts
    if source_totals.customers != target_totals.customers {
        report.warnings.push(format!(
            "Customer count mismatch: source={}, target={}",
            source_totals.customers, target_totals.customers
        ));
    }

    // Check financial totals
    if (source_totals.total_revenue - target_totals.total_revenue).abs() > Decimal::new(1, 2) {
        report.errors.push(format!(
            "Revenue mismatch: source={}, target={}",
            source_totals.total_revenue, target_totals.total_revenue
        ));
    }

    report
}
```

### 14.2 Version Upgrade Path

**v0.4.0 → v1.0.0** (SCM only → SCM + FIN):
```sql
-- No schema changes to SCM
-- Add FIN tables
-- Create linkage table
CREATE TABLE scm_fin_links (
    scm_claim_id TEXT REFERENCES scm_claims(claim_id),
    fin_invoice_id TEXT REFERENCES fin_invoices(invoice_id),
    link_type TEXT NOT NULL,  -- 'shipment_to_invoice', 'batch_to_bill'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (scm_claim_id, fin_invoice_id)
);
```

**v1.0.0 → v2.0.0** (Full ERP suite):
- Add all 7 modules
- Create inter-module linkage tables
- Add shared user/organization tables

---

## 15. Appendices

### Appendix A: Glossary

- **DKG**: Decentralized Knowledge Graph
- **VC**: Verifiable Credential (W3C standard)
- **DID**: Decentralized Identifier
- **BBS+**: Signature scheme for selective disclosure
- **SD-JWT**: Selective Disclosure JSON Web Token
- **RBAC**: Role-Based Access Control
- **GL**: General Ledger
- **AR**: Accounts Receivable
- **AP**: Accounts Payable
- **BOM**: Bill of Materials
- **MRP**: Manufacturing Resource Planning
- **WIP**: Work In Progress

### Appendix B: API Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `INVALID_INPUT` | 400 | Invalid input parameter |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `INVALID_TOKEN` | 401 | JWT token invalid or expired |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource already exists |
| `DUPLICATE_RESOURCE` | 409 | Duplicate unique field |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `DATABASE_ERROR` | 500 | Database operation failed |

### Appendix C: Database Table Naming Conventions

| Pattern | Example | Purpose |
|---------|---------|---------|
| `{module}_{entity}` | `fin_invoices` | Main entity table |
| `{module}_{entity}_history` | `fin_invoices_history` | Audit history |
| `{entity}_line_items` | `invoice_line_items` | Child entities |
| `{module1}_{module2}_links` | `scm_fin_links` | Inter-module relationships |
| `idx_{table}_{column}` | `idx_invoices_customer` | Database index |

### Appendix D: Performance Benchmarks

**Target Hardware**: 4 CPU cores, 16GB RAM, SSD storage

| Operation | Target | Notes |
|-----------|--------|-------|
| Create invoice | <50ms | Single invoice |
| List 100 invoices | <100ms | With pagination |
| Complex lineage query | <200ms | 10+ levels deep |
| Financial report generation | <5s | Full fiscal year |
| Bulk import (1000 records) | <10s | CSV import |
| Concurrent writes (100 req/s) | <100ms p95 | Load test |

### Appendix E: Security Checklist

**Pre-Production Checklist**:
- [ ] All secrets in environment variables (no hardcoded)
- [ ] TLS enabled for all endpoints
- [ ] JWT secret is cryptographically random (32+ bytes)
- [ ] Database credentials rotated
- [ ] Rate limiting enabled
- [ ] CORS configured (no wildcard origins)
- [ ] Security headers (OWASP Top 10)
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (content security policy)
- [ ] Audit logging enabled
- [ ] Error messages don't leak sensitive info
- [ ] Dependencies scanned for vulnerabilities
- [ ] Penetration testing completed

### Appendix F: References

**W3C Standards**:
- Verifiable Credentials: https://www.w3.org/TR/vc-data-model/
- Decentralized Identifiers: https://www.w3.org/TR/did-core/

**RFCs**:
- JWT: RFC 7519
- OAuth 2.0: RFC 6749
- TLS 1.3: RFC 8446

**Books**:
- "Domain-Driven Design" by Eric Evans
- "Building Microservices" by Sam Newman
- "Designing Data-Intensive Applications" by Martin Kleppmann

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-30 | Tristan Stoltz, Claude Code | Initial comprehensive architecture document |

---

**End of Technical Architecture Document**

This document should be updated as the architecture evolves. All major architectural decisions should be documented here or in separate ADR (Architecture Decision Records) documents.
