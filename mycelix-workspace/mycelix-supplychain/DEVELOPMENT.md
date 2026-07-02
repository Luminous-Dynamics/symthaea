# Development Guide

Comprehensive guide for developers contributing to the Mycelix Supply Chain Provenance system.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Environment](#development-environment)
  - [Docker Compose Setup](#docker-compose-setup)
  - [Local Setup (No Docker)](#local-setup-no-docker)
- [Project Structure](#project-structure)
- [Testing](#testing)
  - [Integration Tests](#integration-tests)
  - [End-to-End Tests](#end-to-end-tests)
  - [Load Tests](#load-tests)
- [API Development](#api-development)
  - [Adding New Endpoints](#adding-new-endpoints)
  - [Request/Response Patterns](#requestresponse-patterns)
- [Database](#database)
  - [Migrations](#migrations)
  - [Query Development](#query-development)
- [Monitoring & Observability](#monitoring--observability)
  - [Prometheus Metrics](#prometheus-metrics)
  - [Grafana Dashboards](#grafana-dashboards)
  - [Adding New Metrics](#adding-new-metrics)
- [Debugging](#debugging)
  - [Service Logs](#service-logs)
  - [Database Inspection](#database-inspection)
  - [Tracing Requests](#tracing-requests)
- [Code Style & Conventions](#code-style--conventions)
- [Contributing Workflow](#contributing-workflow)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

**Required**:
- **Rust** 1.75+ ([install](https://rustup.rs/))
- **Docker** 24.0+ and **Docker Compose** 2.0+ ([install](https://docs.docker.com/get-docker/))
- **Git** 2.30+

**Optional** (for SDK/adapter development):
- **Node.js** 18+ ([install](https://nodejs.org/))
- **Make** (usually pre-installed on Linux/Mac)

**System Requirements**:
- 4 GB RAM (8 GB recommended for full stack)
- 2 CPU cores (4 recommended)
- 2 GB disk space

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Luminous-Dynamics/mycelix-supplychain.git
cd mycelix-supplychain

# Start full development stack (PostgreSQL, Service, Prometheus, Grafana)
./scripts/dev-start.sh

# In another terminal, run integration tests
cd rust/service
cargo test --test integration_*

# View monitoring dashboards
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus

# When done, stop services (preserves data)
./scripts/dev-stop.sh

# To completely reset (delete all data)
./scripts/dev-reset.sh
```

---

## Development Environment

### Docker Compose Setup

**Recommended for most development work**. Provides a complete stack with PostgreSQL, Prometheus, and Grafana.

#### Services

The `docker-compose.yml` defines 4 services:

1. **postgres** - PostgreSQL 16 database
   - Port: 5432
   - Database: `supplychain`
   - Credentials: `supplychain / dev_password_change_in_prod`

2. **service** - Rust provenance service
   - Port: 8080
   - Auto-reloads on code changes (when using volume mounts)

3. **prometheus** - Metrics collection
   - Port: 9090
   - Scrapes service every 10 seconds
   - Config: `monitoring/prometheus/prometheus.yml`

4. **grafana** - Dashboard visualization
   - Port: 3000
   - Default credentials: `admin / admin`
   - Auto-provisioned with Supply Chain dashboard

#### Starting the Stack

```bash
# Start all services
./scripts/dev-start.sh

# Or manually with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f service     # Service logs only
docker-compose logs -f             # All services
```

The `dev-start.sh` script includes health checks and waits for the service to be ready before exiting.

#### Stopping the Stack

```bash
# Stop services (data preserved)
./scripts/dev-stop.sh

# Or manually
docker-compose down
```

Volumes are preserved, so your database data, metrics, and dashboards persist between restarts.

#### Resetting the Environment

```bash
# Delete ALL data (PostgreSQL, Prometheus, Grafana, SQLite)
./scripts/dev-reset.sh
```

This prompts for confirmation before deleting:
- PostgreSQL database volume
- Prometheus metrics data
- Grafana dashboards and configuration
- Local SQLite files in `rust/data/`

#### Rebuilding After Code Changes

```bash
# Rebuild service container
docker-compose up -d --build service

# Or rebuild all containers
docker-compose build
docker-compose up -d
```

### Local Setup (No Docker)

For development without Docker (useful for faster iteration with `cargo watch`):

#### 1. Install Dependencies

```bash
# Rust dependencies (managed by Cargo)
cd rust/service
cargo build

# TypeScript dependencies
cd ts/sdk
npm ci
```

#### 2. Start PostgreSQL

```bash
# Option A: Use Docker for just PostgreSQL
docker run -d \
  -p 5432:5432 \
  -e POSTGRES_DB=supplychain \
  -e POSTGRES_USER=supplychain \
  -e POSTGRES_PASSWORD=dev_password \
  --name supplychain-db \
  postgres:16-alpine

# Option B: Use local PostgreSQL installation
createdb supplychain
```

#### 3. Configure Environment

```bash
# Set database URL
export DATABASE_URL="postgresql://supplychain:dev_password@localhost:5432/supplychain"

# Or use SQLite for simple development
export DATABASE_URL="sqlite:./data/supplychain.db"
```

#### 4. Run Migrations

```bash
cd rust/service
sqlx migrate run
```

#### 5. Start Service

```bash
cargo run --bin provenance-service

# Or with auto-reload on file changes
cargo install cargo-watch
cargo watch -x 'run --bin provenance-service'
```

---

## Project Structure

```
mycelix-supplychain/
├── rust/
│   ├── claim-model/          # Core data models (DkgClaim, VC schemas)
│   ├── crypto/                # Cryptographic operations (signing, verification)
│   └── service/               # REST API service (Axum)
│       ├── src/
│       │   ├── main.rs        # Server entry point, router setup
│       │   ├── lib.rs         # Library exports for testing
│       │   ├── batch.rs       # POST /v1/events/batch endpoint
│       │   ├── lineage_api.rs # GET /v1/lineage/:id, search endpoints
│       │   ├── db.rs          # Database layer (SQLite/PostgreSQL)
│       │   └── metrics.rs     # Prometheus metrics
│       ├── tests/
│       │   ├── common/        # Shared test utilities
│       │   ├── integration_batch.rs    # Batch API tests
│       │   ├── integration_lineage.rs  # Lineage query tests
│       │   └── ...
│       └── migrations/        # SQLx database migrations
├── ts/
│   ├── sdk/                   # TypeScript client SDK
│   ├── adapters/              # CSV, MQTT adapters
│   └── dashboard/             # Web UI for verifying claims
├── specs/
│   ├── openapi.yaml           # OpenAPI 3.0 specification
│   ├── schemas/               # JSON schemas for VCs
│   └── examples/              # Example events (JSON)
├── tests/
│   ├── e2e/                   # End-to-end test suites
│   ├── load/                  # K6 load tests
│   └── data/                  # Test datasets (CSV, JSON)
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml     # Scrape configuration
│   │   └── alerts.yml         # Alert rules
│   └── grafana/
│       └── provisioning/      # Auto-provisioned dashboards
├── docker/
│   └── Dockerfile.service     # Multi-stage Rust build
├── scripts/
│   ├── dev-start.sh           # Start development environment
│   ├── dev-stop.sh            # Stop services
│   └── dev-reset.sh           # Reset environment
└── docs/
    ├── api-guide.md           # API reference
    ├── architecture.md        # System architecture
    └── deployment.md          # Deployment guide
```

---

## Testing

### Integration Tests

Integration tests are located in `rust/service/tests/` and test the full HTTP API stack.

#### Running Tests

```bash
cd rust/service

# Run all integration tests
cargo test --test integration_*

# Run specific test file
cargo test --test integration_batch

# Run specific test function
cargo test --test integration_batch test_batch_best_effort_mode_all_success

# Run with output (show println! statements)
cargo test --test integration_batch -- --nocapture
```

#### Test Structure

Each integration test file follows this pattern:

```rust
// tests/integration_myfeature.rs
use axum::{body::Body, http::Request};
use tower::ServiceExt; // for `oneshot`

mod common;
use common::*;  // Shared test utilities

#[tokio::test]
async fn test_my_feature() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/endpoint")
                .header("content-type", "application/json")
                .body(Body::from("..."))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
```

#### Test Database

Integration tests use an in-memory SQLite database (`sqlite::memory:`) for isolation and speed. Each test gets a fresh database.

#### Writing New Tests

1. Add test file: `rust/service/tests/integration_myfeature.rs`
2. Use `mod common; use common::*;` for shared utilities
3. Create test app with `create_test_app().await`
4. Make HTTP requests with `oneshot(Request::builder()...)`
5. Assert responses with `assert_eq!(response.status(), StatusCode::OK)`

**Example**:

```rust
#[tokio::test]
async fn test_get_batch_claims() {
    let app = create_test_app().await;

    // First, create a claim
    let event = create_test_event("BATCH-001", "SKU-001", EventType::Produced);
    let create_response = app
        .oneshot(/* POST /v1/events */)
        .await
        .unwrap();

    // Then query it
    let query_response = app
        .oneshot(
            Request::builder()
                .uri("/v1/batches/BATCH-001/claims")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(query_response.status(), StatusCode::OK);

    let body = query_response.into_body().collect().await.unwrap().to_bytes();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(result["total_claims"], 1);
}
```

### End-to-End Tests

Located in `tests/e2e/`. These tests run against a live service instance.

```bash
cd tests/e2e
npm ci
npm test
```

See `tests/e2e/README.md` for details.

### Load Tests

K6 load tests simulate production traffic patterns.

```bash
cd tests/load

# Quick smoke test
k6 run scenarios/smoke.js

# Sustained load test
k6 run scenarios/load.js

# Stress test
k6 run scenarios/stress.js
```

See `tests/load/README.md` for details and performance targets.

---

## API Development

### Adding New Endpoints

#### 1. Define the Handler

Create a handler function in an appropriate module (e.g., `src/lineage_api.rs`):

```rust
use axum::{extract::State, Json};
use std::sync::Arc;

#[derive(Debug, Serialize)]
pub struct MyResponse {
    pub data: Vec<String>,
    pub total: usize,
}

pub async fn my_endpoint(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MyResponse>, ApiError> {
    // Your logic here
    let data = vec!["example".to_string()];

    Ok(Json(MyResponse {
        data,
        total: 1,
    }))
}
```

#### 2. Add Route to Router

In `src/main.rs`:

```rust
use provenance_service::lineage_api;

let app = Router::new()
    .route("/v1/my-endpoint", get(lineage_api::my_endpoint))
    // ... other routes
    .with_state(Arc::new(state));
```

#### 3. Add Metrics

In the handler:

```rust
use crate::metrics::{API_REQUEST_DURATION, MetricTimer};

pub async fn my_endpoint(
    State(state): State<Arc<AppState>>,
) -> Result<Json<MyResponse>, ApiError> {
    let timer = MetricTimer::new();

    // Your logic here
    let result = do_work().await?;

    // Record metrics
    timer.observe(&API_REQUEST_DURATION, &["GET", "/v1/my-endpoint", "200"]);

    Ok(Json(result))
}
```

#### 4. Write Integration Tests

In `tests/integration_myfeature.rs`:

```rust
#[tokio::test]
async fn test_my_endpoint() {
    let app = create_test_app().await;

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/my-endpoint")
                .body(Body::empty())
                .unwrap()
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}
```

### Request/Response Patterns

#### Pagination

For endpoints returning lists, use consistent pagination:

```rust
#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize { 50 }

#[derive(Debug, Serialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
    pub has_more: bool,
}
```

#### Error Handling

Use the `ApiError` type for consistent error responses:

```rust
pub enum ApiError {
    NotFound(String),
    BadRequest(String),
    InternalError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(json!({ "error": message }));
        (status, body).into_response()
    }
}
```

---

## Database

### Migrations

Migrations are managed by SQLx and located in `rust/service/migrations/`.

#### Creating a Migration

```bash
cd rust/service

# Create new migration
sqlx migrate add my_migration_name

# This creates:
# migrations/YYYYMMDDHHMMSS_my_migration_name.sql
```

Edit the generated SQL file:

```sql
-- Add migration
CREATE TABLE IF NOT EXISTS my_new_table (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_my_new_table_created_at ON my_new_table(created_at);
```

#### Running Migrations

```bash
# Apply pending migrations
sqlx migrate run

# Revert last migration
sqlx migrate revert

# Check migration status
sqlx migrate info
```

Migrations run automatically when the service starts.

### Query Development

#### Adding Database Methods

In `src/db.rs`:

```rust
impl Database {
    pub async fn my_query(&self, param: &str) -> Result<Vec<MyData>> {
        let timer = MetricTimer::new();

        let rows = sqlx::query(
            r#"
            SELECT id, data FROM my_table
            WHERE id = ?
            ORDER BY created_at DESC
            "#
        )
        .bind(param)
        .fetch_all(&self.pool)
        .await?;

        timer.observe(&DB_QUERY_DURATION, &["my_query"]);

        let mut results = Vec::new();
        for row in rows {
            let data: String = row.try_get("data")?;
            results.push(MyData { data });
        }

        Ok(results)
    }
}
```

#### Testing Database Queries

```rust
#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_db() -> Result<Database> {
        Database::new("sqlite::memory:").await
    }

    #[tokio::test]
    async fn test_my_query() {
        let db = create_test_db().await.unwrap();

        // Insert test data
        // ...

        // Test query
        let results = db.my_query("test").await.unwrap();
        assert_eq!(results.len(), 1);
    }
}
```

---

## Monitoring & Observability

### Prometheus Metrics

The service exposes Prometheus metrics at `http://localhost:8080/metrics`.

#### Available Metrics

**Request Metrics**:
- `supplychain_api_request_duration_seconds` - Histogram of API request latency
  - Labels: `method`, `endpoint`, `status`
  - Buckets: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s

**Event Metrics**:
- `supplychain_events_ingested_total` - Counter of ingested events
  - Labels: `event_type` (PRODUCED, TRANSFORMED, etc.)

**Claim Metrics**:
- `supplychain_claims_stored_total` - Counter of stored claims

**Database Metrics**:
- `supplychain_db_query_duration_seconds` - Histogram of database query duration
  - Labels: `operation` (store_claim, get_claim, etc.)
- `supplychain_db_connections_active` - Gauge of active connections

**Lineage Metrics**:
- `supplychain_lineage_depth` - Histogram of lineage chain depths
  - Labels: `batch_id_prefix`

**Query Metrics**:
- `supplychain_query_results_count` - Histogram of query result counts
  - Labels: `query_type` (batch_claims, lineage, search)

**Validation Metrics**:
- `supplychain_validation_errors_total` - Counter of validation errors
  - Labels: `error_type`

#### Querying Metrics

**Request rate**:
```promql
rate(supplychain_api_request_duration_seconds_count[5m])
```

**Error rate**:
```promql
rate(supplychain_api_request_duration_seconds_count{status=~"5.."}[5m])
```

**P95 latency**:
```promql
histogram_quantile(0.95, rate(supplychain_api_request_duration_seconds_bucket[5m]))
```

**Events by type**:
```promql
sum by (event_type) (rate(supplychain_events_ingested_total[5m]))
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (username: `admin`, password: `admin`).

The **Supply Chain Provenance - Overview** dashboard includes:

1. **Request Rate** - Requests per second by endpoint
2. **Error Rate** - Error percentage with alert threshold
3. **Latency Percentiles** - p50, p95, p99 latency
4. **Events Ingested** - Total events ingested (last hour)
5. **Claims Stored** - Total claims in database
6. **Database Connections** - Active connections gauge
7. **Service Health** - UP/DOWN status
8. **Database Query Duration** - Average query time by operation
9. **Lineage Depth Distribution** - Average lineage depth
10. **Events by Type** - Pie chart breakdown
11. **Batch Processing Stats** - Batch events vs requests
12. **Validation Errors** - Validation error rate by type

#### Creating Custom Dashboards

1. Open Grafana (`http://localhost:3000`)
2. Click **+ > Dashboard**
3. Click **Add visualization**
4. Select **Prometheus** datasource
5. Enter PromQL query (e.g., `rate(supplychain_events_ingested_total[5m])`)
6. Configure visualization type (graph, stat, gauge, etc.)
7. Save dashboard

### Adding New Metrics

#### 1. Define Metric

In `src/metrics.rs`:

```rust
use prometheus::{register_counter, Counter};
use once_cell::sync::Lazy;

pub static MY_METRIC: Lazy<Counter> = Lazy::new(|| {
    register_counter!(
        opts!("supplychain_my_metric_total", "Description of my metric")
    )
    .expect("Failed to register my_metric")
});
```

#### 2. Record Metric

In your code:

```rust
use crate::metrics::MY_METRIC;

fn do_something() {
    // ... do work ...
    MY_METRIC.inc();
}
```

#### 3. Add to Grafana Dashboard

Create a panel with PromQL query:
```promql
rate(supplychain_my_metric_total[5m])
```

---

## Debugging

### Service Logs

#### Docker Compose

```bash
# Tail all service logs
docker-compose logs -f service

# View last 100 lines
docker-compose logs --tail=100 service

# Filter by log level
docker-compose logs service | grep ERROR
docker-compose logs service | grep WARN
```

#### Local Service

When running locally with `cargo run`, logs go to stdout/stderr.

```bash
# Run with debug logging
RUST_LOG=debug cargo run

# Run with trace logging (very verbose)
RUST_LOG=trace cargo run

# Filter by module
RUST_LOG=provenance_service::db=debug cargo run
```

#### Log Levels

The service uses `tracing` for structured logging:

- `ERROR` - Critical errors requiring immediate attention
- `WARN` - Warning conditions that should be investigated
- `INFO` - Informational messages (default level)
- `DEBUG` - Detailed debugging information
- `TRACE` - Very detailed tracing (includes SQL queries)

### Database Inspection

#### PostgreSQL (Docker)

```bash
# Connect to database
docker-compose exec postgres psql -U supplychain -d supplychain

# List tables
\dt

# Describe table
\d claims
\d lineage

# Query claims
SELECT id, batch_id, event_type, timestamp FROM claims ORDER BY timestamp DESC LIMIT 10;

# Query lineage
SELECT * FROM lineage LIMIT 10;

# Count claims by batch
SELECT batch_id, COUNT(*) FROM claims GROUP BY batch_id;
```

#### SQLite (Local)

```bash
sqlite3 rust/data/supplychain.db

# List tables
.tables

# Show schema
.schema claims

# Query
SELECT * FROM claims LIMIT 10;

# Exit
.quit
```

### Tracing Requests

#### cURL with Timing

```bash
curl -w "\nTime: %{time_total}s\nStatus: %{http_code}\n" \
  -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @specs/examples/batch_produced.json
```

#### Check Metrics

After making requests, check the `/metrics` endpoint:

```bash
curl http://localhost:8080/metrics | grep supplychain_api_request_duration_seconds_count
```

#### Prometheus Query

Query Prometheus directly for request metrics:

```bash
# Recent requests
curl 'http://localhost:9090/api/v1/query?query=supplychain_api_request_duration_seconds_count'

# Request rate over last 5 minutes
curl 'http://localhost:9090/api/v1/query?query=rate(supplychain_api_request_duration_seconds_count[5m])'
```

---

## Code Style & Conventions

### Rust

Follow the official Rust style guide:

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt -- --check

# Lint with Clippy
cargo clippy

# Fix Clippy warnings
cargo clippy --fix
```

**Naming Conventions**:
- Types: `PascalCase` (e.g., `DkgClaim`, `EventType`)
- Functions: `snake_case` (e.g., `store_claim`, `get_batch_claims`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `MAX_BATCH_SIZE`)
- Modules: `snake_case` (e.g., `lineage_api`, `db`)

**Error Handling**:
- Use `Result<T, E>` for operations that can fail
- Prefer `anyhow::Result` for application errors
- Use `?` operator for error propagation
- Add context with `.context("description")`

**Async**:
- Use `async fn` for I/O operations
- Use `tokio::spawn` for concurrent tasks
- Avoid blocking operations in async code

### TypeScript

```bash
cd ts/sdk

# Format code
npm run fmt

# Lint
npm run lint

# Type check
npm run type-check
```

### Commit Messages

Follow Conventional Commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build process, dependencies

**Examples**:
```
feat(api): add batch event ingestion endpoint

Add POST /v1/events/batch with best-effort and atomic modes.
Supports up to 100 events per request with detailed result reporting.

Closes #123
```

```
fix(db): resolve lineage query performance issue

Optimize get_parent_claims to use indexed lookup instead of full scan.
Reduces query time from 500ms to 50ms for large datasets.
```

---

## Contributing Workflow

### 1. Create Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/my-bugfix
```

### 2. Make Changes

- Write code following style conventions
- Add tests for new functionality
- Update documentation if needed

### 3. Test Locally

```bash
# Run tests
cargo test --all
cd ts/sdk && npm test

# Check formatting
cargo fmt -- --check
cargo clippy

# Run integration tests
cargo test --test integration_*

# (Optional) Run load tests
cd tests/load && k6 run scenarios/smoke.js
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat(api): add new endpoint for X"
```

### 5. Push and Create PR

```bash
git push origin feature/my-feature
```

Then create a Pull Request on GitHub.

### 6. Address Review Feedback

```bash
# Make changes
git add .
git commit -m "refactor: address review feedback"
git push
```

---

## Troubleshooting

### Service Won't Start

**Port 8080 already in use**:
```bash
# Find process using port 8080
lsof -i :8080

# Kill the process
kill -9 <PID>

# Or change service port
export SERVICE_PORT=8081
cargo run
```

**Database connection failed**:
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check connection
psql postgresql://supplychain:dev_password@localhost:5432/supplychain

# Reset database
./scripts/dev-reset.sh
./scripts/dev-start.sh
```

### Tests Failing

**Database migration errors**:
```bash
# Reset test database
rm -f rust/data/*.db
cargo test --test integration_batch
```

**Flaky tests**:
- Check for race conditions in concurrent tests
- Ensure test isolation (each test uses fresh database)
- Check test timeouts

**Type errors after updates**:
```bash
# Clean and rebuild
cargo clean
cargo build
```

### Docker Issues

**Service not healthy**:
```bash
# Check service logs
docker-compose logs service

# Check health endpoint
curl http://localhost:8080/health

# Restart service
docker-compose restart service
```

**Out of disk space**:
```bash
# Clean up Docker
docker system prune -a --volumes

# Remove only stopped containers
docker container prune
```

**Build fails**:
```bash
# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Prometheus/Grafana Issues

**No data in Grafana**:
1. Check Prometheus is scraping: `http://localhost:9090/targets`
2. Verify service is exposing metrics: `curl http://localhost:8080/metrics`
3. Check Grafana datasource configuration

**Dashboard not loading**:
```bash
# Restart Grafana
docker-compose restart grafana

# Check provisioning
docker-compose exec grafana ls -la /etc/grafana/provisioning/dashboards/
```

### Performance Issues

**Slow queries**:
1. Check Prometheus for slow queries: `supplychain_db_query_duration_seconds`
2. Enable query logging: `RUST_LOG=sqlx=debug cargo run`
3. Add database indexes if needed

**High memory usage**:
- Check for large query results being loaded into memory
- Implement pagination for large result sets
- Monitor with `docker stats`

**High latency**:
1. Check Grafana latency dashboard
2. Review Prometheus metrics for bottlenecks
3. Enable debug logging to trace slow operations

---

## Additional Resources

- **API Reference**: [docs/api-guide.md](docs/api-guide.md)
- **Architecture**: [docs/architecture.md](docs/architecture.md)
- **Deployment**: [docs/deployment.md](docs/deployment.md)
- **Production Checklist**: [docs/PRODUCTION_CHECKLIST.md](docs/PRODUCTION_CHECKLIST.md)
- **OpenAPI Spec**: [specs/openapi.yaml](specs/openapi.yaml)

For questions or issues, see [CONTRIBUTING.md](CONTRIBUTING.md) or open a [GitHub Issue](https://github.com/Luminous-Dynamics/mycelix-supplychain/issues).
