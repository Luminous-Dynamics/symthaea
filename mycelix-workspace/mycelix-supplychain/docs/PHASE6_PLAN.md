# Phase 6 Plan - Testing, Tooling & Advanced Features

**Goal**: Enhance system quality, developer experience, and operational excellence
**Status**: 📋 Planning
**Estimated Duration**: 3-4 hours
**Focus**: Testing, Query APIs, Monitoring, Developer Experience

---

## Executive Summary

Phase 5 brought us to 100% production-ready. Phase 6 focuses on **quality, tooling, and advanced features** that make the system not just production-ready, but **production-excellent**.

### Key Objectives

1. ✅ **Testing & Quality** - Comprehensive tests for Phase 5 features
2. ✅ **Query & Search** - Powerful lineage and batch query APIs
3. ✅ **Operational Tooling** - Grafana dashboards for monitoring
4. ✅ **Developer Experience** - Docker Compose for easy local development
5. ✅ **SDK Enhancement** - TypeScript SDK v2 with batch support
6. ✅ **API Documentation** - OpenAPI/Swagger specification

---

## Priority 1: Testing & Quality (HIGH)

### 1.1 Integration Tests for Batch API

**Problem**: Batch API (`POST /v1/events/batch`) was added without integration tests
**Risk**: Production issues with batch processing modes
**Estimated Time**: 45 minutes

**File**: `rust/service/tests/integration_batch.rs`

**Test Scenarios**:

```rust
#[tokio::test]
async fn test_batch_best_effort_mode() {
    // Create 10 valid events
    // POST /v1/events/batch with mode=best-effort
    // Assert 201 Created, total=10, succeeded=10, failed=0
    // Verify all claim_ids are present
    // Verify lineage_hashes are unique
    // Check response time <200ms
}

#[tokio::test]
async fn test_batch_partial_success() {
    // Create 5 valid + 3 invalid events
    // POST /v1/events/batch with mode=best-effort
    // Assert 201 Created, total=8, succeeded=5, failed=3
    // Verify failed events have error messages
    // Verify successful events have claim_ids
}

#[tokio::test]
async fn test_batch_atomic_mode_success() {
    // Create 10 valid events
    // POST /v1/events/batch with mode=atomic
    // Assert 201 Created, all succeeded
    // Verify all claims retrievable
}

#[tokio::test]
async fn test_batch_atomic_mode_failure() {
    // Create 5 valid + 1 invalid event
    // POST /v1/events/batch with mode=atomic
    // Assert 500 Internal Server Error
    // Verify error message mentions index of failed event
    // Note: Due to no DB transactions, previous events may be stored
}

#[tokio::test]
async fn test_batch_max_size_exceeded() {
    // Create 101 events (exceeds MAX_BATCH_SIZE)
    // POST /v1/events/batch
    // Assert 400 Bad Request
    // Verify error message mentions max batch size
}

#[tokio::test]
async fn test_batch_empty_array() {
    // POST with events: []
    // Assert 400 Bad Request
    // Verify error message: "Batch cannot be empty"
}

#[tokio::test]
async fn test_batch_invalid_mode() {
    // POST with mode: "invalid-mode"
    // Assert 400 Bad Request
    // Verify error mentions valid modes
}

#[tokio::test]
async fn test_batch_performance() {
    // Create 50 valid events
    // POST /v1/events/batch with mode=best-effort
    // Measure response time
    // Assert duration <300ms
    // Calculate events/second throughput
}

#[tokio::test]
async fn test_batch_lineage_resolution() {
    // Create event A (batch-001)
    // Create batch of 3 events B, C, D (all transform batch-001)
    // Verify all B, C, D have lineage pointing to A
    // Verify lineage_hash calculation correct
}

#[tokio::test]
async fn test_batch_metrics_recorded() {
    // Get initial metrics
    // POST batch with 10 events
    // Get metrics again
    // Verify supplychain_events_ingested_total{event_type="batch"} increased by 10
}
```

**Success Criteria**:
- All 10 tests pass
- Code coverage >80% for `batch.rs`
- CI/CD integration ready

---

### 1.2 Unit Tests for Security Validation

**Problem**: Security validation functions lack unit tests
**Estimated Time**: 30 minutes

**File**: `rust/service/src/security.rs` (add tests module)

**Test Cases**:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_batch_id_valid() {
        assert!(validation::validate_batch_id("BATCH-001").is_ok());
        assert!(validation::validate_batch_id("batch_2025_001").is_ok());
        assert!(validation::validate_batch_id("ABC-123-XYZ").is_ok());
    }

    #[test]
    fn test_validate_batch_id_invalid() {
        assert!(validation::validate_batch_id("").is_err()); // Too short
        assert!(validation::validate_batch_id("batch with spaces").is_err()); // Spaces
        assert!(validation::validate_batch_id("batch@123").is_err()); // Invalid char
        assert!(validation::validate_batch_id(&"a".repeat(129)).is_err()); // Too long
    }

    #[test]
    fn test_validate_metadata_size() {
        let small = r#"{"key": "value"}"#;
        assert!(validation::validate_metadata_size(small).is_ok());

        let large = "a".repeat(11000); // 11KB
        assert!(validation::validate_metadata_size(&large).is_err());
    }

    #[test]
    fn test_validate_array_size() {
        let small_array = vec![1, 2, 3];
        assert!(validation::validate_array_size(&small_array, "test").is_ok());

        let large_array = vec![0; 101];
        assert!(validation::validate_array_size(&large_array, "test").is_err());
    }

    #[test]
    fn test_sanitize_string() {
        assert_eq!(validation::sanitize_string("hello"), "hello");
        assert_eq!(validation::sanitize_string("hello\nworld"), "helloworld");
        assert_eq!(validation::sanitize_string("test\r\n\t"), "test");
    }
}
```

**Success Criteria**:
- 15+ unit tests for validation functions
- Edge cases covered (empty, max size, invalid patterns)
- All tests pass

---

## Priority 2: Query & Lineage API (HIGH)

### 2.1 Lineage Query Endpoint

**Problem**: No way to query full lineage tree for a batch
**Value**: Core feature for supply chain traceability
**Estimated Time**: 60 minutes

**File**: `rust/service/src/lineage_api.rs` (new module)

**New Endpoints**:

```
GET /v1/lineage/:batch_id
GET /v1/lineage/:batch_id/tree
GET /v1/batches/:batch_id/claims
```

**Implementation**:

```rust
/// Get all claims for a specific batch (flat list)
pub async fn get_batch_claims(
    State(state): State<Arc<AppState>>,
    Path(batch_id): Path<String>,
) -> Result<Json<BatchClaimsResponse>, ApiError> {
    // Query database: SELECT * FROM claims WHERE batch_id = ?
    // Return all claims in chronological order
}

/// Get lineage tree for a batch (hierarchical)
pub async fn get_lineage_tree(
    State(state): State<Arc<AppState>>,
    Path(batch_id): Path<String>,
) -> Result<Json<LineageTreeResponse>, ApiError> {
    // 1. Get all claims for this batch
    // 2. Follow previous_claims to build tree
    // 3. Include upstream (sources) and downstream (derivatives)
    // 4. Return hierarchical structure
}

/// Get lineage in both directions
pub async fn get_full_lineage(
    State(state): State<Arc<AppState>>,
    Path(batch_id): Path<String>,
) -> Result<Json<FullLineageResponse>, ApiError> {
    // Combine upstream and downstream lineage
    // Return: { batch_id, upstream: [...], downstream: [...], depth, total_claims }
}
```

**Response Structures**:

```rust
#[derive(Serialize)]
pub struct BatchClaimsResponse {
    batch_id: String,
    claims: Vec<DkgClaim>,
    total: usize,
}

#[derive(Serialize)]
pub struct LineageTreeResponse {
    batch_id: String,
    tree: LineageNode,
    depth: usize,
    total_claims: usize,
}

#[derive(Serialize)]
pub struct LineageNode {
    claim_id: String,
    batch_id: String,
    event_type: String,
    timestamp: String,
    children: Vec<LineageNode>,  // Downstream transformations
}

#[derive(Serialize)]
pub struct FullLineageResponse {
    batch_id: String,
    upstream: Vec<DkgClaim>,    // Source materials
    current: Vec<DkgClaim>,      // This batch
    downstream: Vec<DkgClaim>,   // Derivatives
    depth: usize,
    total_claims: usize,
}
```

**Database Query Enhancement**:

```rust
// Add to db.rs
impl Database {
    pub async fn get_claims_by_batch(&self, batch_id: &str) -> Result<Vec<DkgClaim>> {
        // SELECT * FROM claims WHERE batch_id = ? ORDER BY timestamp ASC
    }

    pub async fn get_downstream_claims(&self, batch_id: &str) -> Result<Vec<DkgClaim>> {
        // SELECT * FROM claims WHERE previous_batches LIKE '%batch_id%'
        // Note: May need JSON extraction for proper querying
    }
}
```

**Success Criteria**:
- 3 new endpoints working
- Coffee demo lineage queryable
- Response time <100ms for typical lineage
- Integration tests for all endpoints

---

### 2.2 Search & Filter API

**Problem**: No way to search claims by product, facility, date range
**Value**: Essential for enterprise users
**Estimated Time**: 45 minutes

**File**: `rust/service/src/search.rs` (new module)

**New Endpoints**:

```
GET /v1/claims?product_id=PRODUCT-001
GET /v1/claims?facility_id=FACILITY-001
GET /v1/claims?event_type=TRANSFORMED
GET /v1/claims?from=2025-01-01&to=2025-12-31
GET /v1/claims?product_id=X&facility_id=Y&limit=50&offset=0
```

**Implementation**:

```rust
#[derive(Deserialize)]
pub struct ClaimFilters {
    product_id: Option<String>,
    batch_id: Option<String>,
    facility_id: Option<String>,
    event_type: Option<String>,
    from: Option<String>,  // ISO 8601 date
    to: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

pub async fn search_claims(
    State(state): State<Arc<AppState>>,
    Query(filters): Query<ClaimFilters>,
) -> Result<Json<SearchResponse>, ApiError> {
    // Build dynamic SQL query based on filters
    // Apply pagination (limit/offset)
    // Return results with total count
}

#[derive(Serialize)]
pub struct SearchResponse {
    claims: Vec<DkgClaim>,
    total: usize,
    limit: usize,
    offset: usize,
    has_more: bool,
}
```

**Database Enhancement**:

```sql
-- Add indexes for common queries
CREATE INDEX idx_claims_product_id ON claims(product_id);
CREATE INDEX idx_claims_batch_id ON claims(batch_id);
CREATE INDEX idx_claims_facility_id ON claims(facility_id);
CREATE INDEX idx_claims_event_type ON claims(event_type);
CREATE INDEX idx_claims_timestamp ON claims(timestamp);
CREATE INDEX idx_claims_composite ON claims(product_id, facility_id, timestamp);
```

**Success Criteria**:
- Search by all major fields
- Pagination working (limit/offset)
- Indexes created for performance
- Response time <200ms for 1000+ claims

---

## Priority 3: Operational Tooling (MEDIUM)

### 3.1 Grafana Dashboard Templates

**Problem**: Prometheus metrics exist but no visualization
**Value**: Operational visibility for production
**Estimated Time**: 45 minutes

**File**: `monitoring/grafana/dashboards/supplychain-overview.json`

**Dashboard Panels**:

1. **Request Rate** (Graph)
   - Query: `rate(supplychain_api_request_duration_seconds_count[5m])`
   - By endpoint: `/v1/events`, `/v1/events/batch`, `/v1/claims/:id`

2. **Error Rate** (Graph)
   - Query: `rate(supplychain_api_request_duration_seconds_count{status=~"4..|5.."}[5m])`
   - Threshold alert: >1% error rate

3. **Latency Percentiles** (Graph)
   - p50: `histogram_quantile(0.5, ...)`
   - p95: `histogram_quantile(0.95, ...)`
   - p99: `histogram_quantile(0.99, ...)`
   - Threshold: p95 <100ms, p99 <200ms

4. **Events Ingested** (Counter)
   - Query: `supplychain_events_ingested_total`
   - By event_type: PRODUCED, TRANSFORMED, SHIPPED, etc.

5. **Claims Stored** (Counter)
   - Query: `supplychain_claims_stored_total`
   - Rate: `rate(supplychain_claims_stored_total[1m])`

6. **Database Performance** (Graph)
   - Query: `supplychain_db_query_duration_seconds`
   - By operation: store_claim, get_claim, get_batch_claims

7. **Active Connections** (Gauge)
   - Query: `supplychain_db_connections_active`

8. **Lineage Depth** (Histogram)
   - Query: `supplychain_lineage_depth`
   - Average depth over time

9. **Batch Processing** (Graph)
   - Batch events/second: `rate(supplychain_events_ingested_total{event_type="batch"}[5m])`
   - Average batch size: calculated from request duration

10. **System Health** (Table)
    - Service uptime
    - Version info
    - Health check status

**Alert Rules** (`monitoring/grafana/alerts/alerts.yaml`):

```yaml
groups:
  - name: supplychain_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(supplychain_api_request_duration_seconds_count{status=~"5.."}[5m]) > 0.01
        for: 2m
        annotations:
          summary: "High error rate (>1%)"

      - alert: HighLatency
        expr: histogram_quantile(0.95, supplychain_api_request_duration_seconds) > 0.1
        for: 5m
        annotations:
          summary: "P95 latency >100ms"

      - alert: ServiceDown
        expr: up{job="supplychain"} == 0
        for: 1m
        annotations:
          summary: "Supplychain service is down"

      - alert: DatabaseConnectionsHigh
        expr: supplychain_db_connections_active > 40
        for: 5m
        annotations:
          summary: "Database connections >40 (max 50)"
```

**Setup Instructions** (`monitoring/README.md`):

```markdown
# Monitoring Setup

## Prometheus

1. Start Prometheus:
   ```bash
   docker-compose up -d prometheus
   ```

2. Verify scraping:
   ```bash
   curl http://localhost:9090/api/v1/targets
   ```

## Grafana

1. Start Grafana:
   ```bash
   docker-compose up -d grafana
   ```

2. Login: http://localhost:3000 (admin/admin)

3. Add Prometheus datasource:
   - URL: http://prometheus:9090
   - Access: Server (default)

4. Import dashboard:
   - Go to Dashboards > Import
   - Upload: `grafana/dashboards/supplychain-overview.json`

## Alert Manager

1. Configure alerts in `grafana/alerts/alerts.yaml`
2. Set up notification channels (email, Slack, PagerDuty)
3. Test alerts with `curl` to trigger conditions
```

**Success Criteria**:
- Complete Grafana dashboard with 10 panels
- 4 alert rules configured
- Docker Compose setup for easy deployment
- Documentation for setup

---

### 3.2 Prometheus Exporter Enhancement

**Problem**: Limited custom metrics
**Value**: More detailed operational insights
**Estimated Time**: 30 minutes

**File**: `rust/service/src/metrics.rs` (enhance existing)

**New Metrics to Add**:

```rust
// Batch-specific metrics
pub static BATCH_SIZE: Lazy<HistogramVec> = Lazy::new(|| {
    HistogramVec::new(
        HistogramOpts::new("supplychain_batch_size", "Size of batch requests")
            .buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0]),
        &["mode"]
    ).unwrap()
});

pub static BATCH_SUCCESS_RATE: Lazy<GaugeVec> = Lazy::new(|| {
    GaugeVec::new(
        GaugeOpts::new("supplychain_batch_success_rate", "Success rate of batch events"),
        &["mode"]
    ).unwrap()
});

// Query metrics
pub static QUERY_RESULTS_COUNT: Lazy<HistogramVec> = Lazy::new(|| {
    HistogramVec::new(
        HistogramOpts::new("supplychain_query_results_count", "Number of results returned by queries")
            .buckets(vec![1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]),
        &["query_type"]
    ).unwrap()
});

// Security metrics
pub static VALIDATION_FAILURES: Lazy<CounterVec> = Lazy::new(|| {
    CounterVec::new(
        Opts::new("supplychain_validation_failures_total", "Total validation failures"),
        &["field", "reason"]
    ).unwrap()
});

pub static RATE_LIMIT_HITS: Lazy<Counter> = Lazy::new(|| {
    Counter::new("supplychain_rate_limit_hits_total", "Rate limit exceeded events").unwrap()
});

// Cache metrics (for future caching implementation)
pub static CACHE_HITS: Lazy<Counter> = Lazy::new(|| {
    Counter::new("supplychain_cache_hits_total", "Cache hits").unwrap()
});

pub static CACHE_MISSES: Lazy<Counter> = Lazy::new(|| {
    Counter::new("supplychain_cache_misses_total", "Cache misses").unwrap()
});
```

**Success Criteria**:
- 7 new metrics added
- Metrics visible in `/metrics` endpoint
- Grafana dashboard updated

---

## Priority 4: Developer Experience (MEDIUM)

### 4.1 Docker Compose for Local Development

**Problem**: Manual setup of service + PostgreSQL + Prometheus + Grafana
**Value**: One-command local environment
**Estimated Time**: 45 minutes

**File**: `docker-compose.yml` (root of repo)

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:16-alpine
    container_name: supplychain-postgres
    environment:
      POSTGRES_DB: supplychain
      POSTGRES_USER: supplychain
      POSTGRES_PASSWORD: dev_password_change_in_prod
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./rust/service/migrations:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U supplychain"]
      interval: 5s
      timeout: 5s
      retries: 5

  # Supply Chain Service
  service:
    build:
      context: ./rust
      dockerfile: service/Dockerfile
    container_name: supplychain-service
    environment:
      DATABASE_URL: postgresql://supplychain:dev_password_change_in_prod@postgres:5432/supplychain
      RUST_LOG: info
      PORT: 8080
    ports:
      - "8080:8080"
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: supplychain-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus/alerts.yml:/etc/prometheus/alerts.yml
      - prometheus_data:/prometheus
    depends_on:
      - service

  # Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: supplychain-grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: false
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:
```

**Prometheus Config** (`monitoring/prometheus/prometheus.yml`):

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'supplychain'
    static_configs:
      - targets: ['service:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

**Grafana Datasource Provisioning** (`monitoring/grafana/datasources/datasource.yml`):

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

**Development Scripts**:

`scripts/dev-start.sh`:
```bash
#!/bin/bash
echo "🚀 Starting Mycelix Supply Chain development environment..."
docker-compose up -d
echo "✅ Services starting:"
echo "  - Service:    http://localhost:8080"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana:    http://localhost:3000 (admin/admin)"
echo ""
echo "Run 'docker-compose logs -f' to view logs"
```

`scripts/dev-stop.sh`:
```bash
#!/bin/bash
echo "🛑 Stopping development environment..."
docker-compose down
echo "✅ All services stopped"
```

`scripts/dev-reset.sh`:
```bash
#!/bin/bash
echo "🔄 Resetting development environment (WARNING: deletes all data)..."
docker-compose down -v
rm -rf data/*.db
echo "✅ Environment reset. Run ./scripts/dev-start.sh to start fresh"
```

**Success Criteria**:
- Single command starts all services
- Health checks working for all containers
- Services can communicate
- README updated with Docker instructions

---

### 4.2 Development Documentation

**File**: `docs/DEVELOPMENT.md`

```markdown
# Development Guide

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Start all services
./scripts/dev-start.sh

# View logs
docker-compose logs -f service

# Stop all services
./scripts/dev-stop.sh

# Reset environment (deletes data)
./scripts/dev-reset.sh
```

### Option 2: Local Development

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Run service locally
cd rust/service
DATABASE_URL=postgresql://supplychain:dev_password_change_in_prod@localhost:5432/supplychain cargo run

# In another terminal, run tests
cargo test
```

## Testing

### Unit Tests
```bash
cd rust
cargo test --lib
```

### Integration Tests
```bash
cd rust
cargo test --test '*'
```

### Load Tests
```bash
# Start service first
./scripts/dev-start.sh

# Run smoke test
cd tests/load
k6 run smoke-test.js

# Run load test
k6 run load-test.js
```

## API Examples

### Ingest Single Event
```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @examples/01-simple-product/event.json
```

### Ingest Batch
```bash
curl -X POST http://localhost:8080/v1/events/batch \
  -H 'Content-Type: application/json' \
  -d '{"events": [...], "mode": "best-effort"}'
```

### Query Lineage
```bash
curl http://localhost:8080/v1/lineage/BATCH-001
```

## Database Migrations

```bash
cd rust/service
sqlx migrate run
```

## Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)
- Metrics: http://localhost:8080/metrics

## Debugging

### View Service Logs
```bash
docker-compose logs -f service
```

### Access Database
```bash
docker-compose exec postgres psql -U supplychain
```

### Check Health
```bash
curl http://localhost:8080/health | jq
```
```

**Success Criteria**:
- Complete development guide
- Quick start instructions
- Testing instructions
- Troubleshooting section

---

## Priority 5: SDK Enhancement (LOW)

### 5.1 TypeScript SDK v2 with Batch Support

**Problem**: Current SDK is basic, no batch support
**Value**: Better developer experience for TypeScript/JavaScript users
**Estimated Time**: 60 minutes

**File**: `sdk/typescript/src/client.ts` (major rewrite)

**New Features**:

```typescript
export class SupplyChainClient {
  private baseUrl: string;
  private timeout: number;
  private retryAttempts: number;

  constructor(config: ClientConfig) {
    this.baseUrl = config.baseUrl;
    this.timeout = config.timeout || 5000;
    this.retryAttempts = config.retryAttempts || 3;
  }

  // Single event ingestion (existing, enhanced)
  async ingestEvent(event: SupplyEventVC): Promise<EventResponse> {
    return this.retry(() => this.post('/v1/events', event));
  }

  // NEW: Batch event ingestion
  async ingestBatch(
    events: SupplyEventVC[],
    options?: BatchOptions
  ): Promise<BatchIngestResponse> {
    const request: BatchIngestRequest = {
      events,
      mode: options?.mode || 'best-effort'
    };
    return this.retry(() => this.post('/v1/events/batch', request));
  }

  // NEW: Get claim by ID
  async getClaim(claimId: string): Promise<DkgClaim> {
    return this.retry(() => this.get(`/v1/claims/${claimId}`));
  }

  // NEW: Get lineage
  async getLineage(batchId: string): Promise<LineageTreeResponse> {
    return this.retry(() => this.get(`/v1/lineage/${batchId}/tree`));
  }

  // NEW: Search claims
  async searchClaims(filters: ClaimFilters): Promise<SearchResponse> {
    const params = new URLSearchParams(filters as any);
    return this.retry(() => this.get(`/v1/claims?${params}`));
  }

  // NEW: Verify VC
  async verifyVC(vcJwt: string): Promise<VerifyResponse> {
    return this.retry(() => this.post('/v1/verify', { vc_jwt: vcJwt }));
  }

  // NEW: Health check
  async health(): Promise<HealthResponse> {
    return this.get('/health');
  }

  // Retry logic with exponential backoff
  private async retry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error;
    for (let attempt = 0; attempt < this.retryAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        if (attempt < this.retryAttempts - 1) {
          const delay = Math.pow(2, attempt) * 1000; // 1s, 2s, 4s
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    throw lastError!;
  }

  // HTTP methods
  private async get<T>(path: string): Promise<T> { /* ... */ }
  private async post<T>(path: string, body: any): Promise<T> { /* ... */ }
}

// Type definitions
export interface BatchOptions {
  mode?: 'best-effort' | 'atomic';
}

export interface BatchIngestRequest {
  events: SupplyEventVC[];
  mode: string;
}

export interface BatchIngestResponse {
  total: number;
  succeeded: number;
  failed: number;
  duration_ms: number;
  results: EventResult[];
}

export interface EventResult {
  index: number;
  status: 'success' | 'failed';
  claim_id?: string;
  lineage_hash?: string;
  error?: string;
  duration_ms: number;
}

export interface ClaimFilters {
  product_id?: string;
  batch_id?: string;
  facility_id?: string;
  event_type?: string;
  from?: string;
  to?: string;
  limit?: number;
  offset?: number;
}
```

**Examples** (`sdk/typescript/examples/batch-ingest.ts`):

```typescript
import { SupplyChainClient } from '../src/client';

const client = new SupplyChainClient({
  baseUrl: 'http://localhost:8080',
  timeout: 10000,
  retryAttempts: 3
});

async function main() {
  // Create batch of events
  const events = [
    createEvent('BATCH-001', 'PRODUCED'),
    createEvent('BATCH-002', 'PRODUCED'),
    createEvent('BATCH-003', 'PRODUCED'),
  ];

  // Ingest batch
  const result = await client.ingestBatch(events, { mode: 'best-effort' });

  console.log(`Total: ${result.total}`);
  console.log(`Succeeded: ${result.succeeded}`);
  console.log(`Failed: ${result.failed}`);
  console.log(`Duration: ${result.duration_ms}ms`);

  // Get lineage
  const lineage = await client.getLineage('BATCH-001');
  console.log(`Lineage depth: ${lineage.depth}`);
}

main().catch(console.error);
```

**Success Criteria**:
- Batch support implemented
- Retry logic with exponential backoff
- All new endpoints supported
- TypeScript types for all responses
- Example code for common use cases
- Tests for SDK

---

## Priority 6: API Documentation (LOW)

### 6.1 OpenAPI/Swagger Specification

**Problem**: No machine-readable API documentation
**Value**: Auto-generated client libraries, API explorer
**Estimated Time**: 45 minutes

**File**: `docs/openapi.yaml`

```yaml
openapi: 3.0.0
info:
  title: Mycelix Supply Chain Provenance API
  version: 0.3.0
  description: |
    REST API for ingesting supply chain events and creating verifiable claims.

    Features:
    - Cryptographically signed verifiable credentials
    - Lineage tracking and resolution
    - Batch event ingestion (up to 100 events)
    - Prometheus metrics and health checks
  contact:
    name: Luminous Dynamics
    email: support@luminousdynamics.com

servers:
  - url: http://localhost:8080
    description: Local development
  - url: https://api.example.com
    description: Production

paths:
  /health:
    get:
      summary: Health check
      description: Returns service health status with component details
      tags: [Monitoring]
      responses:
        '200':
          description: Service is healthy
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/HealthResponse'

  /metrics:
    get:
      summary: Prometheus metrics
      description: Returns metrics in Prometheus text format
      tags: [Monitoring]
      responses:
        '200':
          description: Metrics data
          content:
            text/plain:
              schema:
                type: string

  /v1/events:
    post:
      summary: Ingest single event
      description: Ingest a supply chain event and create verifiable claim
      tags: [Events]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SupplyEventVC'
      responses:
        '201':
          description: Event ingested successfully
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EventResponse'
        '400':
          description: Validation error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/events/batch:
    post:
      summary: Ingest batch of events
      description: Ingest up to 100 events in a single request
      tags: [Events]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/BatchIngestRequest'
      responses:
        '201':
          description: Batch processed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BatchIngestResponse'
        '400':
          description: Validation error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/claims/{id}:
    get:
      summary: Get claim by ID
      description: Retrieve a verifiable claim by its ID
      tags: [Claims]
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Claim found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ClaimResponse'
        '404':
          description: Claim not found
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /v1/lineage/{batch_id}:
    get:
      summary: Get lineage tree
      description: Get full lineage tree for a batch
      tags: [Lineage]
      parameters:
        - name: batch_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Lineage tree
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LineageTreeResponse'

  /v1/verify:
    post:
      summary: Verify VC
      description: Verify a verifiable credential JWT
      tags: [Verification]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/VerifyRequest'
      responses:
        '200':
          description: Verification result
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/VerifyResponse'

components:
  schemas:
    HealthResponse:
      type: object
      properties:
        status:
          type: string
          enum: [healthy, degraded, unhealthy]
        version:
          type: string
        timestamp:
          type: string
          format: date-time
        components:
          type: object
          properties:
            database:
              $ref: '#/components/schemas/ComponentStatus'
            storage:
              $ref: '#/components/schemas/ComponentStatus'

    # ... (rest of schemas)

tags:
  - name: Events
    description: Event ingestion endpoints
  - name: Claims
    description: Claim retrieval endpoints
  - name: Lineage
    description: Lineage query endpoints
  - name: Verification
    description: Verification endpoints
  - name: Monitoring
    description: Health and metrics endpoints
```

**Swagger UI Setup**:

```bash
# Install Swagger UI as Docker service in docker-compose.yml
swagger-ui:
  image: swaggerapi/swagger-ui
  container_name: supplychain-swagger
  ports:
    - "8081:8080"
  environment:
    SWAGGER_JSON: /docs/openapi.yaml
  volumes:
    - ./docs/openapi.yaml:/docs/openapi.yaml
```

**Success Criteria**:
- Complete OpenAPI 3.0 spec
- All endpoints documented
- Swagger UI accessible
- Example requests/responses
- Can generate client libraries

---

## Timeline & Estimates

| Priority | Task | Estimated Time |
|----------|------|---------------|
| 1 | Integration tests for batch API | 45 min |
| 1 | Unit tests for security validation | 30 min |
| 2 | Lineage query endpoint | 60 min |
| 2 | Search & filter API | 45 min |
| 3 | Grafana dashboard templates | 45 min |
| 3 | Prometheus exporter enhancement | 30 min |
| 4 | Docker Compose setup | 45 min |
| 4 | Development documentation | 30 min |
| 5 | TypeScript SDK v2 | 60 min |
| 6 | OpenAPI specification | 45 min |
| **Total** | | **6 hours 15 minutes** |

**Realistic Completion**: 3-4 hours (focus on P1-P3)

---

## Success Metrics

### Code Quality
- [ ] Test coverage >80% for new features
- [ ] All integration tests passing
- [ ] Zero compilation warnings

### Developer Experience
- [ ] One-command local environment setup
- [ ] Clear documentation for all features
- [ ] SDK supports all major operations

### Operational Excellence
- [ ] Grafana dashboards showing all key metrics
- [ ] Alerts configured for critical issues
- [ ] Monitoring documented

### API Completeness
- [ ] Lineage query working end-to-end
- [ ] Search API supports all common filters
- [ ] OpenAPI spec complete and validated

---

## Dependencies

- Docker & Docker Compose
- PostgreSQL 16
- Prometheus & Grafana
- K6 (for load testing)
- Node.js/npm (for SDK)

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Database schema changes | HIGH | Use migrations, version schema |
| Breaking API changes | MEDIUM | Use versioning (e.g., /v2/...) |
| Performance regression | MEDIUM | Load test before/after changes |
| Complex lineage queries | MEDIUM | Add database indexes, optimize queries |

---

## Future Considerations (Post-Phase 6)

- **Phase 7**: Authentication & Authorization (JWT, API keys, RBAC)
- **Phase 8**: Webhooks & Event Streaming (notify on events)
- **Phase 9**: Export & Reporting (CSV, PDF certificates)
- **Phase 10**: Mobile SDKs (iOS, Android)
- **Phase 11**: Blockchain anchoring
- **Phase 12**: Advanced crypto (SD-JWT, BBS+ signatures)

---

## Conclusion

Phase 6 transforms a production-ready system into a **production-excellent** system with:
- ✅ Comprehensive testing
- ✅ Powerful query capabilities
- ✅ World-class monitoring
- ✅ Excellent developer experience
- ✅ Complete documentation

**Let's build it!** 🚀
