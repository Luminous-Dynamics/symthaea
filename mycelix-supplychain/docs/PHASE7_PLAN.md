# Phase 7 Plan - API Completeness & Performance

**Goal**: Complete API testing, optimize performance, and enhance developer ecosystem
**Status**: 📋 In Progress
**Estimated Duration**: 3-4 hours
**Focus**: Testing, Performance, Documentation, SDK Enhancement

---

## Executive Summary

Phase 6 delivered excellent developer experience and observability. Phase 7 focuses on **completeness, performance, and ecosystem** - ensuring all new features are tested, optimized, documented, and accessible through SDKs.

### Key Objectives

1. **Testing Completeness** - Integration tests for all Phase 6 lineage endpoints
2. **Performance Optimization** - Database indexes and query optimization
3. **API Documentation** - Updated OpenAPI spec with batch and lineage APIs
4. **SDK Enhancement** - TypeScript SDK v2 with batch and lineage support
5. **Structured Logging** - Request tracing and correlation IDs
6. **Performance Testing** - Regression tests and benchmarks

---

## Priority 1: Testing Completeness (HIGH)

### 1.1 Lineage API Integration Tests

**Problem**: New lineage endpoints lack integration tests
**Risk**: Bugs in production lineage queries
**Estimated Time**: 45 minutes

**File**: `rust/service/tests/integration_lineage.rs` (NEW)

**Test Scenarios**:

```rust
#[tokio::test]
async fn test_get_batch_claims_success() {
    // Create 3 events for BATCH-001
    // GET /v1/batches/BATCH-001/claims
    // Assert 200 OK, total_claims=3
    // Verify all claims have correct batch_id
    // Verify claims sorted by timestamp (newest first)
}

#[tokio::test]
async fn test_get_batch_claims_empty() {
    // GET /v1/batches/NONEXISTENT/claims
    // Assert 200 OK, total_claims=0, claims=[]
}

#[tokio::test]
async fn test_get_lineage_single_batch() {
    // Create PRODUCED event for BATCH-001
    // GET /v1/lineage/BATCH-001
    // Assert 200 OK, upstream=None, downstream=None
    // Verify depth=0, total_claims=1
}

#[tokio::test]
async fn test_get_lineage_with_upstream() {
    // Create BATCH-001 (PRODUCED)
    // Create BATCH-002 (PRODUCED)
    // Create BATCH-ASM-001 (TRANSFORMED from BATCH-001, BATCH-002)
    // GET /v1/lineage/BATCH-ASM-001
    // Assert upstream contains BATCH-001 and BATCH-002
    // Verify depth=1
}

#[tokio::test]
async fn test_get_lineage_with_downstream() {
    // Create BATCH-001 (PRODUCED)
    // Create BATCH-PKG-001 (TRANSFORMED from BATCH-001)
    // GET /v1/lineage/BATCH-001
    // Assert downstream contains BATCH-PKG-001
}

#[tokio::test]
async fn test_get_lineage_complex_graph() {
    // Create 5-level lineage tree
    // Verify depth calculation
    // Verify all ancestors/descendants found
}

#[tokio::test]
async fn test_search_claims_by_product() {
    // Create claims for SKU-001, SKU-002, SKU-003
    // GET /v1/claims?product_id=SKU-001
    // Assert only SKU-001 claims returned
}

#[tokio::test]
async fn test_search_claims_by_batch() {
    // Create claims for multiple batches
    // GET /v1/claims?batch_id=BATCH-001
    // Assert only BATCH-001 claims
}

#[tokio::test]
async fn test_search_claims_by_facility() {
    // Create claims at FAC-A, FAC-B, FAC-C
    // GET /v1/claims?facility_id=FAC-A
    // Assert only FAC-A claims
}

#[tokio::test]
async fn test_search_claims_by_event_type() {
    // Create PRODUCED, SHIPPED, TRANSFORMED events
    // GET /v1/claims?event_type=PRODUCED
    // Assert only PRODUCED events
}

#[tokio::test]
async fn test_search_claims_date_range() {
    // Create claims across 3 days
    // GET /v1/claims?from=DAY2&to=DAY2
    // Assert only DAY2 claims
}

#[tokio::test]
async fn test_search_claims_pagination() {
    // Create 100 claims
    // GET /v1/claims?limit=50&offset=0
    // Assert 50 results, has_more=true
    // GET /v1/claims?limit=50&offset=50
    // Assert 50 results, has_more=false
}

#[tokio::test]
async fn test_search_claims_combined_filters() {
    // Create diverse claim set
    // GET /v1/claims?product_id=SKU-001&event_type=PRODUCED
    // Assert only matching claims
}

#[tokio::test]
async fn test_search_claims_performance() {
    // Create 500 claims
    // GET /v1/claims?product_id=SKU-001
    // Measure response time
    // Assert <200ms for filtered results
}
```

**Estimated Tests**: 14 tests
**Coverage Target**: 90%+ of lineage_api.rs

---

## Priority 2: Performance Optimization (HIGH)

### 2.1 Database Indexes

**Problem**: Lineage queries use full table scans
**Impact**: Slow queries on large datasets (>10k claims)
**Estimated Time**: 30 minutes

**File**: `rust/service/migrations/YYYYMMDDHHMMSS_add_lineage_indexes.sql` (NEW)

```sql
-- Index for batch_id lookups (used by get_batch_claims)
CREATE INDEX IF NOT EXISTS idx_claims_batch_id ON claims(batch_id);

-- Index for product_id filtering (used by search)
CREATE INDEX IF NOT EXISTS idx_claims_product_id ON claims(product_id);

-- Index for timestamp range queries
CREATE INDEX IF NOT EXISTS idx_claims_timestamp ON claims(timestamp DESC);

-- Index for event_type filtering
CREATE INDEX IF NOT EXISTS idx_claims_event_type ON claims(event_type);

-- Composite index for common search patterns
CREATE INDEX IF NOT EXISTS idx_claims_product_timestamp
ON claims(product_id, timestamp DESC);

-- Index for facility_id filtering (JSON extraction)
-- Note: This requires extracting facility_id from claim_json
-- For SQLite: Not efficient with JSON. For PostgreSQL:
-- CREATE INDEX idx_claims_facility_id ON claims((claim_json->>'assertion'->>'facility_id'));
```

**Performance Impact**:
- `get_batch_claims`: 100ms → 5ms (20x faster)
- `search by product`: 200ms → 10ms (20x faster)
- `date range queries`: 150ms → 8ms (18x faster)

### 2.2 Query Optimization

**Problem**: `get_all_claims()` loads entire table into memory
**Impact**: Memory exhaustion with >100k claims
**Estimated Time**: 45 minutes

**File**: `rust/service/src/db.rs` (MODIFY)

**Changes**:

1. **Server-Side Filtering** - Push filters to SQL WHERE clause
```rust
pub async fn search_claims(
    &self,
    filters: &ClaimFilters,
) -> Result<Vec<DkgClaim>> {
    let mut query = String::from("SELECT claim_json FROM claims WHERE 1=1");
    let mut bindings = Vec::new();

    if let Some(ref batch_id) = filters.batch_id {
        query.push_str(" AND batch_id = ?");
        bindings.push(batch_id);
    }

    if let Some(ref product_id) = filters.product_id {
        query.push_str(" AND product_id = ?");
        bindings.push(product_id);
    }

    if let Some(ref event_type) = filters.event_type {
        query.push_str(" AND event_type = ?");
        bindings.push(event_type);
    }

    if let Some(ref from) = filters.from {
        query.push_str(" AND timestamp >= ?");
        bindings.push(from);
    }

    if let Some(ref to) = filters.to {
        query.push_str(" AND timestamp <= ?");
        bindings.push(to);
    }

    query.push_str(" ORDER BY timestamp DESC LIMIT ? OFFSET ?");
    bindings.push(&filters.limit.to_string());
    bindings.push(&filters.offset.to_string());

    // Execute query with bindings
    // ...
}
```

2. **Cursor-Based Pagination** (Future)
```rust
pub struct CursorPagination {
    pub cursor: Option<String>,  // encoded (timestamp, id)
    pub limit: usize,
}
```

**File**: `rust/service/src/lineage_api.rs` (MODIFY)

**Changes**:
- Replace `get_all_claims()` with `db.search_claims(&filters)`
- Remove in-memory filtering
- Keep sorting at database level

**Performance Impact**:
- Memory: O(all_claims) → O(limit)
- Query time: O(n) → O(log n) with indexes
- Scalability: 10k claims → 1M+ claims

### 2.3 Connection Pooling Tuning

**File**: `rust/service/src/db.rs` (MODIFY)

```rust
let pool = SqlxPostgresPoolOptions::new()
    .max_connections(20)        // Increase from default 10
    .min_connections(5)         // Keep warm connections
    .acquire_timeout(Duration::from_secs(5))
    .idle_timeout(Some(Duration::from_secs(300)))
    .max_lifetime(Some(Duration::from_secs(1800)))
    .connect(&database_url)
    .await?;
```

---

## Priority 3: API Documentation (MEDIUM)

### 3.1 Update OpenAPI Specification

**Problem**: OpenAPI spec missing batch and new lineage endpoints
**Impact**: Incomplete API documentation, SDK generation issues
**Estimated Time**: 30 minutes

**File**: `specs/openapi.yaml` (MODIFY)

**Add Endpoints**:

1. **POST /v1/events/batch**
```yaml
  /v1/events/batch:
    post:
      summary: Ingest multiple events in a single request
      tags: [events]
      description: |
        High-volume batch ingestion (up to 100 events per request).
        Supports "best-effort" (partial success) and "atomic" (all-or-nothing) modes.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [events]
              properties:
                events:
                  type: array
                  items:
                    $ref: '#/components/schemas/SupplyEvent'
                  minItems: 1
                  maxItems: 100
                mode:
                  type: string
                  enum: [best-effort, atomic]
                  default: best-effort
      responses:
        '201':
          description: Batch processed
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BatchResponse'
```

2. **GET /v1/batches/:batch_id/claims**
```yaml
  /v1/batches/{batchId}/claims:
    get:
      summary: Get all claims for a batch
      tags: [claims]
      parameters:
        - in: path
          name: batchId
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Batch claims retrieved
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/BatchClaimsResponse'
```

3. **GET /v1/lineage/:batch_id**
```yaml
  /v1/lineage/{batchId}:
    get:
      summary: Get full lineage graph for a batch
      tags: [claims]
      description: |
        Returns upstream (source) and downstream (derivative) batches
        with complete lineage traversal.
      parameters:
        - in: path
          name: batchId
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Lineage graph retrieved
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LineageResponse'
```

4. **GET /v1/claims (search)**
```yaml
  /v1/claims:
    get:
      summary: Search and filter claims
      tags: [claims]
      parameters:
        - in: query
          name: product_id
          schema:
            type: string
        - in: query
          name: batch_id
          schema:
            type: string
        - in: query
          name: facility_id
          schema:
            type: string
        - in: query
          name: event_type
          schema:
            type: string
            enum: [PRODUCED, TRANSFORMED, SHIPPED, RECEIVED, CERTIFIED]
        - in: query
          name: from
          schema:
            type: string
            format: date-time
        - in: query
          name: to
          schema:
            type: string
            format: date-time
        - in: query
          name: limit
          schema:
            type: integer
            default: 50
            minimum: 1
            maximum: 1000
        - in: query
          name: offset
          schema:
            type: integer
            default: 0
            minimum: 0
      responses:
        '200':
          description: Claims retrieved
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SearchResponse'
```

**Add Schemas**:
```yaml
components:
  schemas:
    BatchResponse:
      type: object
      properties:
        total: {type: integer}
        succeeded: {type: integer}
        failed: {type: integer}
        results:
          type: array
          items:
            type: object
            properties:
              index: {type: integer}
              success: {type: boolean}
              claim_id: {type: string}
              vc_jwt: {type: string}
              error: {type: string}

    BatchClaimsResponse:
      type: object
      properties:
        batch_id: {type: string}
        claims:
          type: array
          items:
            $ref: '#/components/schemas/Claim'
        total_claims: {type: integer}

    LineageResponse:
      type: object
      properties:
        batch_id: {type: string}
        claims:
          type: array
          items:
            $ref: '#/components/schemas/Claim'
        upstream:
          type: array
          items:
            type: object
            properties:
              batch_id: {type: string}
              claim_count: {type: integer}
              depth: {type: integer}
        downstream:
          type: array
          items:
            type: object
            properties:
              batch_id: {type: string}
              claim_count: {type: integer}
              depth: {type: integer}
        total_claims: {type: integer}
        depth: {type: integer}

    SearchResponse:
      type: object
      properties:
        claims:
          type: array
          items:
            $ref: '#/components/schemas/Claim'
        total: {type: integer}
        limit: {type: integer}
        offset: {type: integer}
        has_more: {type: boolean}
```

---

## Priority 4: SDK Enhancement (MEDIUM)

### 4.1 TypeScript SDK v2

**Problem**: SDK doesn't support batch or new lineage endpoints
**Impact**: Developers must use raw HTTP for new features
**Estimated Time**: 45 minutes

**File**: `ts/sdk/src/client.ts` (MODIFY)

**Add Types** (`ts/sdk/src/types.ts`):
```typescript
export interface BatchRequest {
  events: SupplyEventVC[];
  mode?: 'best-effort' | 'atomic';
}

export interface BatchResult {
  index: number;
  success: boolean;
  claim_id?: string;
  vc_jwt?: string;
  lineage_hash?: string;
  error?: string;
}

export interface BatchResponse {
  total: number;
  succeeded: number;
  failed: number;
  results: BatchResult[];
}

export interface BatchClaimsResponse {
  batch_id: string;
  claims: DkgClaim[];
  total_claims: number;
}

export interface UpstreamBatch {
  batch_id: string;
  claim_count: number;
  depth: number;
}

export interface DownstreamBatch {
  batch_id: string;
  claim_count: number;
  depth: number;
}

export interface LineageResponse {
  batch_id: string;
  claims: DkgClaim[];
  upstream?: UpstreamBatch[];
  downstream?: DownstreamBatch[];
  total_claims: number;
  depth: number;
}

export interface ClaimFilters {
  product_id?: string;
  batch_id?: string;
  facility_id?: string;
  event_type?: 'PRODUCED' | 'TRANSFORMED' | 'SHIPPED' | 'RECEIVED' | 'CERTIFIED';
  from?: string;
  to?: string;
  limit?: number;
  offset?: number;
}

export interface SearchResponse {
  claims: DkgClaim[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}
```

**Add Methods** (`ts/sdk/src/client.ts`):
```typescript
/**
 * Ingest multiple events in a batch
 */
async ingestBatch(request: BatchRequest): Promise<BatchResponse> {
  const response = await this.client.post<BatchResponse>(
    '/v1/events/batch',
    request
  );
  return response.data;
}

/**
 * Get all claims for a batch
 */
async getBatchClaims(batchId: string): Promise<BatchClaimsResponse> {
  const response = await this.client.get<BatchClaimsResponse>(
    `/v1/batches/${batchId}/claims`
  );
  return response.data;
}

/**
 * Get full lineage graph for a batch
 */
async getLineage(batchId: string): Promise<LineageResponse> {
  const response = await this.client.get<LineageResponse>(
    `/v1/lineage/${batchId}`
  );
  return response.data;
}

/**
 * Search and filter claims
 */
async searchClaims(filters: ClaimFilters = {}): Promise<SearchResponse> {
  const response = await this.client.get<SearchResponse>('/v1/claims', {
    params: filters,
  });
  return response.data;
}

/**
 * Helper: Create a batch of events
 */
createBatch(events: SupplyEventVC[], mode?: 'best-effort' | 'atomic'): BatchRequest {
  return {
    events,
    mode: mode || 'best-effort',
  };
}
```

**Update Package Version**: `ts/sdk/package.json`
```json
{
  "version": "0.2.0",
  "description": "Mycelix Supply Chain SDK with batch and lineage support"
}
```

**Add Examples** (`ts/sdk/examples/batch-example.ts`):
```typescript
import { SupplyChainClient } from '@mycelix/supplychain-sdk';

const client = new SupplyChainClient({
  baseUrl: 'http://localhost:8080',
});

// Create batch of events
const events = [
  client.createProducedEvent({...}),
  client.createProducedEvent({...}),
  client.createProducedEvent({...}),
];

// Ingest batch
const result = await client.ingestBatch({
  events,
  mode: 'best-effort',
});

console.log(`Succeeded: ${result.succeeded}/${result.total}`);

// Get lineage
const lineage = await client.getLineage('BATCH-001');
console.log(`Upstream batches: ${lineage.upstream?.length || 0}`);
console.log(`Downstream batches: ${lineage.downstream?.length || 0}`);

// Search claims
const results = await client.searchClaims({
  product_id: 'SKU-COFFEE-ROASTED',
  event_type: 'PRODUCED',
  limit: 50,
});

console.log(`Found ${results.total} claims`);
```

---

## Priority 5: Structured Logging (MEDIUM)

### 5.1 Request Tracing

**Problem**: Difficult to correlate logs across async operations
**Impact**: Slower debugging in production
**Estimated Time**: 30 minutes

**File**: `rust/service/Cargo.toml` (ADD)
```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
uuid = { version = "1.0", features = ["v4"] }
```

**File**: `rust/service/src/tracing.rs` (NEW)
```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use uuid::Uuid;

pub fn init_tracing() {
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer()
            .with_target(true)
            .with_level(true)
            .with_thread_ids(true)
            .json())
        .init();
}

// Middleware to add request_id to all logs
pub async fn request_tracing_middleware(
    req: Request,
    next: Next,
) -> Response {
    let request_id = Uuid::new_v4().to_string();

    let _span = tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %req.method(),
        path = %req.uri().path(),
    ).entered();

    tracing::info!("Request started");

    let response = next.run(req).await;

    tracing::info!(
        status = %response.status(),
        "Request completed"
    );

    response
}
```

**File**: `rust/service/src/main.rs` (MODIFY)
```rust
mod tracing;

#[tokio::main]
async fn main() {
    tracing::init_tracing();

    let app = Router::new()
        .route(...)
        .layer(axum::middleware::from_fn(tracing::request_tracing_middleware))
        .layer(axum::middleware::from_fn(security::security_headers_middleware));

    // ...
}
```

**Log Format**:
```json
{
  "timestamp": "2025-11-16T10:30:45.123Z",
  "level": "INFO",
  "target": "provenance_service",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "path": "/v1/events/batch",
  "message": "Processing batch of 10 events"
}
```

---

## Priority 6: Performance Testing (LOW)

### 6.1 Regression Test Suite

**File**: `rust/service/benches/api_bench.rs` (NEW)

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_batch_ingestion(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    c.bench_function("batch_10_events", |b| {
        b.to_async(&rt).iter(|| async {
            let events = create_test_events(10);
            let result = ingest_batch(events).await;
            black_box(result)
        });
    });

    c.bench_function("batch_50_events", |b| {
        b.to_async(&rt).iter(|| async {
            let events = create_test_events(50);
            let result = ingest_batch(events).await;
            black_box(result)
        });
    });

    c.bench_function("batch_100_events", |b| {
        b.to_async(&rt).iter(|| async {
            let events = create_test_events(100);
            let result = ingest_batch(events).await;
            black_box(result)
        });
    });
}

criterion_group!(benches, bench_batch_ingestion);
criterion_main!(benches);
```

**Run Benchmarks**:
```bash
cargo bench --bench api_bench
```

---

## Implementation Order

### Phase 7.1: Testing & Performance (90 min)
1. ✅ Create `integration_lineage.rs` with 14 tests (45 min)
2. ✅ Add database indexes migration (15 min)
3. ✅ Optimize `search_claims` with server-side filtering (30 min)

### Phase 7.2: Documentation & SDK (75 min)
4. ✅ Update OpenAPI spec with 4 new endpoints (30 min)
5. ✅ Enhance TypeScript SDK with batch/lineage support (45 min)

### Phase 7.3: Observability (30 min)
6. ✅ Add structured logging with request tracing (30 min)

### Phase 7.4: Finalization (15 min)
7. ✅ Create Phase 7 summary document (10 min)
8. ✅ Commit and push Phase 7 completion (5 min)

**Total Estimated Time**: 3.5 hours

---

## Success Criteria

### Testing
- ✅ 14+ integration tests for lineage API
- ✅ 95%+ code coverage for `lineage_api.rs`
- ✅ All tests passing in <2s total

### Performance
- ✅ `get_batch_claims`: <10ms (from ~100ms)
- ✅ `search_claims`: <20ms for 1000 results
- ✅ Lineage traversal: <50ms for depth 5

### Documentation
- ✅ OpenAPI spec 100% complete
- ✅ All endpoints documented with examples
- ✅ Schema definitions for all response types

### SDK
- ✅ TypeScript SDK v0.2.0 published
- ✅ 4 new methods: `ingestBatch`, `getBatchClaims`, `getLineage`, `searchClaims`
- ✅ Type definitions for all new responses
- ✅ Example code for batch and lineage operations

### Observability
- ✅ Request ID in all logs
- ✅ JSON structured logging
- ✅ Correlated logs across async operations

---

## Future Enhancements (Phase 8 Candidates)

1. **GraphQL API** - Alternative to REST for flexible querying
2. **WebSocket Subscriptions** - Real-time claim updates
3. **Distributed Tracing** - OpenTelemetry integration
4. **Advanced Analytics** - Lineage graph visualization
5. **Multi-Tenant Support** - Organization isolation
6. **Rate Limiting** - Token bucket implementation
7. **Caching Layer** - Redis for frequently accessed claims
8. **Async Processing** - Background job queue for batch operations

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Database migration fails in production | Low | High | Test migrations on staging, backup before apply |
| SDK breaking changes | Medium | Medium | Semantic versioning, migration guide |
| Performance regression | Low | Medium | Benchmark suite, load testing before deploy |
| Index overhead on writes | Low | Low | Monitor write latency, adjust if needed |

---

## Metrics to Track

**Before Phase 7**:
- Query performance: 100-200ms for searches
- Test coverage: 85% (lineage API untested)
- SDK version: 0.1.0 (missing features)
- OpenAPI completeness: 70% (4/7 endpoints)

**After Phase 7**:
- Query performance: 5-20ms for searches (10x faster)
- Test coverage: 95%+ (14 new tests)
- SDK version: 0.2.0 (full feature parity)
- OpenAPI completeness: 100% (7/7 endpoints)

---

**Status**: Ready to execute
**Next Step**: Create `integration_lineage.rs` test file
