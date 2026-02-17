# Phase 9: Production Excellence & Developer Experience

**Date**: 2025-11-16
**Previous Phase**: Phase 8 (API Documentation & SDK v2)
**Focus**: Production readiness, observability, performance, security
**Estimated Duration**: 4-5 hours
**Status**: 🚧 In Progress

---

## Overview

Phase 9 transforms the mycelix-supplychain platform from developer-ready to **production-ready enterprise system**. This phase focuses on:

1. **Testing Excellence** - 100% test pass rate, comprehensive coverage
2. **Developer Experience** - Rich examples, clear patterns, quick-start guides
3. **Observability** - Structured logging, request tracing, metrics
4. **Performance** - Benchmarks, optimization, caching
5. **Security** - Rate limiting, input validation, security headers
6. **Operations** - Health checks, graceful shutdown, deployment guides

---

## Success Criteria

### Must Have (Priority 1-3)
- ✅ 100% integration test pass rate (15/15 tests)
- ✅ Comprehensive TypeScript SDK examples (5+ scenarios)
- ✅ Structured logging with request tracing
- ✅ Performance benchmarks for critical paths
- ✅ Rate limiting on all endpoints

### Should Have (Priority 4-5)
- ✅ Response caching layer
- ✅ Security headers (CORS, CSP, etc.)
- ✅ Input validation hardening
- ✅ Database query optimization
- ✅ Prometheus metrics export

### Nice to Have (Priority 6)
- 📋 Load testing suite
- 📋 Deployment guide (Docker, k8s)
- 📋 Operations runbook
- 📋 Performance dashboard

---

## Priority 1: Testing Excellence (15 min)

### Objective
Fix 4 failing integration tests and achieve 100% pass rate.

### Current State
- **Passing**: 11/15 tests (73%)
- **Failing**: 4/15 tests (27%)

### Failed Tests Analysis

From `rust/service/tests/integration_lineage.rs`:

1. **`test_get_batch_claims_empty`**
   - **Issue**: Assertion expects null for empty results
   - **Fix**: Check if response returns empty array or null properly

2. **`test_get_batch_claims_response_structure`**
   - **Issue**: Field type mismatch in response structure
   - **Fix**: Align assertion with actual API response schema

3. **`test_get_lineage_nonexistent_batch`**
   - **Issue**: Error handling not matching expected response
   - **Fix**: Verify 404 response structure and error format

4. **`test_get_lineage_response_structure`**
   - **Issue**: Optional field handling (upstream/downstream can be null)
   - **Fix**: Update assertions to handle optional fields correctly

### Implementation Steps

1. **Read failing test file** (`rust/service/tests/integration_lineage.rs`)
   ```bash
   # Lines to focus on: test definitions for the 4 failing tests
   ```

2. **Run tests individually to see exact failures**
   ```bash
   cd rust/service
   cargo test test_get_batch_claims_empty -- --nocapture
   cargo test test_get_batch_claims_response_structure -- --nocapture
   cargo test test_get_lineage_nonexistent_batch -- --nocapture
   cargo test test_get_lineage_response_structure -- --nocapture
   ```

3. **Fix assertion logic**
   - Update test expectations to match actual API behavior
   - Ensure optional fields handled correctly
   - Verify error response formats

4. **Run full test suite**
   ```bash
   cargo test
   ```

### Files to Modify
- `rust/service/tests/integration_lineage.rs`

### Success Criteria
- ✅ All 15 integration tests passing
- ✅ No test warnings or ignored tests
- ✅ Clean test output

### Time Estimate
**15 minutes**

---

## Priority 2: TypeScript SDK Examples (45 min)

### Objective
Create comprehensive, production-ready examples showing real-world SDK usage patterns.

### Example Scenarios

#### 1. Basic Event Ingestion (`ts/examples/01-basic-ingestion.ts`)
**Scenario**: Single event submission (PRODUCED event)

```typescript
import { SupplyChainClient } from '@mycelix/supplychain-sdk';

async function basicIngestion() {
  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  // Create a PRODUCED event
  const event = client.createProducedEvent(
    'BATCH-2025-001',
    'ORG-ACME',
    'Organic Coffee Beans',
    1000,
    'kg',
    { origin: 'Colombia', grade: 'Premium' }
  );

  // Submit event
  const response = await client.ingestEvent(event);

  console.log('✅ Event ingested successfully');
  console.log('Claim ID:', response.claim_id);
  console.log('VC JWT:', response.vc_jwt.substring(0, 50) + '...');
  console.log('Lineage Hash:', response.lineage_hash);
}

basicIngestion().catch(console.error);
```

**Learning Goals**:
- SDK initialization
- Event creation helpers
- Single event submission
- Response handling

---

#### 2. Batch Processing (`ts/examples/02-batch-processing.ts`)
**Scenario**: High-volume event ingestion with error handling

```typescript
import { SupplyChainClient, BatchRequest } from '@mycelix/supplychain-sdk';

async function batchProcessing() {
  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  // Create multiple events for a production run
  const events = [
    client.createProducedEvent(
      'BATCH-COFFEE-001',
      'ORG-ROASTER',
      'Dark Roast Coffee',
      500,
      'kg',
      { roast_level: 'dark', temperature: '220C' }
    ),
    client.createProducedEvent(
      'BATCH-COFFEE-002',
      'ORG-ROASTER',
      'Medium Roast Coffee',
      500,
      'kg',
      { roast_level: 'medium', temperature: '210C' }
    ),
    client.createTransformedEvent(
      'BATCH-BLEND-001',
      'ORG-ROASTER',
      'House Blend',
      ['BATCH-COFFEE-001', 'BATCH-COFFEE-002'],
      800,
      'kg',
      { blend_ratio: '50:50' }
    ),
  ];

  // Submit as batch (best-effort mode)
  console.log('📦 Submitting batch of', events.length, 'events...');
  const batch = client.createBatch(events, 'best-effort');
  const response = await client.ingestBatch(batch);

  // Process results
  console.log('\n📊 Batch Results:');
  console.log('Total:', response.total);
  console.log('Succeeded:', response.succeeded);
  console.log('Failed:', response.failed);
  console.log('Duration:', response.duration_ms, 'ms');

  // Check individual results
  response.results.forEach((result, index) => {
    if (result.status === 'success') {
      console.log(`✅ Event ${index}: ${result.claim_id}`);
    } else {
      console.error(`❌ Event ${index}: ${result.error}`);
    }
  });

  // Retry failed events (if any)
  const failedEvents = response.results
    .filter(r => r.status === 'error')
    .map(r => events[r.index]);

  if (failedEvents.length > 0) {
    console.log('\n🔄 Retrying', failedEvents.length, 'failed events...');
    const retryBatch = client.createBatch(failedEvents, 'atomic');
    const retryResponse = await client.ingestBatch(retryBatch);
    console.log('Retry succeeded:', retryResponse.succeeded);
  }
}

batchProcessing().catch(console.error);
```

**Learning Goals**:
- Batch creation
- Best-effort vs atomic modes
- Error handling and recovery
- Result processing

---

#### 3. Supply Chain Lineage (`ts/examples/03-lineage-tracking.ts`)
**Scenario**: Trace complete supply chain from source to final product

```typescript
import { SupplyChainClient } from '@mycelix/supplychain-sdk';

async function lineageTracking() {
  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  // Create a supply chain: Raw Material → Processing → Distribution

  // 1. Raw material production
  const rawMaterial = client.createProducedEvent(
    'BATCH-RAW-001',
    'ORG-FARM',
    'Organic Cacao Beans',
    1000,
    'kg',
    { farm: 'Peru Highlands', certification: 'USDA Organic' }
  );

  // 2. Processing/transformation
  const processed = client.createTransformedEvent(
    'BATCH-PROC-001',
    'ORG-FACTORY',
    'Dark Chocolate Bars',
    ['BATCH-RAW-001'],
    800,
    'kg',
    { cocoa_content: '85%', process: 'stone-ground' }
  );

  // 3. Packaging
  const packaged = client.createTransformedEvent(
    'BATCH-PKG-001',
    'ORG-PACKAGER',
    'Retail Chocolate Bars (100g)',
    ['BATCH-PROC-001'],
    8000,
    'units',
    { packaging: 'recyclable', weight_per_unit: '100g' }
  );

  // Submit all events as batch
  const events = [rawMaterial, processed, packaged];
  const batch = client.createBatch(events, 'atomic');
  await client.ingestBatch(batch);

  console.log('✅ Supply chain events ingested\n');

  // Query lineage for final product
  console.log('🔍 Querying lineage for BATCH-PKG-001...\n');
  const lineage = await client.getLineage('BATCH-PKG-001');

  console.log('📊 Lineage Graph:');
  console.log('Batch ID:', lineage.batch_id);
  console.log('Total Claims:', lineage.total_claims);
  console.log('Graph Depth:', lineage.depth);

  // Show upstream sources
  if (lineage.upstream && lineage.upstream.length > 0) {
    console.log('\n⬆️  Upstream Sources:');
    lineage.upstream.forEach(batch => {
      console.log(`  - ${batch.batch_id} (${batch.claim_count} claims, depth ${batch.depth})`);
    });
  }

  // Show all claims in chronological order
  console.log('\n📋 Supply Chain History:');
  lineage.claims.forEach((claim, index) => {
    const event = claim.event;
    console.log(`${index + 1}. ${event.batch_id} - ${event.event_type}`);
    console.log(`   Product: ${event.product_id}`);
    console.log(`   Facility: ${event.facility_id}`);
    console.log(`   Quantity: ${event.quantity} ${event.unit}`);
    if (event.input_batches && event.input_batches.length > 0) {
      console.log(`   Inputs: ${event.input_batches.join(', ')}`);
    }
  });
}

lineageTracking().catch(console.error);
```

**Learning Goals**:
- Multi-stage supply chains
- Lineage graph traversal
- Upstream/downstream relationships
- Complete traceability

---

#### 4. Advanced Search & Filtering (`ts/examples/04-search-filtering.ts`)
**Scenario**: Query and filter claims with various criteria

```typescript
import { SupplyChainClient } from '@mycelix/supplychain-sdk';

async function advancedSearch() {
  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  // Example 1: Find all PRODUCED events
  console.log('📋 Finding all PRODUCED events...\n');
  const producedEvents = await client.searchClaims({
    event_type: 'PRODUCED',
    limit: 10,
  });
  console.log(`Found ${producedEvents.total} PRODUCED events`);
  console.log(`Showing ${producedEvents.claims.length} results`);
  console.log(`Has more: ${producedEvents.has_more}\n`);

  // Example 2: Find events for specific product
  console.log('🔍 Finding events for "Organic Coffee"...\n');
  const coffeeEvents = await client.searchClaims({
    product_id: 'Organic Coffee',
    limit: 20,
  });
  coffeeEvents.claims.forEach(claim => {
    console.log(`- ${claim.event.event_type}: ${claim.event.batch_id}`);
  });

  // Example 3: Find events in date range
  const startDate = new Date('2025-01-01');
  const endDate = new Date('2025-12-31');

  console.log('\n📅 Finding events in 2025...\n');
  const rangeEvents = await client.searchClaims({
    from: startDate.toISOString(),
    to: endDate.toISOString(),
    limit: 50,
  });
  console.log(`Found ${rangeEvents.total} events in date range`);

  // Example 4: Combine multiple filters
  console.log('\n🎯 Complex query: TRANSFORMED events at specific facility...\n');
  const complexQuery = await client.searchClaims({
    event_type: 'TRANSFORMED',
    facility_id: 'ORG-FACTORY',
    limit: 10,
  });
  console.log(`Found ${complexQuery.total} matching events`);

  // Example 5: Pagination through results
  console.log('\n📄 Paginating through all results...\n');
  let offset = 0;
  const limit = 20;
  let allClaims = [];

  while (true) {
    const page = await client.searchClaims({
      product_id: 'Organic Coffee',
      limit,
      offset,
    });

    allClaims.push(...page.claims);
    console.log(`Fetched page at offset ${offset}: ${page.claims.length} claims`);

    if (!page.has_more) break;
    offset += limit;
  }

  console.log(`\n✅ Total claims fetched: ${allClaims.length}`);
}

advancedSearch().catch(console.error);
```

**Learning Goals**:
- Filter by event type, product, facility
- Date range queries
- Pagination patterns
- Combining multiple filters

---

#### 5. Production Workflow (`ts/examples/05-production-workflow.ts`)
**Scenario**: Complete factory production workflow with error handling

```typescript
import { SupplyChainClient } from '@mycelix/supplychain-sdk';

interface ProductionConfig {
  batchId: string;
  facilityId: string;
  productId: string;
  inputBatches?: string[];
  quantity: number;
  unit: string;
  metadata: Record<string, any>;
}

async function productionWorkflow() {
  const client = new SupplyChainClient({
    baseURL: 'http://localhost:3000',
  });

  console.log('🏭 Starting production workflow...\n');

  try {
    // Step 1: Receive raw materials
    console.log('📥 Step 1: Receiving raw materials...');
    const receivedMaterials = [
      client.createReceivedEvent(
        'BATCH-RM-WHEAT-001',
        'FAC-MILL',
        'Organic Wheat',
        5000,
        'kg',
        { source: 'Local Farm Co-op', quality_grade: 'A' }
      ),
      client.createReceivedEvent(
        'BATCH-RM-SALT-001',
        'FAC-MILL',
        'Sea Salt',
        100,
        'kg',
        { source: 'Atlantic Salt Co.' }
      ),
    ];

    const receiveBatch = client.createBatch(receivedMaterials, 'atomic');
    const receiveResponse = await client.ingestBatch(receiveBatch);

    if (receiveResponse.failed > 0) {
      throw new Error('Failed to record received materials');
    }
    console.log(`✅ Received ${receiveResponse.succeeded} material batches\n`);

    // Step 2: Production/transformation
    console.log('⚙️  Step 2: Processing materials...');
    const processedGoods = client.createTransformedEvent(
      'BATCH-FG-FLOUR-001',
      'FAC-MILL',
      'Artisan Bread Flour',
      ['BATCH-RM-WHEAT-001', 'BATCH-RM-SALT-001'],
      4500,
      'kg',
      {
        process: 'stone-milled',
        additives: 'salt',
        yield_rate: '90%',
      }
    );

    const processResponse = await client.ingestEvent(processedGoods);
    console.log(`✅ Processed goods recorded: ${processResponse.claim_id}\n`);

    // Step 3: Quality certification
    console.log('🔬 Step 3: Quality certification...');
    const certified = client.createCertifiedEvent(
      'BATCH-FG-FLOUR-001',
      'FAC-MILL',
      'Artisan Bread Flour',
      4500,
      'kg',
      {
        certification_type: 'USDA Organic',
        certifier: 'Organic Certifiers Inc.',
        cert_number: 'ORG-2025-12345',
        expires: '2026-12-31',
      }
    );

    const certResponse = await client.ingestEvent(certified);
    console.log(`✅ Certification recorded: ${certResponse.claim_id}\n`);

    // Step 4: Ship to distribution
    console.log('🚚 Step 4: Shipping to distributor...');
    const shipped = client.createShippedEvent(
      'BATCH-FG-FLOUR-001',
      'FAC-MILL',
      'Artisan Bread Flour',
      4500,
      'kg',
      {
        destination: 'DIST-REGIONAL',
        carrier: 'GreenTransport LLC',
        tracking: 'GT-2025-98765',
        departure_date: new Date().toISOString(),
      }
    );

    const shipResponse = await client.ingestEvent(shipped);
    console.log(`✅ Shipment recorded: ${shipResponse.claim_id}\n`);

    // Step 5: Query complete workflow lineage
    console.log('🔍 Step 5: Verifying complete lineage...\n');
    const lineage = await client.getLineage('BATCH-FG-FLOUR-001');

    console.log('📊 Production Lineage Summary:');
    console.log(`Total events: ${lineage.total_claims}`);
    console.log(`Supply chain depth: ${lineage.depth}`);
    console.log(`Upstream sources: ${lineage.upstream?.length || 0}`);

    // Verify all events are present
    const eventTypes = lineage.claims.map(c => c.event.event_type);
    const requiredEvents = ['RECEIVED', 'TRANSFORMED', 'CERTIFIED', 'SHIPPED'];
    const hasAllEvents = requiredEvents.every(type => eventTypes.includes(type));

    if (hasAllEvents) {
      console.log('\n✅ Production workflow completed successfully!');
      console.log('All required events recorded and verified.');
    } else {
      console.warn('\n⚠️  Warning: Some events may be missing');
    }

    // Step 6: Get batch claims for audit
    console.log('\n📋 Step 6: Retrieving batch claims for audit...');
    const batchClaims = await client.getBatchClaims('BATCH-FG-FLOUR-001');

    console.log(`\nAudit Trail for ${batchClaims.batch_id}:`);
    batchClaims.claims.forEach((claim, index) => {
      const event = claim.event;
      const timestamp = new Date(event.timestamp).toLocaleString();
      console.log(`\n${index + 1}. ${event.event_type} - ${timestamp}`);
      console.log(`   Facility: ${event.facility_id}`);
      console.log(`   Product: ${event.product_id}`);
      console.log(`   Quantity: ${event.quantity} ${event.unit}`);
      console.log(`   Claim ID: ${claim.claim_id}`);
      console.log(`   Verified: ${claim.verified ? '✅' : '❌'}`);
    });

  } catch (error) {
    console.error('\n❌ Production workflow failed:', error);
    throw error;
  }
}

productionWorkflow().catch(console.error);
```

**Learning Goals**:
- Complete production workflow
- Multi-step process orchestration
- Error handling and recovery
- Audit trail generation
- Lineage verification

---

### Implementation Steps

1. **Create examples directory structure**
   ```bash
   mkdir -p ts/examples
   ```

2. **Create each example file**
   - `01-basic-ingestion.ts`
   - `02-batch-processing.ts`
   - `03-lineage-tracking.ts`
   - `04-search-filtering.ts`
   - `05-production-workflow.ts`

3. **Create package.json for examples**
   ```json
   {
     "name": "@mycelix/supplychain-examples",
     "version": "0.1.0",
     "private": true,
     "scripts": {
       "example:basic": "ts-node 01-basic-ingestion.ts",
       "example:batch": "ts-node 02-batch-processing.ts",
       "example:lineage": "ts-node 03-lineage-tracking.ts",
       "example:search": "ts-node 04-search-filtering.ts",
       "example:workflow": "ts-node 05-production-workflow.ts"
     },
     "dependencies": {
       "@mycelix/supplychain-sdk": "^0.2.0"
     },
     "devDependencies": {
       "ts-node": "^10.9.2",
       "typescript": "^5.3.3"
     }
   }
   ```

4. **Create README for examples**
   ```markdown
   # Mycelix Supply Chain SDK Examples

   This directory contains comprehensive examples showing how to use the
   Mycelix Supply Chain SDK in production scenarios.

   ## Prerequisites
   - Node.js 18+
   - Running Mycelix API server (http://localhost:3000)

   ## Installation
   ```bash
   npm install
   ```

   ## Running Examples
   ```bash
   npm run example:basic      # Basic event ingestion
   npm run example:batch      # Batch processing
   npm run example:lineage    # Lineage tracking
   npm run example:search     # Advanced search
   npm run example:workflow   # Production workflow
   ```
   ```

### Files to Create
- `ts/examples/01-basic-ingestion.ts`
- `ts/examples/02-batch-processing.ts`
- `ts/examples/03-lineage-tracking.ts`
- `ts/examples/04-search-filtering.ts`
- `ts/examples/05-production-workflow.ts`
- `ts/examples/package.json`
- `ts/examples/README.md`

### Success Criteria
- ✅ 5 comprehensive example files
- ✅ All examples executable and well-commented
- ✅ README with clear instructions
- ✅ package.json with run scripts

### Time Estimate
**45 minutes**

---

## Priority 3: Structured Logging with Request Tracing (45 min)

### Objective
Implement production-grade structured logging with request tracing for observability.

### Current State
- Basic logging with `println!` statements
- No request correlation
- No structured output
- No log levels

### Target State
- JSON structured logging
- Request ID propagation
- Log levels (debug, info, warn, error)
- Performance metrics logging
- Error context capture

### Dependencies to Add

**Cargo.toml**:
```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
uuid = { version = "1.6", features = ["v4", "serde"] }
```

### Implementation Steps

#### 1. Create logging module (`rust/service/src/logging.rs`)

```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

pub fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .json()
                .with_current_span(true)
                .with_span_list(true)
        )
        .init();

    tracing::info!("Structured logging initialized");
}
```

#### 2. Add request tracing middleware (`rust/service/src/middleware/tracing.rs`)

```rust
use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
};
use uuid::Uuid;
use tracing::{info_span, Instrument};

pub async fn trace_request(
    req: Request,
    next: Next,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let method = req.method().clone();
    let uri = req.uri().clone();

    let span = info_span!(
        "http_request",
        request_id = %request_id,
        method = %method,
        uri = %uri,
    );

    async move {
        tracing::info!("Request started");
        let start = std::time::Instant::now();

        let response = next.run(req).await;

        let duration = start.elapsed();
        tracing::info!(
            status = response.status().as_u16(),
            duration_ms = duration.as_millis() as u64,
            "Request completed"
        );

        response
    }
    .instrument(span)
    .await
}
```

#### 3. Update handlers with structured logging

**Batch ingestion** (`rust/service/src/handlers/batch.rs`):
```rust
#[tracing::instrument(skip(state, payload), fields(event_count = payload.events.len()))]
pub async fn ingest_batch(
    State(state): State<AppState>,
    Json(payload): Json<BatchIngestRequest>,
) -> Result<Json<BatchResponse>, StatusCode> {
    tracing::info!("Starting batch ingestion");
    let start = std::time::Instant::now();

    // ... processing logic ...

    let duration_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        succeeded = succeeded_count,
        failed = failed_count,
        duration_ms = duration_ms,
        "Batch ingestion completed"
    );

    Ok(Json(response))
}
```

**Lineage query** (`rust/service/src/handlers/lineage.rs`):
```rust
#[tracing::instrument(skip(state), fields(batch_id = %batch_id))]
pub async fn get_lineage(
    State(state): State<AppState>,
    Path(batch_id): Path<String>,
) -> Result<Json<LineageResponse>, StatusCode> {
    tracing::info!("Querying lineage graph");
    let start = std::time::Instant::now();

    // ... query logic ...

    let duration_ms = start.elapsed().as_millis() as u64;
    tracing::info!(
        total_claims = response.total_claims,
        depth = response.depth,
        duration_ms = duration_ms,
        "Lineage query completed"
    );

    Ok(Json(response))
}
```

#### 4. Add error logging

**Error handling** (`rust/service/src/error.rs`):
```rust
use tracing::error;

pub fn log_error<E: std::error::Error>(error: &E, context: &str) {
    error!(
        error = %error,
        context = context,
        "Operation failed"
    );
}
```

#### 5. Update main.rs to initialize logging

```rust
mod logging;

#[tokio::main]
async fn main() {
    // Initialize structured logging
    logging::init_logging();

    tracing::info!("Starting Mycelix Supply Chain API");

    // ... rest of main ...
}
```

### Log Output Examples

**Request trace**:
```json
{
  "timestamp": "2025-11-16T10:30:45.123Z",
  "level": "INFO",
  "message": "Request started",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "uri": "/v1/events/batch",
  "span": "http_request"
}
```

**Batch processing**:
```json
{
  "timestamp": "2025-11-16T10:30:45.456Z",
  "level": "INFO",
  "message": "Batch ingestion completed",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_count": 25,
  "succeeded": 25,
  "failed": 0,
  "duration_ms": 234,
  "span": "http_request::ingest_batch"
}
```

### Files to Create/Modify
- `rust/service/src/logging.rs` (NEW)
- `rust/service/src/middleware/tracing.rs` (NEW)
- `rust/service/src/middleware/mod.rs` (NEW)
- `rust/service/src/handlers/batch.rs` (MODIFY)
- `rust/service/src/handlers/lineage.rs` (MODIFY)
- `rust/service/src/handlers/search.rs` (MODIFY)
- `rust/service/src/main.rs` (MODIFY)
- `rust/service/Cargo.toml` (MODIFY)

### Success Criteria
- ✅ JSON structured logging throughout
- ✅ Request ID propagation
- ✅ Performance metrics captured
- ✅ Error context logging
- ✅ Configurable log levels (ENV)

### Time Estimate
**45 minutes**

---

## Priority 4: Performance Benchmarks & Optimization (30 min)

### Objective
Establish performance baselines and optimize critical paths.

### Benchmarks to Create

#### 1. Batch Ingestion Benchmark

**File**: `rust/service/benches/batch_ingestion.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use mycelix_service::*;

async fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_ingestion");

    for size in [1, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                b.iter(|| {
                    // Create batch of events
                    let events = create_test_events(size);
                    // Measure ingestion time
                    black_box(ingest_batch(events))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_batch_sizes);
criterion_main!(benches);
```

#### 2. Lineage Query Benchmark

**File**: `rust/service/benches/lineage_query.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_lineage_depths(c: &mut Criterion) {
    let mut group = c.benchmark_group("lineage_query");

    // Test different supply chain depths
    for depth in [1, 3, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(depth),
            depth,
            |b, &depth| {
                b.iter(|| {
                    // Create supply chain of given depth
                    let batch_id = create_supply_chain(depth);
                    // Measure query time
                    black_box(query_lineage(batch_id))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_lineage_depths);
criterion_main!(benches);
```

#### 3. Search Performance Benchmark

**File**: `rust/service/benches/search_performance.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_search_filters(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");

    // Populate database with test data
    setup_test_data(10000); // 10k claims

    group.bench_function("filter_by_product", |b| {
        b.iter(|| {
            black_box(search_claims(ClaimFilters {
                product_id: Some("Test Product".to_string()),
                ..Default::default()
            }))
        });
    });

    group.bench_function("filter_by_date_range", |b| {
        b.iter(|| {
            black_box(search_claims(ClaimFilters {
                from: Some("2025-01-01".to_string()),
                to: Some("2025-12-31".to_string()),
                ..Default::default()
            }))
        });
    });

    group.finish();
}

criterion_group!(benches, bench_search_filters);
criterion_main!(benches);
```

### Database Optimization

#### Add indexes for common queries

**Migration**: `rust/service/migrations/004_performance_indexes.sql`

```sql
-- Index for batch_id lookups (most common query)
CREATE INDEX IF NOT EXISTS idx_claims_batch_id ON claims(batch_id);

-- Index for product_id filtering
CREATE INDEX IF NOT EXISTS idx_claims_product_id ON claims(
    json_extract(event_json, '$.product_id')
);

-- Index for facility_id filtering
CREATE INDEX IF NOT EXISTS idx_claims_facility_id ON claims(
    json_extract(event_json, '$.facility_id')
);

-- Index for event_type filtering
CREATE INDEX IF NOT EXISTS idx_claims_event_type ON claims(
    json_extract(event_json, '$.event_type')
);

-- Index for timestamp range queries
CREATE INDEX IF NOT EXISTS idx_claims_timestamp ON claims(
    json_extract(event_json, '$.timestamp')
);

-- Composite index for common filter combinations
CREATE INDEX IF NOT EXISTS idx_claims_product_timestamp ON claims(
    json_extract(event_json, '$.product_id'),
    json_extract(event_json, '$.timestamp')
);

-- Analyze tables for query planner
ANALYZE claims;
```

### Performance Targets

**Batch Ingestion**:
- 1 event: < 10ms
- 10 events: < 50ms
- 50 events: < 150ms
- 100 events: < 300ms

**Lineage Queries**:
- Depth 1: < 10ms
- Depth 3: < 30ms
- Depth 5: < 50ms
- Depth 10: < 100ms

**Search Queries**:
- No filters: < 20ms (with pagination)
- Single filter: < 30ms
- Multiple filters: < 50ms
- Date range: < 40ms

### Files to Create
- `rust/service/benches/batch_ingestion.rs`
- `rust/service/benches/lineage_query.rs`
- `rust/service/benches/search_performance.rs`
- `rust/service/migrations/004_performance_indexes.sql`
- `rust/service/Cargo.toml` (add criterion dev-dependency)

### Success Criteria
- ✅ 3 comprehensive benchmarks
- ✅ Database indexes for all common queries
- ✅ Performance targets documented
- ✅ Baseline metrics established

### Time Estimate
**30 minutes**

---

## Priority 5: Rate Limiting & Caching (60 min)

### Objective
Protect API from abuse and improve performance with caching.

### Rate Limiting Implementation

#### Dependencies

**Cargo.toml**:
```toml
[dependencies]
governor = "0.6"
tower-governor = "0.3"
```

#### Rate limit middleware

**File**: `rust/service/src/middleware/rate_limit.rs`

```rust
use governor::{Quota, RateLimiter};
use std::num::NonZeroU32;
use std::sync::Arc;
use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
    http::StatusCode,
};

pub struct RateLimitConfig {
    pub requests_per_second: NonZeroU32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: NonZeroU32::new(100).unwrap(),
        }
    }
}

pub fn create_rate_limiter(config: RateLimitConfig) -> Arc<RateLimiter<String, _, _>> {
    let quota = Quota::per_second(config.requests_per_second);
    Arc::new(RateLimiter::keyed(quota))
}

pub async fn rate_limit_middleware(
    limiter: Arc<RateLimiter<String, _, _>>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract client IP or API key as rate limit key
    let key = extract_rate_limit_key(&req);

    match limiter.check_key(&key) {
        Ok(_) => Ok(next.run(req).await),
        Err(_) => {
            tracing::warn!(key = %key, "Rate limit exceeded");
            Err(StatusCode::TOO_MANY_REQUESTS)
        }
    }
}

fn extract_rate_limit_key(req: &Request) -> String {
    // Try API key first, fall back to IP
    req.headers()
        .get("X-API-Key")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            // Extract IP from connection info or X-Forwarded-For
            "default_key".to_string()
        })
}
```

#### Configure rate limits per endpoint

**main.rs**:
```rust
use tower_governor::{GovernorConfigBuilder, GovernorLayer};

let governor_conf = Box::new(
    GovernorConfigBuilder::default()
        .per_second(100)  // Global: 100 req/s
        .burst_size(20)   // Allow bursts
        .finish()
        .unwrap()
);

let app = Router::new()
    // High-traffic endpoints: generous limits
    .route("/v1/events", post(ingest_event))
    .layer(governor_conf.clone())

    // Expensive endpoints: stricter limits
    .route("/v1/lineage/:batch_id", get(get_lineage))
    .layer(rate_limit_layer(10)) // 10 req/s for lineage

    // Search: moderate limits
    .route("/v1/claims", get(search_claims))
    .layer(rate_limit_layer(50)); // 50 req/s for search
```

### Caching Layer

#### Dependencies

**Cargo.toml**:
```toml
[dependencies]
moka = { version = "0.12", features = ["future"] }
```

#### Cache implementation

**File**: `rust/service/src/cache.rs`

```rust
use moka::future::Cache;
use std::time::Duration;
use serde::{Serialize, Deserialize};

#[derive(Clone)]
pub struct ResponseCache {
    lineage: Cache<String, LineageResponse>,
    batch_claims: Cache<String, BatchClaimsResponse>,
    search: Cache<String, SearchResponse>,
}

impl ResponseCache {
    pub fn new() -> Self {
        Self {
            // Lineage cache: 1000 entries, 5 min TTL
            lineage: Cache::builder()
                .max_capacity(1000)
                .time_to_live(Duration::from_secs(300))
                .build(),

            // Batch claims cache: 2000 entries, 10 min TTL
            batch_claims: Cache::builder()
                .max_capacity(2000)
                .time_to_live(Duration::from_secs(600))
                .build(),

            // Search cache: 500 entries, 1 min TTL
            search: Cache::builder()
                .max_capacity(500)
                .time_to_live(Duration::from_secs(60))
                .build(),
        }
    }

    pub async fn get_lineage(&self, batch_id: &str) -> Option<LineageResponse> {
        self.lineage.get(batch_id).await
    }

    pub async fn set_lineage(&self, batch_id: String, response: LineageResponse) {
        self.lineage.insert(batch_id, response).await;
    }

    pub async fn invalidate_batch(&self, batch_id: &str) {
        self.lineage.invalidate(batch_id).await;
        self.batch_claims.invalidate(batch_id).await;
    }
}
```

#### Cache-aware handlers

**lineage.rs**:
```rust
#[tracing::instrument(skip(state, cache))]
pub async fn get_lineage(
    State(state): State<AppState>,
    State(cache): State<ResponseCache>,
    Path(batch_id): Path<String>,
) -> Result<Json<LineageResponse>, StatusCode> {
    // Check cache first
    if let Some(cached) = cache.get_lineage(&batch_id).await {
        tracing::debug!("Lineage cache hit");
        return Ok(Json(cached));
    }

    tracing::debug!("Lineage cache miss, querying database");
    let response = query_lineage_from_db(&state, &batch_id).await?;

    // Store in cache
    cache.set_lineage(batch_id, response.clone()).await;

    Ok(Json(response))
}
```

#### Cache invalidation strategy

```rust
// Invalidate cache when new events ingested
pub async fn ingest_event(
    State(state): State<AppState>,
    State(cache): State<ResponseCache>,
    Json(event): Json<SupplyEventVC>,
) -> Result<Json<EventResponse>, StatusCode> {
    let response = process_event(&state, event).await?;

    // Invalidate affected caches
    cache.invalidate_batch(&event.batch_id).await;

    // If transformed event, invalidate input batches too
    if let Some(inputs) = event.input_batches {
        for input in inputs {
            cache.invalidate_batch(&input).await;
        }
    }

    Ok(Json(response))
}
```

### Cache Headers

Add proper HTTP cache headers:

```rust
use axum::http::header;

// For cacheable responses
let mut headers = HeaderMap::new();
headers.insert(
    header::CACHE_CONTROL,
    "public, max-age=300".parse().unwrap()
);
headers.insert(
    header::ETAG,
    compute_etag(&response).parse().unwrap()
);
```

### Files to Create/Modify
- `rust/service/src/middleware/rate_limit.rs` (NEW)
- `rust/service/src/cache.rs` (NEW)
- `rust/service/src/handlers/lineage.rs` (MODIFY - add caching)
- `rust/service/src/handlers/batch.rs` (MODIFY - cache invalidation)
- `rust/service/src/main.rs` (MODIFY - add middleware)
- `rust/service/Cargo.toml` (MODIFY - add dependencies)

### Success Criteria
- ✅ Rate limiting on all endpoints
- ✅ Configurable rate limits per endpoint
- ✅ Response caching for expensive queries
- ✅ Smart cache invalidation
- ✅ HTTP cache headers

### Time Estimate
**60 minutes**

---

## Priority 6: Security Hardening (45 min)

### Objective
Harden API security with input validation, security headers, and CORS.

### Security Headers

**File**: `rust/service/src/middleware/security.rs`

```rust
use axum::{
    http::{HeaderMap, HeaderValue, header},
    middleware::Next,
    response::Response,
    extract::Request,
};

pub async fn security_headers(
    req: Request,
    next: Next,
) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();

    // Prevent XSS
    headers.insert(
        header::HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff")
    );

    // Prevent clickjacking
    headers.insert(
        header::HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY")
    );

    // XSS protection
    headers.insert(
        header::HeaderName::from_static("x-xss-protection"),
        HeaderValue::from_static("1; mode=block")
    );

    // Content Security Policy
    headers.insert(
        header::HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static("default-src 'self'")
    );

    // Strict Transport Security (HTTPS only)
    headers.insert(
        header::HeaderName::from_static("strict-transport-security"),
        HeaderValue::from_static("max-age=31536000; includeSubDomains")
    );

    response
}
```

### CORS Configuration

```rust
use tower_http::cors::{CorsLayer, Any};

let cors = CorsLayer::new()
    .allow_origin(Any) // Configure based on environment
    .allow_methods([Method::GET, Method::POST])
    .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
    .max_age(Duration::from_secs(3600));

let app = Router::new()
    .route(...)
    .layer(cors);
```

### Input Validation

**File**: `rust/service/src/validation.rs`

```rust
use validator::Validate;

#[derive(Validate)]
pub struct SupplyEventVC {
    #[validate(length(min = 1, max = 100))]
    pub batch_id: String,

    #[validate(length(min = 1, max = 100))]
    pub facility_id: String,

    #[validate(length(min = 1, max = 200))]
    pub product_id: String,

    #[validate(range(min = 0.0))]
    pub quantity: f64,

    #[validate(length(min = 1, max = 20))]
    pub unit: String,
}

pub fn validate_event(event: &SupplyEventVC) -> Result<(), ValidationError> {
    event.validate()?;

    // Additional business logic validation
    if event.quantity <= 0.0 {
        return Err(ValidationError::new("quantity must be positive"));
    }

    // Validate batch_id format
    if !is_valid_batch_id(&event.batch_id) {
        return Err(ValidationError::new("invalid batch_id format"));
    }

    Ok(())
}
```

### SQL Injection Prevention

Already using parameterized queries with sqlx, but add validation:

```rust
// Sanitize user inputs for JSON extraction
pub fn sanitize_json_path(input: &str) -> Result<String, ValidationError> {
    if input.contains(['\'', '"', ';', '-', '/', '\\']) {
        return Err(ValidationError::new("invalid characters in input"));
    }
    Ok(input.to_string())
}
```

### Request Size Limits

```rust
use tower_http::limit::RequestBodyLimitLayer;

let app = Router::new()
    .route(...)
    .layer(RequestBodyLimitLayer::new(
        1024 * 1024 // 1MB max request size
    ));
```

### API Key Authentication (Optional)

```rust
pub async fn auth_middleware(
    headers: HeaderMap,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let api_key = headers
        .get("X-API-Key")
        .and_then(|h| h.to_str().ok());

    match api_key {
        Some(key) if is_valid_api_key(key) => Ok(next.run(req).await),
        _ => {
            tracing::warn!("Unauthorized API access attempt");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

fn is_valid_api_key(key: &str) -> bool {
    // Check against configured API keys
    std::env::var("API_KEYS")
        .ok()
        .map(|keys| keys.split(',').any(|k| k == key))
        .unwrap_or(false)
}
```

### Files to Create/Modify
- `rust/service/src/middleware/security.rs` (NEW)
- `rust/service/src/validation.rs` (NEW)
- `rust/service/src/types.rs` (MODIFY - add Validate derives)
- `rust/service/src/main.rs` (MODIFY - add security middleware)
- `rust/service/Cargo.toml` (MODIFY - add validator, tower-http)

### Success Criteria
- ✅ Security headers on all responses
- ✅ CORS properly configured
- ✅ Input validation on all endpoints
- ✅ Request size limits
- ✅ Optional API key authentication

### Time Estimate
**45 minutes**

---

## Priority 7: Metrics & Observability (30 min - OPTIONAL)

### Objective
Export Prometheus metrics for monitoring and alerting.

### Dependencies

```toml
[dependencies]
metrics = "0.21"
metrics-exporter-prometheus = "0.12"
```

### Implementation

**File**: `rust/service/src/metrics.rs`

```rust
use metrics::{counter, histogram, gauge};

pub fn init_metrics() {
    let builder = PrometheusBuilder::new();
    builder.install().expect("Failed to install Prometheus exporter");
}

// Request metrics
pub fn record_request(method: &str, path: &str, status: u16, duration_ms: u64) {
    counter!("http_requests_total", "method" => method, "path" => path, "status" => status.to_string()).increment(1);
    histogram!("http_request_duration_ms", "method" => method, "path" => path).record(duration_ms as f64);
}

// Batch metrics
pub fn record_batch_ingestion(size: usize, succeeded: usize, failed: usize, duration_ms: u64) {
    counter!("batch_events_total").increment(size as u64);
    counter!("batch_events_succeeded").increment(succeeded as u64);
    counter!("batch_events_failed").increment(failed as u64);
    histogram!("batch_duration_ms").record(duration_ms as f64);
}

// Database metrics
pub fn record_query(query_type: &str, duration_ms: u64) {
    counter!("db_queries_total", "type" => query_type).increment(1);
    histogram!("db_query_duration_ms", "type" => query_type).record(duration_ms as f64);
}

// Cache metrics
pub fn record_cache_hit(cache_type: &str) {
    counter!("cache_hits_total", "type" => cache_type).increment(1);
}

pub fn record_cache_miss(cache_type: &str) {
    counter!("cache_misses_total", "type" => cache_type).increment(1);
}
```

### Metrics Endpoint

```rust
use axum::routing::get;
use metrics_exporter_prometheus::PrometheusHandle;

pub async fn metrics_handler(
    State(handle): State<PrometheusHandle>,
) -> String {
    handle.render()
}

// In main.rs
let app = Router::new()
    .route("/metrics", get(metrics_handler))
    // ... other routes
```

---

## Implementation Order

### Phase 1: Foundation (1 hour)
1. Fix 4 failing integration tests (15 min)
2. Create TypeScript example files (45 min)

### Phase 2: Observability (1.5 hours)
3. Implement structured logging (45 min)
4. Add performance benchmarks (30 min)
5. Create database indexes (15 min)

### Phase 3: Production Features (1.5 hours)
6. Add rate limiting (30 min)
7. Implement caching (30 min)
8. Security hardening (30 min)

### Phase 4: Documentation (30 min)
9. Create Phase 9 summary
10. Update README with new features
11. Commit and push

---

## Expected Outcomes

### Developer Experience
- **5 comprehensive SDK examples** covering all major use cases
- **Clear documentation** for getting started
- **Type-safe workflows** for production applications

### Production Readiness
- **100% test pass rate** - all integration tests green
- **Structured logging** - JSON logs with request tracing
- **Performance metrics** - benchmarks and optimization
- **Rate limiting** - protection from abuse
- **Response caching** - improved performance for expensive queries
- **Security hardening** - headers, validation, CORS

### Observability
- **Request tracing** with unique IDs
- **Performance monitoring** with structured logs
- **Error tracking** with context
- **Optional Prometheus metrics** for dashboards

---

## Success Metrics

### Testing
- Test pass rate: 73% → 100%
- All integration tests green
- No skipped or ignored tests

### Performance
- Batch ingestion (100 events): < 300ms
- Lineage query (depth 5): < 50ms
- Search with filters: < 50ms
- Cache hit rate: > 80% for lineage queries

### Security
- All endpoints rate-limited
- Security headers on all responses
- Input validation 100% coverage
- No SQL injection vulnerabilities

### Developer Experience
- 5 comprehensive examples
- README with quick-start guide
- Clear API documentation
- Easy local setup

---

## Risks & Mitigations

### Risk: Performance regression from logging
**Mitigation**: Use async logging, configurable log levels, sample high-volume logs

### Risk: Cache invalidation complexity
**Mitigation**: Simple TTL-based caching, conservative invalidation strategy

### Risk: Rate limiting false positives
**Mitigation**: Generous default limits, configurable per-endpoint, proper error messages

---

## Phase 9 Completion Checklist

- [ ] All 15 integration tests passing
- [ ] 5 TypeScript SDK examples created
- [ ] Structured logging implemented
- [ ] Request tracing functional
- [ ] Performance benchmarks created
- [ ] Database indexes added
- [ ] Rate limiting on all endpoints
- [ ] Response caching implemented
- [ ] Security headers configured
- [ ] Input validation hardened
- [ ] CORS properly configured
- [ ] Phase 9 summary documented
- [ ] All changes committed and pushed

---

**Phase 9 Status**: 🚧 **IN PROGRESS**
**Target Completion**: 2025-11-16
**Next Phase**: Phase 10 (Deployment & Operations)
