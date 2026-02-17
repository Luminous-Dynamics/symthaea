# Phase 2: Advanced Production Readiness

Building on the solid foundation from Phase 1, this plan takes the repository to the next level with integration tests, examples, CLI tools, and persistence.

## Goals

- **Integration Testing**: Full API test coverage with real HTTP calls
- **Example Workflows**: Complete, runnable examples for common scenarios
- **CLI Tool**: Command-line interface for operations
- **Persistence**: SQLite database layer replacing in-memory storage
- **CI/CD**: Docker builds, automated testing, releases
- **Observability**: Basic metrics and structured logging

## Phase 2 Improvements (This Session)

### 1. Integration Tests ⚡ Priority 1

**Goal**: Validate REST API with real HTTP requests

**Tests to Add**:
- ✅ Service startup and health check
- ✅ POST event → verify claim created
- ✅ GET claim by ID
- ✅ Event validation (reject invalid VCs)
- ✅ Lineage tracking (TRANSFORMED event links parents)
- ✅ Concurrent event ingestion
- ✅ Error handling (404, 400, 500)

**Implementation**:
- Create `rust/service/tests/integration_test.rs`
- Use `reqwest` for HTTP client
- Start service in test setup
- Clean shutdown in teardown

**Target**: 10+ integration tests, 100% API coverage

### 2. Example Workflows ⚡ Priority 1

**Goal**: Show real-world usage patterns

**Examples to Create**:
1. **Simple Flow** (`examples/01-simple-flow/`)
   - Single PRODUCED event
   - Retrieve and verify

2. **Full Supply Chain** (`examples/02-full-supply-chain/`)
   - PRODUCED → TRANSFORMED → SHIPPED → RECEIVED → CERTIFIED
   - Complete lineage visualization

3. **Bulk Import** (`examples/03-bulk-import/`)
   - CSV with 100+ events
   - Performance measurement

4. **Multi-Product** (`examples/04-multi-product/`)
   - Multiple products, batches, facilities
   - Complex lineage graph

5. **TypeScript Integration** (`examples/05-typescript-client/`)
   - Using SDK to build an app
   - React dashboard example

**Target**: 5 complete, documented examples

### 3. TypeScript SDK Enhancements ⚡ Priority 2

**Improvements**:
- ✅ Add Jest tests for client
- ✅ Add mock server for testing
- ✅ Add example usage scripts
- ✅ Add retry logic
- ✅ Add better error messages
- ✅ Generate API docs (TypeDoc)

**Target**: 80%+ test coverage, great DX

### 4. CLI Tool ⚡ Priority 2

**Goal**: Command-line interface for common operations

**Commands**:
```bash
# Generate keypair
supplychain keygen --output keypair.json

# Post event from file
supplychain ingest --file event.json

# Post event from JSON
supplychain ingest --json '{"@context":...}'

# Get claim
supplychain get-claim <claim-id>

# Get batch lineage
supplychain lineage <batch-id>

# Verify VC
supplychain verify --jwt <vc-jwt>

# Bulk import CSV
supplychain import-csv --file data.csv

# Export lineage to DOT/JSON
supplychain export-lineage <batch-id> --format dot
```

**Implementation**:
- New binary in `rust/cli/`
- Use `clap` for arg parsing
- Colored output with `colored`
- Progress bars with `indicatif`

**Target**: 8+ commands, beautiful UX

### 5. SQLite Persistence ⚡ Priority 1

**Goal**: Replace in-memory HashMap with SQLite

**Schema**:
```sql
CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    issuer TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    vc_jwt TEXT NOT NULL,
    lineage_hash TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    claim_json TEXT NOT NULL
);

CREATE TABLE lineage (
    claim_id TEXT NOT NULL,
    parent_claim_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, parent_claim_id)
);

CREATE INDEX idx_batch_id ON claims(batch_id);
CREATE INDEX idx_product_id ON claims(product_id);
```

**Implementation**:
- Use `sqlx` with compile-time query checking
- Migrations in `rust/service/migrations/`
- Connection pooling
- Graceful fallback to in-memory for tests

**Target**: Production-ready persistence

### 6. GitHub Actions Enhancements ⚡ Priority 2

**Workflows to Add**:
1. **Docker Build** (`.github/workflows/docker.yml`)
   - Build images on push to main
   - Tag with version + latest
   - Push to GitHub Container Registry

2. **Release** (`.github/workflows/release.yml`)
   - Triggered on version tags
   - Build binaries for Linux/Mac/Windows
   - Create GitHub release with artifacts

3. **Integration Tests** (add to existing workflows)
   - Run integration tests on PR
   - Test with both SQLite and in-memory

**Target**: Automated builds and releases

### 7. Documentation Improvements ⚡ Priority 2

**New Docs**:
1. **API Guide** (`docs/api-guide.md`)
   - Endpoint reference
   - Request/response examples
   - Error codes
   - Rate limiting

2. **Deployment Guide** (`docs/deployment.md`)
   - Development setup
   - Docker Compose
   - Kubernetes
   - Production checklist

3. **Integration Patterns** (`docs/integration-patterns.md`)
   - ERP integration
   - IoT integration
   - Blockchain bridges
   - Webhook handlers

**Target**: Complete documentation suite

### 8. Observability Basics ⚡ Priority 3

**Additions**:
- Prometheus metrics endpoint (`/metrics`)
- Structured JSON logging
- Request tracing with correlation IDs
- Health check with deep checks (DB connection)

**Metrics to Track**:
- `supplychain_events_total` (counter by event_type)
- `supplychain_request_duration_seconds` (histogram)
- `supplychain_claims_total` (gauge)
- `supplychain_db_connections` (gauge)

**Target**: Production observability

### 9. Additional Enhancements ⚡ Priority 3

**Nice-to-Haves**:
- ✅ `.editorconfig` for consistent formatting
- ✅ `.vscode/` debug configurations
- ✅ `docker-compose.dev.yml` with hot-reload
- ✅ Pre-commit hooks configuration
- ✅ Benchmark suite
- ✅ Load testing scripts (K6)

## Success Metrics - Phase 2

| Metric | Target | Stretch |
|--------|--------|---------|
| Integration Tests | 10+ | 20+ |
| API Coverage | 100% | 100% + edge cases |
| Examples | 5 | 7+ |
| CLI Commands | 8 | 12+ |
| SQLite Migration | ✅ Working | + Postgres support |
| Docker Build Time | <5 min | <3 min |
| Documentation Pages | 3 | 5+ |

## Timeline Estimate

- **Integration Tests**: 30 min
- **Examples**: 45 min
- **TypeScript SDK**: 30 min
- **CLI Tool**: 45 min
- **SQLite Layer**: 60 min
- **GitHub Actions**: 20 min
- **Documentation**: 30 min
- **Observability**: 20 min

**Total**: ~4-5 hours of focused work

## Execution Order

1. ✅ Integration tests (validates current code)
2. ✅ Examples (shows how to use)
3. ✅ TypeScript SDK (better DX)
4. ✅ CLI tool (operational tooling)
5. ✅ SQLite layer (persistence)
6. ✅ GitHub Actions (automation)
7. ✅ Documentation (completeness)
8. ✅ Observability (production readiness)

## What This Unlocks

After Phase 2:
- **Production Deployments**: SQLite persistence + Docker builds
- **Easy Integration**: Complete examples + CLI tool
- **Confidence**: Integration tests validate everything works
- **Observability**: Metrics and logging for monitoring
- **Automation**: CI/CD pipeline for releases

Let's execute! 🚀
