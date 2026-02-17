# Mycelix Supply Chain - Architecture

## System Overview

The Mycelix Supply Chain system provides verifiable, end-to-end provenance tracking for supply chain events using **Verifiable Credentials (VCs)** and **Distributed Knowledge Graph (DKG)** integration.

### Core Principles

1. **Cryptographic Verifiability**: All events are signed with Ed25519
2. **Tamper-Evident Lineage**: Hash-linked claims form an immutable trail
3. **Selective Disclosure**: Future SD-JWT/BBS+ support for privacy
4. **Distributed Truth**: Claims published to DKG for Byzantine fault tolerance

## Architecture Diagram

```
┌────────────────┐
│  Data Sources  │
│  (ERP/IoT/CSV) │
└────────┬───────┘
         │
         ▼
┌────────────────────────────────────────┐
│          Adapters Layer                │
│  ┌──────┐  ┌──────┐  ┌──────┐         │
│  │ CSV  │  │ MQTT │  │ REST │         │
│  └──────┘  └──────┘  └──────┘         │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│      Provenance Service (Rust)         │
│  ┌──────────────────────────────────┐  │
│  │   REST API (Axum)                │  │
│  │   - POST /v1/events              │  │
│  │   - GET  /v1/claims/:id          │  │
│  │   - POST /v1/verify              │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│  ┌──────────────▼───────────────────┐  │
│  │   Event Processing Pipeline      │  │
│  │   1. Validate VC structure       │  │
│  │   2. Sign with Ed25519          │  │
│  │   3. Resolve lineage            │  │
│  │   4. Create DKG claim           │  │
│  │   5. Store claim                │  │
│  └──────────────┬───────────────────┘  │
│                 │                       │
│  ┌──────────────▼───────────────────┐  │
│  │   Storage Layer                  │  │
│  │   - In-memory (dev)              │  │
│  │   - SQLite (single node)         │  │
│  │   - PostgreSQL (production)      │  │
│  └──────────────────────────────────┘  │
└────────┬───────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│    DKG Network (Future Integration)    │
│    - Byzantine consensus               │
│    - Distributed claim storage         │
│    - Epistemic reasoning               │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│       Consumers                        │
│   - Dashboard (Next.js)                │
│   - SDK Clients                        │
│   - Verifiers                          │
│   - Product Passports                  │
└────────────────────────────────────────┘
```

## Component Architecture

### 1. Rust Service Components

#### claim-model
**Purpose**: Core data types and validation

- `SupplyEventVC`: W3C Verifiable Credential structure
- `DkgClaim`: DKG-ready epistemic claim
- `EventType`: PRODUCED, TRANSFORMED, SHIPPED, RECEIVED, CERTIFIED
- Validation logic for all event types
- Lineage hash computation

#### crypto
**Purpose**: Cryptographic operations

- Ed25519 keypair management
- JWT signing (VC-JWT format)
- SHA-256 hashing
- Base64URL encoding
- Future: SD-JWT, BBS+ signatures

#### service
**Purpose**: REST API and business logic

**Modules**:
- `api.rs`: HTTP handlers
- `pipeline.rs`: Event processing workflow
- `lineage.rs`: Lineage resolution and graph building
- `vc.rs`: VC signing and verification
- `dkg_client.rs`: DKG integration (placeholder)

### 2. TypeScript Components

#### SDK (`ts/sdk`)
**Purpose**: Client library for API integration

```typescript
const client = new SupplyChainClient({ baseUrl: 'http://localhost:8080' });
const result = await client.ingestEvent(event);
```

**Features**:
- Type-safe API client
- Helper methods for event creation
- Lineage querying
- VC verification

#### Dashboard (`ts/dashboard`)
**Purpose**: Web UI for visualization

- Next.js 14+ app router
- Claim browsing
- Lineage visualization
- VC verification interface

#### Adapters (`ts/adapters`)
**Purpose**: Data source integrations

- **CSV**: Bulk ingestion from files
- **MQTT**: IoT sensor streams
- Future: SAP, Oracle, Dynamics connectors

## Data Flow

### Event Ingestion Flow

```
1. External System
   ↓ (POST JSON)
2. REST API Endpoint
   ↓ (deserialize)
3. VC Validation
   ↓ (schema + business rules)
4. VC Signing
   ↓ (Ed25519 + JWT)
5. Lineage Resolution
   ↓ (query prev claims)
6. DKG Claim Creation
   ↓ (project VC → claim)
7. Storage
   ↓
8. Response (claim_id, vc_jwt, lineage_hash)
```

### Lineage Resolution

**For PRODUCED events**:
- No parents (root of lineage tree)

**For TRANSFORMED events**:
- Find claims for each `prevBatchIds`
- Link as parent claims
- Compute combined lineage hash

**For SHIPPED/RECEIVED events**:
- Find latest claim for same `batchId`
- Link as single parent

**For CERTIFIED events**:
- Find all claims for same `batchId` or `productId`
- Link as related claims

### Lineage Hash Computation

```
hash = SHA256(vc_jwt || parent_claim_id_1 || parent_claim_id_2 || ...)
```

This creates a **Merkle-like structure** where:
- Each claim cryptographically commits to its parents
- Tampering any claim invalidates all descendants
- Full lineage can be verified independently

## Security Architecture

### Threat Model

**Assets**:
- Supply chain event data
- Cryptographic signing keys
- Lineage integrity

**Threats**:
- Event forgery
- Lineage tampering
- Replay attacks
- Key compromise

### Mitigations

1. **Event Forgery**
   - All VCs signed with Ed25519
   - DIDs tie to org identity
   - Signature verification before acceptance

2. **Lineage Tampering**
   - Hash chaining (Merkle structure)
   - Timestamps in claims
   - Future: DKG consensus

3. **Replay Attacks**
   - Unique batch IDs
   - Timestamps
   - Future: nonces/sequence numbers

4. **Key Compromise**
   - Recommend HSM/KMS in production
   - Key rotation procedures (future)
   - DID revocation (future)

### Access Control (Future)

- API authentication (JWT bearer tokens)
- Role-based access (RBAC)
- Policy-based lineage queries (who can see what)

## Storage Architecture

### Current: In-Memory HashMap

```rust
HashMap<claim_id, DkgClaim>
```

**Pros**: Fast, simple
**Cons**: Not persistent, single-node only

### Phase 2: SQLite

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
    PRIMARY KEY (claim_id, parent_claim_id),
    FOREIGN KEY (claim_id) REFERENCES claims(id),
    FOREIGN KEY (parent_claim_id) REFERENCES claims(id)
);

CREATE INDEX idx_batch_id ON claims(batch_id);
CREATE INDEX idx_product_id ON claims(product_id);
CREATE INDEX idx_timestamp ON claims(timestamp);
```

### Phase 3: PostgreSQL

- Same schema as SQLite
- Connection pooling (sqlx::Pool)
- Read replicas for scaling
- JSONB columns for flexible metadata

### Future: Graph Database (Neo4j)

For advanced lineage queries:
- Multi-hop provenance
- Impact analysis
- Cycle detection
- Community detection

## API Design

### RESTful Principles

- **Resources**: events, claims, lineage
- **Methods**: POST (create), GET (read)
- **Stateless**: no session state
- **Cacheable**: claims are immutable once created

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| POST | `/v1/events` | Ingest event |
| GET | `/v1/claims/:id` | Get claim + lineage |
| GET | `/v1/batches/:id/lineage` | Get batch lineage |
| POST | `/v1/verify` | Verify VC signature |

### Error Handling

**HTTP Status Codes**:
- `200 OK`: Success
- `201 Created`: Event ingested
- `400 Bad Request`: Validation error
- `404 Not Found`: Claim not found
- `500 Internal Server Error`: Server fault

**Error Response**:
```json
{
  "error": "validation_error",
  "message": "quantity must be positive"
}
```

## Performance Considerations

### Current Bottlenecks

1. **Synchronous processing**: Single-threaded pipeline
2. **In-memory storage**: Limited capacity
3. **No caching**: Repeated lineage queries expensive

### Optimization Strategies

1. **Async processing**: Tokio async runtime (already used)
2. **Batch ingestion**: Accept arrays of events
3. **Caching**: Redis for frequently-accessed claims
4. **Read replicas**: PostgreSQL replication
5. **Indexing**: Batch ID, product ID, timestamp indexes

### Scalability Targets

- **Current**: ~100 events/sec, single node
- **Phase 2 (SQLite)**: ~500 events/sec
- **Phase 3 (Postgres)**: ~2000 events/sec
- **Future (distributed)**: 10k+ events/sec

## Deployment Architecture

### Development

```
Docker Compose:
- provenance-service
- dashboard (optional)
```

### Production

```
Kubernetes:
- Deployment (3+ replicas)
- Service (ClusterIP)
- Ingress (TLS)
- PVC (SQLite) or external PostgreSQL
```

### High Availability

- **Multi-replica deployment**: 3+ pods
- **Load balancing**: Kubernetes Service
- **Database**: PostgreSQL with replication
- **Monitoring**: Prometheus + Grafana
- **Logging**: Centralized (Loki/ELK)

## Integration Points

### Upstream (Data Sources)

- **ERP systems**: CSV export → adapter
- **IoT devices**: MQTT → adapter
- **Webhooks**: HTTP POST → REST API
- **Blockchain**: Event listener → adapter

### Downstream (Consumers)

- **Dashboards**: REST API
- **Mobile apps**: SDK
- **Smart contracts**: DKG → blockchain bridge
- **Auditors**: Verification API

## Future Enhancements

### Short Term (1-3 months)

- SQLite persistence
- Authentication/authorization
- Rate limiting
- Prometheus metrics

### Medium Term (3-6 months)

- DKG integration
- SD-JWT selective disclosure
- Product passport generation
- GraphQL API

### Long Term (6-12 months)

- Multi-party computation (MPC)
- Zero-knowledge proofs (ZKP)
- Blockchain anchoring
- GS1 EPCIS compliance

## References

- **W3C Verifiable Credentials**: https://www.w3.org/TR/vc-data-model/
- **DID Spec**: https://www.w3.org/TR/did-core/
- **Ed25519**: https://ed25519.cr.yp.to/
- **JWT**: https://datatracker.ietf.org/doc/html/rfc7519
- **SD-JWT**: https://datatracker.ietf.org/doc/draft-ietf-oauth-selective-disclosure-jwt/
