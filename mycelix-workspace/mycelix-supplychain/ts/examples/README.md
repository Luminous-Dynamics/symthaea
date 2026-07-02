# Mycelix Supply Chain SDK - Examples

This directory contains comprehensive, production-ready examples demonstrating how to use the Mycelix Supply Chain SDK for various supply chain provenance tracking scenarios.

## Prerequisites

- **Node.js** 18.0.0 or higher
- **TypeScript** 5.x
- **Running API Server**: The Mycelix Supply Chain API must be running at `http://localhost:3000`

## Installation

```bash
# Install dependencies
npm install

# Build the SDK (from parent directory)
cd ../sdk
npm install
npm run build
cd ../examples
```

## Starting the API Server

Before running the examples, start the Mycelix API server:

```bash
# From the rust/service directory
cd ../../rust/service
cargo run --release

# Or in development mode
cargo run
```

The server will start on `http://localhost:3000`

## Running Examples

### Individual Examples

```bash
# Example 1: Basic event ingestion
npm run example:basic

# Example 2: Batch processing with error handling
npm run example:batch

# Example 3: Supply chain lineage tracking
npm run example:lineage

# Example 4: Advanced search and filtering
npm run example:search

# Example 5: Complete production workflow
npm run example:workflow
```

### Run All Examples

```bash
npm run example:all
```

## Examples Overview

### 1. Basic Event Ingestion (`01-basic-ingestion.ts`)

**What you'll learn:**
- SDK initialization and configuration
- Creating PRODUCED events
- Submitting single events
- Handling API responses
- Working with claim IDs and VCs

**Use case:** Simple event tracking for individual production batches.

**Concepts:**
- Event creation helpers
- Synchronous event submission
- Response structure

---

### 2. Batch Processing (`02-batch-processing.ts`)

**What you'll learn:**
- High-volume event ingestion (up to 100 events)
- Best-effort vs atomic processing modes
- Error handling and recovery strategies
- Batch result processing
- Retry logic for failed events

**Use case:** Production runs with multiple events that need to be tracked together.

**Concepts:**
- Batch creation
- Processing modes (best-effort allows partial success, atomic requires all or nothing)
- Individual result inspection
- Retry patterns

**Key features demonstrated:**
- Submit 4 events in a single batch
- Process results individually
- Retry failed events
- Use atomic mode for critical operations

---

### 3. Supply Chain Lineage Tracking (`03-lineage-tracking.ts`)

**What you'll learn:**
- Creating multi-stage supply chains
- Tracking transformation across multiple steps
- Querying complete lineage graphs
- Understanding upstream/downstream relationships
- Verifying end-to-end traceability

**Use case:** Complete supply chain from raw materials to finished product.

**Concepts:**
- Multi-stage transformations
- Lineage graph traversal
- Upstream sources tracking
- Downstream derivatives tracking
- Supply chain visualization

**Flow demonstrated:**
```
Raw Cacao Beans (PRODUCED)
  ↓
Dark Chocolate (TRANSFORMED)
  ↓
Retail Packages (TRANSFORMED)
  ↓
Certified (CERTIFIED)
  ↓
Shipped (SHIPPED)
```

---

### 4. Advanced Search & Filtering (`04-search-filtering.ts`)

**What you'll learn:**
- Filtering by event type (PRODUCED, TRANSFORMED, etc.)
- Product-based queries
- Facility-based filtering
- Date range queries
- Pagination strategies
- Combining multiple filters
- Batch ID lookups

**Use case:** Querying and analyzing historical supply chain data.

**Concepts:**
- Multi-criteria filtering
- Pagination with offset/limit
- Date range queries
- Result aggregation

**9 search patterns demonstrated:**
1. Filter by event type
2. Filter by product ID
3. Date range filtering
4. Filter by facility
5. Complex multi-filter queries
6. Pagination through large result sets
7. Batch ID filtering
8. Custom pagination limits
9. Advanced multi-criteria searches

---

### 5. Complete Production Workflow (`05-production-workflow.ts`)

**What you'll learn:**
- End-to-end factory workflow orchestration
- Multi-step process tracking
- Quality assurance and certification
- Audit trail generation
- Comprehensive lineage verification
- Production reporting

**Use case:** Complete factory production from raw material receipt to shipment.

**Concepts:**
- Workflow orchestration
- Error handling at each step
- Audit trail generation
- Compliance documentation
- Production metrics

**8-step workflow:**
1. **Receive Raw Materials** - Track incoming materials (wheat, salt, yeast)
2. **Process Materials** - Transform into finished goods (flour)
3. **Quality Certification** - USDA Organic certification
4. **Packaging** - Create distribution units
5. **Shipping** - Record outbound logistics
6. **Verify Lineage** - Confirm complete traceability
7. **Generate Audit Trail** - Create compliance documentation
8. **Production Summary** - Generate reports

---

## Example Output

### Basic Ingestion
```
=== Basic Event Ingestion Example ===

📦 Creating a PRODUCED event...
✓ Event created
  Batch ID: BATCH-2025-001
  Product: Organic Coffee Beans
  Quantity: 1000 kg

🚀 Submitting event to API...
✅ Event ingested successfully!

Response:
  Claim ID: 01JCXXX...
  VC JWT: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9...
  Lineage Hash: abc123def456...
  Verified: true
```

### Lineage Query
```
📊 Lineage Graph Summary:
  Batch ID: BATCH-PKG-RETAIL-002
  Total Claims: 5
  Graph Depth: 2

⬆️  Upstream Sources (1):
  • BATCH-RAW-CACAO-001
    - Claims: 1
    - Depth: 1

📋 Complete Supply Chain History:
1. PRODUCED Event
   ├─ Batch: BATCH-RAW-CACAO-001
   ├─ Product: Organic Cacao Beans
   ├─ Facility: ORG-FARM
   └─ Quantity: 1000 kg

2. TRANSFORMED Event
   ├─ Batch: BATCH-PROC-CHOC-001
   ├─ Product: Dark Chocolate Bars
   ├─ Input Batches: BATCH-RAW-CACAO-001
   └─ Quantity: 800 kg

...
```

## Best Practices

### 1. Error Handling
Always wrap SDK calls in try-catch blocks:

```typescript
try {
  const response = await client.ingestEvent(event);
  // Handle success
} catch (error) {
  console.error('Failed to ingest event:', error);
  // Handle error (retry, log, alert, etc.)
}
```

### 2. Batch Processing
Use batches for high-volume operations:

```typescript
// For critical operations where all must succeed
const batch = client.createBatch(events, 'atomic');

// For best performance with partial success allowed
const batch = client.createBatch(events, 'best-effort');
```

### 3. Pagination
Always handle pagination for search results:

```typescript
let offset = 0;
const limit = 50;

while (true) {
  const result = await client.searchClaims({ limit, offset });
  // Process result.claims

  if (!result.has_more) break;
  offset += limit;
}
```

### 4. Metadata
Include rich metadata for auditability:

```typescript
const event = client.createProducedEvent(
  batchId,
  facilityId,
  productId,
  quantity,
  unit,
  {
    // Add relevant metadata
    certifications: ['USDA Organic', 'Fair Trade'],
    quality_tests: {
      moisture: '12%',
      purity: '99.9%'
    },
    production_date: new Date().toISOString(),
    operator: 'John Doe',
    shift: 'morning'
  }
);
```

## API Reference

For complete API documentation, see:
- **OpenAPI Spec**: `../../specs/openapi.yaml`
- **SDK Documentation**: `../sdk/README.md`

## Troubleshooting

### API Connection Error
```
Error: connect ECONNREFUSED 127.0.0.1:3000
```

**Solution**: Ensure the API server is running on port 3000:
```bash
cd ../../rust/service
cargo run
```

### TypeScript Compilation Error
```
Error: Cannot find module '@mycelix/supplychain-sdk'
```

**Solution**: Build the SDK first:
```bash
cd ../sdk
npm install
npm run build
```

### Batch Ingestion Failure
```
Failed: 25, Succeeded: 0
```

**Solution**: Check individual error messages in the batch response:
```typescript
response.results.forEach(result => {
  if (result.status === 'error') {
    console.error('Error:', result.error);
  }
});
```

## Advanced Usage

### Custom Client Configuration

```typescript
const client = new SupplyChainClient({
  baseURL: 'http://localhost:3000',
  timeout: 30000,  // 30 second timeout
  headers: {
    'X-API-Key': 'your-api-key',  // If authentication enabled
  }
});
```

### Programmatic Event Creation

```typescript
// Instead of using helpers, create events manually:
const event = {
  batch_id: 'BATCH-001',
  facility_id: 'ORG-FACTORY',
  product_id: 'Widget',
  event_type: 'PRODUCED',
  quantity: 1000,
  unit: 'units',
  timestamp: new Date().toISOString(),
  metadata: JSON.stringify({
    custom_field: 'value'
  })
};

await client.ingestEvent(event);
```

## Contributing

Found a bug or want to add an example? Please open an issue or pull request!

## License

See the main project LICENSE file.

## Support

For questions or issues:
- Open an issue on GitHub
- Check the main project documentation
- Review the OpenAPI specification

---

**Happy tracking! 🎯**
