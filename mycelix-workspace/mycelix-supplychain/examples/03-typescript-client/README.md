# Example 3: TypeScript Client

This example shows how to use the TypeScript SDK to integrate with the Mycelix Supply Chain API.

## What You'll Learn

- How to install and use `@mycelix/supplychain-sdk`
- How to create events programmatically
- How to handle responses and errors
- How to build a simple tracking application

## Prerequisites

- Node.js 18+ installed
- Service running on `localhost:8080`
- Basic TypeScript/JavaScript knowledge

## Installation

```bash
cd ../../ts/sdk
npm install
npm run build

cd ../../examples/03-typescript-client
npm install
```

## Running the Examples

### Basic Usage

```bash
npm run example:basic
```

### Full Workflow

```bash
npm run example:workflow
```

### Error Handling

```bash
npm run example:errors
```

## Code Examples

### 1. Basic Setup

```typescript
import { SupplyChainClient } from '@mycelix/supplychain-sdk';

const client = new SupplyChainClient({
  baseUrl: 'http://localhost:8080'
});

// Check health
const health = await client.health();
console.log(`Service version: ${health.version}`);
```

### 2. Create a PRODUCED Event

```typescript
const event = client.createProducedEvent({
  issuer: 'did:mycelix:org:my-company',
  productId: 'SKU-12345',
  batchId: 'BATCH-001',
  quantity: 1000,
  unit: 'kg',
  facility: {
    id: 'FAC-001',
    name: 'Main Factory'
  }
});

const result = await client.ingestEvent(event);
console.log(`Claim ID: ${result.claim_id}`);
console.log(`VC JWT: ${result.vc_jwt}`);
```

### 3. Create a SHIPPED Event

```typescript
const shipment = client.createShippedEvent({
  issuer: 'did:mycelix:org:logistics-co',
  productId: 'SKU-12345',
  batchId: 'BATCH-001',
  quantity: 1000,
  unit: 'kg',
  facility: {
    id: 'FAC-001',
    name: 'Main Factory'
  },
  shipment: {
    shipmentId: 'SHIP-001',
    carrier: 'ACME Logistics',
    trackingNumber: 'TRACK-123456',
    origin: 'FAC-001',
    destination: 'WH-002'
  }
});

const result = await client.ingestEvent(shipment);
```

### 4. Create a TRANSFORMED Event

```typescript
const transformed = client.createTransformedEvent({
  issuer: 'did:mycelix:org:processor',
  productId: 'SKU-PROCESSED',
  batchId: 'BATCH-TRANSFORMED-001',
  prevBatchIds: ['BATCH-001', 'BATCH-002'], // Parent batches
  quantity: 500,
  unit: 'kg',
  facility: {
    id: 'FAC-PROCESSING',
    name: 'Processing Plant'
  },
  metadata: {
    processType: 'roasting',
    temperature: '200C',
    duration: '15min'
  }
});

const result = await client.ingestEvent(transformed);
```

### 5. Retrieve a Claim

```typescript
const claim = await client.getClaim(result.claim_id);
console.log(`Batch: ${claim.claim.subject.batchId}`);
console.log(`Event: ${claim.claim.assertion.eventType}`);
console.log(`Lineage hash: ${claim.claim.lineage.hash}`);
```

### 6. Error Handling

```typescript
try {
  const event = client.createProducedEvent({
    issuer: 'not-a-did', // Invalid!
    productId: 'SKU-001',
    batchId: 'BATCH-001',
    quantity: 1000,
    unit: 'kg',
    facility: { id: 'FAC-001', name: 'Factory' }
  });

  await client.ingestEvent(event);
} catch (error) {
  if (error.response?.status === 400) {
    console.error('Validation error:', error.response.data.message);
  } else {
    console.error('Unexpected error:', error.message);
  }
}
```

## Building a Simple App

See `src/app.ts` for a complete example that:
- Creates a supply chain flow
- Tracks lineage
- Displays results
- Handles errors gracefully

Run it with:

```bash
npm start
```

## SDK Features

### Helper Methods

- `createProducedEvent()` - For production events
- `createShippedEvent()` - For shipment events
- `createTransformedEvent()` - For transformation events
- `createReceivedEvent()` - For receipt events (you can add this!)

### API Methods

- `health()` - Check service health
- `ingestEvent()` - Post an event
- `getClaim()` - Retrieve a claim
- `getBatchLineage()` - Get batch lineage
- `verify()` - Verify a VC

### Type Safety

All methods are fully typed with TypeScript:

```typescript
interface EventResponse {
  vc_jwt: string;
  claim_id: string;
  lineage_hash: string;
  previous_claims?: string[];
}

interface DkgClaim {
  id: string;
  type: string;
  issuer: string;
  subject: Subject;
  assertion: Assertion;
  evidence: Evidence;
  lineage: Lineage;
  timestamp: string;
}
```

## Advanced Usage

### Retry Logic (Future)

```typescript
const client = new SupplyChainClient({
  baseUrl: 'http://localhost:8080',
  timeout: 30000,
  retries: 3,
  retryDelay: 1000
});
```

### Custom Headers

```typescript
const client = new SupplyChainClient({
  baseUrl: 'http://localhost:8080',
  headers: {
    'Authorization': 'Bearer your-api-key',
    'X-Custom-Header': 'value'
  }
});
```

## Next Steps

- Build your own integration using the SDK
- Add retry logic and error handling
- Create a React/Vue/Angular app
- Integrate with your ERP system

## Files

- `package.json` - Dependencies
- `src/basic.ts` - Basic usage example
- `src/workflow.ts` - Full workflow example
- `src/errors.ts` - Error handling
- `src/app.ts` - Complete application
- `README.md` - This file
