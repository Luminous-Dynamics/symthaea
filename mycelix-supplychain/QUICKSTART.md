# Mycelix Supply Chain - Quickstart Guide

Get up and running with the Mycelix Supply Chain provenance system in 5 minutes!

## Prerequisites

- **Rust** 1.75+ ([install](https://rustup.rs/))
- **Node.js** 18+ ([install](https://nodejs.org/))
- **Make** (usually pre-installed on Linux/Mac)
- **curl** (for testing)

## Quick Install

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix-supplychain.git
cd mycelix-supplychain

# One-command setup
make setup
```

## Your First Supply Chain Event

### 1. Start the Service

```bash
make run
```

You should see:
```
INFO Service DID: did:key:...
INFO Starting server on 0.0.0.0:8080
```

### 2. Check Health

Open a new terminal:

```bash
curl http://localhost:8080/health
```

Response:
```json
{"status":"ok","version":"0.1.0"}
```

### 3. Post a "Produced" Event

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d '{
    "@context": [
      "https://www.w3.org/2018/credentials/v1",
      "https://mycelix.org/contexts/supply-chain/v1"
    ],
    "type": ["VerifiableCredential", "SupplyChainEvent"],
    "issuer": "did:mycelix:org:acme-manufacturing",
    "issuanceDate": "2025-11-15T08:00:00Z",
    "credentialSubject": {
      "eventType": "PRODUCED",
      "productId": "SKU-WIDGET-42",
      "batchId": "BATCH-2025-001",
      "quantity": 5000,
      "unit": "pieces",
      "facility": {
        "id": "FAC-PLANT-A",
        "name": "ACME Manufacturing Plant A"
      },
      "timestamp": "2025-11-15T08:00:00Z"
    }
  }'
```

Response:
```json
{
  "vc_jwt": "eyJ...",
  "claim_id": "a7b3c...",
  "lineage_hash": "8f9e...",
  "previous_claims": null
}
```

Save the `claim_id` for the next step!

### 4. Retrieve the Claim

```bash
curl http://localhost:8080/v1/claims/YOUR_CLAIM_ID
```

You'll see the full claim with lineage information.

## Full Supply Chain Flow

Let's trace a product through its lifecycle:

### 1. Production

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @specs/examples/batch_produced.json
```

### 2. Shipment

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @specs/examples/shipment_departed.json
```

### 3. Certification

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @specs/examples/certificate_issued.json
```

### 4. View Lineage

Now each event is linked! The shipment event will show the production event in its `previous_claims`.

## Using the CSV Adapter

Bulk ingest events from a CSV file:

```bash
# Build the adapter
cd ts/adapters/csv
npm ci && npm run build

# Ingest test data
node dist/ingest.js -f ../../../tests/data/small_demo.csv
```

Output:
```
Connected to API (version 0.1.0)
✓ Ingested PRODUCED for batch BATCH-2025-001 → claim abc123
✓ Ingested PRODUCED for batch BATCH-2025-002 → claim def456
✓ Ingested TRANSFORMED for batch BATCH-2025-ASM-001 → claim ghi789
...
Summary: 5 processed, 0 errors
```

## Using the TypeScript SDK

Create a simple client:

```typescript
import { SupplyChainClient } from '@mycelix/supplychain-sdk';

const client = new SupplyChainClient({
  baseUrl: 'http://localhost:8080'
});

// Create a produced event
const event = client.createProducedEvent({
  issuer: 'did:mycelix:org:my-company',
  productId: 'SKU-123',
  batchId: 'BATCH-001',
  quantity: 1000,
  unit: 'kg',
  facility: {
    id: 'FAC-001',
    name: 'My Factory'
  }
});

// Ingest it
const result = await client.ingestEvent(event);
console.log('Claim ID:', result.claim_id);

// Retrieve it
const claim = await client.getClaim(result.claim_id);
console.log('Claim:', claim);
```

## Next Steps

### Explore the API

- **OpenAPI Spec**: `specs/openapi.yaml`
- **Schemas**: `specs/schemas/`
- **Examples**: `specs/examples/`

### Read the Docs

- **Architecture**: `docs/architecture.md` (coming soon)
- **API Guide**: `docs/api-guide.md` (coming soon)
- **Deployment**: `docs/deployment.md` (coming soon)

### Build Your Integration

1. **ERP Integration**: Use the CSV adapter as a template
2. **IoT Sensors**: Use the MQTT adapter
3. **Custom App**: Use the TypeScript SDK

### Deploy to Production

```bash
# Docker Compose (easy mode)
docker-compose -f deployments/docker-compose.yml up

# Kubernetes (production mode)
kubectl apply -f deployments/k8s/deployment.yaml
```

## Common Tasks

```bash
# Run tests
make test

# Format code
make fmt

# Lint code
make lint

# Build everything
make build

# Clean artifacts
make clean

# Generate documentation
make docs

# See all commands
make help
```

## Troubleshooting

### Service won't start

- **Port 8080 in use**: Change `SERVICE_PORT` in `.env`
- **Dependencies missing**: Run `make install-deps`

### Events rejected with validation errors

- Check the JSON schema: `specs/schemas/vc.supplyEvent.v1.json`
- Ensure `issuer` starts with `did:`
- Ensure `eventType` is one of: PRODUCED, TRANSFORMED, SHIPPED, RECEIVED, CERTIFIED
- For TRANSFORMED events, include `prevBatchIds`

### TypeScript errors

```bash
cd ts/sdk
npm ci
npm run build
```

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/Luminous-Dynamics/mycelix-supplychain/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Luminous-Dynamics/mycelix-supplychain/discussions)
- **Documentation**: Check the `docs/` directory

## What's Next?

Now that you have the basics running, explore:

1. **Lineage Tracking**: How to trace products through transformations
2. **Verifiable Credentials**: Understanding the cryptographic signatures
3. **DKG Integration**: Publishing claims to the distributed knowledge graph (coming soon)
4. **Selective Disclosure**: Hiding sensitive fields in VCs (coming soon)
5. **Product Passports**: Creating exportable provenance bundles

Happy tracking! 🚀
