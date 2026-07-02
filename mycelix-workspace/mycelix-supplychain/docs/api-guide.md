# API Usage Guide

Complete reference for integrating with the Mycelix Supply Chain Provenance API.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [POST /v1/events](#post-v1events)
  - [GET /v1/claims/:id](#get-v1claimsid)
  - [GET /v1/batches/:id/lineage](#get-v1batchesidlineage)
  - [POST /v1/verify](#post-v1verify)
  - [GET /health](#get-health)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Best Practices](#best-practices)
- [Code Examples](#code-examples)

## Base URL

Default base URL for local development:
```
http://localhost:8080
```

Production deployments should use HTTPS:
```
https://api.example.com
```

## Authentication

**Current Status**: No authentication required (v0.1.0)

**Future Roadmap**:
- API keys (planned for v0.2.0)
- JWT tokens for service-to-service auth
- OAuth 2.0 for third-party integrations

## Endpoints

### POST /v1/events

Ingest a supply chain event and create a verifiable claim.

**Request**:
```
POST /v1/events
Content-Type: application/json
```

**Body** (Supply Event VC):
```json
{
  "@context": ["https://www.w3.org/2018/credentials/v1"],
  "type": ["VerifiableCredential"],
  "issuer": "did:mycelix:org:acme-corp",
  "issuanceDate": "2025-11-15T10:30:00Z",
  "credentialSubject": {
    "eventType": "PRODUCED",
    "productId": "SKU-12345",
    "batchId": "BATCH-001",
    "quantity": 1000.0,
    "unit": "kg",
    "facility": {
      "id": "FAC-001",
      "name": "Main Factory",
      "location": {
        "country": "US",
        "region": "California"
      }
    },
    "timestamp": "2025-11-15T10:30:00Z"
  }
}
```

**Response** (201 Created):
```json
{
  "vc_jwt": "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9...",
  "claim_id": "550e8400-e29b-41d4-a716-446655440000",
  "lineage_hash": "a3f5b8c9d2e1f4a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0",
  "previous_claims": []
}
```

**Event Types**:

#### PRODUCED
Initial production of goods.
```json
{
  "eventType": "PRODUCED",
  "productId": "SKU-001",
  "batchId": "BATCH-001",
  "quantity": 1000,
  "unit": "kg",
  "facility": {
    "id": "FAC-001",
    "name": "Factory Name"
  }
}
```

#### SHIPPED
Shipment of goods from one location to another.
```json
{
  "eventType": "SHIPPED",
  "productId": "SKU-001",
  "batchId": "BATCH-001",
  "quantity": 1000,
  "unit": "kg",
  "facility": {
    "id": "FAC-001",
    "name": "Factory Name"
  },
  "shipment": {
    "shipmentId": "SHIP-001",
    "carrier": "ACME Logistics",
    "trackingNumber": "TRACK123",
    "origin": "FAC-001",
    "destination": "WH-002"
  }
}
```

#### RECEIVED
Receipt of goods at destination.
```json
{
  "eventType": "RECEIVED",
  "productId": "SKU-001",
  "batchId": "BATCH-001",
  "quantity": 1000,
  "unit": "kg",
  "facility": {
    "id": "WH-002",
    "name": "Warehouse 2"
  }
}
```

#### TRANSFORMED
Transformation of one or more batches into a new product.
```json
{
  "eventType": "TRANSFORMED",
  "productId": "SKU-PROCESSED",
  "batchId": "BATCH-TRANSFORMED-001",
  "prevBatchIds": ["BATCH-001", "BATCH-002"],
  "quantity": 500,
  "unit": "kg",
  "facility": {
    "id": "FAC-PROCESSING",
    "name": "Processing Plant"
  },
  "metadata": {
    "processType": "roasting",
    "temperature": "200C"
  }
}
```

#### CERTIFIED
Quality or compliance certification.
```json
{
  "eventType": "CERTIFIED",
  "productId": "SKU-001",
  "batchId": "BATCH-001",
  "quantity": 1000,
  "unit": "kg",
  "facility": {
    "id": "CERT-LAB",
    "name": "Certification Lab"
  },
  "certification": {
    "certType": "ORGANIC",
    "certBody": "USDA",
    "certId": "CERT-12345",
    "validUntil": "2026-11-15T00:00:00Z"
  }
}
```

**Validation Rules**:
- `issuer` must be a valid DID format
- `eventType` must be one of: PRODUCED, TRANSFORMED, SHIPPED, RECEIVED, CERTIFIED
- `quantity` must be positive
- `prevBatchIds` required for TRANSFORMED events
- Timestamps should be ISO 8601 format

**Error Responses**:

400 Bad Request - Invalid event data:
```json
{
  "error": "Validation failed",
  "message": "Quantity must be positive",
  "code": "VALIDATION_ERROR"
}
```

500 Internal Server Error:
```json
{
  "error": "Internal server error",
  "message": "Failed to store claim",
  "code": "INTERNAL_ERROR"
}
```

---

### GET /v1/claims/:id

Retrieve a claim by ID.

**Request**:
```
GET /v1/claims/550e8400-e29b-41d4-a716-446655440000
```

**Response** (200 OK):
```json
{
  "claim": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "type": "DkgClaim",
    "issuer": "did:mycelix:org:acme-corp",
    "subject": {
      "productId": "SKU-12345",
      "batchId": "BATCH-001"
    },
    "assertion": {
      "eventType": "PRODUCED",
      "quantity": 1000.0,
      "unit": "kg"
    },
    "evidence": {
      "facility": {
        "id": "FAC-001",
        "name": "Main Factory"
      }
    },
    "lineage": {
      "hash": "a3f5b8c9...",
      "previous_claims": []
    },
    "timestamp": "2025-11-15T10:30:00Z"
  },
  "vc_jwt": "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9..."
}
```

**Error Responses**:

404 Not Found:
```json
{
  "error": "Claim not found",
  "message": "No claim found with ID: 550e8400-...",
  "code": "NOT_FOUND"
}
```

---

### GET /v1/batches/:id/lineage

Get the full lineage tree for a batch.

**Request**:
```
GET /v1/batches/BATCH-001/lineage
```

**Response** (200 OK):
```json
{
  "batch_id": "BATCH-001",
  "claims": [
    {
      "id": "claim-1",
      "eventType": "PRODUCED",
      "timestamp": "2025-11-15T10:00:00Z",
      "lineage": {
        "hash": "abc123...",
        "previous_claims": []
      }
    },
    {
      "id": "claim-2",
      "eventType": "SHIPPED",
      "timestamp": "2025-11-15T14:00:00Z",
      "lineage": {
        "hash": "def456...",
        "previous_claims": ["claim-1"]
      }
    }
  ]
}
```

**Query Parameters**:
- `depth` (optional): Maximum depth to traverse (default: unlimited)
- `format` (optional): Response format (json, dot) - future feature

**Error Responses**:

404 Not Found - No lineage found:
```json
{
  "batch_id": "BATCH-UNKNOWN",
  "claims": []
}
```

---

### POST /v1/verify

Verify a Verifiable Credential JWT.

**Request**:
```
POST /v1/verify
Content-Type: application/json
```

**Body**:
```json
{
  "vc_jwt": "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9..."
}
```

**Response** (200 OK):
```json
{
  "valid": true,
  "claim_id": "550e8400-e29b-41d4-a716-446655440000",
  "issuer": "did:mycelix:org:acme-corp"
}
```

**Response** (200 OK - Invalid):
```json
{
  "valid": false,
  "error": "Invalid signature"
}
```

---

### GET /health

Health check endpoint.

**Request**:
```
GET /health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-11-15T10:30:00Z"
}
```

---

## Error Handling

### Error Response Format

All error responses follow this structure:
```json
{
  "error": "Error type",
  "message": "Human-readable error description",
  "code": "ERROR_CODE"
}
```

### HTTP Status Codes

| Code | Meaning | When Used |
|------|---------|-----------|
| 200 | OK | Successful GET request |
| 201 | Created | Successful POST /v1/events |
| 400 | Bad Request | Invalid request data |
| 404 | Not Found | Resource doesn't exist |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service temporarily down |

### Common Error Codes

| Code | Description | Action |
|------|-------------|--------|
| `VALIDATION_ERROR` | Request failed validation | Check request schema |
| `NOT_FOUND` | Resource not found | Verify resource ID |
| `INTERNAL_ERROR` | Server error | Retry with backoff |
| `DATABASE_ERROR` | Database operation failed | Contact support |

### Retry Logic

Implement exponential backoff for 5xx errors:

```typescript
async function ingestWithRetry(event, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await ingestEvent(event);
    } catch (error) {
      if (error.status < 500 || i === maxRetries - 1) {
        throw error;
      }
      await sleep(Math.pow(2, i) * 1000); // 1s, 2s, 4s
    }
  }
}
```

---

## Rate Limiting

**Current Status**: No rate limiting (v0.1.0)

**Future Roadmap** (v0.2.0):
- 100 requests/minute per API key
- 1000 requests/hour per API key
- Rate limit headers:
  - `X-RateLimit-Limit`: Requests allowed per window
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Unix timestamp of window reset

---

## Best Practices

### 1. Use Batch Ingestion for High Volume

For bulk event ingestion, batch requests in groups of 10-100:

```typescript
async function ingestBatch(events) {
  const chunks = chunk(events, 50); // 50 events per batch

  for (const batch of chunks) {
    await Promise.all(
      batch.map(event => client.ingestEvent(event))
    );
  }
}
```

### 2. Store Claim IDs for Later Retrieval

Always store the `claim_id` returned from ingestion:

```typescript
const result = await client.ingestEvent(event);
await db.storeClaim(result.claim_id, result.vc_jwt);
```

### 3. Validate Events Before Ingestion

Validate events client-side to catch errors early:

```typescript
function validateEvent(event) {
  if (!event.issuer.startsWith('did:')) {
    throw new Error('Invalid DID format');
  }
  if (event.credentialSubject.quantity <= 0) {
    throw new Error('Quantity must be positive');
  }
}
```

### 4. Use Idempotency Keys (Future)

For critical events, use idempotency keys to prevent duplicate ingestion:

```typescript
// Future feature
const result = await client.ingestEvent(event, {
  idempotencyKey: `${batchId}-${timestamp}`
});
```

### 5. Monitor Lineage Hashes

Verify lineage integrity by checking hashes:

```typescript
function verifyLineage(claim) {
  const computed = computeLineageHash(
    claim.vc_jwt,
    claim.lineage.previous_claims
  );
  return computed === claim.lineage.hash;
}
```

---

## Code Examples

### cURL

#### Ingest a PRODUCED Event
```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d '{
    "@context": ["https://www.w3.org/2018/credentials/v1"],
    "type": ["VerifiableCredential"],
    "issuer": "did:mycelix:org:acme",
    "issuanceDate": "2025-11-15T10:00:00Z",
    "credentialSubject": {
      "eventType": "PRODUCED",
      "productId": "SKU-001",
      "batchId": "BATCH-001",
      "quantity": 1000,
      "unit": "kg",
      "facility": {
        "id": "FAC-001",
        "name": "Factory"
      },
      "timestamp": "2025-11-15T10:00:00Z"
    }
  }'
```

#### Get a Claim
```bash
curl http://localhost:8080/v1/claims/550e8400-e29b-41d4-a716-446655440000
```

#### Get Batch Lineage
```bash
curl http://localhost:8080/v1/batches/BATCH-001/lineage
```

### TypeScript/JavaScript

```typescript
import { SupplyChainClient } from '@mycelix/supplychain-sdk';

const client = new SupplyChainClient({
  baseUrl: 'http://localhost:8080'
});

// Ingest event
const event = client.createProducedEvent({
  issuer: 'did:mycelix:org:acme',
  productId: 'SKU-001',
  batchId: 'BATCH-001',
  quantity: 1000,
  unit: 'kg',
  facility: { id: 'FAC-001', name: 'Factory' }
});

const result = await client.ingestEvent(event);
console.log(`Claim ID: ${result.claim_id}`);

// Get claim
const claim = await client.getClaim(result.claim_id);
console.log(claim);

// Get lineage
const lineage = await client.getBatchLineage('BATCH-001');
console.log(`${lineage.claims.length} events in lineage`);
```

### Python

```python
import requests
from datetime import datetime

base_url = 'http://localhost:8080'

# Ingest event
event = {
    '@context': ['https://www.w3.org/2018/credentials/v1'],
    'type': ['VerifiableCredential'],
    'issuer': 'did:mycelix:org:acme',
    'issuanceDate': datetime.utcnow().isoformat() + 'Z',
    'credentialSubject': {
        'eventType': 'PRODUCED',
        'productId': 'SKU-001',
        'batchId': 'BATCH-001',
        'quantity': 1000.0,
        'unit': 'kg',
        'facility': {
            'id': 'FAC-001',
            'name': 'Factory'
        },
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
}

response = requests.post(f'{base_url}/v1/events', json=event)
result = response.json()
print(f"Claim ID: {result['claim_id']}")

# Get claim
claim_response = requests.get(f"{base_url}/v1/claims/{result['claim_id']}")
claim = claim_response.json()
print(claim)

# Get lineage
lineage_response = requests.get(f"{base_url}/v1/batches/BATCH-001/lineage")
lineage = lineage_response.json()
print(f"{len(lineage['claims'])} events in lineage")
```

### Rust

```rust
use reqwest::Client;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let base_url = "http://localhost:8080";

    // Ingest event
    let event = json!({
        "@context": ["https://www.w3.org/2018/credentials/v1"],
        "type": ["VerifiableCredential"],
        "issuer": "did:mycelix:org:acme",
        "issuanceDate": "2025-11-15T10:00:00Z",
        "credentialSubject": {
            "eventType": "PRODUCED",
            "productId": "SKU-001",
            "batchId": "BATCH-001",
            "quantity": 1000.0,
            "unit": "kg",
            "facility": {
                "id": "FAC-001",
                "name": "Factory"
            },
            "timestamp": "2025-11-15T10:00:00Z"
        }
    });

    let response = client
        .post(&format!("{}/v1/events", base_url))
        .json(&event)
        .send()
        .await?;

    let result: serde_json::Value = response.json().await?;
    println!("Claim ID: {}", result["claim_id"]);

    Ok(())
}
```

---

## Webhooks (Future Feature)

**Planned for v0.3.0**:

Register webhooks to receive real-time event notifications:

```typescript
// Future API
await client.registerWebhook({
  url: 'https://your-app.com/webhooks/supply-chain',
  events: ['claim.created', 'claim.verified'],
  secret: 'your-webhook-secret'
});
```

Webhook payload:
```json
{
  "event": "claim.created",
  "timestamp": "2025-11-15T10:30:00Z",
  "data": {
    "claim_id": "550e8400-e29b-41d4-a716-446655440000",
    "batch_id": "BATCH-001",
    "event_type": "PRODUCED"
  }
}
```

---

## Support

- **Documentation**: https://docs.mycelix.dev
- **GitHub Issues**: https://github.com/Luminous-Dynamics/mycelix-supplychain/issues
- **Email**: support@luminous-dynamics.dev

---

**Version**: 0.1.0
**Last Updated**: 2025-11-15
