# Example 1: Simple Flow

This example shows the simplest possible workflow: create a single PRODUCED event and retrieve it.

## What You'll Learn

- How to start the service
- How to post a supply chain event
- How to retrieve a claim
- How to verify the response

## Prerequisites

- Service running on `localhost:8080`
- `curl` installed
- Basic understanding of JSON

## Step-by-Step

### 1. Start the Service

```bash
cd ../../rust/service
cargo run
```

You should see:
```
INFO Service DID: did:key:...
INFO Starting server on 0.0.0.0:8080
```

### 2. Check Health

In a new terminal:

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### 3. Post a PRODUCED Event

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @event.json
```

Expected response:
```json
{
  "vc_jwt": "eyJhbGc...long-jwt-string",
  "claim_id": "a7b3c4d5-...",
  "lineage_hash": "8f9e1d2c...",
  "previous_claims": null
}
```

**Save the `claim_id`** for the next step!

### 4. Retrieve the Claim

```bash
curl http://localhost:8080/v1/claims/YOUR_CLAIM_ID_HERE
```

Expected response:
```json
{
  "claim": {
    "id": "a7b3c4d5-...",
    "type": "SupplyChainClaim",
    "issuer": "did:mycelix:org:acme-widgets",
    "subject": {
      "batchId": "BATCH-2025-SIMPLE-001",
      "productId": "SKU-WIDGET-100"
    },
    "assertion": {
      "eventType": "PRODUCED",
      "quantity": 1000,
      "unit": "pieces",
      "facilityId": "FAC-PLANT-1"
    },
    "evidence": {
      "vcJwt": "eyJhbGc..."
    },
    "lineage": {
      "hash": "8f9e1d2c...",
      "previousClaims": null
    },
    "timestamp": "2025-11-15T18:30:00Z",
    "confidence": 1.0
  },
  "lineage": null
}
```

### 5. Verify the VC (Optional)

```bash
curl -X POST http://localhost:8080/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{
    "vc_jwt": "YOUR_VC_JWT_HERE",
    "check_lineage": true
  }'
```

Expected response:
```json
{
  "valid": true,
  "signature_valid": true,
  "lineage_valid": true,
  "issuer": "did:mycelix:org:acme-widgets"
}
```

## Understanding the Response

### claim_id
A unique identifier for this claim in the system. Use this to retrieve the claim later.

### vc_jwt
The signed Verifiable Credential in JWT format. This can be:
- Stored in your database
- Shared with trading partners
- Verified independently
- Used to prove the event occurred

### lineage_hash
A cryptographic hash that commits to:
- The VC content
- All parent claims (if any)

This creates a tamper-evident chain.

### previous_claims
For a PRODUCED event (root of lineage tree), this is `null`.

For other event types, this contains the IDs of parent claims.

## What's Happening Under the Hood

```
1. Your Event (JSON)
   ↓
2. Service validates against schema
   ↓
3. Service creates a Verifiable Credential
   ↓
4. Service signs with Ed25519 key
   ↓
5. Service computes lineage hash
   ↓
6. Service creates DKG claim
   ↓
7. Service stores claim (in-memory for now)
   ↓
8. Response with claim_id + vc_jwt
```

## Common Issues

### "Connection refused"
- Make sure service is running (`cargo run` in rust/service)
- Check it's using port 8080 (`netstat -an | grep 8080`)

### "Validation error: issuer must be a DID"
- Make sure `issuer` starts with `did:`
- Example: `"issuer": "did:mycelix:org:your-company"`

### "Validation error: quantity must be positive"
- Ensure `quantity` is greater than 0
- Use `0.0` as minimum for fractional quantities

## Next Steps

- Try [Example 2: Full Supply Chain](../02-full-supply-chain/) to see lineage tracking
- Try [Example 3: TypeScript Client](../03-typescript-client/) to use the SDK
- Modify `event.json` to create your own products

## Files in This Example

- `event.json` - The supply chain event payload
- `run.sh` - Automated script to run all steps
- `README.md` - This file
