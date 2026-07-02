# Example 2: Full Supply Chain

This example demonstrates a complete supply chain lifecycle with lineage tracking.

## Scenario

We'll track coffee beans from farm to cup:

1. **PRODUCED** - Coffee beans harvested at farm
2. **SHIPPED** - Beans shipped to roastery
3. **RECEIVED** - Roastery receives beans
4. **TRANSFORMED** - Beans roasted into product
5. **CERTIFIED** - Quality certification issued
6. **SHIPPED** - Product shipped to retailer
7. **RECEIVED** - Retailer receives product

## What You'll Learn

- How to create a lineage chain
- How TRANSFORMED events link parent batches
- How to track a product through multiple facilities
- How lineage hashes create tamper-evident trails

## Running the Example

### Automated (Recommended)

```bash
chmod +x run.sh
./run.sh
```

### Manual

```bash
# 1. PRODUCED - Harvest
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @01-produced.json

# 2. SHIPPED - To roastery
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @02-shipped.json

# 3. RECEIVED - At roastery
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @03-received.json

# 4. TRANSFORMED - Roasting
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @04-transformed.json

# 5. CERTIFIED - Quality check
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @05-certified.json

# 6. SHIPPED - To retailer
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @06-shipped.json

# 7. RECEIVED - At retailer
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @07-received.json
```

## Understanding Lineage

### Simple Chain (Same Batch)

```
PRODUCED → SHIPPED → RECEIVED
(BATCH-001) (BATCH-001) (BATCH-001)
```

Each event references the previous one for the same batch.

### Transformation (New Batch)

```
PRODUCED     PRODUCED
(BATCH-RAW1) (BATCH-RAW2)
    ↓            ↓
    └─→ TRANSFORMED ←─┘
       (BATCH-ROASTED)
```

The TRANSFORMED event's `prevBatchIds` links to parent batches.

### Certification (Related)

```
PRODUCED → TRANSFORMED
    ↓          ↓
    └→ CERTIFIED ←┘
```

Certification can link to multiple related claims.

## Lineage Hash

Each claim includes a `lineage_hash`:

```
hash = SHA256(vc_jwt || parent_claim_1_id || parent_claim_2_id || ...)
```

This creates a **Merkle tree structure** where:
- Tampering any event invalidates all descendants
- Full history can be cryptographically verified
- Independent auditors can verify the chain

## Querying Lineage

Once all events are created, you can query the full history:

```bash
# Get batch lineage (when implemented)
curl http://localhost:8080/v1/batches/BATCH-ROASTED/lineage
```

This will return the full tree of events leading to this batch.

## Expected Output

Run `./run.sh` and you should see:

```
╔════════════════════════════════════════╗
║  Full Supply Chain Example             ║
╚════════════════════════════════════════╝

✓ PRODUCED: Coffee beans harvested
  Batch: BATCH-RAW-BEANS-001
  Facility: FARM-COLOMBIA-01
  Claim ID: abc123...

✓ SHIPPED: Beans to roastery
  Previous: abc123...
  Claim ID: def456...

...

╔════════════════════════════════════════╗
║  Complete! 7 events created             ║
║  Full lineage established               ║
╚════════════════════════════════════════╝

Lineage Chain:
  abc123... → def456... → ghi789... → jkl012...
```

## Files

- `01-produced.json` through `07-received.json` - Event payloads
- `run.sh` - Automated script
- `visualize.sh` - Generate lineage graph (requires GraphViz)
- `README.md` - This file

## Next Steps

- Modify event files to create your own supply chain
- Try [Example 3: TypeScript Client](../03-typescript-client/) to use the SDK
- Add more events to create complex lineage graphs
