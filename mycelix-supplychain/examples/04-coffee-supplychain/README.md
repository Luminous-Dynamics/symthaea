# Coffee Supply Chain - Complete End-to-End Example

This example demonstrates a complete coffee supply chain journey from farm to cup, showcasing:
- **5 organizations** (Farm, Processor, Exporter, Roaster, Cafe)
- **8 supply chain events** (Production, Certification, Transformation, Shipping, Receiving)
- **Lineage tracking** across multiple transformations
- **Quality certifications** (Organic, Fair Trade)
- **Real-world data** with facilities, quantities, and metadata

## Coffee Journey Overview

```
┌─────────────┐
│   FARM      │  1. Coffee cherries harvested (5000 kg)
│  Ethiopia   │  2. Organic certification
└──────┬──────┘
       │ cherries
       ↓
┌─────────────┐
│  PROCESSOR  │  3. Transform cherries → green beans (1000 kg)
│  Ethiopia   │
└──────┬──────┘
       │ green beans
       ↓
┌─────────────┐
│   EXPORTER  │  4. Fair Trade certification
│  Ethiopia   │  5. Ship to USA
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   ROASTER   │  6. Receive green beans
│     USA     │  7. Transform green → roasted (850 kg)
└──────┬──────┘
       │ roasted beans
       ↓
┌─────────────┐
│    CAFE     │  8. Receive roasted beans
│     USA     │
└─────────────┘
```

## Prerequisites

```bash
# Start the provenance service
cd rust/service
cargo run --release

# Or use Docker
docker run -p 8080:8080 mycelix-supplychain:latest
```

## Run the Example

### Option 1: Automated Script (Recommended)

```bash
cd examples/04-coffee-supplychain
./run-coffee-demo.sh
```

This script will:
1. Ingest all 8 events in order
2. Show claim IDs and lineage hashes
3. Retrieve the final lineage tree
4. Display verification results

### Option 2: Manual Step-by-Step

Follow the steps below to manually execute each event.

---

## Event 1: Coffee Cherry Production

**Organization**: Highland Coffee Farm, Ethiopia
**Event**: PRODUCED
**Product**: Fresh Coffee Cherries
**Quantity**: 5000 kg

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @events/01-farm-produced.json

# Save the claim_id for later
```

<details>
<summary>View Event JSON</summary>

```json
{
  "@context": ["https://www.w3.org/2018/credentials/v1"],
  "type": ["VerifiableCredential"],
  "issuer": "did:mycelix:org:highland-coffee-farm",
  "issuanceDate": "2025-01-15T06:00:00Z",
  "credentialSubject": {
    "eventType": "PRODUCED",
    "productId": "COFFEE-CHERRIES-ETHIOPIAN-YIRGACHEFFE",
    "batchId": "BATCH-2025-HARVEST-001",
    "quantity": 5000.0,
    "unit": "kg",
    "facility": {
      "id": "FARM-HIGHLAND-ETH",
      "name": "Highland Coffee Farm",
      "location": {
        "country": "Ethiopia",
        "region": "Yirgacheffe",
        "coordinates": {
          "lat": 6.1624,
          "lon": 38.2056
        }
      }
    },
    "timestamp": "2025-01-15T06:00:00Z",
    "metadata": {
      "variety": "Heirloom Ethiopian",
      "altitude": "1800-2200m",
      "harvestMethod": "Hand-picked",
      "season": "2024/2025"
    }
  }
}
```
</details>

---

## Event 2: Organic Certification

**Organization**: Ethiopian Organic Certification Authority
**Event**: CERTIFIED
**Certification**: Organic

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @events/02-farm-certified-organic.json
```

<details>
<summary>View Event JSON</summary>

```json
{
  "@context": ["https://www.w3.org/2018/credentials/v1"],
  "type": ["VerifiableCredential"],
  "issuer": "did:mycelix:org:ethiopian-organic-cert",
  "issuanceDate": "2025-01-16T10:00:00Z",
  "credentialSubject": {
    "eventType": "CERTIFIED",
    "productId": "COFFEE-CHERRIES-ETHIOPIAN-YIRGACHEFFE",
    "batchId": "BATCH-2025-HARVEST-001",
    "quantity": 5000.0,
    "unit": "kg",
    "facility": {
      "id": "CERT-ETHIOPIAN-ORGANIC",
      "name": "Ethiopian Organic Certification Authority"
    },
    "timestamp": "2025-01-16T10:00:00Z",
    "certification": {
      "certType": "ORGANIC",
      "certBody": "Ethiopian Organic Certification Authority",
      "certId": "ETH-ORG-2025-001234",
      "validUntil": "2026-01-16T00:00:00Z",
      "standard": "EU Organic Regulation 834/2007"
    }
  }
}
```
</details>

---

## Event 3: Processing (Cherries → Green Beans)

**Organization**: Yirgacheffe Processing Mill
**Event**: TRANSFORMED
**Input**: 5000 kg cherries → **Output**: 1000 kg green beans (20% yield)

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @events/03-processor-transformed.json
```

<details>
<summary>View Event JSON</summary>

```json
{
  "@context": ["https://www.w3.org/2018/credentials/v1"],
  "type": ["VerifiableCredential"],
  "issuer": "did:mycelix:org:yirgacheffe-processing",
  "issuanceDate": "2025-01-20T14:00:00Z",
  "credentialSubject": {
    "eventType": "TRANSFORMED",
    "productId": "COFFEE-GREEN-BEANS-ETHIOPIAN-YIRGACHEFFE",
    "batchId": "BATCH-2025-GREEN-001",
    "prevBatchIds": ["BATCH-2025-HARVEST-001"],
    "quantity": 1000.0,
    "unit": "kg",
    "facility": {
      "id": "PROC-YIRGACHEFFE-MILL",
      "name": "Yirgacheffe Processing Mill",
      "location": {
        "country": "Ethiopia",
        "region": "Yirgacheffe"
      }
    },
    "timestamp": "2025-01-20T14:00:00Z",
    "metadata": {
      "processType": "Washed (Wet Processing)",
      "fermentationTime": "48 hours",
      "dryingMethod": "Sun-dried on raised beds",
      "moistureContent": "11%",
      "screenSize": "15/16"
    }
  }
}
```
</details>

---

## Event 4: Fair Trade Certification

**Organization**: Fair Trade Certification Body
**Event**: CERTIFIED

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @events/04-exporter-certified-fairtrade.json
```

---

## Event 5: Export Shipment to USA

**Organization**: Ethiopian Coffee Exporters
**Event**: SHIPPED

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @events/05-exporter-shipped.json
```

<details>
<summary>View Event JSON</summary>

```json
{
  "@context": ["https://www.w3.org/2018/credentials/v1"],
  "type": ["VerifiableCredential"],
  "issuer": "did:mycelix:org:ethiopian-coffee-exporters",
  "issuanceDate": "2025-02-01T08:00:00Z",
  "credentialSubject": {
    "eventType": "SHIPPED",
    "productId": "COFFEE-GREEN-BEANS-ETHIOPIAN-YIRGACHEFFE",
    "batchId": "BATCH-2025-GREEN-001",
    "quantity": 1000.0,
    "unit": "kg",
    "facility": {
      "id": "EXPORT-ETHIOPIAN-COFFEE",
      "name": "Ethiopian Coffee Exporters",
      "location": {
        "country": "Ethiopia",
        "region": "Addis Ababa"
      }
    },
    "timestamp": "2025-02-01T08:00:00Z",
    "shipment": {
      "shipmentId": "SHIP-ETH-USA-2025-001",
      "carrier": "Maersk Line",
      "trackingNumber": "MAEU123456789",
      "origin": "Addis Ababa, Ethiopia",
      "destination": "Oakland, CA, USA",
      "containerNumber": "MAEU1234567",
      "vesselName": "Maersk Sealand",
      "departureDate": "2025-02-01",
      "estimatedArrival": "2025-03-05"
    }
  }
}
```
</details>

---

## Event 6: Receiving at USA Roaster

**Organization**: Artisan Coffee Roasters, Oakland CA
**Event**: RECEIVED

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @events/06-roaster-received.json
```

---

## Event 7: Roasting (Green → Roasted Beans)

**Organization**: Artisan Coffee Roasters
**Event**: TRANSFORMED
**Input**: 1000 kg green → **Output**: 850 kg roasted (15% weight loss)

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @events/07-roaster-transformed.json
```

<details>
<summary>View Event JSON</summary>

```json
{
  "@context": ["https://www.w3.org/2018/credentials/v1"],
  "type": ["VerifiableCredential"],
  "issuer": "did:mycelix:org:artisan-coffee-roasters",
  "issuanceDate": "2025-03-08T10:00:00Z",
  "credentialSubject": {
    "eventType": "TRANSFORMED",
    "productId": "COFFEE-ROASTED-BEANS-ETHIOPIAN-YIRGACHEFFE",
    "batchId": "BATCH-2025-ROASTED-001",
    "prevBatchIds": ["BATCH-2025-GREEN-001"],
    "quantity": 850.0,
    "unit": "kg",
    "facility": {
      "id": "ROAST-ARTISAN-OAKLAND",
      "name": "Artisan Coffee Roasters",
      "location": {
        "country": "USA",
        "region": "California",
        "city": "Oakland"
      }
    },
    "timestamp": "2025-03-08T10:00:00Z",
    "metadata": {
      "roastLevel": "Medium (City+)",
      "roastTemperature": "220°C",
      "roastDuration": "12 minutes",
      "roastProfile": "Ethiopian Yirgacheffe Light-Medium",
      "development Time": "2:30 minutes",
      "firstCrack": "9:30 minutes"
    }
  }
}
```
</details>

---

## Event 8: Delivery to Cafe

**Organization**: Downtown Specialty Cafe, San Francisco
**Event**: RECEIVED

```bash
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @events/08-cafe-received.json
```

---

## Verify Lineage

After ingesting all events, retrieve the complete lineage tree:

```bash
# Get lineage for the final roasted batch
curl http://localhost:8080/v1/batches/BATCH-2025-ROASTED-001/lineage | jq

# Or use the CLI
supplychain lineage BATCH-2025-ROASTED-001 --format json

# Visualize as Mermaid diagram
supplychain lineage BATCH-2025-ROASTED-001 --format mermaid
```

**Expected Lineage Tree**:
```
BATCH-2025-HARVEST-001 (PRODUCED)
  ├─ BATCH-2025-HARVEST-001 (CERTIFIED - Organic)
  └─ BATCH-2025-GREEN-001 (TRANSFORMED)
      ├─ BATCH-2025-GREEN-001 (CERTIFIED - Fair Trade)
      ├─ BATCH-2025-GREEN-001 (SHIPPED)
      ├─ BATCH-2025-GREEN-001 (RECEIVED)
      └─ BATCH-2025-ROASTED-001 (TRANSFORMED)
          └─ BATCH-2025-ROASTED-001 (RECEIVED)
```

---

## Data Insights

### Supply Chain Metrics

- **Organizations Involved**: 5 (Farm, Processor, Exporter, Roaster, Cafe)
- **Countries**: 2 (Ethiopia → USA)
- **Total Events**: 8
- **Transformations**: 2 (Cherries→Green, Green→Roasted)
- **Certifications**: 2 (Organic, Fair Trade)
- **Shipments**: 1 (Ethiopia → USA, 33 days transit)
- **Weight Loss**:
  - Processing: 5000kg → 1000kg (80% loss - typical for cherry to green)
  - Roasting: 1000kg → 850kg (15% loss - typical for roasting)
- **Final Yield**: 17% of original cherry weight

### Verifiable Claims

Each event creates a cryptographically signed claim with:
- **Unique Claim ID**: UUID v4
- **Lineage Hash**: SHA-256 hash linking to previous claims
- **VC JWT**: Ed25519-signed Verifiable Credential
- **Tamper-Evident**: Any modification breaks the lineage hash chain

### Use Cases Demonstrated

1. **Origin Verification**: Trace coffee back to specific farm and harvest
2. **Certification Tracking**: Organic and Fair Trade status preserved
3. **Quality Assurance**: Processing and roasting parameters recorded
4. **Shipping Transparency**: Container number, vessel, dates tracked
5. **Sustainability**: Altitude, variety, processing method documented

---

## QR Code Integration (Future Enhancement)

Generate QR codes for customer scanning:

```bash
# Generate QR code for final batch
qrencode -o batch-qr.png "https://verify.mycelix.dev/batch/BATCH-2025-ROASTED-001"
```

Customers can scan to see:
- Farm origin and altitude
- Processing methods
- Certifications
- Shipping route
- Roasting profile

---

## Testing Validation

### Test Invalid Event

```bash
# Try to receive batch that wasn't shipped
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d '{
    "eventType": "RECEIVED",
    "batchId": "BATCH-DOESNT-EXIST",
    "productId": "COFFEE",
    "quantity": 100,
    "unit": "kg"
  }'

# Should return 400 Bad Request
```

### Test Negative Quantity

```bash
# Try to produce negative quantity
curl -X POST http://localhost:8080/v1/events \
  -H 'Content-Type: application/json' \
  -d @events/invalid-negative-quantity.json

# Should return 400 Bad Request with validation error
```

---

## Next Steps

1. **Import to Dashboard**: Load events into web UI for visualization
2. **Export Reports**: Generate PDF supply chain report
3. **Share with Customers**: Provide transparency via QR codes
4. **Integrate IoT**: Add temperature/humidity sensors during shipping
5. **Blockchain Anchoring**: Anchor lineage hashes to public blockchain

---

## Files in This Example

```
04-coffee-supplychain/
├── README.md                            # This file
├── run-coffee-demo.sh                   # Automated demo script
├── events/
│   ├── 01-farm-produced.json           # Farm: Cherry harvest
│   ├── 02-farm-certified-organic.json  # Cert: Organic
│   ├── 03-processor-transformed.json   # Mill: Green beans
│   ├── 04-exporter-certified-fairtrade.json  # Cert: Fair Trade
│   ├── 05-exporter-shipped.json        # Export: Ship to USA
│   ├── 06-roaster-received.json        # Roaster: Receive green
│   ├── 07-roaster-transformed.json     # Roaster: Roast beans
│   └── 08-cafe-received.json           # Cafe: Final delivery
└── visualizations/
    ├── lineage-graph.dot                # GraphViz lineage
    └── supply-chain-map.svg             # Geographic visualization
```

---

## Questions?

- **API Documentation**: See [docs/api-guide.md](../../docs/api-guide.md)
- **Deployment**: See [docs/deployment.md](../../docs/deployment.md)
- **GitHub Issues**: https://github.com/Luminous-Dynamics/mycelix-supplychain/issues

---

**Made with ☕ by Luminous Dynamics**
