# Integration Guide

This guide explains how the Fabrication hApp integrates with other Mycelix civilizational OS components.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MYCELIX CIVILIZATIONAL OS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  KNOWLEDGE  │  │ MARKETPLACE │  │SUPPLY CHAIN │  │  PROPERTY   │        │
│  │             │  │             │  │             │  │             │        │
│  │ - Claims    │  │ - Listings  │  │ - Materials │  │ - Assets    │        │
│  │ - E/N/M     │  │ - Trades    │  │ - Suppliers │  │ - IP        │        │
│  │ - Markets   │  │ - Payments  │  │ - Passports │  │ - Licensing │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┼────────────────┼────────────────┘               │
│                          │                │                                 │
│                    ┌─────▼────────────────▼─────┐                          │
│                    │      FABRICATION hApp      │                          │
│                    │                            │                          │
│                    │  ┌────────┐  ┌──────────┐  │                          │
│                    │  │ BRIDGE │◄─│ DESIGNS  │  │                          │
│                    │  │  ZOME  │  │  + HDC   │  │                          │
│                    │  └────┬───┘  └──────────┘  │                          │
│                    │       │                    │                          │
│                    │  ┌────▼───┐  ┌──────────┐  │                          │
│                    │  │ PRINTS │◄─│ PRINTERS │  │                          │
│                    │  │ + PoGF │  │          │  │                          │
│                    │  └────────┘  └──────────┘  │                          │
│                    └────────────────────────────┘                          │
│                          │                │                                 │
│         ┌────────────────┼────────────────┼────────────────┐               │
│         │                │                │                │               │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐        │
│  │ TERRA ATLAS │  │   HEARTH    │  │  MYCELIUM   │  │  IDENTITY   │        │
│  │             │  │             │  │   (CIV)     │  │             │        │
│  │ - Energy    │  │ - Funding   │  │ - Reputation│  │ - DIDs      │        │
│  │ - Carbon    │  │ - Local Econ│  │ - Rewards   │  │ - Trust     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Knowledge hApp Integration

The Knowledge hApp provides epistemic classification for safety claims.

### Purpose
- Verify safety claims with E/N/M (Empirical/Normative/Mythic) scores
- Create prediction markets for design verification
- Build collective intelligence around design safety

### Integration Points

#### 1. Safety Claims Bridge

```typescript
// Submit a safety claim that bridges to Knowledge hApp
const claim = await client.verification.submitSafetyClaim({
  designHash,
  claimType: { type: 'LoadCapacity', value: '50kg' },
  claimText: 'Tested to support 50kg static load',
  supportingEvidence: ['ipfs://Qm...']
});

// The claim is automatically classified by Knowledge hApp
// Returns E/N/M scores indicating epistemic quality
```

#### 2. Epistemic Score Retrieval

```typescript
// Get aggregated epistemic scores for a design
const score = await client.verification.getEpistemicScore(designHash);

console.log(`Empirical: ${score.empirical}`);   // Measured facts
console.log(`Normative: ${score.normative}`);   // Standards compliance
console.log(`Mythic: ${score.mythic}`);         // Experiential knowledge
console.log(`Overall: ${score.overallConfidence}`);
```

#### 3. Knowledge Query

```typescript
// Query Knowledge hApp for relevant existing claims
const result = await client.verification.queryKnowledge(
  'PETG bracket 50kg load capacity outdoor use'
);

for (const claim of result.claims) {
  console.log(`Claim: ${claim.text}`);
  console.log(`E/N/M: ${claim.epistemic.empirical}/${claim.epistemic.normative}/${claim.epistemic.mythic}`);
}
```

#### 4. Verification Markets

```typescript
// Request verification through Knowledge prediction markets
const marketHash = await client.bridge.requestVerificationMarket(
  designHash,
  'Class3BodyContact'  // Target safety class
);

// Market participants stake on verification outcome
// Result feeds back to Fabrication hApp
```

### Data Flow

```
Fabrication                    Knowledge
    │                              │
    │  Submit Safety Claim         │
    ├─────────────────────────────►│
    │                              │
    │                    Classify E/N/M
    │                              │
    │  Return Epistemic Score      │
    │◄─────────────────────────────┤
    │                              │
    │  Store Linked Claim          │
    │                              │
```

---

## Marketplace hApp Integration

The Marketplace hApp enables trading of designs and print services.

### Purpose
- List designs for sale
- Offer print-on-demand services
- Handle payments and escrow

### Integration Points

#### 1. Design Listings

```typescript
// List a design on the marketplace
const listing = await client.bridge.listOnMarketplace({
  designHash,
  price: 500,  // In HEARTH tokens
  listingType: 'DesignSale'
});

// Listing types:
// - DesignSale: Sell the design file
// - PrintService: Offer to print for others
// - PrintedProduct: Sell finished prints
```

#### 2. Purchase Callbacks

```typescript
// When a design is purchased, Marketplace calls back
// Fabrication hApp handles license activation
await client.bridge.onDesignPurchased(designHash, buyerPubKey);
```

#### 3. License Compliance

```typescript
// Check if usage complies with license
const compliance = await client.bridge.checkLicenseCompliance(
  designHash,
  'commercial'  // personal | commercial | derivative | distribution
);

if (!compliance.compliant) {
  console.log('Restrictions:', compliance.restrictions);
}
```

### Data Flow

```
Fabrication                    Marketplace
    │                              │
    │  List Design                 │
    ├─────────────────────────────►│
    │                              │
    │                    Display Listing
    │                    Handle Payment
    │                              │
    │  Purchase Callback           │
    │◄─────────────────────────────┤
    │                              │
    │  Activate License            │
    │                              │
```

---

## Supply Chain hApp Integration

The Supply Chain hApp tracks material provenance and availability.

### Purpose
- Track material origin for circularity scoring
- Find local suppliers
- Manage Material Passports

### Integration Points

#### 1. Material Passport Linking

```typescript
// Link material to supply chain record
await client.bridge.linkMaterialToSupplier(
  materialHash,
  'did:mycelix:supplier123',
  supplyChainItemHash  // Optional: specific batch
);
```

#### 2. Availability Queries

```typescript
// Check material availability at location
const availability = await client.bridge.queryMaterialAvailability(
  materialHash,
  { geohash: 'gcpvj0', country: 'UK' }
);

console.log(`Available: ${availability.available}`);
console.log(`Suppliers: ${availability.suppliers.length}`);
console.log(`Local options: ${availability.localOptions}`);
```

#### 3. Material Passports for PoGF

```typescript
// Material Passport data feeds into PoGF scoring
interface MaterialPassport {
  materialHash: ActionHash;
  batchId: string;
  origin: 'Virgin' | 'PostIndustrial' | 'PostConsumer' | 'Biobased' | 'UrbanMined';
  recycledContentPercent: number;
  supplyChainHash: ActionHash;  // Link to Supply Chain hApp
  certifications: string[];
  endOfLife: EndOfLifeStrategy;
}
```

### Data Flow

```
Fabrication                    Supply Chain
    │                              │
    │  Query Availability          │
    ├─────────────────────────────►│
    │                              │
    │  Return Suppliers            │
    │◄─────────────────────────────┤
    │                              │
    │  Order Material              │
    ├─────────────────────────────►│
    │                              │
    │  Delivery + Passport         │
    │◄─────────────────────────────┤
    │                              │
    │  Record in PoGF              │
    │                              │
```

---

## Property hApp Integration

The Property hApp manages digital twins and IP/licensing.

### Purpose
- Anticipatory Repair Loop (failure prediction)
- Design IP registration
- License management

### Integration Points

#### 1. Anticipatory Repair Loop

```typescript
// Property hApp sends sensor data and failure prediction
// Fabrication creates repair workflow

const prediction = await client.bridge.createRepairPrediction({
  propertyAssetHash: assetHash,  // Digital twin in Property hApp
  assetModel: 'Bosch GSR 18V-60',
  predictedFailureComponent: 'Battery clip',
  failureProbability: 0.75,
  estimatedFailureDate: Date.now() + 30 * 24 * 60 * 60 * 1000,
  confidenceIntervalDays: 7,
  sensorDataSummary: 'Vibration +40% over 6 months'
});

// If probability > 0.7, workflow is auto-created
// Searches for matching repair design
// Matches to local printer
// Requests HEARTH funding
```

#### 2. Repair Prediction from Sensors

```typescript
// Use Symthaea to predict repairs from sensor data
const prediction = await client.symthaea.predictRepairNeeds({
  propertyAssetHash: assetHash,
  sensorHistory: [
    { timestamp: Date.now() - 86400000, sensorType: 'vibration', value: 0.5, unit: 'g' },
    { timestamp: Date.now(), sensorType: 'vibration', value: 0.7, unit: 'g' }
  ],
  usageHours: 1500
});

console.log(`Predicted: ${prediction.predictedComponent}`);
console.log(`Probability: ${prediction.failureProbability}`);
console.log(`Remaining: ${prediction.estimatedRemainingHours} hours`);
console.log(`Action: ${prediction.recommendedAction}`);
```

#### 3. IP Registration

```typescript
// Register design IP via Property hApp
const ipHash = await client.bridge.registerDesignIp(
  designHash,
  { type: 'CreativeCommons', variant: 'BY_SA' }
);
```

### Anticipatory Repair Flow

```
Property hApp                   Fabrication
    │                              │
    │  Sensor Data                 │
    │  (vibration, temp, usage)    │
    ├─────────────────────────────►│
    │                              │
    │                    Predict Failure
    │                              │
    │                    Search Repair Design
    │                              │
    │                    Match Local Printer
    │                              │
    │  Request HEARTH Funding      │
    │◄─────────────────────────────┤
    │                              │
    │  Funding Approved            │
    ├─────────────────────────────►│
    │                              │
    │                    Start Print
    │                              │
    │  Part Ready                  │
    │◄─────────────────────────────┤
    │                              │
    │  Install (before failure!)   │
    │                              │
```

---

## Terra Atlas Integration

Terra Atlas provides renewable energy grounding for PoGF.

### Purpose
- Link prints to renewable energy sources
- Track grid carbon intensity
- Support energy democracy

### Integration Points

#### 1. Energy Source Linking

```typescript
// Link print to Terra Atlas energy project
const groundingCertificate: GroundingCertificate = {
  certificateId: 'cert_123',
  terraAtlasEnergyHash: terraProjectHash,  // Link to Terra Atlas
  energyType: 'Solar',
  gridCarbonIntensity: 50,  // gCO2/kWh at print time
  materialPassports: [...],
  hearthFundingHash: fundingHash,
  issuedAt: Date.now(),
  issuerSignature: new Uint8Array([...])
};

// Include in print job
await client.prints.createJob({
  designHash,
  printerHash,
  settings: {...},
  groundingCertificate
});
```

#### 2. Energy Score for PoGF

```typescript
// Energy grounding affects PoGF score
// E component = renewable fraction × energy weight (0.3)

// Solar/Wind/Hydro: E = 1.0 × 0.3 = 0.30
// Grid Mix (50% renewable): E = 0.5 × 0.3 = 0.15
// Unknown: E = 0.0 × 0.3 = 0.00
```

---

## HEARTH Integration

HEARTH provides local economy funding for community repairs.

### Purpose
- Fund anticipatory repairs
- Support local manufacturing
- Build community resilience

### Integration Points

#### 1. Auto-Funding for Repairs

```typescript
// When repair workflow is created, HEARTH funding is requested
const workflow = await client.bridge.createRepairWorkflow(predictionHash);

// If HEARTH funding is approved, workflow status updates
if (workflow.hearthFundingHash) {
  console.log('Community funding approved!');
}
```

#### 2. Local Economy Participation (PoGF)

```typescript
// HEARTH participation affects L component of PoGF
// L = 1.0 if HEARTH funded + local printer
// L = 0.5 if only local printer
// L = 0.0 if neither

const localParticipation =
  (hearthFunded ? 0.5 : 0) + (localPrinter ? 0.5 : 0);
```

---

## MYCELIUM (CIV) Integration

MYCELIUM is the reputation system rewarding quality sustainable manufacturing.

### Purpose
- Reward high PoGF scores
- Track manufacturing reputation
- Enable trust-based matching

### Integration Points

#### 1. Earning MYCELIUM

```typescript
// MYCELIUM earned based on PoGF and quality
// Calculated automatically when recording print result

const record = await client.prints.recordResult({
  jobHash,
  result: { type: 'Success' },
  qualityScore: 0.9,
  energyUsedKwh: 0.5
});

// If PoGF >= 0.6, MYCELIUM is awarded
// reward = BASE × (pogScore + qualityScore) / 2 + BONUS × (pogScore - 0.6)
```

#### 2. Reputation-Based Matching

```typescript
// Printer matching can factor in MYCELIUM reputation
const matches = await client.printers.matchDesign(designHash);

// Matches sorted by compatibility + reputation
for (const match of matches) {
  console.log(`Printer: ${match.printerHash}`);
  console.log(`Compatibility: ${match.compatibilityScore}`);
  // High-reputation printers ranked higher
}
```

---

## Identity hApp Integration

The Identity hApp provides DIDs and trust graphs.

### Purpose
- Verify agent identity
- Build trust relationships
- Enable credential verification

### Integration Points

#### 1. Verifier Credentials

```typescript
// Verification requires credentials from Identity hApp
await client.verification.submitVerification({
  designHash,
  verificationType: 'SafetyReview',
  result: { type: 'Passed', confidence: 0.95, notes: '...' },
  evidence: [...],
  credentials: [
    'did:mycelix:verifier123#engineer-cert',
    'did:mycelix:verifier123#safety-reviewer'
  ]
});
```

#### 2. Supplier DIDs

```typescript
// Suppliers identified by DID
await client.bridge.linkMaterialToSupplier(
  materialHash,
  'did:mycelix:supplier:recycled-plastics-uk',
  supplyChainHash
);
```

---

## Event System

The Bridge zome provides an event system for cross-hApp communication.

### Event Types

| Event | Trigger | Listeners |
|-------|---------|-----------|
| DesignPublished | New design created | Marketplace |
| DesignVerified | Verification complete | Marketplace, Property |
| PrintCompleted | Print job done | Property, MYCELIUM |
| PrinterRegistered | New printer | Marketplace |
| MaterialShortage | Low availability | Supply Chain |
| VerificationRequested | Need verification | Knowledge |

### Emitting Events

```typescript
await client.bridge.emitEvent(
  'DesignPublished',
  designHash,
  JSON.stringify({ category: 'Parts', safetyClass: 'Class1Functional' })
);
```

### Subscribing to Events

```typescript
// Poll for recent events
const events = await client.bridge.getRecentEvents(
  Date.now() - 3600000  // Last hour
);

for (const event of events) {
  console.log(`Event: ${event.eventType}`);
  console.log(`Design: ${event.designId}`);
  console.log(`Time: ${event.timestamp}`);
}
```

---

## Best Practices

### 1. Always Include Grounding Certificates

```typescript
// Every print should have a grounding certificate for PoGF
const job = await client.prints.createJob({
  ...settings,
  groundingCertificate: {
    terraAtlasEnergyHash: energySource,
    energyType: 'Solar',
    gridCarbonIntensity: 50,
    materialPassports: passports,
    hearthFundingHash: funding,
    issuedAt: Date.now(),
    issuerSignature: signature
  }
});
```

### 2. Bridge Safety Claims to Knowledge

```typescript
// Safety claims should always be bridged for epistemic validation
const claim = await client.verification.submitSafetyClaim({...});

// Bridge to Knowledge hApp
const knowledgeHash = await client.verification.bridgeToKnowledge(
  claim.signed_action.hashed.hash
);
```

### 3. Use Local Resources When Possible

```typescript
// Optimize for local to maximize PoGF
const optimized = await client.symthaea.optimizeForLocal({
  designHash,
  localMaterials: await getLocalMaterials(location),
  localPrinters: await client.printers.findNearby(location, 50),
  energyPreference: 'Solar'
});
```

### 4. Enable Anticipatory Repair

```typescript
// Register assets with repair manifests
const design = await client.designs.create({
  ...designInput,
  repairManifest: {
    parentProductModel: 'Product XYZ',
    partName: 'Clip',
    failureModes: ['MechanicalWear', 'UvDegradation'],
    replacementInterval: 2000,  // hours
    repairDifficulty: 'Easy'
  }
});
```

---

## Troubleshooting

### Connection Issues

```typescript
try {
  await client.connect();
} catch (error) {
  if (error.message.includes('WebSocket')) {
    console.error('Cannot connect to conductor. Is it running?');
  }
}
```

### Cross-hApp Calls Failing

Ensure the bridge zome is installed and the target hApp is available:

```typescript
// Check if Knowledge hApp is accessible
try {
  const score = await client.verification.getEpistemicScore(designHash);
} catch (error) {
  console.error('Knowledge hApp not available:', error);
}
```

### PoGF Score Not Calculating

Ensure all components are provided:

```typescript
// All four components required for full PoGF
const pogScore = client.calculatePogScore({
  energyRenewableFraction: 0.8,   // E: Terra Atlas link
  materialCircularity: 0.6,       // M: Material Passport
  qualityVerified: 0.9,           // Q: Cincinnati report
  localParticipation: 1.0         // L: HEARTH + local
});
```

---

## Next Steps

- [Core Concepts](CONCEPTS.md) - Deep dive into HDC, PoGF, Cincinnati
- [API Reference](API_REFERENCE.md) - Complete function documentation
- [Getting Started](GETTING_STARTED.md) - Set up your environment
