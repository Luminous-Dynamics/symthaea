# Getting Started

This guide will help you set up and start using the Mycelix Fabrication hApp.

## Prerequisites

- **Holochain**: v0.4.x installed
- **Node.js**: v18+ (for TypeScript SDK)
- **Rust**: 1.70+ (for building zomes)

## Installation

### Option 1: Using the TypeScript SDK

```bash
# Install the SDK
npm install @mycelix/fabrication-sdk

# Or with yarn
yarn add @mycelix/fabrication-sdk

# Or with pnpm
pnpm add @mycelix/fabrication-sdk
```

### Option 2: Building from Source

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/mycelix
cd mycelix-workspace/happs/fabrication

# Build all zomes
cargo build --release

# Package the DNA
hc dna pack dna/

# Package the hApp
hc app pack .
```

## Quick Start

### 1. Connect to the Fabrication hApp

```typescript
import { FabricationClient } from '@mycelix/fabrication-sdk';

const client = new FabricationClient({
  url: 'ws://localhost:8888',      // Conductor URL
  installedAppId: 'fabrication'    // App ID
});

await client.connect();
console.log('Connected!', client.appInfo);
```

### 2. Create Your First Design

```typescript
const design = await client.designs.create({
  title: 'Universal Pipe Bracket',
  description: 'Adjustable bracket for standard pipes, weatherproof',
  category: 'Parts',
  files: [{
    filename: 'bracket.stl',
    format: 'STL',
    ipfsCid: 'QmXxx...',  // Upload to IPFS first
    sizeBytes: 45000,
    checksumSha256: 'abc123...'
  }],
  license: { type: 'CreativeCommons', variant: 'BY_SA' },
  safetyClass: 'Class1Functional'
});

console.log('Design created:', design.signed_action.hashed.hash);
```

### 3. Register a Printer

```typescript
const printer = await client.printers.register({
  name: 'Workshop Printer 1',
  printerType: 'FDM',
  capabilities: {
    buildVolume: { x: 220, y: 220, z: 250 },
    layerHeights: [0.1, 0.15, 0.2, 0.3],
    nozzleDiameters: [0.4, 0.6],
    heatedBed: true,
    enclosure: false,
    multiMaterial: null,
    maxTempHotend: 300,
    maxTempBed: 110,
    features: ['AutoLeveling', 'FilamentRunout']
  },
  materialsAvailable: ['PLA', 'PETG', 'ABS'],
  location: {
    geohash: 'gcpvj0',  // London
    city: 'London',
    country: 'UK'
  }
});

console.log('Printer registered:', printer.signed_action.hashed.hash);
```

### 4. Search for Designs

```typescript
// Using natural language search
const results = await client.searchDesigns('bracket for 12mm pipe');

for (const result of results) {
  console.log(`Found: ${result.designHash}`);
  console.log(`Similarity: ${result.similarityScore}`);
  console.log(`Matched concepts: ${result.matchedBindings.join(', ')}`);
}

// Or search by category
const toolDesigns = await client.designs.getByCategory('Tools');
```

### 5. Create a Print Job

```typescript
const designHash = results[0].designHash;
const printerHash = printer.signed_action.hashed.hash;

const job = await client.prints.createJob({
  designHash,
  printerHash,
  settings: {
    layerHeight: 0.2,
    infillPercent: 20,
    material: 'PETG',
    supports: false,
    raft: false,
    temperatures: {
      hotend: 240,
      bed: 70
    }
  }
});

console.log('Print job created:', job.signed_action.hashed.hash);
```

### 6. Record Print Result with PoGF

```typescript
const record = await client.prints.recordResult({
  jobHash: job.signed_action.hashed.hash,
  result: { type: 'Success' },
  qualityScore: 0.9,
  energyUsedKwh: 0.5,
  photos: ['ipfs://Qm...'],
  notes: 'Perfect print, no issues',
  issues: []
});

// The PoGF score is calculated automatically
// based on energy source, materials, quality, and local participation
```

### 7. Calculate PoGF Score (Client-Side)

```typescript
const pogScore = client.calculatePogScore({
  energyRenewableFraction: 0.8,  // 80% solar
  materialCircularity: 0.6,      // 60% recycled content
  qualityVerified: 0.9,          // Cincinnati Algorithm score
  localParticipation: 1.0        // Local printer, HEARTH funded
});

console.log(`PoGF Score: ${pogScore}`);  // 0.76

// Estimate MYCELIUM reward
const mycelium = client.estimateMyceliumReward(pogScore, 0.9);
console.log(`MYCELIUM Earned: ${mycelium}`);  // ~108 tokens
```

## Working with HDC (Symthaea)

### Generate Intent Vectors

```typescript
// Generate an intent vector from natural language
const intent = await client.symthaea.generateIntentVector({
  description: 'heavy-duty bracket for outdoor use, 25mm pipe',
  language: 'en'
});

console.log('Vector hash:', intent.vectorHash);
console.log('Bindings:', intent.bindings);
// [
//   { concept: 'bracket', role: 'Base', weight: 1.0 },
//   { concept: 'heavy-duty', role: 'Modifier', weight: 0.9 },
//   { concept: '25mm', role: 'Dimensional', weight: 0.9 },
//   { concept: 'outdoor', role: 'Modifier', weight: 0.8 }
// ]
```

### Lateral Binding

```typescript
// Start with a base intent
const base = await client.symthaea.generateIntentVector({
  description: 'bracket'
});

// Add modifiers via lateral binding
const combined = await client.symthaea.lateralBind({
  baseIntentHash: base.record.signed_action.hashed.hash,
  modifierDescriptions: ['25mm pipe', 'weatherproof', 'heavy-duty']
});

// combined.vectorHash is semantically similar to all components
// but distinct from each
```

### Semantic Search

```typescript
// Search for similar designs
const matches = await client.symthaea.semanticSearch({
  intentHash: combined.record.signed_action.hashed.hash,
  threshold: 0.7,
  limit: 10
});

for (const match of matches) {
  console.log(`Design: ${match.designHash}`);
  console.log(`Similarity: ${match.similarityScore}`);
}
```

## Verification and Safety

### Submit a Safety Claim

```typescript
const claim = await client.verification.submitSafetyClaim({
  designHash,
  claimType: { type: 'LoadCapacity', value: '50kg tested' },
  claimText: 'This bracket has been tested to support 50kg static load',
  supportingEvidence: ['Test report: ipfs://Qm...']
});
```

### Get Epistemic Score

```typescript
const epistemic = await client.verification.getEpistemicScore(designHash);

console.log('Epistemic Score:');
console.log(`  Empirical: ${epistemic.empirical}`);   // Measured facts
console.log(`  Normative: ${epistemic.normative}`);   // Standards compliance
console.log(`  Mythic: ${epistemic.mythic}`);         // Experiential knowledge
console.log(`  Overall: ${epistemic.overallConfidence}`);
```

## Cross-hApp Integration

### Anticipatory Repair Loop

```typescript
// Create a repair prediction (from Property hApp data)
const prediction = await client.bridge.createRepairPrediction({
  propertyAssetHash: assetHash,  // Digital twin in Property hApp
  assetModel: 'Bosch GSR 18V-60',
  predictedFailureComponent: 'Battery clip',
  failureProbability: 0.75,
  estimatedFailureDate: Date.now() + 30 * 24 * 60 * 60 * 1000,  // 30 days
  confidenceIntervalDays: 7,
  sensorDataSummary: 'Vibration increased 40% over 6 months'
});

// A repair workflow is automatically created if probability > 0.7
```

### Marketplace Integration

```typescript
// List a design on the marketplace
const listing = await client.bridge.listOnMarketplace({
  designHash,
  price: 500,  // In HEARTH tokens
  listingType: 'DesignSale'
});
```

## Next Steps

- [Core Concepts](CONCEPTS.md) - Deep dive into HDC, PoGF, Cincinnati
- [API Reference](API_REFERENCE.md) - Complete function documentation
- [Integration Guide](INTEGRATION_GUIDE.md) - Connect with other Mycelix hApps
