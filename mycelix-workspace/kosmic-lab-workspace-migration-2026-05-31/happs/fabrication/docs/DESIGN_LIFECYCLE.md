# Design Lifecycle

> From concept to verified production

This document describes the complete lifecycle of a design in the Mycelix Fabrication ecosystem.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DESIGN LIFECYCLE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. CREATION          2. CLASSIFICATION      3. VERIFICATION               │
│   ┌──────────┐         ┌──────────────┐       ┌───────────────┐            │
│   │ Designer │────────▶│ Safety Class │──────▶│ Knowledge hApp│            │
│   │ submits  │         │ Assignment   │       │ Epistemic     │            │
│   └──────────┘         └──────────────┘       │ Verification  │            │
│                                               └───────┬───────┘            │
│                                                       │                     │
│   6. EVOLUTION         5. PRODUCTION          4. PUBLICATION               │
│   ┌──────────┐         ┌──────────────┐       ┌───────┴───────┐            │
│   │ Forks &  │◀────────│ Print Jobs   │◀──────│ Marketplace   │            │
│   │ Versions │         │ Quality Data │       │ Listing       │            │
│   └──────────┘         └──────────────┘       └───────────────┘            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Creation

### 1.1 Design Submission

A designer creates a new entry with:

```typescript
const design = fab.createDesign({
  title: 'Ergonomic Tool Handle',
  description: 'Universal replacement handle for garden tools',
  category: 'Repair',
  safetyClass: 'Class2LoadBearing',
  license: 'OpenHardware',
  files: [
    {
      filename: 'handle.stl',
      format: 'STL',
      ipfsCid: 'QmXxx...',
      sizeBytes: 245000,
      checksumSha256: 'abc123...',
    },
    {
      filename: 'handle.scad',
      format: 'SCAD',
      ipfsCid: 'QmYyy...',
      sizeBytes: 3500,
      checksumSha256: 'def456...',
    },
  ],
});
```

### 1.2 HDC Vector Generation

The Symthaea zome automatically generates an HDC hypervector:

1. **Parse title & description** - Extract semantic concepts
2. **Bind category** - Category vector (Repair = specific encoding)
3. **Bind parameters** - If parametric, encode each parameter
4. **Bind material hints** - Material compatibility vectors
5. **Composite** - Lateral binding of all components

Result: 10,000-dimensional bipolar vector for semantic search.

### 1.3 Parametric Schema (Optional)

For generative designs:

```typescript
parametricSchema: {
  engine: 'OpenSCAD',
  sourceTemplate: 'QmTemplate...',
  parameters: [
    {
      name: 'grip_diameter',
      paramType: 'Length',
      defaultValue: 32,
      minValue: 25,
      maxValue: 45,
      unit: 'mm',
      hdcBinding: 'diameter:grip',
    },
    {
      name: 'length',
      paramType: 'Length',
      defaultValue: 120,
      minValue: 80,
      maxValue: 200,
      unit: 'mm',
    },
    {
      name: 'texture',
      paramType: { Enum: ['smooth', 'ribbed', 'knurled'] },
      defaultValue: 'ribbed',
    },
  ],
  constraints: ['grip_diameter < length / 2'],
  autoGenerate: true,
}
```

### 1.4 Repair Manifest (For Repair Designs)

Linking to parent products for anticipatory repair:

```typescript
repairManifest: {
  parentProductModel: 'Fiskars Pro Shovel',
  partName: 'D-Handle Grip',
  failureModes: ['MechanicalWear', 'UvDegradation'],
  replacementIntervalHours: 500,
  repairDifficulty: 'Easy',
}
```

## Stage 2: Classification

### 2.1 Safety Class Assignment

The designer selects an initial safety class:

| Class | Name | Meaning |
|-------|------|---------|
| 0 | Decorative | No safety concerns |
| 1 | Functional | Basic mechanical use |
| 2 | Load Bearing | Structural requirements |
| 3 | Body Contact | Material safety critical |
| 4 | Medical | Professional oversight required |
| 5 | Critical | Multi-party certification |

### 2.2 Epistemic Initial Scoring

Default scores set based on safety class:

```typescript
// Class 0-1: Self-certification acceptable
epistemic: {
  manufacturability: 0.5,
  safety: 0.5,
  usability: 0.5,
}

// Class 2-3: Requires community verification
epistemic: {
  manufacturability: 0.3,  // Lower until verified
  safety: 0.3,
  usability: 0.3,
}

// Class 4-5: Requires professional verification
epistemic: {
  manufacturability: 0.1,
  safety: 0.1,
  usability: 0.1,
}
```

### 2.3 Material Compatibility

System validates material declarations:

```typescript
// Design declares compatible materials
materialsCompatible: ['PETG', 'ABS', 'ASA']

// System checks against safety class
if (safetyClass === 'Class3BodyContact') {
  // Verify food-safe certifications
  requireMaterialCertification('FoodSafe');
}
```

## Stage 3: Verification

### 3.1 Knowledge hApp Integration

Safety claims bridge to the Knowledge hApp:

```typescript
// Submit claim to Knowledge
const claim = await knowledge.submitClaim({
  designHash: design.id,
  claimText: 'Supports 50kg static load',
  empiricalLevel: 'E2PrivateVerify', // Community verifiable
  normativeLevel: 'N1Communal',
  materialityLevel: 'M2Persistent',
});
```

### 3.2 Verification Types

| Type | Method | Required For |
|------|--------|--------------|
| Structural Analysis | FEA simulation | Class 2+ |
| Material Compatibility | Lab testing | Class 3+ |
| Printability Test | Sample prints | All |
| Safety Review | Expert review | Class 4+ |
| Food Safe Cert | Lab certification | Food contact |
| Medical Cert | Professional board | Class 4 |
| Community Review | Crowd verification | Class 0-2 |

### 3.3 Verification Workflow

```typescript
// Request verification
const request = await fab.requestVerification({
  designHash: design.id,
  targetSafetyClass: 'Class2LoadBearing',
  verificationType: 'StructuralAnalysis',
  bounty: 50, // MYCELIUM tokens
  deadline: Date.now() + 7 * 24 * 60 * 60 * 1000,
});

// Verifier submits result
const verification = await fab.submitVerification({
  designHash: design.id,
  verificationType: 'StructuralAnalysis',
  result: {
    Passed: {
      confidence: 0.92,
      notes: 'FEA shows safety factor of 3.2 under 50kg load',
    },
  },
  evidence: ['QmFEAReport...', 'QmTestPhotos...'],
  verifierCredentials: ['MechanicalEngineer', 'PE-License-CA-12345'],
});
```

### 3.4 Epistemic Score Update

After verification:

```typescript
// Updated based on verification results
epistemic: {
  manufacturability: 0.85,  // Verified printable
  safety: 0.92,             // FEA passed
  usability: 0.78,          // Community feedback
}
```

## Stage 4: Publication

### 4.1 Marketplace Listing

Verified designs can be listed:

```typescript
// List on marketplace
await fab.listDesignOnMarketplace(design.id, {
  listingType: 'DesignSale',
  price: 25, // MYCELIUM
});

// Or offer print service
await fab.listDesignOnMarketplace(design.id, {
  listingType: 'PrintService',
  price: 15, // Per print
});
```

### 4.2 Discovery

Designs are discoverable via:

- **Category browsing** - Tools, Repair, Medical, etc.
- **Semantic search** - Natural language queries
- **Repair matching** - By product model and part name
- **Printer compatibility** - What can print this?

## Stage 5: Production

### 5.1 Print Job Creation

See [PoGF Specification](./POGF_SPECIFICATION.md) for full details.

### 5.2 Quality Data Collection

Every print contributes:

```typescript
printRecord: {
  result: 'Success',
  qualityScore: 0.92,
  pogScore: 0.78,
  cincinnatiReport: { ... },
  issues: [],
}
```

### 5.3 Statistical Aggregation

Design accumulates production statistics:

```typescript
designStats: {
  totalPrints: 847,
  successRate: 0.94,
  averageQuality: 0.89,
  averagePogScore: 0.72,
  commonIssues: ['Warping (3%)', 'Stringing (2%)'],
  recommendedSettings: {
    layerHeight: 0.2,
    infillPercent: 25,
    material: 'PETG',
  },
}
```

## Stage 6: Evolution

### 6.1 Version History

All changes tracked:

```typescript
versionHistory: [
  { hash: 'v1...', date: '2026-01-01', note: 'Initial release' },
  { hash: 'v2...', date: '2026-01-15', note: 'Improved grip texture' },
  { hash: 'v3...', date: '2026-02-01', note: 'Added knurled option' },
]
```

### 6.2 Forking

Community improvements:

```typescript
const fork = fab.forkDesign(design.id, {
  title: 'Ergonomic Tool Handle - Left Hand',
  description: 'Left-handed variant with reversed ergonomics',
});
```

### 6.3 Reputation Feedback

Designer reputation grows with:

- Successful prints
- Positive quality scores
- Community forks
- Verification passes

## Lifecycle Metrics

### Design Health Score

Composite metric:

```typescript
healthScore = (
  verificationScore * 0.3 +
  productionSuccessRate * 0.3 +
  averageQuality * 0.2 +
  communityEngagement * 0.2
)
```

### Time-to-Production

Track efficiency:

```typescript
timeToProduction: {
  created: '2026-01-01T00:00:00Z',
  firstVerification: '2026-01-03T12:00:00Z',
  marketplaceListed: '2026-01-04T00:00:00Z',
  firstPrint: '2026-01-04T14:00:00Z',
  daysToProd: 3.5,
}
```

## Best Practices

### For Designers

1. **Start with lower safety class** - Easier to verify, upgrade later
2. **Include parametric schema** - Enables customization
3. **Add repair manifest** - Links to anticipatory repair system
4. **Respond to verification feedback** - Iterate quickly
5. **Document recommended settings** - Help printers succeed

### For Verifiers

1. **Provide detailed evidence** - Photos, reports, test data
2. **Be specific in notes** - Help designers improve
3. **Declare credentials** - Build trust in your verifications
4. **Follow up on conditionals** - Track conditional passes

### For Printers

1. **Report issues accurately** - Helps design improvement
2. **Follow recommended settings** - Higher success rate
3. **Complete Cincinnati monitoring** - Required for Class 3+
4. **Document post-processing** - Part of quality record

---

*Every design tells a story. From first concept to thousandth print, the lifecycle captures that journey.*
