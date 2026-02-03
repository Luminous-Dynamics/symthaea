# Mycelix Fabrication hApp

> The Physical Hands of the Civilizational OS

Mycelix Fabrication is a decentralized manufacturing commons that transforms bits into atoms through parametric intelligence, metabolic accountability, and anticipatory repair.

## Vision

In a world of planned obsolescence and centralized manufacturing, Fabrication enables:

- **Community-owned production** - Local printers serving local needs
- **Circular economy** - Material passports track every gram from origin to end-of-life
- **Anticipatory repair** - Replace parts before they fail, extending product life
- **Metabolic accountability** - Every print tied to energy sources and ecological health

## Core Concepts

### HDC-Encoded Parametric Designs

Designs aren't just static files - they're semantic vectors in a 10,000-dimensional hyperdimensional space. This enables:

- **Natural language search** - "Find a bracket for 12mm pipe, weatherproof"
- **Generative customization** - Automatic parameter adjustment for local materials
- **Lateral binding** - Combine concepts: `bracket + 12mm + outdoor = new design`

### Proof of Grounded Fabrication (PoGF)

Every print carries metabolic accountability:

```
PoG_score = (E_renewable × 0.3) + (M_circular × 0.3) + (Q_verified × 0.2) + (L_local × 0.2)
```

Where:
- **E_renewable** - Renewable energy percentage (Terra Atlas integration)
- **M_circular** - Material circularity (recycled content, end-of-life strategy)
- **Q_verified** - Quality verification (Cincinnati Algorithm + safety certification)
- **L_local** - Local economy participation (HEARTH funding)

### Cincinnati Algorithm

Real-time teleomorphic monitoring ensures print quality:

- **1000Hz sensor sampling** - Temperature, vibration, current, tension
- **Anomaly detection** - Extrusion issues, layer adhesion, nozzle clogs
- **Adaptive response** - Auto-adjust parameters or pause for inspection
- **Quality assurance** - Layer-by-layer scoring for Class 3+ safety prints

### Anticipatory Repair Loop

The autopoietic cycle that ends planned obsolescence:

```
Property hApp (Digital Twin)
    ↓ Wear Prediction
Fabrication (Find/Create Repair Design)
    ↓ Printer Matching
Supply Chain (Material Sourcing)
    ↓ HEARTH Funding
Local Printer (PoGF Verified)
    ↓ Installation
Property hApp (Update Digital Twin)
```

## Architecture

### Zome Structure

| Zome | Purpose |
|------|---------|
| `designs` | Parametric design CRUD with HDC vectors |
| `printers` | Printer registry and capability matching |
| `prints` | Print job lifecycle with PoGF scoring |
| `materials` | Material specifications and passports |
| `verification` | Safety classification and Knowledge bridge |
| `bridge` | Cross-hApp communication hub |
| `cincinnati` | Quality monitoring telemetry |

### Integration Map

```
┌─────────────────────────────────────────────────────────────────┐
│                      FABRICATION hApp                           │
│                                                                 │
│   designs ←→ prints ←→ printers ←→ materials                   │
│       ↓         ↓                     ↓                        │
│   verification  ↓                     ↓                        │
│       ↓         ↓                     ↓                        │
│       └─────────┴─────────┬───────────┘                        │
│                           ↓                                     │
│                        bridge                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
   Knowledge           Marketplace        Supply Chain
   (Safety Claims)    (Design Trade)    (Material Source)
        ↓                   ↓                   ↓
   Property            HEARTH            Terra Atlas
   (Digital Twin)   (Local Economy)   (Renewable Energy)
```

## Quick Start

### Install SDK

```bash
npm install @mycelix/sdk
```

### Create a Design

```typescript
import { getFabricationService } from '@mycelix/sdk/integrations/fabrication';

const fab = getFabricationService();

const design = fab.createDesign({
  title: 'Replacement Battery Clip',
  description: 'Universal clip for power tool batteries',
  category: 'Repair',
  safetyClass: 'Class2LoadBearing',
  license: 'OpenHardware',
  repairManifest: {
    parentProductModel: 'DeWalt DCD771',
    partName: 'Battery Clip',
    failureModes: ['MechanicalWear', 'ImpactDamage'],
    repairDifficulty: 'Easy',
  },
});
```

### Register a Printer

```typescript
const printer = fab.registerPrinter({
  name: 'Community Makerspace Prusa',
  printerType: 'FDM',
  capabilities: {
    buildVolume: { x: 250, y: 210, z: 210 },
    layerHeights: [0.1, 0.15, 0.2, 0.3],
    nozzleDiameters: [0.4],
    heatedBed: true,
    enclosure: false,
    maxTempHotend: 285,
    maxTempBed: 100,
    features: ['auto-leveling', 'power-recovery'],
  },
  materialsAvailable: ['PLA', 'PETG', 'TPU'],
  location: {
    geohash: '9q8yy',
    city: 'San Francisco',
    country: 'USA',
  },
  rates: {
    baseFee: 5,
    perHour: 2,
    perGram: 0.05,
  },
});
```

### Create a Print Job

```typescript
// Match design to available printers
const matches = fab.matchDesignToPrinters(design.id);

// Create job with best match
const job = fab.createPrintJob({
  designId: design.id,
  printerId: matches[0].printer.id,
  settings: {
    layerHeight: 0.2,
    infillPercent: 20,
    material: 'PETG',
    supports: false,
    raft: false,
  },
  groundingRequest: {
    requireRenewable: true,
    requireRecycledMaterial: false,
    targetPogScore: 0.7,
  },
});

// Job lifecycle
fab.acceptPrintJob(job.id);
fab.startPrint(job.id);

// Complete with quality assessment
const record = fab.completePrint(
  job.id,
  'Success',
  {
    dimensionalAccuracy: 0.95,
    surfaceQuality: 0.90,
    structuralIntegrity: 0.92,
  },
  45 // grams used
);

console.log(`PoGF Score: ${record.pogScore}`);
console.log(`MYCELIUM Earned: ${record.myceliumEarned}`);
```

## Safety Classification

| Class | Name | Requirements | Examples |
|-------|------|--------------|----------|
| 0 | Decorative | Self-certification | Art, models, figurines |
| 1 | Functional | Basic testing | Tool holders, brackets |
| 2 | Load Bearing | Stress testing | Furniture parts, mounts |
| 3 | Body Contact | Material certification | Utensils, handles |
| 4 | Medical | Professional review | Assistive devices |
| 5 | Critical | Multi-party certification | Safety equipment |

## MYCELIUM (CIV) Reputation

High PoGF scores earn MYCELIUM reputation tokens:

- **Renewable energy** - Use solar/wind powered printing
- **Recycled materials** - Post-consumer/post-industrial content
- **Quality prints** - Pass Cincinnati monitoring
- **Local economy** - HEARTH funding participation

Reputation unlocks:
- Priority matching with high-reputation printers
- Access to advanced safety class designs
- Voting power in governance decisions
- Reduced verification requirements

## Documentation

- [Design Lifecycle](./DESIGN_LIFECYCLE.md) - From concept to verified production
- [PoGF Specification](./POGF_SPECIFICATION.md) - Metabolic accountability in detail
- [Cincinnati Algorithm](./CINCINNATI_ALGORITHM.md) - Quality monitoring deep dive
- [API Reference](./API_REFERENCE.md) - Complete zome function reference

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:

- Submitting designs
- Registering printers
- Improving verification
- Extending the SDK

## License

MIT License - see [LICENSE](../LICENSE)

---

*Building the physical infrastructure of a regenerative civilization, one print at a time.*
