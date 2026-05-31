# Core Concepts

This document explains the key innovations in the Mycelix Fabrication hApp.

## Table of Contents

1. [Hyperdimensional Computing (HDC)](#hyperdimensional-computing-hdc)
2. [Proof of Grounded Fabrication (PoGF)](#proof-of-grounded-fabrication-pogf)
3. [Cincinnati Algorithm](#cincinnati-algorithm)
4. [Anticipatory Repair Loop](#anticipatory-repair-loop)
5. [Safety Classification](#safety-classification)
6. [Metabolic Economy Integration](#metabolic-economy-integration)

---

## Hyperdimensional Computing (HDC)

### What is HDC?

Hyperdimensional Computing (also called Vector Symbolic Architectures) represents concepts as very high-dimensional vectors (typically 10,000 dimensions). Unlike traditional embeddings, HDC vectors are:

- **Bipolar**: Each dimension is either -1 or +1
- **Nearly orthogonal**: Any two random vectors are almost certainly unrelated
- **Composable**: Multiple concepts can be combined while remaining searchable

### How We Use HDC

Every design in the Fabrication hApp has an associated **Intent Vector** - a 10,000-dimensional hypervector that encodes its semantic meaning.

```
Example: "bracket for 12mm pipe, weatherproof"

Semantic Bindings:
├── bracket     (Base, weight: 1.0)
├── 12mm        (Dimensional, weight: 0.9)
├── pipe        (Base, weight: 0.8)
└── weatherproof (Modifier, weight: 0.8)

Intent Vector = bracket ⊛ 12mm ⊛ pipe ⊛ weatherproof
```

### Operations

| Operation | Symbol | Purpose |
|-----------|--------|---------|
| **Bundling** | + | Combine multiple concepts (superposition) |
| **Binding** | ⊛ | Associate concepts (role binding) |
| **Permutation** | ρ | Create sequences/positions |
| **Similarity** | cos(a,b) | Compare vectors (cosine similarity) |

### Benefits

1. **Natural Language Search**: Find designs by description, not exact keywords
2. **Cross-Language**: Same concept maps to same region of HDC space
3. **Typo Tolerance**: Similar descriptions yield similar vectors
4. **Generative Design**: AI can compose new intent vectors

---

## Proof of Grounded Fabrication (PoGF)

### Philosophy

PoGF ensures that manufacturing is "grounded" in ecological reality. Every print carries a certificate of its metabolic impact.

### The Formula

```
PoGF = (E × 0.3) + (M × 0.3) + (Q × 0.2) + (L × 0.2)
```

Where:
- **E** (Energy, 30%): Renewable energy fraction
- **M** (Material, 30%): Material circularity score
- **Q** (Quality, 20%): Cincinnati verification score
- **L** (Local, 20%): Local economy participation

### Components

#### Energy Grounding (E)
- Linked to Terra Atlas energy sources
- Measures grid carbon intensity at time of print
- Renewable sources score higher

```typescript
// Example energy scores
Solar/Wind/Hydro: 1.0
Nuclear: 0.9
Grid Mix: 0.4 (varies by time/location)
```

#### Material Circularity (M)
- Tracked via Material Passports
- Considers origin and end-of-life

```typescript
// Example material scores
PostConsumer recycled: 0.9
PostIndustrial recycled: 0.7
Biobased: 0.8
Virgin: 0.2
```

#### Quality Verification (Q)
- Cincinnati Algorithm health score
- Dimensional accuracy
- Visual inspection results

#### Local Participation (L)
- HEARTH funding involvement
- Local printer used
- Community benefit

### MYCELIUM Rewards

Prints with PoGF ≥ 0.6 earn MYCELIUM (CIV) tokens:

```typescript
if (pogScore < 0.6) return 0;

reward = BASE_REWARD × (pogScore + qualityScore) / 2
       + BONUS × (pogScore - 0.6)
```

---

## Cincinnati Algorithm

### Overview

The Cincinnati Algorithm provides real-time "teleomorphic monitoring" during 3D prints. It detects anomalies by comparing current sensor signatures to a learned baseline.

### How It Works

```
1. Learn baseline "healthy print" signature
2. Sample sensors at 1000 Hz during print
3. Compare each sample to baseline
4. Detect anomalies (deviation > threshold)
5. Take automatic action if needed
```

### Sensor Data

```typescript
interface SensorSnapshot {
  hotendTemp: number;      // °C
  bedTemp: number;         // °C
  stepperCurrents: [       // Amps
    xCurrent,
    yCurrent,
    zCurrent,
    extruderCurrent
  ];
  vibrationRms: number;    // g
  filamentTension?: number; // grams
  ambientTemp?: number;    // °C
  humidity?: number;       // %
}
```

### Anomaly Types

| Type | Detection Method | Auto-Action |
|------|------------------|-------------|
| ExtrusionInconsistency | Extruder current variance | Adjust flow |
| TemperatureDeviation | Temp outside bounds | Pause |
| VibrationAnomaly | RMS spike | Reduce speed |
| LayerAdhesionFailure | Z-force pattern | Abort |
| NozzleClog | High pressure, low flow | Pause |
| BedLevelDrift | First layer force | Re-level |

### Actions

```typescript
type CincinnatiAction =
  | { type: 'Continue' }
  | { type: 'AdjustParameters', adjustment: {...} }
  | { type: 'PauseForInspection' }
  | { type: 'AbortPrint', reason: string }
  | { type: 'AlertOperator', message: string };
```

### Configuration

In `dna.yaml`:

```yaml
cincinnati:
  sampling_rate_hz: 1000
  anomaly_threshold: 0.15
  auto_pause_severity: 0.8
  auto_abort_severity: 0.95
```

---

## Anticipatory Repair Loop

### The Problem

Traditional repair is reactive:
1. Part breaks
2. User notices
3. User searches for replacement
4. Order/print part
5. Wait for delivery
6. Install

**Downtime: Days to weeks**

### The Solution

Anticipatory Repair is proactive:

```
Property hApp              Fabrication hApp
     │                           │
     │  Sensor Data              │
     │  (vibration, temp, usage) │
     ▼                           │
┌──────────────┐                 │
│ Digital Twin │                 │
│ Prediction   │─────────────────┤
│ Model        │                 │
└──────────────┘                 │
     │                           │
     │ Failure Prediction        │
     │ (component, probability,  │
     │  estimated date)          │
     ▼                           ▼
┌──────────────┐          ┌──────────────┐
│ Create       │          │ Search for   │
│ Repair       │──────────│ Matching     │
│ Workflow     │          │ Design       │
└──────────────┘          └──────────────┘
     │                           │
     ▼                           ▼
┌──────────────┐          ┌──────────────┐
│ HEARTH       │          │ Match Local  │
│ Auto-Fund    │──────────│ Printer      │
└──────────────┘          └──────────────┘
     │                           │
     ▼                           ▼
┌──────────────┐          ┌──────────────┐
│ Print        │◀─────────│ Start Print  │
│ Completes    │          │ Job          │
└──────────────┘          └──────────────┘
     │
     ▼
┌──────────────┐
│ Part Arrives │
│ BEFORE       │
│ Failure      │
└──────────────┘
```

**Downtime: Zero (part ready before failure)**

### Workflow States

```
Predicted → DesignFound → PrinterMatched → FundingApproved → Printing → ReadyForInstall → Installed
                                                                                              ↑
                                                                            (or Cancelled)───┘
```

### Integration Points

1. **Property hApp**: Provides digital twin sensor data and failure predictions
2. **Knowledge hApp**: Verifies repair design safety
3. **HEARTH**: Auto-funds community repairs
4. **Supply Chain**: Sources materials locally

---

## Safety Classification

### Overview

Every design has a Safety Class determining verification requirements:

| Class | Name | Risk Level | Verification |
|-------|------|------------|--------------|
| 0 | Decorative | None | None |
| 1 | Functional | Low | Self-certification |
| 2 | Load Bearing | Medium | Community verification |
| 3 | Body Contact | Medium | Material certification |
| 4 | Medical | High | Professional verification |
| 5 | Critical | Extreme | Multi-party certification |

### Verification Flow

```
Design Created
     │
     ▼
┌─────────────────┐
│ Safety Class    │
│ Determined      │
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
Class 0-1  Class 2  Class 3-4  Class 5
    │         │        │         │
    │    Community  Material   Multi-
  Self    Review   + Safety   Party
  Cert      │     Cert        │
    │         │        │         │
    └────┬────┴────┬───┴────┬────┘
         ▼         ▼        ▼
    Safety Claims Created
    (Linked to Knowledge hApp)
```

### Epistemic Classification

Safety claims use the Knowledge hApp's E/N/M framework:

- **Empirical (E)**: Measurable, testable facts
  - "Load capacity: 50kg tested"
  - "Dimensional accuracy: ±0.1mm measured"

- **Normative (N)**: Standards and best practices
  - "Meets ISO 3382-1 requirements"
  - "Uses FDA-approved food-safe material"

- **Mythic (M)**: Theoretical/experiential knowledge
  - "Design based on 100-year-old proven pattern"
  - "Follows traditional joinery principles"

---

## Metabolic Economy Integration

The Fabrication hApp is deeply integrated with Mycelix's metabolic economy:

### Terra Atlas (Energy)

```typescript
interface GroundingCertificate {
  terraAtlasEnergyHash: ActionHash;  // Link to renewable source
  energyType: 'Solar' | 'Wind' | 'Hydro' | ...;
  gridCarbonIntensity: number;       // gCO2/kWh at time of print
}
```

### Material Passports

```typescript
interface MaterialPassport {
  origin: 'Virgin' | 'PostIndustrial' | 'PostConsumer' | 'Biobased' | 'UrbanMined';
  recycledContentPercent: number;
  supplyChainHash: ActionHash;  // Link to Supply Chain hApp
  endOfLife: 'MechanicalRecycling' | 'ChemicalRecycling' | 'Biodegradable' | ...;
}
```

### HEARTH (Local Economy)

When a repair workflow is created with HEARTH funding:
1. Community members can contribute to repair fund
2. Funds released automatically when print completes
3. Supports local economy and reduces waste

### MYCELIUM (Reputation)

The CIV reputation system rewards:
- High PoGF scores
- Quality prints (Cincinnati verified)
- Community contributions
- Design sharing

```
Quality × Sustainability = Reputation Growth
```

---

## Next Steps

- [Getting Started](GETTING_STARTED.md) - Set up your environment
- [API Reference](API_REFERENCE.md) - Complete function documentation
- [Integration Guide](INTEGRATION_GUIDE.md) - Connect with other hApps
