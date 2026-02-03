# Proof of Grounded Fabrication (PoGF) Specification

> Manufacturing tied to ecological health

Proof of Grounded Fabrication (PoGF) is the metabolic accountability layer that ensures every print contributes to regenerative civilization rather than extractive consumption.

## Overview

PoGF creates a direct link between:

- **Energy sources** - Where does the electricity come from?
- **Material origins** - What's the lifecycle of each gram?
- **Quality verification** - Does it meet specifications?
- **Local economy** - Does it support community resilience?

## The Formula

```
PoG_score = (E_renewable × 0.3) + (M_circular × 0.3) + (Q_verified × 0.2) + (L_local × 0.2)
```

### Component Breakdown

| Component | Weight | Range | Source |
|-----------|--------|-------|--------|
| E_renewable | 30% | 0.0-1.0 | Terra Atlas integration |
| M_circular | 30% | 0.0-1.0 | Material Passport |
| Q_verified | 20% | 0.0-1.0 | Cincinnati Algorithm |
| L_local | 20% | 0.0-1.0 | HEARTH funding |

## E_renewable: Energy Score

### Calculation

```typescript
function calculateEnergyScore(certificate: GroundingCertificate): number {
  const energyWeights: Record<EnergyType, number> = {
    Solar: 1.0,
    Wind: 1.0,
    Hydro: 0.95,      // Ecological impact considerations
    Geothermal: 0.95,
    Nuclear: 0.85,    // Low carbon, but other concerns
    GridMix: 0.3,     // Varies by region
    Unknown: 0.2,
  };

  const baseScore = energyWeights[certificate.energyType];

  // Adjust for grid carbon intensity
  // Lower intensity = higher score
  const carbonFactor = Math.max(0, 1 - (certificate.gridCarbonIntensity / 500));

  return baseScore * 0.7 + carbonFactor * 0.3;
}
```

### Terra Atlas Integration

When available, link to verified renewable energy:

```typescript
// In print job
groundingCertificate: {
  certificateId: 'cert-xxx',
  terraAtlasEnergyHash: 'QmTerraAtlas...', // Link to energy project
  energyType: 'Solar',
  gridCarbonIntensity: 45, // gCO2/kWh at time of print
  issuedAt: Date.now(),
}
```

### Verification Methods

1. **Smart meter integration** - Real-time energy source verification
2. **Terra Atlas certificate** - Linked renewable investment
3. **Timestamp correlation** - Grid mix at time of print
4. **Geographic verification** - Regional energy mix data

## M_circular: Material Circularity Score

### Calculation

```typescript
function calculateCircularityScore(passports: MaterialPassport[]): number {
  if (passports.length === 0) return 0.2; // Default for unknown

  const originWeights: Record<MaterialOrigin, number> = {
    UrbanMined: 1.0,     // Recycled from local waste stream
    PostConsumer: 0.9,    // Consumer recycled content
    PostIndustrial: 0.8,  // Industrial recycled content
    Biobased: 0.7,        // Renewable biomass source
    Virgin: 0.3,          // New raw material
  };

  const eolWeights: Record<EndOfLifeStrategy, number> = {
    MechanicalRecycling: 1.0,
    ChemicalRecycling: 0.9,
    IndustrialCompost: 0.85,
    Biodegradable: 0.8,
    Downcycle: 0.5,
    Landfill: 0.1,
  };

  let totalScore = 0;

  for (const passport of passports) {
    const originScore = originWeights[passport.origin];
    const eolScore = eolWeights[passport.endOfLife];
    const recycledBonus = passport.recycledContentPercent / 100;

    // Weight: 40% origin, 30% end-of-life, 30% recycled content
    const passportScore =
      originScore * 0.4 +
      eolScore * 0.3 +
      recycledBonus * 0.3;

    totalScore += passportScore;
  }

  return totalScore / passports.length;
}
```

### Material Passport

Every material used carries a passport:

```typescript
interface MaterialPassport {
  materialHash: string;           // Reference to material entry
  batchId: string;                // Specific batch/lot
  origin: MaterialOrigin;         // Where it came from
  recycledContentPercent: number; // 0-100
  supplyChainHash?: string;       // Link to Supply Chain hApp
  certifications: string[];       // Food-safe, etc.
  endOfLife: EndOfLifeStrategy;   // What happens after use
}
```

### Urban Mining

Highest score for materials reclaimed from local waste:

```typescript
// Urban mined material example
{
  materialHash: 'mat-xxx',
  batchId: 'SF-2026-001',
  origin: 'UrbanMined',
  recycledContentPercent: 100,
  supplyChainHash: 'QmLocalRecycler...',
  certifications: ['LocalRecycler-Certified'],
  endOfLife: 'MechanicalRecycling',
}
```

## Q_verified: Quality Verification Score

### Calculation

```typescript
function calculateQualityScore(
  assessment: QualityAssessment,
  cincinnatiReport?: CincinnatiReport
): number {
  // Base quality from physical assessment
  const baseQuality = (
    assessment.dimensionalAccuracy +
    assessment.surfaceQuality +
    assessment.structuralIntegrity
  ) / 3;

  if (!cincinnatiReport) {
    return baseQuality * 0.7; // Penalty for no monitoring
  }

  // Cincinnati monitoring bonus
  const monitoringScore = cincinnatiReport.overallHealthScore;

  // Weight: 60% physical, 40% monitoring
  return baseQuality * 0.6 + monitoringScore * 0.4;
}
```

### Quality Assessment

Manual assessment by printer operator:

```typescript
interface QualityAssessment {
  dimensionalAccuracy: number;   // 0-1: Matches design dimensions
  surfaceQuality: number;        // 0-1: Layer lines, finish
  structuralIntegrity: number;   // 0-1: No delamination, voids
}
```

### Cincinnati Integration

For Class 3+ safety prints, Cincinnati monitoring is required:

```typescript
// Cincinnati report contributes to quality score
cincinnatiReport: {
  sessionId: 'cin-xxx',
  overallHealthScore: 0.94,
  anomaliesDetected: 2,
  anomalyEvents: [...],
  layerByLayerScores: [0.95, 0.96, 0.93, ...],
}
```

## L_local: Local Economy Score

### Calculation

```typescript
function calculateLocalScore(
  certificate: GroundingCertificate,
  printerLocation?: GeoLocation
): number {
  let score = 0.2; // Base score

  // HEARTH funding participation
  if (certificate.hearthFundingHash) {
    score += 0.5;
  }

  // Local material sourcing
  const localMaterials = certificate.materialPassports.filter(
    mp => mp.origin === 'UrbanMined' || mp.supplyChainHash
  );
  if (localMaterials.length > 0) {
    score += 0.2 * (localMaterials.length / certificate.materialPassports.length);
  }

  // Printer is in same region as requester
  if (printerLocation?.region) {
    score += 0.1;
  }

  return Math.min(1.0, score);
}
```

### HEARTH Funding

Community-funded prints get highest local score:

```typescript
// When HEARTH funds the print
groundingCertificate: {
  hearthFundingHash: 'QmHearthProject...',
  // ... other fields
}
```

### Local Economy Indicators

- **HEARTH funding** - Community investment in the print
- **Local materials** - Supply Chain hApp verified local sourcing
- **Local printer** - Same region as design requester
- **Local designer** - Design created in community

## Score Thresholds

### Certification Levels

| Score | Level | Meaning |
|-------|-------|---------|
| 0.9+ | Platinum | Exemplary regenerative manufacturing |
| 0.7-0.9 | Gold | Strong ecological accountability |
| 0.5-0.7 | Silver | Above average sustainability |
| 0.3-0.5 | Bronze | Basic compliance |
| <0.3 | None | Below minimum standards |

### Minimum Requirements

| Safety Class | Min PoGF | Rationale |
|--------------|----------|-----------|
| Class 0-1 | None | Decorative/basic functional |
| Class 2 | 0.3 | Structural needs accountability |
| Class 3 | 0.5 | Body contact requires care |
| Class 4-5 | 0.7 | Critical applications need high standards |

## MYCELIUM Rewards

PoGF directly determines MYCELIUM (CIV) reputation earned:

```typescript
function calculateMyceliumReward(pogScore: number, printValue: number): number {
  // Base reward scales with PoGF
  const baseReward = Math.round(pogScore * 100);

  // Bonus tiers
  if (pogScore >= 0.9) {
    return baseReward * 1.5; // 50% bonus for platinum
  } else if (pogScore >= 0.7) {
    return baseReward * 1.2; // 20% bonus for gold
  }

  return baseReward;
}
```

### Reputation Benefits

Higher MYCELIUM unlocks:

- **Priority matching** - First in queue for popular printers
- **Reduced verification** - Trusted actors need less oversight
- **Governance votes** - Influence protocol decisions
- **Access levels** - Higher safety class permissions

## Implementation

### Grounding Certificate Creation

```typescript
// Generate certificate before print starts
async function generateGroundingCertificate(
  energySource: EnergySource,
  materials: MaterialPassport[],
  hearthProject?: string
): Promise<GroundingCertificate> {
  return {
    certificateId: generateId(),
    terraAtlasEnergyHash: await queryTerraAtlas(energySource),
    energyType: energySource.type,
    gridCarbonIntensity: await getGridCarbonIntensity(energySource.location),
    materialPassports: materials,
    hearthFundingHash: hearthProject,
    issuedAt: Date.now(),
    issuerSignature: await signCertificate(),
  };
}
```

### Score Calculation

```typescript
// Calculate final PoGF score
function calculatePogScore(
  certificate: GroundingCertificate,
  assessment: QualityAssessment,
  cincinnatiReport?: CincinnatiReport
): number {
  const energyScore = calculateEnergyScore(certificate);
  const circularityScore = calculateCircularityScore(certificate.materialPassports);
  const qualityScore = calculateQualityScore(assessment, cincinnatiReport);
  const localScore = calculateLocalScore(certificate);

  return (
    energyScore * 0.3 +
    circularityScore * 0.3 +
    qualityScore * 0.2 +
    localScore * 0.2
  );
}
```

### Print Record Storage

```typescript
// Store with print record
printRecord: {
  jobHash: 'job-xxx',
  result: 'Success',
  pogScore: 0.78,
  energyUsedKwh: 0.45,
  carbonOffsetKg: 0.02,
  materialCircularity: 0.85,
  myceliumEarned: 78,
  groundingCertificate: { ... },
  qualityAssessment: { ... },
  cincinnatiReport: { ... },
}
```

## Verification & Auditing

### On-Chain Verification

All PoGF components are verifiable:

1. **Energy hash** - Query Terra Atlas for renewable certificate
2. **Material hashes** - Query Supply Chain hApp for passports
3. **Quality data** - Cincinnati telemetry stored on DHT
4. **HEARTH hash** - Query HEARTH for funding record

### Audit Trail

Every score component links to source:

```typescript
pogAuditTrail: {
  pogScore: 0.78,
  breakdown: {
    energy: { score: 0.85, source: 'terraAtlas:QmXxx...' },
    circularity: { score: 0.72, source: 'supplyChain:QmYyy...' },
    quality: { score: 0.89, source: 'cincinnati:cin-xxx' },
    local: { score: 0.60, source: 'hearth:QmZzz...' },
  },
  calculatedAt: Date.now(),
  calculatorVersion: '1.0.0',
}
```

### Dispute Resolution

If PoGF score is challenged:

1. **Retrieve audit trail** - Get all source hashes
2. **Verify each component** - Cross-reference with source hApps
3. **Knowledge hApp claim** - Submit dispute as epistemic claim
4. **Governance vote** - If unresolved, community decides

## Future Extensions

### Carbon Credits

Link PoGF to carbon markets:

```typescript
// High PoGF prints could generate carbon credits
if (pogScore >= 0.9 && carbonOffsetKg > 0) {
  await generateCarbonCredit({
    amount: carbonOffsetKg,
    source: 'fabrication',
    certificate: printRecord.groundingCertificate,
  });
}
```

### Regional Multipliers

Adjust weights for regional context:

```typescript
// Areas with dirty grid get bonus for renewables
const regionalMultipliers = {
  'high-carbon-grid': { energy: 1.2, circularity: 1.0 },
  'low-recycling': { energy: 1.0, circularity: 1.2 },
  'rural': { local: 1.3 },
};
```

### Supply Chain Depth

Track full material provenance:

```typescript
// Deep supply chain verification
materialPassport: {
  origin: 'UrbanMined',
  supplyChainDepth: 3,
  fullChain: [
    'QmLocalCollector...',
    'QmRegionalProcessor...',
    'QmFilamentMaker...',
  ],
}
```

---

*Every print is an economic vote. PoGF ensures that vote counts toward regeneration.*
