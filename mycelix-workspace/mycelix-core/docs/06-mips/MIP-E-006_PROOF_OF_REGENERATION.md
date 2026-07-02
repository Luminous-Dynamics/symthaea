# MIP-E-006: Proof of Regeneration (PoR)

**Title**: Proof of Regeneration - Ecological Impact Verification
**Author**: Mycelix Ecological Working Group
**Status**: DRAFT
**Type**: Standards Track (Economic)
**Category**: Ecological Value Mechanics
**Created**: 2026-01-04
**Requires**: MIP-E-002, MIP-E-003
**Supersedes**: None

---

## Abstract

Proof of Grounding (MIP-E-003) measures **capacity** - what infrastructure exists. This proposal introduces **Proof of Regeneration (PoR)** - measuring **impact** - what ecological improvement results from that infrastructure.

MIP-E-006 introduces:
1. **Regeneration Domains**: Carbon, Biodiversity, Water, Soil, Air
2. **Baseline-Delta Verification**: Before/after measurement of ecological state
3. **Oracle Networks**: Multi-source verification of regeneration claims
4. **Regeneration Multipliers**: Rewards for net-positive ecological impact
5. **Harm Accounting**: Penalties for degradation

The formula becomes: **Grounding Score = PoG × PoR**

Infrastructure that regenerates ecosystems receives dramatically higher weight than infrastructure that merely exists.

---

## Motivation

### The Limits of Capacity Measurement

PoG verifies:
- "This solar panel produces 5kW"
- "This storage node replicates 10TB"
- "This compute cluster processes X FLOPS"

But it doesn't verify:
- "This solar farm is on degraded land being restored"
- "This data center uses 100% captured rainwater"
- "This node operator is sequestering carbon"

### Why Impact Matters

Two solar installations, both producing 100kW:
1. **Installation A**: Built on clearcut forest, concrete foundation, grid-tied with coal backup
2. **Installation B**: Built on brownfield, sheep grazing underneath, battery storage, watershed restoration integrated

PoG treats these identically. PoR differentiates them.

### Alignment with Mycelix Values

The Spore Constitution's Axiomatic Value 2 is **Pan-Sentient Flourishing** - "unconditional care, intrinsic value, holistic well-being." This cannot be achieved by ignoring ecological impact.

The Commons Charter establishes stewardship of shared resources. Regeneration is the practice of that stewardship.

---

## Specification

### 1. Regeneration Domains

#### 1.1 Carbon Domain

```rust
pub struct CarbonRegeneration {
    /// Baseline carbon state (tCO2e)
    pub baseline: CarbonBaseline,
    /// Current carbon state
    pub current: CarbonMeasurement,
    /// Net sequestration (positive) or emission (negative)
    pub net_sequestration_tco2e: f64,
    /// Verification method
    pub verification: CarbonVerification,
    /// Measurement period
    pub period: MeasurementPeriod,
}

pub struct CarbonBaseline {
    /// Timestamp of baseline
    pub timestamp: u64,
    /// Soil carbon (tCO2e/hectare)
    pub soil_carbon: f64,
    /// Biomass carbon (tCO2e/hectare)
    pub biomass_carbon: f64,
    /// Area (hectares)
    pub area_hectares: f64,
    /// Methodology used
    pub methodology: CarbonMethodology,
}

pub enum CarbonVerification {
    /// Satellite imagery + ML analysis
    Satellite {
        provider: String,
        confidence: f64,
        imagery_hash: String,
    },
    /// Ground-truth soil sampling
    SoilSampling {
        lab: String,
        sample_count: u32,
        report_hash: String,
    },
    /// Biochar/enhanced weathering receipts
    CarbonRemovalReceipts {
        provider: String,
        tonnes: f64,
        registry_id: String,
    },
    /// Composite (multiple methods)
    Composite(Vec<CarbonVerification>),
}
```

#### 1.2 Biodiversity Domain

```rust
pub struct BiodiversityRegeneration {
    /// Baseline biodiversity index
    pub baseline: BiodiversityBaseline,
    /// Current biodiversity state
    pub current: BiodiversityMeasurement,
    /// Net biodiversity change
    pub net_change: BiodiversityDelta,
    /// Verification method
    pub verification: BiodiversityVerification,
}

pub struct BiodiversityBaseline {
    /// Species richness (count)
    pub species_richness: u32,
    /// Shannon diversity index
    pub shannon_index: f64,
    /// Functional diversity score
    pub functional_diversity: f64,
    /// Habitat connectivity score
    pub connectivity: f64,
    /// Native vs invasive ratio
    pub native_ratio: f64,
    /// Timestamp
    pub timestamp: u64,
}

pub enum BiodiversityVerification {
    /// Acoustic monitoring (bird/insect/amphibian calls)
    Acoustic {
        device_ids: Vec<String>,
        analysis_model: String,
        duration_days: u32,
    },
    /// Camera trap surveys
    CameraTrap {
        trap_count: u32,
        species_detected: Vec<SpeciesRecord>,
    },
    /// eDNA sampling
    EnvironmentalDna {
        sample_locations: Vec<GeoCoordinate>,
        lab: String,
        species_detected: Vec<String>,
    },
    /// Expert survey attestation
    ExpertSurvey {
        experts: Vec<String>,
        methodology: String,
        report_hash: String,
    },
    /// Composite
    Composite(Vec<BiodiversityVerification>),
}
```

#### 1.3 Water Domain

```rust
pub struct WaterRegeneration {
    /// Baseline water state
    pub baseline: WaterBaseline,
    /// Current water state
    pub current: WaterMeasurement,
    /// Net water improvement
    pub net_improvement: WaterDelta,
    /// Verification
    pub verification: WaterVerification,
}

pub struct WaterBaseline {
    /// Groundwater level (meters below surface)
    pub groundwater_level: f64,
    /// Surface water quality index (0-100)
    pub surface_quality_index: f64,
    /// Watershed retention capacity (cubic meters)
    pub retention_capacity: f64,
    /// Riparian buffer extent (hectares)
    pub riparian_buffer: f64,
    /// Timestamp
    pub timestamp: u64,
}

pub enum WaterVerification {
    /// Hydrological sensors
    HydrologicalSensors {
        sensor_ids: Vec<String>,
        readings: Vec<SensorReading>,
    },
    /// Lab water quality tests
    LabTests {
        lab: String,
        parameters: Vec<WaterParameter>,
        results_hash: String,
    },
    /// Satellite wetland analysis
    SatelliteWetland {
        provider: String,
        wetland_extent_change: f64,
    },
    /// Community water monitoring
    CommunityMonitoring {
        monitors: Vec<String>,
        reports: Vec<String>,
    },
}
```

#### 1.4 Soil Domain

```rust
pub struct SoilRegeneration {
    /// Baseline soil state
    pub baseline: SoilBaseline,
    /// Current soil state
    pub current: SoilMeasurement,
    /// Net soil improvement
    pub net_improvement: SoilDelta,
    /// Verification
    pub verification: SoilVerification,
}

pub struct SoilBaseline {
    /// Organic matter percentage
    pub organic_matter_pct: f64,
    /// Microbial biomass (mg C/kg soil)
    pub microbial_biomass: f64,
    /// Aggregate stability (%)
    pub aggregate_stability: f64,
    /// Infiltration rate (mm/hour)
    pub infiltration_rate: f64,
    /// Compaction (bulk density g/cm3)
    pub bulk_density: f64,
    /// Timestamp
    pub timestamp: u64,
}

pub enum SoilVerification {
    /// Lab soil testing
    LabTesting {
        lab: String,
        sample_count: u32,
        parameters: Vec<SoilParameter>,
        results_hash: String,
    },
    /// In-field sensors
    InFieldSensors {
        sensor_ids: Vec<String>,
        continuous_data: bool,
    },
    /// Regenerative practice attestation
    PracticeAttestation {
        practices: Vec<RegenerativePractice>,
        attestors: Vec<String>,
    },
}

pub enum RegenerativePractice {
    NoTill,
    CoverCropping,
    CompostApplication,
    Silvopasture,
    ManagedGrazing,
    Agroforestry,
    Biochar,
    Mulching,
}
```

#### 1.5 Air Domain

```rust
pub struct AirRegeneration {
    /// Baseline air quality
    pub baseline: AirBaseline,
    /// Current air quality
    pub current: AirMeasurement,
    /// Net improvement
    pub net_improvement: AirDelta,
    /// Verification
    pub verification: AirVerification,
}

pub struct AirBaseline {
    /// PM2.5 (ug/m3)
    pub pm25: f64,
    /// PM10 (ug/m3)
    pub pm10: f64,
    /// NO2 (ppb)
    pub no2: f64,
    /// O3 (ppb)
    pub ozone: f64,
    /// VOCs (ppb)
    pub vocs: f64,
    /// Timestamp
    pub timestamp: u64,
}

pub enum AirVerification {
    /// Air quality sensors
    AirSensors {
        sensor_ids: Vec<String>,
        calibration_date: u64,
    },
    /// Government monitoring data
    GovernmentData {
        agency: String,
        station_id: String,
    },
    /// Tree canopy analysis (air filtration proxy)
    CanopyAnalysis {
        imagery_provider: String,
        canopy_change_pct: f64,
    },
}
```

### 2. Composite Regeneration Score

#### 2.1 Domain Weights

```rust
pub struct RegenerationWeights {
    /// Carbon weight (climate priority)
    pub carbon: f64,      // Default: 0.30
    /// Biodiversity weight
    pub biodiversity: f64, // Default: 0.25
    /// Water weight
    pub water: f64,        // Default: 0.20
    /// Soil weight
    pub soil: f64,         // Default: 0.15
    /// Air weight
    pub air: f64,          // Default: 0.10
}
```

#### 2.2 Score Calculation

```rust
pub struct RegenerationScore {
    /// Individual domain scores (-1.0 to +1.0)
    pub carbon_score: f64,
    pub biodiversity_score: f64,
    pub water_score: f64,
    pub soil_score: f64,
    pub air_score: f64,
    /// Weighted composite (-1.0 to +1.0)
    pub composite_score: f64,
    /// Converted to multiplier (0.5 to 2.0)
    pub multiplier: f64,
    /// Verification confidence (0.0 to 1.0)
    pub confidence: f64,
}

impl RegenerationScore {
    pub fn calculate_multiplier(&self) -> f64 {
        // Map composite score to multiplier range
        // -1.0 (degradation) -> 0.5x
        //  0.0 (neutral)     -> 1.0x
        // +1.0 (regeneration) -> 2.0x
        1.0 + (self.composite_score * 0.5 * self.confidence)
    }
}
```

#### 2.3 Integration with PoG

```rust
pub fn calculate_grounding_score(
    pog: &ProofOfGrounding,
    por: &RegenerationScore,
) -> f64 {
    let base_pog = pog.composite_score(); // From MIP-E-003
    let por_multiplier = por.calculate_multiplier();

    // Final grounding = capacity × impact
    base_pog * por_multiplier
}
```

**Example**:
- Node A: PoG = 0.8 (good infrastructure), PoR = 1.5 (regenerating) → 1.2 Grounding
- Node B: PoG = 0.8 (same infrastructure), PoR = 0.7 (degrading) → 0.56 Grounding

### 3. Oracle Networks

#### 3.1 Regeneration Oracle Structure

```rust
pub struct RegenerationOracle {
    /// Oracle ID
    pub oracle_id: OracleId,
    /// Domains covered
    pub domains: Vec<RegenerationDomain>,
    /// Geographic coverage
    pub coverage: GeographicCoverage,
    /// Data sources
    pub sources: Vec<DataSource>,
    /// Update frequency
    pub update_frequency_epochs: u32,
    /// Dispute resolution
    pub dispute_mechanism: DisputeMechanism,
}

pub enum DataSource {
    /// Satellite imagery provider
    Satellite { provider: String, resolution_m: f64 },
    /// IoT sensor network
    SensorNetwork { network_id: String, sensor_count: u32 },
    /// Scientific institution
    Institution { name: String, accreditation: String },
    /// Community monitoring program
    CommunityProgram { program_id: String, participant_count: u32 },
    /// Government agency
    GovernmentAgency { agency: String, data_type: String },
}
```

#### 3.2 Multi-Source Verification

For claims to be accepted, they require:
- **Minimum 2 independent sources** for same domain
- **Cross-domain consistency** (e.g., improving soil should correlate with carbon)
- **Temporal consistency** (no sudden impossible jumps)
- **Geographic plausibility** (claims must match location capabilities)

```rust
pub struct VerificationResult {
    /// Is the claim accepted?
    pub accepted: bool,
    /// Confidence level
    pub confidence: f64,
    /// Sources that verified
    pub verifying_sources: Vec<DataSource>,
    /// Discrepancies found
    pub discrepancies: Vec<Discrepancy>,
    /// Final adjusted score
    pub adjusted_score: f64,
}
```

### 4. Harm Accounting

#### 4.1 Degradation Tracking

Infrastructure that causes ecological harm receives negative PoR:

```rust
pub struct HarmAssessment {
    /// Domain affected
    pub domain: RegenerationDomain,
    /// Type of harm
    pub harm_type: HarmType,
    /// Magnitude (-1.0 to 0.0)
    pub magnitude: f64,
    /// Reversibility
    pub reversibility: Reversibility,
    /// Required remediation
    pub remediation: Option<RemediationRequirement>,
}

pub enum HarmType {
    /// Habitat destruction
    HabitatDestruction,
    /// Pollution (water, air, soil)
    Pollution { pollutant: String },
    /// Carbon emission
    CarbonEmission { tco2e: f64 },
    /// Water depletion
    WaterDepletion,
    /// Soil degradation
    SoilDegradation,
    /// Biodiversity loss
    BiodiversityLoss { species_affected: Vec<String> },
}

pub enum Reversibility {
    /// Easily reversed (months)
    HighlyReversible,
    /// Reversible with effort (years)
    Reversible,
    /// Difficult to reverse (decades)
    DifficultlyReversible,
    /// Permanent or near-permanent
    Irreversible,
}
```

#### 4.2 Remediation Requirements

Harmful activities may continue if:
1. **Remediation bond** posted (SAP locked against future restoration)
2. **Remediation plan** approved by bioregional DAO
3. **Progress tracking** enabled via oracle network

```rust
pub struct RemediationRequirement {
    /// Required bond (SAP)
    pub bond_amount: u64,
    /// Restoration timeline (epochs)
    pub timeline_epochs: u32,
    /// Success metrics
    pub success_metrics: Vec<Metric>,
    /// Monitoring requirements
    pub monitoring: MonitoringRequirement,
    /// Penalty if not met
    pub penalty: RemediationPenalty,
}
```

### 5. Regeneration Incentives

#### 5.1 Validator Weight Boost

Validators with positive PoR receive weight boost in consensus:

```
Validator_Weight = Base_Weight × (1 + PoR_Bonus)
```

Where `PoR_Bonus = max(0, (RegenerationMultiplier - 1.0))`

A validator with 1.5x regeneration multiplier gets 50% bonus weight.

#### 5.2 Fee Reduction

Transactions from accounts with positive PoR receive fee discounts:

| PoR Multiplier | Fee Discount |
|----------------|--------------|
| < 1.0 (harm) | +25% surcharge |
| 1.0 - 1.2 | No change |
| 1.2 - 1.5 | 10% discount |
| 1.5 - 1.8 | 20% discount |
| > 1.8 | 30% discount |

#### 5.3 HEARTH Access

HEARTH pools can require minimum PoR for membership:

```rust
pub struct HearthEcologicalRequirements {
    /// Minimum PoR multiplier to join
    pub min_por_multiplier: f64,
    /// Domains that must be positive
    pub required_positive_domains: Vec<RegenerationDomain>,
    /// Allowed harm domains (with remediation)
    pub allowed_harm_with_remediation: Vec<RegenerationDomain>,
}
```

---

## Rationale

### Why Separate PoG and PoR?

PoG measures *what exists*. PoR measures *what it does*. Combining them into a single metric would obscure both:
- High-capacity, high-harm infrastructure would score medium
- Low-capacity, high-regeneration efforts would be undervalued

Multiplication preserves both signals: `Grounding = Capacity × Impact`

### Why Include Harm Accounting?

Without harm accounting, the protocol is blind to negative externalities. A data center that pollutes a river would have the same PoG as a solar-powered one. Including harm creates:
1. **Price signals** for ecological responsibility
2. **Incentives** for remediation
3. **Accountability** for negative impacts

### Why Multi-Source Verification?

Single-source oracles are vulnerable to:
- Corruption
- Technical failure
- Gaming

Multi-source verification with cross-domain consistency makes fraud extremely difficult.

### Why Domain Weights?

Different domains have different urgency:
- **Carbon** (0.30): Climate crisis is existential
- **Biodiversity** (0.25): Sixth extinction is accelerating
- **Water** (0.20): Freshwater scarcity affects billions
- **Soil** (0.15): Foundation of terrestrial life
- **Air** (0.10): Most areas have improving air quality

Weights are protocol parameters adjustable via MIP process.

---

## Backwards Compatibility

### PoG Integration

Existing PoG implementations continue unchanged. PoR is an optional multiplier. Nodes without PoR data default to 1.0x multiplier (neutral).

### Validator Requirements

Validators are not required to have PoR. However, validators with positive PoR gain competitive advantage through weight boost.

### Gradual Rollout

1. **Phase 1**: PoR as optional metadata (no protocol effects)
2. **Phase 2**: PoR affects validator weight
3. **Phase 3**: PoR affects fee structure
4. **Phase 4**: HEARTH pools can require PoR

---

## Security Considerations

### 1. Oracle Manipulation

Multi-source verification prevents single-point manipulation. Cross-domain consistency checks catch fabricated data.

### 2. Baseline Gaming

Baselines are timestamped and immutable. Artificially degraded baselines (to show false improvement) are detectable through historical satellite imagery.

### 3. Greenwashing

Regeneration claims without corresponding PoG are suspicious. A pure services company claiming carbon sequestration would face scrutiny.

### 4. Oracle Collusion

Oracle operators must stake collateral. False attestations result in slashing. Geographic distribution of oracles prevents regional collusion.

---

## Implementation Status

| Component | SDK Module | Status |
|-----------|------------|--------|
| Carbon Regeneration | `por::carbon` | Planned |
| Biodiversity Regeneration | `por::biodiversity` | Planned |
| Water Regeneration | `por::water` | Planned |
| Soil Regeneration | `por::soil` | Planned |
| Air Regeneration | `por::air` | Planned |
| Composite Score | `por::composite` | Planned |
| Oracle Network | `por::oracle` | Planned |
| Harm Accounting | `por::harm` | Planned |

---

## References

- [MIP-E-003: Proof of Grounding](./MIP-E-003_PROOF_OF_GROUNDING.md)
- [Commons Charter v1.0](../architecture/THE%20COMMONS%20CHARTER%20(v1.0).md)
- [Spore Constitution v0.24 - Axiomatic Value 2](../architecture/THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)

---

## Copyright

This document is licensed under Apache 2.0.

---

*"We do not inherit the Earth from our ancestors; we borrow it from our children. Proof of Regeneration ensures we return it better than we found it."*
