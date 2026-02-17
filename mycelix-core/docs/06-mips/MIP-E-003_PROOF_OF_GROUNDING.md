# MIP-E-003: Proof of Grounding (PoG) Integration

**Title**: Proof of Grounding: DePIN Integration for Physical Infrastructure Binding
**Author**: Tristan Stoltz (tstoltz), Claude AI (Co-Author)
**Status**: DRAFT
**Type**: Standards Track (Economic)
**Category**: Economic Charter Extension
**Created**: 2026-01-04
**Requires**: MIP-E-002 (Metabolic Bridge Amendment)
**Supersedes**: None

---

## Abstract

This proposal specifies **Proof of Grounding (PoG)**—a mechanism binding Mycelix economic value to verified physical infrastructure. PoG addresses the fundamental disconnect between digital currencies and real-world utility by requiring network participants to demonstrate contribution of tangible resources (energy, storage, compute, bandwidth).

**Core Principle**: *A currency grounded in physical infrastructure cannot collapse below the utility value of that infrastructure.*

---

## Motivation

### The Abstraction Problem

Digital currencies suffer from a fundamental vulnerability: their value is purely consensual. When network belief falters, value can evaporate instantly. This creates boom-bust cycles antithetical to stable economic coordination.

### The DePIN Solution

Decentralized Physical Infrastructure Networks (DePIN) have demonstrated that digital tokens can represent claims on real-world utility:
- **Filecoin**: Storage capacity
- **Helium**: Wireless coverage
- **Render**: GPU compute
- **Energy Web**: Renewable energy

### Mycelix Integration Opportunity

By requiring physical infrastructure contribution as component of Proof of Contribution, Mycelix:

1. **Creates Value Floor**: Network value cannot drop below aggregate infrastructure utility
2. **Aligns Incentives**: Validators are infrastructure operators, not speculators
3. **Enables Real Economy**: SAP/FLOW tokens become claims on actual resources
4. **Builds Resilience**: Distributed physical infrastructure is harder to attack than pure software

---

## Specification

### Article I: PoG Fundamentals

#### Section 1. Definition

**Proof of Grounding (PoG)** is a verifiable attestation that a network participant contributes physical infrastructure resources to the Mycelix ecosystem.

#### Section 2. Infrastructure Categories

| Category | Description | Minimum Contribution | Verification Method |
|----------|-------------|---------------------|---------------------|
| **Energy** | Renewable energy generation | 1 kWh/day average | Hardware attestation + oracle |
| **Storage** | Decentralized data storage | 100 GB persistent | Proof of replication |
| **Compute** | Processing capacity | 1 TFLOP available | TEE attestation |
| **Bandwidth** | Network connectivity | 10 Mbps sustained | Peer measurement |

#### Section 3. PoG Weight in PoC

Per MIP-E-002, Proof of Contribution comprises:
```
PoC_Score = (Behavioral × 0.60) + (Physical_Grounding × 0.40)
```

PoG constitutes the full 40% physical grounding component.

---

### Article II: Energy Infrastructure

#### Section 1. Eligible Generation

The following renewable energy sources qualify for PoG:

| Source | PoG Multiplier | Attestation Requirement |
|--------|----------------|------------------------|
| Solar PV | 1.4x | Smart inverter + GPS |
| Wind | 1.4x | Turbine telemetry + GPS |
| Micro-hydro | 1.3x | Flow sensor + GPS |
| Biogas | 1.2x | Production meter |
| Grid-tied Renewable | 1.1x | Utility API integration |

#### Section 2. Energy Oracle

**2.1 Oracle Architecture**:

```rust
pub struct EnergyOracle {
    /// Verified energy sources registered to this oracle
    sources: HashMap<SourceId, EnergySource>,

    /// Rolling 30-day production history
    production_history: RingBuffer<DailyProduction>,

    /// External verification integrations
    verifiers: Vec<Box<dyn EnergyVerifier>>,
}

pub struct EnergySource {
    pub source_id: SourceId,
    pub owner_did: String,
    pub source_type: EnergyType,
    pub location: GeoCoordinate,
    pub rated_capacity_kw: f64,
    pub hardware_attestation: Option<HardwareAttestation>,
    pub verification_level: VerificationLevel,
}

#[derive(Clone, Copy)]
pub enum VerificationLevel {
    SelfReported,           // 0.3x weight
    CommunityWitnessed,     // 0.6x weight
    OracleVerified,         // 0.8x weight
    HardwareAttested,       // 1.0x weight
}
```

**2.2 Verification Methods**:

| Method | Weight | Requirements |
|--------|--------|--------------|
| Self-Reported | 30% | Member attestation only |
| Community Witnessed | 60% | 3+ community member confirmations |
| Oracle Verified | 80% | Third-party energy API integration |
| Hardware Attested | 100% | Cryptographic hardware attestation (TPM/TEE) |

#### Section 3. Energy Credit Calculation

```rust
pub fn calculate_energy_pog(
    source: &EnergySource,
    production_kwh: f64,
    period_days: u32,
) -> PogScore {
    let daily_average = production_kwh / period_days as f64;

    // Minimum threshold: 1 kWh/day
    if daily_average < 1.0 {
        return PogScore::zero();
    }

    // Logarithmic scaling prevents mega-farm dominance
    let scaled_production = (daily_average.ln() + 1.0).min(5.0);

    // Apply source multiplier and verification weight
    let base_score = scaled_production
        * source.source_type.multiplier()
        * source.verification_level.weight();

    PogScore::new(base_score.min(1.0))
}
```

---

### Article III: Storage Infrastructure

#### Section 1. Eligible Storage

| Storage Type | PoG Multiplier | Protocol |
|--------------|----------------|----------|
| IPFS pinning | 1.3x | IPFS/Filecoin |
| Holochain DHT | 1.4x | Native (bonus) |
| Arweave | 1.2x | Permaweb |
| Custom S3-compatible | 1.1x | Community verification |

#### Section 2. Proof of Replication

Storage PoG requires periodic proof of data availability:

```rust
pub struct StorageAttestation {
    pub provider_did: String,
    pub storage_commitment_gb: u64,
    pub replication_proofs: Vec<ReplicationProof>,
    pub last_challenge_response: DateTime<Utc>,
    pub availability_score: f64,  // 0.0 - 1.0
}

pub struct ReplicationProof {
    pub challenge_id: ChallengeId,
    pub merkle_root: Hash,
    pub response_time_ms: u32,
    pub verified_at: DateTime<Utc>,
}
```

**Challenge Protocol**:
1. Random challenge issued every 24 hours
2. Provider must respond within 5 minutes
3. Response includes Merkle proof of random data segment
4. Availability score = (successful_challenges / total_challenges)

#### Section 3. Storage Credit Calculation

```rust
pub fn calculate_storage_pog(
    attestation: &StorageAttestation,
) -> PogScore {
    // Minimum: 100 GB
    if attestation.storage_commitment_gb < 100 {
        return PogScore::zero();
    }

    // Logarithmic scaling (1 TB = ~3x score of 100 GB)
    let storage_factor = (attestation.storage_commitment_gb as f64 / 100.0).ln() + 1.0;

    // Availability multiplier (99.9% = 1.0, 99% = 0.9, etc.)
    let availability_multiplier = attestation.availability_score;

    // Storage type multiplier
    let type_multiplier = 1.3; // Holochain DHT native

    let score = (storage_factor * availability_multiplier * type_multiplier)
        .min(1.0);

    PogScore::new(score)
}
```

---

### Article IV: Compute Infrastructure

#### Section 1. Eligible Compute

| Compute Type | PoG Multiplier | Verification |
|--------------|----------------|--------------|
| CPU cycles | 1.1x | Benchmark + TEE |
| GPU compute | 1.3x | Hardware attestation |
| TEE enclaves | 1.4x | Remote attestation |
| FPGA | 1.2x | Bitstream verification |

#### Section 2. TEE Attestation

Trusted Execution Environments provide cryptographic proof of compute contribution:

```rust
pub struct ComputeAttestation {
    pub provider_did: String,
    pub tee_type: TeeType,
    pub remote_attestation: RemoteAttestation,
    pub benchmark_results: BenchmarkResults,
    pub availability_hours_per_week: u32,
}

pub enum TeeType {
    IntelSGX,
    ArmTrustZone,
    AmdSEV,
    RiscVKeystone,
}

pub struct BenchmarkResults {
    pub flops_available: u64,
    pub memory_gb: u32,
    pub measured_at: DateTime<Utc>,
    pub verifier_signature: Signature,
}
```

#### Section 3. Compute Credit Calculation

```rust
pub fn calculate_compute_pog(
    attestation: &ComputeAttestation,
) -> PogScore {
    // Minimum: 1 TFLOP available
    let tflops = attestation.benchmark_results.flops_available as f64 / 1e12;
    if tflops < 1.0 {
        return PogScore::zero();
    }

    // Logarithmic scaling
    let compute_factor = tflops.ln() + 1.0;

    // Availability factor (40 hrs/week = 1.0)
    let availability = (attestation.availability_hours_per_week as f64 / 40.0).min(1.0);

    // TEE bonus
    let tee_multiplier = match attestation.tee_type {
        TeeType::IntelSGX => 1.4,
        TeeType::ArmTrustZone => 1.3,
        TeeType::AmdSEV => 1.4,
        TeeType::RiscVKeystone => 1.2,
    };

    let score = (compute_factor * availability * tee_multiplier * 0.2).min(1.0);

    PogScore::new(score)
}
```

---

### Article V: Bandwidth Infrastructure

#### Section 1. Eligible Bandwidth

| Connection Type | PoG Multiplier | Measurement |
|-----------------|----------------|-------------|
| Residential fiber | 1.1x | Peer speed test |
| Business dedicated | 1.2x | SLA verification |
| Community mesh | 1.4x | Network mapping |
| Satellite uplink | 1.3x | Latency measurement |

#### Section 2. Peer Measurement Protocol

Bandwidth verification through multi-peer attestation:

```rust
pub struct BandwidthAttestation {
    pub provider_did: String,
    pub measurements: Vec<PeerMeasurement>,
    pub sustained_mbps: f64,
    pub latency_ms_avg: u32,
    pub uptime_percentage: f64,
}

pub struct PeerMeasurement {
    pub peer_did: String,
    pub download_mbps: f64,
    pub upload_mbps: f64,
    pub latency_ms: u32,
    pub measured_at: DateTime<Utc>,
}

impl BandwidthAttestation {
    pub fn aggregate_measurements(&self) -> (f64, f64) {
        // Median of peer measurements (Sybil-resistant)
        let mut downloads: Vec<f64> = self.measurements.iter()
            .map(|m| m.download_mbps)
            .collect();
        downloads.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_download = downloads[downloads.len() / 2];
        let median_upload = /* similar */;

        (median_download, median_upload)
    }
}
```

#### Section 3. Bandwidth Credit Calculation

```rust
pub fn calculate_bandwidth_pog(
    attestation: &BandwidthAttestation,
) -> PogScore {
    let (download, upload) = attestation.aggregate_measurements();

    // Minimum: 10 Mbps sustained
    let combined_mbps = (download + upload) / 2.0;
    if combined_mbps < 10.0 {
        return PogScore::zero();
    }

    // Logarithmic scaling (100 Mbps = ~2.3x score of 10 Mbps)
    let bandwidth_factor = (combined_mbps / 10.0).ln() + 1.0;

    // Uptime factor
    let uptime_factor = attestation.uptime_percentage / 100.0;

    // Latency bonus (lower is better, <50ms = 1.2x)
    let latency_bonus = if attestation.latency_ms_avg < 50 {
        1.2
    } else if attestation.latency_ms_avg < 100 {
        1.1
    } else {
        1.0
    };

    let score = (bandwidth_factor * uptime_factor * latency_bonus * 0.15).min(1.0);

    PogScore::new(score)
}
```

---

### Article VI: Composite PoG Calculation

#### Section 1. Multi-Category Aggregation

Members may contribute across multiple infrastructure categories:

```rust
pub struct CompositePoG {
    pub energy: Option<PogScore>,
    pub storage: Option<PogScore>,
    pub compute: Option<PogScore>,
    pub bandwidth: Option<PogScore>,
}

impl CompositePoG {
    pub fn calculate_total(&self) -> PogScore {
        let mut total = 0.0;
        let mut count = 0;

        // Weighted average across contributed categories
        if let Some(e) = &self.energy {
            total += e.value() * 0.35;  // Energy weighted highest
            count += 1;
        }
        if let Some(s) = &self.storage {
            total += s.value() * 0.25;
            count += 1;
        }
        if let Some(c) = &self.compute {
            total += c.value() * 0.25;
            count += 1;
        }
        if let Some(b) = &self.bandwidth {
            total += b.value() * 0.15;
            count += 1;
        }

        // Bonus for multi-category contribution
        let diversity_bonus = match count {
            1 => 1.0,
            2 => 1.1,
            3 => 1.2,
            4 => 1.3,
            _ => 1.0,
        };

        PogScore::new((total * diversity_bonus).min(1.0))
    }
}
```

#### Section 2. PoG Dashboard Integration

Members view real-time PoG status:

```
┌─────────────────────────────────────────────────┐
│              🌱 PROOF OF GROUNDING              │
├─────────────────────────────────────────────────┤
│  ⚡ Energy:    0.72  │ Solar 3.2 kWh/day        │
│  💾 Storage:  0.58  │ 500 GB IPFS pinned       │
│  🖥️ Compute:  0.45  │ 2.1 TFLOP available      │
│  📡 Bandwidth: 0.31 │ 45 Mbps sustained        │
├─────────────────────────────────────────────────┤
│  🌍 Composite PoG: 0.68 (+0.13 diversity bonus) │
│  📊 PoC Contribution: 27.2% of total            │
└─────────────────────────────────────────────────┘
```

---

### Article VII: Verification Network

#### Section 1. Oracle Network Architecture

PoG verification requires distributed oracle infrastructure:

```rust
pub struct PogOracleNetwork {
    /// Geographic distribution of oracle nodes
    nodes: HashMap<Region, Vec<OracleNode>>,

    /// Consensus threshold for attestation acceptance
    consensus_threshold: f64,  // Default: 0.67

    /// Challenge frequency per category
    challenge_intervals: HashMap<InfraCategory, Duration>,
}

pub struct OracleNode {
    pub node_id: NodeId,
    pub operator_did: String,
    pub supported_categories: Vec<InfraCategory>,
    pub reputation_score: f64,
    pub last_attestation: DateTime<Utc>,
}
```

#### Section 2. Consensus Requirements

| Category | Min Oracle Confirmations | Challenge Frequency |
|----------|-------------------------|---------------------|
| Energy | 3 of 5 | Weekly |
| Storage | 5 of 7 | Daily |
| Compute | 3 of 5 | On-demand |
| Bandwidth | 7 of 11 (peer) | Hourly |

#### Section 3. Oracle Incentives

Oracle operators receive:
- 0.5% of PoG-weighted protocol fees
- CIV reputation bonus for accurate attestations
- CIV penalty for false attestations (slashing)

---

### Article VIII: Anti-Gaming Mechanisms

#### Section 1. Sybil Resistance

**Problem**: Single operator registers multiple "separate" infrastructure contributions.

**Mitigation**:
- Geographic clustering detection
- Hardware fingerprinting
- Network topology analysis
- Economic correlation (same funding source)

```rust
pub fn detect_sybil_cluster(
    attestations: &[Attestation],
    matl_graph: &MatlGraph,
) -> Vec<SybilCluster> {
    // Graph-based clustering via MATL integration
    let clusters = matl_graph.detect_correlated_entities(
        attestations.iter().map(|a| &a.provider_did).collect(),
        CorrelationThreshold::High,
    );

    clusters.into_iter()
        .filter(|c| c.size() > 1)
        .map(|c| SybilCluster::from(c))
        .collect()
}
```

#### Section 2. Fake Infrastructure Prevention

**Problem**: Attestations for non-existent infrastructure.

**Mitigation**:
- Random physical verification (for high-value claims)
- Cross-reference with utility APIs
- Community whistleblower rewards
- Stake-at-risk for oracle operators

#### Section 3. Gaming Penalties

| Violation | First Offense | Second Offense | Third Offense |
|-----------|--------------|----------------|---------------|
| False attestation | CIV -0.1 | CIV -0.3 | 90-day PoG ban |
| Sybil clustering | Cluster merged | CIV -0.2 all | Cluster banned |
| Oracle collusion | Slashing 50% | Oracle removal | Permanent ban |

---

### Article IX: Implementation Timeline

#### Phase 1: Energy Pilot (Q2 2026)

- [ ] Energy oracle deployment
- [ ] Solar/wind hardware integration partnerships
- [ ] 10 pilot energy contributors
- [ ] Dashboard MVP

#### Phase 2: Storage Integration (Q3 2026)

- [ ] IPFS/Filecoin oracle
- [ ] Holochain DHT native integration
- [ ] Challenge protocol deployment
- [ ] 50+ storage contributors

#### Phase 3: Compute + Bandwidth (Q4 2026)

- [ ] TEE attestation framework
- [ ] Peer bandwidth measurement protocol
- [ ] Multi-category aggregation
- [ ] 200+ total PoG contributors

#### Phase 4: Full Integration (Q1 2027)

- [ ] Oracle network decentralization
- [ ] Cross-network PoG portability
- [ ] Advanced anti-gaming
- [ ] 1000+ PoG contributors

---

### Article X: Economic Impact Projections

#### Network Value Floor

With 1000 PoG contributors providing:
- 500 kWh/day renewable energy (~$75/day at grid rates)
- 50 TB distributed storage (~$500/month equivalent)
- 10 TFLOP compute availability (~$1000/month equivalent)

**Projected Infrastructure Value Floor**: ~$2,500/day or ~$75,000/month

This creates minimum network value regardless of token speculation.

#### Validator Composition Target

| Validator Category | Current | Year 1 Target | Year 3 Target |
|--------------------|---------|---------------|---------------|
| Pure Stake | 100% | 60% | 20% |
| Stake + PoG | 0% | 30% | 50% |
| Pure PoG | 0% | 10% | 30% |

---

## Rationale

### Why Physical Infrastructure?

**Intrinsic Value**: A token backed by real energy, storage, and compute has utility value independent of market sentiment.

**Alignment**: Infrastructure operators have long-term network interest—they've invested in physical assets.

**Resilience**: Distributed physical infrastructure is expensive to attack and impossible to fork.

### Why Logarithmic Scaling?

Linear scaling would allow mega-infrastructure operators to dominate. Logarithmic scaling ensures:
- Small contributors are meaningful
- Large contributors get diminishing returns
- No single entity can exceed ~30% of category

### Why Multiple Categories?

Infrastructure diversity creates network robustness:
- Energy-only network is compute-limited
- Storage-only network is bandwidth-constrained
- Multi-category encourages complete infrastructure ecosystems

---

## Backwards Compatibility

This MIP extends MIP-E-002 without conflicts:

- PoG replaces placeholder "Physical_Grounding" component
- Existing CIV scores remain valid (behavioral component unchanged)
- No forced migration—PoG is additive contribution opportunity

---

## Security Considerations

### Oracle Centralization

**Risk**: Oracle network becomes centralized trust point.

**Mitigation**:
- Minimum 11 oracle nodes geographically distributed
- MATL composite scoring for oracle reputation
- Stake slashing for collusion
- Community-operated oracle pathway

### Hardware Attestation Compromise

**Risk**: TPM/TEE vulnerabilities enable fake attestations.

**Mitigation**:
- Multiple attestation methods required
- Regular security audits
- Rapid key rotation capability
- Fallback to community verification

---

## Reference Implementation

Complete Rust implementation in:
- `mycelix-core/src/pog/mod.rs`
- `mycelix-core/src/pog/energy.rs`
- `mycelix-core/src/pog/storage.rs`
- `mycelix-core/src/pog/compute.rs`
- `mycelix-core/src/pog/bandwidth.rs`
- `mycelix-core/src/oracles/physical_infra.rs`

---

## Copyright

This MIP is licensed under CC0 1.0 Universal (Public Domain).

---

*Submitted to Global DAO for ratification consideration.*

**Ratification Requirements**:
- ⅔ supermajority of Global DAO
- MIP-E-002 ratification prerequisite
- 45-day comment period (technical complexity)
- Audit Guild infrastructure feasibility review
- Pilot program success metrics (10+ contributors)
