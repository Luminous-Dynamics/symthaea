# MIP-E-005: Temporal Economics Framework

**Title**: Temporal Economics - Patient Capital and Intergenerational Value
**Author**: Mycelix Economic Working Group
**Status**: DRAFT
**Type**: Standards Track (Economic)
**Category**: Temporal Value Mechanics
**Created**: 2026-01-04
**Requires**: MIP-E-001, MIP-E-002
**Supersedes**: None

---

## Abstract

This proposal introduces **Temporal Economics** to the Mycelix Protocol - a framework for aligning economic incentives with long-term thinking. Current economic systems optimize for quarterly returns; Mycelix should optimize for generational flourishing.

MIP-E-005 introduces:
1. **Time-Locked Commitments** with graduated governance weight
2. **Covenant Bonds** that bind capital to specific purposes
3. **Intergenerational Trusts** for assets held across generations
4. **Temporal Multipliers** that reward patience throughout the protocol
5. **Succession Mechanisms** for orderly value transfer

This creates a "slow money" layer that counterbalances short-term speculation with patient, purpose-aligned capital.

---

## Motivation

### The Problem of Temporal Myopia

Modern economies suffer from extreme short-termism:
- Public companies optimize for quarterly earnings
- Politicians optimize for election cycles
- Individuals optimize for immediate consumption
- Markets discount future value at rates that make 30-year planning irrational

This temporal myopia is incompatible with:
- Ecological regeneration (decades to centuries)
- Infrastructure development (multi-generational)
- Cultural evolution (generational)
- Civilizational resilience (centuries to millennia)

### Why Mycelix Must Think Long

The Spore Constitution establishes **Evolutionary Progression** as an axiomatic value. The Metabolic Bridge (MIP-E-002) creates circulation, but circulation without *direction* is mere churning.

Temporal Economics provides:
1. **Incentives for patience** - Those who commit longer gain more influence
2. **Purpose preservation** - Capital can be bound to intentions that outlive individuals
3. **Intergenerational equity** - Future generations have representation in current decisions
4. **Stability anchors** - Long-term commitments create predictable value floors

---

## Specification

### 1. Time-Locked Commitments

#### 1.1 Commitment Structure

```rust
pub struct TemporalCommitment {
    /// Unique commitment ID
    pub commitment_id: CommitmentId,
    /// Member DID
    pub member_did: String,
    /// SAP amount locked
    pub sap_locked: u64,
    /// Lock start timestamp
    pub lock_start: u64,
    /// Lock duration in epochs (1 epoch = 30 days)
    pub duration_epochs: u32,
    /// Unlock timestamp (computed)
    pub unlock_at: u64,
    /// Purpose covenant (optional)
    pub covenant: Option<Covenant>,
    /// Early exit penalty percentage
    pub exit_penalty_pct: f64,
    /// Governance weight multiplier earned
    pub governance_multiplier: f64,
    /// TEND earning multiplier
    pub tend_multiplier: f64,
}
```

#### 1.2 Duration Tiers

| Tier | Duration | Gov Multiplier | TEND Multiplier | Exit Penalty |
|------|----------|----------------|-----------------|--------------|
| **Sprout** | 1-6 epochs (1-6 months) | 1.0x | 1.0x | 5% |
| **Sapling** | 7-24 epochs (7mo-2yr) | 1.5x | 1.25x | 10% |
| **Tree** | 25-84 epochs (2-7yr) | 2.5x | 1.75x | 15% |
| **Grove** | 85-168 epochs (7-14yr) | 4.0x | 2.5x | 20% |
| **Forest** | 169+ epochs (14yr+) | 7.0x | 4.0x | 25% |

**Rationale**: Exponential governance weight rewards extreme patience. A 14-year commitment carries 7x the governance weight of a 1-month commitment, creating a "deep time" constituency in all decisions.

#### 1.3 Governance Weight Calculation

```
Effective_Vote_Weight = Base_CIV × Σ(Commitment_SAP × Governance_Multiplier) / Total_SAP_Held
```

This means:
- A member with 0.8 CIV, 10,000 SAP (5,000 in Forest tier, 5,000 liquid) has:
  - Forest portion: 5,000 × 7.0 = 35,000 effective
  - Liquid portion: 5,000 × 1.0 = 5,000 effective
  - Total effective: 40,000 / 10,000 = 4.0x multiplier
  - Final vote weight: 0.8 × 4.0 = 3.2 (vs 0.8 with no commitments)

#### 1.4 Early Exit Mechanism

Members may exit commitments early, but:
1. **Penalty Deduction**: Exit penalty percentage deducted from locked SAP
2. **Penalty Distribution**: 50% to source HEARTH, 50% to Global Commons
3. **CIV Impact**: -0.02 CIV penalty per early exit (discourages pattern)
4. **Cooldown**: Cannot create new commitment in same tier for 6 epochs

```rust
pub struct EarlyExitResult {
    /// SAP returned to member
    pub sap_returned: u64,
    /// Penalty amount
    pub penalty_amount: u64,
    /// HEARTH portion
    pub to_hearth: u64,
    /// Commons portion
    pub to_commons: u64,
    /// CIV adjustment
    pub civ_adjustment: f64,
    /// Cooldown until
    pub cooldown_until: u64,
}
```

### 2. Covenant Bonds

#### 2.1 Purpose-Bound Capital

Covenants bind locked capital to specific purposes, ensuring the "why" survives beyond the individual:

```rust
pub struct Covenant {
    /// Covenant ID
    pub covenant_id: CovenantId,
    /// Human-readable purpose
    pub purpose: String,
    /// Structured intention (from Civilizational Intentions - MIP-E-007)
    pub intention: Option<CivilizationalIntention>,
    /// Beneficiary specification
    pub beneficiaries: BeneficiarySpec,
    /// Success metrics
    pub success_metrics: Vec<Metric>,
    /// Dissolution conditions
    pub dissolution_conditions: Vec<Condition>,
    /// Successor covenant if this dissolves
    pub successor: Option<CovenantId>,
}
```

#### 2.2 Beneficiary Types

```rust
pub enum BeneficiarySpec {
    /// Specific DIDs
    Named(Vec<String>),
    /// Members of a Local DAO
    LocalDao(String),
    /// Members of a bioregion
    Bioregion(BioregionId),
    /// Future generations (demographic oracle)
    FutureGenerations {
        /// Born after this timestamp
        born_after: u64,
        /// In this geographic scope
        scope: GeographicScope,
    },
    /// Ecological entity (river, forest, etc.)
    EcologicalEntity {
        /// Entity identifier
        entity_id: String,
        /// Legal guardian DAO
        guardian_dao: String,
    },
    /// Universal (all Mycelix members)
    Universal,
}
```

#### 2.3 Covenant Multipliers

Covenanted commitments receive additional benefits:

| Covenant Type | Additional Gov Multiplier | Additional TEND |
|---------------|---------------------------|-----------------|
| No covenant | 1.0x | 1.0x |
| Named beneficiaries | 1.1x | 1.1x |
| Local DAO / Bioregion | 1.25x | 1.2x |
| Future Generations | 1.5x | 1.5x |
| Ecological Entity | 1.5x | 1.5x |
| Universal | 1.75x | 1.75x |

**Rationale**: More expansive beneficiary scope = more aligned with collective flourishing = more reward.

### 3. Intergenerational Trusts

#### 3.1 Trust Structure

```rust
pub struct IntergenerationalTrust {
    /// Trust ID
    pub trust_id: TrustId,
    /// Founding members
    pub founders: Vec<String>,
    /// Trust covenant
    pub covenant: Covenant,
    /// SAP corpus (principal)
    pub corpus: u64,
    /// TEND accumulated
    pub accumulated_tend: u64,
    /// Governance structure
    pub governance: TrustGovernance,
    /// Distribution rules
    pub distribution: DistributionRules,
    /// Trust duration (None = perpetual)
    pub duration: Option<u64>,
    /// Created timestamp
    pub created_at: u64,
}
```

#### 3.2 Trust Governance

```rust
pub struct TrustGovernance {
    /// Steward DIDs (current managers)
    pub stewards: Vec<String>,
    /// Maximum steward count
    pub max_stewards: u32,
    /// Steward term length (epochs)
    pub steward_term: u32,
    /// Succession method
    pub succession: SuccessionMethod,
    /// Amendment threshold (% of stewards)
    pub amendment_threshold: f64,
    /// Dissolution threshold
    pub dissolution_threshold: f64,
}

pub enum SuccessionMethod {
    /// Stewards elect successors
    Election,
    /// Beneficiaries elect stewards
    BeneficiaryElection,
    /// Lottery among qualified candidates
    Sortition { min_civ: f64 },
    /// Automatic based on contribution
    MeritocraticRotation,
    /// Hybrid approaches
    Hybrid(Vec<SuccessionMethod>),
}
```

#### 3.3 Distribution Rules

```rust
pub struct DistributionRules {
    /// What can be distributed
    pub distributable: Distributable,
    /// Distribution frequency
    pub frequency: DistributionFrequency,
    /// Per-beneficiary limits
    pub per_beneficiary_limit: Option<u64>,
    /// Total distribution limit per period
    pub period_limit: Option<u64>,
    /// Conditions for distribution
    pub conditions: Vec<DistributionCondition>,
}

pub enum Distributable {
    /// Only TEND (preserve corpus)
    TendOnly,
    /// TEND + percentage of corpus
    TendPlusCorpus { max_corpus_pct: f64 },
    /// Everything (winding down)
    Full,
}

pub enum DistributionFrequency {
    /// Every N epochs
    Periodic(u32),
    /// On request (subject to conditions)
    OnDemand,
    /// At specific timestamps
    Scheduled(Vec<u64>),
    /// When conditions are met
    Conditional,
}
```

### 4. Temporal Multipliers Throughout Protocol

#### 4.1 Unified Patience Coefficient

All protocol operations should respect temporal commitment:

```rust
pub struct PatienceCoefficient {
    /// Member's weighted average commitment duration
    pub avg_duration_epochs: f64,
    /// Longest active commitment
    pub max_duration: u32,
    /// Covenant depth (how purpose-bound)
    pub covenant_depth: f64,
    /// Computed coefficient (0.5 - 3.0)
    pub coefficient: f64,
}
```

**Applications**:
- **Validation weight**: Validators with higher patience coefficient have more influence
- **Proposal weight**: MIP proposals from patient members get more initial support
- **Dispute resolution**: Arbitrators weighted by patience
- **HEARTH governance**: Voting power in HEARTH pools includes patience factor

#### 4.2 Integration with Proof of Contribution

Extend PoC (MIP-E-002) to include temporal dimension:

```
PoC_Score = (Behavioral_60% + PoG_40%) × Patience_Coefficient
```

This means someone with:
- High behavioral score (0.8)
- Strong PoG (0.7)
- 10-year commitment (patience coefficient 1.5)

Gets: (0.8 × 0.6 + 0.7 × 0.4) × 1.5 = 1.14 PoC (vs 0.76 without patience)

### 5. Succession Mechanisms

#### 5.1 Commitment Inheritance

When a member dies or becomes incapacitated:

```rust
pub struct SuccessionPlan {
    /// Member DID
    pub member_did: String,
    /// Designated successors (ordered)
    pub successors: Vec<SuccessorDesignation>,
    /// Fallback if no successor available
    pub fallback: SuccessionFallback,
    /// Notification requirements
    pub notification: NotificationRequirements,
    /// Last updated
    pub updated_at: u64,
}

pub struct SuccessorDesignation {
    /// Successor DID
    pub successor_did: String,
    /// Percentage of commitments
    pub percentage: f64,
    /// Conditions
    pub conditions: Vec<Condition>,
}

pub enum SuccessionFallback {
    /// Convert to trust
    CreateTrust { covenant: Covenant },
    /// Donate to HEARTH
    ToHearth(String),
    /// To Global Commons
    ToCommons,
    /// Dissolve (return to protocol)
    Dissolve,
}
```

#### 5.2 Proof of Incapacity

Triggering succession requires proof (to prevent fraud):

```rust
pub enum IncapacityProof {
    /// Legal death certificate (attested)
    DeathCertificate {
        attestor: String,
        document_hash: String
    },
    /// Court-ordered incapacity
    LegalIncapacity {
        jurisdiction: String,
        case_id: String
    },
    /// Multi-sig declaration by trusted parties
    TrustedDeclaration {
        declarers: Vec<String>,
        threshold: u32
    },
    /// Extended inactivity (configurable, default 7 years)
    ExtendedInactivity {
        last_activity: u64,
        threshold_epochs: u32
    },
}
```

---

## Rationale

### Why Exponential Governance Multipliers?

Linear multipliers don't create sufficient differentiation. If 10-year commitment only gets 2x vs 1-year, the incentive is weak. Exponential scaling (1x → 7x) makes long-term commitment genuinely valuable.

### Why Allow Early Exit?

Complete lock-in creates:
1. **Liquidity crises** for individuals facing emergencies
2. **Reduced commitment** (fear of lock-in prevents initial commitment)
3. **Black markets** for commitment transfers

Allowing exit with penalties balances commitment with flexibility. The penalty isn't punitive - it's redistributive (flows to commons).

### Why Intergenerational Trusts?

Most human institutions that last centuries (universities, religious endowments, land trusts) have trust-like structures. Mycelix needs native support for multi-generational capital preservation.

### Why Patience Coefficient Throughout?

If patience only affects governance voting, it's siloed. By threading patience through PoC, validation weight, and HEARTH governance, we create a coherent "slow money" philosophy that permeates the entire protocol.

---

## Backwards Compatibility

### Existing Commitments

All current SAP holdings are implicitly "Sprout" tier (1.0x multipliers). Members can optionally lock existing holdings into higher tiers.

### HEARTH Integration

Existing HEARTH pools continue unchanged. Temporal Commitment is a complementary mechanism, not a replacement.

### CIV Calculation

CIV calculation (per Economic Charter) remains unchanged. Patience Coefficient is multiplicative on top of CIV, not a component of it.

---

## Security Considerations

### 1. Sybil Resistance

Long lock periods naturally resist Sybil attacks - it's expensive to maintain many long-term commitments across multiple identities.

### 2. Oracle Attacks (Death/Incapacity)

Succession triggers require multi-party attestation or extended inactivity. Single-party claims are insufficient.

### 3. Covenant Manipulation

Covenants are immutable once created. Beneficiary specifications cannot be changed (only dissolved according to dissolution conditions).

### 4. Time Manipulation

All timestamps use protocol consensus time, not individual node clocks.

---

## Implementation Status

| Component | SDK Module | Status |
|-----------|------------|--------|
| Commitment Structure | `temporal::commitment` | Planned |
| Duration Tiers | `temporal::tiers` | Planned |
| Covenant Bonds | `temporal::covenant` | Planned |
| Intergenerational Trusts | `temporal::trust` | Planned |
| Patience Coefficient | `temporal::patience` | Planned |
| Succession | `temporal::succession` | Planned |

---

## References

- [MIP-E-001: Validator Economics Framework](../architecture/THE%20ECONOMIC%20CHARTER%20(v1.0).md)
- [MIP-E-002: Metabolic Bridge Amendment](./MIP-E-002_METABOLIC_BRIDGE_AMENDMENT.md)
- [The Spore Constitution v0.24](../architecture/THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)
- [Commons Charter v1.0](../architecture/THE%20COMMONS%20CHARTER%20(v1.0).md)

---

## Copyright

This document is licensed under Apache 2.0.

---

*"The best time to plant a tree was twenty years ago. The second best time is now. The best economic system rewards those who plant trees they'll never sit under."*
