# MIP-E-007: Civilizational Intention Framework

**Title**: Civilizational Intention Framework - Goal-Oriented Economics
**Author**: Mycelix Vision Working Group
**Status**: DRAFT
**Type**: Standards Track (Economic / Constitutional)
**Category**: Teleological Infrastructure
**Created**: 2026-01-04
**Requires**: MIP-E-001, MIP-E-002
**Supersedes**: None

---

## Abstract

Economic systems optimize. But optimize for *what*?

Traditional economies optimize for GDP growth. DeFi optimizes for TVL and yields. Even well-intentioned protocols optimize for metrics that may not serve human flourishing.

This proposal introduces the **Civilizational Intention Framework** - explicit, measurable goals that the Mycelix Protocol is designed to achieve. The economic machinery becomes *instrumental* to these intentions, not an end in itself.

MIP-E-007 introduces:
1. **Intention Domains**: The major areas of civilizational concern
2. **Intention Trees**: Hierarchical goal structures
3. **Progress Oracles**: Measurement of real-world intention progress
4. **Intention-Aligned Incentives**: Protocol rewards tied to intention advancement
5. **Intention Governance**: How intentions evolve over time

This transforms Mycelix from "an economy" to "an economy *for something*."

---

## Motivation

### The Problem of Optimization Without Purpose

Every optimization system needs an objective function. Modern economics implicitly optimizes for:
- **GDP growth** (regardless of quality or distribution)
- **Shareholder returns** (regardless of externalities)
- **Engagement metrics** (regardless of well-being)

These objectives lead to:
- Climate crisis (externalities ignored)
- Inequality (distribution ignored)
- Mental health epidemic (well-being ignored)

### Why Mycelix Needs Explicit Intentions

The Spore Constitution establishes **Seven Axiomatic Values**:
1. Resonant Coherence
2. Pan-Sentient Flourishing
3. Integral Wisdom
4. Infinite Play
5. Universal Interconnectedness
6. Sacred Reciprocity
7. Evolutionary Progression

But these are *values*, not *goals*. They describe qualities, not outcomes.

**Intentions** translate values into measurable objectives:
- Value: "Pan-Sentient Flourishing"
- Intention: "Universal access to clean water by 2040"
- Metric: "% of global population with access"

### The Teleological Layer

Mycelix economic components:
- **MIP-E-002**: Metabolic Bridge (how value circulates)
- **MIP-E-003**: Proof of Grounding (what infrastructure exists)
- **MIP-E-004**: Agentic Economy (who participates)
- **MIP-E-005**: Temporal Economics (when value manifests)
- **MIP-E-006**: Proof of Regeneration (what ecological impact)

What's missing: **Why** all of this exists.

MIP-E-007 is the teleological layer - the purpose that gives meaning to the mechanisms.

---

## Specification

### 1. Intention Domains

#### 1.1 Core Domains

```rust
pub enum IntentionDomain {
    /// Basic human needs for all
    UniversalBasicServices,
    /// Planetary ecosystem health
    EcologicalRegeneration,
    /// Freely accessible knowledge
    KnowledgeCommons,
    /// Violence reduction and justice
    ConflictTransformation,
    /// Human consciousness development
    ConsciousnessEvolution,
    /// Democratic participation
    GovernanceEvolution,
    /// Infrastructure for all
    DigitalSovereignty,
    /// Cultural preservation and creation
    CulturalFlourishing,
    /// Scientific advancement
    ScientificFrontier,
    /// Joy and meaning
    CollectiveWellbeing,
}
```

#### 1.2 Domain Descriptions

| Domain | Description | Example Intentions |
|--------|-------------|-------------------|
| **UniversalBasicServices** | Ensuring all humans have access to essentials | Clean water, food security, shelter, healthcare, education |
| **EcologicalRegeneration** | Restoring planetary health | Climate stability, biodiversity, ocean health, forest cover |
| **KnowledgeCommons** | Making knowledge freely accessible | Open science, education access, cultural heritage |
| **ConflictTransformation** | Reducing violence and achieving justice | War prevention, restorative justice, reconciliation |
| **ConsciousnessEvolution** | Supporting human development | Mental health, wisdom traditions, contemplative practice |
| **GovernanceEvolution** | Better collective decision-making | Participatory democracy, transparency, accountability |
| **DigitalSovereignty** | Technology that serves humanity | Privacy, data rights, digital access |
| **CulturalFlourishing** | Supporting creativity and meaning | Arts, languages, traditions |
| **ScientificFrontier** | Advancing understanding | Research funding, open science, breakthrough technologies |
| **CollectiveWellbeing** | Joy and meaning in life | Community, belonging, purpose |

### 2. Intention Trees

#### 2.1 Structure

Intentions form hierarchical trees from abstract to concrete:

```rust
pub struct IntentionTree {
    /// Root domain
    pub domain: IntentionDomain,
    /// Tree structure
    pub root: IntentionNode,
}

pub struct IntentionNode {
    /// Unique intention ID
    pub intention_id: IntentionId,
    /// Human-readable name
    pub name: String,
    /// Detailed description
    pub description: String,
    /// Quantitative target (optional)
    pub target: Option<QuantitativeTarget>,
    /// Target achievement date (optional)
    pub target_date: Option<u64>,
    /// How to measure progress
    pub metrics: Vec<ProgressMetric>,
    /// Sub-intentions (more specific)
    pub children: Vec<IntentionNode>,
    /// Status
    pub status: IntentionStatus,
}

pub struct QuantitativeTarget {
    /// Metric name
    pub metric_name: String,
    /// Current value
    pub current_value: f64,
    /// Target value
    pub target_value: f64,
    /// Unit
    pub unit: String,
}

pub enum IntentionStatus {
    /// Actively being worked toward
    Active,
    /// Achieved
    Achieved { achieved_at: u64 },
    /// Superseded by new intention
    Superseded { by: IntentionId },
    /// Abandoned (with reason)
    Abandoned { reason: String },
}
```

#### 2.2 Example Tree: Universal Basic Services

```
UniversalBasicServices (Domain)
├── CleanWaterForAll
│   ├── SafeDrinkingWater
│   │   ├── UrbanWaterAccess (Target: 99% by 2030)
│   │   ├── RuralWaterAccess (Target: 95% by 2035)
│   │   └── WaterQualityStandards
│   ├── Sanitation
│   │   ├── BasicSanitation (Target: 95% by 2035)
│   │   └── AdvancedSanitation (Target: 80% by 2040)
│   └── WaterSecurityResilience
│       ├── DroughtResilience
│       └── FloodResilience
├── FoodSecurity
│   ├── ZeroHunger (Target: <3% food insecurity by 2035)
│   ├── NutritionSecurity
│   └── SustainableFoodSystems
├── ShelterForAll
│   ├── HomelessnessElimination
│   ├── AffordableHousing
│   └── ClimateResilientShelter
├── HealthcareAccess
│   ├── UniversalPrimaryCare
│   ├── MentalHealthServices
│   └── MaternalChildHealth
└── EducationAccess
    ├── UniversalLiteracy
    ├── SecondaryEducation
    └── LifelongLearning
```

### 3. Progress Oracles

#### 3.1 Oracle Structure

```rust
pub struct IntentionOracle {
    /// Oracle ID
    pub oracle_id: OracleId,
    /// Intentions covered
    pub intentions: Vec<IntentionId>,
    /// Data sources
    pub sources: Vec<IntentionDataSource>,
    /// Update frequency
    pub update_frequency_epochs: u32,
    /// Methodology documentation
    pub methodology_doc: String,
    /// Dispute mechanism
    pub dispute_mechanism: DisputeMechanism,
}

pub enum IntentionDataSource {
    /// UN/World Bank statistics
    InternationalOrganization {
        org: String,
        dataset: String,
    },
    /// Government statistics
    NationalStatistics {
        countries: Vec<String>,
        indicators: Vec<String>,
    },
    /// Academic research
    AcademicResearch {
        institutions: Vec<String>,
        methodologies: Vec<String>,
    },
    /// Satellite/remote sensing
    RemoteSensing {
        provider: String,
        indicators: Vec<String>,
    },
    /// On-chain Mycelix data
    OnChainData {
        metrics: Vec<String>,
    },
    /// Community reporting
    CommunityReporting {
        program_id: String,
        verification_method: String,
    },
}
```

#### 3.2 Progress Calculation

```rust
pub struct IntentionProgress {
    /// Intention ID
    pub intention_id: IntentionId,
    /// Current progress (0.0 to 1.0)
    pub progress: f64,
    /// Progress delta since last measurement
    pub delta: f64,
    /// Trajectory (are we on track?)
    pub trajectory: Trajectory,
    /// Blockers identified
    pub blockers: Vec<Blocker>,
    /// Mycelix contribution estimate
    pub mycelix_contribution: f64,
    /// Measurement timestamp
    pub measured_at: u64,
    /// Confidence level
    pub confidence: f64,
}

pub enum Trajectory {
    /// Ahead of schedule
    Accelerating,
    /// On track
    OnTrack,
    /// Behind but recoverable
    Lagging,
    /// Significantly behind
    AtRisk,
    /// Moving backwards
    Regressing,
}
```

### 4. Intention-Aligned Incentives

#### 4.1 Contribution Tracking

```rust
pub struct IntentionContribution {
    /// Contributor (DID)
    pub contributor_did: String,
    /// Intention contributed to
    pub intention_id: IntentionId,
    /// Contribution type
    pub contribution_type: ContributionType,
    /// Estimated impact
    pub estimated_impact: f64,
    /// Verification
    pub verification: ContributionVerification,
    /// Timestamp
    pub timestamp: u64,
}

pub enum ContributionType {
    /// Direct action toward intention
    DirectAction {
        description: String,
        evidence: Vec<Evidence>,
    },
    /// Funding allocation
    Funding {
        amount_sap: u64,
        recipient: String,
    },
    /// Infrastructure provision (via PoG)
    Infrastructure {
        pog_contribution: ProofOfGrounding,
        intention_alignment: f64,
    },
    /// Knowledge contribution
    Knowledge {
        artifact: String,
        artifact_hash: String,
    },
    /// Coordination/governance
    Coordination {
        role: String,
        hours: f64,
    },
}
```

#### 4.2 Intention Multipliers

Contributors to high-priority intentions receive multipliers:

```rust
pub struct IntentionMultiplier {
    /// Domain weights (protocol parameter)
    pub domain_weights: HashMap<IntentionDomain, f64>,
    /// Urgency multiplier (based on trajectory)
    pub urgency_multiplier: f64,
    /// Contribution depth (specific vs general)
    pub depth_multiplier: f64,
    /// Final multiplier
    pub total_multiplier: f64,
}
```

**Example weights**:
| Domain | Weight | Rationale |
|--------|--------|-----------|
| UniversalBasicServices | 1.5 | Most urgent human needs |
| EcologicalRegeneration | 1.5 | Existential priority |
| ConflictTransformation | 1.3 | Foundational for others |
| KnowledgeCommons | 1.2 | Enables all else |
| DigitalSovereignty | 1.1 | Mycelix specialty |
| Others | 1.0 | Important but less urgent |

#### 4.3 PoC Integration

Extend Proof of Contribution (MIP-E-002) to include intention alignment:

```
PoC_Score = (Behavioral_50% + PoG_30% + Intention_20%) × Patience_Coefficient
```

Intention component:
```rust
pub fn calculate_intention_score(
    contributions: &[IntentionContribution],
    intention_progress: &HashMap<IntentionId, IntentionProgress>,
    weights: &IntentionMultiplier,
) -> f64 {
    contributions.iter()
        .map(|c| {
            let intention = &intention_progress[&c.intention_id];
            let urgency = match intention.trajectory {
                Trajectory::Regressing => 2.0,
                Trajectory::AtRisk => 1.5,
                Trajectory::Lagging => 1.2,
                Trajectory::OnTrack => 1.0,
                Trajectory::Accelerating => 0.8,
            };
            c.estimated_impact * urgency * weights.domain_weights[&c.intention_id.domain()]
        })
        .sum()
}
```

### 5. Intention Governance

#### 5.1 Intention Lifecycle

```
PROPOSED → REVIEW → RATIFIED → ACTIVE → ACHIEVED/SUPERSEDED/ABANDONED
```

```rust
pub enum IntentionLifecycle {
    /// Initial proposal
    Proposed {
        proposer: String,
        proposed_at: u64,
    },
    /// Under community review
    Review {
        review_start: u64,
        comments: Vec<Comment>,
        endorsements: u32,
    },
    /// Ratified by Global DAO
    Ratified {
        ratified_at: u64,
        vote_result: VoteResult,
    },
    /// Actively being pursued
    Active {
        activated_at: u64,
        current_progress: f64,
    },
    /// Successfully achieved
    Achieved {
        achieved_at: u64,
        celebration: Option<String>,
    },
    /// Replaced by better intention
    Superseded {
        superseded_at: u64,
        successor: IntentionId,
        reason: String,
    },
    /// Abandoned
    Abandoned {
        abandoned_at: u64,
        reason: String,
        lessons: String,
    },
}
```

#### 5.2 Voting Requirements

| Action | Quorum | Approval | Review Period |
|--------|--------|----------|---------------|
| Add new intention | 30% | Simple majority | 21 days |
| Modify intention | 30% | Simple majority | 14 days |
| Achieve intention | 20% | Simple majority | 7 days |
| Abandon intention | 40% | ⅔ supermajority | 30 days |
| Change domain weights | 50% | ⅔ supermajority | 60 days |

#### 5.3 Intention Councils

Each domain has an advisory council:

```rust
pub struct IntentionCouncil {
    /// Domain covered
    pub domain: IntentionDomain,
    /// Council members (experts + community)
    pub members: Vec<CouncilMember>,
    /// Selection method
    pub selection: CouncilSelection,
    /// Term length (epochs)
    pub term_epochs: u32,
    /// Responsibilities
    pub responsibilities: Vec<Responsibility>,
}

pub struct CouncilMember {
    /// Member DID
    pub member_did: String,
    /// Role
    pub role: CouncilRole,
    /// Term start
    pub term_start: u64,
    /// Expertise documentation
    pub expertise: String,
}

pub enum CouncilRole {
    /// Domain expert (academic/practitioner)
    Expert,
    /// Community representative
    CommunityRep,
    /// Youth representative (< 30)
    YouthRep,
    /// Indigenous/traditional knowledge holder
    TraditionalKnowledge,
    /// Mycelix protocol representative
    ProtocolRep,
}
```

### 6. Civilizational Dashboard

#### 6.1 Public Progress Display

```rust
pub struct CivilizationalDashboard {
    /// Overall civilizational progress (aggregate)
    pub overall_progress: f64,
    /// Per-domain progress
    pub domain_progress: HashMap<IntentionDomain, DomainProgress>,
    /// Top intentions (most urgent)
    pub urgent_intentions: Vec<IntentionProgress>,
    /// Recent achievements
    pub recent_achievements: Vec<Achievement>,
    /// Top contributors (privacy-respecting)
    pub top_contributors: Vec<ContributorSummary>,
    /// Mycelix impact estimate
    pub mycelix_impact: ImpactEstimate,
}

pub struct DomainProgress {
    /// Domain
    pub domain: IntentionDomain,
    /// Active intentions count
    pub active_intentions: u32,
    /// Achieved intentions count
    pub achieved_intentions: u32,
    /// Average progress
    pub avg_progress: f64,
    /// Overall trajectory
    pub trajectory: Trajectory,
}
```

#### 6.2 Integration with Protocol

The dashboard is:
- Embedded in every Mycelix client
- Updated every epoch (30 days)
- Linked to individual contribution history
- Source of TEND bonus calculations

---

## Rationale

### Why Explicit Intentions?

Implicit optimization is dangerous. Facebook optimized for engagement; the result was social harm. Mycelix should be explicit about what it's optimizing for.

### Why Hierarchical Trees?

Abstract intentions ("human flourishing") are inspiring but unmeasurable. Concrete intentions ("clean water for village X") are measurable but lack vision. Trees connect both.

### Why Multiple Domains?

Single-metric optimization fails. GDP maximization ignores ecology. Carbon minimization might ignore poverty. Multi-domain optimization creates balance.

### Why Councils?

Pure democracy risks uninformed decisions on technical matters. Pure technocracy risks ignoring community wisdom. Councils balance expertise with representation.

### Why Public Dashboard?

Transparency creates accountability. Every Mycelix user should know what the protocol is trying to achieve and whether it's succeeding.

---

## Backwards Compatibility

### Existing Components

All existing MIPs remain valid. MIP-E-007 adds a new dimension:
- **PoC**: Gains intention alignment component
- **Covenants (MIP-E-005)**: Can reference intentions
- **HEARTH pools**: Can focus on specific intentions
- **Agent collectives (MIP-E-008)**: Can form around intentions

### Gradual Adoption

1. **Phase 1**: Intention registry launched (informational only)
2. **Phase 2**: Intention contribution tracking
3. **Phase 3**: Intention multipliers in PoC
4. **Phase 4**: Full integration with governance

---

## Security Considerations

### 1. Intention Gaming

Members might claim false contributions to intentions. Mitigation:
- Contribution verification requirements
- Peer attestation
- Impact measurement via oracles

### 2. Council Capture

Councils might be captured by special interests. Mitigation:
- Term limits
- Diverse selection methods
- Override by Global DAO supermajority

### 3. Metrics Gaming

Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."

Mitigation:
- Multiple metrics per intention
- Qualitative oracle components
- Regular intention review and refinement

### 4. Priority Disputes

Disagreement on domain weights is inevitable. Mitigation:
- High threshold for weight changes (⅔ + 60 days)
- Transparent methodology
- Annual review cycles

---

## Implementation Status

| Component | SDK Module | Status |
|-----------|------------|--------|
| Intention Domains | `intentions::domains` | Planned |
| Intention Trees | `intentions::trees` | Planned |
| Progress Oracles | `intentions::oracles` | Planned |
| Contribution Tracking | `intentions::contributions` | Planned |
| Intention Multipliers | `intentions::multipliers` | Planned |
| Councils | `intentions::governance` | Planned |
| Dashboard | `intentions::dashboard` | Planned |

---

## References

- [Spore Constitution v0.24](../architecture/THE%20MYCELIX%20SPORE%20CONSTITUTION%20(v0.24).md)
- [MIP-E-002: Metabolic Bridge](./MIP-E-002_METABOLIC_BRIDGE_AMENDMENT.md)
- [MIP-E-005: Temporal Economics](./MIP-E-005_TEMPORAL_ECONOMICS.md)
- UN Sustainable Development Goals (for domain inspiration)
- Doughnut Economics (Kate Raworth)
- Effective Altruism frameworks

---

## Copyright

This document is licensed under Apache 2.0.

---

*"An economy without purpose is just accounting. An economy with purpose is civilization."*
