# Extended hApps: Civilizational Completeness

## Overview

Beyond the core 8 Tier 1 hApps, a complete civilizational OS requires additional primitives for economic coordination, community formation, resource management, and emergent phenomena. This document provides design summaries for 19 additional hApps organized into Tier 2 (Essential) and Tier 3 (Advanced).

---

## Tier 2: Essential Happs (12)

### 1. Mycelix-Treasury
**Purpose**: Collective resource management and allocation

**Core Functions**:
- Multi-sig treasury management
- Budget allocation and tracking
- Transparent spending proposals
- Streaming payments

**Key Integrations**:
- Agora (governance-controlled spending)
- Collab (project funding)
- Marketplace (procurement)

**Data Model Highlights**:
```rust
pub struct Treasury {
    pub treasury_id: String,
    pub governance_space: ActionHash, // Agora space that controls this
    pub balance: HashMap<String, u64>, // Currency -> Amount
    pub signers: Vec<AgentPubKey>,
    pub threshold: u32, // Multi-sig threshold
    pub budget_allocations: Vec<BudgetAllocation>,
}

pub struct SpendingProposal {
    pub amount: u64,
    pub currency: String,
    pub recipient: Recipient,
    pub purpose: String,
    pub approvals: Vec<AgentPubKey>,
    pub status: ProposalStatus,
}
```

---

### 2. Mycelix-Commons
**Purpose**: Shared resource governance (Ostrom-inspired)

**Core Functions**:
- Define common pool resources
- Set access rules and boundaries
- Monitor usage and enforce limits
- Graduated sanctions for violations

**Key Integrations**:
- Agora (rule-making)
- Arbiter (dispute resolution)
- Oracle (usage monitoring)

**Design Principles** (Ostrom's 8):
1. Clearly defined boundaries
2. Congruence between rules and local conditions
3. Collective choice arrangements
4. Monitoring
5. Graduated sanctions
6. Conflict resolution mechanisms
7. Minimal recognition of rights
8. Nested enterprises

---

### 3. Mycelix-Guild
**Purpose**: Professional associations and credentialing

**Core Functions**:
- Professional standards definition
- Skill certification paths
- Mentorship matching
- Knowledge sharing

**Key Integrations**:
- Praxis (credentials)
- Collab (project staffing)
- Attest (professional identity)

**Data Model Highlights**:
```rust
pub struct Guild {
    pub guild_id: String,
    pub profession: String,
    pub standards: Vec<Standard>,
    pub certification_paths: Vec<CertificationPath>,
    pub membership_tiers: Vec<MembershipTier>,
    pub governance: GuildGovernance,
}

pub struct CertificationPath {
    pub name: String,
    pub requirements: Vec<Requirement>,
    pub credential_template: ActionHash,
    pub assessors: Vec<AgentPubKey>,
}
```

---

### 4. Mycelix-Mutual
**Purpose**: Mutual aid and risk pooling

**Core Functions**:
- Define mutual aid protocols
- Risk pool management
- Claims processing
- Premium/contribution calculation

**Key Integrations**:
- Oracle (event verification)
- Arbiter (claims disputes)
- Treasury (pool management)
- HealthVault (health claims)

**Use Cases**:
- Crop insurance (Oracle: weather data)
- Health mutual aid
- Equipment breakdown coverage
- Income protection

---

### 5. Mycelix-Beacon
**Purpose**: Emergency coordination and alerting

**Core Functions**:
- Emergency declaration
- Resource mobilization
- Volunteer coordination
- Status tracking

**Key Integrations**:
- Sentinel (threat detection)
- HealthVault (medical emergencies)
- Oracle (event verification)
- Agora (emergency governance)

**Data Model Highlights**:
```rust
pub struct Emergency {
    pub emergency_id: String,
    pub emergency_type: EmergencyType,
    pub severity: EmergencySeverity,
    pub affected_area: GeographicScope,
    pub status: EmergencyStatus,
    pub responders: Vec<Responder>,
    pub resources_needed: Vec<ResourceNeed>,
    pub resources_offered: Vec<ResourceOffer>,
}

pub enum EmergencyType {
    Natural { disaster_type: String },
    Health { condition: String },
    Security { threat_type: String },
    Infrastructure { system: String },
    Community { issue: String },
}
```

---

### 6. Mycelix-Nexus
**Purpose**: Event coordination and gathering

**Core Functions**:
- Event creation and discovery
- RSVP and ticketing
- Schedule coordination
- Post-event archiving

**Key Integrations**:
- Attest (attendee verification)
- Treasury (ticketing revenue)
- Chronicle (event history)
- Anchor (venue management)

---

### 7. Mycelix-Loom
**Purpose**: Narrative weaving and storytelling

**Core Functions**:
- Collaborative story creation
- Cultural narrative archiving
- Myth and legend tracking
- Narrative analysis

**Key Integrations**:
- Chronicle (archival)
- Attest (author attribution)
- Agora (community narratives)

**Philosophy**: Every civilization needs stories. Loom captures and preserves the narratives that give meaning to collective action.

---

### 8. Mycelix-Seed
**Purpose**: Project incubation and early-stage support

**Core Functions**:
- Project proposal submission
- Community feedback
- Resource matching
- Milestone tracking

**Key Integrations**:
- Collab (graduation to full projects)
- Treasury (seed funding)
- Guild (mentor matching)
- Praxis (skill development)

**Data Model Highlights**:
```rust
pub struct SeedProject {
    pub seed_id: String,
    pub title: String,
    pub description: String,
    pub stage: IncubationStage,
    pub founder: AgentPubKey,
    pub mentors: Vec<AgentPubKey>,
    pub milestones: Vec<SeedMilestone>,
    pub community_support: u64, // Endorsements
    pub resources_requested: Vec<ResourceRequest>,
}

pub enum IncubationStage {
    Ideation,
    Validation,
    Prototyping,
    PilotTesting,
    LaunchReady,
    Graduated,
}
```

---

### 9. Mycelix-Pulse
**Purpose**: Collective sentiment and temperature-checking

**Core Functions**:
- Anonymous sentiment surveys
- Trend detection
- Community health metrics
- Early warning signals

**Key Integrations**:
- Agora (input to governance)
- Sentinel (anomaly detection)
- Chronicle (historical tracking)

**Privacy Features**:
- ZKP for anonymous participation
- Differential privacy for aggregates
- No individual tracking

---

### 10. Mycelix-Weave
**Purpose**: Relationship mapping and social graph

**Core Functions**:
- Relationship declaration
- Trust path computation
- Community structure analysis
- Connection recommendations

**Key Integrations**:
- Attest (vouch verification)
- Arbiter (conflict of interest detection)
- MATL (trust scoring)

**Data Model Highlights**:
```rust
pub struct Relationship {
    pub from: AgentPubKey,
    pub to: AgentPubKey,
    pub relationship_type: RelationshipType,
    pub strength: f64,
    pub contexts: Vec<String>,
    pub declared_at: Timestamp,
    pub mutual: bool,
}

pub enum RelationshipType {
    Professional { domain: String },
    Personal { nature: String },
    Organizational { org: String },
    Mentorship,
    Collaboration,
    Community { community: String },
}
```

---

### 11. Mycelix-Anchor
**Purpose**: Physical location binding and verification

**Core Functions**:
- Location attestation
- Physical asset registration
- Geo-bounded communities
- Location-based access

**Key Integrations**:
- SupplyChain (asset tracking)
- Oracle (GPS verification)
- Commons (geographic resources)
- Nexus (event venues)

**Use Cases**:
- Proof of physical presence
- Geographic community membership
- Asset location tracking
- Place-based governance

---

### 12. Mycelix-Cycle
**Purpose**: Resource circulation and sharing economy

**Core Functions**:
- Lending/borrowing coordination
- Gift economy tracking
- Circular economy flows
- Waste-to-resource matching

**Key Integrations**:
- Marketplace (value exchange)
- Commons (shared resources)
- SupplyChain (material tracking)
- MATL (trust for lending)

**Design Philosophy**:
Move beyond linear economy (extract-use-dispose) to circular economy (use-return-reuse).

---

## Tier 3: Advanced hApps (7)

### 13. Mycelix-Diplomat
**Purpose**: Inter-community relations and federation

**Core Functions**:
- Treaty negotiation
- Inter-community messaging
- Resource sharing agreements
- Conflict mediation

**Key Integrations**:
- Agora (treaty ratification)
- Arbiter (inter-community disputes)
- Treasury (inter-community payments)

**Federation Model**:
```rust
pub struct Treaty {
    pub treaty_id: String,
    pub parties: Vec<GovernanceSpace>,
    pub terms: Vec<TreatyTerm>,
    pub ratification_status: HashMap<String, RatificationStatus>,
    pub effective_from: Option<Timestamp>,
    pub expires_at: Option<Timestamp>,
}

pub enum TreatyTerm {
    MutualRecognition { scope: String },
    ResourceSharing { resources: Vec<String>, terms: String },
    DisputeResolution { method: String },
    JointGovernance { domain: String, mechanism: String },
}
```

---

### 14. Mycelix-Forecast
**Purpose**: Collective prediction and futarchy

**Core Functions**:
- Prediction market creation
- Outcome resolution
- Track record tracking
- Policy futarchy support

**Key Integrations**:
- Oracle (outcome verification)
- Agora (futarchy voting)
- Chronicle (prediction history)
- MATL (prediction reputation)

**Use Cases**:
- Policy outcome prediction
- Event forecasting
- Expert opinion aggregation
- Decision support

---

### 15. Mycelix-Legacy
**Purpose**: Generational transfer and succession

**Core Functions**:
- Digital will creation
- Asset succession planning
- Knowledge transfer
- Memory preservation

**Key Integrations**:
- Attest (identity verification)
- Chronicle (permanent storage)
- Treasury (asset transfer)
- HealthVault (end-of-life wishes)

**Ethical Considerations**:
- Privacy of deceased
- Dispute resolution for succession
- Long-term key management

---

### 16. Mycelix-Resonance
**Purpose**: Value alignment and collective identity

**Core Functions**:
- Value definition and articulation
- Alignment scoring
- Community fit assessment
- Collective identity formation

**Key Integrations**:
- Attest (value attestation)
- Agora (value governance)
- Pulse (value sentiment)
- Chronicle (value history)

**Philosophy**:
How do communities form around shared values? How do we measure alignment without enforcing conformity?

---

### 17. Mycelix-Membrane
**Purpose**: Boundary management and permeability

**Core Functions**:
- Community boundary definition
- Access control policies
- Gradual integration pathways
- Exit procedures

**Key Integrations**:
- Attest (membership verification)
- Agora (boundary governance)
- Diplomat (inter-community access)

**Metaphor**:
Like a cell membrane: selective permeability, not a wall. Some things flow freely, others require specific channels.

---

### 18. Mycelix-Metabolism
**Purpose**: System health and sustainability monitoring

**Core Functions**:
- Resource flow tracking
- System health metrics
- Sustainability assessment
- Bottleneck detection

**Key Integrations**:
- All hApps (aggregated metrics)
- Sentinel (anomaly correlation)
- Oracle (external benchmarks)
- Chronicle (historical trends)

**Metrics Tracked**:
- Transaction velocity
- Participation rates
- Resource utilization
- Trust network density
- Governance responsiveness

---

### 19. Mycelix-Emergence
**Purpose**: Pattern detection and collective intelligence

**Core Functions**:
- Cross-hApp pattern analysis
- Emergent behavior detection
- Collective learning synthesis
- Innovation identification

**Key Integrations**:
- 0TML (federated learning)
- Chronicle (historical patterns)
- Sentinel (anomaly patterns)
- Pulse (sentiment patterns)

**AI/ML Capabilities**:
- Unsupervised pattern discovery
- Trend prediction
- Anomaly classification
- Recommendation generation

---

## hApp Dependency Graph

```
                            [Core Infrastructure]
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
       [Holochain]            [Mycelix-SDK]           [Bridge Protocol]
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
   [Identity/Trust]           [Coordination]              [Economic]
        │                           │                           │
   ┌────┴────┐              ┌───────┴───────┐           ┌──────┴──────┐
   │         │              │               │           │             │
 Attest    MATL          Agora          Arbiter    Marketplace   Treasury
   │         │              │               │           │             │
   └────┬────┘              │               │           └──────┬──────┘
        │                   │               │                  │
   ┌────┴────┐         ┌────┴────┐     ┌────┴────┐        ┌────┴────┐
   │         │         │         │     │         │        │         │
 Weave   Chronicle   Collab    Guild  Arbiter  Mutual  Commons   Cycle
                                                │
                                            ┌───┴───┐
                                            │       │
                                        Oracle  Sentinel
```

---

## Priority Ranking for Implementation

### Phase 1 (2025 Q1-Q2): Core + Tier 1
1. Existing hApps (Mail, Marketplace, Praxis, SupplyChain, Civitas)
2. HealthVault, Arbiter, Oracle, Attest
3. Agora, Collab, Sentinel, Chronicle

### Phase 2 (2025 Q3-Q4): Essential Tier 2
4. Treasury, Commons, Guild
5. Mutual, Beacon, Nexus
6. Seed, Pulse, Weave
7. Anchor, Cycle, Loom

### Phase 3 (2026): Advanced Tier 3
8. Diplomat, Forecast
9. Legacy, Resonance
10. Membrane, Metabolism
11. Emergence

---

## Cross-Cutting Concerns

### All hApps Must Support:

1. **Bridge Protocol Integration**
   - Register with Bridge
   - Respond to cross-hApp queries
   - Emit events for interested subscribers

2. **MATL Trust Integration**
   - Report reputation impacts
   - Query trust scores
   - Weight interactions by trust

3. **Epistemic Classification**
   - Classify claims on 3D cube
   - Support for ZKP verification
   - Materiality-based retention

4. **Chronicle Archival**
   - Auto-archive M2+ events
   - Support citation
   - Preserve provenance

5. **Sentinel Monitoring**
   - Emit security-relevant events
   - Respond to security queries
   - Support quarantine actions

---

*"A complete civilization requires complete infrastructure. Each hApp is a vital organ in the body of collective intelligence."*
