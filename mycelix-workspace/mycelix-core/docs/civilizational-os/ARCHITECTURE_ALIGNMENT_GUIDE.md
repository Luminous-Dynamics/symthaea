# Mycelix Civilizational OS - Architecture Alignment Guide

> **Version**: 1.0
> **Status**: Canonical Reference
> **Last Updated**: 2025-12-31
> **Purpose**: Ensures all civilizational-os documentation aligns with core Mycelix architecture

---

## 1. Document Hierarchy

All civilizational-os documentation operates under the authority of these foundational documents (in order of precedence):

1. **THE MYCELIX SPORE CONSTITUTION (v0.24)** - Supreme governing document
2. **THE GOVERNANCE CHARTER (v1.0)** - Polycentric DAO structure
3. **THE ECONOMIC CHARTER (v1.0)** - Financial governance & token system
4. **THE EPISTEMIC CHARTER (v2.0)** - Knowledge classification (Epistemic Cube)
5. **THE COMMONS CHARTER (v1.0)** - Commons governance & CGC mechanisms
6. **Mycelix Protocol: Integrated System Architecture v5.2** - Technical architecture

Any conflict between civilizational-os documentation and these foundational documents should be resolved in favor of the foundational documents.

---

## 2. The Epistemic Cube (LEM v2.0)

All data types, claims, and knowledge artifacts in civilizational-os hApps MUST be classified according to the three-dimensional Epistemic Cube:

### 2.1 E-Tier: Empirical Verifiability

| Tier | Name | Description | Example in Civilizational-OS |
|------|------|-------------|------------------------------|
| E0 | Null | Cannot be verified | Personal intuitions, dreams |
| E1 | Testimonial | Based on witness accounts | Meeting minutes, oral histories |
| E2 | Privately Verifiable | Verifiable by limited parties | Member credentials, private contracts |
| E3 | Cryptographically Proven | Mathematically verifiable | DHT entries, token transfers, votes |
| E4 | Publicly Reproducible | Anyone can verify | Open-source code, public algorithms |

### 2.2 N-Tier: Normative Authority

| Tier | Name | Description | Example in Civilizational-OS |
|------|------|-------------|------------------------------|
| N0 | Personal | Individual opinion only | Personal preferences, beliefs |
| N1 | Communal | Local community consensus | Local DAO decisions, circle agreements |
| N2 | Network | Network-wide consensus | Global DAO resolutions, protocol rules |
| N3 | Axiomatic | Constitutional/foundational | Core rights, constitutional amendments |

### 2.3 M-Tier: Materiality (Persistence)

| Tier | Name | Description | Example in Civilizational-OS |
|------|------|-------------|------------------------------|
| M0 | Ephemeral | Temporary, non-stored | Real-time chat, live collaboration |
| M1 | Temporal | Time-limited storage | Session data, temporary credentials |
| M2 | Persistent | Long-term storage | Member profiles, historical records |
| M3 | Foundational | Immutable/canonical | Constitution, charter amendments |

### 2.4 Classification Examples for hApp Data Types

```rust
// Crisis Response Alert
pub struct CrisisAlert {
    // E2: Verifiable by emergency coordinators
    // N1: Communal authority (local crisis team)
    // M2: Persistent (historical record)
    epistemic_classification: (E2, N1, M2),
}

// Collective Intelligence Insight
pub struct CollectiveInsight {
    // E1: Testimonial (aggregated opinions)
    // N1: Communal (circle consensus)
    // M2: Persistent
    epistemic_classification: (E1, N1, M2),
}

// Knowledge Repository Entry
pub struct WisdomEntry {
    // E3: Cryptographically proven (signed, timestamped)
    // N2: Network authority (validated by Knowledge Council)
    // M2: Persistent
    epistemic_classification: (E3, N2, M2),
}

// System Evolution Proposal (MIP)
pub struct EvolutionProposal {
    // E3: Cryptographically proven
    // N2-N3: Network or Axiomatic (depends on MIP type)
    // M3: Foundational (if approved)
    epistemic_classification: (E3, N2_N3, M3),
}
```

---

## 3. Token System Integration

All economic mechanisms in civilizational-os MUST align with the three-token system defined in the Economic Charter:

### 3.1 CIV (Civic Standing)

- **Nature**: Non-transferable reputation token
- **Acquisition**: Earned through verified contributions
- **Function**: Governance weight, access tiers, trust scoring
- **Decay**: Subject to activity-based decay (use it or lose it)

**Civilizational-OS Applications**:
- Facilitation Leadership: CIV gates access to higher facilitation tiers
- Crisis Response: CIV weight determines coordinator eligibility
- Knowledge Management: CIV influences curation authority
- Collective Intelligence: CIV affects synthesis voting weight

### 3.2 CGC (Civic Gifting Credit)

- **Nature**: Transferable social signal currency
- **Acquisition**: Regular distribution to all members
- **Function**: Recognize contributions, fund commons, express gratitude
- **Constraint**: Cannot be exchanged for fiat (pure gift economy)

**Civilizational-OS Applications**:
- Facilitation: CGC rewards for excellent facilitation
- Knowledge: CGC recognition for wisdom contributions
- Crisis Response: CGC appreciation for emergency responders
- Collective Intelligence: CGC rewards for valuable insights

### 3.3 FLOW (Utility Token - Optional)

- **Nature**: Transferable utility token
- **Function**: Resource allocation, service access, external exchange
- **Governance**: Subject to Economic Charter constraints

**Civilizational-OS Applications**:
- Resource allocation for major initiatives
- Access to premium features (if implemented)
- Cross-network value transfer

### 3.4 MATL Integration (Mycelix Adaptive Trust Layer)

All civilizational-os features involving trust or reputation MUST integrate with MATL:

```
MATL Score = f(CIV_standing, CGC_given/received, Activity_metrics,
               Identity_assurance, Network_vouching)
```

---

## 4. Governance Structure Mapping

All governance patterns in civilizational-os MUST map to the formal DAO tier structure:

### 4.1 DAO Tier Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                      GLOBAL DAO                              │
│  ┌─────────────────────┬─────────────────────────────────┐  │
│  │   Citizen Chamber   │      Knowledge Council          │  │
│  │   (1 member 1 vote) │   (Merit-weighted expertise)    │  │
│  └─────────────────────┴─────────────────────────────────┘  │
│                    Bicameral Consensus                       │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                    REGIONAL DAOs                             │
│         (Geographic coordination - cities, regions)          │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                     SECTOR DAOs                              │
│    (Thematic: Energy, Housing, Food, Healthcare, etc.)       │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                      LOCAL DAOs                              │
│      (Neighborhoods, organizations, communities)             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Civilizational-OS Governance Mapping

| Civilizational-OS Component | Primary DAO Tier | Secondary Tier |
|-----------------------------|------------------|----------------|
| Wisdom Circles | Local DAO | Sector DAO |
| Crisis Response Teams | Local/Regional DAO | Sector DAO |
| Knowledge Council Integration | Global DAO | - |
| Facilitation Circles | Local DAO | Regional DAO |
| Protocol Stewards (Evolution) | Global DAO | - |
| hApp Maintainer Circles | Sector DAO | Global DAO |
| Community Voice Assembly | Regional DAO | Global DAO |

### 4.3 Voting Mechanisms Reference

All voting in civilizational-os MUST use mechanisms defined in the Governance Charter:

- **Simple Majority**: Routine decisions (>50% + quorum)
- **Supermajority**: Significant changes (>66.67% + quorum)
- **Consensus Minus N**: Sensitive matters (all but N objections)
- **Quadratic Voting**: Resource allocation (sqrt weighting)
- **Conviction Voting**: Long-term priorities (time-weighted)

---

## 5. MIP (Mycelix Improvement Proposal) Categories

All system evolution proposals MUST be categorized according to the MIP framework:

### 5.1 MIP Categories

| Category | Code | Description | Approval Path |
|----------|------|-------------|---------------|
| Technical | MIP-T | Protocol, code, infrastructure | Protocol Stewards → Global DAO |
| Economic | MIP-E | Token mechanics, incentives | Economic Council → Global DAO |
| Governance | MIP-G | Voting rules, DAO structure | Governance Council → Global DAO |
| Social | MIP-S | Community norms, practices | Community Assembly → Global DAO |
| Constitutional | MIP-C | Charter/Constitution amendments | Supermajority + Ratification |

### 5.2 Civilizational-OS Evolution Mapping

| Evolution Type | MIP Category | Authority |
|----------------|--------------|-----------|
| hApp Feature Updates | MIP-T | hApp Maintainer Circle |
| Economic Model Changes | MIP-E | Economic Council |
| Governance Pattern Changes | MIP-G | Governance Council |
| Community Practice Changes | MIP-S | Community Voice Assembly |
| Core Protocol Changes | MIP-T + MIP-G | Protocol Stewards |
| Foundational Changes | MIP-C | Constitutional Process |

---

## 6. Key Institutional References

All civilizational-os documentation should reference these canonical institutions:

### 6.1 Knowledge Council
- **Role**: Epistemic oversight, knowledge validation, truth-seeking
- **Authority**: E-Tier classification, N2 normative decisions
- **Integration**: Wisdom Capture, Collective Intelligence validation

### 6.2 Audit Guild
- **Role**: Financial accountability, economic integrity
- **Authority**: Token system audits, economic model validation
- **Integration**: Resource allocation oversight, CGC distribution audits

### 6.3 Member Redress Council
- **Role**: Dispute resolution, grievance handling
- **Authority**: Binding arbitration within network
- **Integration**: Crisis Response appeals, facilitation complaints

### 6.4 Foundation
- **Role**: Legal entity, external interface, emergency powers
- **Authority**: Golden Veto (emergency constitutional protection)
- **Integration**: External crisis coordination, legal compliance

---

## 7. Technical Standards

### 7.1 Holochain Version Alignment

All civilizational-os hApps MUST target:
- **HDK**: 0.6.0
- **HDI**: 0.7.0
- **Holochain**: 0.6.x

### 7.2 Entry Type Standards

All entry types MUST include:

```rust
pub struct StandardEntry {
    // Required fields
    pub id: ActionHash,
    pub created_at: Timestamp,
    pub created_by: AgentPubKey,

    // Epistemic classification
    pub epistemic_e: EpistemicTierE,  // E0-E4
    pub epistemic_n: EpistemicTierN,  // N0-N3
    pub epistemic_m: EpistemicTierM,  // M0-M3

    // Governance reference
    pub governance_tier: GovernanceTier,  // Local/Sector/Regional/Global

    // Content (type-specific)
    pub content: T,
}
```

### 7.3 Zome Naming Convention

```
{happ_name}_{zome_type}_{function}
e.g., crisis_response_coordinator_alert
      collective_intelligence_integrity_insight
      knowledge_management_coordinator_wisdom
```

---

## 8. Cross-Reference Table

| Civilizational-OS Document | Primary Architecture References |
|---------------------------|--------------------------------|
| COMPLETE_HAPP_CATALOG.md | Architecture v5.2, SDK Design |
| COLLECTIVE_INTELLIGENCE_GUIDE.md | Epistemic Charter, Governance Charter |
| CRISIS_RESPONSE_PROTOCOL.md | Governance Charter, Commons Charter |
| KNOWLEDGE_WISDOM_SYSTEM.md | Epistemic Charter, DKG Plan |
| FACILITATION_CURRICULUM.md | Governance Charter, Constitution |
| SYSTEM_EVOLUTION_PROTOCOL.md | Constitution, All Charters |

---

## 9. Validation Checklist

When creating or updating civilizational-os documentation, verify:

- [ ] Document references its position in the document hierarchy
- [ ] All data types have E/N/M tier classifications
- [ ] Economic mechanisms use CIV/CGC/FLOW terminology
- [ ] Governance patterns map to Local/Sector/Regional/Global tiers
- [ ] Voting mechanisms reference Governance Charter options
- [ ] Proposals use MIP category codes
- [ ] Key institutions are referenced where appropriate
- [ ] Technical specs align with Holochain 0.6.x standards
- [ ] Entry types include standard fields

---

## 10. Amendment Process

Updates to this alignment guide require:

1. **Minor Updates** (typos, clarifications): Direct edit by Protocol Stewards
2. **Mapping Changes**: MIP-G proposal + Governance Council approval
3. **Structural Changes**: MIP-G proposal + Global DAO ratification
4. **Foundational References**: MIP-C process (constitutional amendment)

---

*This guide ensures the Mycelix Civilizational OS documentation remains in harmony with the foundational architecture while enabling the ecosystem to evolve coherently.*

**Document Authority**: Canonical reference for all civilizational-os documentation alignment.
