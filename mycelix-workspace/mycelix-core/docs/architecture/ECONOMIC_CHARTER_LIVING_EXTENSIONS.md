# **ECONOMIC CHARTER: LIVING EXTENSIONS (v1.1)**

*Updated March 2026: CIV renamed MYCEL, FLOW renamed SAP, CGC renamed MYCEL recognition — aligned with production implementation*

**Amendments to the Economic Charter (v1.0) under the Living Protocol Initiative**

---

## Editor's Note

This document specifies two extensions to the Economic Charter (v1.0):

1. **Wound Healing Protocol** (Article II-A): A regenerative alternative to punitive slashing for good-faith failures.

2. **Kenosis Protocol** (Article II-B): A voluntary mechanism for releasing accumulated MYCEL to the commons.

These extensions are activated via MIP-E (Economic) proposal and integrated with existing Economic Charter provisions. They align with the Metabolism Charter (v1.0) temporal framework and the Constitutional principle of Reciprocity and Fair Exchange.

**Constitutional Alignment**: Constitution Article I, Section 2 (Reciprocity), Article VI (Member Rights), Article VII (Justice and Redress).

---

## **ARTICLE II-A: WOUND HEALING PROTOCOL**

### **Section 1. Purpose and Scope**

**1.1 Purpose**: The Wound Healing Protocol provides a regenerative alternative to punitive slashing when harm occurs through good-faith failure rather than Byzantine malice.

**1.2 Philosophical Basis**: Recognizing that:
- Humans err; communities that heal errors are more resilient than those that only punish
- Accountability and restoration are not mutually exclusive
- Some harms cannot be quantified in tokens alone
- Relationships have value beyond individual transactions

**1.3 Scope**: This Protocol applies when:
- Harm has occurred within the Network
- The harm was not the result of intentional Byzantine behavior
- The harmer acknowledges responsibility
- Both harmed and harmer consent to the healing process

**1.4 Exclusions**: This Protocol does NOT apply to:
- Proven Byzantine behavior (use MATL slashing)
- Sybil attacks (use identity quarantine)
- Validator collusion (use Economic Charter Section 5.6)
- Unacknowledged harm (use Member Redress Council)

### **Section 2. Wound Classification**

#### **2.1 Wound Types**

| Type | Definition | Example |
|------|------------|---------|
| **Bilateral** | Harm between two identified parties | Alice's code broke Bob's integration |
| **Self-Inflicted** | Harm an agent causes to themselves | Burnout from overcommitment |
| **Systemic** | Harm from policy affecting many | DAO decision harmed a cohort |
| **Commons** | Harm to shared resources | Accidental data corruption |
| **Inherited** | Harm from onboarding into dysfunction | New member joins broken process |

#### **2.2 Wound Severity**

| Severity | Impact | Escrow Requirement | Resolution Body |
|----------|--------|-------------------|-----------------|
| **Minor** | <1,000 SAP equivalent | 1% of harmer's stake | Peer mediation |
| **Moderate** | 1,000-10,000 SAP | 5% of harmer's stake | Sector DAO panel |
| **Severe** | 10,000-100,000 SAP | 20% of harmer's stake | MRC mediation |
| **Critical** | >100,000 SAP or rights violation | 50% of harmer's stake | MRC + Global DAO |

#### **2.3 Wound Categories**

| Category | Definition | Typical Resolution |
|----------|------------|-------------------|
| **BrokenCommitment** | Failed to deliver promised work | Restitution + process improvement |
| **Miscommunication** | Harm from misunderstanding | Clarification + reconciliation |
| **CapacityFailure** | Couldn't meet obligations | Workload adjustment + support |
| **NormViolation** | Broke community norms | Education + commitment renewal |
| **TrustBreach** | Betrayed confidence | Extended accountability period |

### **Section 3. Healing Phases**

The Wound Healing Protocol follows a four-phase process inspired by biological wound healing:

#### **3.1 Inflammation Phase**

**Purpose**: Acknowledge the harm; contain the damage.

**Duration**: 1-7 days

**Requirements**:
- Wound formally created by harmed party or witness
- Harmer notified within 24 hours
- Escrow locked (per severity table)
- Initial documentation of harm

**Outputs**:
- Wound Record created on DHT
- Parties identified and notified
- Escrow contract deployed
- Phase 1 complete attestation

**Transition Criteria**:
- Harmer acknowledges the wound exists (not necessarily fault)
- Both parties consent to healing process
- OR 7 days elapsed (auto-advance with "contested" flag)

#### **3.2 Proliferation Phase**

**Purpose**: Understand root cause; document fully.

**Duration**: 7-21 days

**Requirements**:
- Root cause analysis documented
- Witnesses heard (if any)
- Full impact assessment
- Restitution amount proposed

**Outputs**:
- Root Cause Document
- Impact Assessment
- Witness Statements (optional)
- Proposed Restitution Agreement

**Transition Criteria**:
- Both parties agree on root cause (or mediator determines)
- Both parties agree on restitution amount
- OR mediator issues binding determination

#### **3.3 Remodeling Phase**

**Purpose**: Make amends; repair trust.

**Duration**: 14-56 days

**Requirements**:
- Restitution paid (may be staged)
- Process improvements implemented
- Apology delivered and accepted
- Relationship repair actions taken

**Outputs**:
- Restitution Payment Record
- Process Improvement Commitments
- Relationship Repair Attestations
- Phase 3 complete attestation

**Transition Criteria**:
- Restitution >= restitution_required
- Harmed party attests to acceptance
- OR mediator certifies good-faith completion

#### **3.4 Integration Phase**

**Purpose**: Restore relationship; release escrow.

**Duration**: 7-28 days

**Requirements**:
- Final reconciliation meeting (optional but encouraged)
- Lessons extracted for commons
- Escrow released
- Wound status updated

**Outputs**:
- Integration Attestation (both parties)
- Lessons Learned Document (for Wisdom Library)
- Escrow Release Transaction
- Final Wound Record (status: Healed)

**Transition Criteria**:
- Both parties sign Integration Attestation
- OR 28 days elapsed post-restitution (auto-complete)

### **Section 4. Wound Escrow Mechanics**

#### **4.1 Escrow Contract**

Wounds use the `WoundEscrow.sol` smart contract on the L4 Bridge layer:

```solidity
// Simplified interface
interface IWoundEscrow {
    function createWound(
        bytes32 woundId,
        address harmer,
        address harmed,
        uint256 escrowAmount,
        uint256 restitutionRequired,
        WoundSeverity severity
    ) external;

    function advancePhase(bytes32 woundId) external;

    function payRestitution(bytes32 woundId, uint256 amount) external;

    function releaseEscrow(bytes32 woundId) external;

    function escalate(bytes32 woundId) external; // To MRC
}
```

#### **4.2 Escrow Source**

Escrow may be drawn from:
- Harmer's staked SAP (primary)
- Harmer's locked MYCEL equivalent (secondary)
- Wound Insurance Pool (if harmer insolvent)

#### **4.3 Escrow Release**

Upon successful Integration:
- Full escrow returned to harmer
- Restitution paid from escrow to harmed
- Excess returned to harmer
- If restitution > escrow: harmer debt created

### **Section 5. Failure Modes**

#### **5.1 Stalled Healing**

If a wound stalls in any phase for >2 cycles (56 days):
- Automatic escalation to Member Redress Council
- MRC may: force advancement, convert to slash, or dismiss

#### **5.2 Bad Faith Participation**

If either party demonstrates bad faith:
- Wound converts to standard MATL slashing
- Bad faith party receives 2x penalty
- Good faith party receives escrow

#### **5.3 Repeat Offenses**

| Offense Count (12 months) | Consequence |
|---------------------------|-------------|
| 1st wound | Standard healing process |
| 2nd wound | Healing + 1.5x escrow |
| 3rd wound | Healing + 2x escrow + probation |
| 4th+ wound | Mandatory MRC review; possible MATL slash |

#### **5.4 Scarred Status**

Wounds that complete but leave lasting impact may receive "Scarred" status:
- Wound record permanently flagged
- Harmer's MATL profile notes scar
- Does not prevent participation
- Used for pattern detection

### **Section 6. Integration with MATL**

#### **6.1 Trust Score Impact**

| Wound Outcome | MATL Impact |
|---------------|-------------|
| Healed (full reconciliation) | +0.02 resilience bonus (both parties) |
| Healed (mediated) | Neutral |
| Scarred | -0.05 for harmer |
| Escalated to slash | Standard slash penalty |

#### **6.2 Healing Track Record**

MATL tracks:
- Wounds healed (positive signal)
- Wounds initiated (neutral)
- Wounds escalated (negative signal)
- Time to healing (efficiency signal)

### **Section 7. Metabolism Integration**

Wound Healing aligns with the Metabolism Cycle:

| Phase | Wound Activity |
|-------|----------------|
| **Release** | Wound creation encouraged; acknowledgment |
| **Stillness** | Root cause analysis; listening |
| **Creation** | Restitution planning; process design |
| **Integration** | Completion; lessons harvesting |

Wounds may span multiple cycles. Each phase may complete in any Metabolism phase.

---

## **ARTICLE II-B: KENOSIS PROTOCOL**

### **Section 1. Purpose and Philosophy**

**1.1 Definition**: Kenosis (Greek: κένωσις, "emptying") is the voluntary release of accumulated reputation (MYCEL) or resources (SAP) to the commons pool.

**1.2 Philosophical Basis**: Recognizing that:
- Accumulated power tends toward concentration
- Generosity strengthens community bonds
- Renewal requires release
- Status can be held lightly

**1.3 Purpose**: The Kenosis Protocol enables agents to:
- Redistribute accumulated influence voluntarily
- Signal deep commitment to collective welfare
- Support newcomer onboarding
- Practice non-attachment to status

**1.4 Non-Purposes**: Kenosis is NOT:
- Mandatory (always voluntary)
- Punitive (not a penalty mechanism)
- Transferable (cannot kenosis "to" a specific recipient)
- Reversible (commitments are irrevocable)

### **Section 2. Kenosis Mechanics**

#### **2.1 Release Types**

| Type | What is Released | Mechanism |
|------|------------------|-----------|
| **MYCEL Kenosis** | Civic Standing (reputation) | MYCEL reduced; goes to Newcomer Pool |
| **SAP Kenosis** | Utility tokens | Burned or pooled (configurable) |
| **Voting Kenosis** | Governance weight | Temporarily reduced voting power |
| **Blended Kenosis** | Combination of above | Proportional release |

#### **2.2 Release Limits**

| Limit | Value | Rationale |
|-------|-------|-----------|
| **Maximum per cycle** | 20% of holdings | Prevent impulsive over-release |
| **Minimum floor** | 0.1 MYCEL / 100 SAP | Maintain minimum participation |
| **Cooldown** | 1 cycle between releases | Prevent gaming |
| **Lifetime maximum** | 80% cumulative | Ensure agent retains stake |

#### **2.3 Release Destinations**

Kenosis releases flow to designated pools:

| Pool | Allocation | Purpose |
|------|------------|---------|
| **Newcomer Pool** | 50% | Boosts new member starting reputation |
| **Wound Fund** | 30% | Funds wound healing escrow for insolvent harmers |
| **Research Pool** | 20% | Funds experimental governance and research |

Pool allocations may be adjusted via MIP-E with ⅔ supermajority.

### **Section 3. Kenosis Process**

#### **3.1 Commitment Phase**

1. Agent declares kenosis intention
2. Specifies: release type, amount, justification (optional)
3. System validates against limits
4. Commitment recorded (irrevocable)

#### **3.2 Execution Phase**

Execution occurs during the Integration phase of the current Metabolism Cycle:

1. MYCEL/SAP deducted from agent
2. Value distributed to destination pools
3. Recognition recorded in agent's profile
4. Kenosis event published to network

#### **3.3 Smart Contract Interface**

```solidity
// KenosisBurn.sol interface
interface IKenosisBurn {
    function commitKenosis(
        bytes32 agentId,
        ReleaseType releaseType,
        uint256 basisPoints, // 1-2000 (0.01% - 20%)
        string calldata justification
    ) external returns (bytes32 commitmentId);

    function executeKenosis(bytes32 commitmentId) external;

    function getAgentKenosisHistory(bytes32 agentId)
        external view returns (KenosisRecord[] memory);
}
```

### **Section 4. Recognition and Incentives**

#### **4.1 Recognition Tiers**

| Cumulative Kenosis | Recognition | Benefits |
|-------------------|-------------|----------|
| 1-5% | "Contributor" | Badge display |
| 5-10% | "Generous" | 1.05x MYCEL recognition multiplier |
| 10-20% | "Selfless" | Advisory voice on pool allocation |
| 20%+ | "Elder" | Emeritus recognition |

#### **4.2 Non-Monetary Benefits**

Kenosis contributors receive:
- Public recognition in MATL profile
- Invitation to Kenosis Council (advisory body for pool allocation)
- Priority access to Research Pool grants (if applying)
- Ceremony acknowledgment during Integration phase

#### **4.3 Anti-Gaming Provisions**

To prevent gaming:
- Kenosis cannot be used to avoid slashing (already-committed slashes unaffected)
- Kenosis during active wound healing does not affect escrow
- Kenosis followed by rapid re-accumulation triggers review
- Coordinated mass kenosis (>10 agents, >10% each, same cycle) requires Knowledge Council review

### **Section 5. Pool Governance**

#### **5.1 Newcomer Pool**

**Purpose**: Boost starting reputation for new verified members.

**Allocation**: New members receive bonus MYCEL from pool:
- Standard bonus: 0.05 MYCEL (on top of 0.1 baseline)
- Pool-dependent: If pool depleted, bonus reduces proportionally

**Governance**: Automatic distribution; no discretion.

#### **5.2 Wound Fund**

**Purpose**: Provide escrow for insolvent harmers in wound healing.

**Allocation**: Used when harmer cannot meet escrow requirement:
- Maximum coverage: 50% of required escrow
- Harmer incurs debt to fund
- Repayment required before new wounds can heal

**Governance**: MRC authorizes draws; automatic repayment tracking.

#### **5.3 Research Pool**

**Purpose**: Fund experimental governance and protocol research.

**Allocation**: Via quarterly grant rounds:
- Proposals submitted during Creation phase
- Voting during Integration phase
- Quadratic funding mechanism

**Governance**: Kenosis Council advises; Global DAO ratifies.

### **Section 6. MATL Integration**

#### **6.1 Trust Score Impact**

| Kenosis Action | MATL Impact |
|----------------|-------------|
| Small kenosis (1-5%) | +0.01 temporary bonus (3 cycles) |
| Medium kenosis (5-10%) | +0.02 permanent bonus |
| Large kenosis (10-20%) | +0.03 permanent bonus |
| Recognized Elder (20%+) | +0.05 permanent bonus |

#### **6.2 Kenosis History**

MATL tracks:
- Total kenosis (lifetime)
- Kenosis frequency
- Kenosis timing (aligned with Integration phase = bonus)
- Pool destination choices

### **Section 7. Metabolism Integration**

Kenosis aligns with the Metabolism Cycle:

| Phase | Kenosis Activity |
|-------|------------------|
| **Release** | Contemplation of what to release |
| **Stillness** | Reflection on generosity |
| **Creation** | (Kenosis discouraged - focus on creation) |
| **Integration** | Commitment and execution encouraged |

Kenosis commitments made during Integration phase receive 1.1x recognition bonus.

---

## **ARTICLE II-C: COMPOSTING EXTENSION**

*(Reference to Commons Charter)*

This extension adds Composting as a Commons mechanism. See **Commons Charter Extension: Composting Protocol** for full specification.

**Summary**: Composting is the graceful retirement of failed patterns, outdated knowledge, or deprecated structures. It operates as a Commons mechanism (like TEND or ROOT) and integrates with:
- Release phase of Metabolism Cycle
- Wound Healing (composting of harmful patterns)
- DKG (composting of superseded claims)

---

## **APPENDIX A – DATA SCHEMAS**

### **A.1 Wound Record Schema**

```rust
pub struct WoundRecord {
    pub wound_id: Uuid,
    pub created_at: Timestamp,
    pub wound_type: WoundType,
    pub severity: WoundSeverity,
    pub category: WoundCategory,
    pub phase: WoundPhase,

    // Parties
    pub harmed: DID,
    pub harmer: DID,
    pub witnesses: Vec<DID>,
    pub mediator: Option<DID>,

    // Financial
    pub escrow_contract: Address,
    pub escrow_amount: u64,
    pub restitution_required: u64,
    pub restitution_paid: u64,

    // Documentation
    pub initial_description: String,
    pub root_cause_document: Option<ContentHash>,
    pub impact_assessment: Option<ContentHash>,
    pub lessons_learned: Option<ContentHash>,

    // Status
    pub phase_history: Vec<PhaseTransition>,
    pub final_status: Option<WoundFinalStatus>,
}

pub enum WoundPhase {
    Inflammation,
    Proliferation,
    Remodeling,
    Integration,
}

pub enum WoundFinalStatus {
    Healed,
    Scarred,
    Escalated,
    Dismissed,
}
```

### **A.2 Kenosis Record Schema**

```rust
pub struct KenosisRecord {
    pub commitment_id: Uuid,
    pub agent_did: DID,
    pub committed_at: Timestamp,
    pub executed_at: Option<Timestamp>,
    pub cycle_number: u32,

    // Release specification
    pub release_type: ReleaseType,
    pub amount_basis_points: u16, // 1-2000
    pub actual_amount: u64,
    pub justification: Option<String>,

    // Destination
    pub pool_allocation: PoolAllocation,

    // Recognition
    pub recognition_tier: RecognitionTier,
    pub integration_phase_bonus: bool,
}

pub enum ReleaseType {
    MYCEL,
    SAP,
    Voting,
    Blended { civ_pct: u8, flow_pct: u8, voting_pct: u8 },
}

pub struct PoolAllocation {
    pub newcomer_pool: u64,
    pub wound_fund: u64,
    pub research_pool: u64,
}
```

---

## **APPENDIX B – INTEGRATION DIAGRAM**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ECONOMIC CHARTER: LIVING EXTENSIONS                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │  WOUND HEALING  │    │     KENOSIS     │    │   COMPOSTING    │     │
│  │                 │    │                 │    │                 │     │
│  │ Inflammation    │    │ Commitment      │    │ Initiation      │     │
│  │ Proliferation   │    │ Execution       │    │ Decomposition   │     │
│  │ Remodeling      │    │ Recognition     │    │ Mineralization  │     │
│  │ Integration     │    │                 │    │                 │     │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘     │
│           │                      │                      │              │
│           ▼                      ▼                      ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        INTEGRATION LAYER                         │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │                                                                   │   │
│  │  MATL ◀───────── Trust impacts ──────────▶ MYCEL/SAP             │   │
│  │                                                                   │   │
│  │  Metabolism Cycle ◀─── Phase alignment ───▶ Timing bonuses       │   │
│  │                                                                   │   │
│  │  MRC ◀─────────── Escalation ─────────────▶ Dispute resolution   │   │
│  │                                                                   │   │
│  │  DKG ◀─────────── Records ────────────────▶ Lessons library      │   │
│  │                                                                   │   │
│  │  Smart Contracts ◀─ Escrow/Burn ──────────▶ L4 Bridge            │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**Version**: 1.0
**Status**: Draft for Ratification
**Published**: February 2026
**Amends**: Economic Charter v1.0 (Articles II-A, II-B, II-C)
**Related Documents**:
- [Economic Charter v1.0](./THE%20ECONOMIC%20CHARTER%20(v1.0).md)
- [Metabolism Charter v1.0](./THE%20METABOLISM%20CHARTER%20(v1.0).md)
- [Commons Charter v1.0](./THE%20COMMONS%20CHARTER%20(v1.0).md)
- [MATL Whitepaper](../0TML/docs/06-architecture/matl_whitepaper.md)

---

*"From each according to their generosity, to each according to their need for healing."*
