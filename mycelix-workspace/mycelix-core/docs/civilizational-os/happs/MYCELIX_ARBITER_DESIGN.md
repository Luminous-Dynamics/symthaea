# Mycelix-Arbiter: Decentralized Dispute Resolution

## Vision & Design Document

**Version**: 1.0.0
**Created**: December 30, 2025
**Status**: Design Phase
**Priority**: Tier 1 - High Synergy

---

## Executive Summary

Mycelix-Arbiter is a decentralized dispute resolution system that provides fair, transparent, and efficient conflict resolution across the Mycelix ecosystem. By leveraging the Epistemic Charter for evidence classification and MATL trust scores for arbitrator selection, Arbiter creates a justice system that is both rigorous and accessible.

### Why Arbiter?

Every coordination system eventually encounters disputes:
- **Marketplace**: Buyer claims product wasn't delivered
- **SupplyChain**: Supplier disputes quality assessment
- **Praxis**: Student challenges credential denial
- **Collab**: Team members disagree on contribution credit

Without dispute resolution, participants must either accept injustice or exit the system. Arbiter provides a third path: fair resolution within the ecosystem.

---

## Core Principles

### 1. Procedural Justice
The process itself must be fair. Parties must feel heard, understand the reasoning, and believe the arbitrators were impartial.

### 2. Epistemic Rigor
Evidence is classified using the 3D Epistemic Cube. Higher empirical evidence (E3-E4) carries more weight than testimonial claims (E1).

### 3. Proportional Stakes
The dispute resolution process should be proportional to the stakes involved. A $10 dispute doesn't need a 3-person tribunal.

### 4. Reputation Consequences
Dishonest behavior in disputes affects MATL scores. This creates incentives for good-faith participation.

### 5. Appeal Rights
Parties can appeal decisions through escalating tiers, with increasing cost and rigor.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Dispute Resolution Flow                         │
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Dispute │ →  │ Evidence │ →  │  Review  │ →  │ Decision │      │
│  │  Filed   │    │ Submitted│    │  Period  │    │ Rendered │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│       │              │               │               │              │
│       ↓              ↓               ↓               ↓              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Mediation│    │ Epistemic│    │ Arbitrator│   │  Enforce │      │
│  │  Option  │    │ Classify │    │ Deliberate│   │  or      │      │
│  │          │    │          │    │          │    │  Appeal  │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        Arbiter hApp                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      Coordinator Zomes                           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Dispute  │ │ Evidence │ │ Arbitrator│ │ Decision │           ││
│  │  │ Manager  │ │ Manager  │ │ Selection │ │ Engine   │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐                        ││
│  │  │ Mediation│ │ Appeals  │ │ Enforce  │                        ││
│  │  │ Service  │ │ Manager  │ │ Actions  │                        ││
│  │  └──────────┘ └──────────┘ └──────────┘                        ││
│  ├─────────────────────────────────────────────────────────────────┤│
│  │                      Integrity Zomes                             ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           ││
│  │  │ Dispute  │ │ Evidence │ │ Ruling   │ │ Appeal   │           ││
│  │  │ Entry    │ │ Entry    │ │ Entry    │ │ Entry    │           ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                     Cross-hApp Integrations                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Bridge       │  │ MATL         │  │ Source hApp              │   │
│  │ (Reputation) │  │ (Trust)      │  │ (Marketplace/Supply/etc) │   │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Model

### Core Entry Types

```rust
/// A dispute between parties
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Dispute {
    /// Unique dispute identifier
    pub dispute_id: String,
    /// Source hApp where dispute originated
    pub source_happ: String,
    /// Reference to transaction/interaction in source hApp
    pub source_reference: ActionHash,
    /// Party filing the dispute (claimant)
    pub claimant: AgentPubKey,
    /// Party being disputed against (respondent)
    pub respondent: AgentPubKey,
    /// Type of dispute
    pub dispute_type: DisputeType,
    /// Detailed claim
    pub claim: String,
    /// Requested remedy
    pub remedy_sought: RemedySought,
    /// Estimated value at stake
    pub stake_value: StakeValue,
    /// Current status
    pub status: DisputeStatus,
    /// Resolution tier (escalates with appeals)
    pub tier: ResolutionTier,
    /// Filed timestamp
    pub filed_at: Timestamp,
    /// Deadline for response
    pub response_deadline: Timestamp,
    /// Assigned arbitrators (if beyond mediation)
    pub arbitrators: Vec<AgentPubKey>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum DisputeType {
    // Marketplace
    NonDelivery,
    QualityDispute,
    DescriptionMismatch,
    PaymentDispute,
    RefundRequest,

    // Supply Chain
    ProvenanceFraud,
    QualityCertification,
    DeliveryTimeline,
    ContractBreach,

    // Praxis
    CredentialDenial,
    PlagiarismAccusation,
    GradeDispute,

    // General
    ReputationDispute,
    IntellectualProperty,
    PrivacyViolation,
    Other(String),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct RemedySought {
    /// Financial compensation
    pub compensation: Option<CompensationRequest>,
    /// Specific performance requested
    pub specific_action: Option<String>,
    /// Reputation adjustment
    pub reputation_adjustment: Option<ReputationAdjustment>,
    /// Credential action
    pub credential_action: Option<CredentialAction>,
    /// Other remedies
    pub other: Vec<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct StakeValue {
    /// Monetary value (if applicable)
    pub monetary: Option<MonetaryValue>,
    /// Reputation at stake
    pub reputation: f64,
    /// Credential at stake
    pub credential: Option<ActionHash>,
    /// Other value description
    pub other: Option<String>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum DisputeStatus {
    Filed,
    AwaitingResponse,
    InMediation,
    MediationFailed,
    ArbitratorAssigned,
    EvidencePhase,
    Deliberation,
    RulingIssued,
    AppealPeriod,
    Appealed,
    Final,
    Enforced,
    Settled, // Parties resolved directly
    Withdrawn,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionTier {
    /// Direct negotiation between parties
    Tier0Negotiation,
    /// Single mediator (non-binding)
    Tier1Mediation,
    /// Single arbitrator (binding)
    Tier2SingleArbitrator,
    /// Panel of 3 arbitrators
    Tier3Panel,
    /// Community jury (7+ members)
    Tier4CommunityJury,
    /// Constitutional review (for precedent-setting cases)
    Tier5ConstitutionalReview,
}

/// Evidence submitted in a dispute
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Evidence {
    /// Dispute this evidence relates to
    pub dispute: ActionHash,
    /// Who submitted this evidence
    pub submitter: AgentPubKey,
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Evidence content (or hash if large)
    pub content: EvidenceContent,
    /// Epistemic classification
    pub epistemic: EpistemicClaim,
    /// When submitted
    pub submitted_at: Timestamp,
    /// Witnesses who can attest to this evidence
    pub witnesses: Vec<AgentPubKey>,
    /// Cross-hApp reference (if evidence from another hApp)
    pub cross_happ_ref: Option<CrossHappReference>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Written statement/testimony
    Testimony,
    /// On-chain transaction record
    TransactionRecord,
    /// Smart contract/agreement
    ContractRecord,
    /// Communication logs
    CommunicationLog,
    /// External document
    ExternalDocument,
    /// Photo/video evidence
    MediaEvidence,
    /// Expert opinion
    ExpertOpinion,
    /// Cross-hApp data
    CrossHappData,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossHappReference {
    pub happ: String,
    pub entry_type: String,
    pub action_hash: ActionHash,
    pub verified: bool,
    pub verification_method: String,
}

/// Arbitrator profile
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ArbitratorProfile {
    /// Arbitrator's agent key
    pub agent: AgentPubKey,
    /// Dispute types they can arbitrate
    pub specializations: Vec<DisputeType>,
    /// Required Praxis credentials
    pub credentials: Vec<ActionHash>,
    /// Languages spoken
    pub languages: Vec<String>,
    /// MATL trust score (from Bridge)
    pub trust_score: f64,
    /// Cases completed
    pub cases_completed: u32,
    /// Satisfaction rating
    pub satisfaction_rating: f64,
    /// Appeal reversal rate
    pub appeal_reversal_rate: f64,
    /// Availability status
    pub available: bool,
    /// Stake deposited (for accountability)
    pub stake_deposited: MonetaryValue,
}

/// A ruling on a dispute
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Ruling {
    /// Dispute being ruled on
    pub dispute: ActionHash,
    /// Arbitrator(s) who made the ruling
    pub arbitrators: Vec<AgentPubKey>,
    /// The decision
    pub decision: Decision,
    /// Detailed reasoning
    pub reasoning: String,
    /// Evidence considered
    pub evidence_considered: Vec<ActionHash>,
    /// Evidence weight analysis
    pub evidence_analysis: Vec<EvidenceAnalysis>,
    /// Remedies ordered
    pub remedies_ordered: Vec<OrderedRemedy>,
    /// Reputation impacts
    pub reputation_impacts: Vec<ReputationImpact>,
    /// Issued timestamp
    pub issued_at: Timestamp,
    /// Appeal deadline
    pub appeal_deadline: Timestamp,
    /// Is this ruling final?
    pub is_final: bool,
    /// Precedent category (if applicable)
    pub precedent_category: Option<String>,
    /// Dissenting opinions (for panels)
    pub dissenting_opinions: Vec<DissentingOpinion>,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum Decision {
    /// Claimant prevails fully
    ClaimantPrevails,
    /// Respondent prevails fully
    RespondentPrevails,
    /// Mixed decision with specific allocations
    SplitDecision { claimant_portion: f64 },
    /// Claim dismissed (procedural)
    Dismissed(DismissalReason),
    /// Case remanded for additional evidence
    Remanded(RemandReason),
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceAnalysis {
    pub evidence: ActionHash,
    pub weight_assigned: f64, // 0.0 to 1.0
    pub epistemic_upgrade: Option<EmpiricalLevel>, // If evidence corroborated
    pub analysis: String,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct ReputationImpact {
    pub agent: AgentPubKey,
    pub happ: String,
    pub impact: f64, // Positive or negative
    pub reason: String,
}

/// Appeal of a ruling
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Appeal {
    /// Original ruling being appealed
    pub ruling: ActionHash,
    /// Appellant (who's appealing)
    pub appellant: AgentPubKey,
    /// Grounds for appeal
    pub grounds: Vec<AppealGround>,
    /// Detailed argument
    pub argument: String,
    /// New evidence (if allowed)
    pub new_evidence: Vec<ActionHash>,
    /// Appeal fee deposited
    pub fee_deposited: MonetaryValue,
    /// Filed timestamp
    pub filed_at: Timestamp,
    /// Status
    pub status: AppealStatus,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum AppealGround {
    /// Arbitrator bias
    ArbitratorBias,
    /// Procedural error
    ProceduralError,
    /// Evidence improperly excluded
    EvidenceExclusion,
    /// Evidence improperly admitted
    EvidenceAdmission,
    /// Misapplication of rules
    RuleMisapplication,
    /// New evidence discovered
    NewEvidence,
    /// Disproportionate remedy
    DisproportionateRemedy,
}

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub enum AppealStatus {
    Filed,
    UnderReview,
    Accepted,
    Denied,
    Upheld, // Original ruling stands
    Reversed, // Original ruling overturned
    Modified, // Original ruling modified
}
```

### Link Types

```rust
#[hdk_link_types]
pub enum LinkTypes {
    // Dispute organization
    ClaimantToDispute,
    RespondentToDispute,
    SourceHappToDispute,
    DisputeTypeToDispute,
    StatusToDispute,

    // Evidence
    DisputeToEvidence,
    SubmitterToEvidence,
    WitnessToEvidence,

    // Arbitration
    ArbitratorToCase,
    DisputeToArbitrator,
    ArbitratorToProfile,

    // Rulings
    DisputeToRuling,
    RulingToAppeal,
    ArbitratorToRuling,

    // Precedent
    CategoryToPrecedent,
}
```

---

## Resolution Tiers

### Tier 0: Direct Negotiation
```
Stakes: Any
Duration: Up to 7 days
Process: Parties communicate directly
Binding: No (parties can escalate)
Cost: None
```

### Tier 1: Mediation
```
Stakes: Any
Duration: Up to 14 days
Process: Neutral mediator facilitates discussion
Binding: Only if both parties agree
Cost: Minimal (mediator's time)
Mediator Selection: Random from available pool, conflict check
```

### Tier 2: Single Arbitrator
```
Stakes: Up to $5,000 equivalent
Duration: Up to 30 days
Process: Evidence submission, arbitrator reviews and decides
Binding: Yes (appealable to Tier 3)
Cost: Proportional to stakes (1-5%)
Arbitrator Selection: Weighted random by trust score, conflict check
```

### Tier 3: Panel (3 Arbitrators)
```
Stakes: $5,000 - $50,000 equivalent
Duration: Up to 45 days
Process: Detailed evidence phase, panel deliberation
Binding: Yes (appealable to Tier 4)
Cost: Higher (3x single arbitrator)
Panel Selection: One chosen by each party + one neutral
```

### Tier 4: Community Jury
```
Stakes: Over $50,000 or community-wide implications
Duration: Up to 60 days
Process: Full trial-like process, community jury of 7+
Binding: Yes (appealable to Tier 5 for precedent only)
Cost: Significant (shared by community)
Jury Selection: Random from high-trust agents, extensive conflict check
```

### Tier 5: Constitutional Review
```
Stakes: Precedent-setting, constitutional interpretation
Duration: Up to 90 days
Process: Review by Constitutional Council (elected)
Binding: Yes, creates binding precedent
Cost: Community-funded
Council: Elected representatives with term limits
```

---

## Zome Specifications

### 1. Dispute Manager Zome

```rust
// dispute_manager/src/lib.rs

/// File a new dispute
#[hdk_extern]
pub fn file_dispute(input: FileDisputeInput) -> ExternResult<ActionHash> {
    let claimant = agent_info()?.agent_latest_pubkey;

    // Validate source reference exists
    let source_valid = verify_source_reference(&input.source_happ, &input.source_reference)?;
    if !source_valid {
        return Err(WasmError::Guest("Invalid source reference".into()));
    }

    // Determine initial tier based on stake value
    let tier = determine_initial_tier(&input.stake_value);

    // Calculate response deadline
    let response_deadline = calculate_deadline(tier, DisputePhase::Response)?;

    // Create dispute
    let dispute = Dispute {
        dispute_id: generate_dispute_id(),
        source_happ: input.source_happ,
        source_reference: input.source_reference,
        claimant: claimant.clone(),
        respondent: input.respondent.clone(),
        dispute_type: input.dispute_type,
        claim: input.claim,
        remedy_sought: input.remedy_sought,
        stake_value: input.stake_value,
        status: DisputeStatus::Filed,
        tier,
        filed_at: sys_time()?,
        response_deadline,
        arbitrators: vec![],
    };

    let hash = create_entry(&EntryTypes::Dispute(dispute.clone()))?;

    // Create links for indexing
    create_dispute_links(&hash, &dispute)?;

    // Notify respondent
    send_signal_to_agent(&input.respondent, Signal::DisputeFiled(hash.clone()))?;

    // Notify source hApp
    bridge_call::<()>(
        &dispute.source_happ,
        "on_dispute_filed",
        DisputeNotification {
            dispute: hash.clone(),
            claimant,
            respondent: input.respondent,
            reference: dispute.source_reference,
        },
    )?;

    Ok(hash)
}

/// Respond to a dispute
#[hdk_extern]
pub fn respond_to_dispute(input: DisputeResponseInput) -> ExternResult<ActionHash> {
    let respondent = agent_info()?.agent_latest_pubkey;

    // Get dispute
    let dispute = get_dispute(&input.dispute)?;

    // Verify responder is the respondent
    if dispute.respondent != respondent {
        return Err(WasmError::Guest("Only respondent can respond".into()));
    }

    // Check deadline
    if sys_time()? > dispute.response_deadline {
        return Err(WasmError::Guest("Response deadline passed".into()));
    }

    // Create response
    let response = DisputeResponse {
        dispute: input.dispute,
        respondent,
        response_type: input.response_type,
        counter_claim: input.counter_claim,
        proposed_resolution: input.proposed_resolution,
        responded_at: sys_time()?,
    };

    let hash = create_entry(&EntryTypes::DisputeResponse(response))?;

    // Update dispute status
    update_dispute_status(&input.dispute, DisputeStatus::AwaitingResponse)?;

    // If direct resolution proposed and accepted, can resolve here
    if let ResponseType::ProposeSettlement(settlement) = input.response_type {
        // Notify claimant of settlement offer
        send_signal_to_agent(
            &dispute.claimant,
            Signal::SettlementProposed(input.dispute.clone(), settlement),
        )?;
    }

    // Notify claimant
    send_signal_to_agent(&dispute.claimant, Signal::DisputeResponsed(hash.clone()))?;

    Ok(hash)
}

/// Escalate dispute to next tier
#[hdk_extern]
pub fn escalate_dispute(dispute_hash: ActionHash) -> ExternResult<()> {
    let caller = agent_info()?.agent_latest_pubkey;
    let dispute = get_dispute(&dispute_hash)?;

    // Verify caller is a party
    if dispute.claimant != caller && dispute.respondent != caller {
        return Err(WasmError::Guest("Only parties can escalate".into()));
    }

    // Check escalation is appropriate
    let can_escalate = match dispute.status {
        DisputeStatus::MediationFailed => true,
        DisputeStatus::AppealPeriod => true,
        _ => false,
    };

    if !can_escalate {
        return Err(WasmError::Guest("Cannot escalate at current status".into()));
    }

    // Determine next tier
    let next_tier = escalate_tier(&dispute.tier)?;

    // Update dispute
    update_dispute_tier(&dispute_hash, next_tier)?;

    // If moving to arbitration, assign arbitrators
    if matches!(next_tier, ResolutionTier::Tier2SingleArbitrator | ResolutionTier::Tier3Panel) {
        let arbitrators = select_arbitrators(&dispute, next_tier)?;
        assign_arbitrators(&dispute_hash, arbitrators)?;
    }

    Ok(())
}

/// Settle dispute directly (mutual agreement)
#[hdk_extern]
pub fn settle_dispute(input: SettlementInput) -> ExternResult<ActionHash> {
    let caller = agent_info()?.agent_latest_pubkey;
    let dispute = get_dispute(&input.dispute)?;

    // Verify caller is a party
    if dispute.claimant != caller && dispute.respondent != caller {
        return Err(WasmError::Guest("Only parties can settle".into()));
    }

    // Check both parties have signed off
    if !verify_settlement_signatures(&input)? {
        return Err(WasmError::Guest("Both parties must agree".into()));
    }

    // Create settlement record
    let settlement = Settlement {
        dispute: input.dispute,
        terms: input.terms,
        agreed_by: vec![dispute.claimant.clone(), dispute.respondent.clone()],
        agreed_at: sys_time()?,
    };

    let hash = create_entry(&EntryTypes::Settlement(settlement))?;

    // Update dispute status
    update_dispute_status(&input.dispute, DisputeStatus::Settled)?;

    // Execute settlement terms
    execute_settlement(&input.terms)?;

    Ok(hash)
}
```

### 2. Evidence Manager Zome

```rust
// evidence_manager/src/lib.rs

/// Submit evidence for a dispute
#[hdk_extern]
pub fn submit_evidence(input: SubmitEvidenceInput) -> ExternResult<ActionHash> {
    let submitter = agent_info()?.agent_latest_pubkey;

    // Get dispute
    let dispute = get_dispute(&input.dispute)?;

    // Verify submitter is a party or witness
    let is_party = dispute.claimant == submitter || dispute.respondent == submitter;
    let is_witness = input.as_witness;

    if !is_party && !is_witness {
        return Err(WasmError::Guest("Only parties or witnesses can submit evidence".into()));
    }

    // Check evidence phase is open
    if !is_evidence_phase_open(&dispute)? {
        return Err(WasmError::Guest("Evidence phase is closed".into()));
    }

    // Classify evidence epistemically
    let epistemic = classify_evidence(&input.evidence_type, &input.content)?;

    // If cross-hApp reference, verify it
    let cross_happ_ref = if let Some(ref xref) = input.cross_happ_reference {
        let verified = verify_cross_happ_evidence(xref)?;
        Some(CrossHappReference {
            happ: xref.happ.clone(),
            entry_type: xref.entry_type.clone(),
            action_hash: xref.action_hash.clone(),
            verified,
            verification_method: "bridge_call".to_string(),
        })
    } else {
        None
    };

    // Create evidence entry
    let evidence = Evidence {
        dispute: input.dispute,
        submitter,
        evidence_type: input.evidence_type,
        content: input.content,
        epistemic,
        submitted_at: sys_time()?,
        witnesses: input.witnesses,
        cross_happ_ref,
    };

    let hash = create_entry(&EntryTypes::Evidence(evidence))?;

    // Link to dispute
    create_link(
        input.dispute,
        hash.clone(),
        LinkTypes::DisputeToEvidence,
        (),
    )?;

    // Notify other party
    let other_party = if dispute.claimant == submitter {
        dispute.respondent
    } else {
        dispute.claimant
    };
    send_signal_to_agent(&other_party, Signal::EvidenceSubmitted(hash.clone()))?;

    Ok(hash)
}

/// Challenge evidence
#[hdk_extern]
pub fn challenge_evidence(input: EvidenceChallengeInput) -> ExternResult<ActionHash> {
    let challenger = agent_info()?.agent_latest_pubkey;

    // Get evidence and dispute
    let evidence = get_evidence(&input.evidence)?;
    let dispute = get_dispute(&evidence.dispute)?;

    // Verify challenger is a party
    if dispute.claimant != challenger && dispute.respondent != challenger {
        return Err(WasmError::Guest("Only parties can challenge evidence".into()));
    }

    // Can't challenge own evidence
    if evidence.submitter == challenger {
        return Err(WasmError::Guest("Cannot challenge your own evidence".into()));
    }

    // Create challenge
    let challenge = EvidenceChallenge {
        evidence: input.evidence,
        challenger,
        grounds: input.grounds,
        argument: input.argument,
        counter_evidence: input.counter_evidence,
        challenged_at: sys_time()?,
    };

    let hash = create_entry(&EntryTypes::EvidenceChallenge(challenge))?;

    // Notify submitter
    send_signal_to_agent(&evidence.submitter, Signal::EvidenceChallenged(hash.clone()))?;

    Ok(hash)
}

/// Classify evidence epistemically
fn classify_evidence(
    evidence_type: &EvidenceType,
    content: &EvidenceContent,
) -> ExternResult<EpistemicClaim> {
    let empirical = match evidence_type {
        EvidenceType::Testimony => EmpiricalLevel::E1Testimonial,
        EvidenceType::TransactionRecord => EmpiricalLevel::E3Cryptographic,
        EvidenceType::ContractRecord => EmpiricalLevel::E3Cryptographic,
        EvidenceType::CommunicationLog => EmpiricalLevel::E2PrivateVerify,
        EvidenceType::ExternalDocument => EmpiricalLevel::E1Testimonial,
        EvidenceType::MediaEvidence => EmpiricalLevel::E2PrivateVerify,
        EvidenceType::ExpertOpinion => EmpiricalLevel::E2PrivateVerify,
        EvidenceType::CrossHappData => EmpiricalLevel::E3Cryptographic,
    };

    Ok(EpistemicClaim::new(
        "Dispute evidence",
        empirical,
        NormativeLevel::N1Communal, // Dispute-specific agreement
        MaterialityLevel::M2Persistent, // Keep for record
    ))
}
```

### 3. Arbitrator Selection Zome

```rust
// arbitrator_selection/src/lib.rs

/// Register as an arbitrator
#[hdk_extern]
pub fn register_as_arbitrator(input: ArbitratorRegistration) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_latest_pubkey;

    // Verify credentials via Praxis
    let credentials_valid = verify_arbitrator_credentials(&input.credentials)?;
    if !credentials_valid {
        return Err(WasmError::Guest("Invalid credentials".into()));
    }

    // Get MATL trust score
    let trust_score = bridge_call::<f64>(
        "bridge",
        "get_cross_happ_reputation",
        agent.clone(),
    )?;

    // Minimum trust requirement
    if trust_score < 0.7 {
        return Err(WasmError::Guest("Trust score too low (min 0.7)".into()));
    }

    // Create profile
    let profile = ArbitratorProfile {
        agent,
        specializations: input.specializations,
        credentials: input.credentials,
        languages: input.languages,
        trust_score,
        cases_completed: 0,
        satisfaction_rating: 0.0,
        appeal_reversal_rate: 0.0,
        available: true,
        stake_deposited: input.stake,
    };

    let hash = create_entry(&EntryTypes::ArbitratorProfile(profile))?;

    // Index by specialization
    for spec in &input.specializations {
        create_link(
            specialization_path(spec).path_entry_hash()?,
            hash.clone(),
            LinkTypes::SpecializationToArbitrator,
            (),
        )?;
    }

    Ok(hash)
}

/// Select arbitrators for a dispute
pub fn select_arbitrators(
    dispute: &Dispute,
    tier: ResolutionTier,
) -> ExternResult<Vec<AgentPubKey>> {
    let count = match tier {
        ResolutionTier::Tier1Mediation => 1,
        ResolutionTier::Tier2SingleArbitrator => 1,
        ResolutionTier::Tier3Panel => 3,
        ResolutionTier::Tier4CommunityJury => 7,
        _ => return Err(WasmError::Guest("Invalid tier for arbitrator selection".into())),
    };

    // Get available arbitrators for this dispute type
    let candidates = get_available_arbitrators(&dispute.dispute_type)?;

    // Filter out conflicts of interest
    let eligible: Vec<_> = candidates
        .into_iter()
        .filter(|a| !has_conflict(&a.agent, dispute))
        .collect();

    if eligible.len() < count {
        return Err(WasmError::Guest("Not enough eligible arbitrators".into()));
    }

    // Weight by trust score and experience
    let selected = weighted_random_selection(&eligible, count)?;

    Ok(selected.into_iter().map(|a| a.agent).collect())
}

/// Check for conflicts of interest
fn has_conflict(arbitrator: &AgentPubKey, dispute: &Dispute) -> bool {
    // Check: previous interactions with parties
    // Check: same organization/community
    // Check: previous disputes involving same parties
    // Check: financial relationships
    // Check: family/personal relationships (self-declared)

    // Query relationship data via Bridge
    let claimant_relation = bridge_call::<f64>(
        "weave", // Relationship mapping hApp
        "get_relationship_strength",
        (arbitrator, &dispute.claimant),
    ).unwrap_or(0.0);

    let respondent_relation = bridge_call::<f64>(
        "weave",
        "get_relationship_strength",
        (arbitrator, &dispute.respondent),
    ).unwrap_or(0.0);

    // Conflict if strong relationship with either party
    claimant_relation > 0.5 || respondent_relation > 0.5
}

/// Weighted random selection based on trust and experience
fn weighted_random_selection(
    candidates: &[ArbitratorProfile],
    count: usize,
) -> ExternResult<Vec<ArbitratorProfile>> {
    let mut selected = vec![];
    let mut remaining = candidates.to_vec();

    for _ in 0..count {
        if remaining.is_empty() {
            break;
        }

        // Calculate weights
        let weights: Vec<f64> = remaining
            .iter()
            .map(|a| {
                a.trust_score * 0.4
                    + (a.cases_completed as f64 / 100.0).min(1.0) * 0.2
                    + a.satisfaction_rating * 0.3
                    + (1.0 - a.appeal_reversal_rate) * 0.1
            })
            .collect();

        // Weighted random selection
        let total: f64 = weights.iter().sum();
        let mut random = random_in_range(0.0, total)?;

        let mut selected_idx = 0;
        for (idx, weight) in weights.iter().enumerate() {
            random -= weight;
            if random <= 0.0 {
                selected_idx = idx;
                break;
            }
        }

        selected.push(remaining.remove(selected_idx));
    }

    Ok(selected)
}
```

### 4. Decision Engine Zome

```rust
// decision_engine/src/lib.rs

/// Issue a ruling on a dispute
#[hdk_extern]
pub fn issue_ruling(input: RulingInput) -> ExternResult<ActionHash> {
    let arbitrator = agent_info()?.agent_latest_pubkey;

    // Get dispute
    let dispute = get_dispute(&input.dispute)?;

    // Verify caller is assigned arbitrator
    if !dispute.arbitrators.contains(&arbitrator) {
        return Err(WasmError::Guest("Not an assigned arbitrator".into()));
    }

    // For panels, check if this is the deciding vote
    if dispute.arbitrators.len() > 1 {
        let existing_votes = get_arbitrator_votes(&input.dispute)?;
        let total_arbitrators = dispute.arbitrators.len();
        let current_decision = input.decision.clone();

        // Count votes for this decision
        let votes_for: usize = existing_votes
            .iter()
            .filter(|v| v.decision == current_decision)
            .count() + 1; // +1 for this vote

        let majority = total_arbitrators / 2 + 1;

        if votes_for < majority {
            // Record vote but don't issue ruling yet
            record_vote(&input.dispute, &arbitrator, &input)?;
            return Ok(input.dispute); // Return dispute hash, not ruling
        }
    }

    // Analyze evidence
    let evidence_analysis = analyze_evidence(&input.dispute, &input.evidence_considered)?;

    // Calculate reputation impacts
    let reputation_impacts = calculate_reputation_impacts(&dispute, &input.decision)?;

    // Calculate appeal deadline
    let appeal_deadline = sys_time()?.add_days(match dispute.tier {
        ResolutionTier::Tier2SingleArbitrator => 7,
        ResolutionTier::Tier3Panel => 14,
        ResolutionTier::Tier4CommunityJury => 21,
        _ => 7,
    });

    // Create ruling
    let ruling = Ruling {
        dispute: input.dispute,
        arbitrators: vec![arbitrator], // Will be updated for panels
        decision: input.decision,
        reasoning: input.reasoning,
        evidence_considered: input.evidence_considered,
        evidence_analysis,
        remedies_ordered: input.remedies,
        reputation_impacts: reputation_impacts.clone(),
        issued_at: sys_time()?,
        appeal_deadline,
        is_final: false,
        precedent_category: input.precedent_category,
        dissenting_opinions: vec![],
    };

    let hash = create_entry(&EntryTypes::Ruling(ruling))?;

    // Update dispute status
    update_dispute_status(&input.dispute, DisputeStatus::RulingIssued)?;

    // Apply reputation impacts via Bridge
    for impact in reputation_impacts {
        bridge_call::<()>(
            "bridge",
            "apply_reputation_impact",
            impact,
        )?;
    }

    // Notify parties
    send_signal_to_agent(&dispute.claimant, Signal::RulingIssued(hash.clone()))?;
    send_signal_to_agent(&dispute.respondent, Signal::RulingIssued(hash.clone()))?;

    // Notify source hApp
    bridge_call::<()>(
        &dispute.source_happ,
        "on_ruling_issued",
        RulingNotification {
            ruling: hash.clone(),
            dispute: input.dispute,
            decision: input.decision,
        },
    )?;

    Ok(hash)
}

/// Analyze evidence and assign weights
fn analyze_evidence(
    dispute: &ActionHash,
    considered: &[ActionHash],
) -> ExternResult<Vec<EvidenceAnalysis>> {
    let mut analysis = vec![];

    for evidence_hash in considered {
        let evidence = get_evidence(evidence_hash)?;

        // Base weight from epistemic level
        let base_weight = match evidence.epistemic.empirical {
            EmpiricalLevel::E0Null => 0.1,
            EmpiricalLevel::E1Testimonial => 0.3,
            EmpiricalLevel::E2PrivateVerify => 0.6,
            EmpiricalLevel::E3Cryptographic => 0.9,
            EmpiricalLevel::E4PublicRepro => 1.0,
        };

        // Adjust for corroboration
        let corroboration_bonus = count_corroborating_evidence(&evidence, dispute)? * 0.1;

        // Adjust for challenges
        let challenge_penalty = get_successful_challenges(&evidence_hash)? * 0.2;

        let final_weight = (base_weight + corroboration_bonus - challenge_penalty).clamp(0.0, 1.0);

        analysis.push(EvidenceAnalysis {
            evidence: evidence_hash.clone(),
            weight_assigned: final_weight,
            epistemic_upgrade: None,
            analysis: format!(
                "Base: {:.2}, Corroboration: +{:.2}, Challenges: -{:.2}",
                base_weight, corroboration_bonus, challenge_penalty
            ),
        });
    }

    Ok(analysis)
}

/// Calculate reputation impacts based on decision
fn calculate_reputation_impacts(
    dispute: &Dispute,
    decision: &Decision,
) -> ExternResult<Vec<ReputationImpact>> {
    let mut impacts = vec![];

    match decision {
        Decision::ClaimantPrevails => {
            // Claimant vindicated
            impacts.push(ReputationImpact {
                agent: dispute.claimant.clone(),
                happ: dispute.source_happ.clone(),
                impact: 0.02, // Small positive
                reason: "Prevailed in dispute".to_string(),
            });
            // Respondent penalized
            impacts.push(ReputationImpact {
                agent: dispute.respondent.clone(),
                happ: dispute.source_happ.clone(),
                impact: -0.05, // Larger negative
                reason: "Lost dispute".to_string(),
            });
        }
        Decision::RespondentPrevails => {
            // Opposite
            impacts.push(ReputationImpact {
                agent: dispute.respondent.clone(),
                happ: dispute.source_happ.clone(),
                impact: 0.02,
                reason: "Prevailed in dispute".to_string(),
            });
            // Claimant may be penalized for frivolous claim
            impacts.push(ReputationImpact {
                agent: dispute.claimant.clone(),
                happ: dispute.source_happ.clone(),
                impact: -0.03,
                reason: "Claim rejected".to_string(),
            });
        }
        Decision::SplitDecision { claimant_portion } => {
            // Proportional impacts
            impacts.push(ReputationImpact {
                agent: dispute.claimant.clone(),
                happ: dispute.source_happ.clone(),
                impact: (claimant_portion - 0.5) * 0.04,
                reason: format!("Split decision: {:.0}%", claimant_portion * 100.0),
            });
            impacts.push(ReputationImpact {
                agent: dispute.respondent.clone(),
                happ: dispute.source_happ.clone(),
                impact: (0.5 - claimant_portion) * 0.04,
                reason: format!("Split decision: {:.0}%", (1.0 - claimant_portion) * 100.0),
            });
        }
        Decision::Dismissed(_) => {
            // Claimant penalized for dismissed claim
            impacts.push(ReputationImpact {
                agent: dispute.claimant.clone(),
                happ: dispute.source_happ.clone(),
                impact: -0.02,
                reason: "Claim dismissed".to_string(),
            });
        }
        Decision::Remanded(_) => {
            // No reputation impact for remand
        }
    }

    Ok(impacts)
}
```

### 5. Appeals Manager Zome

```rust
// appeals_manager/src/lib.rs

/// File an appeal
#[hdk_extern]
pub fn file_appeal(input: FileAppealInput) -> ExternResult<ActionHash> {
    let appellant = agent_info()?.agent_latest_pubkey;

    // Get ruling
    let ruling = get_ruling(&input.ruling)?;
    let dispute = get_dispute(&ruling.dispute)?;

    // Verify appellant is a party
    if dispute.claimant != appellant && dispute.respondent != appellant {
        return Err(WasmError::Guest("Only parties can appeal".into()));
    }

    // Check within appeal deadline
    if sys_time()? > ruling.appeal_deadline {
        return Err(WasmError::Guest("Appeal deadline passed".into()));
    }

    // Check appeal is allowed for this tier
    let current_tier = dispute.tier;
    let appeal_allowed = matches!(
        current_tier,
        ResolutionTier::Tier2SingleArbitrator |
        ResolutionTier::Tier3Panel |
        ResolutionTier::Tier4CommunityJury
    );

    if !appeal_allowed {
        return Err(WasmError::Guest("Appeals not allowed at this tier".into()));
    }

    // Calculate appeal fee
    let appeal_fee = calculate_appeal_fee(&dispute.stake_value, &current_tier)?;

    // Verify fee deposited
    if input.fee_deposited < appeal_fee {
        return Err(WasmError::Guest("Insufficient appeal fee".into()));
    }

    // Verify grounds are valid
    validate_appeal_grounds(&input.grounds, &ruling)?;

    // Create appeal
    let appeal = Appeal {
        ruling: input.ruling,
        appellant,
        grounds: input.grounds,
        argument: input.argument,
        new_evidence: input.new_evidence,
        fee_deposited: input.fee_deposited,
        filed_at: sys_time()?,
        status: AppealStatus::Filed,
    };

    let hash = create_entry(&EntryTypes::Appeal(appeal))?;

    // Update dispute status
    update_dispute_status(&ruling.dispute, DisputeStatus::Appealed)?;

    // Escalate to next tier
    let next_tier = escalate_tier(&current_tier)?;
    update_dispute_tier(&ruling.dispute, next_tier)?;

    // Select new arbitrators for appeal
    let new_arbitrators = select_arbitrators(&dispute, next_tier)?;
    assign_arbitrators(&ruling.dispute, new_arbitrators)?;

    // Notify other party
    let other_party = if dispute.claimant == appellant {
        dispute.respondent
    } else {
        dispute.claimant
    };
    send_signal_to_agent(&other_party, Signal::AppealFiled(hash.clone()))?;

    Ok(hash)
}

/// Process appeal decision
#[hdk_extern]
pub fn decide_appeal(input: AppealDecisionInput) -> ExternResult<ActionHash> {
    let arbitrator = agent_info()?.agent_latest_pubkey;

    // Get appeal and related data
    let appeal = get_appeal(&input.appeal)?;
    let original_ruling = get_ruling(&appeal.ruling)?;
    let dispute = get_dispute(&original_ruling.dispute)?;

    // Verify caller is assigned arbitrator
    if !dispute.arbitrators.contains(&arbitrator) {
        return Err(WasmError::Guest("Not an assigned arbitrator".into()));
    }

    // Process decision
    let appeal_status = match &input.decision {
        AppealDecision::Deny => AppealStatus::Denied,
        AppealDecision::Uphold => AppealStatus::Upheld,
        AppealDecision::Reverse => AppealStatus::Reversed,
        AppealDecision::Modify(_) => AppealStatus::Modified,
    };

    // Update appeal status
    update_appeal_status(&input.appeal, appeal_status)?;

    // If reversed or modified, create new ruling
    let new_ruling = if matches!(input.decision, AppealDecision::Reverse | AppealDecision::Modify(_)) {
        let new_ruling = match input.decision {
            AppealDecision::Reverse => {
                // Reverse the original decision
                let reversed_decision = reverse_decision(&original_ruling.decision);
                issue_appeal_ruling(&dispute, &original_ruling, reversed_decision, input.reasoning)?
            }
            AppealDecision::Modify(modifications) => {
                // Modify specific aspects
                issue_modified_ruling(&dispute, &original_ruling, modifications, input.reasoning)?
            }
            _ => unreachable!(),
        };
        Some(new_ruling)
    } else {
        None
    };

    // Handle appeal fee
    match appeal_status {
        AppealStatus::Upheld | AppealStatus::Reversed | AppealStatus::Modified => {
            // Return fee to appellant if appeal had merit
            if matches!(appeal_status, AppealStatus::Reversed | AppealStatus::Modified) {
                return_appeal_fee(&appeal.appellant, &appeal.fee_deposited)?;
            }
        }
        AppealStatus::Denied => {
            // Fee forfeited (goes to arbitrator pool or community)
            forfeit_appeal_fee(&appeal.fee_deposited)?;
        }
        _ => {}
    }

    // Update arbitrator's appeal reversal rate
    update_arbitrator_stats(&original_ruling.arbitrators, appeal_status)?;

    // Return hash of new ruling or appeal
    Ok(new_ruling.unwrap_or(input.appeal))
}
```

---

## Cross-hApp Integration

### Marketplace Integration
```rust
// When dispute filed from Marketplace
bridge_call::<()>(
    "marketplace",
    "pause_transaction",
    dispute.source_reference,
)?;

// On ruling
bridge_call::<()>(
    "marketplace",
    "execute_remedy",
    OrderedRemedy { ... },
)?;
```

### SupplyChain Integration
```rust
// Verify provenance claims
let provenance = bridge_call::<ProvenanceChain>(
    "supplychain",
    "get_provenance",
    product_id,
)?;
```

### MATL/Bridge Integration
```rust
// Apply reputation impacts
bridge_call::<()>(
    "bridge",
    "apply_reputation_impact",
    ReputationImpact { ... },
)?;
```

---

## Precedent System

### Creating Precedent
```rust
// When a ruling has broader implications
if ruling.precedent_category.is_some() {
    // Store as precedent
    let precedent = Precedent {
        ruling: ruling_hash,
        category: ruling.precedent_category.unwrap(),
        summary: generate_precedent_summary(&ruling),
        key_principles: extract_key_principles(&ruling.reasoning),
        created_at: sys_time()?,
        superseded_by: None,
    };

    create_entry(&EntryTypes::Precedent(precedent))?;
}
```

### Consulting Precedent
```rust
// During deliberation
let relevant_precedents = get_precedents_for_category(&dispute.dispute_type)?;

// Arbitrator considers precedent in reasoning
// Not strictly binding, but deviation requires explanation
```

---

## Economic Model

### Fee Structure

| Tier | Filing Fee | Decision Fee | Appeal Fee |
|------|------------|--------------|------------|
| T1 Mediation | 0 | 0 | N/A |
| T2 Single Arb | 1% of stake (min $10) | 2% of stake | 3x filing |
| T3 Panel | 2% of stake (min $50) | 4% of stake | 3x filing |
| T4 Jury | 3% of stake (min $200) | 6% of stake | 4x filing |

### Fee Distribution
- 60% to arbitrators
- 20% to Arbiter development fund
- 10% to unsuccessful party (if clear frivolous claim)
- 10% to community treasury

---

## Success Metrics

| Metric | 1 Year | 3 Years |
|--------|--------|---------|
| Disputes resolved | 500 | 10,000 |
| Median resolution time | 14 days | 10 days |
| Settlement rate (pre-ruling) | 40% | 60% |
| Appeal reversal rate | <20% | <15% |
| Satisfaction rating | 4.0/5 | 4.5/5 |
| Active arbitrators | 50 | 500 |

---

*"Justice delayed is justice denied. Justice decentralized is justice for all."*
