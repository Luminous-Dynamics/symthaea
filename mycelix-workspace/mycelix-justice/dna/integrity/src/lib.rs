// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Mycelix Justice Integrity Zome
//!
//! Entry types and validation for decentralized dispute resolution.
//!
//! Implements:
//! - Three-tier justice (mediation, arbitration, appeal)
//! - Tamper-proof evidence management
//! - Restorative justice pathways
//! - Cross-hApp enforcement
//! - Community-based adjudication

use hdi::prelude::*;

// ============================================================================
// CASE TYPES
// ============================================================================

/// A dispute case
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Case {
    /// Case ID
    pub id: String,
    /// Case title
    pub title: String,
    /// Description of the dispute
    pub description: String,
    /// Case type
    pub case_type: CaseType,
    /// Complainant DID
    pub complainant: String,
    /// Respondent DID
    pub respondent: String,
    /// Additional parties
    pub parties: Vec<CaseParty>,
    /// Current phase
    pub phase: CasePhase,
    /// Status
    pub status: CaseStatus,
    /// Severity assessment
    pub severity: CaseSeverity,
    /// Related hApp/context
    pub context: CaseContext,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Last updated
    pub updated_at: Timestamp,
    /// Deadline for current phase
    pub phase_deadline: Option<Timestamp>,
}

/// Types of cases
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CaseType {
    /// Breach of agreement/contract
    ContractDispute,
    /// Violation of community guidelines
    ConductViolation,
    /// Property/asset dispute
    PropertyDispute,
    /// Financial dispute
    FinancialDispute,
    /// Governance dispute
    GovernanceDispute,
    /// Identity/reputation dispute
    IdentityDispute,
    /// Intellectual property
    IPDispute,
    /// Other
    Other { category: String },
}

/// Party to a case
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseParty {
    pub did: String,
    pub role: PartyRole,
    pub joined_at: Timestamp,
}

/// Roles parties can have
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PartyRole {
    Complainant,
    Respondent,
    Witness,
    Expert,
    Intervenor,
    Affected,
}

/// Case phases (three-tier system)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CasePhase {
    /// Initial filing
    Filed,
    /// Direct negotiation between parties
    Negotiation,
    /// Third-party mediation
    Mediation,
    /// Formal arbitration
    Arbitration,
    /// Appeal of arbitration decision
    Appeal,
    /// Enforcement of decision
    Enforcement,
    /// Case closed
    Closed,
}

/// Case status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CaseStatus {
    Active,
    OnHold,
    AwaitingResponse,
    InDeliberation,
    DecisionRendered,
    Enforcing,
    Resolved,
    Dismissed,
    Withdrawn,
}

/// Case severity
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CaseSeverity {
    /// Minor, can be resolved informally
    Minor,
    /// Moderate, requires structured process
    Moderate,
    /// Serious, requires formal adjudication
    Serious,
    /// Critical, may require external intervention
    Critical,
}

/// Context of the dispute
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CaseContext {
    /// Originating hApp
    pub happ: Option<String>,
    /// Specific entry/transaction
    pub reference_id: Option<String>,
    /// Community
    pub community: Option<String>,
    /// Jurisdiction rules
    pub jurisdiction: Option<String>,
}

// ============================================================================
// EVIDENCE TYPES
// ============================================================================

/// Evidence submitted for a case
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Evidence {
    /// Evidence ID
    pub id: String,
    /// Case ID
    pub case_id: String,
    /// Submitter DID
    pub submitter: String,
    /// Evidence type
    pub evidence_type: EvidenceType,
    /// Content reference
    pub content: EvidenceContent,
    /// Description
    pub description: String,
    /// Chain of custody
    pub custody: Vec<CustodyEvent>,
    /// Verification status
    pub verification: EvidenceVerification,
    /// Visibility
    pub visibility: EvidenceVisibility,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Sealed (no more changes)
    pub sealed: bool,
}

/// Types of evidence
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EvidenceType {
    /// Document/text
    Document,
    /// Transaction record
    Transaction,
    /// Communication log
    Communication,
    /// Testimony
    Testimony,
    /// Expert opinion
    ExpertOpinion,
    /// Media (image, video, audio)
    Media,
    /// On-chain data
    OnChainData { happ: String, entry_hash: String },
    /// External evidence
    External { source: String },
}

/// Evidence content
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceContent {
    /// Content hash (for integrity)
    pub hash: String,
    /// Storage reference (CID, entry hash, etc.)
    pub reference: String,
    /// MIME type
    pub mime_type: String,
    /// Size in bytes
    pub size: u64,
    /// Encrypted?
    pub encrypted: bool,
    /// Encryption key reference (if encrypted)
    pub key_reference: Option<String>,
}

/// Chain of custody event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CustodyEvent {
    pub action: CustodyAction,
    pub actor: String,
    pub timestamp: Timestamp,
    pub notes: Option<String>,
}

/// Custody actions
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CustodyAction {
    Submitted,
    Accessed,
    Copied,
    Verified,
    Challenged,
    Sealed,
    Released,
}

/// Evidence verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceVerification {
    pub status: VerificationStatus,
    pub verifier: Option<String>,
    pub method: Option<String>,
    pub verified_at: Option<Timestamp>,
    pub notes: Option<String>,
}

/// Verification status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerificationStatus {
    Unverified,
    Pending,
    Verified,
    Disputed,
    Rejected,
}

/// Evidence visibility
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EvidenceVisibility {
    /// All parties can see
    AllParties,
    /// Only adjudicators can see
    AdjudicatorsOnly,
    /// Specific parties
    Restricted { parties: Vec<String> },
    /// Sealed (court order)
    Sealed,
}

// ============================================================================
// MEDIATION TYPES
// ============================================================================

/// Mediation session
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Mediation {
    /// Mediation ID
    pub id: String,
    /// Case ID
    pub case_id: String,
    /// Mediator DID
    pub mediator: String,
    /// Status
    pub status: MediationStatus,
    /// Scheduled sessions
    pub sessions: Vec<MediationSession>,
    /// Proposed settlements
    pub proposals: Vec<String>, // Settlement IDs
    /// Created timestamp
    pub created_at: Timestamp,
    /// Deadline
    pub deadline: Option<Timestamp>,
}

/// Mediation status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MediationStatus {
    Scheduled,
    InProgress,
    SettlementReached,
    Failed,
    Cancelled,
}

/// Mediation session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediationSession {
    pub session_number: u32,
    pub scheduled_at: Timestamp,
    pub actual_start: Option<Timestamp>,
    pub actual_end: Option<Timestamp>,
    pub notes: Option<String>,
    pub outcome: Option<String>,
}

// ============================================================================
// ARBITRATION TYPES
// ============================================================================

/// Arbitration panel
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Arbitration {
    /// Arbitration ID
    pub id: String,
    /// Case ID
    pub case_id: String,
    /// Arbitrators (odd number for voting)
    pub arbitrators: Vec<Arbitrator>,
    /// Selection method used
    pub selection_method: ArbitratorSelection,
    /// Status
    pub status: ArbitrationStatus,
    /// Deliberation deadline
    pub deliberation_deadline: Option<Timestamp>,
    /// Created timestamp
    pub created_at: Timestamp,
}

/// Arbitrator info
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Arbitrator {
    pub did: String,
    pub role: ArbitratorRole,
    pub selected_at: Timestamp,
    pub accepted: bool,
    pub recused: bool,
    pub recusal_reason: Option<String>,
}

/// Arbitrator roles
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArbitratorRole {
    /// Primary arbitrator
    Primary,
    /// Panel member
    PanelMember,
    /// Alternate (in case of recusal)
    Alternate,
}

/// How arbitrators were selected
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArbitratorSelection {
    /// Random from qualified pool
    Random,
    /// Weighted by reputation/MATL
    MATLWeighted,
    /// Party agreement
    PartyAgreed,
    /// Expertise-based
    ExpertiseBased { domain: String },
}

/// Arbitration status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArbitrationStatus {
    PanelFormation,
    EvidenceReview,
    Hearing,
    Deliberation,
    DecisionDrafting,
    DecisionRendered,
    Appealed,
}

/// Arbitration decision
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Decision {
    /// Decision ID
    pub id: String,
    /// Case ID
    pub case_id: String,
    /// Arbitration ID
    pub arbitration_id: String,
    /// Decision type
    pub decision_type: DecisionType,
    /// Outcome
    pub outcome: DecisionOutcome,
    /// Reasoning
    pub reasoning: String,
    /// Remedies ordered
    pub remedies: Vec<Remedy>,
    /// Voting record
    pub votes: Vec<ArbitratorVote>,
    /// Dissenting opinions
    pub dissents: Vec<DissentingOpinion>,
    /// Rendered timestamp
    pub rendered_at: Timestamp,
    /// Appeal deadline
    pub appeal_deadline: Timestamp,
    /// Finalized (no more appeals)
    pub finalized: bool,
}

/// Decision types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecisionType {
    /// Full decision on merits
    MeritsDecision,
    /// Preliminary/interim decision
    InterimDecision,
    /// Default (party didn't respond)
    DefaultDecision,
    /// Consent (parties agreed)
    ConsentDecision,
    /// Dismissal
    Dismissal,
}

/// Decision outcomes
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DecisionOutcome {
    /// Complainant prevails
    ForComplainant,
    /// Respondent prevails
    ForRespondent,
    /// Split decision
    SplitDecision,
    /// Case dismissed
    Dismissed,
    /// Settled before decision
    Settled,
}

/// Remedies that can be ordered
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Remedy {
    pub remedy_type: RemedyType,
    pub responsible_party: String,
    pub deadline: Option<Timestamp>,
    pub amount: Option<u128>,
    pub currency: Option<String>,
    pub description: String,
}

/// Types of remedies
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RemedyType {
    /// Monetary compensation
    Compensation,
    /// Return of property
    Restitution,
    /// Specific performance
    SpecificPerformance,
    /// Cease and desist
    Injunction,
    /// Formal apology
    Apology,
    /// Community service
    CommunityService,
    /// Reputation adjustment
    ReputationAdjustment,
    /// Access restriction
    AccessRestriction,
    /// Training/education
    Education,
    /// Restorative circle
    RestorativeCircle,
}

/// Arbitrator's vote
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArbitratorVote {
    pub arbitrator: String,
    pub vote: VoteChoice,
    pub timestamp: Timestamp,
}

/// Vote choices
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum VoteChoice {
    ForComplainant,
    ForRespondent,
    Abstain,
}

/// Dissenting opinion
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DissentingOpinion {
    pub arbitrator: String,
    pub opinion: String,
    pub timestamp: Timestamp,
}

// ============================================================================
// APPEAL TYPES
// ============================================================================

/// Appeal of a decision
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Appeal {
    /// Appeal ID
    pub id: String,
    /// Case ID
    pub case_id: String,
    /// Decision being appealed
    pub decision_id: String,
    /// Appellant DID
    pub appellant: String,
    /// Grounds for appeal
    pub grounds: Vec<AppealGround>,
    /// Argument
    pub argument: String,
    /// Status
    pub status: AppealStatus,
    /// Appeal number (1st, 2nd, etc.)
    pub appeal_number: u8,
    /// Created timestamp
    pub created_at: Timestamp,
}

/// Grounds for appeal
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AppealGround {
    /// Procedural error
    ProceduralError,
    /// New evidence
    NewEvidence,
    /// Legal/rule misinterpretation
    LegalError,
    /// Bias/conflict of interest
    Bias,
    /// Excessive remedy
    ExcessiveRemedy,
    /// Insufficient remedy
    InsufficientRemedy,
}

/// Appeal status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AppealStatus {
    Filed,
    UnderReview,
    Granted,
    Denied,
    Remanded,
    Resolved,
}

// ============================================================================
// ENFORCEMENT TYPES
// ============================================================================

/// Enforcement action
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Enforcement {
    /// Enforcement ID
    pub id: String,
    /// Decision ID
    pub decision_id: String,
    /// Remedy being enforced
    pub remedy_index: u32,
    /// Enforcer DID (may be system)
    pub enforcer: String,
    /// Status
    pub status: EnforcementStatus,
    /// Actions taken
    pub actions: Vec<EnforcementAction>,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Completed timestamp
    pub completed_at: Option<Timestamp>,
}

/// Enforcement status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnforcementStatus {
    Pending,
    InProgress,
    PartiallyCompleted,
    Completed,
    Failed,
    Contested,
}

/// Enforcement action taken
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnforcementAction {
    pub action_type: EnforcementActionType,
    pub target_happ: Option<String>,
    pub target_entry: Option<String>,
    pub executed_at: Timestamp,
    pub result: String,
}

/// Types of enforcement actions
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnforcementActionType {
    /// Transfer funds
    FundsTransfer,
    /// Lock/freeze assets
    AssetFreeze,
    /// Reputation adjustment
    ReputationUpdate,
    /// Access revocation
    AccessRevocation,
    /// Notification sent
    Notification,
    /// Manual action required
    ManualRequired,
    /// Cross-hApp action
    CrossHappAction,
}

// ============================================================================
// RESTORATIVE JUSTICE TYPES
// ============================================================================

/// Restorative justice circle
#[hdk_entry_helper]
#[derive(Clone)]
pub struct RestorativeCircle {
    /// Circle ID
    pub id: String,
    /// Case ID
    pub case_id: String,
    /// Facilitator DID
    pub facilitator: String,
    /// Participants
    pub participants: Vec<CircleParticipant>,
    /// Status
    pub status: CircleStatus,
    /// Sessions held
    pub sessions: Vec<CircleSession>,
    /// Agreements reached
    pub agreements: Vec<String>,
    /// Created timestamp
    pub created_at: Timestamp,
}

/// Circle participant
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircleParticipant {
    pub did: String,
    pub role: CircleRole,
    pub consented: bool,
    pub attended_sessions: Vec<u32>,
}

/// Roles in restorative circle
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CircleRole {
    Facilitator,
    HarmDoer,
    HarmReceiver,
    CommunityMember,
    SupportPerson,
    Elder,
}

/// Circle status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CircleStatus {
    Forming,
    Active,
    AgreementReached,
    Monitoring,
    Completed,
    Discontinued,
}

/// Circle session
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircleSession {
    pub session_number: u32,
    pub held_at: Timestamp,
    pub attendees: Vec<String>,
    pub summary: String,
    pub next_steps: Vec<String>,
}

// ============================================================================
// ENTRY ENUM
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Case(Case),
    #[entry_type(visibility = "public")]
    Evidence(Evidence),
    #[entry_type(visibility = "public")]
    Mediation(Mediation),
    #[entry_type(visibility = "public")]
    Arbitration(Arbitration),
    #[entry_type(visibility = "public")]
    Decision(Decision),
    #[entry_type(visibility = "public")]
    Appeal(Appeal),
    #[entry_type(visibility = "public")]
    Enforcement(Enforcement),
    #[entry_type(visibility = "public")]
    RestorativeCircle(RestorativeCircle),
}

// ============================================================================
// LINK TYPES
// ============================================================================

#[hdk_link_types]
pub enum LinkTypes {
    /// Complainant -> Cases
    ComplainantToCases,
    /// Respondent -> Cases
    RespondentToCases,
    /// Case -> Evidence
    CaseToEvidence,
    /// Case -> Mediation
    CaseToMediation,
    /// Case -> Arbitration
    CaseToArbitration,
    /// Case -> Decisions
    CaseToDecisions,
    /// Decision -> Appeals
    DecisionToAppeals,
    /// Decision -> Enforcement
    DecisionToEnforcement,
    /// Case -> RestorativeCircle
    CaseToRestorativeCircle,
    /// Arbitrator -> Cases
    ArbitratorToCases,
    /// All cases path
    AllCases,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(OpEntry::CreateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Case(c) => validate_case(&c),
            EntryTypes::Evidence(e) => validate_evidence(&e),
            EntryTypes::Mediation(m) => validate_mediation(&m),
            EntryTypes::Arbitration(a) => validate_arbitration(&a),
            EntryTypes::Decision(d) => validate_decision(&d),
            EntryTypes::Appeal(a) => validate_appeal(&a),
            EntryTypes::Enforcement(e) => validate_enforcement(&e),
            EntryTypes::RestorativeCircle(r) => validate_restorative(&r),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::ComplainantToCases => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RespondentToCases => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CaseToEvidence => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CaseToMediation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CaseToArbitration => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CaseToDecisions => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DecisionToAppeals => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DecisionToEnforcement => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CaseToRestorativeCircle => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ArbitratorToCases => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllCases => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_case(case: &Case) -> ExternResult<ValidateCallbackResult> {
    // Title required
    if case.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Case title required".into(),
        ));
    }

    // Description required
    if case.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Case description required".into(),
        ));
    }

    // Parties must be DIDs
    if !case.complainant.starts_with("did:") || !case.respondent.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Case parties must be DIDs".into(),
        ));
    }

    // Can't file case against self
    if case.complainant == case.respondent {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot file case against self".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_evidence(evidence: &Evidence) -> ExternResult<ValidateCallbackResult> {
    // Submitter must be DID
    if !evidence.submitter.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence submitter must be a DID".into(),
        ));
    }

    // Description required
    if evidence.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence description required".into(),
        ));
    }

    // Content hash required
    if evidence.content.hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence content hash required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_mediation(mediation: &Mediation) -> ExternResult<ValidateCallbackResult> {
    // Mediator must be DID
    if !mediation.mediator.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Mediator must be a DID".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_arbitration(arb: &Arbitration) -> ExternResult<ValidateCallbackResult> {
    // Must have odd number of arbitrators for voting
    if arb.arbitrators.len() % 2 == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Arbitration panel must have odd number of arbitrators".into(),
        ));
    }

    // All arbitrators must be DIDs
    for a in &arb.arbitrators {
        if !a.did.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "All arbitrators must be DIDs".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_decision(decision: &Decision) -> ExternResult<ValidateCallbackResult> {
    // Reasoning required
    if decision.reasoning.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision reasoning required".into(),
        ));
    }

    // Must have votes
    if decision.votes.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision must have votes".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_appeal(appeal: &Appeal) -> ExternResult<ValidateCallbackResult> {
    // Appellant must be DID
    if !appeal.appellant.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Appellant must be a DID".into(),
        ));
    }

    // Must have grounds
    if appeal.grounds.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Appeal must state grounds".into(),
        ));
    }

    // Argument required
    if appeal.argument.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Appeal argument required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_enforcement(enforcement: &Enforcement) -> ExternResult<ValidateCallbackResult> {
    // Enforcer must be DID
    if !enforcement.enforcer.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Enforcer must be a DID".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_restorative(circle: &RestorativeCircle) -> ExternResult<ValidateCallbackResult> {
    // Facilitator must be DID
    if !circle.facilitator.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Facilitator must be a DID".into(),
        ));
    }

    // Must have participants
    if circle.participants.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Restorative circle must have participants".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
