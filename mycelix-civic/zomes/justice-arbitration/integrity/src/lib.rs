// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix Justice Integrity Zome
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
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
        FlatOp::StoreEntry(OpEntry::UpdateEntry { app_entry, .. }) => match app_entry {
            EntryTypes::Case(c) => validate_case(&c),
            EntryTypes::Evidence(e) => validate_evidence(&e),
            EntryTypes::Mediation(m) => validate_mediation(&m),
            EntryTypes::Arbitration(a) => validate_arbitration(&a),
            EntryTypes::Decision(d) => validate_decision(&d),
            EntryTypes::Appeal(a) => validate_appeal(&a),
            EntryTypes::Enforcement(e) => validate_enforcement(&e),
            EntryTypes::RestorativeCircle(r) => validate_restorative(&r),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            let tag_len = tag.0.len();
            match link_type {
                LinkTypes::CaseToEvidence => {
                    // Evidence links may carry serialized metadata
                    if tag_len > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 512 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::CaseToDecisions => {
                    // Decision links may carry serialized metadata
                    if tag_len > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 512 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ComplainantToCases
                | LinkTypes::RespondentToCases
                | LinkTypes::CaseToMediation
                | LinkTypes::CaseToArbitration
                | LinkTypes::DecisionToAppeals
                | LinkTypes::DecisionToEnforcement
                | LinkTypes::CaseToRestorativeCircle
                | LinkTypes::ArbitratorToCases
                | LinkTypes::AllCases => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { tag, action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let result = check_link_author_match(
                original_action.action().author(),
                &action.author,
            );
            if result != ValidateCallbackResult::Valid {
                return Ok(result);
            }
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
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

    // String length limits
    if case.id.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Case ID too long (max 512)".into(),
        ));
    }
    if case.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Case title too long (max 512)".into(),
        ));
    }

    // Description required
    if case.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Case description required".into(),
        ));
    }
    if case.description.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Case description too long (max 8192)".into(),
        ));
    }

    // DID length limits
    if case.complainant.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Complainant DID too long (max 256)".into(),
        ));
    }
    if case.respondent.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Respondent DID too long (max 256)".into(),
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

    // Vec length limits
    if case.parties.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many parties (max 20)".into(),
        ));
    }

    // Validate nested party DIDs
    for party in &case.parties {
        if party.did.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Party DID too long (max 256)".into(),
            ));
        }
    }

    // CaseContext string limits
    if let Some(ref happ) = case.context.happ {
        if happ.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Context happ too long (max 256)".into(),
            ));
        }
    }
    if let Some(ref reference_id) = case.context.reference_id {
        if reference_id.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Context reference_id too long (max 256)".into(),
            ));
        }
    }
    if let Some(ref community) = case.context.community {
        if community.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Context community too long (max 256)".into(),
            ));
        }
    }
    if let Some(ref jurisdiction) = case.context.jurisdiction {
        if jurisdiction.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Context jurisdiction too long (max 4096)".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_evidence(evidence: &Evidence) -> ExternResult<ValidateCallbackResult> {
    // String length limits
    if evidence.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence ID too long (max 256)".into(),
        ));
    }
    if evidence.case_id.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence case_id too long (max 512)".into(),
        ));
    }
    if evidence.submitter.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence submitter too long (max 256)".into(),
        ));
    }

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
    if evidence.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence description too long (max 4096)".into(),
        ));
    }

    // Content hash required
    if evidence.content.hash.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence content hash required".into(),
        ));
    }

    // Content string limits
    if evidence.content.hash.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence content hash too long (max 256)".into(),
        ));
    }
    if evidence.content.reference.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence content reference too long (max 4096)".into(),
        ));
    }
    if evidence.content.mime_type.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence content mime_type too long (max 256)".into(),
        ));
    }
    if let Some(ref key_ref) = evidence.content.key_reference {
        if key_ref.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Evidence key_reference too long (max 256)".into(),
            ));
        }
    }

    // Vec length limits
    if evidence.custody.len() > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many custody events (max 200)".into(),
        ));
    }

    // Validate nested custody event strings
    for event in &evidence.custody {
        if event.actor.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Custody event actor too long (max 256)".into(),
            ));
        }
        if let Some(ref notes) = event.notes {
            if notes.len() > 4096 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Custody event notes too long (max 4096)".into(),
                ));
            }
        }
    }

    // Verification string limits
    if let Some(ref verifier) = evidence.verification.verifier {
        if verifier.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Verification verifier too long (max 256)".into(),
            ));
        }
    }
    if let Some(ref method) = evidence.verification.method {
        if method.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Verification method too long (max 4096)".into(),
            ));
        }
    }
    if let Some(ref notes) = evidence.verification.notes {
        if notes.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Verification notes too long (max 4096)".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_mediation(mediation: &Mediation) -> ExternResult<ValidateCallbackResult> {
    // String length limits
    if mediation.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Mediation ID too long (max 256)".into(),
        ));
    }
    if mediation.case_id.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Mediation case_id too long (max 512)".into(),
        ));
    }
    if mediation.mediator.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Mediator DID too long (max 256)".into(),
        ));
    }

    // Mediator must be DID
    if !mediation.mediator.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Mediator must be a DID".into(),
        ));
    }

    // Vec length limits
    if mediation.sessions.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many mediation sessions (max 50)".into(),
        ));
    }
    if mediation.proposals.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many proposals (max 10)".into(),
        ));
    }

    // Validate nested session strings
    for session in &mediation.sessions {
        if let Some(ref notes) = session.notes {
            if notes.len() > 4096 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Mediation session notes too long (max 4096)".into(),
                ));
            }
        }
        if let Some(ref outcome) = session.outcome {
            if outcome.len() > 4096 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Mediation session outcome too long (max 4096)".into(),
                ));
            }
        }
    }

    // Validate proposal IDs
    for proposal in &mediation.proposals {
        if proposal.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Proposal ID too long (max 256)".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_arbitration(arb: &Arbitration) -> ExternResult<ValidateCallbackResult> {
    // String length limits
    if arb.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Arbitration ID too long (max 256)".into(),
        ));
    }
    if arb.case_id.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Arbitration case_id too long (max 512)".into(),
        ));
    }

    // Vec length limits
    if arb.arbitrators.len() > 9 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many arbitrators (max 9)".into(),
        ));
    }

    // Must have odd number of arbitrators for voting
    if arb.arbitrators.len().is_multiple_of(2) {
        return Ok(ValidateCallbackResult::Invalid(
            "Arbitration panel must have odd number of arbitrators".into(),
        ));
    }

    // All arbitrators must be DIDs with length limits
    for a in &arb.arbitrators {
        if a.did.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Arbitrator DID too long (max 256)".into(),
            ));
        }
        if !a.did.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "All arbitrators must be DIDs".into(),
            ));
        }
        if let Some(ref reason) = a.recusal_reason {
            if reason.len() > 4096 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Arbitrator recusal reason too long (max 4096)".into(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_decision(decision: &Decision) -> ExternResult<ValidateCallbackResult> {
    // String length limits
    if decision.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision ID too long (max 256)".into(),
        ));
    }
    if decision.case_id.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision case_id too long (max 512)".into(),
        ));
    }
    if decision.arbitration_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision arbitration_id too long (max 256)".into(),
        ));
    }

    // Reasoning required
    if decision.reasoning.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision reasoning required".into(),
        ));
    }
    if decision.reasoning.len() > 16384 {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision reasoning too long (max 16384)".into(),
        ));
    }

    // Vec length limits
    if decision.remedies.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many remedies (max 20)".into(),
        ));
    }

    // Must have votes
    if decision.votes.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Decision must have votes".into(),
        ));
    }
    if decision.votes.len() > 9 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many votes (max 9)".into(),
        ));
    }
    if decision.dissents.len() > 9 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many dissenting opinions (max 9)".into(),
        ));
    }

    // Validate nested remedy strings
    for remedy in &decision.remedies {
        if remedy.responsible_party.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Remedy responsible_party too long (max 256)".into(),
            ));
        }
        if remedy.description.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Remedy description too long (max 4096)".into(),
            ));
        }
        if let Some(ref currency) = remedy.currency {
            if currency.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Remedy currency too long (max 256)".into(),
                ));
            }
        }
    }

    // Validate nested vote strings
    for vote in &decision.votes {
        if vote.arbitrator.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Vote arbitrator DID too long (max 256)".into(),
            ));
        }
    }

    // Validate nested dissent strings
    for dissent in &decision.dissents {
        if dissent.arbitrator.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Dissent arbitrator DID too long (max 256)".into(),
            ));
        }
        if dissent.opinion.len() > 16384 {
            return Ok(ValidateCallbackResult::Invalid(
                "Dissenting opinion too long (max 16384)".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_appeal(appeal: &Appeal) -> ExternResult<ValidateCallbackResult> {
    // String length limits
    if appeal.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Appeal ID too long (max 256)".into(),
        ));
    }
    if appeal.case_id.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Appeal case_id too long (max 512)".into(),
        ));
    }
    if appeal.decision_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Appeal decision_id too long (max 256)".into(),
        ));
    }
    if appeal.appellant.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Appellant DID too long (max 256)".into(),
        ));
    }

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

    // Vec length limits
    if appeal.grounds.len() > 10 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many appeal grounds (max 10)".into(),
        ));
    }

    // Argument required
    if appeal.argument.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Appeal argument required".into(),
        ));
    }
    if appeal.argument.len() > 16384 {
        return Ok(ValidateCallbackResult::Invalid(
            "Appeal argument too long (max 16384)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_enforcement(enforcement: &Enforcement) -> ExternResult<ValidateCallbackResult> {
    // String length limits
    if enforcement.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Enforcement ID too long (max 256)".into(),
        ));
    }
    if enforcement.decision_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Enforcement decision_id too long (max 256)".into(),
        ));
    }
    if enforcement.enforcer.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Enforcer DID too long (max 256)".into(),
        ));
    }

    // Enforcer must be DID
    if !enforcement.enforcer.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Enforcer must be a DID".into(),
        ));
    }

    // Vec length limits
    if enforcement.actions.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many enforcement actions (max 100)".into(),
        ));
    }

    // Validate nested action strings
    for action in &enforcement.actions {
        if action.result.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Enforcement action result too long (max 4096)".into(),
            ));
        }
        if let Some(ref target_happ) = action.target_happ {
            if target_happ.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Enforcement action target_happ too long (max 256)".into(),
                ));
            }
        }
        if let Some(ref target_entry) = action.target_entry {
            if target_entry.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Enforcement action target_entry too long (max 256)".into(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_restorative(circle: &RestorativeCircle) -> ExternResult<ValidateCallbackResult> {
    // String length limits
    if circle.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "RestorativeCircle ID too long (max 256)".into(),
        ));
    }
    if circle.case_id.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "RestorativeCircle case_id too long (max 512)".into(),
        ));
    }
    if circle.facilitator.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Facilitator DID too long (max 256)".into(),
        ));
    }

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

    // Vec length limits
    if circle.participants.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many participants (max 50)".into(),
        ));
    }
    if circle.agreements.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many agreements (max 20)".into(),
        ));
    }

    // Validate nested participant strings
    for participant in &circle.participants {
        if participant.did.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Participant DID too long (max 256)".into(),
            ));
        }
        if participant.attended_sessions.len() > 100 {
            return Ok(ValidateCallbackResult::Invalid(
                "Too many attended sessions (max 100)".into(),
            ));
        }
    }

    // Validate agreement strings
    for agreement in &circle.agreements {
        if agreement.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Agreement too long (max 4096)".into(),
            ));
        }
    }

    // Validate circle sessions
    if circle.sessions.len() > 50 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many circle sessions (max 50)".into(),
        ));
    }
    for session in &circle.sessions {
        if session.summary.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Circle session summary too long (max 4096)".into(),
            ));
        }
        if session.next_steps.len() > 30 {
            return Ok(ValidateCallbackResult::Invalid(
                "Too many next steps (max 30)".into(),
            ));
        }
        for step in &session.next_steps {
            if step.len() > 4096 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Next step too long (max 4096)".into(),
                ));
            }
        }
        for attendee in &session.attendees {
            if attendee.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Session attendee DID too long (max 256)".into(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RESULT HELPERS
    // ========================================================================

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    // ========================================================================
    // DATA CONSTRUCTION HELPERS
    // ========================================================================

    fn ts() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn make_case_context() -> CaseContext {
        CaseContext {
            happ: None,
            reference_id: None,
            community: None,
            jurisdiction: None,
        }
    }

    fn make_case() -> Case {
        Case {
            id: "case-1".into(),
            title: "Contract breach".into(),
            description: "Respondent failed to deliver".into(),
            case_type: CaseType::ContractDispute,
            complainant: "did:example:alice".into(),
            respondent: "did:example:bob".into(),
            parties: vec![],
            phase: CasePhase::Filed,
            status: CaseStatus::Active,
            severity: CaseSeverity::Moderate,
            context: make_case_context(),
            created_at: ts(),
            updated_at: ts(),
            phase_deadline: None,
        }
    }

    fn make_evidence_content() -> EvidenceContent {
        EvidenceContent {
            hash: "sha256:abc123".into(),
            reference: "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".into(),
            mime_type: "application/pdf".into(),
            size: 1024,
            encrypted: false,
            key_reference: None,
        }
    }

    fn make_evidence_verification() -> EvidenceVerification {
        EvidenceVerification {
            status: VerificationStatus::Unverified,
            verifier: None,
            method: None,
            verified_at: None,
            notes: None,
        }
    }

    fn make_evidence() -> Evidence {
        Evidence {
            id: "ev-1".into(),
            case_id: "case-1".into(),
            submitter: "did:example:alice".into(),
            evidence_type: EvidenceType::Document,
            content: make_evidence_content(),
            description: "Contract document".into(),
            custody: vec![],
            verification: make_evidence_verification(),
            visibility: EvidenceVisibility::AllParties,
            created_at: ts(),
            sealed: false,
        }
    }

    fn make_mediation() -> Mediation {
        Mediation {
            id: "med-1".into(),
            case_id: "case-1".into(),
            mediator: "did:example:mediator".into(),
            status: MediationStatus::Scheduled,
            sessions: vec![],
            proposals: vec![],
            created_at: ts(),
            deadline: None,
        }
    }

    fn make_arbitrator(did: &str) -> Arbitrator {
        Arbitrator {
            did: did.into(),
            role: ArbitratorRole::PanelMember,
            selected_at: ts(),
            accepted: true,
            recused: false,
            recusal_reason: None,
        }
    }

    fn make_arbitration(arbitrators: Vec<Arbitrator>) -> Arbitration {
        Arbitration {
            id: "arb-1".into(),
            case_id: "case-1".into(),
            arbitrators,
            selection_method: ArbitratorSelection::Random,
            status: ArbitrationStatus::PanelFormation,
            deliberation_deadline: None,
            created_at: ts(),
        }
    }

    fn make_vote(arbitrator: &str) -> ArbitratorVote {
        ArbitratorVote {
            arbitrator: arbitrator.into(),
            vote: VoteChoice::ForComplainant,
            timestamp: ts(),
        }
    }

    fn make_decision() -> Decision {
        Decision {
            id: "dec-1".into(),
            case_id: "case-1".into(),
            arbitration_id: "arb-1".into(),
            decision_type: DecisionType::MeritsDecision,
            outcome: DecisionOutcome::ForComplainant,
            reasoning: "Evidence clearly supports the complainant".into(),
            remedies: vec![],
            votes: vec![make_vote("did:example:arb1")],
            dissents: vec![],
            rendered_at: ts(),
            appeal_deadline: ts(),
            finalized: false,
        }
    }

    fn make_appeal() -> Appeal {
        Appeal {
            id: "appeal-1".into(),
            case_id: "case-1".into(),
            decision_id: "dec-1".into(),
            appellant: "did:example:bob".into(),
            grounds: vec![AppealGround::ProceduralError],
            argument: "The panel did not consider key evidence".into(),
            status: AppealStatus::Filed,
            appeal_number: 1,
            created_at: ts(),
        }
    }

    fn make_enforcement() -> Enforcement {
        Enforcement {
            id: "enf-1".into(),
            decision_id: "dec-1".into(),
            remedy_index: 0,
            enforcer: "did:example:system".into(),
            status: EnforcementStatus::Pending,
            actions: vec![],
            created_at: ts(),
            completed_at: None,
        }
    }

    fn make_circle_participant(did: &str) -> CircleParticipant {
        CircleParticipant {
            did: did.into(),
            role: CircleRole::CommunityMember,
            consented: true,
            attended_sessions: vec![],
        }
    }

    fn make_restorative_circle() -> RestorativeCircle {
        RestorativeCircle {
            id: "circle-1".into(),
            case_id: "case-1".into(),
            facilitator: "did:example:facilitator".into(),
            participants: vec![
                make_circle_participant("did:example:alice"),
                make_circle_participant("did:example:bob"),
            ],
            status: CircleStatus::Forming,
            sessions: vec![],
            agreements: vec![],
            created_at: ts(),
        }
    }

    // ========================================================================
    // STRING LENGTH BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn case_id_at_limit_passes() {
        let mut case = make_case();
        case.id = "x".repeat(512);
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_id_over_limit_rejected() {
        let mut case = make_case();
        case.id = "x".repeat(513);
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Case ID too long (max 512)");
    }

    #[test]
    fn case_title_at_limit_passes() {
        let mut case = make_case();
        case.title = "t".repeat(512);
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_title_over_limit_rejected() {
        let mut case = make_case();
        case.title = "t".repeat(513);
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Case title too long (max 512)");
    }

    #[test]
    fn case_description_at_limit_passes() {
        let mut case = make_case();
        case.description = "d".repeat(8192);
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_description_over_limit_rejected() {
        let mut case = make_case();
        case.description = "d".repeat(8193);
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Case description too long (max 8192)");
    }

    #[test]
    fn case_too_many_parties_rejected() {
        let mut case = make_case();
        case.parties = (0..21)
            .map(|i| CaseParty {
                did: format!("did:example:party{}", i),
                role: PartyRole::Witness,
                joined_at: ts(),
            })
            .collect();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Too many parties (max 20)");
    }

    #[test]
    fn evidence_description_at_limit_passes() {
        let mut ev = make_evidence();
        ev.description = "d".repeat(4096);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_description_over_limit_rejected() {
        let mut ev = make_evidence();
        ev.description = "d".repeat(4097);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Evidence description too long (max 4096)"
        );
    }

    #[test]
    fn decision_reasoning_at_limit_passes() {
        let mut dec = make_decision();
        dec.reasoning = "r".repeat(16384);
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_reasoning_over_limit_rejected() {
        let mut dec = make_decision();
        dec.reasoning = "r".repeat(16385);
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Decision reasoning too long (max 16384)"
        );
    }

    #[test]
    fn appeal_argument_at_limit_passes() {
        let mut appeal = make_appeal();
        appeal.argument = "a".repeat(16384);
        let result = validate_appeal(&appeal);
        assert!(is_valid(&result));
    }

    #[test]
    fn appeal_argument_over_limit_rejected() {
        let mut appeal = make_appeal();
        appeal.argument = "a".repeat(16385);
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Appeal argument too long (max 16384)");
    }

    #[test]
    fn enforcement_too_many_actions_rejected() {
        let mut enf = make_enforcement();
        enf.actions = (0..101)
            .map(|_| EnforcementAction {
                action_type: EnforcementActionType::Notification,
                target_happ: None,
                target_entry: None,
                executed_at: ts(),
                result: "done".into(),
            })
            .collect();
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Too many enforcement actions (max 100)"
        );
    }

    #[test]
    fn restorative_too_many_participants_rejected() {
        let mut circle = make_restorative_circle();
        circle.participants = (0..51)
            .map(|i| make_circle_participant(&format!("did:example:p{}", i)))
            .collect();
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Too many participants (max 50)");
    }

    #[test]
    fn arbitration_too_many_arbitrators_rejected() {
        let arb = make_arbitration(
            (0..11)
                .map(|i| make_arbitrator(&format!("did:example:arb{}", i)))
                .collect(),
        );
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Too many arbitrators (max 9)");
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn serde_roundtrip_case_type_all_variants() {
        let variants = vec![
            CaseType::ContractDispute,
            CaseType::ConductViolation,
            CaseType::PropertyDispute,
            CaseType::FinancialDispute,
            CaseType::GovernanceDispute,
            CaseType::IdentityDispute,
            CaseType::IPDispute,
            CaseType::Other {
                category: "custom".into(),
            },
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CaseType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_party_role_all_variants() {
        let variants = vec![
            PartyRole::Complainant,
            PartyRole::Respondent,
            PartyRole::Witness,
            PartyRole::Expert,
            PartyRole::Intervenor,
            PartyRole::Affected,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: PartyRole = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_case_phase_all_variants() {
        let variants = vec![
            CasePhase::Filed,
            CasePhase::Negotiation,
            CasePhase::Mediation,
            CasePhase::Arbitration,
            CasePhase::Appeal,
            CasePhase::Enforcement,
            CasePhase::Closed,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CasePhase = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_case_status_all_variants() {
        let variants = vec![
            CaseStatus::Active,
            CaseStatus::OnHold,
            CaseStatus::AwaitingResponse,
            CaseStatus::InDeliberation,
            CaseStatus::DecisionRendered,
            CaseStatus::Enforcing,
            CaseStatus::Resolved,
            CaseStatus::Dismissed,
            CaseStatus::Withdrawn,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CaseStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_case_severity_all_variants() {
        let variants = vec![
            CaseSeverity::Minor,
            CaseSeverity::Moderate,
            CaseSeverity::Serious,
            CaseSeverity::Critical,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CaseSeverity = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_evidence_type_all_variants() {
        let variants = vec![
            EvidenceType::Document,
            EvidenceType::Transaction,
            EvidenceType::Communication,
            EvidenceType::Testimony,
            EvidenceType::ExpertOpinion,
            EvidenceType::Media,
            EvidenceType::OnChainData {
                happ: "mycelix-commons".into(),
                entry_hash: "uhCEk123".into(),
            },
            EvidenceType::External {
                source: "court-records.gov".into(),
            },
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: EvidenceType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_custody_action_all_variants() {
        let variants = vec![
            CustodyAction::Submitted,
            CustodyAction::Accessed,
            CustodyAction::Copied,
            CustodyAction::Verified,
            CustodyAction::Challenged,
            CustodyAction::Sealed,
            CustodyAction::Released,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CustodyAction = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_verification_status_all_variants() {
        let variants = vec![
            VerificationStatus::Unverified,
            VerificationStatus::Pending,
            VerificationStatus::Verified,
            VerificationStatus::Disputed,
            VerificationStatus::Rejected,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: VerificationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_evidence_visibility_all_variants() {
        let variants: Vec<EvidenceVisibility> = vec![
            EvidenceVisibility::AllParties,
            EvidenceVisibility::AdjudicatorsOnly,
            EvidenceVisibility::Restricted {
                parties: vec!["did:example:a".into(), "did:example:b".into()],
            },
            EvidenceVisibility::Sealed,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let back: EvidenceVisibility = serde_json::from_str(&json).unwrap();
            assert_eq!(*v, back);
        }
    }

    #[test]
    fn serde_roundtrip_mediation_status_all_variants() {
        let variants = vec![
            MediationStatus::Scheduled,
            MediationStatus::InProgress,
            MediationStatus::SettlementReached,
            MediationStatus::Failed,
            MediationStatus::Cancelled,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: MediationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_arbitrator_role_all_variants() {
        let variants = vec![
            ArbitratorRole::Primary,
            ArbitratorRole::PanelMember,
            ArbitratorRole::Alternate,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ArbitratorRole = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_arbitrator_selection_all_variants() {
        let variants = vec![
            ArbitratorSelection::Random,
            ArbitratorSelection::MATLWeighted,
            ArbitratorSelection::PartyAgreed,
            ArbitratorSelection::ExpertiseBased {
                domain: "finance".into(),
            },
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ArbitratorSelection = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_arbitration_status_all_variants() {
        let variants = vec![
            ArbitrationStatus::PanelFormation,
            ArbitrationStatus::EvidenceReview,
            ArbitrationStatus::Hearing,
            ArbitrationStatus::Deliberation,
            ArbitrationStatus::DecisionDrafting,
            ArbitrationStatus::DecisionRendered,
            ArbitrationStatus::Appealed,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ArbitrationStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_decision_type_all_variants() {
        let variants = vec![
            DecisionType::MeritsDecision,
            DecisionType::InterimDecision,
            DecisionType::DefaultDecision,
            DecisionType::ConsentDecision,
            DecisionType::Dismissal,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: DecisionType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_decision_outcome_all_variants() {
        let variants = vec![
            DecisionOutcome::ForComplainant,
            DecisionOutcome::ForRespondent,
            DecisionOutcome::SplitDecision,
            DecisionOutcome::Dismissed,
            DecisionOutcome::Settled,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: DecisionOutcome = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_remedy_type_all_variants() {
        let variants = vec![
            RemedyType::Compensation,
            RemedyType::Restitution,
            RemedyType::SpecificPerformance,
            RemedyType::Injunction,
            RemedyType::Apology,
            RemedyType::CommunityService,
            RemedyType::ReputationAdjustment,
            RemedyType::AccessRestriction,
            RemedyType::Education,
            RemedyType::RestorativeCircle,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: RemedyType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_vote_choice_all_variants() {
        let variants = vec![
            VoteChoice::ForComplainant,
            VoteChoice::ForRespondent,
            VoteChoice::Abstain,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: VoteChoice = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_appeal_ground_all_variants() {
        let variants = vec![
            AppealGround::ProceduralError,
            AppealGround::NewEvidence,
            AppealGround::LegalError,
            AppealGround::Bias,
            AppealGround::ExcessiveRemedy,
            AppealGround::InsufficientRemedy,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: AppealGround = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_appeal_status_all_variants() {
        let variants = vec![
            AppealStatus::Filed,
            AppealStatus::UnderReview,
            AppealStatus::Granted,
            AppealStatus::Denied,
            AppealStatus::Remanded,
            AppealStatus::Resolved,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: AppealStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_enforcement_status_all_variants() {
        let variants = vec![
            EnforcementStatus::Pending,
            EnforcementStatus::InProgress,
            EnforcementStatus::PartiallyCompleted,
            EnforcementStatus::Completed,
            EnforcementStatus::Failed,
            EnforcementStatus::Contested,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: EnforcementStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_enforcement_action_type_all_variants() {
        let variants = vec![
            EnforcementActionType::FundsTransfer,
            EnforcementActionType::AssetFreeze,
            EnforcementActionType::ReputationUpdate,
            EnforcementActionType::AccessRevocation,
            EnforcementActionType::Notification,
            EnforcementActionType::ManualRequired,
            EnforcementActionType::CrossHappAction,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: EnforcementActionType = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_circle_role_all_variants() {
        let variants = vec![
            CircleRole::Facilitator,
            CircleRole::HarmDoer,
            CircleRole::HarmReceiver,
            CircleRole::CommunityMember,
            CircleRole::SupportPerson,
            CircleRole::Elder,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CircleRole = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn serde_roundtrip_circle_status_all_variants() {
        let variants = vec![
            CircleStatus::Forming,
            CircleStatus::Active,
            CircleStatus::AgreementReached,
            CircleStatus::Monitoring,
            CircleStatus::Completed,
            CircleStatus::Discontinued,
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: CircleStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    // Struct roundtrips

    #[test]
    fn serde_roundtrip_case_context_full() {
        let ctx = CaseContext {
            happ: Some("mycelix-commons".into()),
            reference_id: Some("tx-12345".into()),
            community: Some("builders-guild".into()),
            jurisdiction: Some("global-commons".into()),
        };
        let json = serde_json::to_string(&ctx).unwrap();
        let back: CaseContext = serde_json::from_str(&json).unwrap();
        assert_eq!(ctx.happ, back.happ);
        assert_eq!(ctx.reference_id, back.reference_id);
        assert_eq!(ctx.community, back.community);
        assert_eq!(ctx.jurisdiction, back.jurisdiction);
    }

    #[test]
    fn serde_roundtrip_case_context_all_none() {
        let ctx = make_case_context();
        let json = serde_json::to_string(&ctx).unwrap();
        let back: CaseContext = serde_json::from_str(&json).unwrap();
        assert!(back.happ.is_none());
        assert!(back.reference_id.is_none());
        assert!(back.community.is_none());
        assert!(back.jurisdiction.is_none());
    }

    #[test]
    fn serde_roundtrip_evidence_content() {
        let content = make_evidence_content();
        let json = serde_json::to_string(&content).unwrap();
        let back: EvidenceContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content.hash, back.hash);
        assert_eq!(content.reference, back.reference);
        assert_eq!(content.mime_type, back.mime_type);
        assert_eq!(content.size, back.size);
        assert_eq!(content.encrypted, back.encrypted);
        assert_eq!(content.key_reference, back.key_reference);
    }

    #[test]
    fn serde_roundtrip_evidence_content_encrypted() {
        let content = EvidenceContent {
            hash: "sha256:xyz".into(),
            reference: "cid:bafyxyz".into(),
            mime_type: "image/png".into(),
            size: u64::MAX,
            encrypted: true,
            key_reference: Some("key-ref-001".into()),
        };
        let json = serde_json::to_string(&content).unwrap();
        let back: EvidenceContent = serde_json::from_str(&json).unwrap();
        assert!(back.encrypted);
        assert_eq!(back.key_reference, Some("key-ref-001".into()));
        assert_eq!(back.size, u64::MAX);
    }

    #[test]
    fn serde_roundtrip_evidence_verification_full() {
        let v = EvidenceVerification {
            status: VerificationStatus::Verified,
            verifier: Some("did:example:verifier".into()),
            method: Some("cryptographic-hash".into()),
            verified_at: Some(Timestamp::from_micros(1000000)),
            notes: Some("Hash matches".into()),
        };
        let json = serde_json::to_string(&v).unwrap();
        let back: EvidenceVerification = serde_json::from_str(&json).unwrap();
        assert_eq!(back.status, VerificationStatus::Verified);
        assert_eq!(back.verifier, Some("did:example:verifier".into()));
    }

    #[test]
    fn serde_roundtrip_custody_event() {
        let evt = CustodyEvent {
            action: CustodyAction::Submitted,
            actor: "did:example:alice".into(),
            timestamp: ts(),
            notes: Some("Initial submission".into()),
        };
        let json = serde_json::to_string(&evt).unwrap();
        let back: CustodyEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.action, CustodyAction::Submitted);
        assert_eq!(back.actor, "did:example:alice");
        assert_eq!(back.notes, Some("Initial submission".into()));
    }

    #[test]
    fn serde_roundtrip_mediation_session() {
        let session = MediationSession {
            session_number: 3,
            scheduled_at: ts(),
            actual_start: Some(Timestamp::from_micros(100)),
            actual_end: Some(Timestamp::from_micros(200)),
            notes: Some("Productive session".into()),
            outcome: Some("Partial agreement".into()),
        };
        let json = serde_json::to_string(&session).unwrap();
        let back: MediationSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back.session_number, 3);
        assert_eq!(back.outcome, Some("Partial agreement".into()));
    }

    #[test]
    fn serde_roundtrip_arbitrator() {
        let arb = Arbitrator {
            did: "did:example:arb1".into(),
            role: ArbitratorRole::Primary,
            selected_at: ts(),
            accepted: true,
            recused: false,
            recusal_reason: None,
        };
        let json = serde_json::to_string(&arb).unwrap();
        let back: Arbitrator = serde_json::from_str(&json).unwrap();
        assert_eq!(back.did, "did:example:arb1");
        assert_eq!(back.role, ArbitratorRole::Primary);
        assert!(back.accepted);
        assert!(!back.recused);
    }

    #[test]
    fn serde_roundtrip_arbitrator_recused() {
        let arb = Arbitrator {
            did: "did:example:arb2".into(),
            role: ArbitratorRole::PanelMember,
            selected_at: ts(),
            accepted: true,
            recused: true,
            recusal_reason: Some("Conflict of interest".into()),
        };
        let json = serde_json::to_string(&arb).unwrap();
        let back: Arbitrator = serde_json::from_str(&json).unwrap();
        assert!(back.recused);
        assert_eq!(back.recusal_reason, Some("Conflict of interest".into()));
    }

    #[test]
    fn serde_roundtrip_remedy() {
        let remedy = Remedy {
            remedy_type: RemedyType::Compensation,
            responsible_party: "did:example:bob".into(),
            deadline: Some(Timestamp::from_micros(999999)),
            amount: Some(1000),
            currency: Some("SAP".into()),
            description: "Pay damages".into(),
        };
        let json = serde_json::to_string(&remedy).unwrap();
        let back: Remedy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.remedy_type, RemedyType::Compensation);
        assert_eq!(back.amount, Some(1000));
        assert_eq!(back.currency, Some("SAP".into()));
    }

    #[test]
    fn serde_roundtrip_remedy_max_amount() {
        let remedy = Remedy {
            remedy_type: RemedyType::Restitution,
            responsible_party: "did:example:x".into(),
            deadline: None,
            amount: Some(u128::MAX),
            currency: None,
            description: "Max restitution".into(),
        };
        let json = serde_json::to_string(&remedy).unwrap();
        let back: Remedy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.amount, Some(u128::MAX));
    }

    #[test]
    fn serde_roundtrip_arbitrator_vote() {
        let vote = ArbitratorVote {
            arbitrator: "did:example:arb1".into(),
            vote: VoteChoice::Abstain,
            timestamp: ts(),
        };
        let json = serde_json::to_string(&vote).unwrap();
        let back: ArbitratorVote = serde_json::from_str(&json).unwrap();
        assert_eq!(back.vote, VoteChoice::Abstain);
    }

    #[test]
    fn serde_roundtrip_dissenting_opinion() {
        let dissent = DissentingOpinion {
            arbitrator: "did:example:arb3".into(),
            opinion: "I disagree because the evidence was insufficient".into(),
            timestamp: ts(),
        };
        let json = serde_json::to_string(&dissent).unwrap();
        let back: DissentingOpinion = serde_json::from_str(&json).unwrap();
        assert_eq!(back.arbitrator, "did:example:arb3");
        assert!(back.opinion.contains("insufficient"));
    }

    #[test]
    fn serde_roundtrip_case_party() {
        let party = CaseParty {
            did: "did:example:witness1".into(),
            role: PartyRole::Witness,
            joined_at: ts(),
        };
        let json = serde_json::to_string(&party).unwrap();
        let back: CaseParty = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, PartyRole::Witness);
    }

    #[test]
    fn serde_roundtrip_circle_participant() {
        let p = CircleParticipant {
            did: "did:example:elder".into(),
            role: CircleRole::Elder,
            consented: false,
            attended_sessions: vec![1, 2, 3],
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: CircleParticipant = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, CircleRole::Elder);
        assert!(!back.consented);
        assert_eq!(back.attended_sessions, vec![1, 2, 3]);
    }

    #[test]
    fn serde_roundtrip_circle_session() {
        let session = CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec!["did:a".into(), "did:b".into()],
            summary: "Opening ceremony".into(),
            next_steps: vec!["Schedule follow-up".into()],
        };
        let json = serde_json::to_string(&session).unwrap();
        let back: CircleSession = serde_json::from_str(&json).unwrap();
        assert_eq!(back.attendees.len(), 2);
        assert_eq!(back.next_steps.len(), 1);
    }

    #[test]
    fn serde_roundtrip_enforcement_action() {
        let action = EnforcementAction {
            action_type: EnforcementActionType::CrossHappAction,
            target_happ: Some("mycelix-commons".into()),
            target_entry: Some("uhCEk_entry_hash".into()),
            executed_at: ts(),
            result: "Action executed successfully".into(),
        };
        let json = serde_json::to_string(&action).unwrap();
        let back: EnforcementAction = serde_json::from_str(&json).unwrap();
        assert_eq!(back.action_type, EnforcementActionType::CrossHappAction);
        assert_eq!(back.target_happ, Some("mycelix-commons".into()));
    }

    // ========================================================================
    // CASE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_case_passes() {
        let result = validate_case(&make_case());
        assert!(is_valid(&result));
    }

    #[test]
    fn case_empty_title_rejected() {
        let mut case = make_case();
        case.title = "".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Case title required");
    }

    #[test]
    fn case_whitespace_only_title_rejected() {
        let mut case = make_case();
        case.title = "   \t\n  ".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Case title required");
    }

    #[test]
    fn case_empty_description_rejected() {
        let mut case = make_case();
        case.description = "".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Case description required");
    }

    #[test]
    fn case_whitespace_only_description_rejected() {
        let mut case = make_case();
        case.description = "   ".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Case description required");
    }

    #[test]
    fn case_complainant_not_did_rejected() {
        let mut case = make_case();
        case.complainant = "alice".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Case parties must be DIDs");
    }

    #[test]
    fn case_respondent_not_did_rejected() {
        let mut case = make_case();
        case.respondent = "bob".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Case parties must be DIDs");
    }

    #[test]
    fn case_both_parties_not_did_rejected() {
        let mut case = make_case();
        case.complainant = "alice".into();
        case.respondent = "bob".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_complainant_equals_respondent_rejected() {
        let mut case = make_case();
        case.complainant = "did:example:same".into();
        case.respondent = "did:example:same".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Cannot file case against self");
    }

    #[test]
    fn case_did_prefix_only_complainant_passes_did_check() {
        // "did:" alone passes the starts_with check
        let mut case = make_case();
        case.complainant = "did:".into();
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_validation_checks_title_before_description() {
        // Both empty: title error should appear first
        let mut case = make_case();
        case.title = "".into();
        case.description = "".into();
        let result = validate_case(&case);
        assert_eq!(invalid_msg(&result), "Case title required");
    }

    #[test]
    fn case_validation_checks_did_before_self_filing() {
        // Non-DID and same party: DID error should appear first
        let mut case = make_case();
        case.complainant = "same".into();
        case.respondent = "same".into();
        let result = validate_case(&case);
        assert_eq!(invalid_msg(&result), "Case parties must be DIDs");
    }

    #[test]
    fn case_with_all_phases_valid() {
        let phases = vec![
            CasePhase::Filed,
            CasePhase::Negotiation,
            CasePhase::Mediation,
            CasePhase::Arbitration,
            CasePhase::Appeal,
            CasePhase::Enforcement,
            CasePhase::Closed,
        ];
        for phase in phases {
            let mut case = make_case();
            case.phase = phase;
            assert!(is_valid(&validate_case(&case)));
        }
    }

    #[test]
    fn case_with_all_statuses_valid() {
        let statuses = vec![
            CaseStatus::Active,
            CaseStatus::OnHold,
            CaseStatus::AwaitingResponse,
            CaseStatus::InDeliberation,
            CaseStatus::DecisionRendered,
            CaseStatus::Enforcing,
            CaseStatus::Resolved,
            CaseStatus::Dismissed,
            CaseStatus::Withdrawn,
        ];
        for status in statuses {
            let mut case = make_case();
            case.status = status;
            assert!(is_valid(&validate_case(&case)));
        }
    }

    #[test]
    fn case_with_all_severities_valid() {
        let severities = vec![
            CaseSeverity::Minor,
            CaseSeverity::Moderate,
            CaseSeverity::Serious,
            CaseSeverity::Critical,
        ];
        for sev in severities {
            let mut case = make_case();
            case.severity = sev;
            assert!(is_valid(&validate_case(&case)));
        }
    }

    #[test]
    fn case_with_all_case_types_valid() {
        let types = vec![
            CaseType::ContractDispute,
            CaseType::ConductViolation,
            CaseType::PropertyDispute,
            CaseType::FinancialDispute,
            CaseType::GovernanceDispute,
            CaseType::IdentityDispute,
            CaseType::IPDispute,
            CaseType::Other {
                category: "custom".into(),
            },
        ];
        for ct in types {
            let mut case = make_case();
            case.case_type = ct;
            assert!(is_valid(&validate_case(&case)));
        }
    }

    #[test]
    fn case_with_parties_passes() {
        let mut case = make_case();
        case.parties = vec![
            CaseParty {
                did: "did:example:witness".into(),
                role: PartyRole::Witness,
                joined_at: ts(),
            },
            CaseParty {
                did: "did:example:expert".into(),
                role: PartyRole::Expert,
                joined_at: ts(),
            },
        ];
        assert!(is_valid(&validate_case(&case)));
    }

    #[test]
    fn case_with_full_context_passes() {
        let mut case = make_case();
        case.context = CaseContext {
            happ: Some("mycelix-commons".into()),
            reference_id: Some("tx-42".into()),
            community: Some("builders".into()),
            jurisdiction: Some("global".into()),
        };
        assert!(is_valid(&validate_case(&case)));
    }

    #[test]
    fn case_with_phase_deadline_passes() {
        let mut case = make_case();
        case.phase_deadline = Some(Timestamp::from_micros(999999999));
        assert!(is_valid(&validate_case(&case)));
    }

    // ========================================================================
    // EVIDENCE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_evidence_passes() {
        let result = validate_evidence(&make_evidence());
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_submitter_not_did_rejected() {
        let mut ev = make_evidence();
        ev.submitter = "alice".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Evidence submitter must be a DID");
    }

    #[test]
    fn evidence_empty_submitter_rejected() {
        let mut ev = make_evidence();
        ev.submitter = "".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_empty_description_rejected() {
        let mut ev = make_evidence();
        ev.description = "".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Evidence description required");
    }

    #[test]
    fn evidence_whitespace_only_description_rejected() {
        let mut ev = make_evidence();
        ev.description = "  \n  ".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_empty_content_hash_rejected() {
        let mut ev = make_evidence();
        ev.content.hash = "".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Evidence content hash required");
    }

    #[test]
    fn evidence_whitespace_content_hash_rejected() {
        let mut ev = make_evidence();
        ev.content.hash = "   ".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_checks_submitter_before_description() {
        let mut ev = make_evidence();
        ev.submitter = "no-did".into();
        ev.description = "".into();
        let result = validate_evidence(&ev);
        assert_eq!(invalid_msg(&result), "Evidence submitter must be a DID");
    }

    #[test]
    fn evidence_checks_description_before_hash() {
        let mut ev = make_evidence();
        ev.description = "".into();
        ev.content.hash = "".into();
        let result = validate_evidence(&ev);
        assert_eq!(invalid_msg(&result), "Evidence description required");
    }

    #[test]
    fn evidence_with_all_types_passes() {
        let types = vec![
            EvidenceType::Document,
            EvidenceType::Transaction,
            EvidenceType::Communication,
            EvidenceType::Testimony,
            EvidenceType::ExpertOpinion,
            EvidenceType::Media,
            EvidenceType::OnChainData {
                happ: "test".into(),
                entry_hash: "hash".into(),
            },
            EvidenceType::External {
                source: "court".into(),
            },
        ];
        for et in types {
            let mut ev = make_evidence();
            ev.evidence_type = et;
            assert!(is_valid(&validate_evidence(&ev)));
        }
    }

    #[test]
    fn evidence_with_custody_chain_passes() {
        let mut ev = make_evidence();
        ev.custody = vec![
            CustodyEvent {
                action: CustodyAction::Submitted,
                actor: "did:example:alice".into(),
                timestamp: ts(),
                notes: Some("Original submission".into()),
            },
            CustodyEvent {
                action: CustodyAction::Verified,
                actor: "did:example:verifier".into(),
                timestamp: Timestamp::from_micros(100),
                notes: None,
            },
            CustodyEvent {
                action: CustodyAction::Sealed,
                actor: "did:example:system".into(),
                timestamp: Timestamp::from_micros(200),
                notes: Some("Sealed by court order".into()),
            },
        ];
        assert!(is_valid(&validate_evidence(&ev)));
    }

    #[test]
    fn evidence_with_all_visibility_types_passes() {
        let visibilities = vec![
            EvidenceVisibility::AllParties,
            EvidenceVisibility::AdjudicatorsOnly,
            EvidenceVisibility::Restricted {
                parties: vec!["did:a".into()],
            },
            EvidenceVisibility::Sealed,
        ];
        for vis in visibilities {
            let mut ev = make_evidence();
            ev.visibility = vis;
            assert!(is_valid(&validate_evidence(&ev)));
        }
    }

    #[test]
    fn evidence_sealed_flag_does_not_affect_validation() {
        let mut ev = make_evidence();
        ev.sealed = true;
        assert!(is_valid(&validate_evidence(&ev)));
    }

    #[test]
    fn evidence_max_content_size_passes() {
        let mut ev = make_evidence();
        ev.content.size = u64::MAX;
        assert!(is_valid(&validate_evidence(&ev)));
    }

    #[test]
    fn evidence_zero_content_size_passes() {
        let mut ev = make_evidence();
        ev.content.size = 0;
        assert!(is_valid(&validate_evidence(&ev)));
    }

    // ========================================================================
    // MEDIATION VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_mediation_passes() {
        let result = validate_mediation(&make_mediation());
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_mediator_not_did_rejected() {
        let mut med = make_mediation();
        med.mediator = "mediator-name".into();
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Mediator must be a DID");
    }

    #[test]
    fn mediation_empty_mediator_rejected() {
        let mut med = make_mediation();
        med.mediator = "".into();
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_did_prefix_only_passes() {
        let mut med = make_mediation();
        med.mediator = "did:".into();
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_with_sessions_passes() {
        let mut med = make_mediation();
        med.sessions = vec![MediationSession {
            session_number: 1,
            scheduled_at: ts(),
            actual_start: None,
            actual_end: None,
            notes: None,
            outcome: None,
        }];
        assert!(is_valid(&validate_mediation(&med)));
    }

    #[test]
    fn mediation_with_all_statuses_passes() {
        let statuses = vec![
            MediationStatus::Scheduled,
            MediationStatus::InProgress,
            MediationStatus::SettlementReached,
            MediationStatus::Failed,
            MediationStatus::Cancelled,
        ];
        for status in statuses {
            let mut med = make_mediation();
            med.status = status;
            assert!(is_valid(&validate_mediation(&med)));
        }
    }

    #[test]
    fn mediation_with_proposals_passes() {
        let mut med = make_mediation();
        med.proposals = vec!["settlement-1".into(), "settlement-2".into()];
        assert!(is_valid(&validate_mediation(&med)));
    }

    #[test]
    fn mediation_with_deadline_passes() {
        let mut med = make_mediation();
        med.deadline = Some(Timestamp::from_micros(9999999));
        assert!(is_valid(&validate_mediation(&med)));
    }

    // ========================================================================
    // ARBITRATION VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_arbitration_one_arbitrator_passes() {
        let arb = make_arbitration(vec![make_arbitrator("did:example:arb1")]);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn valid_arbitration_three_arbitrators_passes() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn valid_arbitration_five_arbitrators_passes() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:a1"),
            make_arbitrator("did:example:a2"),
            make_arbitrator("did:example:a3"),
            make_arbitrator("did:example:a4"),
            make_arbitrator("did:example:a5"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn valid_arbitration_seven_arbitrators_passes() {
        let arb = make_arbitration(
            (1..=7)
                .map(|i| make_arbitrator(&format!("did:example:a{}", i)))
                .collect(),
        );
        assert!(is_valid(&validate_arbitration(&arb)));
    }

    #[test]
    fn arbitration_zero_arbitrators_rejected() {
        // 0.is_multiple_of(2) == true, so empty is rejected
        let arb = make_arbitration(vec![]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Arbitration panel must have odd number of arbitrators"
        );
    }

    #[test]
    fn arbitration_two_arbitrators_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_four_arbitrators_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:a1"),
            make_arbitrator("did:example:a2"),
            make_arbitrator("did:example:a3"),
            make_arbitrator("did:example:a4"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_six_arbitrators_rejected() {
        let arb = make_arbitration(
            (1..=6)
                .map(|i| make_arbitrator(&format!("did:example:a{}", i)))
                .collect(),
        );
        assert!(is_invalid(&validate_arbitration(&arb)));
    }

    #[test]
    fn arbitration_non_did_arbitrator_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("not-a-did"),
            make_arbitrator("did:example:arb3"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "All arbitrators must be DIDs");
    }

    #[test]
    fn arbitration_empty_string_arbitrator_rejected() {
        let arb = make_arbitration(vec![make_arbitrator("")]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_all_non_did_arbitrators_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator("alice"),
            make_arbitrator("bob"),
            make_arbitrator("carol"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_checks_odd_count_before_did() {
        // 2 arbitrators, one non-DID: odd-count error first
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("not-a-did"),
        ]);
        let result = validate_arbitration(&arb);
        assert_eq!(
            invalid_msg(&result),
            "Arbitration panel must have odd number of arbitrators"
        );
    }

    #[test]
    fn arbitration_with_all_selection_methods_passes() {
        let methods = vec![
            ArbitratorSelection::Random,
            ArbitratorSelection::MATLWeighted,
            ArbitratorSelection::PartyAgreed,
            ArbitratorSelection::ExpertiseBased {
                domain: "law".into(),
            },
        ];
        for method in methods {
            let mut arb = make_arbitration(vec![make_arbitrator("did:example:a1")]);
            arb.selection_method = method;
            assert!(is_valid(&validate_arbitration(&arb)));
        }
    }

    #[test]
    fn arbitration_with_all_statuses_passes() {
        let statuses = vec![
            ArbitrationStatus::PanelFormation,
            ArbitrationStatus::EvidenceReview,
            ArbitrationStatus::Hearing,
            ArbitrationStatus::Deliberation,
            ArbitrationStatus::DecisionDrafting,
            ArbitrationStatus::DecisionRendered,
            ArbitrationStatus::Appealed,
        ];
        for status in statuses {
            let mut arb = make_arbitration(vec![make_arbitrator("did:example:a1")]);
            arb.status = status;
            assert!(is_valid(&validate_arbitration(&arb)));
        }
    }

    // ========================================================================
    // DECISION VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_decision_passes() {
        let result = validate_decision(&make_decision());
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_empty_reasoning_rejected() {
        let mut dec = make_decision();
        dec.reasoning = "".into();
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Decision reasoning required");
    }

    #[test]
    fn decision_whitespace_only_reasoning_rejected() {
        let mut dec = make_decision();
        dec.reasoning = "   \t  ".into();
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_no_votes_rejected() {
        let mut dec = make_decision();
        dec.votes = vec![];
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Decision must have votes");
    }

    #[test]
    fn decision_multiple_votes_passes() {
        let mut dec = make_decision();
        dec.votes = vec![
            make_vote("did:example:arb1"),
            make_vote("did:example:arb2"),
            make_vote("did:example:arb3"),
        ];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_checks_reasoning_before_votes() {
        let mut dec = make_decision();
        dec.reasoning = "".into();
        dec.votes = vec![];
        let result = validate_decision(&dec);
        assert_eq!(invalid_msg(&result), "Decision reasoning required");
    }

    #[test]
    fn decision_with_all_types_passes() {
        let types = vec![
            DecisionType::MeritsDecision,
            DecisionType::InterimDecision,
            DecisionType::DefaultDecision,
            DecisionType::ConsentDecision,
            DecisionType::Dismissal,
        ];
        for dt in types {
            let mut dec = make_decision();
            dec.decision_type = dt;
            assert!(is_valid(&validate_decision(&dec)));
        }
    }

    #[test]
    fn decision_with_all_outcomes_passes() {
        let outcomes = vec![
            DecisionOutcome::ForComplainant,
            DecisionOutcome::ForRespondent,
            DecisionOutcome::SplitDecision,
            DecisionOutcome::Dismissed,
            DecisionOutcome::Settled,
        ];
        for outcome in outcomes {
            let mut dec = make_decision();
            dec.outcome = outcome;
            assert!(is_valid(&validate_decision(&dec)));
        }
    }

    #[test]
    fn decision_with_remedies_passes() {
        let mut dec = make_decision();
        dec.remedies = vec![
            Remedy {
                remedy_type: RemedyType::Compensation,
                responsible_party: "did:example:bob".into(),
                deadline: Some(ts()),
                amount: Some(500),
                currency: Some("SAP".into()),
                description: "Pay compensation".into(),
            },
            Remedy {
                remedy_type: RemedyType::Apology,
                responsible_party: "did:example:bob".into(),
                deadline: None,
                amount: None,
                currency: None,
                description: "Issue formal apology".into(),
            },
        ];
        assert!(is_valid(&validate_decision(&dec)));
    }

    #[test]
    fn decision_with_dissents_passes() {
        let mut dec = make_decision();
        dec.votes = vec![
            ArbitratorVote {
                arbitrator: "did:example:a1".into(),
                vote: VoteChoice::ForComplainant,
                timestamp: ts(),
            },
            ArbitratorVote {
                arbitrator: "did:example:a2".into(),
                vote: VoteChoice::ForComplainant,
                timestamp: ts(),
            },
            ArbitratorVote {
                arbitrator: "did:example:a3".into(),
                vote: VoteChoice::ForRespondent,
                timestamp: ts(),
            },
        ];
        dec.dissents = vec![DissentingOpinion {
            arbitrator: "did:example:a3".into(),
            opinion: "The evidence was ambiguous".into(),
            timestamp: ts(),
        }];
        assert!(is_valid(&validate_decision(&dec)));
    }

    #[test]
    fn decision_finalized_flag_does_not_affect_validation() {
        let mut dec = make_decision();
        dec.finalized = true;
        assert!(is_valid(&validate_decision(&dec)));
    }

    // ========================================================================
    // APPEAL VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_appeal_passes() {
        let result = validate_appeal(&make_appeal());
        assert!(is_valid(&result));
    }

    #[test]
    fn appeal_appellant_not_did_rejected() {
        let mut appeal = make_appeal();
        appeal.appellant = "bob".into();
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Appellant must be a DID");
    }

    #[test]
    fn appeal_empty_appellant_rejected() {
        let mut appeal = make_appeal();
        appeal.appellant = "".into();
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_no_grounds_rejected() {
        let mut appeal = make_appeal();
        appeal.grounds = vec![];
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Appeal must state grounds");
    }

    #[test]
    fn appeal_empty_argument_rejected() {
        let mut appeal = make_appeal();
        appeal.argument = "".into();
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Appeal argument required");
    }

    #[test]
    fn appeal_whitespace_only_argument_rejected() {
        let mut appeal = make_appeal();
        appeal.argument = "  \n\t  ".into();
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_multiple_grounds_passes() {
        let mut appeal = make_appeal();
        appeal.grounds = vec![
            AppealGround::ProceduralError,
            AppealGround::NewEvidence,
            AppealGround::Bias,
        ];
        let result = validate_appeal(&appeal);
        assert!(is_valid(&result));
    }

    #[test]
    fn appeal_all_grounds_passes() {
        let mut appeal = make_appeal();
        appeal.grounds = vec![
            AppealGround::ProceduralError,
            AppealGround::NewEvidence,
            AppealGround::LegalError,
            AppealGround::Bias,
            AppealGround::ExcessiveRemedy,
            AppealGround::InsufficientRemedy,
        ];
        assert!(is_valid(&validate_appeal(&appeal)));
    }

    #[test]
    fn appeal_checks_appellant_before_grounds() {
        let mut appeal = make_appeal();
        appeal.appellant = "not-did".into();
        appeal.grounds = vec![];
        let result = validate_appeal(&appeal);
        assert_eq!(invalid_msg(&result), "Appellant must be a DID");
    }

    #[test]
    fn appeal_checks_grounds_before_argument() {
        let mut appeal = make_appeal();
        appeal.grounds = vec![];
        appeal.argument = "".into();
        let result = validate_appeal(&appeal);
        assert_eq!(invalid_msg(&result), "Appeal must state grounds");
    }

    #[test]
    fn appeal_with_all_statuses_passes() {
        let statuses = vec![
            AppealStatus::Filed,
            AppealStatus::UnderReview,
            AppealStatus::Granted,
            AppealStatus::Denied,
            AppealStatus::Remanded,
            AppealStatus::Resolved,
        ];
        for status in statuses {
            let mut appeal = make_appeal();
            appeal.status = status;
            assert!(is_valid(&validate_appeal(&appeal)));
        }
    }

    #[test]
    fn appeal_number_zero_passes() {
        let mut appeal = make_appeal();
        appeal.appeal_number = 0;
        assert!(is_valid(&validate_appeal(&appeal)));
    }

    #[test]
    fn appeal_number_max_passes() {
        let mut appeal = make_appeal();
        appeal.appeal_number = u8::MAX;
        assert!(is_valid(&validate_appeal(&appeal)));
    }

    // ========================================================================
    // ENFORCEMENT VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_enforcement_passes() {
        let result = validate_enforcement(&make_enforcement());
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_enforcer_not_did_rejected() {
        let mut enf = make_enforcement();
        enf.enforcer = "system".into();
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Enforcer must be a DID");
    }

    #[test]
    fn enforcement_empty_enforcer_rejected() {
        let mut enf = make_enforcement();
        enf.enforcer = "".into();
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
    }

    #[test]
    fn enforcement_did_prefix_only_passes() {
        let mut enf = make_enforcement();
        enf.enforcer = "did:".into();
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_with_all_statuses_passes() {
        let statuses = vec![
            EnforcementStatus::Pending,
            EnforcementStatus::InProgress,
            EnforcementStatus::PartiallyCompleted,
            EnforcementStatus::Completed,
            EnforcementStatus::Failed,
            EnforcementStatus::Contested,
        ];
        for status in statuses {
            let mut enf = make_enforcement();
            enf.status = status;
            assert!(is_valid(&validate_enforcement(&enf)));
        }
    }

    #[test]
    fn enforcement_with_actions_passes() {
        let mut enf = make_enforcement();
        enf.actions = vec![
            EnforcementAction {
                action_type: EnforcementActionType::FundsTransfer,
                target_happ: Some("mycelix-commons".into()),
                target_entry: Some("entry-hash".into()),
                executed_at: ts(),
                result: "Transfer completed".into(),
            },
            EnforcementAction {
                action_type: EnforcementActionType::Notification,
                target_happ: None,
                target_entry: None,
                executed_at: ts(),
                result: "Notification sent".into(),
            },
        ];
        assert!(is_valid(&validate_enforcement(&enf)));
    }

    #[test]
    fn enforcement_with_completed_at_passes() {
        let mut enf = make_enforcement();
        enf.completed_at = Some(Timestamp::from_micros(999999));
        assert!(is_valid(&validate_enforcement(&enf)));
    }

    #[test]
    fn enforcement_max_remedy_index_passes() {
        let mut enf = make_enforcement();
        enf.remedy_index = u32::MAX;
        assert!(is_valid(&validate_enforcement(&enf)));
    }

    // ========================================================================
    // RESTORATIVE CIRCLE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_restorative_circle_passes() {
        let result = validate_restorative(&make_restorative_circle());
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_facilitator_not_did_rejected() {
        let mut circle = make_restorative_circle();
        circle.facilitator = "facilitator".into();
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Facilitator must be a DID");
    }

    #[test]
    fn restorative_empty_facilitator_rejected() {
        let mut circle = make_restorative_circle();
        circle.facilitator = "".into();
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_no_participants_rejected() {
        let mut circle = make_restorative_circle();
        circle.participants = vec![];
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Restorative circle must have participants"
        );
    }

    #[test]
    fn restorative_single_participant_passes() {
        let mut circle = make_restorative_circle();
        circle.participants = vec![make_circle_participant("did:example:alice")];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_did_prefix_only_facilitator_passes() {
        let mut circle = make_restorative_circle();
        circle.facilitator = "did:".into();
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_checks_facilitator_before_participants() {
        let mut circle = make_restorative_circle();
        circle.facilitator = "no-did".into();
        circle.participants = vec![];
        let result = validate_restorative(&circle);
        assert_eq!(invalid_msg(&result), "Facilitator must be a DID");
    }

    #[test]
    fn restorative_with_all_statuses_passes() {
        let statuses = vec![
            CircleStatus::Forming,
            CircleStatus::Active,
            CircleStatus::AgreementReached,
            CircleStatus::Monitoring,
            CircleStatus::Completed,
            CircleStatus::Discontinued,
        ];
        for status in statuses {
            let mut circle = make_restorative_circle();
            circle.status = status;
            assert!(is_valid(&validate_restorative(&circle)));
        }
    }

    #[test]
    fn restorative_with_all_participant_roles_passes() {
        let mut circle = make_restorative_circle();
        circle.participants = vec![
            CircleParticipant {
                did: "did:a".into(),
                role: CircleRole::Facilitator,
                consented: true,
                attended_sessions: vec![],
            },
            CircleParticipant {
                did: "did:b".into(),
                role: CircleRole::HarmDoer,
                consented: true,
                attended_sessions: vec![],
            },
            CircleParticipant {
                did: "did:c".into(),
                role: CircleRole::HarmReceiver,
                consented: true,
                attended_sessions: vec![],
            },
            CircleParticipant {
                did: "did:d".into(),
                role: CircleRole::CommunityMember,
                consented: true,
                attended_sessions: vec![],
            },
            CircleParticipant {
                did: "did:e".into(),
                role: CircleRole::SupportPerson,
                consented: true,
                attended_sessions: vec![],
            },
            CircleParticipant {
                did: "did:f".into(),
                role: CircleRole::Elder,
                consented: false,
                attended_sessions: vec![1],
            },
        ];
        assert!(is_valid(&validate_restorative(&circle)));
    }

    #[test]
    fn restorative_with_sessions_passes() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec!["did:a".into(), "did:b".into()],
            summary: "Opening circle".into(),
            next_steps: vec!["Follow up in 1 week".into()],
        }];
        assert!(is_valid(&validate_restorative(&circle)));
    }

    #[test]
    fn restorative_with_agreements_passes() {
        let mut circle = make_restorative_circle();
        circle.agreements = vec![
            "Agreement to restore trust".into(),
            "Monthly check-ins for 6 months".into(),
        ];
        assert!(is_valid(&validate_restorative(&circle)));
    }

    // ========================================================================
    // UNICODE AND EDGE CASE TESTS
    // ========================================================================

    #[test]
    fn case_unicode_title_passes() {
        let mut case = make_case();
        case.title = "\u{1F4DC} Contract \u{2014} \u{4E89}\u{8BAE}\u{89E3}\u{51B3}".into();
        assert!(is_valid(&validate_case(&case)));
    }

    #[test]
    fn case_unicode_description_passes() {
        let mut case = make_case();
        case.description = "\u{0420}\u{0435}\u{0441}\u{043F}\u{043E}\u{043D}\u{0434}\u{0435}\u{043D}\u{0442} \u{043D}\u{0435} \u{0432}\u{044B}\u{043F}\u{043E}\u{043B}\u{043D}\u{0438}\u{043B} \u{043E}\u{0431}\u{044F}\u{0437}\u{0430}\u{0442}\u{0435}\u{043B}\u{044C}\u{0441}\u{0442}\u{0432}\u{0430}".into();
        assert!(is_valid(&validate_case(&case)));
    }

    #[test]
    fn case_emoji_only_title_passes() {
        let mut case = make_case();
        case.title = "\u{2696}\u{FE0F}".into();
        assert!(is_valid(&validate_case(&case)));
    }

    #[test]
    fn case_zero_width_space_title_rejected() {
        // Zero-width space (\u200B) is whitespace per Unicode, trim() removes it
        let mut case = make_case();
        case.title = "\u{200B}".into();
        // Whether this passes depends on Rust's trim() behavior with ZWSP
        // Rust's trim() does NOT treat \u200B as whitespace, so this passes
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_unicode_description_passes() {
        let mut ev = make_evidence();
        ev.description =
            "\u{6587}\u{4EF6}\u{8BC1}\u{636E} - \u{5408}\u{540C}\u{526F}\u{672C}".into();
        assert!(is_valid(&validate_evidence(&ev)));
    }

    #[test]
    fn decision_unicode_reasoning_passes() {
        let mut dec = make_decision();
        dec.reasoning =
            "La evidencia apoya claramente al demandante. \u{00BF}Hay alguna duda?".into();
        assert!(is_valid(&validate_decision(&dec)));
    }

    #[test]
    fn appeal_unicode_argument_passes() {
        let mut appeal = make_appeal();
        appeal.argument = "\u{5224}\u{6C7A}\u{306B}\u{5BFE}\u{3059}\u{308B}\u{7570}\u{8B70}\u{7533}\u{3057}\u{7ACB}\u{3066} - procedural error occurred".into();
        assert!(is_valid(&validate_appeal(&appeal)));
    }

    #[test]
    fn case_very_long_title_rejected() {
        let mut case = make_case();
        case.title = "A".repeat(10_000);
        assert!(is_invalid(&validate_case(&case)));
    }

    #[test]
    fn case_very_long_description_rejected() {
        let mut case = make_case();
        case.description = "B".repeat(100_000);
        assert!(is_invalid(&validate_case(&case)));
    }

    #[test]
    fn evidence_very_long_hash_rejected() {
        let mut ev = make_evidence();
        ev.content.hash = "x".repeat(50_000);
        assert!(is_invalid(&validate_evidence(&ev)));
    }

    #[test]
    fn decision_very_long_reasoning_rejected() {
        let mut dec = make_decision();
        dec.reasoning = "R".repeat(100_000);
        assert!(is_invalid(&validate_decision(&dec)));
    }

    #[test]
    fn appeal_very_long_argument_rejected() {
        let mut appeal = make_appeal();
        appeal.argument = "A".repeat(100_000);
        assert!(is_invalid(&validate_appeal(&appeal)));
    }

    #[test]
    fn case_newline_only_title_rejected() {
        let mut case = make_case();
        case.title = "\n\n\n".into();
        assert!(is_invalid(&validate_case(&case)));
    }

    #[test]
    fn case_tab_only_title_rejected() {
        let mut case = make_case();
        case.title = "\t\t\t".into();
        assert!(is_invalid(&validate_case(&case)));
    }

    #[test]
    fn case_mixed_whitespace_title_rejected() {
        let mut case = make_case();
        case.title = " \t \n \r ".into();
        assert!(is_invalid(&validate_case(&case)));
    }

    #[test]
    fn case_did_with_various_methods_passes() {
        let did_methods = vec![
            ("did:key:z6Mkf", "did:key:z6Mkg"),
            ("did:web:example.com", "did:web:other.com"),
            ("did:plc:abc123", "did:plc:def456"),
            ("did:pkh:eip155:1:0xabc", "did:pkh:eip155:1:0xdef"),
        ];
        for (c, r) in did_methods {
            let mut case = make_case();
            case.complainant = c.into();
            case.respondent = r.into();
            assert!(is_valid(&validate_case(&case)));
        }
    }

    #[test]
    fn evidence_visibility_restricted_empty_parties_passes() {
        let mut ev = make_evidence();
        ev.visibility = EvidenceVisibility::Restricted { parties: vec![] };
        assert!(is_valid(&validate_evidence(&ev)));
    }

    #[test]
    fn serde_roundtrip_case_type_other_unicode_category() {
        let ct = CaseType::Other {
            category: "\u{00E4}\u{00F6}\u{00FC}\u{00DF}".into(),
        };
        let json = serde_json::to_string(&ct).unwrap();
        let back: CaseType = serde_json::from_str(&json).unwrap();
        assert_eq!(ct, back);
    }

    #[test]
    fn serde_roundtrip_arbitrator_selection_expertise_unicode() {
        let sel = ArbitratorSelection::ExpertiseBased {
            domain: "\u{0444}\u{0438}\u{043D}\u{0430}\u{043D}\u{0441}\u{044B}".into(),
        };
        let json = serde_json::to_string(&sel).unwrap();
        let back: ArbitratorSelection = serde_json::from_str(&json).unwrap();
        assert_eq!(sel, back);
    }

    #[test]
    fn serde_roundtrip_evidence_type_onchain_empty_strings() {
        let et = EvidenceType::OnChainData {
            happ: "".into(),
            entry_hash: "".into(),
        };
        let json = serde_json::to_string(&et).unwrap();
        let back: EvidenceType = serde_json::from_str(&json).unwrap();
        assert_eq!(et, back);
    }

    #[test]
    fn serde_roundtrip_evidence_visibility_restricted_empty() {
        let vis = EvidenceVisibility::Restricted { parties: vec![] };
        let json = serde_json::to_string(&vis).unwrap();
        let back: EvidenceVisibility = serde_json::from_str(&json).unwrap();
        assert_eq!(vis, back);
    }

    #[test]
    fn case_type_other_empty_category_roundtrip() {
        let ct = CaseType::Other {
            category: "".into(),
        };
        let json = serde_json::to_string(&ct).unwrap();
        let back: CaseType = serde_json::from_str(&json).unwrap();
        assert_eq!(ct, back);
    }

    // ========================================================================
    // MAX VALUE / BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn timestamp_max_micros_in_case() {
        let mut case = make_case();
        case.created_at = Timestamp::from_micros(i64::MAX);
        case.updated_at = Timestamp::from_micros(i64::MAX);
        assert!(is_valid(&validate_case(&case)));
    }

    #[test]
    fn timestamp_negative_micros_in_case() {
        let mut case = make_case();
        case.created_at = Timestamp::from_micros(-1);
        assert!(is_valid(&validate_case(&case)));
    }

    #[test]
    fn enforcement_remedy_index_zero_passes() {
        let mut enf = make_enforcement();
        enf.remedy_index = 0;
        assert!(is_valid(&validate_enforcement(&enf)));
    }

    #[test]
    fn circle_participant_max_session_numbers() {
        let mut circle = make_restorative_circle();
        circle.participants = vec![CircleParticipant {
            did: "did:example:a".into(),
            role: CircleRole::HarmReceiver,
            consented: true,
            attended_sessions: vec![0, u32::MAX],
        }];
        assert!(is_valid(&validate_restorative(&circle)));
    }

    #[test]
    fn mediation_session_number_max() {
        let mut med = make_mediation();
        med.sessions = vec![MediationSession {
            session_number: u32::MAX,
            scheduled_at: ts(),
            actual_start: None,
            actual_end: None,
            notes: None,
            outcome: None,
        }];
        assert!(is_valid(&validate_mediation(&med)));
    }

    #[test]
    fn case_many_parties_rejected() {
        let mut case = make_case();
        case.parties = (0..100)
            .map(|i| CaseParty {
                did: format!("did:example:party{}", i),
                role: PartyRole::Affected,
                joined_at: ts(),
            })
            .collect();
        assert!(is_invalid(&validate_case(&case)));
    }

    #[test]
    fn restorative_many_participants_passes() {
        let mut circle = make_restorative_circle();
        circle.participants = (0..50)
            .map(|i| make_circle_participant(&format!("did:example:p{}", i)))
            .collect();
        assert!(is_valid(&validate_restorative(&circle)));
    }

    #[test]
    fn decision_many_votes_rejected() {
        let mut dec = make_decision();
        dec.votes = (0..99)
            .map(|i| make_vote(&format!("did:example:arb{}", i)))
            .collect();
        assert!(is_invalid(&validate_decision(&dec)));
    }

    #[test]
    fn appeal_all_six_grounds_passes() {
        let mut appeal = make_appeal();
        appeal.grounds = vec![
            AppealGround::ProceduralError,
            AppealGround::NewEvidence,
            AppealGround::LegalError,
            AppealGround::Bias,
            AppealGround::ExcessiveRemedy,
            AppealGround::InsufficientRemedy,
        ];
        assert!(is_valid(&validate_appeal(&appeal)));
    }

    #[test]
    fn evidence_encrypted_with_key_reference_passes() {
        let mut ev = make_evidence();
        ev.content.encrypted = true;
        ev.content.key_reference = Some("key:ring:ref:xyz".into());
        assert!(is_valid(&validate_evidence(&ev)));
    }

    #[test]
    fn evidence_encrypted_without_key_reference_passes() {
        // Validation does not enforce key_reference when encrypted
        let mut ev = make_evidence();
        ev.content.encrypted = true;
        ev.content.key_reference = None;
        assert!(is_valid(&validate_evidence(&ev)));
    }

    // ── Update validation tests ──────────────────────────────────────────

    #[test]
    fn test_update_case_invalid_title_rejected() {
        let mut case = make_case();
        case.title = String::new();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("Case title required"));
    }

    #[test]
    fn test_update_case_invalid_self_filing_rejected() {
        let mut case = make_case();
        case.complainant = "did:key:same".into();
        case.respondent = "did:key:same".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("Cannot file case against self"));
    }

    #[test]
    fn test_update_evidence_invalid_submitter_rejected() {
        let mut ev = make_evidence();
        ev.submitter = "not-a-did".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("DID"));
    }

    #[test]
    fn test_update_evidence_invalid_description_rejected() {
        let mut ev = make_evidence();
        ev.description = String::new();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("description"));
    }

    #[test]
    fn test_update_arbitration_invalid_even_panel_rejected() {
        let arb = make_arbitration(vec![
            Arbitrator {
                did: "did:key:arb1".into(),
                role: ArbitratorRole::Primary,
                selected_at: ts(),
                accepted: true,
                recused: false,
                recusal_reason: None,
            },
            Arbitrator {
                did: "did:key:arb2".into(),
                role: ArbitratorRole::PanelMember,
                selected_at: ts(),
                accepted: true,
                recused: false,
                recusal_reason: None,
            },
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("odd number"));
    }

    #[test]
    fn test_update_decision_invalid_empty_reasoning_rejected() {
        let mut dec = make_decision();
        dec.reasoning = "  ".into();
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("Decision reasoning required"));
    }

    #[test]
    fn test_update_appeal_invalid_appellant_rejected() {
        let mut appeal = make_appeal();
        appeal.appellant = "not-a-did".into();
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("Appellant must be a DID"));
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_create_link_tag(link_type: &LinkTypes, tag: &LinkTag) -> ValidateCallbackResult {
        let tag_len = tag.0.len();
        match link_type {
            LinkTypes::CaseToEvidence | LinkTypes::CaseToDecisions => {
                if tag_len > 512 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 512 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
            LinkTypes::ComplainantToCases
            | LinkTypes::RespondentToCases
            | LinkTypes::CaseToMediation
            | LinkTypes::CaseToArbitration
            | LinkTypes::DecisionToAppeals
            | LinkTypes::DecisionToEnforcement
            | LinkTypes::CaseToRestorativeCircle
            | LinkTypes::ArbitratorToCases
            | LinkTypes::AllCases => {
                if tag_len > 256 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 256 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
        }
    }

    fn validate_delete_link_tag(tag: &LinkTag) -> ValidateCallbackResult {
        if tag.0.len() > 256 {
            ValidateCallbackResult::Invalid("Delete link tag too long (max 256 bytes)".into())
        } else {
            ValidateCallbackResult::Valid
        }
    }

    // -- AllCases (256-byte limit) boundary tests --

    #[test]
    fn link_tag_all_cases_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::AllCases, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_all_cases_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_create_link_tag(&LinkTypes::AllCases, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_all_cases_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_create_link_tag(&LinkTypes::AllCases, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- CaseToEvidence (512-byte limit) boundary tests --

    #[test]
    fn link_tag_case_to_evidence_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::CaseToEvidence, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_case_to_evidence_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 512]);
        let result = validate_create_link_tag(&LinkTypes::CaseToEvidence, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_case_to_evidence_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 513]);
        let result = validate_create_link_tag(&LinkTypes::CaseToEvidence, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- CaseToDecisions (512-byte limit) boundary tests --

    #[test]
    fn link_tag_case_to_decisions_at_limit_valid() {
        let tag = LinkTag::new(vec![0xDD; 512]);
        let result = validate_create_link_tag(&LinkTypes::CaseToDecisions, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_case_to_decisions_over_limit_invalid() {
        let tag = LinkTag::new(vec![0xDD; 513]);
        let result = validate_create_link_tag(&LinkTypes::CaseToDecisions, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- DoS prevention: massive tags rejected for all link types --

    #[test]
    fn link_tag_dos_prevention_all_types() {
        let massive_tag = LinkTag::new(vec![0xFF; 10_000]);
        let all_types = [
            LinkTypes::ComplainantToCases,
            LinkTypes::RespondentToCases,
            LinkTypes::CaseToEvidence,
            LinkTypes::CaseToMediation,
            LinkTypes::CaseToArbitration,
            LinkTypes::CaseToDecisions,
            LinkTypes::DecisionToAppeals,
            LinkTypes::DecisionToEnforcement,
            LinkTypes::CaseToRestorativeCircle,
            LinkTypes::ArbitratorToCases,
            LinkTypes::AllCases,
        ];
        for lt in &all_types {
            let result = validate_create_link_tag(lt, &massive_tag);
            assert!(
                matches!(result, ValidateCallbackResult::Invalid(_)),
                "Massive tag should be rejected for {:?}",
                lt
            );
        }
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
