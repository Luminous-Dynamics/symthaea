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
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            let tag_len = tag.0.len();
            match link_type {
                LinkTypes::ComplainantToCases => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::RespondentToCases => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::CaseToEvidence => {
                    // Evidence links may carry content hash metadata in tags
                    if tag_len > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 512 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::CaseToMediation => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::CaseToArbitration => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::CaseToDecisions => {
                    // Decision links may carry ruling metadata in tags
                    if tag_len > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 512 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::DecisionToAppeals => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::DecisionToEnforcement => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::CaseToRestorativeCircle => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ArbitratorToCases => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::AllCases => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink {
            link_type,
            tag,
            action,
            ..
        } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            let tag_len = tag.0.len();
            // All delete link tags get the same 256-byte limit
            if tag_len > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
                ));
            }
            let _ = link_type;
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            let original_author = original_action.action().author().clone();
            if action.author != original_author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this entry".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
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
    }

    #[test]
    fn case_whitespace_only_title_rejected() {
        let mut case = make_case();
        case.title = "   \t\n  ".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_empty_description_rejected() {
        let mut case = make_case();
        case.description = "".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_complainant_not_did_rejected() {
        let mut case = make_case();
        case.complainant = "alice".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_respondent_not_did_rejected() {
        let mut case = make_case();
        case.respondent = "bob".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_against_self_rejected() {
        let mut case = make_case();
        case.complainant = "did:example:same".into();
        case.respondent = "did:example:same".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    // -- Case string length limits --

    #[test]
    fn case_id_too_long_rejected() {
        let mut case = make_case();
        case.id = "x".repeat(513);
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_id_at_limit_accepted() {
        let mut case = make_case();
        case.id = "x".repeat(512);
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_title_too_long_rejected() {
        let mut case = make_case();
        case.title = "x".repeat(513);
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_title_at_limit_accepted() {
        let mut case = make_case();
        case.title = "x".repeat(512);
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_description_too_long_rejected() {
        let mut case = make_case();
        case.description = "x".repeat(8193);
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_description_at_limit_accepted() {
        let mut case = make_case();
        case.description = "x".repeat(8192);
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_complainant_did_too_long_rejected() {
        let mut case = make_case();
        case.complainant = format!("did:{}", "x".repeat(253));
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_complainant_did_at_limit_accepted() {
        let mut case = make_case();
        case.complainant = format!("did:{}", "x".repeat(252));
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_respondent_did_too_long_rejected() {
        let mut case = make_case();
        case.respondent = format!("did:{}", "x".repeat(253));
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_respondent_did_at_limit_accepted() {
        let mut case = make_case();
        case.respondent = format!("did:{}", "y".repeat(252));
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    // -- Case Vec length limits --

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
    }

    #[test]
    fn case_parties_at_limit_accepted() {
        let mut case = make_case();
        case.parties = (0..20)
            .map(|i| CaseParty {
                did: format!("did:example:party{}", i),
                role: PartyRole::Witness,
                joined_at: ts(),
            })
            .collect();
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_party_did_too_long_rejected() {
        let mut case = make_case();
        case.parties = vec![CaseParty {
            did: "x".repeat(257),
            role: PartyRole::Witness,
            joined_at: ts(),
        }];
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_party_did_at_limit_accepted() {
        let mut case = make_case();
        case.parties = vec![CaseParty {
            did: "x".repeat(256),
            role: PartyRole::Witness,
            joined_at: ts(),
        }];
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    // -- Case context string limits --

    #[test]
    fn case_context_happ_too_long_rejected() {
        let mut case = make_case();
        case.context.happ = Some("x".repeat(257));
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_context_happ_at_limit_accepted() {
        let mut case = make_case();
        case.context.happ = Some("x".repeat(256));
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_context_reference_id_too_long_rejected() {
        let mut case = make_case();
        case.context.reference_id = Some("x".repeat(257));
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_context_reference_id_at_limit_accepted() {
        let mut case = make_case();
        case.context.reference_id = Some("x".repeat(256));
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_context_community_too_long_rejected() {
        let mut case = make_case();
        case.context.community = Some("x".repeat(257));
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_context_community_at_limit_accepted() {
        let mut case = make_case();
        case.context.community = Some("x".repeat(256));
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_context_jurisdiction_too_long_rejected() {
        let mut case = make_case();
        case.context.jurisdiction = Some("x".repeat(4097));
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn case_context_jurisdiction_at_limit_accepted() {
        let mut case = make_case();
        case.context.jurisdiction = Some("x".repeat(4096));
        let result = validate_case(&case);
        assert!(is_valid(&result));
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
    }

    #[test]
    fn evidence_empty_description_rejected() {
        let mut ev = make_evidence();
        ev.description = "".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_empty_content_hash_rejected() {
        let mut ev = make_evidence();
        ev.content.hash = "".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    // -- Evidence string length limits --

    #[test]
    fn evidence_id_too_long_rejected() {
        let mut ev = make_evidence();
        ev.id = "x".repeat(257);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_id_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.id = "x".repeat(256);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_case_id_too_long_rejected() {
        let mut ev = make_evidence();
        ev.case_id = "x".repeat(513);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_case_id_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.case_id = "x".repeat(512);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_submitter_too_long_rejected() {
        let mut ev = make_evidence();
        ev.submitter = format!("did:{}", "x".repeat(253));
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_submitter_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.submitter = format!("did:{}", "x".repeat(252));
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_description_too_long_rejected() {
        let mut ev = make_evidence();
        ev.description = "x".repeat(4097);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_description_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.description = "x".repeat(4096);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_content_hash_too_long_rejected() {
        let mut ev = make_evidence();
        ev.content.hash = "x".repeat(257);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_content_hash_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.content.hash = "x".repeat(256);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_content_reference_too_long_rejected() {
        let mut ev = make_evidence();
        ev.content.reference = "x".repeat(4097);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_content_reference_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.content.reference = "x".repeat(4096);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_content_mime_type_too_long_rejected() {
        let mut ev = make_evidence();
        ev.content.mime_type = "x".repeat(257);
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_content_mime_type_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.content.mime_type = "x".repeat(256);
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_key_reference_too_long_rejected() {
        let mut ev = make_evidence();
        ev.content.key_reference = Some("x".repeat(257));
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_key_reference_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.content.key_reference = Some("x".repeat(256));
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    // -- Evidence Vec length limits --

    #[test]
    fn evidence_too_many_custody_events_rejected() {
        let mut ev = make_evidence();
        ev.custody = (0..201)
            .map(|_| CustodyEvent {
                action: CustodyAction::Accessed,
                actor: "did:example:actor".into(),
                timestamp: ts(),
                notes: None,
            })
            .collect();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_custody_events_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.custody = (0..200)
            .map(|_| CustodyEvent {
                action: CustodyAction::Accessed,
                actor: "did:example:actor".into(),
                timestamp: ts(),
                notes: None,
            })
            .collect();
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_custody_actor_too_long_rejected() {
        let mut ev = make_evidence();
        ev.custody = vec![CustodyEvent {
            action: CustodyAction::Submitted,
            actor: "x".repeat(257),
            timestamp: ts(),
            notes: None,
        }];
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_custody_actor_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.custody = vec![CustodyEvent {
            action: CustodyAction::Submitted,
            actor: "x".repeat(256),
            timestamp: ts(),
            notes: None,
        }];
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_custody_notes_too_long_rejected() {
        let mut ev = make_evidence();
        ev.custody = vec![CustodyEvent {
            action: CustodyAction::Submitted,
            actor: "did:example:actor".into(),
            timestamp: ts(),
            notes: Some("x".repeat(4097)),
        }];
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_custody_notes_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.custody = vec![CustodyEvent {
            action: CustodyAction::Submitted,
            actor: "did:example:actor".into(),
            timestamp: ts(),
            notes: Some("x".repeat(4096)),
        }];
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    // -- Evidence verification string limits --

    #[test]
    fn evidence_verification_verifier_too_long_rejected() {
        let mut ev = make_evidence();
        ev.verification.verifier = Some("x".repeat(257));
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_verification_verifier_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.verification.verifier = Some("x".repeat(256));
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_verification_method_too_long_rejected() {
        let mut ev = make_evidence();
        ev.verification.method = Some("x".repeat(4097));
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_verification_method_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.verification.method = Some("x".repeat(4096));
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_verification_notes_too_long_rejected() {
        let mut ev = make_evidence();
        ev.verification.notes = Some("x".repeat(4097));
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn evidence_verification_notes_at_limit_accepted() {
        let mut ev = make_evidence();
        ev.verification.notes = Some("x".repeat(4096));
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
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
        med.mediator = "mediator-person".into();
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    // -- Mediation string length limits --

    #[test]
    fn mediation_id_too_long_rejected() {
        let mut med = make_mediation();
        med.id = "x".repeat(257);
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_id_at_limit_accepted() {
        let mut med = make_mediation();
        med.id = "x".repeat(256);
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_case_id_too_long_rejected() {
        let mut med = make_mediation();
        med.case_id = "x".repeat(513);
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_case_id_at_limit_accepted() {
        let mut med = make_mediation();
        med.case_id = "x".repeat(512);
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_mediator_did_too_long_rejected() {
        let mut med = make_mediation();
        med.mediator = format!("did:{}", "x".repeat(253));
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_mediator_did_at_limit_accepted() {
        let mut med = make_mediation();
        med.mediator = format!("did:{}", "x".repeat(252));
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    // -- Mediation Vec length limits --

    #[test]
    fn mediation_too_many_sessions_rejected() {
        let mut med = make_mediation();
        med.sessions = (0..51)
            .map(|i| MediationSession {
                session_number: i,
                scheduled_at: ts(),
                actual_start: None,
                actual_end: None,
                notes: None,
                outcome: None,
            })
            .collect();
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_sessions_at_limit_accepted() {
        let mut med = make_mediation();
        med.sessions = (0..50)
            .map(|i| MediationSession {
                session_number: i,
                scheduled_at: ts(),
                actual_start: None,
                actual_end: None,
                notes: None,
                outcome: None,
            })
            .collect();
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_too_many_proposals_rejected() {
        let mut med = make_mediation();
        med.proposals = (0..11).map(|i| format!("settlement-{}", i)).collect();
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_proposals_at_limit_accepted() {
        let mut med = make_mediation();
        med.proposals = (0..10).map(|i| format!("settlement-{}", i)).collect();
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_session_notes_too_long_rejected() {
        let mut med = make_mediation();
        med.sessions = vec![MediationSession {
            session_number: 1,
            scheduled_at: ts(),
            actual_start: None,
            actual_end: None,
            notes: Some("x".repeat(4097)),
            outcome: None,
        }];
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_session_notes_at_limit_accepted() {
        let mut med = make_mediation();
        med.sessions = vec![MediationSession {
            session_number: 1,
            scheduled_at: ts(),
            actual_start: None,
            actual_end: None,
            notes: Some("x".repeat(4096)),
            outcome: None,
        }];
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_session_outcome_too_long_rejected() {
        let mut med = make_mediation();
        med.sessions = vec![MediationSession {
            session_number: 1,
            scheduled_at: ts(),
            actual_start: None,
            actual_end: None,
            notes: None,
            outcome: Some("x".repeat(4097)),
        }];
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_session_outcome_at_limit_accepted() {
        let mut med = make_mediation();
        med.sessions = vec![MediationSession {
            session_number: 1,
            scheduled_at: ts(),
            actual_start: None,
            actual_end: None,
            notes: None,
            outcome: Some("x".repeat(4096)),
        }];
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_proposal_id_too_long_rejected() {
        let mut med = make_mediation();
        med.proposals = vec!["x".repeat(257)];
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_proposal_id_at_limit_accepted() {
        let mut med = make_mediation();
        med.proposals = vec!["x".repeat(256)];
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // ARBITRATION VALIDATION TESTS
    // ========================================================================

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
    fn arbitration_even_number_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
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
    }

    // -- Arbitration string length limits --

    #[test]
    fn arbitration_id_too_long_rejected() {
        let mut arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        arb.id = "x".repeat(257);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_id_at_limit_accepted() {
        let mut arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        arb.id = "x".repeat(256);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitration_case_id_too_long_rejected() {
        let mut arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        arb.case_id = "x".repeat(513);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_case_id_at_limit_accepted() {
        let mut arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        arb.case_id = "x".repeat(512);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    // -- Arbitration Vec length limits --

    #[test]
    fn arbitration_too_many_arbitrators_rejected() {
        let arb = make_arbitration(
            (0..11)
                .map(|i| make_arbitrator(&format!("did:example:arb{}", i)))
                .collect(),
        );
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_arbitrators_at_limit_accepted() {
        let arb = make_arbitration(
            (0..9)
                .map(|i| make_arbitrator(&format!("did:example:arb{}", i)))
                .collect(),
        );
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitration_arbitrator_did_too_long_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator(&format!("did:{}", "x".repeat(253))),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_arbitrator_did_at_limit_accepted() {
        let arb = make_arbitration(vec![
            make_arbitrator(&format!("did:{}", "x".repeat(252))),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitration_recusal_reason_too_long_rejected() {
        let mut arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        arb.arbitrators[0].recusal_reason = Some("x".repeat(4097));
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_recusal_reason_at_limit_accepted() {
        let mut arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        arb.arbitrators[0].recusal_reason = Some("x".repeat(4096));
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
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
    }

    #[test]
    fn decision_no_votes_rejected() {
        let mut dec = make_decision();
        dec.votes = vec![];
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    // -- Decision string length limits --

    #[test]
    fn decision_id_too_long_rejected() {
        let mut dec = make_decision();
        dec.id = "x".repeat(257);
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_id_at_limit_accepted() {
        let mut dec = make_decision();
        dec.id = "x".repeat(256);
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_case_id_too_long_rejected() {
        let mut dec = make_decision();
        dec.case_id = "x".repeat(513);
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_case_id_at_limit_accepted() {
        let mut dec = make_decision();
        dec.case_id = "x".repeat(512);
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_arbitration_id_too_long_rejected() {
        let mut dec = make_decision();
        dec.arbitration_id = "x".repeat(257);
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_arbitration_id_at_limit_accepted() {
        let mut dec = make_decision();
        dec.arbitration_id = "x".repeat(256);
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_reasoning_too_long_rejected() {
        let mut dec = make_decision();
        dec.reasoning = "x".repeat(16385);
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_reasoning_at_limit_accepted() {
        let mut dec = make_decision();
        dec.reasoning = "x".repeat(16384);
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    // -- Decision Vec length limits --

    #[test]
    fn decision_too_many_remedies_rejected() {
        let mut dec = make_decision();
        dec.remedies = (0..21)
            .map(|_| Remedy {
                remedy_type: RemedyType::Compensation,
                responsible_party: "did:example:party".into(),
                deadline: None,
                amount: Some(100),
                currency: Some("USD".into()),
                description: "Pay up".into(),
            })
            .collect();
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_remedies_at_limit_accepted() {
        let mut dec = make_decision();
        dec.remedies = (0..20)
            .map(|_| Remedy {
                remedy_type: RemedyType::Compensation,
                responsible_party: "did:example:party".into(),
                deadline: None,
                amount: Some(100),
                currency: Some("USD".into()),
                description: "Pay up".into(),
            })
            .collect();
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_too_many_votes_rejected() {
        let mut dec = make_decision();
        dec.votes = (0..10)
            .map(|i| make_vote(&format!("did:example:arb{}", i)))
            .collect();
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_votes_at_limit_accepted() {
        let mut dec = make_decision();
        dec.votes = (0..9)
            .map(|i| make_vote(&format!("did:example:arb{}", i)))
            .collect();
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_too_many_dissents_rejected() {
        let mut dec = make_decision();
        dec.dissents = (0..10)
            .map(|i| DissentingOpinion {
                arbitrator: format!("did:example:arb{}", i),
                opinion: "I disagree".into(),
                timestamp: ts(),
            })
            .collect();
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_dissents_at_limit_accepted() {
        let mut dec = make_decision();
        dec.dissents = (0..9)
            .map(|i| DissentingOpinion {
                arbitrator: format!("did:example:arb{}", i),
                opinion: "I disagree".into(),
                timestamp: ts(),
            })
            .collect();
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    // -- Decision nested string limits --

    #[test]
    fn decision_remedy_responsible_party_too_long_rejected() {
        let mut dec = make_decision();
        dec.remedies = vec![Remedy {
            remedy_type: RemedyType::Compensation,
            responsible_party: "x".repeat(257),
            deadline: None,
            amount: None,
            currency: None,
            description: "Test".into(),
        }];
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_remedy_responsible_party_at_limit_accepted() {
        let mut dec = make_decision();
        dec.remedies = vec![Remedy {
            remedy_type: RemedyType::Compensation,
            responsible_party: "x".repeat(256),
            deadline: None,
            amount: None,
            currency: None,
            description: "Test".into(),
        }];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_remedy_description_too_long_rejected() {
        let mut dec = make_decision();
        dec.remedies = vec![Remedy {
            remedy_type: RemedyType::Compensation,
            responsible_party: "did:example:party".into(),
            deadline: None,
            amount: None,
            currency: None,
            description: "x".repeat(4097),
        }];
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_remedy_description_at_limit_accepted() {
        let mut dec = make_decision();
        dec.remedies = vec![Remedy {
            remedy_type: RemedyType::Compensation,
            responsible_party: "did:example:party".into(),
            deadline: None,
            amount: None,
            currency: None,
            description: "x".repeat(4096),
        }];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_remedy_currency_too_long_rejected() {
        let mut dec = make_decision();
        dec.remedies = vec![Remedy {
            remedy_type: RemedyType::Compensation,
            responsible_party: "did:example:party".into(),
            deadline: None,
            amount: Some(100),
            currency: Some("x".repeat(257)),
            description: "Test".into(),
        }];
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_remedy_currency_at_limit_accepted() {
        let mut dec = make_decision();
        dec.remedies = vec![Remedy {
            remedy_type: RemedyType::Compensation,
            responsible_party: "did:example:party".into(),
            deadline: None,
            amount: Some(100),
            currency: Some("x".repeat(256)),
            description: "Test".into(),
        }];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_vote_arbitrator_too_long_rejected() {
        let mut dec = make_decision();
        dec.votes = vec![ArbitratorVote {
            arbitrator: "x".repeat(257),
            vote: VoteChoice::ForComplainant,
            timestamp: ts(),
        }];
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_vote_arbitrator_at_limit_accepted() {
        let mut dec = make_decision();
        dec.votes = vec![ArbitratorVote {
            arbitrator: "x".repeat(256),
            vote: VoteChoice::ForComplainant,
            timestamp: ts(),
        }];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_dissent_arbitrator_too_long_rejected() {
        let mut dec = make_decision();
        dec.dissents = vec![DissentingOpinion {
            arbitrator: "x".repeat(257),
            opinion: "I disagree".into(),
            timestamp: ts(),
        }];
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_dissent_arbitrator_at_limit_accepted() {
        let mut dec = make_decision();
        dec.dissents = vec![DissentingOpinion {
            arbitrator: "x".repeat(256),
            opinion: "I disagree".into(),
            timestamp: ts(),
        }];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_dissent_opinion_too_long_rejected() {
        let mut dec = make_decision();
        dec.dissents = vec![DissentingOpinion {
            arbitrator: "did:example:arb1".into(),
            opinion: "x".repeat(16385),
            timestamp: ts(),
        }];
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
    }

    #[test]
    fn decision_dissent_opinion_at_limit_accepted() {
        let mut dec = make_decision();
        dec.dissents = vec![DissentingOpinion {
            arbitrator: "did:example:arb1".into(),
            opinion: "x".repeat(16384),
            timestamp: ts(),
        }];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
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
    }

    #[test]
    fn appeal_no_grounds_rejected() {
        let mut appeal = make_appeal();
        appeal.grounds = vec![];
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_empty_argument_rejected() {
        let mut appeal = make_appeal();
        appeal.argument = "".into();
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    // -- Appeal string length limits --

    #[test]
    fn appeal_id_too_long_rejected() {
        let mut appeal = make_appeal();
        appeal.id = "x".repeat(257);
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_id_at_limit_accepted() {
        let mut appeal = make_appeal();
        appeal.id = "x".repeat(256);
        let result = validate_appeal(&appeal);
        assert!(is_valid(&result));
    }

    #[test]
    fn appeal_case_id_too_long_rejected() {
        let mut appeal = make_appeal();
        appeal.case_id = "x".repeat(513);
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_case_id_at_limit_accepted() {
        let mut appeal = make_appeal();
        appeal.case_id = "x".repeat(512);
        let result = validate_appeal(&appeal);
        assert!(is_valid(&result));
    }

    #[test]
    fn appeal_decision_id_too_long_rejected() {
        let mut appeal = make_appeal();
        appeal.decision_id = "x".repeat(257);
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_decision_id_at_limit_accepted() {
        let mut appeal = make_appeal();
        appeal.decision_id = "x".repeat(256);
        let result = validate_appeal(&appeal);
        assert!(is_valid(&result));
    }

    #[test]
    fn appeal_appellant_did_too_long_rejected() {
        let mut appeal = make_appeal();
        appeal.appellant = format!("did:{}", "x".repeat(253));
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_appellant_did_at_limit_accepted() {
        let mut appeal = make_appeal();
        appeal.appellant = format!("did:{}", "x".repeat(252));
        let result = validate_appeal(&appeal);
        assert!(is_valid(&result));
    }

    #[test]
    fn appeal_argument_too_long_rejected() {
        let mut appeal = make_appeal();
        appeal.argument = "x".repeat(16385);
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_argument_at_limit_accepted() {
        let mut appeal = make_appeal();
        appeal.argument = "x".repeat(16384);
        let result = validate_appeal(&appeal);
        assert!(is_valid(&result));
    }

    // -- Appeal Vec length limits --

    #[test]
    fn appeal_too_many_grounds_rejected() {
        let mut appeal = make_appeal();
        appeal.grounds = (0..11).map(|_| AppealGround::ProceduralError).collect();
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_grounds_at_limit_accepted() {
        let mut appeal = make_appeal();
        appeal.grounds = (0..10).map(|_| AppealGround::ProceduralError).collect();
        let result = validate_appeal(&appeal);
        assert!(is_valid(&result));
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
    }

    // -- Enforcement string length limits --

    #[test]
    fn enforcement_id_too_long_rejected() {
        let mut enf = make_enforcement();
        enf.id = "x".repeat(257);
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
    }

    #[test]
    fn enforcement_id_at_limit_accepted() {
        let mut enf = make_enforcement();
        enf.id = "x".repeat(256);
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_decision_id_too_long_rejected() {
        let mut enf = make_enforcement();
        enf.decision_id = "x".repeat(257);
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
    }

    #[test]
    fn enforcement_decision_id_at_limit_accepted() {
        let mut enf = make_enforcement();
        enf.decision_id = "x".repeat(256);
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_enforcer_did_too_long_rejected() {
        let mut enf = make_enforcement();
        enf.enforcer = format!("did:{}", "x".repeat(253));
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
    }

    #[test]
    fn enforcement_enforcer_did_at_limit_accepted() {
        let mut enf = make_enforcement();
        enf.enforcer = format!("did:{}", "x".repeat(252));
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    // -- Enforcement Vec length limits --

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
    }

    #[test]
    fn enforcement_actions_at_limit_accepted() {
        let mut enf = make_enforcement();
        enf.actions = (0..100)
            .map(|_| EnforcementAction {
                action_type: EnforcementActionType::Notification,
                target_happ: None,
                target_entry: None,
                executed_at: ts(),
                result: "done".into(),
            })
            .collect();
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_action_result_too_long_rejected() {
        let mut enf = make_enforcement();
        enf.actions = vec![EnforcementAction {
            action_type: EnforcementActionType::Notification,
            target_happ: None,
            target_entry: None,
            executed_at: ts(),
            result: "x".repeat(4097),
        }];
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
    }

    #[test]
    fn enforcement_action_result_at_limit_accepted() {
        let mut enf = make_enforcement();
        enf.actions = vec![EnforcementAction {
            action_type: EnforcementActionType::Notification,
            target_happ: None,
            target_entry: None,
            executed_at: ts(),
            result: "x".repeat(4096),
        }];
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_action_target_happ_too_long_rejected() {
        let mut enf = make_enforcement();
        enf.actions = vec![EnforcementAction {
            action_type: EnforcementActionType::CrossHappAction,
            target_happ: Some("x".repeat(257)),
            target_entry: None,
            executed_at: ts(),
            result: "done".into(),
        }];
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
    }

    #[test]
    fn enforcement_action_target_happ_at_limit_accepted() {
        let mut enf = make_enforcement();
        enf.actions = vec![EnforcementAction {
            action_type: EnforcementActionType::CrossHappAction,
            target_happ: Some("x".repeat(256)),
            target_entry: None,
            executed_at: ts(),
            result: "done".into(),
        }];
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_action_target_entry_too_long_rejected() {
        let mut enf = make_enforcement();
        enf.actions = vec![EnforcementAction {
            action_type: EnforcementActionType::CrossHappAction,
            target_happ: None,
            target_entry: Some("x".repeat(257)),
            executed_at: ts(),
            result: "done".into(),
        }];
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
    }

    #[test]
    fn enforcement_action_target_entry_at_limit_accepted() {
        let mut enf = make_enforcement();
        enf.actions = vec![EnforcementAction {
            action_type: EnforcementActionType::CrossHappAction,
            target_happ: None,
            target_entry: Some("x".repeat(256)),
            executed_at: ts(),
            result: "done".into(),
        }];
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
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
    fn restorative_no_participants_rejected() {
        let mut circle = make_restorative_circle();
        circle.participants = vec![];
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_facilitator_not_did_rejected() {
        let mut circle = make_restorative_circle();
        circle.facilitator = "facilitator".into();
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    // -- RestorativeCircle string length limits --

    #[test]
    fn restorative_id_too_long_rejected() {
        let mut circle = make_restorative_circle();
        circle.id = "x".repeat(257);
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_id_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.id = "x".repeat(256);
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_case_id_too_long_rejected() {
        let mut circle = make_restorative_circle();
        circle.case_id = "x".repeat(513);
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_case_id_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.case_id = "x".repeat(512);
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_facilitator_did_too_long_rejected() {
        let mut circle = make_restorative_circle();
        circle.facilitator = format!("did:{}", "x".repeat(253));
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_facilitator_did_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.facilitator = format!("did:{}", "x".repeat(252));
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    // -- RestorativeCircle Vec length limits --

    #[test]
    fn restorative_too_many_participants_rejected() {
        let mut circle = make_restorative_circle();
        circle.participants = (0..51)
            .map(|i| make_circle_participant(&format!("did:example:p{}", i)))
            .collect();
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_participants_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.participants = (0..50)
            .map(|i| make_circle_participant(&format!("did:example:p{}", i)))
            .collect();
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_too_many_agreements_rejected() {
        let mut circle = make_restorative_circle();
        circle.agreements = (0..21).map(|i| format!("Agreement {}", i)).collect();
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_agreements_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.agreements = (0..20).map(|i| format!("Agreement {}", i)).collect();
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_too_many_sessions_rejected() {
        let mut circle = make_restorative_circle();
        circle.sessions = (0..51)
            .map(|i| CircleSession {
                session_number: i,
                held_at: ts(),
                attendees: vec![],
                summary: "Summary".into(),
                next_steps: vec![],
            })
            .collect();
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_sessions_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.sessions = (0..50)
            .map(|i| CircleSession {
                session_number: i,
                held_at: ts(),
                attendees: vec![],
                summary: "Summary".into(),
                next_steps: vec![],
            })
            .collect();
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    // -- RestorativeCircle nested string/vec limits --

    #[test]
    fn restorative_participant_did_too_long_rejected() {
        let mut circle = make_restorative_circle();
        circle.participants = vec![CircleParticipant {
            did: "x".repeat(257),
            role: CircleRole::CommunityMember,
            consented: true,
            attended_sessions: vec![],
        }];
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_participant_did_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.participants = vec![CircleParticipant {
            did: "x".repeat(256),
            role: CircleRole::CommunityMember,
            consented: true,
            attended_sessions: vec![],
        }];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_participant_too_many_attended_sessions_rejected() {
        let mut circle = make_restorative_circle();
        circle.participants[0].attended_sessions = (0..101).collect();
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_participant_attended_sessions_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.participants[0].attended_sessions = (0..100).collect();
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_agreement_too_long_rejected() {
        let mut circle = make_restorative_circle();
        circle.agreements = vec!["x".repeat(4097)];
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_agreement_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.agreements = vec!["x".repeat(4096)];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_session_summary_too_long_rejected() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec![],
            summary: "x".repeat(4097),
            next_steps: vec![],
        }];
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_session_summary_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec![],
            summary: "x".repeat(4096),
            next_steps: vec![],
        }];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_session_too_many_next_steps_rejected() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec![],
            summary: "Summary".into(),
            next_steps: (0..31).map(|i| format!("Step {}", i)).collect(),
        }];
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_session_next_steps_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec![],
            summary: "Summary".into(),
            next_steps: (0..30).map(|i| format!("Step {}", i)).collect(),
        }];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_session_next_step_too_long_rejected() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec![],
            summary: "Summary".into(),
            next_steps: vec!["x".repeat(4097)],
        }];
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_session_next_step_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec![],
            summary: "Summary".into(),
            next_steps: vec!["x".repeat(4096)],
        }];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_session_attendee_too_long_rejected() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec!["x".repeat(257)],
            summary: "Summary".into(),
            next_steps: vec![],
        }];
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
    }

    #[test]
    fn restorative_session_attendee_at_limit_accepted() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec!["x".repeat(256)],
            summary: "Summary".into(),
            next_steps: vec![],
        }];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    /// Helper to test create-link tag validation for a given link type.
    /// Mirrors the logic in the validate() FlatOp::RegisterCreateLink arm.
    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag_len = tag.len();
        match link_type {
            LinkTypes::CaseToEvidence | LinkTypes::CaseToDecisions => {
                if tag_len > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            _ => {
                if tag_len > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    fn validate_delete_link_tag(tag: Vec<u8>) -> ExternResult<ValidateCallbackResult> {
        if tag.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Delete link tag too long (max 256 bytes)".into(),
            ));
        }
        Ok(ValidateCallbackResult::Valid)
    }

    // -- ComplainantToCases (256 limit) --

    #[test]
    fn complainant_to_cases_link_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::ComplainantToCases, vec![0u8; 100]);
        assert!(is_valid(&result));
    }

    #[test]
    fn complainant_to_cases_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::ComplainantToCases, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn complainant_to_cases_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::ComplainantToCases, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // -- CaseToEvidence (512 limit) --

    #[test]
    fn case_to_evidence_link_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::CaseToEvidence, vec![0u8; 400]);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_to_evidence_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::CaseToEvidence, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_to_evidence_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::CaseToEvidence, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    // -- CaseToDecisions (512 limit) --

    #[test]
    fn case_to_decisions_link_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::CaseToDecisions, vec![0u8; 300]);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_to_decisions_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::CaseToDecisions, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_to_decisions_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::CaseToDecisions, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    // -- AllCases (256 limit) --

    #[test]
    fn all_cases_link_tag_empty() {
        let result = validate_create_link_tag(LinkTypes::AllCases, vec![]);
        assert!(is_valid(&result));
    }

    #[test]
    fn all_cases_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::AllCases, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn all_cases_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::AllCases, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // -- RespondentToCases (256 limit) --

    #[test]
    fn respondent_to_cases_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::RespondentToCases, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn respondent_to_cases_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::RespondentToCases, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // -- ArbitratorToCases (256 limit) --

    #[test]
    fn arbitrator_to_cases_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::ArbitratorToCases, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitrator_to_cases_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::ArbitratorToCases, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // -- DecisionToAppeals (256 limit) --

    #[test]
    fn decision_to_appeals_link_tag_at_limit() {
        let result = validate_create_link_tag(LinkTypes::DecisionToAppeals, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_to_appeals_link_tag_over_limit() {
        let result = validate_create_link_tag(LinkTypes::DecisionToAppeals, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    // -- Large tag DoS prevention --

    #[test]
    fn massive_link_tag_rejected_all_standard_types() {
        let huge_tag = vec![0xFFu8; 10_000];
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::ComplainantToCases,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::RespondentToCases,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::CaseToMediation,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::CaseToArbitration,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::DecisionToAppeals,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::DecisionToEnforcement,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::CaseToRestorativeCircle,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::ArbitratorToCases,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::AllCases,
            huge_tag.clone()
        )));
    }

    #[test]
    fn massive_link_tag_rejected_extended_types() {
        let huge_tag = vec![0xFFu8; 10_000];
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::CaseToEvidence,
            huge_tag.clone()
        )));
        assert!(is_invalid(&validate_create_link_tag(
            LinkTypes::CaseToDecisions,
            huge_tag.clone()
        )));
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_valid() {
        let result = validate_delete_link_tag(vec![0u8; 128]);
        assert!(is_valid(&result));
    }

    #[test]
    fn delete_link_tag_at_limit() {
        let result = validate_delete_link_tag(vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn delete_link_tag_over_limit() {
        let result = validate_delete_link_tag(vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn delete_link_tag_massive_rejected() {
        let result = validate_delete_link_tag(vec![0xFFu8; 10_000]);
        assert!(is_invalid(&result));
    }

    // ── Update validation tests ──────────────────────────────────────────

    #[test]
    fn test_update_case_invalid_title_rejected() {
        let mut case = make_case();
        case.title = String::new();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn test_update_case_invalid_description_rejected() {
        let mut case = make_case();
        case.description = "  ".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn test_update_case_invalid_self_filing_rejected() {
        let mut case = make_case();
        case.complainant = "did:key:same".into();
        case.respondent = "did:key:same".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
    }

    #[test]
    fn test_update_evidence_invalid_submitter_rejected() {
        let mut ev = make_evidence();
        ev.submitter = "not-a-did".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn test_update_evidence_invalid_description_rejected() {
        let mut ev = make_evidence();
        ev.description = String::new();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn test_update_evidence_invalid_content_hash_rejected() {
        let mut ev = make_evidence();
        ev.content.hash = "  ".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }
}
