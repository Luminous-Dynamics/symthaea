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
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
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

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid variant"),
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
        case.description = "  \t  ".into();
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
    fn case_complainant_empty_not_did_rejected() {
        let mut case = make_case();
        case.complainant = "".into();
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
        assert_eq!(invalid_msg(&result), "Cannot file case against self");
    }

    #[test]
    fn case_different_did_methods_passes() {
        let mut case = make_case();
        case.complainant = "did:key:z6MkpTHR8VNs5z".into();
        case.respondent = "did:web:example.com".into();
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_all_types_pass_validation() {
        for case_type in [
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
        ] {
            let mut case = make_case();
            case.case_type = case_type;
            let result = validate_case(&case);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn case_all_phases_pass_validation() {
        for phase in [
            CasePhase::Filed,
            CasePhase::Negotiation,
            CasePhase::Mediation,
            CasePhase::Arbitration,
            CasePhase::Appeal,
            CasePhase::Enforcement,
            CasePhase::Closed,
        ] {
            let mut case = make_case();
            case.phase = phase;
            let result = validate_case(&case);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn case_all_severities_pass_validation() {
        for severity in [
            CaseSeverity::Minor,
            CaseSeverity::Moderate,
            CaseSeverity::Serious,
            CaseSeverity::Critical,
        ] {
            let mut case = make_case();
            case.severity = severity;
            let result = validate_case(&case);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn case_with_parties_passes() {
        let mut case = make_case();
        case.parties = vec![
            CaseParty {
                did: "did:example:witness1".into(),
                role: PartyRole::Witness,
                joined_at: ts(),
            },
            CaseParty {
                did: "did:example:expert1".into(),
                role: PartyRole::Expert,
                joined_at: ts(),
            },
        ];
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_with_context_passes() {
        let mut case = make_case();
        case.context = CaseContext {
            happ: Some("mycelix-commons".into()),
            reference_id: Some("tx-12345".into()),
            community: Some("test-community".into()),
            jurisdiction: Some("global".into()),
        };
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
        assert_eq!(invalid_msg(&result), "Evidence submitter must be a DID");
    }

    #[test]
    fn evidence_submitter_empty_rejected() {
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
        ev.description = "   \n\t  ".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Evidence description required");
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
    fn evidence_all_types_pass_validation() {
        for ev_type in [
            EvidenceType::Document,
            EvidenceType::Transaction,
            EvidenceType::Communication,
            EvidenceType::Testimony,
            EvidenceType::ExpertOpinion,
            EvidenceType::Media,
            EvidenceType::OnChainData {
                happ: "commons".into(),
                entry_hash: "uhCkk...".into(),
            },
            EvidenceType::External {
                source: "court-records.gov".into(),
            },
        ] {
            let mut ev = make_evidence();
            ev.evidence_type = ev_type;
            let result = validate_evidence(&ev);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn evidence_all_visibility_types_pass() {
        for vis in [
            EvidenceVisibility::AllParties,
            EvidenceVisibility::AdjudicatorsOnly,
            EvidenceVisibility::Restricted {
                parties: vec!["did:example:alice".into()],
            },
            EvidenceVisibility::Sealed,
        ] {
            let mut ev = make_evidence();
            ev.visibility = vis;
            let result = validate_evidence(&ev);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn evidence_sealed_passes() {
        let mut ev = make_evidence();
        ev.sealed = true;
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_with_custody_chain_passes() {
        let mut ev = make_evidence();
        ev.custody = vec![
            CustodyEvent {
                action: CustodyAction::Submitted,
                actor: "did:example:alice".into(),
                timestamp: ts(),
                notes: Some("Initial submission".into()),
            },
            CustodyEvent {
                action: CustodyAction::Verified,
                actor: "did:example:verifier".into(),
                timestamp: ts(),
                notes: None,
            },
        ];
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_encrypted_with_key_ref_passes() {
        let mut ev = make_evidence();
        ev.content.encrypted = true;
        ev.content.key_reference = Some("key:abc123".into());
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
        assert_eq!(invalid_msg(&result), "Mediator must be a DID");
    }

    #[test]
    fn mediation_mediator_empty_rejected() {
        let mut med = make_mediation();
        med.mediator = "".into();
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_all_statuses_pass() {
        for status in [
            MediationStatus::Scheduled,
            MediationStatus::InProgress,
            MediationStatus::SettlementReached,
            MediationStatus::Failed,
            MediationStatus::Cancelled,
        ] {
            let mut med = make_mediation();
            med.status = status;
            let result = validate_mediation(&med);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn mediation_with_sessions_passes() {
        let mut med = make_mediation();
        med.sessions = vec![MediationSession {
            session_number: 1,
            scheduled_at: ts(),
            actual_start: Some(ts()),
            actual_end: Some(ts()),
            notes: Some("Productive session".into()),
            outcome: Some("Partial agreement".into()),
        }];
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_with_deadline_passes() {
        let mut med = make_mediation();
        med.deadline = Some(Timestamp::from_micros(1_000_000_000));
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
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
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
            make_arbitrator("did:example:arb4"),
            make_arbitrator("did:example:arb5"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitration_even_number_two_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Arbitration panel must have odd number of arbitrators"
        );
    }

    #[test]
    fn arbitration_even_number_four_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
            make_arbitrator("did:example:arb4"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_zero_arbitrators_rejected() {
        // 0.is_multiple_of(2) == true, so empty is rejected
        let arb = make_arbitration(vec![]);
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
        // Even count check passes (3 is odd), but DID check fails
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "All arbitrators must be DIDs");
    }

    #[test]
    fn arbitration_empty_did_arbitrator_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator(""),
            make_arbitrator("did:example:arb3"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn arbitration_all_selection_methods_pass() {
        for method in [
            ArbitratorSelection::Random,
            ArbitratorSelection::MATLWeighted,
            ArbitratorSelection::PartyAgreed,
            ArbitratorSelection::ExpertiseBased {
                domain: "finance".into(),
            },
        ] {
            let mut arb = make_arbitration(vec![make_arbitrator("did:example:arb1")]);
            arb.selection_method = method;
            let result = validate_arbitration(&arb);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn arbitration_all_statuses_pass() {
        for status in [
            ArbitrationStatus::PanelFormation,
            ArbitrationStatus::EvidenceReview,
            ArbitrationStatus::Hearing,
            ArbitrationStatus::Deliberation,
            ArbitrationStatus::DecisionDrafting,
            ArbitrationStatus::DecisionRendered,
            ArbitrationStatus::Appealed,
        ] {
            let mut arb = make_arbitration(vec![make_arbitrator("did:example:arb1")]);
            arb.status = status;
            let result = validate_arbitration(&arb);
            assert!(is_valid(&result));
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
        dec.reasoning = "   \n\t  ".into();
        let result = validate_decision(&dec);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Decision reasoning required");
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
            ArbitratorVote {
                arbitrator: "did:example:arb3".into(),
                vote: VoteChoice::ForRespondent,
                timestamp: ts(),
            },
        ];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_with_abstain_vote_passes() {
        let mut dec = make_decision();
        dec.votes = vec![ArbitratorVote {
            arbitrator: "did:example:arb1".into(),
            vote: VoteChoice::Abstain,
            timestamp: ts(),
        }];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_all_types_pass() {
        for dt in [
            DecisionType::MeritsDecision,
            DecisionType::InterimDecision,
            DecisionType::DefaultDecision,
            DecisionType::ConsentDecision,
            DecisionType::Dismissal,
        ] {
            let mut dec = make_decision();
            dec.decision_type = dt;
            let result = validate_decision(&dec);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn decision_all_outcomes_pass() {
        for outcome in [
            DecisionOutcome::ForComplainant,
            DecisionOutcome::ForRespondent,
            DecisionOutcome::SplitDecision,
            DecisionOutcome::Dismissed,
            DecisionOutcome::Settled,
        ] {
            let mut dec = make_decision();
            dec.outcome = outcome;
            let result = validate_decision(&dec);
            assert!(is_valid(&result));
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
                amount: Some(1000),
                currency: Some("SAP".into()),
                description: "Pay damages".into(),
            },
            Remedy {
                remedy_type: RemedyType::Apology,
                responsible_party: "did:example:bob".into(),
                deadline: None,
                amount: None,
                currency: None,
                description: "Public apology".into(),
            },
        ];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_with_dissents_passes() {
        let mut dec = make_decision();
        dec.dissents = vec![DissentingOpinion {
            arbitrator: "did:example:arb2".into(),
            opinion: "I disagree with the majority".into(),
            timestamp: ts(),
        }];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_finalized_passes() {
        let mut dec = make_decision();
        dec.finalized = true;
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
        assert_eq!(invalid_msg(&result), "Appellant must be a DID");
    }

    #[test]
    fn appeal_appellant_empty_rejected() {
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
        appeal.argument = "   \t\n  ".into();
        let result = validate_appeal(&appeal);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Appeal argument required");
    }

    #[test]
    fn appeal_all_grounds_pass() {
        for ground in [
            AppealGround::ProceduralError,
            AppealGround::NewEvidence,
            AppealGround::LegalError,
            AppealGround::Bias,
            AppealGround::ExcessiveRemedy,
            AppealGround::InsufficientRemedy,
        ] {
            let mut appeal = make_appeal();
            appeal.grounds = vec![ground];
            let result = validate_appeal(&appeal);
            assert!(is_valid(&result));
        }
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
    fn appeal_all_statuses_pass() {
        for status in [
            AppealStatus::Filed,
            AppealStatus::UnderReview,
            AppealStatus::Granted,
            AppealStatus::Denied,
            AppealStatus::Remanded,
            AppealStatus::Resolved,
        ] {
            let mut appeal = make_appeal();
            appeal.status = status;
            let result = validate_appeal(&appeal);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn appeal_various_appeal_numbers_pass() {
        for n in [1u8, 2, 3, u8::MAX] {
            let mut appeal = make_appeal();
            appeal.appeal_number = n;
            let result = validate_appeal(&appeal);
            assert!(is_valid(&result));
        }
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
    fn enforcement_enforcer_empty_rejected() {
        let mut enf = make_enforcement();
        enf.enforcer = "".into();
        let result = validate_enforcement(&enf);
        assert!(is_invalid(&result));
    }

    #[test]
    fn enforcement_all_statuses_pass() {
        for status in [
            EnforcementStatus::Pending,
            EnforcementStatus::InProgress,
            EnforcementStatus::PartiallyCompleted,
            EnforcementStatus::Completed,
            EnforcementStatus::Failed,
            EnforcementStatus::Contested,
        ] {
            let mut enf = make_enforcement();
            enf.status = status;
            let result = validate_enforcement(&enf);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn enforcement_with_actions_passes() {
        let mut enf = make_enforcement();
        enf.actions = vec![
            EnforcementAction {
                action_type: EnforcementActionType::FundsTransfer,
                target_happ: Some("mycelix-commons".into()),
                target_entry: Some("entry-123".into()),
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
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_all_action_types_pass() {
        for action_type in [
            EnforcementActionType::FundsTransfer,
            EnforcementActionType::AssetFreeze,
            EnforcementActionType::ReputationUpdate,
            EnforcementActionType::AccessRevocation,
            EnforcementActionType::Notification,
            EnforcementActionType::ManualRequired,
            EnforcementActionType::CrossHappAction,
        ] {
            let mut enf = make_enforcement();
            enf.actions = vec![EnforcementAction {
                action_type,
                target_happ: None,
                target_entry: None,
                executed_at: ts(),
                result: "done".into(),
            }];
            let result = validate_enforcement(&enf);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn enforcement_with_completed_at_passes() {
        let mut enf = make_enforcement();
        enf.completed_at = Some(Timestamp::from_micros(1_000_000));
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_various_remedy_indices_pass() {
        for idx in [0u32, 1, 5, 100, u32::MAX] {
            let mut enf = make_enforcement();
            enf.remedy_index = idx;
            let result = validate_enforcement(&enf);
            assert!(is_valid(&result));
        }
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
    fn restorative_facilitator_empty_rejected() {
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
    fn restorative_all_circle_roles_pass() {
        for role in [
            CircleRole::Facilitator,
            CircleRole::HarmDoer,
            CircleRole::HarmReceiver,
            CircleRole::CommunityMember,
            CircleRole::SupportPerson,
            CircleRole::Elder,
        ] {
            let mut circle = make_restorative_circle();
            circle.participants = vec![CircleParticipant {
                did: "did:example:person".into(),
                role,
                consented: true,
                attended_sessions: vec![],
            }];
            let result = validate_restorative(&circle);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn restorative_all_statuses_pass() {
        for status in [
            CircleStatus::Forming,
            CircleStatus::Active,
            CircleStatus::AgreementReached,
            CircleStatus::Monitoring,
            CircleStatus::Completed,
            CircleStatus::Discontinued,
        ] {
            let mut circle = make_restorative_circle();
            circle.status = status;
            let result = validate_restorative(&circle);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn restorative_with_sessions_passes() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![CircleSession {
            session_number: 1,
            held_at: ts(),
            attendees: vec!["did:example:alice".into(), "did:example:bob".into()],
            summary: "First circle gathering".into(),
            next_steps: vec!["Schedule follow-up".into()],
        }];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_with_agreements_passes() {
        let mut circle = make_restorative_circle();
        circle.agreements = vec!["agreement-1".into(), "agreement-2".into()];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_unconsented_participant_passes() {
        let mut circle = make_restorative_circle();
        circle.participants = vec![CircleParticipant {
            did: "did:example:person".into(),
            role: CircleRole::HarmDoer,
            consented: false,
            attended_sessions: vec![],
        }];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn case_serde_roundtrip() {
        let case = make_case();
        let json = serde_json::to_string(&case).expect("serialize case");
        let deserialized: Case = serde_json::from_str(&json).expect("deserialize case");
        assert_eq!(deserialized.id, case.id);
        assert_eq!(deserialized.title, case.title);
        assert_eq!(deserialized.description, case.description);
        assert_eq!(deserialized.complainant, case.complainant);
        assert_eq!(deserialized.respondent, case.respondent);
        assert_eq!(deserialized.case_type, case.case_type);
        assert_eq!(deserialized.phase, case.phase);
        assert_eq!(deserialized.status, case.status);
        assert_eq!(deserialized.severity, case.severity);
    }

    #[test]
    fn evidence_serde_roundtrip() {
        let evidence = make_evidence();
        let json = serde_json::to_string(&evidence).expect("serialize evidence");
        let deserialized: Evidence = serde_json::from_str(&json).expect("deserialize evidence");
        assert_eq!(deserialized.id, evidence.id);
        assert_eq!(deserialized.case_id, evidence.case_id);
        assert_eq!(deserialized.submitter, evidence.submitter);
        assert_eq!(deserialized.evidence_type, evidence.evidence_type);
        assert_eq!(deserialized.description, evidence.description);
        assert_eq!(deserialized.sealed, evidence.sealed);
    }

    #[test]
    fn mediation_serde_roundtrip() {
        let mediation = make_mediation();
        let json = serde_json::to_string(&mediation).expect("serialize mediation");
        let deserialized: Mediation = serde_json::from_str(&json).expect("deserialize mediation");
        assert_eq!(deserialized.id, mediation.id);
        assert_eq!(deserialized.case_id, mediation.case_id);
        assert_eq!(deserialized.mediator, mediation.mediator);
        assert_eq!(deserialized.status, mediation.status);
    }

    #[test]
    fn arbitration_serde_roundtrip() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
        ]);
        let json = serde_json::to_string(&arb).expect("serialize arbitration");
        let deserialized: Arbitration =
            serde_json::from_str(&json).expect("deserialize arbitration");
        assert_eq!(deserialized.id, arb.id);
        assert_eq!(deserialized.case_id, arb.case_id);
        assert_eq!(deserialized.arbitrators.len(), 3);
        assert_eq!(deserialized.selection_method, arb.selection_method);
        assert_eq!(deserialized.status, arb.status);
    }

    #[test]
    fn decision_serde_roundtrip() {
        let decision = make_decision();
        let json = serde_json::to_string(&decision).expect("serialize decision");
        let deserialized: Decision = serde_json::from_str(&json).expect("deserialize decision");
        assert_eq!(deserialized.id, decision.id);
        assert_eq!(deserialized.case_id, decision.case_id);
        assert_eq!(deserialized.arbitration_id, decision.arbitration_id);
        assert_eq!(deserialized.decision_type, decision.decision_type);
        assert_eq!(deserialized.outcome, decision.outcome);
        assert_eq!(deserialized.reasoning, decision.reasoning);
        assert_eq!(deserialized.finalized, decision.finalized);
    }

    #[test]
    fn appeal_serde_roundtrip() {
        let appeal = make_appeal();
        let json = serde_json::to_string(&appeal).expect("serialize appeal");
        let deserialized: Appeal = serde_json::from_str(&json).expect("deserialize appeal");
        assert_eq!(deserialized.id, appeal.id);
        assert_eq!(deserialized.case_id, appeal.case_id);
        assert_eq!(deserialized.decision_id, appeal.decision_id);
        assert_eq!(deserialized.appellant, appeal.appellant);
        assert_eq!(deserialized.grounds, appeal.grounds);
        assert_eq!(deserialized.argument, appeal.argument);
        assert_eq!(deserialized.status, appeal.status);
        assert_eq!(deserialized.appeal_number, appeal.appeal_number);
    }

    #[test]
    fn enforcement_serde_roundtrip() {
        let enf = make_enforcement();
        let json = serde_json::to_string(&enf).expect("serialize enforcement");
        let deserialized: Enforcement =
            serde_json::from_str(&json).expect("deserialize enforcement");
        assert_eq!(deserialized.id, enf.id);
        assert_eq!(deserialized.decision_id, enf.decision_id);
        assert_eq!(deserialized.remedy_index, enf.remedy_index);
        assert_eq!(deserialized.enforcer, enf.enforcer);
        assert_eq!(deserialized.status, enf.status);
    }

    #[test]
    fn restorative_circle_serde_roundtrip() {
        let circle = make_restorative_circle();
        let json = serde_json::to_string(&circle).expect("serialize circle");
        let deserialized: RestorativeCircle =
            serde_json::from_str(&json).expect("deserialize circle");
        assert_eq!(deserialized.id, circle.id);
        assert_eq!(deserialized.case_id, circle.case_id);
        assert_eq!(deserialized.facilitator, circle.facilitator);
        assert_eq!(deserialized.participants.len(), 2);
        assert_eq!(deserialized.status, circle.status);
    }

    #[test]
    fn case_type_other_serde_roundtrip() {
        let ct = CaseType::Other {
            category: "environmental".into(),
        };
        let json = serde_json::to_string(&ct).expect("serialize");
        let deserialized: CaseType = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, ct);
    }

    #[test]
    fn evidence_on_chain_data_serde_roundtrip() {
        let et = EvidenceType::OnChainData {
            happ: "commons".into(),
            entry_hash: "uhCkk123".into(),
        };
        let json = serde_json::to_string(&et).expect("serialize");
        let deserialized: EvidenceType = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, et);
    }

    #[test]
    fn remedy_all_types_serde_roundtrip() {
        for rt in [
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
        ] {
            let json = serde_json::to_string(&rt).expect("serialize");
            let deserialized: RemedyType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, rt);
        }
    }

    #[test]
    fn decision_with_full_remedy_serde_roundtrip() {
        let remedy = Remedy {
            remedy_type: RemedyType::Compensation,
            responsible_party: "did:example:bob".into(),
            deadline: Some(ts()),
            amount: Some(50_000),
            currency: Some("SAP".into()),
            description: "Compensate for damages".into(),
        };
        let json = serde_json::to_string(&remedy).expect("serialize");
        let deserialized: Remedy = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.remedy_type, remedy.remedy_type);
        assert_eq!(deserialized.responsible_party, remedy.responsible_party);
        assert_eq!(deserialized.amount, remedy.amount);
        assert_eq!(deserialized.currency, remedy.currency);
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
    // BOUNDARY / EDGE CASE TESTS
    // ========================================================================

    #[test]
    fn case_did_prefix_only_passes() {
        // "did:" alone technically passes the starts_with check
        let mut case = make_case();
        case.complainant = "did:".into();
        case.respondent = "did:other".into();
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_title_single_char_passes() {
        let mut case = make_case();
        case.title = "X".into();
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_description_single_char_passes() {
        let mut case = make_case();
        case.description = "D".into();
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_content_hash_whitespace_rejected() {
        let mut ev = make_evidence();
        ev.content.hash = "   ".into();
        let result = validate_evidence(&ev);
        assert!(is_invalid(&result));
    }

    #[test]
    fn appeal_zero_appeal_number_passes() {
        let mut appeal = make_appeal();
        appeal.appeal_number = 0;
        let result = validate_appeal(&appeal);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_did_system_account_passes() {
        let mut enf = make_enforcement();
        enf.enforcer = "did:system:enforcement-agent".into();
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_unicode_title_passes() {
        let mut case = make_case();
        case.title = "Contrat en litige".into();
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_unicode_description_passes() {
        let mut case = make_case();
        case.description = "This involves damages".into();
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_zero_size_passes() {
        let mut ev = make_evidence();
        ev.content.size = 0;
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_large_size_passes() {
        let mut ev = make_evidence();
        ev.content.size = u64::MAX;
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // ADDITIONAL ENUM VARIANT SERDE TESTS
    // ========================================================================

    #[test]
    fn verification_status_all_variants_serde() {
        for status in [
            VerificationStatus::Unverified,
            VerificationStatus::Pending,
            VerificationStatus::Verified,
            VerificationStatus::Disputed,
            VerificationStatus::Rejected,
        ] {
            let json = serde_json::to_string(&status).expect("serialize");
            let deserialized: VerificationStatus =
                serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, status);
        }
    }

    #[test]
    fn custody_action_all_variants_serde() {
        for action in [
            CustodyAction::Submitted,
            CustodyAction::Accessed,
            CustodyAction::Copied,
            CustodyAction::Verified,
            CustodyAction::Challenged,
            CustodyAction::Sealed,
            CustodyAction::Released,
        ] {
            let json = serde_json::to_string(&action).expect("serialize");
            let deserialized: CustodyAction = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, action);
        }
    }

    #[test]
    fn party_role_all_variants_serde() {
        for role in [
            PartyRole::Complainant,
            PartyRole::Respondent,
            PartyRole::Witness,
            PartyRole::Expert,
            PartyRole::Intervenor,
            PartyRole::Affected,
        ] {
            let json = serde_json::to_string(&role).expect("serialize");
            let deserialized: PartyRole = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, role);
        }
    }

    #[test]
    fn vote_choice_all_variants_serde() {
        for choice in [
            VoteChoice::ForComplainant,
            VoteChoice::ForRespondent,
            VoteChoice::Abstain,
        ] {
            let json = serde_json::to_string(&choice).expect("serialize");
            let deserialized: VoteChoice = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, choice);
        }
    }

    #[test]
    fn arbitrator_role_all_variants_serde() {
        for role in [
            ArbitratorRole::Primary,
            ArbitratorRole::PanelMember,
            ArbitratorRole::Alternate,
        ] {
            let json = serde_json::to_string(&role).expect("serialize");
            let deserialized: ArbitratorRole = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, role);
        }
    }

    #[test]
    fn case_status_all_variants_serde() {
        for status in [
            CaseStatus::Active,
            CaseStatus::OnHold,
            CaseStatus::AwaitingResponse,
            CaseStatus::InDeliberation,
            CaseStatus::DecisionRendered,
            CaseStatus::Enforcing,
            CaseStatus::Resolved,
            CaseStatus::Dismissed,
            CaseStatus::Withdrawn,
        ] {
            let json = serde_json::to_string(&status).expect("serialize");
            let deserialized: CaseStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, status);
        }
    }

    #[test]
    fn case_phase_all_variants_serde() {
        for phase in [
            CasePhase::Filed,
            CasePhase::Negotiation,
            CasePhase::Mediation,
            CasePhase::Arbitration,
            CasePhase::Appeal,
            CasePhase::Enforcement,
            CasePhase::Closed,
        ] {
            let json = serde_json::to_string(&phase).expect("serialize");
            let deserialized: CasePhase = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, phase);
        }
    }

    #[test]
    fn case_severity_all_variants_serde() {
        for severity in [
            CaseSeverity::Minor,
            CaseSeverity::Moderate,
            CaseSeverity::Serious,
            CaseSeverity::Critical,
        ] {
            let json = serde_json::to_string(&severity).expect("serialize");
            let deserialized: CaseSeverity = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(deserialized, severity);
        }
    }

    #[test]
    fn evidence_visibility_restricted_serde() {
        let vis = EvidenceVisibility::Restricted {
            parties: vec!["did:example:alice".into(), "did:example:bob".into()],
        };
        let json = serde_json::to_string(&vis).expect("serialize");
        let deserialized: EvidenceVisibility = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, vis);
    }

    #[test]
    fn evidence_visibility_restricted_empty_list_passes() {
        // Edge case: restricted but no parties listed (validation doesn't check this)
        let mut ev = make_evidence();
        ev.visibility = EvidenceVisibility::Restricted { parties: vec![] };
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitrator_selection_expertise_based_serde() {
        let sel = ArbitratorSelection::ExpertiseBased {
            domain: "intellectual-property".into(),
        };
        let json = serde_json::to_string(&sel).expect("serialize");
        let deserialized: ArbitratorSelection = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized, sel);
    }

    // ========================================================================
    // ADDITIONAL EDGE CASE TESTS
    // ========================================================================

    #[test]
    fn arbitrator_recused_passes_validation() {
        let arb = make_arbitration(vec![Arbitrator {
            did: "did:example:arb1".into(),
            role: ArbitratorRole::PanelMember,
            selected_at: ts(),
            accepted: true,
            recused: true,
            recusal_reason: Some("Conflict of interest".into()),
        }]);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitrator_not_accepted_passes_validation() {
        let arb = make_arbitration(vec![Arbitrator {
            did: "did:example:arb1".into(),
            role: ArbitratorRole::Alternate,
            selected_at: ts(),
            accepted: false,
            recused: false,
            recusal_reason: None,
        }]);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_party_all_roles_pass() {
        for role in [
            PartyRole::Complainant,
            PartyRole::Respondent,
            PartyRole::Witness,
            PartyRole::Expert,
            PartyRole::Intervenor,
            PartyRole::Affected,
        ] {
            let mut case = make_case();
            case.parties = vec![CaseParty {
                did: "did:example:party".into(),
                role,
                joined_at: ts(),
            }];
            let result = validate_case(&case);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn decision_all_remedy_types_pass_in_remedies() {
        for remedy_type in [
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
        ] {
            let mut dec = make_decision();
            dec.remedies = vec![Remedy {
                remedy_type,
                responsible_party: "did:example:bob".into(),
                deadline: None,
                amount: None,
                currency: None,
                description: "Remedy ordered".into(),
            }];
            let result = validate_decision(&dec);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn evidence_all_verification_statuses_pass() {
        for status in [
            VerificationStatus::Unverified,
            VerificationStatus::Pending,
            VerificationStatus::Verified,
            VerificationStatus::Disputed,
            VerificationStatus::Rejected,
        ] {
            let mut ev = make_evidence();
            ev.verification.status = status;
            let result = validate_evidence(&ev);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn evidence_all_custody_actions_pass() {
        for action in [
            CustodyAction::Submitted,
            CustodyAction::Accessed,
            CustodyAction::Copied,
            CustodyAction::Verified,
            CustodyAction::Challenged,
            CustodyAction::Sealed,
            CustodyAction::Released,
        ] {
            let mut ev = make_evidence();
            ev.custody = vec![CustodyEvent {
                action,
                actor: "did:example:actor".into(),
                timestamp: ts(),
                notes: None,
            }];
            let result = validate_evidence(&ev);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn evidence_verified_with_verifier_passes() {
        let mut ev = make_evidence();
        ev.verification = EvidenceVerification {
            status: VerificationStatus::Verified,
            verifier: Some("did:example:verifier".into()),
            method: Some("cryptographic-signature".into()),
            verified_at: Some(ts()),
            notes: Some("Signature verified".into()),
        };
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_with_phase_deadline_passes() {
        let mut case = make_case();
        case.phase_deadline = Some(Timestamp::from_micros(2_000_000_000));
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitration_seven_arbitrators_passes() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
            make_arbitrator("did:example:arb4"),
            make_arbitrator("did:example:arb5"),
            make_arbitrator("did:example:arb6"),
            make_arbitrator("did:example:arb7"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitration_nine_arbitrators_passes() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
            make_arbitrator("did:example:arb4"),
            make_arbitrator("did:example:arb5"),
            make_arbitrator("did:example:arb6"),
            make_arbitrator("did:example:arb7"),
            make_arbitrator("did:example:arb8"),
            make_arbitrator("did:example:arb9"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_valid(&result));
    }

    #[test]
    fn arbitration_six_arbitrators_rejected() {
        let arb = make_arbitration(vec![
            make_arbitrator("did:example:arb1"),
            make_arbitrator("did:example:arb2"),
            make_arbitrator("did:example:arb3"),
            make_arbitrator("did:example:arb4"),
            make_arbitrator("did:example:arb5"),
            make_arbitrator("did:example:arb6"),
        ]);
        let result = validate_arbitration(&arb);
        assert!(is_invalid(&result));
    }

    #[test]
    fn mediation_empty_sessions_passes() {
        let mut med = make_mediation();
        med.sessions = vec![];
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn mediation_empty_proposals_passes() {
        let mut med = make_mediation();
        med.proposals = vec![];
        let result = validate_mediation(&med);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_circle_empty_sessions_passes() {
        let mut circle = make_restorative_circle();
        circle.sessions = vec![];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_circle_empty_agreements_passes() {
        let mut circle = make_restorative_circle();
        circle.agreements = vec![];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn restorative_participant_attended_multiple_sessions_passes() {
        let mut circle = make_restorative_circle();
        circle.participants = vec![CircleParticipant {
            did: "did:example:participant".into(),
            role: CircleRole::HarmReceiver,
            consented: true,
            attended_sessions: vec![1, 2, 3, 4],
        }];
        let result = validate_restorative(&circle);
        assert!(is_valid(&result));
    }

    #[test]
    fn enforcement_empty_actions_passes() {
        let mut enf = make_enforcement();
        enf.actions = vec![];
        let result = validate_enforcement(&enf);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_empty_remedies_passes() {
        let mut dec = make_decision();
        dec.remedies = vec![];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn decision_empty_dissents_passes() {
        let mut dec = make_decision();
        dec.dissents = vec![];
        let result = validate_decision(&dec);
        assert!(is_valid(&result));
    }

    #[test]
    fn case_empty_parties_passes() {
        let mut case = make_case();
        case.parties = vec![];
        let result = validate_case(&case);
        assert!(is_valid(&result));
    }

    #[test]
    fn evidence_empty_custody_chain_passes() {
        let mut ev = make_evidence();
        ev.custody = vec![];
        let result = validate_evidence(&ev);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // COMPLEX STRUCT SERDE TESTS
    // ========================================================================

    #[test]
    fn case_context_full_serde_roundtrip() {
        let ctx = CaseContext {
            happ: Some("mycelix-commons".into()),
            reference_id: Some("entry-abc-123".into()),
            community: Some("global-community".into()),
            jurisdiction: Some("international".into()),
        };
        let json = serde_json::to_string(&ctx).expect("serialize");
        let deserialized: CaseContext = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.happ, ctx.happ);
        assert_eq!(deserialized.reference_id, ctx.reference_id);
        assert_eq!(deserialized.community, ctx.community);
        assert_eq!(deserialized.jurisdiction, ctx.jurisdiction);
    }

    #[test]
    fn case_context_empty_serde_roundtrip() {
        let ctx = CaseContext {
            happ: None,
            reference_id: None,
            community: None,
            jurisdiction: None,
        };
        let json = serde_json::to_string(&ctx).expect("serialize");
        let deserialized: CaseContext = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.happ, None);
        assert_eq!(deserialized.reference_id, None);
    }

    #[test]
    fn evidence_content_full_serde_roundtrip() {
        let content = EvidenceContent {
            hash: "sha256:abcdef123456".into(),
            reference: "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi".into(),
            mime_type: "application/pdf".into(),
            size: 2048,
            encrypted: true,
            key_reference: Some("key:xyz789".into()),
        };
        let json = serde_json::to_string(&content).expect("serialize");
        let deserialized: EvidenceContent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.hash, content.hash);
        assert_eq!(deserialized.reference, content.reference);
        assert_eq!(deserialized.mime_type, content.mime_type);
        assert_eq!(deserialized.size, content.size);
        assert_eq!(deserialized.encrypted, content.encrypted);
        assert_eq!(deserialized.key_reference, content.key_reference);
    }

    #[test]
    fn custody_event_serde_roundtrip() {
        let event = CustodyEvent {
            action: CustodyAction::Verified,
            actor: "did:example:verifier".into(),
            timestamp: ts(),
            notes: Some("Verified hash matches".into()),
        };
        let json = serde_json::to_string(&event).expect("serialize");
        let deserialized: CustodyEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.action, event.action);
        assert_eq!(deserialized.actor, event.actor);
        assert_eq!(deserialized.notes, event.notes);
    }

    #[test]
    fn evidence_verification_full_serde_roundtrip() {
        let verif = EvidenceVerification {
            status: VerificationStatus::Verified,
            verifier: Some("did:example:expert".into()),
            method: Some("chain-of-custody-audit".into()),
            verified_at: Some(ts()),
            notes: Some("All checks passed".into()),
        };
        let json = serde_json::to_string(&verif).expect("serialize");
        let deserialized: EvidenceVerification = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.status, verif.status);
        assert_eq!(deserialized.verifier, verif.verifier);
        assert_eq!(deserialized.method, verif.method);
        assert_eq!(deserialized.notes, verif.notes);
    }

    #[test]
    fn mediation_session_serde_roundtrip() {
        let session = MediationSession {
            session_number: 3,
            scheduled_at: ts(),
            actual_start: Some(ts()),
            actual_end: Some(ts()),
            notes: Some("Productive discussion".into()),
            outcome: Some("Partial agreement reached".into()),
        };
        let json = serde_json::to_string(&session).expect("serialize");
        let deserialized: MediationSession = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.session_number, session.session_number);
        assert_eq!(deserialized.notes, session.notes);
        assert_eq!(deserialized.outcome, session.outcome);
    }

    #[test]
    fn arbitrator_serde_roundtrip() {
        let arb = Arbitrator {
            did: "did:example:arbitrator".into(),
            role: ArbitratorRole::Primary,
            selected_at: ts(),
            accepted: true,
            recused: false,
            recusal_reason: None,
        };
        let json = serde_json::to_string(&arb).expect("serialize");
        let deserialized: Arbitrator = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.did, arb.did);
        assert_eq!(deserialized.role, arb.role);
        assert_eq!(deserialized.accepted, arb.accepted);
        assert_eq!(deserialized.recused, arb.recused);
    }

    #[test]
    fn arbitrator_vote_serde_roundtrip() {
        let vote = ArbitratorVote {
            arbitrator: "did:example:arb1".into(),
            vote: VoteChoice::ForComplainant,
            timestamp: ts(),
        };
        let json = serde_json::to_string(&vote).expect("serialize");
        let deserialized: ArbitratorVote = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.arbitrator, vote.arbitrator);
        assert_eq!(deserialized.vote, vote.vote);
    }

    #[test]
    fn dissenting_opinion_serde_roundtrip() {
        let dissent = DissentingOpinion {
            arbitrator: "did:example:arb2".into(),
            opinion: "I believe the evidence was misinterpreted".into(),
            timestamp: ts(),
        };
        let json = serde_json::to_string(&dissent).expect("serialize");
        let deserialized: DissentingOpinion = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.arbitrator, dissent.arbitrator);
        assert_eq!(deserialized.opinion, dissent.opinion);
    }

    #[test]
    fn enforcement_action_serde_roundtrip() {
        let action = EnforcementAction {
            action_type: EnforcementActionType::FundsTransfer,
            target_happ: Some("mycelix-commons".into()),
            target_entry: Some("entry-xyz".into()),
            executed_at: ts(),
            result: "Transfer successful".into(),
        };
        let json = serde_json::to_string(&action).expect("serialize");
        let deserialized: EnforcementAction = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.action_type, action.action_type);
        assert_eq!(deserialized.target_happ, action.target_happ);
        assert_eq!(deserialized.target_entry, action.target_entry);
        assert_eq!(deserialized.result, action.result);
    }

    #[test]
    fn circle_participant_serde_roundtrip() {
        let participant = CircleParticipant {
            did: "did:example:participant".into(),
            role: CircleRole::HarmDoer,
            consented: true,
            attended_sessions: vec![1, 2, 3],
        };
        let json = serde_json::to_string(&participant).expect("serialize");
        let deserialized: CircleParticipant = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.did, participant.did);
        assert_eq!(deserialized.role, participant.role);
        assert_eq!(deserialized.consented, participant.consented);
        assert_eq!(
            deserialized.attended_sessions,
            participant.attended_sessions
        );
    }

    #[test]
    fn circle_session_serde_roundtrip() {
        let session = CircleSession {
            session_number: 2,
            held_at: ts(),
            attendees: vec!["did:example:alice".into(), "did:example:bob".into()],
            summary: "Agreement on restoration plan".into(),
            next_steps: vec!["Community service".into(), "Follow-up in 30 days".into()],
        };
        let json = serde_json::to_string(&session).expect("serialize");
        let deserialized: CircleSession = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.session_number, session.session_number);
        assert_eq!(deserialized.attendees, session.attendees);
        assert_eq!(deserialized.summary, session.summary);
        assert_eq!(deserialized.next_steps, session.next_steps);
    }

    #[test]
    fn case_party_serde_roundtrip() {
        let party = CaseParty {
            did: "did:example:witness".into(),
            role: PartyRole::Witness,
            joined_at: ts(),
        };
        let json = serde_json::to_string(&party).expect("serialize");
        let deserialized: CaseParty = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.did, party.did);
        assert_eq!(deserialized.role, party.role);
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
    fn test_update_case_invalid_description_rejected() {
        let mut case = make_case();
        case.description = "  ".into();
        let result = validate_case(&case);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("Case description required"));
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
    fn test_update_mediation_invalid_mediator_rejected() {
        let mut med = make_mediation();
        med.mediator = "not-a-did".into();
        let result = validate_mediation(&med);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("Mediator must be a DID"));
    }

    #[test]
    fn test_update_restorative_invalid_facilitator_rejected() {
        let mut circle = make_restorative_circle();
        circle.facilitator = "not-a-did".into();
        let result = validate_restorative(&circle);
        assert!(is_invalid(&result));
        assert!(invalid_msg(&result).contains("Facilitator must be a DID"));
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
