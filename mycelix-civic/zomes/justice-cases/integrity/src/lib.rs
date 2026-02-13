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
    pub proposals: Vec<String>,  // Settlement IDs
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
    if arb.arbitrators.len().is_multiple_of(2) {
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
}
