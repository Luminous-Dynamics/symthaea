//! Proposals Integrity Zome
//! Defines entry types and validation for governance proposals
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A governance proposal (MIP - Mycelix Improvement Proposal)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Proposal {
    /// Unique proposal identifier (MIP-XXX)
    pub id: String,
    /// Proposal title
    pub title: String,
    /// Full description/content
    pub description: String,
    /// Proposal type
    pub proposal_type: ProposalType,
    /// Author's DID
    pub author: String,
    /// Current status
    pub status: ProposalStatus,
    /// Actions to execute if approved (JSON)
    pub actions: String,
    /// Discussion/forum link
    pub discussion_url: Option<String>,
    /// Voting start timestamp
    pub voting_starts: Timestamp,
    /// Voting end timestamp
    pub voting_ends: Timestamp,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last update timestamp
    pub updated: Timestamp,
    /// Version for updates
    pub version: u32,
}

/// Types of proposals
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProposalType {
    /// Standard proposal (7 day voting)
    Standard,
    /// Emergency proposal (24 hour voting)
    Emergency,
    /// Constitutional amendment (30 day voting)
    Constitutional,
    /// Parameter change (variable)
    Parameter,
    /// Funding request
    Funding,
}

/// Status of a proposal
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ProposalStatus {
    /// Draft, not yet submitted for voting
    Draft,
    /// Open for voting
    Active,
    /// Voting ended, awaiting tally
    Ended,
    /// Approved, in timelock
    Approved,
    /// Threshold signature obtained, ready for execution
    Signed,
    /// Rejected by vote
    Rejected,
    /// Timelock complete, executed
    Executed,
    /// Cancelled by author or governance
    Cancelled,
    /// Failed execution
    Failed,
}

/// Proposal amendment/update request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProposalAmendment {
    /// Amendment identifier
    pub id: String,
    /// Original proposal ID
    pub proposal_id: String,
    /// Proposed changes
    pub changes: String,
    /// Reason for amendment
    pub reason: String,
    /// Proposer's DID
    pub proposer: String,
    /// Amendment status
    pub status: AmendmentStatus,
    /// Creation timestamp
    pub created: Timestamp,
}

/// Status of an amendment
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AmendmentStatus {
    Proposed,
    Accepted,
    Rejected,
    Withdrawn,
}

// ============================================================================
// DISCUSSION SYSTEM WITH COLLECTIVE SENSING
// ============================================================================

/// A contribution to proposal discussion
///
/// This is where collective sensing begins - in the conversation itself.
/// Each contribution carries metadata for the Mirror.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DiscussionContribution {
    /// Unique contribution ID
    pub id: String,
    /// Proposal being discussed
    pub proposal_id: String,
    /// Contributor's DID
    pub contributor: String,
    /// The contribution content
    pub content: String,
    /// Which harmonies does this contribution invoke?
    pub harmony_tags: Vec<String>,
    /// Sentiment/position (optional)
    pub stance: Option<Stance>,
    /// Parent contribution ID for threading
    pub parent_id: Option<String>,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Has this been edited?
    pub edited: bool,
}

/// Stance on the proposal
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Stance {
    /// Supports the proposal
    Support,
    /// Opposes the proposal
    Oppose,
    /// Neutral/questioning
    Neutral,
    /// Suggests modification
    Amend,
}

/// Collective sensing of the discussion phase
///
/// Philosophy: Mirror the discussion, not just the votes.
/// The quality of deliberation matters as much as the outcome.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DiscussionReflection {
    /// Unique reflection ID
    pub id: String,
    /// Proposal being reflected on
    pub proposal_id: String,
    /// When this reflection was generated
    pub timestamp: Timestamp,

    // === PARTICIPATION METRICS ===
    /// Number of unique contributors
    pub contributor_count: u64,
    /// Total contributions
    pub contribution_count: u64,
    /// Average contributions per participant
    pub avg_contributions_per_participant: f64,
    /// Thread depth distribution
    pub max_thread_depth: u8,

    // === HARMONY ANALYSIS ===
    /// Which harmonies are represented
    pub harmony_coverage: Vec<HarmonyPresence>,
    /// Overall harmony diversity (0-1)
    pub harmony_diversity: f64,
    /// Absent harmonies
    pub absent_harmonies: Vec<String>,

    // === STANCE DISTRIBUTION ===
    pub support_count: u64,
    pub oppose_count: u64,
    pub neutral_count: u64,
    pub amend_count: u64,
    /// Preliminary approval sentiment (0-1)
    pub preliminary_sentiment: f64,

    // === DISCUSSION HEALTH ===
    /// Is discussion concentrated in few voices?
    pub voice_concentration: f64,
    /// Are opposing views engaging with each other?
    pub cross_camp_engagement: f64,
    /// Is discussion substantive or echo-chamber?
    pub substantiveness_score: f64,

    // === READINESS SIGNALS ===
    /// Has discussion reached saturation?
    pub discussion_saturated: bool,
    /// Are there unaddressed concerns?
    pub unaddressed_concerns: Vec<String>,
    /// Recommendation for proceeding to vote
    pub ready_for_vote: bool,
    /// Reasoning for readiness assessment
    pub readiness_reasoning: String,

    /// Human-readable summary
    pub summary: String,
}

/// Presence level of a harmony in discussion
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct HarmonyPresence {
    pub harmony: String,
    /// How present is this harmony (0-1)
    pub presence: f64,
    /// Example contribution invoking this harmony
    pub example_contribution_id: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Proposal(Proposal),
    ProposalAmendment(ProposalAmendment),
    DiscussionContribution(DiscussionContribution),
    DiscussionReflection(DiscussionReflection),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Author to their proposals
    AuthorToProposal,
    /// Proposal type to proposals
    TypeToProposal,
    /// Status to proposals (for querying)
    StatusToProposal,
    /// Proposal to amendments
    ProposalToAmendment,
    /// All active proposals
    ActiveProposals,
    /// Proposal to discussion contributions
    ProposalToContribution,
    /// Contributor to their contributions
    ContributorToContribution,
    /// Contribution to replies (threading)
    ContributionToReply,
    /// Proposal to discussion reflections
    ProposalToDiscussionReflection,
    /// O(1) lookup: proposal ID anchor → proposal record
    ProposalById,
}

// ============================================================================
// PURE VALIDATION FUNCTIONS (testable without HDI host)
// ============================================================================

pub fn check_create_proposal(proposal: &Proposal) -> Result<(), String> {
    if !proposal.id.starts_with("MIP-") {
        return Err("Proposal ID must start with 'MIP-'".into());
    }
    if !proposal.author.starts_with("did:") {
        return Err("Author must be a valid DID".into());
    }
    if proposal.title.is_empty() {
        return Err("Proposal title cannot be empty".into());
    }
    if proposal.description.is_empty() {
        return Err("Proposal description cannot be empty".into());
    }
    if proposal.voting_ends <= proposal.voting_starts {
        return Err("Voting end must be after voting start".into());
    }
    if serde_json::from_str::<serde_json::Value>(&proposal.actions).is_err() {
        return Err("Actions must be valid JSON".into());
    }
    if proposal.version != 1 {
        return Err("Initial proposal version must be 1".into());
    }
    if proposal.status != ProposalStatus::Draft {
        return Err("New proposals must start in Draft status".into());
    }
    Ok(())
}

pub fn check_update_proposal(original: &Proposal, updated: &Proposal) -> Result<(), String> {
    if updated.id != original.id {
        return Err("Cannot change proposal ID".into());
    }
    if updated.author != original.author {
        return Err("Cannot change proposal author".into());
    }
    if updated.status != original.status {
        let valid = matches!(
            (&original.status, &updated.status),
            (ProposalStatus::Draft, ProposalStatus::Active)
            | (ProposalStatus::Draft, ProposalStatus::Cancelled)
            | (ProposalStatus::Active, ProposalStatus::Ended)
            | (ProposalStatus::Active, ProposalStatus::Cancelled)
            | (ProposalStatus::Ended, ProposalStatus::Approved)
            | (ProposalStatus::Ended, ProposalStatus::Rejected)
            | (ProposalStatus::Approved, ProposalStatus::Signed)
            | (ProposalStatus::Approved, ProposalStatus::Cancelled)
            | (ProposalStatus::Signed, ProposalStatus::Executed)
            | (ProposalStatus::Signed, ProposalStatus::Failed)
            | (ProposalStatus::Signed, ProposalStatus::Cancelled)
        );
        if !valid {
            return Err(format!(
                "Invalid status transition: {:?} -> {:?}",
                original.status, updated.status
            ));
        }
    }
    if original.status != ProposalStatus::Draft
        && (updated.title != original.title
            || updated.description != original.description
            || updated.actions != original.actions
            || updated.proposal_type != original.proposal_type)
    {
        return Err("Cannot modify proposal content after leaving Draft status".into());
    }
    if updated.version != original.version + 1 {
        return Err("Version must be incremented by 1".into());
    }
    Ok(())
}

pub fn check_create_amendment(amendment: &ProposalAmendment) -> Result<(), String> {
    if !amendment.proposer.starts_with("did:") {
        return Err("Proposer must be a valid DID".into());
    }
    if amendment.changes.is_empty() {
        return Err("Amendment changes cannot be empty".into());
    }
    if amendment.reason.is_empty() {
        return Err("Amendment reason cannot be empty".into());
    }
    Ok(())
}

pub fn check_create_contribution(contribution: &DiscussionContribution) -> Result<(), String> {
    if !contribution.contributor.starts_with("did:") {
        return Err("Contributor must be a valid DID".into());
    }
    if contribution.proposal_id.is_empty() {
        return Err("Proposal ID cannot be empty".into());
    }
    if contribution.content.is_empty() {
        return Err("Contribution content cannot be empty".into());
    }
    if contribution.harmony_tags.len() > 7 {
        return Err("Cannot have more than 7 harmony tags".into());
    }
    Ok(())
}

pub fn check_create_reflection(reflection: &DiscussionReflection) -> Result<(), String> {
    if reflection.proposal_id.is_empty() {
        return Err("Proposal ID cannot be empty".into());
    }
    if reflection.harmony_diversity < 0.0 || reflection.harmony_diversity > 1.0 {
        return Err("Harmony diversity must be between 0 and 1".into());
    }
    if reflection.voice_concentration < 0.0 || reflection.voice_concentration > 1.0 {
        return Err("Voice concentration must be between 0 and 1".into());
    }
    if reflection.preliminary_sentiment < 0.0 || reflection.preliminary_sentiment > 1.0 {
        return Err("Preliminary sentiment must be between 0 and 1".into());
    }
    Ok(())
}

// ============================================================================
// VALIDATION CALLBACK
// ============================================================================

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Proposal(proposal) => validate_create_proposal(action, proposal),
                EntryTypes::ProposalAmendment(amendment) => validate_create_amendment(action, amendment),
                EntryTypes::DiscussionContribution(contribution) => validate_create_contribution(action, contribution),
                EntryTypes::DiscussionReflection(reflection) => validate_create_discussion_reflection(action, reflection),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Proposal(proposal) => {
                    validate_update_proposal(action, proposal, original_action_hash)
                }
                EntryTypes::ProposalAmendment(amendment) => {
                    validate_update_amendment(action, amendment)
                }
                EntryTypes::DiscussionContribution(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::DiscussionReflection(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AuthorToProposal => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TypeToProposal => Ok(ValidateCallbackResult::Valid),
            LinkTypes::StatusToProposal => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToAmendment => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveProposals => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToContribution => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ContributorToContribution => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ContributionToReply => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToDiscussionReflection => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalById => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            // Allow deleting status links when status changes
            LinkTypes::StatusToProposal => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveProposals => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToContribution => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ContributorToContribution => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ContributionToReply => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToDiscussionReflection => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate proposal creation
fn validate_create_proposal(
    _action: Create,
    proposal: Proposal,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_proposal(&proposal) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

/// Validate proposal update
fn validate_update_proposal(
    _action: Update,
    proposal: Proposal,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_proposal: Proposal = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original proposal not found".into()
        )))?;

    match check_update_proposal(&original_proposal, &proposal) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

/// Validate amendment creation
fn validate_create_amendment(
    _action: Create,
    amendment: ProposalAmendment,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_amendment(&amendment) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

/// Validate amendment update
fn validate_update_amendment(
    _action: Update,
    _amendment: ProposalAmendment,
) -> ExternResult<ValidateCallbackResult> {
    // Amendments can be updated (e.g., status changes)
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// DISCUSSION VALIDATION
// ============================================================================

/// Validate discussion contribution creation
fn validate_create_contribution(
    _action: Create,
    contribution: DiscussionContribution,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_contribution(&contribution) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

/// Validate discussion reflection creation
fn validate_create_discussion_reflection(
    _action: Create,
    reflection: DiscussionReflection,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_reflection(&reflection) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn make_proposal() -> Proposal {
        Proposal {
            id: "MIP-001".into(),
            title: "Test Proposal".into(),
            description: "A test proposal".into(),
            proposal_type: ProposalType::Standard,
            author: "did:mycelix:test123".into(),
            status: ProposalStatus::Draft,
            actions: "{}".into(),
            discussion_url: None,
            voting_starts: ts(1000000),
            voting_ends: ts(2000000),
            created: ts(1000000),
            updated: ts(1000000),
            version: 1,
        }
    }

    #[test]
    fn test_valid_proposal_accepted() {
        assert!(check_create_proposal(&make_proposal()).is_ok());
    }

    #[test]
    fn test_proposal_id_must_start_with_mip() {
        let mut p = make_proposal();
        p.id = "BAD-001".into();
        assert!(check_create_proposal(&p).unwrap_err().contains("MIP-"));
    }

    #[test]
    fn test_proposal_author_must_be_did() {
        let mut p = make_proposal();
        p.author = "not-a-did".into();
        assert!(check_create_proposal(&p).unwrap_err().contains("DID"));
    }

    #[test]
    fn test_proposal_title_not_empty() {
        let mut p = make_proposal();
        p.title = "".into();
        assert!(check_create_proposal(&p).unwrap_err().contains("title"));
    }

    #[test]
    fn test_proposal_description_not_empty() {
        let mut p = make_proposal();
        p.description = "".into();
        assert!(check_create_proposal(&p).unwrap_err().contains("description"));
    }

    #[test]
    fn test_proposal_voting_period_end_after_start() {
        let mut p = make_proposal();
        p.voting_ends = ts(500000); // before start
        assert!(check_create_proposal(&p).unwrap_err().contains("Voting end"));
    }

    #[test]
    fn test_proposal_actions_must_be_json() {
        let mut p = make_proposal();
        p.actions = "not json {{{".into();
        assert!(check_create_proposal(&p).unwrap_err().contains("JSON"));
    }

    #[test]
    fn test_proposal_initial_version_must_be_1() {
        let mut p = make_proposal();
        p.version = 5;
        assert!(check_create_proposal(&p).unwrap_err().contains("version"));
    }

    #[test]
    fn test_proposal_must_start_draft() {
        let mut p = make_proposal();
        p.status = ProposalStatus::Active;
        assert!(check_create_proposal(&p).unwrap_err().contains("Draft"));
    }

    // --- Update validation ---

    #[test]
    fn test_valid_status_transitions() {
        let original = make_proposal();
        let transitions = vec![
            (ProposalStatus::Draft, ProposalStatus::Active),
            (ProposalStatus::Draft, ProposalStatus::Cancelled),
            (ProposalStatus::Active, ProposalStatus::Ended),
            (ProposalStatus::Active, ProposalStatus::Cancelled),
            (ProposalStatus::Ended, ProposalStatus::Approved),
            (ProposalStatus::Ended, ProposalStatus::Rejected),
            (ProposalStatus::Approved, ProposalStatus::Signed),
            (ProposalStatus::Signed, ProposalStatus::Executed),
            (ProposalStatus::Signed, ProposalStatus::Failed),
        ];
        for (from, to) in transitions {
            let mut orig = original.clone();
            orig.status = from.clone();
            let mut updated = orig.clone();
            updated.status = to.clone();
            updated.version = orig.version + 1;
            assert!(check_update_proposal(&orig, &updated).is_ok(),
                "Transition {:?} -> {:?} should be valid", from, to);
        }
    }

    #[test]
    fn test_invalid_status_transition_rejected() {
        let mut orig = make_proposal();
        orig.status = ProposalStatus::Rejected;
        let mut updated = orig.clone();
        updated.status = ProposalStatus::Active;
        updated.version = orig.version + 1;
        assert!(check_update_proposal(&orig, &updated).unwrap_err().contains("Invalid status"));
    }

    #[test]
    fn test_cannot_change_proposal_id() {
        let orig = make_proposal();
        let mut updated = orig.clone();
        updated.id = "MIP-999".into();
        updated.version = orig.version + 1;
        assert!(check_update_proposal(&orig, &updated).unwrap_err().contains("proposal ID"));
    }

    #[test]
    fn test_cannot_modify_content_after_draft() {
        let mut orig = make_proposal();
        orig.status = ProposalStatus::Active;
        let mut updated = orig.clone();
        updated.title = "Changed Title".into();
        updated.version = orig.version + 1;
        assert!(check_update_proposal(&orig, &updated).unwrap_err().contains("content"));
    }

    #[test]
    fn test_version_must_increment() {
        let orig = make_proposal();
        let mut updated = orig.clone();
        updated.status = ProposalStatus::Active;
        updated.version = orig.version; // same version
        assert!(check_update_proposal(&orig, &updated).unwrap_err().contains("Version"));
    }

    // --- Amendment validation ---

    #[test]
    fn test_amendment_proposer_must_be_did() {
        let a = ProposalAmendment {
            id: "A-001".into(),
            proposal_id: "MIP-001".into(),
            changes: "some changes".into(),
            reason: "good reason".into(),
            proposer: "not-a-did".into(),
            status: AmendmentStatus::Proposed,
            created: ts(1000000),
        };
        assert!(check_create_amendment(&a).unwrap_err().contains("DID"));
    }

    #[test]
    fn test_amendment_changes_not_empty() {
        let a = ProposalAmendment {
            id: "A-001".into(),
            proposal_id: "MIP-001".into(),
            changes: "".into(),
            reason: "good reason".into(),
            proposer: "did:mycelix:test".into(),
            status: AmendmentStatus::Proposed,
            created: ts(1000000),
        };
        assert!(check_create_amendment(&a).unwrap_err().contains("changes"));
    }

    // --- Contribution validation ---

    #[test]
    fn test_contribution_harmony_tags_max_7() {
        let c = DiscussionContribution {
            id: "C-001".into(),
            proposal_id: "MIP-001".into(),
            contributor: "did:mycelix:test".into(),
            content: "Some content".into(),
            harmony_tags: vec!["a".into(), "b".into(), "c".into(), "d".into(),
                               "e".into(), "f".into(), "g".into(), "h".into()],
            stance: None,
            parent_id: None,
            created_at: ts(1000000),
            edited: false,
        };
        assert!(check_create_contribution(&c).unwrap_err().contains("7"));
    }

    // --- Reflection validation ---

    #[test]
    fn test_reflection_harmony_diversity_range() {
        let r = DiscussionReflection {
            id: "R-001".into(),
            proposal_id: "MIP-001".into(),
            timestamp: ts(1000000),
            contributor_count: 5,
            contribution_count: 10,
            avg_contributions_per_participant: 2.0,
            max_thread_depth: 3,
            harmony_coverage: vec![],
            harmony_diversity: 1.5, // out of range
            absent_harmonies: vec![],
            support_count: 3,
            oppose_count: 2,
            neutral_count: 1,
            amend_count: 1,
            preliminary_sentiment: 0.5,
            voice_concentration: 0.3,
            cross_camp_engagement: 0.5,
            substantiveness_score: 0.7,
            discussion_saturated: false,
            unaddressed_concerns: vec![],
            ready_for_vote: false,
            readiness_reasoning: "Not ready".into(),
            summary: "Test reflection".into(),
        };
        assert!(check_create_reflection(&r).unwrap_err().contains("Harmony diversity"));
    }
}
