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
}

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
    // Validate ID format (MIP-XXX)
    if !proposal.id.starts_with("MIP-") {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID must start with 'MIP-'".into(),
        ));
    }

    // Validate author is a DID
    if !proposal.author.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Author must be a valid DID".into(),
        ));
    }

    // Validate title not empty
    if proposal.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal title cannot be empty".into(),
        ));
    }

    // Validate description not empty
    if proposal.description.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal description cannot be empty".into(),
        ));
    }

    // Validate voting period
    if proposal.voting_ends <= proposal.voting_starts {
        return Ok(ValidateCallbackResult::Invalid(
            "Voting end must be after voting start".into(),
        ));
    }

    // Validate actions is valid JSON
    if serde_json::from_str::<serde_json::Value>(&proposal.actions).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Actions must be valid JSON".into(),
        ));
    }

    // Validate initial version is 1
    if proposal.version != 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial proposal version must be 1".into(),
        ));
    }

    // New proposals must start in Draft status
    if proposal.status != ProposalStatus::Draft {
        return Ok(ValidateCallbackResult::Invalid(
            "New proposals must start in Draft status".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate proposal update
fn validate_update_proposal(
    _action: Update,
    proposal: Proposal,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Get original proposal for comparison
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_proposal: Proposal = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original proposal not found".into()
        )))?;

    // Cannot change proposal ID
    if proposal.id != original_proposal.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change proposal ID".into(),
        ));
    }

    // Cannot change author
    if proposal.author != original_proposal.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change proposal author".into(),
        ));
    }

    // Enforce valid status transitions
    if proposal.status != original_proposal.status {
        let valid = matches!(
            (&original_proposal.status, &proposal.status),
            // Draft: submit for voting or cancel
            (ProposalStatus::Draft, ProposalStatus::Active)
            | (ProposalStatus::Draft, ProposalStatus::Cancelled)
            // Active: end voting period or cancel
            | (ProposalStatus::Active, ProposalStatus::Ended)
            | (ProposalStatus::Active, ProposalStatus::Cancelled)
            // Ended: tally determines outcome
            | (ProposalStatus::Ended, ProposalStatus::Approved)
            | (ProposalStatus::Ended, ProposalStatus::Rejected)
            // Approved: threshold signature obtained, or cancel
            | (ProposalStatus::Approved, ProposalStatus::Signed)
            | (ProposalStatus::Approved, ProposalStatus::Cancelled)
            // Signed: execution outcome
            | (ProposalStatus::Signed, ProposalStatus::Executed)
            | (ProposalStatus::Signed, ProposalStatus::Failed)
            | (ProposalStatus::Signed, ProposalStatus::Cancelled)
        );

        if !valid {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid status transition: {:?} -> {:?}",
                original_proposal.status, proposal.status
            )));
        }
    }

    // Cannot modify content fields after voting starts (only status changes allowed)
    if original_proposal.status != ProposalStatus::Draft
        && (proposal.title != original_proposal.title
            || proposal.description != original_proposal.description
            || proposal.actions != original_proposal.actions
            || proposal.proposal_type != original_proposal.proposal_type)
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot modify proposal content after leaving Draft status".into(),
        ));
    }

    // Version must increment
    if proposal.version != original_proposal.version + 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Version must be incremented by 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate amendment creation
fn validate_create_amendment(
    _action: Create,
    amendment: ProposalAmendment,
) -> ExternResult<ValidateCallbackResult> {
    // Validate proposer is a DID
    if !amendment.proposer.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposer must be a valid DID".into(),
        ));
    }

    // Validate changes not empty
    if amendment.changes.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Amendment changes cannot be empty".into(),
        ));
    }

    // Validate reason not empty
    if amendment.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Amendment reason cannot be empty".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
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
    // Validate contributor is a DID
    if !contribution.contributor.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Contributor must be a valid DID".into(),
        ));
    }

    // Validate proposal ID not empty
    if contribution.proposal_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID cannot be empty".into(),
        ));
    }

    // Validate content not empty
    if contribution.content.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Contribution content cannot be empty".into(),
        ));
    }

    // Validate harmony tags (max 7 - the Seven Harmonies)
    if contribution.harmony_tags.len() > 7 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot have more than 7 harmony tags".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate discussion reflection creation
fn validate_create_discussion_reflection(
    _action: Create,
    reflection: DiscussionReflection,
) -> ExternResult<ValidateCallbackResult> {
    // Validate proposal ID not empty
    if reflection.proposal_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID cannot be empty".into(),
        ));
    }

    // Validate harmony diversity in range
    if reflection.harmony_diversity < 0.0 || reflection.harmony_diversity > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Harmony diversity must be between 0 and 1".into(),
        ));
    }

    // Validate voice concentration in range
    if reflection.voice_concentration < 0.0 || reflection.voice_concentration > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Voice concentration must be between 0 and 1".into(),
        ));
    }

    // Validate preliminary sentiment in range
    if reflection.preliminary_sentiment < 0.0 || reflection.preliminary_sentiment > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Preliminary sentiment must be between 0 and 1".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
