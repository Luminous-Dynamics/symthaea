use hdi::prelude::*;

/// Dispute entry - represents a disputed transaction
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Dispute {
    /// Transaction being disputed
    pub transaction_hash: ActionHash,

    /// Agent who filed the dispute
    pub filed_by: AgentPubKey,

    /// Buyer in the transaction
    pub buyer: AgentPubKey,

    /// Seller in the transaction
    pub seller: AgentPubKey,

    /// Reason for dispute
    pub reason: String,

    /// IPFS CIDs of evidence (photos, documents, etc.)
    pub evidence_cids: Vec<String>,

    /// Current status
    pub status: DisputeStatus,

    /// Assigned arbitrators (high MATL score agents)
    pub arbitrators: Vec<AgentPubKey>,

    /// Creation timestamp
    pub created_at: Timestamp,

    /// Last update timestamp
    pub updated_at: Timestamp,
}

/// Dispute status lifecycle
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DisputeStatus {
    /// Newly filed, awaiting arbitrator assignment
    Filed,

    /// Arbitrators assigned, awaiting votes
    UnderReview,

    /// All votes collected, awaiting finalization
    Voting,

    /// Resolved in favor of buyer
    ResolvedBuyer,

    /// Resolved in favor of seller
    ResolvedSeller,

    /// Withdrawn by filer
    Withdrawn,
}

/// Arbitration vote - an arbitrator's decision
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ArbitrationVote {
    /// Dispute being voted on
    pub dispute_hash: ActionHash,

    /// Arbitrator who voted
    pub arbitrator: AgentPubKey,

    /// Decision (true = favor buyer, false = favor seller)
    pub favor_buyer: bool,

    /// Reasoning for decision
    pub reasoning: String,

    /// Arbitrator's MATL score at time of vote (for transparency)
    pub arbitrator_matl_score: f64,

    /// Timestamp
    pub voted_at: Timestamp,
}

/// Arbitration result - final outcome with compensation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ArbitrationResult {
    /// Dispute that was resolved
    pub dispute_hash: ActionHash,

    /// Winner (buyer or seller)
    pub winner: AgentPubKey,

    /// Loser
    pub loser: AgentPubKey,

    /// Weighted vote score (0.0-1.0, >0.66 = buyer wins)
    pub weighted_vote: f64,

    /// Total votes cast
    pub total_votes: u32,

    /// Optional compensation amount (in cents)
    pub compensation_cents: Option<u64>,

    /// Resolution summary
    pub summary: String,

    /// Timestamp
    pub finalized_at: Timestamp,
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Transaction -> Dispute
    TransactionToDispute,

    /// Agent -> Disputes (as filer)
    AgentToFiledDisputes,

    /// Agent -> Disputes (as arbitrator)
    AgentToArbitrationOpportunities,

    /// Dispute -> Votes
    DisputeToVotes,

    /// Dispute -> Result
    DisputeToResult,

    /// All Disputes (for browsing)
    AllDisputes,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Dispute(Dispute),
    ArbitrationVote(ArbitrationVote),
    ArbitrationResult(ArbitrationResult),
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Dispute(dispute) => validate_dispute(&dispute),
                EntryTypes::ArbitrationVote(vote) => validate_vote(&vote),
                EntryTypes::ArbitrationResult(result) => validate_result(&result),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_dispute(dispute: &Dispute) -> ExternResult<ValidateCallbackResult> {
    // Reason must not be empty
    if dispute.reason.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute reason cannot be empty".into(),
        ));
    }

    // Reason length limit
    if dispute.reason.len() > 5000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Dispute reason too long (max 5000 characters)".into(),
        ));
    }

    // Filer must be either buyer or seller
    if dispute.filed_by != dispute.buyer && dispute.filed_by != dispute.seller {
        return Ok(ValidateCallbackResult::Invalid(
            "Only buyer or seller can file dispute".into(),
        ));
    }

    // Buyer and seller must be different
    if dispute.buyer == dispute.seller {
        return Ok(ValidateCallbackResult::Invalid(
            "Buyer and seller must be different".into(),
        ));
    }

    // Evidence CIDs validation
    for cid in &dispute.evidence_cids {
        if !is_valid_ipfs_cid(cid) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid IPFS CID: {}",
                cid
            )));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_vote(vote: &ArbitrationVote) -> ExternResult<ValidateCallbackResult> {
    // Reasoning must not be empty
    if vote.reasoning.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote reasoning cannot be empty".into(),
        ));
    }

    // Reasoning length limit
    if vote.reasoning.len() > 2000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote reasoning too long (max 2000 characters)".into(),
        ));
    }

    // MATL score must be valid
    if vote.arbitrator_matl_score < 0.0 || vote.arbitrator_matl_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid MATL score (must be 0.0-1.0)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_result(result: &ArbitrationResult) -> ExternResult<ValidateCallbackResult> {
    // Weighted vote must be valid
    if result.weighted_vote < 0.0 || result.weighted_vote > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid weighted vote (must be 0.0-1.0)".into(),
        ));
    }

    // Must have at least one vote
    if result.total_votes == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Must have at least one vote".into(),
        ));
    }

    // Winner must be either buyer or seller
    // (This would require access to the dispute entry, so we'll skip for now)

    Ok(ValidateCallbackResult::Valid)
}

/// Validate IPFS CID format (simplified)
fn is_valid_ipfs_cid(cid: &str) -> bool {
    // Basic validation: starts with Qm or b (CIDv0 or CIDv1)
    // and has reasonable length
    (cid.starts_with("Qm") && cid.len() == 46)
        || (cid.starts_with('b') && cid.len() >= 50 && cid.len() <= 100)
}
