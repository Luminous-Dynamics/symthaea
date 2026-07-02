// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

    /// Anchor -> registered arbitrator pool member
    AllArbitrators,
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
    // An arbitrator must not be a party to the dispute they're voting on
    // (buyer, seller, or the agent who filed it). Enforced here, not just
    // in coordinator code, so it can't be bypassed by a direct entry write.
    let dispute_record = must_get_valid_record(vote.dispute_hash.clone())?;
    let dispute: Dispute = dispute_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(e))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Dispute entry missing for vote".into()
        )))?;
    if vote.arbitrator == dispute.buyer
        || vote.arbitrator == dispute.seller
        || vote.arbitrator == dispute.filed_by
    {
        return Ok(ValidateCallbackResult::Invalid(
            "Arbitrator cannot be a party to the dispute (self-dealing)".into(),
        ));
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Helpers =====

    fn mock_agent(byte: u8) -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![byte; 36])
    }

    fn valid_dispute() -> Dispute {
        Dispute {
            transaction_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            filed_by: mock_agent(2),
            buyer: mock_agent(2),
            seller: mock_agent(3),
            reason: "Item not as described".to_string(),
            evidence_cids: vec![],
            status: DisputeStatus::Filed,
            arbitrators: vec![],
            created_at: Timestamp::from_micros(1000000),
            updated_at: Timestamp::from_micros(1000000),
        }
    }

    fn valid_vote() -> ArbitrationVote {
        ArbitrationVote {
            dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            arbitrator: mock_agent(4),
            favor_buyer: true,
            reasoning: "Evidence supports buyer claim".to_string(),
            arbitrator_matl_score: 0.85,
            voted_at: Timestamp::from_micros(2000000),
        }
    }

    fn valid_result() -> ArbitrationResult {
        ArbitrationResult {
            dispute_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            winner: mock_agent(2),
            loser: mock_agent(3),
            weighted_vote: 0.75,
            total_votes: 3,
            compensation_cents: Some(1999),
            summary: "Resolved in favor of buyer".to_string(),
            finalized_at: Timestamp::from_micros(3000000),
        }
    }

    // ===== Dispute Validation Tests =====

    #[test]
    fn test_validate_dispute_valid() {
        let dispute = valid_dispute();
        let result = validate_dispute(&dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_dispute_empty_reason() {
        let dispute = Dispute {
            reason: String::new(),
            ..valid_dispute()
        };
        let result = validate_dispute(&dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_dispute_whitespace_only_reason() {
        let dispute = Dispute {
            reason: "   \t\n  ".to_string(),
            ..valid_dispute()
        };
        let result = validate_dispute(&dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_dispute_reason_too_long() {
        let dispute = Dispute {
            reason: "x".repeat(5001),
            ..valid_dispute()
        };
        let result = validate_dispute(&dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_dispute_filer_not_party() {
        let dispute = Dispute {
            filed_by: mock_agent(99),
            ..valid_dispute()
        };
        let result = validate_dispute(&dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_dispute_buyer_equals_seller() {
        let dispute = Dispute {
            buyer: mock_agent(3),
            seller: mock_agent(3),
            filed_by: mock_agent(3),
            ..valid_dispute()
        };
        let result = validate_dispute(&dispute).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_dispute_seller_files() {
        let dispute = Dispute {
            filed_by: mock_agent(3),
            ..valid_dispute()
        };
        let result = validate_dispute(&dispute).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ===== Vote Validation Tests =====

    #[test]
    fn test_validate_vote_valid() {
        let vote = valid_vote();
        let result = validate_vote(&vote).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_vote_empty_reasoning() {
        let vote = ArbitrationVote {
            reasoning: String::new(),
            ..valid_vote()
        };
        let result = validate_vote(&vote).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_vote_whitespace_reasoning() {
        let vote = ArbitrationVote {
            reasoning: "  \t ".to_string(),
            ..valid_vote()
        };
        let result = validate_vote(&vote).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_vote_reasoning_too_long() {
        let vote = ArbitrationVote {
            reasoning: "x".repeat(2001),
            ..valid_vote()
        };
        let result = validate_vote(&vote).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_vote_matl_below_zero() {
        let vote = ArbitrationVote {
            arbitrator_matl_score: -0.1,
            ..valid_vote()
        };
        let result = validate_vote(&vote).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_vote_matl_above_one() {
        let vote = ArbitrationVote {
            arbitrator_matl_score: 1.1,
            ..valid_vote()
        };
        let result = validate_vote(&vote).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_vote_matl_boundary_zero() {
        let vote = ArbitrationVote {
            arbitrator_matl_score: 0.0,
            ..valid_vote()
        };
        let result = validate_vote(&vote).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_vote_matl_boundary_one() {
        let vote = ArbitrationVote {
            arbitrator_matl_score: 1.0,
            ..valid_vote()
        };
        let result = validate_vote(&vote).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // ===== Result Validation Tests =====

    #[test]
    fn test_validate_result_valid() {
        let res = valid_result();
        let result = validate_result(&res).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_validate_result_weighted_vote_below_zero() {
        let res = ArbitrationResult {
            weighted_vote: -0.1,
            ..valid_result()
        };
        let result = validate_result(&res).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_result_weighted_vote_above_one() {
        let res = ArbitrationResult {
            weighted_vote: 1.1,
            ..valid_result()
        };
        let result = validate_result(&res).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_validate_result_zero_votes() {
        let res = ArbitrationResult {
            total_votes: 0,
            ..valid_result()
        };
        let result = validate_result(&res).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // ===== IPFS CID Tests =====

    #[test]
    fn test_ipfs_cid_valid_v0() {
        assert!(is_valid_ipfs_cid(
            "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG"
        ));
    }

    #[test]
    fn test_ipfs_cid_invalid_short() {
        assert!(!is_valid_ipfs_cid("QmShort"));
    }

    #[test]
    fn test_ipfs_cid_invalid_empty() {
        assert!(!is_valid_ipfs_cid(""));
    }

    #[test]
    fn test_ipfs_cid_v1_too_short() {
        assert!(!is_valid_ipfs_cid("bshort"));
    }

    // ===== Dispute Status Tests =====

    #[test]
    fn test_all_dispute_statuses() {
        let statuses = vec![
            DisputeStatus::Filed,
            DisputeStatus::UnderReview,
            DisputeStatus::Voting,
            DisputeStatus::ResolvedBuyer,
            DisputeStatus::ResolvedSeller,
            DisputeStatus::Withdrawn,
        ];
        assert_eq!(statuses.len(), 6);
    }

    // ===== Serde Tests =====

    #[test]
    fn test_dispute_serde_roundtrip() {
        let dispute = valid_dispute();
        let json = serde_json::to_string(&dispute).unwrap();
        let parsed: Dispute = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.reason, dispute.reason);
    }

    #[test]
    fn test_vote_serde_roundtrip() {
        let vote = valid_vote();
        let json = serde_json::to_string(&vote).unwrap();
        let parsed: ArbitrationVote = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.arbitrator_matl_score, vote.arbitrator_matl_score);
    }

    #[test]
    fn test_result_serde_roundtrip() {
        let res = valid_result();
        let json = serde_json::to_string(&res).unwrap();
        let parsed: ArbitrationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.weighted_vote, res.weighted_vote);
    }
}
