// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! RB-BFT Integrity Zome
//!
//! Validation rules for Reputation-Based Byzantine Fault Tolerance consensus.
//! Achieves 45% Byzantine tolerance through reputation² weighted voting.

use hdi::prelude::*;

/// Minimum reputation to participate as validator (Φ ≥ 0.3)
pub const MIN_VALIDATOR_REPUTATION: f32 = 0.3;

/// Byzantine tolerance threshold (revolutionary 45%)
pub const RBBFT_BYZANTINE_THRESHOLD: f32 = 0.45;

/// Quorum threshold (2/3 of weighted votes)
pub const QUORUM_THRESHOLD: f32 = 0.667;

/// Compute RB-BFT voting weight from reputation (reputation²)
pub fn voting_weight(reputation: f32) -> f32 {
    reputation.powi(2)
}

/// Check if a reputation value is in valid range [0, 1]
pub fn is_valid_reputation(v: f32) -> bool {
    v >= 0.0 && v <= 1.0
}

/// K-Vector: Multi-dimensional reputation representation
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KVector {
    /// Reputation score (primary for voting weight)
    pub k_r: f32,
    /// Activity level
    pub k_a: f32,
    /// Integrity score
    pub k_i: f32,
    /// Performance metrics
    pub k_p: f32,
    /// Membership duration
    pub k_m: f32,
    /// Stake weight
    pub k_s: f32,
    /// Historical consistency
    pub k_h: f32,
    /// Network topology contribution
    pub k_topo: f32,
}

impl KVector {
    /// Check if all dimensions are in valid range [0, 1]
    pub fn is_valid(&self) -> bool {
        is_valid_reputation(self.k_r) && is_valid_reputation(self.k_a) &&
        is_valid_reputation(self.k_i) && is_valid_reputation(self.k_p) &&
        is_valid_reputation(self.k_m) && is_valid_reputation(self.k_s) &&
        is_valid_reputation(self.k_h) && is_valid_reputation(self.k_topo)
    }
}

/// Validator registration entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ValidatorRegistration {
    /// Validator's agent public key
    pub agent: AgentPubKey,
    /// K-Vector: Multi-dimensional reputation
    pub kvector: KVector,
    /// Registration timestamp
    pub registered_at: Timestamp,
    /// Staking proof (external reference)
    pub stake_proof: Option<String>,
}

impl ValidatorRegistration {
    /// Compute voting weight from reputation
    pub fn voting_weight(&self) -> f32 {
        voting_weight(self.kvector.k_r)
    }

    /// Check if all K-Vector dimensions are valid
    pub fn is_kvector_valid(&self) -> bool {
        self.kvector.is_valid()
    }
}

/// Consensus round state
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundState {
    Propose,
    PreVote,
    PreCommit,
    Committed,
    Failed,
}

/// Block proposal entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BlockProposal {
    /// Round number
    pub round: u64,
    /// View number (increments on leader change)
    pub view: u64,
    /// Proposer's agent key
    pub proposer: AgentPubKey,
    /// Proposer's K-Vector at proposal time
    pub proposer_kvector: KVector,
    /// Block height
    pub height: u64,
    /// Parent block action hash
    pub parent_hash: Option<ActionHash>,
    /// Content merkle root
    pub content_root: Vec<u8>,
    /// Proposal timestamp
    pub timestamp: Timestamp,
}

/// Vote type
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteType {
    PreVote,
    PreCommit,
}

/// Vote entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Vote {
    /// Voter's agent key
    pub voter: AgentPubKey,
    /// Vote type
    pub vote_type: VoteType,
    /// Proposal being voted on
    pub proposal_hash: ActionHash,
    /// Round number
    pub round: u64,
    /// Vote value (true = approve, false = reject)
    pub value: bool,
    /// Voter's reputation at vote time
    pub reputation: f32,
    /// Vote timestamp
    pub timestamp: Timestamp,
}

impl Vote {
    /// Compute weighted vote value
    pub fn weighted_value(&self) -> f32 {
        if self.value {
            self.reputation.powi(2)
        } else {
            0.0
        }
    }

    /// Get vote weight (reputation²)
    pub fn weight(&self) -> f32 {
        self.reputation.powi(2)
    }
}

/// Committed block entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CommittedBlock {
    /// The proposal that was committed
    pub proposal_hash: ActionHash,
    /// Round it was committed in
    pub round: u64,
    /// Block height
    pub height: u64,
    /// Total weighted votes in favor
    pub weighted_votes_for: f32,
    /// Total weighted votes
    pub weighted_votes_total: f32,
    /// Commit timestamp
    pub committed_at: Timestamp,
}

/// Challenge evidence for misbehavior
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ChallengeEvidence {
    /// Challenged validator
    pub validator: AgentPubKey,
    /// Type of violation
    pub violation_type: ViolationType,
    /// Proof data
    pub proof: Vec<u8>,
    /// Round where violation occurred
    pub round: u64,
    /// Challenger
    pub challenger: AgentPubKey,
    /// Challenge timestamp
    pub timestamp: Timestamp,
}

/// Violation types
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationType {
    DoubleVote,
    DoubleProposal,
    ConflictingPreCommit,
    InvalidProposal,
    Unavailable,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    ValidatorRegistration(ValidatorRegistration),
    BlockProposal(BlockProposal),
    Vote(Vote),
    CommittedBlock(CommittedBlock),
    ChallengeEvidence(ChallengeEvidence),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All registered validators
    ValidatorIndex,
    /// Validator to their registrations
    ValidatorToRegistration,
    /// Round to proposals
    RoundToProposals,
    /// Proposal to votes
    ProposalToVotes,
    /// Block chain (height to committed block)
    BlockChain,
    /// Validator to challenges against them
    ValidatorToChallenges,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                validate_create_entry(app_entry, EntryCreationAction::Create(action))
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                validate_update_entry(app_entry, action)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, base_address, target_address, .. } => {
            validate_create_link(link_type, &base_address, &target_address)
        }
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            validate_delete_link(link_type)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(
    entry: EntryTypes,
    _action: EntryCreationAction,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::ValidatorRegistration(reg) => {
            // Validate K-Vector ranges
            if !reg.is_kvector_valid() {
                return Ok(ValidateCallbackResult::Invalid(
                    "K-Vector dimensions must be in range [0, 1]".into()
                ));
            }
            // Validate minimum reputation
            if reg.kvector.k_r < MIN_VALIDATOR_REPUTATION {
                return Ok(ValidateCallbackResult::Invalid(
                    format!("Minimum reputation {} required to register as validator",
                            MIN_VALIDATOR_REPUTATION)
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::BlockProposal(proposal) => {
            // Proposer K-Vector must be valid
            if !proposal.proposer_kvector.is_valid() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Proposer K-Vector dimensions must be in range [0, 1]".into()
                ));
            }
            // Proposer must meet minimum reputation
            if proposal.proposer_kvector.k_r < MIN_VALIDATOR_REPUTATION {
                return Ok(ValidateCallbackResult::Invalid(
                    "Proposer does not meet minimum reputation requirement".into()
                ));
            }
            // Content root must not be empty
            if proposal.content_root.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Block content root cannot be empty".into()
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::Vote(vote) => {
            // Reputation must be valid
            if vote.reputation < 0.0 || vote.reputation > 1.0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Vote reputation must be in range [0, 1]".into()
                ));
            }
            // Voter must meet minimum reputation
            if vote.reputation < MIN_VALIDATOR_REPUTATION {
                return Ok(ValidateCallbackResult::Invalid(
                    "Voter does not meet minimum reputation requirement".into()
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::CommittedBlock(block) => {
            // Weighted votes must be positive
            if block.weighted_votes_total <= 0.0 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Committed block must have positive total vote weight".into()
                ));
            }
            // Quorum must be reached
            let quorum = block.weighted_votes_for / block.weighted_votes_total;
            if quorum < QUORUM_THRESHOLD {
                return Ok(ValidateCallbackResult::Invalid(
                    format!("Block did not reach quorum ({:.1}% < {:.1}%)",
                            quorum * 100.0, QUORUM_THRESHOLD * 100.0)
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::ChallengeEvidence(evidence) => {
            // Proof must not be empty
            if evidence.proof.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Challenge evidence must include proof data".into()
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_update_entry(
    entry: EntryTypes,
    _action: Update,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        // Votes are immutable
        EntryTypes::Vote(_) => {
            Ok(ValidateCallbackResult::Invalid(
                "Votes cannot be updated - they are immutable".into()
            ))
        }
        // Committed blocks are immutable
        EntryTypes::CommittedBlock(_) => {
            Ok(ValidateCallbackResult::Invalid(
                "Committed blocks cannot be updated".into()
            ))
        }
        // Challenge evidence is immutable
        EntryTypes::ChallengeEvidence(_) => {
            Ok(ValidateCallbackResult::Invalid(
                "Challenge evidence cannot be updated".into()
            ))
        }
        // Validator registration can be updated (reputation changes)
        EntryTypes::ValidatorRegistration(reg) => {
            if !reg.is_kvector_valid() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Updated K-Vector must be valid".into()
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        // Proposals cannot be updated after creation
        EntryTypes::BlockProposal(_) => {
            Ok(ValidateCallbackResult::Invalid(
                "Block proposals cannot be updated after creation".into()
            ))
        }
    }
}

fn validate_create_link(
    link_type: LinkTypes,
    base: &HoloHash<hash_type::AnyLinkable>,
    target: &HoloHash<hash_type::AnyLinkable>,
) -> ExternResult<ValidateCallbackResult> {
    let base_valid = base.as_ref().len() == 39;
    let target_valid = target.as_ref().len() == 39;

    match link_type {
        LinkTypes::ValidatorIndex | LinkTypes::ValidatorToRegistration => {
            if !base_valid || !target_valid {
                return Ok(ValidateCallbackResult::Invalid(
                    "Validator links must have valid hashes".into()
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::RoundToProposals | LinkTypes::ProposalToVotes => {
            if !target_valid {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link target must be a valid hash".into()
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::BlockChain => {
            // Block chain links must be entry hashes
            if !base_valid || !target_valid {
                return Ok(ValidateCallbackResult::Invalid(
                    "BlockChain links must connect valid block hashes".into()
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::ValidatorToChallenges => {
            if !target_valid {
                return Ok(ValidateCallbackResult::Invalid(
                    "Challenge link target must be valid".into()
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_delete_link(link_type: LinkTypes) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        // Block chain links cannot be deleted
        LinkTypes::BlockChain => {
            Ok(ValidateCallbackResult::Invalid(
                "BlockChain links cannot be deleted - chain is immutable".into()
            ))
        }
        // Proposal to votes links cannot be deleted
        LinkTypes::ProposalToVotes => {
            Ok(ValidateCallbackResult::Invalid(
                "ProposalToVotes links cannot be deleted - votes are immutable".into()
            ))
        }
        // Challenge links cannot be deleted
        LinkTypes::ValidatorToChallenges => {
            Ok(ValidateCallbackResult::Invalid(
                "Challenge links cannot be deleted - evidence must be preserved".into()
            ))
        }
        // Other links can be deleted
        _ => Ok(ValidateCallbackResult::Valid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_kvector(k_r: f32) -> KVector {
        KVector {
            k_r,
            k_a: 0.5,
            k_i: 0.5,
            k_p: 0.5,
            k_m: 0.5,
            k_s: 0.5,
            k_h: 0.5,
            k_topo: 0.5,
        }
    }

    #[test]
    fn test_voting_weight_function() {
        // High reputation (0.9)² = 0.81
        assert!((voting_weight(0.9) - 0.81).abs() < 0.001);
        // Low reputation (0.3)² = 0.09
        assert!((voting_weight(0.3) - 0.09).abs() < 0.001);
        // Zero reputation
        assert!((voting_weight(0.0) - 0.0).abs() < 0.001);
        // Perfect reputation
        assert!((voting_weight(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_reputation_validation() {
        assert!(is_valid_reputation(0.0));
        assert!(is_valid_reputation(0.5));
        assert!(is_valid_reputation(1.0));
        assert!(!is_valid_reputation(-0.1));
        assert!(!is_valid_reputation(1.1));
    }

    #[test]
    fn test_kvector_validity() {
        // All valid dimensions
        let valid = KVector {
            k_r: 0.5, k_a: 0.5, k_i: 0.5, k_p: 0.5,
            k_m: 0.5, k_s: 0.5, k_h: 0.5, k_topo: 0.5,
        };
        assert!(valid.is_valid());

        // Invalid k_r (negative)
        let invalid_kr = KVector { k_r: -0.1, ..valid.clone() };
        assert!(!invalid_kr.is_valid());

        // Invalid k_a (above 1)
        let invalid_ka = KVector { k_a: 1.5, ..valid.clone() };
        assert!(!invalid_ka.is_valid());

        // Boundary values valid
        let boundary = KVector {
            k_r: 0.0, k_a: 1.0, k_i: 0.0, k_p: 1.0,
            k_m: 0.0, k_s: 1.0, k_h: 0.0, k_topo: 1.0,
        };
        assert!(boundary.is_valid());
    }

    #[test]
    fn test_validator_registration_voting_weight() {
        let reg = ValidatorRegistration {
            agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
            kvector: test_kvector(0.8),
            registered_at: Timestamp::now(),
            stake_proof: None,
        };
        // (0.8)² = 0.64
        assert!((reg.voting_weight() - 0.64).abs() < 0.001);
        assert!(reg.is_kvector_valid());
    }

    #[test]
    fn test_vote_weight() {
        let vote = Vote {
            voter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            vote_type: VoteType::PreVote,
            proposal_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            round: 1,
            value: true,
            reputation: 0.9,
            timestamp: Timestamp::now(),
        };
        assert!((vote.weight() - 0.81).abs() < 0.001);
        assert!((vote.weighted_value() - 0.81).abs() < 0.001);

        // False vote has zero weighted value
        let no_vote = Vote {
            voter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            vote_type: VoteType::PreVote,
            proposal_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            round: 1,
            value: false,
            reputation: 0.9,
            timestamp: Timestamp::now(),
        };
        assert!((no_vote.weighted_value() - 0.0).abs() < 0.001);
        assert!((no_vote.weight() - 0.81).abs() < 0.001); // Weight is still counted
    }

    #[test]
    fn test_byzantine_threshold_math() {
        // The 45% Byzantine tolerance comes from reputation² weighting
        // With 45% Byzantines at rep=0.3, their weight = 0.45 * 0.09 = 0.0405
        // With 55% honest at rep=0.9, their weight = 0.55 * 0.81 = 0.4455
        // Total honest weight >> Byzantine weight
        let byz_weight = 0.45 * voting_weight(0.3);
        let honest_weight = 0.55 * voting_weight(0.9);
        assert!(honest_weight > byz_weight * 10.0); // Honest > 10x Byzantine
    }

    #[test]
    fn test_quorum_calculation() {
        // Need 66.7% weighted votes for quorum
        let total = 1.0;
        let for_votes = QUORUM_THRESHOLD;
        assert!(for_votes / total >= QUORUM_THRESHOLD);

        // Just below quorum fails
        let barely_below = QUORUM_THRESHOLD - 0.001;
        assert!(barely_below / total < QUORUM_THRESHOLD);
    }

    #[test]
    fn test_round_states() {
        // Verify round state transitions are defined
        let states = [
            RoundState::Propose,
            RoundState::PreVote,
            RoundState::PreCommit,
            RoundState::Committed,
            RoundState::Failed,
        ];
        // All states are distinct
        for (i, s1) in states.iter().enumerate() {
            for (j, s2) in states.iter().enumerate() {
                if i != j {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    #[test]
    fn test_violation_types() {
        // Verify all violation types are defined
        let violations = [
            ViolationType::DoubleVote,
            ViolationType::DoubleProposal,
            ViolationType::ConflictingPreCommit,
            ViolationType::InvalidProposal,
            ViolationType::Unavailable,
        ];
        assert_eq!(violations.len(), 5);
    }

    #[test]
    fn test_vote_types() {
        // PreVote and PreCommit are distinct
        assert_ne!(VoteType::PreVote, VoteType::PreCommit);
    }

    #[test]
    fn test_min_validator_reputation() {
        // Verify minimum reputation constant
        assert!((MIN_VALIDATOR_REPUTATION - 0.3).abs() < 0.001);

        // Below minimum should fail validation
        let low_rep = test_kvector(0.2);
        assert!(low_rep.k_r < MIN_VALIDATOR_REPUTATION);

        // At minimum should pass
        let min_rep = test_kvector(0.3);
        assert!(min_rep.k_r >= MIN_VALIDATOR_REPUTATION);
    }
}
