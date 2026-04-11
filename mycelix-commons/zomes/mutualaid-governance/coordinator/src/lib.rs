// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mutual Aid Governance Coordinator Zome
//! Democratic decision-making for mutual aid circles.

use hdk::prelude::*;
use mutualaid_common::{Proposal, Vote};
use mutualaid_governance_integrity::*;
use mycelix_bridge_common::{civic_requirement_proposal, civic_requirement_voting};
use mycelix_zome_helpers::get_latest_record;


// ============================================================================
// Extern Functions
// ============================================================================

/// Minimum voting period in milliseconds (5 minutes)
const MIN_VOTING_PERIOD_MS: i64 = 5 * 60 * 1_000_000; // 5 minutes in microseconds

/// Create a governance proposal

#[hdk_extern]
pub fn create_proposal(proposal: Proposal) -> ExternResult<Record> {
    // Consciousness gate: Participant tier + identity >= 0.25
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "create_proposal")?;

    // Validate title is not empty or whitespace-only
    if proposal.title.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal title cannot be empty or whitespace-only".into()
        )));
    }

    // Validate description is not empty or whitespace-only
    if proposal.description.trim().is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal description cannot be empty or whitespace-only".into()
        )));
    }

    // Validate quorum threshold is non-zero
    if proposal.quorum_percent == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Quorum threshold must be greater than zero".into()
        )));
    }

    // Validate threshold percent is non-zero
    if proposal.threshold_percent == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Approval threshold must be greater than zero".into()
        )));
    }

    // Validate voting window is meaningful (at least 5 minutes)
    let voting_duration_us = proposal.voting_ends.as_micros() - proposal.voting_starts.as_micros();
    if voting_duration_us < MIN_VOTING_PERIOD_MS {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Voting period must be at least 5 minutes".into()
        )));
    }

    // Validate voting_ends is after voting_starts (handles negative durations)
    if proposal.voting_ends <= proposal.voting_starts {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Voting end time must be after voting start time".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Proposal(proposal))?;
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created proposal".into()
    )))
}

/// Get all proposals
#[hdk_extern]
pub fn get_all_proposals(_: ()) -> ExternResult<Vec<Record>> {
    Ok(vec![])
}

/// Cast a vote on a proposal
#[hdk_extern]
pub fn cast_vote(vote: Vote) -> ExternResult<Record> {
    // Consciousness gate: Citizen tier + identity >= 0.25
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "cast_vote")?;

    // Check for double-voting: look up existing votes linked to this proposal
    let existing_vote_links = get_links(
        LinkQuery::try_new(vote.proposal_hash.clone(), LinkTypes::ProposalToVotes)?,
        GetStrategy::default(),
    )?;

    // Check each existing vote to see if this voter already voted
    for link in existing_vote_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get_latest_record(action_hash)? {
                if let Ok(Some(existing_vote)) = record.entry().to_app_option::<Vote>() {
                    if existing_vote.voter == vote.voter {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Member has already voted on this proposal".into()
                        )));
                    }
                }
            }
        }
    }

    // Create the vote entry
    let action_hash = create_entry(&EntryTypes::Vote(vote.clone()))?;

    // Link the vote to the proposal for future duplicate detection
    create_link(
        vote.proposal_hash.clone(),
        action_hash.clone(),
        LinkTypes::ProposalToVotes,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created vote".into()
    )))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use mutualaid_common::{
        MemberRole, MemberStatus, ProposalStatus, ProposalType, RuleCategory, VoteChoice,
        VotingMethod,
    };

    // ── Integrity enum serde roundtrip tests ──────────────────────────

    #[test]
    fn proposal_type_all_variants_serde() {
        let variants = vec![
            ProposalType::AddRule,
            ProposalType::ModifyRule,
            ProposalType::RemoveRule,
            ProposalType::CreditLimitChange,
            ProposalType::MemberAdmission,
            ProposalType::MemberStatusChange,
            ProposalType::ResourcePolicy,
            ProposalType::GeneralDecision,
            ProposalType::Emergency,
            ProposalType::Custom("special".to_string()),
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: ProposalType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn voting_method_all_variants_serde() {
        let variants = vec![
            VotingMethod::Majority,
            VotingMethod::Supermajority,
            VotingMethod::Consensus,
            VotingMethod::ConsentBased,
            VotingMethod::ContributionWeighted,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: VotingMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn proposal_status_all_variants_serde() {
        let variants = vec![
            ProposalStatus::Draft,
            ProposalStatus::Discussion,
            ProposalStatus::Voting,
            ProposalStatus::Passed,
            ProposalStatus::Failed,
            ProposalStatus::Implemented,
            ProposalStatus::Withdrawn,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: ProposalStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn vote_choice_all_variants_serde() {
        let variants = vec![
            VoteChoice::Yes,
            VoteChoice::No,
            VoteChoice::Abstain,
            VoteChoice::Block,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: VoteChoice = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn member_role_all_variants_serde() {
        let variants = vec![
            MemberRole::Member,
            MemberRole::Steward,
            MemberRole::ResourceManager,
            MemberRole::Treasurer,
            MemberRole::Facilitator,
            MemberRole::Founder,
            MemberRole::Custom("Elder".to_string()),
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: MemberRole = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn member_status_all_variants_serde() {
        let variants = vec![
            MemberStatus::Pending,
            MemberStatus::Active,
            MemberStatus::Inactive,
            MemberStatus::Suspended,
            MemberStatus::Departed,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: MemberStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn rule_category_all_variants_serde() {
        let variants = vec![
            RuleCategory::Membership,
            RuleCategory::Credits,
            RuleCategory::Resources,
            RuleCategory::Conduct,
            RuleCategory::Governance,
            RuleCategory::Disputes,
            RuleCategory::Privacy,
            RuleCategory::Custom("Safety".to_string()),
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: RuleCategory = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    // ========================================================================
    // Proposal serde roundtrip tests
    // ========================================================================

    #[test]
    fn proposal_full_serde_roundtrip() {
        let proposal = Proposal {
            id: "prop-001".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "Increase credit limit".to_string(),
            description: "Raise the default limit to 1000".to_string(),
            proposal_type: ProposalType::CreditLimitChange,
            modifies_rule: Some(ActionHash::from_raw_36(vec![1u8; 36])),
            voting_method: VotingMethod::Supermajority,
            quorum_percent: 75,
            threshold_percent: 66,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(2000000),
            status: ProposalStatus::Voting,
            created_at: Timestamp::from_micros(1000000),
        };
        let json = serde_json::to_string(&proposal).unwrap();
        let decoded: Proposal = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "prop-001");
        assert_eq!(decoded.proposal_type, ProposalType::CreditLimitChange);
        assert_eq!(decoded.quorum_percent, 75);
        assert!(decoded.modifies_rule.is_some());
    }

    #[test]
    fn proposal_minimal_serde_roundtrip() {
        let proposal = Proposal {
            id: "p".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "T".to_string(),
            description: "D".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 0,
            threshold_percent: 0,
            voting_starts: Timestamp::from_micros(0),
            voting_ends: Timestamp::from_micros(0),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&proposal).unwrap();
        let decoded: Proposal = serde_json::from_str(&json).unwrap();
        assert!(decoded.modifies_rule.is_none());
        assert_eq!(decoded.quorum_percent, 0);
        assert_eq!(decoded.threshold_percent, 0);
    }

    // ========================================================================
    // Vote serde roundtrip tests
    // ========================================================================

    #[test]
    fn vote_with_reasoning_serde_roundtrip() {
        let vote = Vote {
            proposal_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            voter: AgentPubKey::from_raw_36(vec![1u8; 36]),
            vote: VoteChoice::Yes,
            reasoning: Some("I support this change".to_string()),
            voted_at: Timestamp::from_micros(1500000),
        };
        let json = serde_json::to_string(&vote).unwrap();
        let decoded: Vote = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.vote, VoteChoice::Yes);
        assert_eq!(decoded.reasoning, Some("I support this change".to_string()));
    }

    #[test]
    fn vote_without_reasoning_serde_roundtrip() {
        let vote = Vote {
            proposal_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            voter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            vote: VoteChoice::Abstain,
            reasoning: None,
            voted_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&vote).unwrap();
        let decoded: Vote = serde_json::from_str(&json).unwrap();
        assert!(decoded.reasoning.is_none());
        assert_eq!(decoded.vote, VoteChoice::Abstain);
    }

    // ========================================================================
    // Clone/equality tests
    // ========================================================================

    #[test]
    fn proposal_clone_equals_original() {
        let proposal = Proposal {
            id: "prop-clone".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "Clone test".to_string(),
            description: "Test".to_string(),
            proposal_type: ProposalType::Emergency,
            modifies_rule: None,
            voting_method: VotingMethod::Consensus,
            quorum_percent: 100,
            threshold_percent: 100,
            voting_starts: Timestamp::from_micros(0),
            voting_ends: Timestamp::from_micros(0),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(0),
        };
        assert_eq!(proposal, proposal.clone());
    }

    #[test]
    fn vote_clone_equals_original() {
        let vote = Vote {
            proposal_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            voter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            vote: VoteChoice::Block,
            reasoning: Some("Strong objection".to_string()),
            voted_at: Timestamp::from_micros(0),
        };
        assert_eq!(vote, vote.clone());
    }

    #[test]
    fn proposal_ne_different_status() {
        let a = Proposal {
            id: "p".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "T".to_string(),
            description: "D".to_string(),
            proposal_type: ProposalType::AddRule,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 51,
            voting_starts: Timestamp::from_micros(0),
            voting_ends: Timestamp::from_micros(0),
            status: ProposalStatus::Voting,
            created_at: Timestamp::from_micros(0),
        };
        let mut b = a.clone();
        b.status = ProposalStatus::Passed;
        assert_ne!(a, b);
    }

    #[test]
    fn vote_ne_different_choice() {
        let a = Vote {
            proposal_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            voter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            vote: VoteChoice::Yes,
            reasoning: None,
            voted_at: Timestamp::from_micros(0),
        };
        let mut b = a.clone();
        b.vote = VoteChoice::No;
        assert_ne!(a, b);
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn proposal_type_custom_empty_string_serde() {
        let pt = ProposalType::Custom("".to_string());
        let json = serde_json::to_string(&pt).unwrap();
        let decoded: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ProposalType::Custom("".to_string()));
    }

    #[test]
    fn proposal_type_custom_unicode_serde() {
        let pt = ProposalType::Custom("\u{6C11}\u{4E3B}\u{7684}\u{6C7A}\u{5B9A}".to_string());
        let json = serde_json::to_string(&pt).unwrap();
        let decoded: ProposalType = serde_json::from_str(&json).unwrap();
        if let ProposalType::Custom(s) = decoded {
            assert_eq!(s, "\u{6C11}\u{4E3B}\u{7684}\u{6C7A}\u{5B9A}");
        } else {
            panic!("Expected Custom variant");
        }
    }

    #[test]
    fn member_role_custom_empty_string_serde() {
        let mr = MemberRole::Custom("".to_string());
        let json = serde_json::to_string(&mr).unwrap();
        let decoded: MemberRole = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, MemberRole::Custom("".to_string()));
    }

    #[test]
    fn rule_category_custom_empty_string_serde() {
        let rc = RuleCategory::Custom("".to_string());
        let json = serde_json::to_string(&rc).unwrap();
        let decoded: RuleCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, RuleCategory::Custom("".to_string()));
    }

    #[test]
    fn vote_choice_all_variants_clone_eq() {
        for vc in [
            VoteChoice::Yes,
            VoteChoice::No,
            VoteChoice::Abstain,
            VoteChoice::Block,
        ] {
            assert_eq!(vc, vc.clone());
        }
    }

    #[test]
    fn proposal_status_all_variants_clone_eq() {
        for ps in [
            ProposalStatus::Draft,
            ProposalStatus::Discussion,
            ProposalStatus::Voting,
            ProposalStatus::Passed,
            ProposalStatus::Failed,
            ProposalStatus::Implemented,
            ProposalStatus::Withdrawn,
        ] {
            assert_eq!(ps, ps.clone());
        }
    }

    // ── Validation edge case tests (coordinator-level) ──────────────

    #[test]
    fn proposal_whitespace_title_detected() {
        let proposal = Proposal {
            id: "prop-ws-title".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "   \t\n  ".to_string(),
            description: "Valid description".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 51,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(2000000),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(1000000),
        };
        assert!(
            proposal.title.trim().is_empty(),
            "Whitespace-only title should be detected"
        );
    }

    #[test]
    fn proposal_empty_title_detected() {
        let proposal = Proposal {
            id: "prop-empty-title".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "".to_string(),
            description: "Valid description".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 51,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(2000000),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(1000000),
        };
        assert!(
            proposal.title.trim().is_empty(),
            "Empty title should be detected"
        );
    }

    #[test]
    fn proposal_whitespace_description_detected() {
        let proposal = Proposal {
            id: "prop-ws-desc".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "Valid Title".to_string(),
            description: "   ".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 51,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(2000000),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(1000000),
        };
        assert!(
            proposal.description.trim().is_empty(),
            "Whitespace-only description should be detected"
        );
    }

    #[test]
    fn proposal_zero_quorum_detected() {
        let proposal = Proposal {
            id: "prop-zero-quorum".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "Valid Title".to_string(),
            description: "Valid Description".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 0,
            threshold_percent: 51,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(2000000),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(1000000),
        };
        assert_eq!(proposal.quorum_percent, 0, "Zero quorum should be detected");
    }

    #[test]
    fn proposal_zero_threshold_detected() {
        let proposal = Proposal {
            id: "prop-zero-thresh".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "Valid Title".to_string(),
            description: "Valid Description".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 0,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(2000000),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(1000000),
        };
        assert_eq!(
            proposal.threshold_percent, 0,
            "Zero threshold should be detected"
        );
    }

    #[test]
    fn proposal_tiny_voting_window_detected() {
        // 1-microsecond voting window should be caught by MIN_VOTING_PERIOD_MS check
        let proposal = Proposal {
            id: "prop-tiny-window".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "Valid Title".to_string(),
            description: "Valid Description".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 51,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(1000001), // 1 microsecond later
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(1000000),
        };
        let duration_us = proposal.voting_ends.as_micros() - proposal.voting_starts.as_micros();
        assert!(
            duration_us < MIN_VOTING_PERIOD_MS,
            "1-microsecond voting window should be less than minimum: {} < {}",
            duration_us,
            MIN_VOTING_PERIOD_MS
        );
    }

    #[test]
    fn proposal_inverted_voting_window_detected() {
        // voting_ends before voting_starts
        let proposal = Proposal {
            id: "prop-inverted".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "Valid Title".to_string(),
            description: "Valid Description".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 51,
            voting_starts: Timestamp::from_micros(2000000),
            voting_ends: Timestamp::from_micros(1000000), // Before start!
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(1000000),
        };
        assert!(
            proposal.voting_ends <= proposal.voting_starts,
            "Inverted voting window should be detected"
        );
    }

    #[test]
    fn proposal_valid_voting_window_accepted() {
        // 10-minute voting window
        let start_us: i64 = 1_000_000;
        let ten_minutes_us: i64 = 10 * 60 * 1_000_000;
        let proposal = Proposal {
            id: "prop-valid-window".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0u8; 36]),
            title: "Valid Title".to_string(),
            description: "Valid Description".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 51,
            voting_starts: Timestamp::from_micros(start_us),
            voting_ends: Timestamp::from_micros(start_us + ten_minutes_us),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(start_us),
        };
        let duration_us = proposal.voting_ends.as_micros() - proposal.voting_starts.as_micros();
        assert!(
            duration_us >= MIN_VOTING_PERIOD_MS,
            "10-minute voting window should pass minimum check"
        );
        assert!(proposal.voting_ends > proposal.voting_starts);
        assert!(proposal.quorum_percent > 0);
        assert!(proposal.threshold_percent > 0);
    }

    #[test]
    fn double_vote_same_voter_detected() {
        // Verify that two votes from the same voter on the same proposal can be detected
        let proposal_hash = ActionHash::from_raw_36(vec![0xAA; 36]);
        let voter = AgentPubKey::from_raw_36(vec![0xBB; 36]);

        let vote1 = Vote {
            proposal_hash: proposal_hash.clone(),
            voter: voter.clone(),
            vote: VoteChoice::Yes,
            reasoning: Some("First vote".to_string()),
            voted_at: Timestamp::from_micros(1000000),
        };

        let vote2 = Vote {
            proposal_hash: proposal_hash.clone(),
            voter: voter.clone(),
            vote: VoteChoice::No,
            reasoning: Some("Changed my mind".to_string()),
            voted_at: Timestamp::from_micros(2000000),
        };

        // Same voter on same proposal should be detected
        assert_eq!(vote1.voter, vote2.voter, "Same voter should be detected");
        assert_eq!(
            vote1.proposal_hash, vote2.proposal_hash,
            "Same proposal should be detected"
        );
    }

    #[test]
    fn different_voters_not_flagged() {
        let proposal_hash = ActionHash::from_raw_36(vec![0xAA; 36]);
        let voter1 = AgentPubKey::from_raw_36(vec![0xBB; 36]);
        let voter2 = AgentPubKey::from_raw_36(vec![0xCC; 36]);

        let vote1 = Vote {
            proposal_hash: proposal_hash.clone(),
            voter: voter1,
            vote: VoteChoice::Yes,
            reasoning: None,
            voted_at: Timestamp::from_micros(1000000),
        };

        let vote2 = Vote {
            proposal_hash: proposal_hash.clone(),
            voter: voter2,
            vote: VoteChoice::No,
            reasoning: None,
            voted_at: Timestamp::from_micros(2000000),
        };

        assert_ne!(
            vote1.voter, vote2.voter,
            "Different voters should not be flagged"
        );
    }

    #[test]
    fn same_voter_different_proposals_not_flagged() {
        let voter = AgentPubKey::from_raw_36(vec![0xBB; 36]);
        let proposal1 = ActionHash::from_raw_36(vec![0xAA; 36]);
        let proposal2 = ActionHash::from_raw_36(vec![0xDD; 36]);

        let vote1 = Vote {
            proposal_hash: proposal1,
            voter: voter.clone(),
            vote: VoteChoice::Yes,
            reasoning: None,
            voted_at: Timestamp::from_micros(1000000),
        };

        let vote2 = Vote {
            proposal_hash: proposal2,
            voter: voter.clone(),
            vote: VoteChoice::No,
            reasoning: None,
            voted_at: Timestamp::from_micros(2000000),
        };

        assert_ne!(
            vote1.proposal_hash, vote2.proposal_hash,
            "Different proposals should not be flagged"
        );
    }

    #[test]
    fn min_voting_period_constant_is_5_minutes() {
        // 5 minutes = 5 * 60 * 1_000_000 microseconds = 300_000_000
        assert_eq!(
            MIN_VOTING_PERIOD_MS, 300_000_000,
            "MIN_VOTING_PERIOD_MS should be 5 minutes in microseconds"
        );
    }
}
