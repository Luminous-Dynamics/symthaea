//! Mutual Aid Governance Integrity Zome
//! Democratic decision-making for mutual aid circles.

use hdi::prelude::*;
use mutualaid_common::*;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Proposal(Proposal),
    Vote(Vote),
    Rule(Rule),
    Member(Member),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllProposals,
    ProposalToVotes,
    AllRules,
    AllMembers,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Proposal(proposal) => validate_proposal(proposal),
                EntryTypes::Vote(vote) => validate_vote(vote),
                EntryTypes::Rule(rule) => validate_rule(rule),
                EntryTypes::Member(member) => validate_member(member),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Proposal(proposal) => validate_proposal(proposal),
                EntryTypes::Vote(vote) => validate_vote(vote),
                EntryTypes::Rule(rule) => validate_rule(rule),
                EntryTypes::Member(member) => validate_member(member),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => match link_type {
            LinkTypes::AllProposals => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllProposals link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ProposalToVotes => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ProposalToVotes link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllRules => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllRules link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllMembers => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllMembers link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_proposal(proposal: Proposal) -> ExternResult<ValidateCallbackResult> {
    if proposal.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID cannot be empty".into(),
        ));
    }
    if proposal.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal ID must be 256 characters or fewer".into(),
        ));
    }
    if proposal.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal title cannot be empty".into(),
        ));
    }
    if proposal.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal title must be 512 characters or fewer".into(),
        ));
    }
    if proposal.description.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal description cannot be empty".into(),
        ));
    }
    if proposal.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Proposal description must be 4096 characters or fewer".into(),
        ));
    }
    if proposal.quorum_percent > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quorum percent must be 0-100".into(),
        ));
    }
    if proposal.threshold_percent > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Threshold percent must be 0-100".into(),
        ));
    }
    // Voting window: starts must be before ends
    if proposal.voting_starts >= proposal.voting_ends {
        return Ok(ValidateCallbackResult::Invalid(
            "voting_starts must be before voting_ends".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_rule(rule: Rule) -> ExternResult<ValidateCallbackResult> {
    if rule.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Rule ID cannot be empty".into(),
        ));
    }
    if rule.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rule ID must be 256 characters or fewer".into(),
        ));
    }
    if rule.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Rule title cannot be empty".into(),
        ));
    }
    if rule.title.len() > 512 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rule title must be 512 characters or fewer".into(),
        ));
    }
    if rule.text.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Rule text cannot be empty".into(),
        ));
    }
    if rule.text.len() > 8192 {
        return Ok(ValidateCallbackResult::Invalid(
            "Rule text must be 8192 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_member(member: Member) -> ExternResult<ValidateCallbackResult> {
    if member.display_name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Member display name cannot be empty".into(),
        ));
    }
    if member.display_name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Member display name must be 256 characters or fewer".into(),
        ));
    }
    if member.roles.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Member must have at least one role".into(),
        ));
    }
    if member.roles.len() > 20 {
        return Ok(ValidateCallbackResult::Invalid(
            "Member cannot have more than 20 roles".into(),
        ));
    }
    if let Some(score) = member.matl_score {
        if !score.is_finite() || !(0.0..=1.0).contains(&score) {
            return Ok(ValidateCallbackResult::Invalid(
                "MATL score must be a finite number between 0.0 and 1.0".into(),
            ));
        }
    }
    // Validate custom role name lengths
    for role in &member.roles {
        if let MemberRole::Custom(name) = role {
            if name.trim().is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Custom role name cannot be empty".into(),
                ));
            }
            if name.len() > 128 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Custom role name must be 128 characters or fewer".into(),
                ));
            }
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_vote(vote: Vote) -> ExternResult<ValidateCallbackResult> {
    if let Some(ref reasoning) = vote.reasoning {
        if reasoning.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Vote reasoning must be 4096 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // PROPOSAL TESTS
    // ========================================================================

    #[test]
    fn test_proposal_construction() {
        let proposer = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let proposal = Proposal {
            id: "proposal-001".to_string(),
            proposer: proposer.clone(),
            title: "Add new credit limit rule".to_string(),
            description: "Increase default credit limit to 1000".to_string(),
            proposal_type: ProposalType::AddRule,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 66,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(2000000),
            status: ProposalStatus::Voting,
            created_at: Timestamp::from_micros(1000000),
        };

        assert_eq!(proposal.id, "proposal-001");
        assert_eq!(proposal.proposer, proposer);
        assert_eq!(proposal.quorum_percent, 50);
        assert_eq!(proposal.threshold_percent, 66);
    }

    #[test]
    fn test_proposal_serde_roundtrip() {
        let proposer = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let proposal = Proposal {
            id: "proposal-002".to_string(),
            proposer: proposer.clone(),
            title: "Emergency decision".to_string(),
            description: "Temporary suspension of fees".to_string(),
            proposal_type: ProposalType::Emergency,
            modifies_rule: Some(ActionHash::from_raw_36(vec![0xab; 36])),
            voting_method: VotingMethod::Consensus,
            quorum_percent: 75,
            threshold_percent: 100,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(2000000),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(1000000),
        };

        let serialized = serde_json::to_string(&proposal).unwrap();
        let deserialized: Proposal = serde_json::from_str(&serialized).unwrap();

        assert_eq!(proposal, deserialized);
        assert_eq!(deserialized.id, "proposal-002");
        assert_eq!(
            deserialized.modifies_rule,
            Some(ActionHash::from_raw_36(vec![0xab; 36]))
        );
    }

    #[test]
    fn test_proposal_with_all_statuses() {
        let proposer = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let statuses = vec![
            ProposalStatus::Draft,
            ProposalStatus::Discussion,
            ProposalStatus::Voting,
            ProposalStatus::Passed,
            ProposalStatus::Failed,
            ProposalStatus::Implemented,
            ProposalStatus::Withdrawn,
        ];

        for status in statuses {
            let proposal = Proposal {
                id: format!("proposal-{:?}", status),
                proposer: proposer.clone(),
                title: "Test".to_string(),
                description: "Test".to_string(),
                proposal_type: ProposalType::GeneralDecision,
                modifies_rule: None,
                voting_method: VotingMethod::Majority,
                quorum_percent: 50,
                threshold_percent: 51,
                voting_starts: Timestamp::from_micros(1000000),
                voting_ends: Timestamp::from_micros(2000000),
                status: status.clone(),
                created_at: Timestamp::from_micros(1000000),
            };

            let serialized = serde_json::to_string(&proposal).unwrap();
            let deserialized: Proposal = serde_json::from_str(&serialized).unwrap();
            assert_eq!(deserialized.status, status);
        }
    }

    // ========================================================================
    // PROPOSAL TYPE TESTS (10 variants)
    // ========================================================================

    #[test]
    fn test_proposal_type_add_rule_serde() {
        let pt = ProposalType::AddRule;
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
    }

    #[test]
    fn test_proposal_type_modify_rule_serde() {
        let pt = ProposalType::ModifyRule;
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
    }

    #[test]
    fn test_proposal_type_remove_rule_serde() {
        let pt = ProposalType::RemoveRule;
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
    }

    #[test]
    fn test_proposal_type_credit_limit_change_serde() {
        let pt = ProposalType::CreditLimitChange;
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
    }

    #[test]
    fn test_proposal_type_member_admission_serde() {
        let pt = ProposalType::MemberAdmission;
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
    }

    #[test]
    fn test_proposal_type_member_status_change_serde() {
        let pt = ProposalType::MemberStatusChange;
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
    }

    #[test]
    fn test_proposal_type_resource_policy_serde() {
        let pt = ProposalType::ResourcePolicy;
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
    }

    #[test]
    fn test_proposal_type_general_decision_serde() {
        let pt = ProposalType::GeneralDecision;
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
    }

    #[test]
    fn test_proposal_type_emergency_serde() {
        let pt = ProposalType::Emergency;
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
    }

    #[test]
    fn test_proposal_type_custom_serde() {
        let pt = ProposalType::Custom("Budget Allocation".to_string());
        let json = serde_json::to_string(&pt).unwrap();
        let parsed: ProposalType = serde_json::from_str(&json).unwrap();
        assert_eq!(pt, parsed);
        if let ProposalType::Custom(name) = parsed {
            assert_eq!(name, "Budget Allocation");
        } else {
            panic!("Expected Custom variant");
        }
    }

    // ========================================================================
    // VOTING METHOD TESTS (5 variants)
    // ========================================================================

    #[test]
    fn test_voting_method_majority_serde() {
        let vm = VotingMethod::Majority;
        let json = serde_json::to_string(&vm).unwrap();
        let parsed: VotingMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(vm, parsed);
    }

    #[test]
    fn test_voting_method_supermajority_serde() {
        let vm = VotingMethod::Supermajority;
        let json = serde_json::to_string(&vm).unwrap();
        let parsed: VotingMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(vm, parsed);
    }

    #[test]
    fn test_voting_method_consensus_serde() {
        let vm = VotingMethod::Consensus;
        let json = serde_json::to_string(&vm).unwrap();
        let parsed: VotingMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(vm, parsed);
    }

    #[test]
    fn test_voting_method_consent_based_serde() {
        let vm = VotingMethod::ConsentBased;
        let json = serde_json::to_string(&vm).unwrap();
        let parsed: VotingMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(vm, parsed);
    }

    #[test]
    fn test_voting_method_contribution_weighted_serde() {
        let vm = VotingMethod::ContributionWeighted;
        let json = serde_json::to_string(&vm).unwrap();
        let parsed: VotingMethod = serde_json::from_str(&json).unwrap();
        assert_eq!(vm, parsed);
    }

    // ========================================================================
    // PROPOSAL STATUS TESTS (7 variants)
    // ========================================================================

    #[test]
    fn test_proposal_status_draft_serde() {
        let ps = ProposalStatus::Draft;
        let json = serde_json::to_string(&ps).unwrap();
        let parsed: ProposalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ps, parsed);
    }

    #[test]
    fn test_proposal_status_discussion_serde() {
        let ps = ProposalStatus::Discussion;
        let json = serde_json::to_string(&ps).unwrap();
        let parsed: ProposalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ps, parsed);
    }

    #[test]
    fn test_proposal_status_voting_serde() {
        let ps = ProposalStatus::Voting;
        let json = serde_json::to_string(&ps).unwrap();
        let parsed: ProposalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ps, parsed);
    }

    #[test]
    fn test_proposal_status_passed_serde() {
        let ps = ProposalStatus::Passed;
        let json = serde_json::to_string(&ps).unwrap();
        let parsed: ProposalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ps, parsed);
    }

    #[test]
    fn test_proposal_status_failed_serde() {
        let ps = ProposalStatus::Failed;
        let json = serde_json::to_string(&ps).unwrap();
        let parsed: ProposalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ps, parsed);
    }

    #[test]
    fn test_proposal_status_implemented_serde() {
        let ps = ProposalStatus::Implemented;
        let json = serde_json::to_string(&ps).unwrap();
        let parsed: ProposalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ps, parsed);
    }

    #[test]
    fn test_proposal_status_withdrawn_serde() {
        let ps = ProposalStatus::Withdrawn;
        let json = serde_json::to_string(&ps).unwrap();
        let parsed: ProposalStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ps, parsed);
    }

    // ========================================================================
    // VOTE TESTS
    // ========================================================================

    #[test]
    fn test_vote_construction() {
        let voter = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let proposal_hash = ActionHash::from_raw_36(vec![0xab; 36]);
        let vote = Vote {
            proposal_hash: proposal_hash.clone(),
            voter: voter.clone(),
            vote: VoteChoice::Yes,
            reasoning: Some("I support this proposal".to_string()),
            voted_at: Timestamp::from_micros(1000000),
        };

        assert_eq!(vote.proposal_hash, proposal_hash);
        assert_eq!(vote.voter, voter);
        assert_eq!(vote.vote, VoteChoice::Yes);
        assert!(vote.reasoning.is_some());
    }

    #[test]
    fn test_vote_serde_roundtrip() {
        let voter = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let proposal_hash = ActionHash::from_raw_36(vec![0xab; 36]);
        let vote = Vote {
            proposal_hash: proposal_hash.clone(),
            voter: voter.clone(),
            vote: VoteChoice::Block,
            reasoning: None,
            voted_at: Timestamp::from_micros(1500000),
        };

        let serialized = serde_json::to_string(&vote).unwrap();
        let deserialized: Vote = serde_json::from_str(&serialized).unwrap();

        assert_eq!(vote, deserialized);
        assert_eq!(deserialized.vote, VoteChoice::Block);
        assert!(deserialized.reasoning.is_none());
    }

    // ========================================================================
    // VOTE CHOICE TESTS (4 variants)
    // ========================================================================

    #[test]
    fn test_vote_choice_yes_serde() {
        let vc = VoteChoice::Yes;
        let json = serde_json::to_string(&vc).unwrap();
        let parsed: VoteChoice = serde_json::from_str(&json).unwrap();
        assert_eq!(vc, parsed);
    }

    #[test]
    fn test_vote_choice_no_serde() {
        let vc = VoteChoice::No;
        let json = serde_json::to_string(&vc).unwrap();
        let parsed: VoteChoice = serde_json::from_str(&json).unwrap();
        assert_eq!(vc, parsed);
    }

    #[test]
    fn test_vote_choice_abstain_serde() {
        let vc = VoteChoice::Abstain;
        let json = serde_json::to_string(&vc).unwrap();
        let parsed: VoteChoice = serde_json::from_str(&json).unwrap();
        assert_eq!(vc, parsed);
    }

    #[test]
    fn test_vote_choice_block_serde() {
        let vc = VoteChoice::Block;
        let json = serde_json::to_string(&vc).unwrap();
        let parsed: VoteChoice = serde_json::from_str(&json).unwrap();
        assert_eq!(vc, parsed);
    }

    // ========================================================================
    // RULE TESTS
    // ========================================================================

    #[test]
    fn test_rule_construction() {
        let proposal_hash = ActionHash::from_raw_36(vec![0xab; 36]);
        let rule = Rule {
            id: "rule-001".to_string(),
            title: "Credit Limit Rule".to_string(),
            text: "All members have a default credit limit of 500".to_string(),
            category: RuleCategory::Credits,
            priority: 10,
            created_by_proposal: proposal_hash.clone(),
            active_since: Timestamp::from_micros(1000000),
            active: true,
            superseded_by: None,
        };

        assert_eq!(rule.id, "rule-001");
        assert_eq!(rule.category, RuleCategory::Credits);
        assert_eq!(rule.priority, 10);
        assert!(rule.active);
    }

    #[test]
    fn test_rule_serde_roundtrip() {
        let proposal_hash = ActionHash::from_raw_36(vec![0xab; 36]);
        let superseded_hash = ActionHash::from_raw_36(vec![0xcd; 36]);
        let rule = Rule {
            id: "rule-002".to_string(),
            title: "Governance Rule".to_string(),
            text: "Quorum is 66% for all proposals".to_string(),
            category: RuleCategory::Governance,
            priority: 5,
            created_by_proposal: proposal_hash.clone(),
            active_since: Timestamp::from_micros(2000000),
            active: false,
            superseded_by: Some(superseded_hash.clone()),
        };

        let serialized = serde_json::to_string(&rule).unwrap();
        let deserialized: Rule = serde_json::from_str(&serialized).unwrap();

        assert_eq!(rule, deserialized);
        assert!(!deserialized.active);
        assert_eq!(deserialized.superseded_by, Some(superseded_hash));
    }

    // ========================================================================
    // RULE CATEGORY TESTS (8 variants)
    // ========================================================================

    #[test]
    fn test_rule_category_membership_serde() {
        let rc = RuleCategory::Membership;
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: RuleCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
    }

    #[test]
    fn test_rule_category_credits_serde() {
        let rc = RuleCategory::Credits;
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: RuleCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
    }

    #[test]
    fn test_rule_category_resources_serde() {
        let rc = RuleCategory::Resources;
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: RuleCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
    }

    #[test]
    fn test_rule_category_conduct_serde() {
        let rc = RuleCategory::Conduct;
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: RuleCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
    }

    #[test]
    fn test_rule_category_governance_serde() {
        let rc = RuleCategory::Governance;
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: RuleCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
    }

    #[test]
    fn test_rule_category_disputes_serde() {
        let rc = RuleCategory::Disputes;
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: RuleCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
    }

    #[test]
    fn test_rule_category_privacy_serde() {
        let rc = RuleCategory::Privacy;
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: RuleCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
    }

    #[test]
    fn test_rule_category_custom_serde() {
        let rc = RuleCategory::Custom("Environmental".to_string());
        let json = serde_json::to_string(&rc).unwrap();
        let parsed: RuleCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(rc, parsed);
        if let RuleCategory::Custom(name) = parsed {
            assert_eq!(name, "Environmental");
        } else {
            panic!("Expected Custom variant");
        }
    }

    // ========================================================================
    // MEMBER TESTS
    // ========================================================================

    #[test]
    fn test_member_construction() {
        let agent = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let member = Member {
            agent: agent.clone(),
            display_name: "Alice Smith".to_string(),
            identity_hash: None,
            roles: vec![MemberRole::Member],
            joined_at: Timestamp::from_micros(1000000),
            status: MemberStatus::Active,
            endorsement_count: 5,
            matl_score: Some(0.85),
        };

        assert_eq!(member.agent, agent);
        assert_eq!(member.display_name, "Alice Smith");
        assert_eq!(member.roles.len(), 1);
        assert_eq!(member.endorsement_count, 5);
        assert_eq!(member.matl_score, Some(0.85));
    }

    #[test]
    fn test_member_serde_roundtrip() {
        let agent = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let identity_hash = ActionHash::from_raw_36(vec![0xef; 36]);
        let member = Member {
            agent: agent.clone(),
            display_name: "Bob Jones".to_string(),
            identity_hash: Some(identity_hash.clone()),
            roles: vec![MemberRole::Steward, MemberRole::Facilitator],
            joined_at: Timestamp::from_micros(2000000),
            status: MemberStatus::Pending,
            endorsement_count: 0,
            matl_score: None,
        };

        let serialized = serde_json::to_string(&member).unwrap();
        let deserialized: Member = serde_json::from_str(&serialized).unwrap();

        assert_eq!(member, deserialized);
        assert_eq!(deserialized.roles.len(), 2);
        assert_eq!(deserialized.identity_hash, Some(identity_hash));
    }

    #[test]
    fn test_member_with_all_roles() {
        let agent = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let roles = vec![
            MemberRole::Member,
            MemberRole::Steward,
            MemberRole::ResourceManager,
            MemberRole::Treasurer,
            MemberRole::Facilitator,
            MemberRole::Founder,
            MemberRole::Custom("Mediator".to_string()),
        ];

        let member = Member {
            agent: agent.clone(),
            display_name: "Multi-Role Member".to_string(),
            identity_hash: None,
            roles: roles.clone(),
            joined_at: Timestamp::from_micros(1000000),
            status: MemberStatus::Active,
            endorsement_count: 10,
            matl_score: Some(0.95),
        };

        assert_eq!(member.roles.len(), 7);
        assert_eq!(member.roles, roles);
    }

    // ========================================================================
    // MEMBER ROLE TESTS (7 variants)
    // ========================================================================

    #[test]
    fn test_member_role_member_serde() {
        let mr = MemberRole::Member;
        let json = serde_json::to_string(&mr).unwrap();
        let parsed: MemberRole = serde_json::from_str(&json).unwrap();
        assert_eq!(mr, parsed);
    }

    #[test]
    fn test_member_role_steward_serde() {
        let mr = MemberRole::Steward;
        let json = serde_json::to_string(&mr).unwrap();
        let parsed: MemberRole = serde_json::from_str(&json).unwrap();
        assert_eq!(mr, parsed);
    }

    #[test]
    fn test_member_role_resource_manager_serde() {
        let mr = MemberRole::ResourceManager;
        let json = serde_json::to_string(&mr).unwrap();
        let parsed: MemberRole = serde_json::from_str(&json).unwrap();
        assert_eq!(mr, parsed);
    }

    #[test]
    fn test_member_role_treasurer_serde() {
        let mr = MemberRole::Treasurer;
        let json = serde_json::to_string(&mr).unwrap();
        let parsed: MemberRole = serde_json::from_str(&json).unwrap();
        assert_eq!(mr, parsed);
    }

    #[test]
    fn test_member_role_facilitator_serde() {
        let mr = MemberRole::Facilitator;
        let json = serde_json::to_string(&mr).unwrap();
        let parsed: MemberRole = serde_json::from_str(&json).unwrap();
        assert_eq!(mr, parsed);
    }

    #[test]
    fn test_member_role_founder_serde() {
        let mr = MemberRole::Founder;
        let json = serde_json::to_string(&mr).unwrap();
        let parsed: MemberRole = serde_json::from_str(&json).unwrap();
        assert_eq!(mr, parsed);
    }

    #[test]
    fn test_member_role_custom_serde() {
        let mr = MemberRole::Custom("Conflict Resolver".to_string());
        let json = serde_json::to_string(&mr).unwrap();
        let parsed: MemberRole = serde_json::from_str(&json).unwrap();
        assert_eq!(mr, parsed);
        if let MemberRole::Custom(name) = parsed {
            assert_eq!(name, "Conflict Resolver");
        } else {
            panic!("Expected Custom variant");
        }
    }

    // ========================================================================
    // MEMBER STATUS TESTS (5 variants)
    // ========================================================================

    #[test]
    fn test_member_status_pending_serde() {
        let ms = MemberStatus::Pending;
        let json = serde_json::to_string(&ms).unwrap();
        let parsed: MemberStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ms, parsed);
    }

    #[test]
    fn test_member_status_active_serde() {
        let ms = MemberStatus::Active;
        let json = serde_json::to_string(&ms).unwrap();
        let parsed: MemberStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ms, parsed);
    }

    #[test]
    fn test_member_status_inactive_serde() {
        let ms = MemberStatus::Inactive;
        let json = serde_json::to_string(&ms).unwrap();
        let parsed: MemberStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ms, parsed);
    }

    #[test]
    fn test_member_status_suspended_serde() {
        let ms = MemberStatus::Suspended;
        let json = serde_json::to_string(&ms).unwrap();
        let parsed: MemberStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ms, parsed);
    }

    #[test]
    fn test_member_status_departed_serde() {
        let ms = MemberStatus::Departed;
        let json = serde_json::to_string(&ms).unwrap();
        let parsed: MemberStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(ms, parsed);
    }

    // ========================================================================
    // ENTRY TYPES TESTS
    // ========================================================================

    #[test]
    fn test_entry_types_proposal_construction() {
        let proposer = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let proposal = Proposal {
            id: "proposal-entry".to_string(),
            proposer,
            title: "Test Entry".to_string(),
            description: "Test".to_string(),
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

        let entry = EntryTypes::Proposal(proposal.clone());

        match entry {
            EntryTypes::Proposal(p) => assert_eq!(p.id, "proposal-entry"),
            _ => panic!("Expected Proposal entry type"),
        }
    }

    #[test]
    fn test_entry_types_vote_construction() {
        let voter = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let proposal_hash = ActionHash::from_raw_36(vec![0xab; 36]);
        let vote = Vote {
            proposal_hash,
            voter,
            vote: VoteChoice::Yes,
            reasoning: None,
            voted_at: Timestamp::from_micros(1000000),
        };

        let entry = EntryTypes::Vote(vote.clone());

        match entry {
            EntryTypes::Vote(v) => assert_eq!(v.vote, VoteChoice::Yes),
            _ => panic!("Expected Vote entry type"),
        }
    }

    #[test]
    fn test_entry_types_rule_construction() {
        let proposal_hash = ActionHash::from_raw_36(vec![0xab; 36]);
        let rule = Rule {
            id: "rule-entry".to_string(),
            title: "Test Rule".to_string(),
            text: "Test rule text".to_string(),
            category: RuleCategory::Governance,
            priority: 1,
            created_by_proposal: proposal_hash,
            active_since: Timestamp::from_micros(1000000),
            active: true,
            superseded_by: None,
        };

        let entry = EntryTypes::Rule(rule.clone());

        match entry {
            EntryTypes::Rule(r) => assert_eq!(r.id, "rule-entry"),
            _ => panic!("Expected Rule entry type"),
        }
    }

    #[test]
    fn test_entry_types_member_construction() {
        let agent = AgentPubKey::from_raw_36(vec![0xdb; 36]);
        let member = Member {
            agent,
            display_name: "Test Member".to_string(),
            identity_hash: None,
            roles: vec![MemberRole::Member],
            joined_at: Timestamp::from_micros(1000000),
            status: MemberStatus::Active,
            endorsement_count: 0,
            matl_score: None,
        };

        let entry = EntryTypes::Member(member.clone());

        match entry {
            EntryTypes::Member(m) => assert_eq!(m.display_name, "Test Member"),
            _ => panic!("Expected Member entry type"),
        }
    }

    // ========================================================================
    // PROPOSAL VALIDATION TESTS
    // ========================================================================

    fn make_valid_proposal() -> Proposal {
        Proposal {
            id: "proposal-001".to_string(),
            proposer: AgentPubKey::from_raw_36(vec![0xdb; 36]),
            title: "Test proposal".to_string(),
            description: "A meaningful description".to_string(),
            proposal_type: ProposalType::GeneralDecision,
            modifies_rule: None,
            voting_method: VotingMethod::Majority,
            quorum_percent: 50,
            threshold_percent: 51,
            voting_starts: Timestamp::from_micros(1000000),
            voting_ends: Timestamp::from_micros(2000000),
            status: ProposalStatus::Draft,
            created_at: Timestamp::from_micros(1000000),
        }
    }

    #[test]
    fn valid_proposal_passes_validation() {
        let result = validate_proposal(make_valid_proposal());
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn proposal_empty_id_rejected() {
        let mut p = make_valid_proposal();
        p.id = "".to_string();
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn proposal_whitespace_id_rejected() {
        let mut p = make_valid_proposal();
        p.id = "   ".to_string();
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn proposal_empty_title_rejected() {
        let mut p = make_valid_proposal();
        p.title = "".to_string();
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn proposal_empty_description_rejected() {
        let mut p = make_valid_proposal();
        p.description = "".to_string();
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn proposal_quorum_over_100_rejected() {
        let mut p = make_valid_proposal();
        p.quorum_percent = 101;
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn proposal_threshold_over_100_rejected() {
        let mut p = make_valid_proposal();
        p.threshold_percent = 200;
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn proposal_quorum_100_passes() {
        let mut p = make_valid_proposal();
        p.quorum_percent = 100;
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn proposal_quorum_0_passes() {
        let mut p = make_valid_proposal();
        p.quorum_percent = 0;
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    // ========================================================================
    // RULE VALIDATION TESTS
    // ========================================================================

    fn make_valid_rule() -> Rule {
        Rule {
            id: "rule-001".to_string(),
            title: "Test Rule".to_string(),
            text: "Members must contribute monthly".to_string(),
            category: RuleCategory::Governance,
            priority: 5,
            created_by_proposal: ActionHash::from_raw_36(vec![0xab; 36]),
            active_since: Timestamp::from_micros(1000000),
            active: true,
            superseded_by: None,
        }
    }

    #[test]
    fn valid_rule_passes_validation() {
        let result = validate_rule(make_valid_rule());
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn rule_empty_id_rejected() {
        let mut r = make_valid_rule();
        r.id = "".to_string();
        let result = validate_rule(r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn rule_whitespace_id_rejected() {
        let mut r = make_valid_rule();
        r.id = "  \t ".to_string();
        let result = validate_rule(r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn rule_empty_title_rejected() {
        let mut r = make_valid_rule();
        r.title = "".to_string();
        let result = validate_rule(r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn rule_empty_text_rejected() {
        let mut r = make_valid_rule();
        r.text = "".to_string();
        let result = validate_rule(r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ========================================================================
    // MEMBER VALIDATION TESTS
    // ========================================================================

    fn make_valid_member() -> Member {
        Member {
            agent: AgentPubKey::from_raw_36(vec![0xdb; 36]),
            display_name: "Alice".to_string(),
            identity_hash: None,
            roles: vec![MemberRole::Member],
            joined_at: Timestamp::from_micros(1000000),
            status: MemberStatus::Active,
            endorsement_count: 0,
            matl_score: Some(0.5),
        }
    }

    #[test]
    fn valid_member_passes_validation() {
        let result = validate_member(make_valid_member());
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn member_empty_display_name_rejected() {
        let mut m = make_valid_member();
        m.display_name = "".to_string();
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_whitespace_display_name_rejected() {
        let mut m = make_valid_member();
        m.display_name = "   ".to_string();
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_empty_roles_rejected() {
        let mut m = make_valid_member();
        m.roles = vec![];
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_matl_score_negative_rejected() {
        let mut m = make_valid_member();
        m.matl_score = Some(-0.1);
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_matl_score_over_1_rejected() {
        let mut m = make_valid_member();
        m.matl_score = Some(1.1);
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_matl_score_none_passes() {
        let mut m = make_valid_member();
        m.matl_score = None;
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn member_matl_score_boundary_passes() {
        let mut m = make_valid_member();
        m.matl_score = Some(0.0);
        assert!(matches!(
            validate_member(m.clone()),
            Ok(ValidateCallbackResult::Valid)
        ));
        m.matl_score = Some(1.0);
        assert!(matches!(
            validate_member(m),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    // ========================================================================
    // STRING MAX-LENGTH BOUNDARY TESTS
    // ========================================================================

    // -- Proposal ID --

    #[test]
    fn proposal_id_at_max_length_passes() {
        let mut p = make_valid_proposal();
        p.id = "x".repeat(64);
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn proposal_id_over_max_length_rejected() {
        let mut p = make_valid_proposal();
        p.id = "x".repeat(257);
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    // -- Proposal title --

    #[test]
    fn proposal_title_at_max_length_passes() {
        let mut p = make_valid_proposal();
        p.title = "t".repeat(512);
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn proposal_title_over_max_length_rejected() {
        let mut p = make_valid_proposal();
        p.title = "t".repeat(513);
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    // -- Proposal description --

    #[test]
    fn proposal_description_at_max_length_passes() {
        let mut p = make_valid_proposal();
        p.description = "d".repeat(4096);
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn proposal_description_over_max_length_rejected() {
        let mut p = make_valid_proposal();
        p.description = "d".repeat(4097);
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    // -- Rule ID --

    #[test]
    fn rule_id_at_max_length_passes() {
        let mut r = make_valid_rule();
        r.id = "r".repeat(64);
        assert!(matches!(
            validate_rule(r),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn rule_id_over_max_length_rejected() {
        let mut r = make_valid_rule();
        r.id = "r".repeat(257);
        assert!(matches!(
            validate_rule(r),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    // -- Rule title --

    #[test]
    fn rule_title_at_max_length_passes() {
        let mut r = make_valid_rule();
        r.title = "t".repeat(512);
        assert!(matches!(
            validate_rule(r),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn rule_title_over_max_length_rejected() {
        let mut r = make_valid_rule();
        r.title = "t".repeat(513);
        assert!(matches!(
            validate_rule(r),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    // -- Rule text --

    #[test]
    fn rule_text_at_max_length_passes() {
        let mut r = make_valid_rule();
        r.text = "x".repeat(8192);
        assert!(matches!(
            validate_rule(r),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn rule_text_over_max_length_rejected() {
        let mut r = make_valid_rule();
        r.text = "x".repeat(8193);
        assert!(matches!(
            validate_rule(r),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    // -- Member display_name --

    #[test]
    fn member_display_name_at_max_length_passes() {
        let mut m = make_valid_member();
        m.display_name = "n".repeat(256);
        assert!(matches!(
            validate_member(m),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn member_display_name_over_max_length_rejected() {
        let mut m = make_valid_member();
        m.display_name = "n".repeat(257);
        assert!(matches!(
            validate_member(m),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    // -- Member roles Vec --

    #[test]
    fn member_roles_at_max_count_passes() {
        let mut m = make_valid_member();
        m.roles = (0..20)
            .map(|i| MemberRole::Custom(format!("role-{}", i)))
            .collect();
        assert!(matches!(
            validate_member(m),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn member_roles_over_max_count_rejected() {
        let mut m = make_valid_member();
        m.roles = (0..21)
            .map(|i| MemberRole::Custom(format!("role-{}", i)))
            .collect();
        assert!(matches!(
            validate_member(m),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    // ========================================================================
    // VOTE VALIDATION TESTS
    // ========================================================================

    fn make_valid_vote() -> Vote {
        Vote {
            proposal_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            voter: AgentPubKey::from_raw_36(vec![0xdb; 36]),
            vote: VoteChoice::Yes,
            reasoning: Some("I support this".to_string()),
            voted_at: Timestamp::from_micros(1000000),
        }
    }

    #[test]
    fn valid_vote_passes_validation() {
        let result = validate_vote(make_valid_vote());
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn vote_no_reasoning_passes() {
        let mut v = make_valid_vote();
        v.reasoning = None;
        assert!(matches!(
            validate_vote(v),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn vote_reasoning_at_max_length_passes() {
        let mut v = make_valid_vote();
        v.reasoning = Some("r".repeat(4096));
        assert!(matches!(
            validate_vote(v),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn vote_reasoning_over_max_length_rejected() {
        let mut v = make_valid_vote();
        v.reasoning = Some("r".repeat(4097));
        assert!(matches!(
            validate_vote(v),
            Ok(ValidateCallbackResult::Invalid(_))
        ));
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllProposals => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllProposals link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ProposalToVotes => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ProposalToVotes link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllRules => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllRules link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllMembers => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllMembers link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn test_link_all_proposals_tag_at_limit() {
        let result = validate_link_tag(LinkTypes::AllProposals, vec![0u8; 256]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_all_proposals_tag_too_long() {
        let result = validate_link_tag(LinkTypes::AllProposals, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_proposal_to_votes_tag_at_limit() {
        let result = validate_link_tag(LinkTypes::ProposalToVotes, vec![0u8; 256]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_proposal_to_votes_tag_too_long() {
        let result = validate_link_tag(LinkTypes::ProposalToVotes, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_all_rules_tag_at_limit() {
        let result = validate_link_tag(LinkTypes::AllRules, vec![0u8; 256]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_all_rules_tag_too_long() {
        let result = validate_link_tag(LinkTypes::AllRules, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn test_link_all_members_tag_at_limit() {
        let result = validate_link_tag(LinkTypes::AllMembers, vec![0u8; 256]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Valid)));
    }

    #[test]
    fn test_link_all_members_tag_too_long() {
        let result = validate_link_tag(LinkTypes::AllMembers, vec![0u8; 257]);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ========================================================================
    // HARDENING: Byzantine & Edge Case Tests
    // ========================================================================

    // ── Proposal voting window validation ───────────────────────────────

    #[test]
    fn proposal_voting_starts_before_ends_ok() {
        let mut p = make_valid_proposal();
        p.voting_starts = Timestamp::from_micros(1000);
        p.voting_ends = Timestamp::from_micros(2000);
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn proposal_voting_starts_equals_ends_rejected() {
        let mut p = make_valid_proposal();
        p.voting_starts = Timestamp::from_micros(1000);
        p.voting_ends = Timestamp::from_micros(1000);
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
        if let Ok(ValidateCallbackResult::Invalid(msg)) = result {
            assert!(
                msg.contains("voting_starts must be before voting_ends"),
                "Got: {}",
                msg
            );
        }
    }

    #[test]
    fn proposal_voting_starts_after_ends_rejected() {
        let mut p = make_valid_proposal();
        p.voting_starts = Timestamp::from_micros(5000);
        p.voting_ends = Timestamp::from_micros(1000);
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn proposal_voting_window_1_microsecond_ok() {
        let mut p = make_valid_proposal();
        p.voting_starts = Timestamp::from_micros(1000);
        p.voting_ends = Timestamp::from_micros(1001);
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    // ── Quorum edge cases ───────────────────────────────────────────────

    #[test]
    fn proposal_quorum_0_threshold_0_ok() {
        let mut p = make_valid_proposal();
        p.quorum_percent = 0;
        p.threshold_percent = 0;
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn proposal_quorum_100_threshold_100_ok() {
        let mut p = make_valid_proposal();
        p.quorum_percent = 100;
        p.threshold_percent = 100;
        assert!(matches!(
            validate_proposal(p),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn proposal_quorum_255_rejected() {
        let mut p = make_valid_proposal();
        p.quorum_percent = 255;
        let result = validate_proposal(p);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ── Member edge cases ───────────────────────────────────────────────

    #[test]
    fn member_matl_score_exactly_0_ok() {
        let mut m = make_valid_member();
        m.matl_score = Some(0.0);
        assert!(matches!(
            validate_member(m),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn member_matl_score_exactly_1_ok() {
        let mut m = make_valid_member();
        m.matl_score = Some(1.0);
        assert!(matches!(
            validate_member(m),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn member_matl_score_nan_rejected() {
        let mut m = make_valid_member();
        m.matl_score = Some(f64::NAN);
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_matl_score_infinity_rejected() {
        let mut m = make_valid_member();
        m.matl_score = Some(f64::INFINITY);
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_matl_score_neg_infinity_rejected() {
        let mut m = make_valid_member();
        m.matl_score = Some(f64::NEG_INFINITY);
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_single_role_ok() {
        let mut m = make_valid_member();
        m.roles = vec![MemberRole::Member];
        assert!(matches!(
            validate_member(m),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn member_role_name_too_long_rejected() {
        let mut m = make_valid_member();
        m.roles = vec![MemberRole::Custom("x".repeat(129))];
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_role_name_exactly_128_ok() {
        let mut m = make_valid_member();
        m.roles = vec![MemberRole::Custom("x".repeat(128))];
        assert!(matches!(
            validate_member(m),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn member_empty_custom_role_rejected() {
        let mut m = make_valid_member();
        m.roles = vec![MemberRole::Custom(String::new())];
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn member_whitespace_custom_role_rejected() {
        let mut m = make_valid_member();
        m.roles = vec![MemberRole::Custom("  \t ".to_string())];
        let result = validate_member(m);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    // ── Vote edge cases ─────────────────────────────────────────────────

    #[test]
    fn vote_empty_reasoning_ok() {
        let mut v = make_valid_vote();
        v.reasoning = Some(String::new());
        assert!(matches!(
            validate_vote(v),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    #[test]
    fn vote_reasoning_exactly_4096_ok() {
        let mut v = make_valid_vote();
        v.reasoning = Some("v".repeat(4096));
        assert!(matches!(
            validate_vote(v),
            Ok(ValidateCallbackResult::Valid)
        ));
    }

    // ── Rule edge cases ─────────────────────────────────────────────────

    #[test]
    fn rule_whitespace_text_rejected() {
        let mut r = make_valid_rule();
        r.text = "   \n\t  ".to_string();
        let result = validate_rule(r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }

    #[test]
    fn rule_whitespace_title_rejected() {
        let mut r = make_valid_rule();
        r.title = "  \t ".to_string();
        let result = validate_rule(r);
        assert!(matches!(result, Ok(ValidateCallbackResult::Invalid(_))));
    }
}
