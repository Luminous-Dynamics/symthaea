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
pub fn validate(_op: Op) -> ExternResult<ValidateCallbackResult> {
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
        assert_eq!(deserialized.modifies_rule, Some(ActionHash::from_raw_36(vec![0xab; 36])));
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
}
