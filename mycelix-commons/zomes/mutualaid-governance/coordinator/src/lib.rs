//! Mutual Aid Governance Coordinator Zome
//! Democratic decision-making for mutual aid circles.

use hdk::prelude::*;
use mutualaid_governance_integrity::*;
use mutualaid_common::{Proposal, Vote};

/// Create a governance proposal
#[hdk_extern]
pub fn create_proposal(proposal: Proposal) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Proposal(proposal))?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created proposal".into())))
}

/// Get all proposals
#[hdk_extern]
pub fn get_all_proposals(_: ()) -> ExternResult<Vec<Record>> {
    Ok(vec![])
}

/// Cast a vote on a proposal
#[hdk_extern]
pub fn cast_vote(vote: Vote) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Vote(vote))?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find created vote".into())))
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    use mutualaid_common::{
        MemberRole, MemberStatus, ProposalStatus, ProposalType,
        RuleCategory, VoteChoice, VotingMethod,
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
}
