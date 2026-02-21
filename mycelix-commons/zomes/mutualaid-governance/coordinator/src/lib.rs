//! Mutual Aid Governance Coordinator Zome
//! Democratic decision-making for mutual aid circles.

use hdk::prelude::*;
use mutualaid_governance_integrity::*;
use mutualaid_common::{Proposal, Vote};
use mycelix_bridge_common::{
    ConsciousnessCredential, GovernanceEligibility, GovernanceRequirement,
    evaluate_governance, requirement_for_proposal, requirement_for_voting,
};

// ============================================================================
// Consciousness Gating
// ============================================================================

/// Fetch the calling agent's consciousness credential via the commons bridge
/// and evaluate it against the given governance requirement.
fn require_consciousness(
    requirement: &GovernanceRequirement,
) -> ExternResult<GovernanceEligibility> {
    let agent = agent_info()?.agent_initial_pubkey;
    let did = format!("did:mycelix:{}", agent);

    let response = call(
        CallTargetCell::Local,
        ZomeName::new("commons_bridge"),
        FunctionName::new("get_consciousness_credential"),
        None,
        did,
    )?;

    let credential: ConsciousnessCredential = match response {
        ZomeCallResponse::Ok(extern_io) => extern_io.decode().map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to decode consciousness credential: {}", e
            )))
        })?,
        other => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Consciousness credential call failed: {:?}", other
            ))));
        }
    };

    let eligibility = evaluate_governance(&credential.profile, requirement);
    if !eligibility.eligible {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Consciousness gate: tier {:?} insufficient. Reasons: {}",
            eligibility.tier,
            eligibility.reasons.join(", ")
        ))));
    }
    Ok(eligibility)
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Create a governance proposal
#[hdk_extern]
pub fn create_proposal(proposal: Proposal) -> ExternResult<Record> {
    // Consciousness gate: Participant tier + identity >= 0.25
    let _eligibility = require_consciousness(&requirement_for_proposal())?;

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
    // Consciousness gate: Citizen tier + identity >= 0.25
    let _eligibility = require_consciousness(&requirement_for_voting())?;

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
        for vc in [VoteChoice::Yes, VoteChoice::No, VoteChoice::Abstain, VoteChoice::Block] {
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
}
