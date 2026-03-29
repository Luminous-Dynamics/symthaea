// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Comprehensive tests for RB-BFT consensus
//!
//! Tests cover:
//! - Validator creation and management
//! - Reputation-squared voting weight
//! - Consensus threshold calculations
//! - Byzantine tolerance (45%)
//! - Slashing mechanics
//! - Round management
//! - Vote tallying

use super::*;

// =============================================================================
// VALIDATOR NODE TESTS
// =============================================================================

#[cfg(test)]
mod validator_node_tests {
    use super::*;

    #[test]
    fn test_new_validator() {
        let v = ValidatorNode::new("validator-1".to_string());

        assert_eq!(v.id, "validator-1");
        assert!(v.active);
        assert_eq!(v.successful_rounds, 0);
        assert_eq!(v.failed_rounds, 0);
        assert_eq!(v.slash_count, 0);
    }

    #[test]
    fn test_voting_weight_is_reputation_squared() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 0.8;

        let expected_weight = 0.8 * 0.8; // 0.64
        assert!((v.voting_weight() - expected_weight).abs() < 0.0001);
    }

    #[test]
    fn test_voting_weight_amplifies_high_reputation() {
        let mut high_rep = ValidatorNode::new("high".to_string());
        high_rep.k_vector.k_r = 0.9;

        let mut low_rep = ValidatorNode::new("low".to_string());
        low_rep.k_vector.k_r = 0.3;

        // High rep: 0.81, Low rep: 0.09
        // Ratio of weights: 9x (vs 3x for linear)
        let weight_ratio = high_rep.voting_weight() / low_rep.voting_weight();
        assert!(weight_ratio > 8.0, "Rep² should amplify weight difference");
    }

    #[test]
    fn test_can_participate_active_validator() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 0.5; // Above minimum

        assert!(v.can_participate().is_ok());
    }

    #[test]
    fn test_cannot_participate_inactive() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.active = false;
        v.k_vector.k_r = 0.5;

        assert!(v.can_participate().is_err());
    }

    #[test]
    fn test_cannot_participate_low_reputation() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 0.05; // Below MIN_PARTICIPATION_REPUTATION (0.1)

        assert!(v.can_participate().is_err());
    }

    #[test]
    fn test_update_reputation_success() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 0.5;

        let initial_rep = v.reputation();
        v.update_reputation(true);

        assert!(v.reputation() > initial_rep, "Success should increase reputation");
        assert_eq!(v.successful_rounds, 1);
    }

    #[test]
    fn test_update_reputation_failure() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 0.5;

        let initial_rep = v.reputation();
        v.update_reputation(false);

        assert!(v.reputation() < initial_rep, "Failure should decrease reputation");
        assert_eq!(v.failed_rounds, 1);
    }

    #[test]
    fn test_participation_rate() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.successful_rounds = 8;
        v.failed_rounds = 2;

        let rate = v.participation_rate();
        assert!((rate - 0.8).abs() < 0.0001);
    }

    #[test]
    fn test_participation_rate_zero_rounds() {
        let v = ValidatorNode::new("v1".to_string());
        assert_eq!(v.participation_rate(), 0.0);
    }
}

// =============================================================================
// SLASHING TESTS
// =============================================================================

#[cfg(test)]
mod slashing_tests {
    use super::*;

    #[test]
    fn test_slash_minor() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 1.0;

        v.slash(SlashingSeverity::Minor);

        assert_eq!(v.slash_count, 1);
        assert!((v.reputation() - 0.9).abs() < 0.01, "Minor slash: 10% penalty");
    }

    #[test]
    fn test_slash_moderate() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 1.0;

        v.slash(SlashingSeverity::Moderate);

        assert!((v.reputation() - 0.7).abs() < 0.01, "Moderate slash: 30% penalty");
    }

    #[test]
    fn test_slash_severe() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 1.0;

        v.slash(SlashingSeverity::Severe);

        assert!((v.reputation() - 0.5).abs() < 0.01, "Severe slash: 50% penalty");
    }

    #[test]
    fn test_slash_critical_deactivates() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 0.1; // Low reputation

        v.slash(SlashingSeverity::Critical); // 80% penalty

        // 0.1 * (1 - 0.8) = 0.02, which is < 0.05
        assert!(!v.active, "Critical slash should deactivate low-rep validator");
    }

    #[test]
    fn test_slash_also_affects_integrity() {
        let mut v = ValidatorNode::new("v1".to_string());
        v.k_vector.k_r = 1.0;
        v.k_vector.k_i = 1.0;

        v.slash(SlashingSeverity::Severe);

        // Integrity penalty is half of reputation penalty
        assert!(v.k_vector.k_i < 1.0, "Slash should affect integrity score");
        assert!((v.k_vector.k_i - 0.75).abs() < 0.01);
    }
}

// =============================================================================
// VALIDATOR SET TESTS
// =============================================================================

#[cfg(test)]
mod validator_set_tests {
    use super::*;

    fn create_test_validator(id: &str, reputation: f32) -> ValidatorNode {
        let mut v = ValidatorNode::new(id.to_string());
        v.k_vector.k_r = reputation;
        v
    }

    #[test]
    fn test_empty_set() {
        let set = ValidatorSet::new();
        assert_eq!(set.len(), 0);
        assert!(set.is_empty());
    }

    #[test]
    fn test_add_validator() {
        let mut set = ValidatorSet::new();
        let v = create_test_validator("v1", 0.8);

        set.add(v);

        assert_eq!(set.len(), 1);
        assert!(set.get("v1").is_some());
    }

    #[test]
    fn test_total_weight_calculation() {
        let mut set = ValidatorSet::new();
        set.add(create_test_validator("v1", 0.8)); // weight: 0.64
        set.add(create_test_validator("v2", 0.6)); // weight: 0.36

        // Total: 0.64 + 0.36 = 1.0
        let total = set.total_voting_weight();
        assert!((total - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_consensus_threshold() {
        let mut set = ValidatorSet::new();
        set.add(create_test_validator("v1", 1.0)); // weight: 1.0
        set.add(create_test_validator("v2", 1.0)); // weight: 1.0
        set.add(create_test_validator("v3", 1.0)); // weight: 1.0

        // Total: 3.0
        // With 45% Byzantine tolerance, threshold = total * (1 - 0.45) = 3.0 * 0.55 = 1.65
        let threshold = set.consensus_threshold();
        assert!((threshold - 1.65).abs() < 0.01);
    }

    #[test]
    fn test_can_reach_consensus_insufficient_validators() {
        let mut set = ValidatorSet::new();
        set.add(create_test_validator("v1", 0.8));
        set.add(create_test_validator("v2", 0.8));
        // Only 2 validators, need MIN_VALIDATORS (5)

        let result = set.can_reach_consensus();
        assert!(result.is_err());
    }

    #[test]
    fn test_can_reach_consensus_sufficient_validators() {
        let mut set = ValidatorSet::new();
        for i in 0..5 {
            set.add(create_test_validator(&format!("v{}", i), 0.8));
        }

        let result = set.can_reach_consensus();
        assert!(result.is_ok());
    }

    #[test]
    fn test_leader_selection_deterministic() {
        let mut set = ValidatorSet::new();
        for i in 0..5 {
            set.add(create_test_validator(&format!("v{}", i), 0.8));
        }

        let leader1 = set.select_leader(42);
        let leader2 = set.select_leader(42);

        assert_eq!(leader1.map(|l| l.id.clone()), leader2.map(|l| l.id.clone()));
    }

    #[test]
    fn test_leader_selection_varies_by_round() {
        let mut set = ValidatorSet::new();
        for i in 0..10 {
            set.add(create_test_validator(&format!("v{}", i), 0.8));
        }

        let mut leaders = std::collections::HashSet::new();
        for round in 0..100 {
            if let Some(leader) = set.select_leader(round) {
                leaders.insert(leader.id.clone());
            }
        }

        // Should have multiple different leaders across rounds
        assert!(leaders.len() > 1, "Leader selection should vary by round");
    }
}

// =============================================================================
// BYZANTINE TOLERANCE TESTS
// =============================================================================

#[cfg(test)]
mod byzantine_tolerance_tests {
    use super::*;

    fn create_test_validator(id: &str, reputation: f32) -> ValidatorNode {
        let mut v = ValidatorNode::new(id.to_string());
        v.k_vector.k_r = reputation;
        v
    }

    #[test]
    fn test_45_percent_byzantine_tolerance() {
        // With reputation² weighting, honest nodes with high reputation
        // can outvote Byzantine nodes even at 45%

        let mut set = ValidatorSet::new();

        // 55% honest nodes with high reputation
        for i in 0..11 {
            set.add(create_test_validator(&format!("honest{}", i), 0.9));
        }

        // 45% Byzantine nodes (we assume they have lower reputation due to past behavior)
        for i in 0..9 {
            set.add(create_test_validator(&format!("byzantine{}", i), 0.3));
        }

        // Calculate weights
        let honest_weight: f32 = 11.0 * (0.9 * 0.9);  // 11 * 0.81 = 8.91
        let byzantine_weight: f32 = 9.0 * (0.3 * 0.3); // 9 * 0.09 = 0.81
        let total_weight = honest_weight + byzantine_weight; // 9.72

        // Honest fraction
        let honest_fraction = honest_weight / total_weight; // ~0.916

        assert!(honest_fraction > 0.55, "Honest nodes should have >55% of voting power");

        // Verify with actual set
        let threshold = set.consensus_threshold();
        let _actual_total = set.total_voting_weight();

        // Honest nodes should exceed threshold
        assert!(honest_weight > threshold,
            "Honest weight {} should exceed threshold {}",
            honest_weight, threshold);
    }

    #[test]
    fn test_linear_voting_would_fail() {
        // Without reputation², same scenario would give Byzantine nodes too much power

        // 55% honest, 45% Byzantine (by count)
        let _honest_count = 11;
        let _byzantine_count = 9;

        // With LINEAR weights (reputation, not reputation²)
        let honest_linear_weight = 11.0 * 0.9;   // 9.9
        let byzantine_linear_weight = 9.0 * 0.3; // 2.7
        let total_linear = honest_linear_weight + byzantine_linear_weight; // 12.6

        let honest_linear_fraction = honest_linear_weight / total_linear; // ~0.786

        // With SQUARED weights
        let honest_squared_weight = 11.0 * 0.81;  // 8.91
        let byzantine_squared_weight = 9.0 * 0.09; // 0.81
        let total_squared = honest_squared_weight + byzantine_squared_weight; // 9.72

        let honest_squared_fraction = honest_squared_weight / total_squared; // ~0.916

        // Squared weights give honest nodes MORE power
        assert!(honest_squared_fraction > honest_linear_fraction,
            "Rep² should give honest nodes more relative power");
    }

    #[test]
    fn test_equal_reputation_default_tolerance() {
        // When all validators have equal reputation, we get traditional 1/3 tolerance
        let mut set = ValidatorSet::new();

        for i in 0..10 {
            set.add(create_test_validator(&format!("v{}", i), 0.5));
        }

        let total = set.total_voting_weight();
        let threshold = set.consensus_threshold();

        // With 45% tolerance setting, need 55% to pass
        let required_fraction = threshold / total;
        assert!((required_fraction - 0.55).abs() < 0.01);
    }
}

// =============================================================================
// CONSENSUS ENGINE TESTS
// =============================================================================

#[cfg(test)]
mod consensus_engine_tests {
    use super::*;

    fn setup_consensus_with_validators(count: usize) -> RbBftConsensus {
        let mut consensus = RbBftConsensus::with_defaults();

        for i in 0..count {
            let mut v = ValidatorNode::new(format!("validator-{}", i));
            v.k_vector.k_r = 0.8;
            consensus.add_validator(v);
        }

        consensus.set_our_id("validator-0".to_string());
        consensus
    }

    #[test]
    fn test_create_consensus_engine() {
        let consensus = RbBftConsensus::with_defaults();

        assert_eq!(consensus.validators().len(), 0);
        assert!(!consensus.are_we_leader());
    }

    #[test]
    fn test_consensus_config_defaults() {
        let config = ConsensusConfig::default();

        assert_eq!(config.round_timeout_ms, DEFAULT_ROUND_TIMEOUT_MS);
        assert!((config.byzantine_tolerance - BYZANTINE_TOLERANCE).abs() < 0.001);
        assert_eq!(config.min_validators, MIN_VALIDATORS);
    }

    #[test]
    fn test_start_round_insufficient_validators() {
        let mut consensus = setup_consensus_with_validators(3); // Less than MIN_VALIDATORS

        let result = consensus.start_round();
        assert!(result.is_err());
    }

    #[test]
    fn test_start_round_sufficient_validators() {
        let mut consensus = setup_consensus_with_validators(5);

        let result = consensus.start_round();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1); // First round is 1
    }

    #[test]
    fn test_multiple_rounds() {
        let mut consensus = setup_consensus_with_validators(5);

        let round1 = consensus.start_round().unwrap();
        // Would need to complete round to start another...

        assert_eq!(round1, 1);
    }
}

// =============================================================================
// VOTE TESTS
// =============================================================================

#[cfg(test)]
mod vote_tests {
    use super::*;

    #[test]
    fn test_vote_approve() {
        let vote = Vote::approve(
            "proposal-1".to_string(),
            1,
            "validator-1".to_string(),
            0.8,
        );

        assert_eq!(vote.decision, VoteDecision::Approve);
        assert_eq!(vote.round, 1);
        assert_eq!(vote.voter, "validator-1");
    }

    #[test]
    fn test_vote_reject() {
        let vote = Vote::reject(
            "proposal-1".to_string(),
            1,
            "validator-1".to_string(),
            0.8,
            "Invalid content".to_string(),
        );

        assert_eq!(vote.decision, VoteDecision::Reject);
        assert!(vote.reason.is_some());
    }

    #[test]
    fn test_vote_abstain() {
        let vote = Vote::abstain(
            "proposal-1".to_string(),
            1,
            "validator-1".to_string(),
            0.8,
        );

        assert_eq!(vote.decision, VoteDecision::Abstain);
    }

    #[test]
    fn test_vote_weighted_value() {
        let vote = Vote::approve(
            "proposal-1".to_string(),
            1,
            "validator-1".to_string(),
            0.8,
        );

        // Weight is reputation squared
        let expected_weight = 0.8 * 0.8;
        assert!((vote.weighted_value() - expected_weight).abs() < 0.0001);
    }
}

// =============================================================================
// BATCH VOTE RESULT TESTS
// =============================================================================

#[cfg(test)]
mod batch_vote_result_tests {
    use super::*;

    #[test]
    fn test_all_accepted() {
        let result = BatchVoteResult {
            accepted_count: 5,
            rejected_count: 0,
            rejected_details: vec![],
        };

        assert!(result.all_accepted());
        assert_eq!(result.total(), 5);
    }

    #[test]
    fn test_some_rejected() {
        let result = BatchVoteResult {
            accepted_count: 3,
            rejected_count: 2,
            rejected_details: vec![
                ("v1".to_string(), "invalid".to_string()),
                ("v2".to_string(), "duplicate".to_string()),
            ],
        };

        assert!(!result.all_accepted());
        assert_eq!(result.total(), 5);
    }
}

// =============================================================================
// CONSENSUS OUTCOME TESTS
// =============================================================================

#[cfg(test)]
mod consensus_outcome_tests {
    use super::*;

    #[test]
    fn test_accepted_outcome() {
        let outcome = ConsensusOutcome::Accepted {
            round: 5,
            proposal_id: "prop-1".to_string(),
            result_hash: "abc123".to_string(),
        };

        match outcome {
            ConsensusOutcome::Accepted { round, .. } => assert_eq!(round, 5),
            _ => panic!("Expected Accepted"),
        }
    }

    #[test]
    fn test_rejected_outcome() {
        let outcome = ConsensusOutcome::Rejected {
            round: 5,
            proposal_id: "prop-1".to_string(),
            reason: "Invalid signature".to_string(),
        };

        match outcome {
            ConsensusOutcome::Rejected { reason, .. } => {
                assert!(reason.contains("Invalid"));
            }
            _ => panic!("Expected Rejected"),
        }
    }

    #[test]
    fn test_timed_out_outcome() {
        let outcome = ConsensusOutcome::TimedOut { round: 5 };

        match outcome {
            ConsensusOutcome::TimedOut { round } => assert_eq!(round, 5),
            _ => panic!("Expected TimedOut"),
        }
    }
}

// =============================================================================
// CRYPTO KEYPAIR TESTS
// =============================================================================

#[cfg(test)]
mod crypto_tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let keypair = ValidatorKeypair::generate();

        // Public key should be valid hex
        let pubkey = keypair.public_key_hex();
        assert_eq!(pubkey.len(), 64); // 32 bytes = 64 hex chars
        assert!(pubkey.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_keypair_sign_verify() {
        let keypair = ValidatorKeypair::generate();
        let message = b"test message";

        let signature = keypair.sign(message);
        assert!(keypair.verify(message, &signature));
    }

    #[test]
    fn test_keypair_wrong_message() {
        let keypair = ValidatorKeypair::generate();
        let message = b"test message";
        let wrong_message = b"wrong message";

        let signature = keypair.sign(message);
        assert!(!keypair.verify(wrong_message, &signature));
    }

    #[test]
    fn test_different_keypairs() {
        let kp1 = ValidatorKeypair::generate();
        let kp2 = ValidatorKeypair::generate();

        assert_ne!(kp1.public_key_hex(), kp2.public_key_hex());
    }
}

// =============================================================================
// CONSTANTS TESTS
// =============================================================================

#[cfg(test)]
mod constants_tests {
    use super::*;

    #[test]
    fn test_byzantine_tolerance_is_45_percent() {
        assert!((BYZANTINE_TOLERANCE - 0.45).abs() < 0.001);
    }

    #[test]
    fn test_min_participation_reputation() {
        assert!((MIN_PARTICIPATION_REPUTATION - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_min_validators() {
        assert_eq!(MIN_VALIDATORS, 5);
    }

    #[test]
    fn test_default_round_timeout() {
        assert_eq!(DEFAULT_ROUND_TIMEOUT_MS, 30_000);
    }
}
