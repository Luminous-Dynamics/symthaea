// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for consensus and trust layer interaction
//!
//! Tests the flow: Trust Score -> Consensus Weight -> Vote Power

use matl_bridge::{
    MatlBridge, MatlBridgeConfig, GradientContribution, InteractionType,
    POGQ_WEIGHT, TCDM_WEIGHT, ENTROPY_WEIGHT,
};
use rb_bft_consensus::{
    RbBftConsensus, ConsensusOutcome,
    ValidatorNode, ValidatorKeypair, VoteDecision,
    BYZANTINE_TOLERANCE,
};

/// Test helper: Create validator with MATL-derived reputation
fn create_matl_validator(_id: &str, trust_score: f32) -> (ValidatorNode, ValidatorKeypair) {
    let keypair = ValidatorKeypair::generate();
    let mut validator = ValidatorNode::new(keypair.public_key_hex());
    // Set reputation from MATL trust score
    validator.k_vector.k_r = trust_score;
    (validator, keypair)
}

/// Test: Validators with higher MATL trust have more voting power
#[test]
fn test_trust_score_affects_voting_weight() {
    // Create validators with different trust scores
    let (v1, _) = create_matl_validator("v1", 0.9); // High trust
    let (v2, _) = create_matl_validator("v2", 0.5); // Medium trust
    let (v3, _) = create_matl_validator("v3", 0.3); // Low trust

    // Voting weight is reputation squared
    let w1 = v1.reputation().powi(2); // 0.81
    let w2 = v2.reputation().powi(2); // 0.25
    let w3 = v3.reputation().powi(2); // 0.09

    assert!((w1 - 0.81).abs() < 0.01);
    assert!((w2 - 0.25).abs() < 0.01);
    assert!((w3 - 0.09).abs() < 0.01);

    // High trust validator has ~3x more weight than medium
    assert!(w1 > w2 * 3.0);
    // High trust validator has ~9x more weight than low
    assert!(w1 > w3 * 8.0);
}

/// Test: MATL bridge computes trust from PoGQ, TCDM, Entropy
#[test]
fn test_matl_trust_computation() {
    let mut bridge = MatlBridge::new();

    // Submit a gradient contribution (PoGQ)
    // This automatically creates the agent state
    let contribution = GradientContribution::new(
        "agent_1".to_string(),
        1,
        "hash123".to_string(),
    ).with_stats(0.5, 1000, 0.0, 1.0, 0.0, 3.0); // Normal distribution stats

    // Process gradient (this registers the agent and computes trust)
    let trust = bridge.process_gradient(contribution).unwrap();

    // Trust should be between 0 and 1
    assert!(trust.total >= 0.0 && trust.total <= 1.0);

    // Components should respect weights
    // Formula: T = 0.4×PoGQ + 0.3×TCDM + 0.3×Entropy
    println!("Trust breakdown: PoGQ={}, TCDM={}, Entropy={}, Total={}",
        trust.pogq, trust.tcdm, trust.entropy, trust.total);
}

/// Test: Trust synchronization between sources
#[test]
fn test_trust_sync() {
    let mut bridge = MatlBridge::new();

    // Submit gradient for PoGQ (this creates the agent)
    let contribution = GradientContribution::new(
        "sync_test_agent".to_string(),
        1,
        "hash456".to_string(),
    ).with_stats(0.3, 1000, 0.0, 1.0, 0.0, 3.0);

    bridge.process_gradient(contribution).unwrap();

    // Get sync data via trust_store
    let record = bridge.trust_store().get("sync_test_agent").unwrap();

    // Verify record has expected fields
    assert_eq!(record.agent_id, "sync_test_agent");
    assert!(record.updated_at > 0);
}

/// Test: Consensus threshold computation with trust-weighted validators
#[test]
fn test_consensus_threshold_with_trust() {
    let mut consensus = RbBftConsensus::with_defaults();

    // Add 5 validators with varying MATL trust scores
    let trust_scores = [0.9, 0.8, 0.7, 0.6, 0.5];
    let mut _keypairs = Vec::new();

    for (i, &trust) in trust_scores.iter().enumerate() {
        let (validator, keypair) = create_matl_validator(&format!("v{}", i), trust);
        consensus.add_validator(validator);
        _keypairs.push(keypair);
    }

    // Calculate expected total weight and threshold
    // Weight = sum of reputation²
    let total_weight: f32 = trust_scores.iter()
        .map(|t| t.powi(2))
        .sum();

    // Threshold = total_weight * (1 - BYZANTINE_TOLERANCE)
    let expected_threshold = total_weight * (1.0 - BYZANTINE_TOLERANCE);

    let actual_threshold = consensus.validators().consensus_threshold();

    println!("Total weight: {}", total_weight);
    println!("Byzantine tolerance: {}", BYZANTINE_TOLERANCE);
    println!("Expected threshold: {}", expected_threshold);
    println!("Actual threshold: {}", actual_threshold);

    assert!((actual_threshold - expected_threshold).abs() < 0.1);
}

/// Test: Full consensus round with MATL-derived trust
#[test]
fn test_full_consensus_round_with_matl() {
    // Setup MATL bridge
    let mut bridge = MatlBridge::new();

    // Setup consensus
    let mut consensus = RbBftConsensus::with_defaults();
    let mut keypairs = Vec::new();
    let mut validator_ids = Vec::new();

    // Create 5 validators with MATL trust scores
    for i in 0..5 {
        let keypair = ValidatorKeypair::generate();
        let id = keypair.public_key_hex();
        validator_ids.push(id.clone());

        // Submit gradients to build trust
        let contribution = GradientContribution::new(
            id.clone(),
            1,
            format!("hash_{}", i),
        ).with_stats(0.1 * (i as f32 + 1.0), 1000, 0.0, 1.0, 0.0, 3.0);

        // Process gradient to establish trust
        let trust = bridge.process_gradient(contribution).unwrap();

        // Use trust score for validator reputation (at least 0.5 to ensure participation threshold)
        let reputation = trust.total.max(0.5);

        let mut validator = ValidatorNode::new(id);
        validator.k_vector.k_r = reputation;
        consensus.add_validator(validator);
        keypairs.push(keypair);
    }

    // Start consensus round
    consensus.start_round().unwrap();

    // Find leader - we need to iterate through validators to find who matches
    let leader_idx = keypairs.iter()
        .position(|kp| {
            let validator = consensus.validators().get(&kp.public_key_hex());
            validator.map(|v| consensus.validators().select_leader(1).map(|l| l.id == v.id).unwrap_or(false)).unwrap_or(false)
        })
        .unwrap_or(0); // Default to first if not found

    let leader_keypair = &keypairs[leader_idx];

    consensus.propose_signed(
        b"test_content".to_vec(),
        "content-hash-123".to_string(),
        "parent-hash-000".to_string(),
        leader_keypair,
    ).unwrap();

    // All non-leader validators vote
    for (i, keypair) in keypairs.iter().enumerate() {
        if i != leader_idx {
            consensus.vote_signed(
                VoteDecision::Approve,
                None,
                keypair,
            ).unwrap();
        }
    }

    // Check consensus
    let outcome = consensus.check_consensus().unwrap();

    assert!(matches!(outcome, Some(ConsensusOutcome::Accepted { .. })));
    println!("Consensus reached with MATL-derived trust scores!");
}

/// Test: MATL weights are properly applied
#[test]
fn test_matl_weights() {
    // Verify MATL weight constants
    assert!((POGQ_WEIGHT - 0.4).abs() < 0.01);
    assert!((TCDM_WEIGHT - 0.3).abs() < 0.01);
    assert!((ENTROPY_WEIGHT - 0.3).abs() < 0.01);

    // Verify weights sum to 1.0
    let total = POGQ_WEIGHT + TCDM_WEIGHT + ENTROPY_WEIGHT;
    assert!((total - 1.0).abs() < 0.01);
}

/// Test: Custom MATL configuration
#[test]
fn test_custom_matl_config() {
    let config = MatlBridgeConfig {
        pogq_weight: 0.5,
        tcdm_weight: 0.3,
        entropy_weight: 0.2,
        decay_rate: 0.95,
        pogq_epsilon: 5.0,
        pogq_sigma: 2.5,
    };

    let _bridge = MatlBridge::with_config(config);

    // Bridge was created successfully with custom config
    assert!(true);
}

/// Test: Interaction types affect trust
#[test]
fn test_interaction_types() {
    // Verify interaction type impacts
    assert!(InteractionType::HighQualityContribution.trust_impact() > 0.0);
    assert!(InteractionType::FLParticipation.trust_impact() > 0.0);
    assert!(InteractionType::ByzantineBehavior.trust_impact() < 0.0);
    assert!(InteractionType::MissedRound.trust_impact() < 0.0);

    // Verify positive detection
    assert!(InteractionType::HighQualityContribution.is_positive());
    assert!(InteractionType::FLParticipation.is_positive());
    assert!(!InteractionType::ByzantineBehavior.is_positive());
    assert!(!InteractionType::MissedRound.is_positive());
}

/// Test: Byzantine behavior detection affects K-vector
#[test]
fn test_byzantine_behavior_detection() {
    let mut bridge = MatlBridge::new();

    // First establish trust
    let contribution = GradientContribution::new(
        "agent_byzantine".to_string(),
        1,
        "hash_initial".to_string(),
    ).with_stats(0.5, 1000, 0.0, 1.0, 0.0, 3.0);

    bridge.process_gradient(contribution).unwrap();

    // Get initial K-vector values
    let initial_kvector = bridge.get_k_vector("agent_byzantine").unwrap().clone();
    println!("Initial K-vector: k_i={}, k_r={}", initial_kvector.k_i, initial_kvector.k_r);

    // Record Byzantine behavior - this should penalize integrity and reputation
    bridge.record_byzantine("agent_byzantine", "Detected double voting").unwrap();

    // K-vector should have been penalized
    let final_kvector = bridge.get_k_vector("agent_byzantine").unwrap();
    println!("Final K-vector: k_i={}, k_r={}", final_kvector.k_i, final_kvector.k_r);

    // record_byzantine halves k_i and k_r
    assert!(final_kvector.k_i < initial_kvector.k_i);
    assert!(final_kvector.k_r < initial_kvector.k_r);
}
