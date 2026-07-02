// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consensus integration tests
//!
//! Tests the interaction between:
//! - RB-BFT consensus with reputation weighting
//! - DKG for threshold signatures
//! - K-Vector trust computation

use feldman_dkg::{
    ceremony::{DkgCeremony, DkgConfig},
    dealer::Dealer,
    participant::ParticipantId,
};
use mycelix_core_types::KVector;
use rb_bft_consensus::{ConsensusConfig, RbBftConsensus, ValidatorNode};

/// Simulates a full validator node with all components
struct ValidatorInfo {
    id: String,
    k_vector: KVector,
    reputation: f32,
    dkg_participant_id: ParticipantId,
}

impl ValidatorInfo {
    fn new(id: &str, participant_num: u32, k_vector: KVector) -> Self {
        let reputation = k_vector.trust_score();
        Self {
            id: id.to_string(),
            k_vector,
            reputation,
            dkg_participant_id: ParticipantId(participant_num),
        }
    }
}

/// Test DKG + Consensus integration with a validator committee
#[test]
fn test_dkg_with_validator_committee() {
    let num_validators = 5;
    let threshold = 3; // 3-of-5 for finality

    // Create validators with different K-Vectors
    let validators: Vec<ValidatorInfo> = vec![
        ValidatorInfo::new(
            "v1",
            1,
            KVector {
                k_r: 0.9, k_a: 0.85, k_i: 0.95, k_p: 0.88,
                k_m: 0.7, k_s: 0.8, k_h: 0.9, k_topo: 0.85,
            },
        ),
        ValidatorInfo::new(
            "v2",
            2,
            KVector {
                k_r: 0.85, k_a: 0.8, k_i: 0.9, k_p: 0.82,
                k_m: 0.65, k_s: 0.75, k_h: 0.85, k_topo: 0.8,
            },
        ),
        ValidatorInfo::new(
            "v3",
            3,
            KVector {
                k_r: 0.75, k_a: 0.7, k_i: 0.85, k_p: 0.75,
                k_m: 0.5, k_s: 0.65, k_h: 0.78, k_topo: 0.7,
            },
        ),
        ValidatorInfo::new(
            "v4",
            4,
            KVector {
                k_r: 0.5, k_a: 0.45, k_i: 0.6, k_p: 0.55,
                k_m: 0.3, k_s: 0.4, k_h: 0.5, k_topo: 0.45,
            },
        ),
        ValidatorInfo::new(
            "v5",
            5,
            KVector {
                k_r: 0.3, k_a: 0.25, k_i: 0.4, k_p: 0.35,
                k_m: 0.15, k_s: 0.2, k_h: 0.3, k_topo: 0.25,
            },
        ),
    ];

    // Print computed reputations
    println!("\nValidator Reputations (from K-Vector trust scores):");
    for v in &validators {
        println!("  {}: {:.4}", v.id, v.reputation);
    }

    // Run DKG ceremony for threshold signing keys
    let config = DkgConfig::new(threshold, num_validators).expect("Valid config");
    let mut ceremony = DkgCeremony::new(config);

    for v in &validators {
        ceremony
            .add_participant(v.dkg_participant_id)
            .expect("Registration should succeed");
    }

    // Each validator contributes to DKG
    for v in &validators {
        let dealer = Dealer::new(v.dkg_participant_id, threshold, num_validators)
            .expect("Valid dealer");
        let deal = dealer.generate_deal();
        ceremony
            .submit_deal(v.dkg_participant_id, deal)
            .expect("Deal should succeed");
    }

    let dkg_result = ceremony.finalize().expect("DKG should succeed");
    println!("\nDKG Result:");
    println!("  Qualified participants: {}", dkg_result.qualified_count);
    assert_eq!(dkg_result.qualified_count, num_validators);

    // Verify threshold signatures would work
    // Get combined shares from top-3 validators (threshold)
    let top_validators: Vec<_> = validators
        .iter()
        .take(threshold)
        .map(|v| v.dkg_participant_id)
        .collect();

    let shares: Vec<_> = top_validators
        .iter()
        .map(|&pid| ceremony.get_combined_share(pid).expect("Should get share"))
        .collect();

    let _reconstructed = ceremony
        .reconstruct_secret(&shares)
        .expect("Threshold reconstruction should work");

    println!("\nThreshold signature simulation successful!");
    println!("  Signers: {:?}", top_validators);
}

/// Test consensus validator setup with DKG-derived committee
#[test]
fn test_consensus_validator_setup() {
    let num_validators = 5;
    let threshold = 3;

    // Step 1: Run DKG
    let config = DkgConfig::new(threshold, num_validators).expect("Valid config");
    let mut ceremony = DkgCeremony::new(config);

    for i in 1..=num_validators {
        ceremony
            .add_participant(ParticipantId(i as u32))
            .expect("Registration should succeed");
    }

    for i in 1..=num_validators {
        let dealer = Dealer::new(ParticipantId(i as u32), threshold, num_validators)
            .expect("Valid dealer");
        let deal = dealer.generate_deal();
        ceremony
            .submit_deal(ParticipantId(i as u32), deal)
            .expect("Deal should succeed");
    }

    let dkg_result = ceremony.finalize().expect("DKG should succeed");
    assert_eq!(dkg_result.qualified_count, num_validators);

    // Step 2: Initialize RB-BFT consensus
    let consensus_config = ConsensusConfig::default();
    let mut consensus = RbBftConsensus::new(consensus_config);

    // Register validators with trust-based reputations from K-Vectors
    let reputations = [0.9, 0.85, 0.75, 0.5, 0.3];
    for (i, &rep) in reputations.iter().enumerate() {
        let mut validator = ValidatorNode::new(format!("v{}", i + 1));
        validator.k_vector.k_r = rep; // Set reputation
        consensus.add_validator(validator);
    }

    // Verify validator set is ready
    let validator_set = consensus.validators();
    assert_eq!(validator_set.active_count(), 5);

    // Check total weight calculation
    let total_weight = validator_set.total_weight();
    let expected_weight: f32 = reputations.iter().map(|r| r * r).sum(); // reputation² weighting
    assert!((total_weight - expected_weight).abs() < 0.01);

    println!("\nConsensus Validator Setup:");
    println!("  Active validators: {}", validator_set.active_count());
    println!("  Total voting weight: {:.4}", total_weight);
    println!("  Consensus threshold: {:.4}", validator_set.consensus_threshold());
}

/// Test weighted voting calculation for RB-BFT
#[test]
fn test_weighted_voting_calculation() {
    // Simulate weighted voting calculation
    let reputations = [0.9, 0.85, 0.75, 0.5, 0.3];

    // Calculate reputation² weights
    let weights: Vec<f32> = reputations.iter().map(|r| r * r).collect();
    let total_weight: f32 = weights.iter().sum();

    println!("\nWeighted Voting Analysis:");
    println!("  Reputations: {:?}", reputations);
    println!("  Weights (rep²): {:?}", weights);
    println!("  Total weight: {:.4}", total_weight);

    // High-trust validators (v1, v2, v3) approve
    let approve_weight: f32 = weights[0..3].iter().sum();
    // Low-trust validators (v4, v5) reject
    let reject_weight: f32 = weights[3..5].iter().sum();

    println!("  Approve weight: {:.4} ({:.1}%)", approve_weight, approve_weight / total_weight * 100.0);
    println!("  Reject weight: {:.4} ({:.1}%)", reject_weight, reject_weight / total_weight * 100.0);

    // With reputation² weighting, high-trust validators should dominate
    assert!(approve_weight > reject_weight * 2.0, "High-trust validators should have much more weight");

    // Consensus threshold is 55% of total weight
    let threshold = total_weight * 0.55;
    assert!(approve_weight > threshold, "High-trust approval should exceed threshold");
}

/// Test Byzantine fault tolerance with reputation weighting
#[test]
fn test_byzantine_fault_tolerance_with_reputation() {
    // Scenario: 3 honest high-rep validators vs 4 Byzantine low-rep validators
    // Traditional BFT would fail (4 > 3), but RB-BFT should succeed due to reputation weighting

    let honest_reps = [0.9, 0.85, 0.8]; // 3 honest validators
    let byzantine_reps = [0.2, 0.15, 0.15, 0.1]; // 4 Byzantine validators

    // Calculate reputation² weights
    let honest_weight: f32 = honest_reps.iter().map(|r| r * r).sum();
    let byzantine_weight: f32 = byzantine_reps.iter().map(|r| r * r).sum();

    println!("\nByzantine Fault Tolerance Analysis:");
    println!("  Honest validators: {} with reps {:?}", honest_reps.len(), honest_reps);
    println!("  Byzantine validators: {} with reps {:?}", byzantine_reps.len(), byzantine_reps);
    println!("  Honest weight (rep²): {:.4}", honest_weight);
    println!("  Byzantine weight (rep²): {:.4}", byzantine_weight);
    println!("  Honest percentage: {:.1}%", honest_weight / (honest_weight + byzantine_weight) * 100.0);

    // Even though Byzantine validators are majority by count (4 vs 3),
    // honest validators should win due to reputation² weighting
    assert!(
        honest_weight > byzantine_weight,
        "Reputation-weighted honest validators should outweigh Byzantine majority"
    );

    // Calculate equivalent Byzantine tolerance
    let byzantine_fraction = byzantine_weight / (honest_weight + byzantine_weight);
    println!("  Effective Byzantine fraction: {:.1}%", byzantine_fraction * 100.0);

    assert!(
        byzantine_fraction < 0.45,
        "Byzantine fraction should be below 45% tolerance"
    );
}

/// Test DKG key rotation
#[test]
fn test_dkg_key_rotation() {
    let threshold = 2;
    let num_participants = 3;

    // Epoch 1: Initial DKG
    let mut epoch1_ceremony = DkgCeremony::new(
        DkgConfig::new(threshold, num_participants).unwrap()
    );

    for i in 1..=num_participants {
        epoch1_ceremony.add_participant(ParticipantId(i as u32)).unwrap();
    }

    for i in 1..=num_participants {
        let dealer = Dealer::new(ParticipantId(i as u32), threshold, num_participants).unwrap();
        epoch1_ceremony.submit_deal(ParticipantId(i as u32), dealer.generate_deal()).unwrap();
    }

    let epoch1_result = epoch1_ceremony.finalize().unwrap();
    let epoch1_pubkey = epoch1_result.public_key.to_bytes();

    // Epoch 2: New DKG (key rotation)
    let mut epoch2_ceremony = DkgCeremony::new(
        DkgConfig::new(threshold, num_participants).unwrap()
    );

    for i in 1..=num_participants {
        epoch2_ceremony.add_participant(ParticipantId(i as u32)).unwrap();
    }

    for i in 1..=num_participants {
        let dealer = Dealer::new(ParticipantId(i as u32), threshold, num_participants).unwrap();
        epoch2_ceremony.submit_deal(ParticipantId(i as u32), dealer.generate_deal()).unwrap();
    }

    let epoch2_result = epoch2_ceremony.finalize().unwrap();
    let epoch2_pubkey = epoch2_result.public_key.to_bytes();

    // Public keys should be different (new random secrets)
    assert_ne!(epoch1_pubkey, epoch2_pubkey, "Rotated keys should be different");

    println!("\nKey Rotation Test:");
    println!("  Epoch 1 public key: {} bytes", epoch1_pubkey.len());
    println!("  Epoch 2 public key: {} bytes", epoch2_pubkey.len());
    println!("  Keys are different: {}", epoch1_pubkey != epoch2_pubkey);
}

// ============================================================
// ZK Proof + Consensus Integration Tests
// ============================================================

use kvector_zkp::{KVectorWitness, proof::KVectorRangeProof};

/// Test validator onboarding with ZK-verified K-Vector
/// This is the full flow: prove K-Vector → verify → join consensus
#[test]
fn test_validator_onboarding_with_zkp() {
    // New validator wants to join with their K-Vector
    let validator_k_vector = KVector {
        k_r: 0.85,
        k_a: 0.78,
        k_i: 0.92,
        k_p: 0.80,
        k_m: 0.60,
        k_s: 0.70,
        k_h: 0.88,
        k_topo: 0.75,
    };

    // Step 1: Validator creates ZK proof of K-Vector validity
    let witness = KVectorWitness {
        k_r: validator_k_vector.k_r,
        k_a: validator_k_vector.k_a,
        k_i: validator_k_vector.k_i,
        k_p: validator_k_vector.k_p,
        k_m: validator_k_vector.k_m,
        k_s: validator_k_vector.k_s,
        k_h: validator_k_vector.k_h,
        k_topo: validator_k_vector.k_topo,
    };

    let proof = KVectorRangeProof::prove(&witness)
        .expect("Validator should be able to prove their K-Vector");

    // Step 2: Network verifies the proof
    proof.verify().expect("Network should accept valid proof");

    // Step 3: Only after verification, validator joins consensus
    let consensus_config = ConsensusConfig::default();
    let mut consensus = RbBftConsensus::new(consensus_config);

    let mut validator = ValidatorNode::new("new-validator".to_string());
    validator.k_vector = validator_k_vector;
    consensus.add_validator(validator);

    // Step 4: Verify validator is active with correct reputation
    let validator_set = consensus.validators();
    assert_eq!(validator_set.active_count(), 1);

    let trust_score = validator_k_vector.trust_score();
    println!("\nValidator Onboarding with ZKP:");
    println!("  K-Vector proof verified: true");
    println!("  Trust score: {:.4}", trust_score);
    println!("  Validator active in consensus: true");
}

/// Test that fraudulent K-Vectors cannot pass ZK verification
#[test]
fn test_fraudulent_kvector_rejected() {
    // Attacker tries to claim impossibly good K-Vector
    let fraudulent_witness = KVectorWitness {
        k_r: 1.5, // Invalid: > 1.0
        k_a: 0.95,
        k_i: 0.95,
        k_p: 0.95,
        k_m: 0.95,
        k_s: 0.95,
        k_h: 0.95,
        k_topo: 0.95,
    };

    // ZK proof should fail for out-of-range values
    let result = KVectorRangeProof::prove(&fraudulent_witness);
    assert!(result.is_err(), "Fraudulent K-Vector should be rejected");

    println!("\nFraudulent K-Vector Rejection:");
    println!("  Out-of-range value detected: true");
    println!("  Proof generation rejected: true");
}

/// Test multi-validator committee formation with ZK-verified K-Vectors
#[test]
fn test_committee_formation_with_zkp() {
    let validator_data = vec![
        ("v1", KVector { k_r: 0.95, k_a: 0.90, k_i: 0.98, k_p: 0.92, k_m: 0.85, k_s: 0.80, k_h: 0.95, k_topo: 0.88 }),
        ("v2", KVector { k_r: 0.80, k_a: 0.75, k_i: 0.85, k_p: 0.78, k_m: 0.60, k_s: 0.65, k_h: 0.82, k_topo: 0.70 }),
        ("v3", KVector { k_r: 0.65, k_a: 0.60, k_i: 0.70, k_p: 0.62, k_m: 0.45, k_s: 0.50, k_h: 0.68, k_topo: 0.55 }),
        ("v4", KVector { k_r: 0.50, k_a: 0.45, k_i: 0.55, k_p: 0.48, k_m: 0.30, k_s: 0.35, k_h: 0.52, k_topo: 0.40 }),
    ];

    let consensus_config = ConsensusConfig::default();
    let mut consensus = RbBftConsensus::new(consensus_config);

    println!("\nCommittee Formation with ZK Proofs:");

    for (name, k_vec) in &validator_data {
        // Each validator proves their K-Vector
        let witness = KVectorWitness {
            k_r: k_vec.k_r, k_a: k_vec.k_a, k_i: k_vec.k_i, k_p: k_vec.k_p,
            k_m: k_vec.k_m, k_s: k_vec.k_s, k_h: k_vec.k_h, k_topo: k_vec.k_topo,
        };

        let proof = KVectorRangeProof::prove(&witness)
            .expect(&format!("{} proof should succeed", name));

        proof.verify()
            .expect(&format!("{} proof should verify", name));

        // Only after verification, add to consensus
        let mut validator = ValidatorNode::new(name.to_string());
        validator.k_vector = k_vec.clone();
        consensus.add_validator(validator);

        println!("  {} joined (trust: {:.4}, proof verified)", name, k_vec.trust_score());
    }

    let validator_set = consensus.validators();
    assert_eq!(validator_set.active_count(), 4);

    // Calculate total voting power
    let total_weight = validator_set.total_weight();
    println!("  Total committee voting weight: {:.4}", total_weight);
    println!("  Consensus threshold: {:.4}", validator_set.consensus_threshold());
}

/// Test K-Vector to validator reputation conversion
#[test]
fn test_kvector_to_validator_reputation() {
    // High-trust K-Vector
    let high_trust = KVector {
        k_r: 0.95,
        k_a: 0.9,
        k_i: 0.98,
        k_p: 0.92,
        k_m: 0.85,
        k_s: 0.8,
        k_h: 0.95,
        k_topo: 0.88,
    };

    // Low-trust K-Vector
    let low_trust = KVector {
        k_r: 0.3,
        k_a: 0.2,
        k_i: 0.5,
        k_p: 0.4,
        k_m: 0.1,
        k_s: 0.15,
        k_h: 0.3,
        k_topo: 0.25,
    };

    let high_score = high_trust.trust_score();
    let low_score = low_trust.trust_score();

    // Create validators with these K-Vectors
    let mut high_validator = ValidatorNode::new("high-trust".to_string());
    high_validator.k_vector = high_trust;

    let mut low_validator = ValidatorNode::new("low-trust".to_string());
    low_validator.k_vector = low_trust;

    // Voting weights are reputation² (using k_r for voting weight)
    let high_weight = high_validator.voting_weight();
    let low_weight = low_validator.voting_weight();

    println!("\nK-Vector to Validator Conversion:");
    println!("  High-trust score: {:.4}", high_score);
    println!("  Low-trust score: {:.4}", low_score);
    println!("  High voting weight (k_r²): {:.4}", high_weight);
    println!("  Low voting weight (k_r²): {:.4}", low_weight);
    println!("  Weight ratio: {:.1}x", high_weight / low_weight);

    assert!(high_score > 0.8, "High-trust score should exceed 0.8");
    assert!(low_score < 0.4, "Low-trust score should be below 0.4");
    assert!(high_weight > low_weight * 5.0, "High-trust voting power should be much greater");
}
