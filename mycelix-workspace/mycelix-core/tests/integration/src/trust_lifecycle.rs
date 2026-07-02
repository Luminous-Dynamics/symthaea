// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust lifecycle integration tests
//!
//! Tests the full journey of an agent:
//! 1. Join network with initial K-Vector
//! 2. Prove K-Vector validity with ZKP
//! 3. Compute trust score

use kvector_zkp::KVectorWitness;
use mycelix_core_types::KVector;

/// Simulates an agent in the Mycelix network
struct SimulatedAgent {
    id: String,
    k_vector: KVector,
    trust_score: f32,
}

impl SimulatedAgent {
    fn with_kvector(id: &str, k_vector: KVector) -> Self {
        let trust_score = k_vector.trust_score();
        Self {
            id: id.to_string(),
            k_vector,
            trust_score,
        }
    }
}

/// Test the complete trust lifecycle for a new agent
#[test]
fn test_agent_trust_lifecycle() {
    // Step 1: Agent joins with initial K-Vector
    let agent = SimulatedAgent::with_kvector(
        "agent_alice",
        KVector {
            k_r: 0.5,    // Initial reputation
            k_a: 0.3,    // Low activity (new)
            k_i: 0.8,    // High integrity
            k_p: 0.6,    // Moderate performance
            k_m: 0.1,    // Short membership
            k_s: 0.4,    // Some stake
            k_h: 0.7,    // Good history
            k_topo: 0.5, // Average topology contribution
        },
    );

    // Step 2: Generate ZKP proof of K-Vector validity
    let witness = KVectorWitness {
        k_r: agent.k_vector.k_r,
        k_a: agent.k_vector.k_a,
        k_i: agent.k_vector.k_i,
        k_p: agent.k_vector.k_p,
        k_m: agent.k_vector.k_m,
        k_s: agent.k_vector.k_s,
        k_h: agent.k_vector.k_h,
        k_topo: agent.k_vector.k_topo,
    };

    // Validate witness (would fail if out of range)
    assert!(witness.validate().is_ok());

    // Store commitment (in real system, proof would be generated and verified)
    let commitment = witness.commitment();
    assert_ne!(commitment, [0u8; 32]);

    // Step 3: Trust score from K-Vector
    assert!(agent.trust_score >= 0.0 && agent.trust_score <= 1.0);
    println!("Agent {} trust score: {:.4}", agent.id, agent.trust_score);

    // Trust score should qualify for network participation (threshold 0.3)
    assert!(agent.trust_score > 0.3, "Agent should qualify for participation");
}

/// Test multiple agents with different trust profiles
#[test]
fn test_multi_agent_trust_comparison() {
    // High-trust agent (long-term, active contributor)
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

    // Low-trust agent (new, minimal activity)
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

    println!("High-trust agent score: {:.4}", high_score);
    println!("Low-trust agent score: {:.4}", low_score);

    assert!(high_score > low_score, "High-trust agent should score higher");
    assert!(high_score > 0.8, "High-trust agent should be above 0.8");
    assert!(low_score < 0.4, "Low-trust agent should be below 0.4");
}

/// Test K-Vector evolution over time
#[test]
fn test_kvector_evolution() {
    let initial = KVector {
        k_r: 0.5,
        k_a: 0.3,
        k_i: 0.8,
        k_p: 0.6,
        k_m: 0.1,
        k_s: 0.4,
        k_h: 0.7,
        k_topo: 0.5,
    };

    // Simulate positive contributions
    let after_contributions = KVector {
        k_r: 0.6,    // Reputation increased
        k_a: 0.5,    // Activity increased
        k_i: 0.85,   // Integrity maintained
        k_p: 0.7,    // Performance improved
        k_m: 0.2,    // Membership grew
        k_s: 0.5,    // Stake increased
        k_h: 0.75,   // History improved
        k_topo: 0.6, // Network contribution grew
    };

    let initial_score = initial.trust_score();
    let evolved_score = after_contributions.trust_score();

    println!("Initial trust score: {:.4}", initial_score);
    println!("Evolved trust score: {:.4}", evolved_score);

    assert!(
        evolved_score > initial_score,
        "Trust should increase with positive contributions"
    );
}

/// Test that invalid K-Vectors are rejected by ZKP
#[test]
fn test_invalid_kvector_rejection() {
    // K-Vector with out-of-range values
    let invalid_witness = KVectorWitness {
        k_r: 1.5, // Invalid: > 1.0
        k_a: 0.5,
        k_i: 0.5,
        k_p: 0.5,
        k_m: 0.5,
        k_s: 0.5,
        k_h: 0.5,
        k_topo: 0.5,
    };

    let result = invalid_witness.validate();
    assert!(result.is_err(), "Should reject out-of-range K-Vector");

    // Negative value
    let negative_witness = KVectorWitness {
        k_r: -0.1, // Invalid: < 0.0
        k_a: 0.5,
        k_i: 0.5,
        k_p: 0.5,
        k_m: 0.5,
        k_s: 0.5,
        k_h: 0.5,
        k_topo: 0.5,
    };

    let result = negative_witness.validate();
    assert!(result.is_err(), "Should reject negative K-Vector component");
}

/// Test K-Vector commitment consistency
#[test]
fn test_commitment_consistency() {
    let witness = KVectorWitness {
        k_r: 0.75,
        k_a: 0.6,
        k_i: 0.85,
        k_p: 0.7,
        k_m: 0.4,
        k_s: 0.55,
        k_h: 0.8,
        k_topo: 0.65,
    };

    // Commitment should be deterministic
    let commit1 = witness.commitment();
    let commit2 = witness.commitment();
    assert_eq!(commit1, commit2, "Commitments should be identical");

    // Different values should produce different commitments
    let different_witness = KVectorWitness {
        k_r: 0.76, // Slightly different
        k_a: 0.6,
        k_i: 0.85,
        k_p: 0.7,
        k_m: 0.4,
        k_s: 0.55,
        k_h: 0.8,
        k_topo: 0.65,
    };

    let different_commit = different_witness.commitment();
    assert_ne!(commit1, different_commit, "Different values should produce different commitments");
}

/// Test weighted trust score calculation
#[test]
fn test_trust_score_weights() {
    // Test that weights are properly applied
    // Trust = 0.25×k_r + 0.15×k_a + 0.20×k_i + 0.15×k_p + 0.05×k_m + 0.10×k_s + 0.05×k_h + 0.05×k_topo

    let k_vector = KVector {
        k_r: 1.0,    // 0.25
        k_a: 1.0,    // 0.15
        k_i: 1.0,    // 0.20
        k_p: 1.0,    // 0.15
        k_m: 1.0,    // 0.05
        k_s: 1.0,    // 0.10
        k_h: 1.0,    // 0.05
        k_topo: 1.0, // 0.05
    };

    let score = k_vector.trust_score();
    assert!((score - 1.0).abs() < 0.0001, "All 1.0 should give trust score of 1.0");

    let half_vector = KVector {
        k_r: 0.5,
        k_a: 0.5,
        k_i: 0.5,
        k_p: 0.5,
        k_m: 0.5,
        k_s: 0.5,
        k_h: 0.5,
        k_topo: 0.5,
    };

    let half_score = half_vector.trust_score();
    assert!((half_score - 0.5).abs() < 0.0001, "All 0.5 should give trust score of 0.5");
}

// ============================================================
// Full ZK Proof Integration Tests
// ============================================================

use kvector_zkp::proof::KVectorRangeProof;

/// Test complete ZK proof generation and verification for an agent
#[test]
fn test_full_zkp_prove_verify_cycle() {
    // Agent's K-Vector (representing a moderately trusted node)
    let witness = KVectorWitness {
        k_r: 0.75,
        k_a: 0.60,
        k_i: 0.85,
        k_p: 0.70,
        k_m: 0.45,
        k_s: 0.55,
        k_h: 0.80,
        k_topo: 0.65,
    };

    // Generate ZK proof that all components are in [0, 1]
    let proof = KVectorRangeProof::prove(&witness)
        .expect("Proof generation should succeed");

    // Verify the proof
    proof.verify().expect("Proof verification should succeed");

    // Verify commitment matches
    assert_eq!(proof.commitment, witness.commitment());

    // Verify targets match scaled values
    assert_eq!(proof.targets, witness.scaled_values());

    println!("ZK proof generated and verified successfully");
    println!("Proof size: {} bytes", proof.size());
}

/// Test that ZK proof can be serialized, transmitted, and verified by another party
#[test]
fn test_zkp_serialization_for_network_transmission() {
    let witness = KVectorWitness {
        k_r: 0.8,
        k_a: 0.7,
        k_i: 0.9,
        k_p: 0.6,
        k_m: 0.5,
        k_s: 0.4,
        k_h: 0.85,
        k_topo: 0.75,
    };

    // Prover generates proof
    let original_proof = KVectorRangeProof::prove(&witness)
        .expect("Proof generation should succeed");

    // Simulate network transmission (serialize to bytes)
    let proof_bytes = original_proof.to_bytes();
    println!("Transmitting {} bytes over network", proof_bytes.len());

    // Verifier receives and deserializes
    let received_proof = KVectorRangeProof::from_bytes(&proof_bytes)
        .expect("Deserialization should succeed");

    // Verifier checks the proof
    received_proof.verify().expect("Remote verification should succeed");

    // Verify data integrity
    assert_eq!(original_proof.commitment, received_proof.commitment);
    assert_eq!(original_proof.targets, received_proof.targets);
}

/// Test that tampering with proof data is detected
#[test]
fn test_zkp_tamper_detection() {
    let witness = KVectorWitness {
        k_r: 0.75,
        k_a: 0.65,
        k_i: 0.85,
        k_p: 0.55,
        k_m: 0.45,
        k_s: 0.35,
        k_h: 0.95,
        k_topo: 0.25,
    };

    let mut proof = KVectorRangeProof::prove(&witness)
        .expect("Proof generation should succeed");

    // Attacker tries to change the commitment to claim different values
    proof.commitment[0] ^= 0xFF;

    // Verification should fail
    let result = proof.verify();
    assert!(result.is_err(), "Tampered proof should be rejected");
}

/// Test integration of ZK proof with trust score calculation
#[test]
fn test_zkp_trust_score_integration() {
    // Step 1: Agent creates K-Vector
    let k_vector = KVector {
        k_r: 0.82,
        k_a: 0.71,
        k_i: 0.93,
        k_p: 0.68,
        k_m: 0.55,
        k_s: 0.62,
        k_h: 0.88,
        k_topo: 0.73,
    };

    // Step 2: Convert to witness and generate ZK proof
    let witness = KVectorWitness {
        k_r: k_vector.k_r,
        k_a: k_vector.k_a,
        k_i: k_vector.k_i,
        k_p: k_vector.k_p,
        k_m: k_vector.k_m,
        k_s: k_vector.k_s,
        k_h: k_vector.k_h,
        k_topo: k_vector.k_topo,
    };

    let proof = KVectorRangeProof::prove(&witness)
        .expect("Proof generation should succeed");

    // Step 3: Verifier checks proof
    proof.verify().expect("Proof should be valid");

    // Step 4: Now verifier can trust the K-Vector is valid and compute trust score
    let trust_score = k_vector.trust_score();

    println!("Agent K-Vector verified via ZK proof");
    println!("Trust score: {:.4}", trust_score);

    // Agent should be highly trusted
    assert!(trust_score > 0.7, "High-integrity agent should have high trust");
}

/// Test multiple agents proving their K-Vectors in parallel (simulated)
#[test]
fn test_multi_agent_zkp_verification() {
    let agents = vec![
        ("alice", KVectorWitness {
            k_r: 0.9, k_a: 0.85, k_i: 0.95, k_p: 0.88,
            k_m: 0.7, k_s: 0.75, k_h: 0.92, k_topo: 0.8,
        }),
        ("bob", KVectorWitness {
            k_r: 0.6, k_a: 0.55, k_i: 0.7, k_p: 0.65,
            k_m: 0.4, k_s: 0.45, k_h: 0.6, k_topo: 0.5,
        }),
        ("charlie", KVectorWitness {
            k_r: 0.3, k_a: 0.25, k_i: 0.4, k_p: 0.35,
            k_m: 0.15, k_s: 0.2, k_h: 0.3, k_topo: 0.25,
        }),
    ];

    let mut verified_agents = Vec::new();

    for (name, witness) in &agents {
        // Each agent generates proof
        let proof = KVectorRangeProof::prove(witness)
            .expect(&format!("{} proof generation should succeed", name));

        // Network verifies proof
        proof.verify()
            .expect(&format!("{} proof should verify", name));

        // Compute trust score
        let k_vec = KVector {
            k_r: witness.k_r,
            k_a: witness.k_a,
            k_i: witness.k_i,
            k_p: witness.k_p,
            k_m: witness.k_m,
            k_s: witness.k_s,
            k_h: witness.k_h,
            k_topo: witness.k_topo,
        };

        verified_agents.push((name, k_vec.trust_score()));
    }

    // Print results
    println!("\nVerified Agent Trust Scores:");
    for (name, score) in &verified_agents {
        println!("  {}: {:.4}", name, score);
    }

    // Alice should have highest trust, Charlie lowest
    assert!(verified_agents[0].1 > verified_agents[1].1);
    assert!(verified_agents[1].1 > verified_agents[2].1);
}
