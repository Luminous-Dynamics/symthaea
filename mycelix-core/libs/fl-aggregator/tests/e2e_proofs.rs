//! End-to-end integration tests for zkSTARK proofs
//!
//! These tests verify complete workflows across multiple modules.

#![cfg(feature = "proofs")]

use fl_aggregator::proofs::{
    // Circuits
    RangeProof, MembershipProof, GradientIntegrityProof,
    IdentityAssuranceProof, VoteEligibilityProof,
    // Types
    ProofConfig, SecurityLevel, ProofAssuranceLevel,
    ProofIdentityFactor, ProofProposalType, ProofVoterProfile,
    // Integration
    VerifiedGradientSubmission, GradientProofBundle, ProofEnvelope,
    // Helpers
    build_merkle_tree,
};
use fl_aggregator::Gradient;
use ndarray::Array1;

fn test_config() -> ProofConfig {
    ProofConfig {
        security_level: SecurityLevel::Standard96,
        parallel: false,
        max_proof_size: 0,
    }
}

// =============================================================================
// E2E Test: Complete FL Round with Proofs
// =============================================================================

/// Simulates a complete FL round where:
/// 1. Multiple participants submit gradients with proofs
/// 2. Server verifies all proofs
/// 3. Gradients are aggregated only if proofs are valid
#[test]
fn test_fl_round_with_verified_submissions() {
    // Setup: Use gradients that match the working unit test pattern
    // The gradient proof works well with small integer-like values
    let gradient_data = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let max_norm = 5.0;

    // Generate proof bundle for this gradient
    let bundle = GradientProofBundle::generate(
        &gradient_data,
        max_norm,
        test_config(),
    ).expect("Proof generation should succeed");

    // Verify the bundle
    let result = bundle.verify().expect("Verification should complete");
    assert!(result.valid, "Proof should be valid");

    // Create verified submission
    let gradient: Gradient = Array1::from_vec(gradient_data.clone());
    let mut submission = VerifiedGradientSubmission::with_proofs(
        "node_1".to_string(),
        gradient,
        bundle,
    );

    // Verify the submission
    let submission_result = submission.verify().expect("Submission verification should complete");
    assert!(submission_result.valid, "Submission proof should be valid");
    assert!(submission.is_verified());

    println!("FL round completed with verified submission");
    println!("Gradient: {:?}", gradient_data);
    println!("Proof size: {} bytes", submission.proofs().unwrap().size());
}

/// Test that invalid gradients (exceeding norm) are rejected
#[test]
fn test_fl_round_rejects_invalid_gradient() {
    // Gradient with very large values that exceed norm
    let large_gradient = vec![10.0, 10.0, 10.0, 10.0, 10.0]; // L2 norm ≈ 22.4
    let max_norm = 1.0;

    // Generate proof - this should FAIL because norm exceeds max
    let result = GradientIntegrityProof::generate(
        &large_gradient,
        max_norm,
        test_config(),
    );

    // The proof generation should fail with an error
    assert!(result.is_err(), "Proof generation should fail for norm-exceeding gradient");

    // Match the error to verify it's about the norm
    match result {
        Err(e) => {
            let err_msg = format!("{:?}", e);
            assert!(err_msg.contains("norm") || err_msg.contains("Norm"),
                "Error should mention norm: {}", err_msg);
            println!("Invalid gradient correctly rejected: {}", err_msg);
        }
        Ok(_) => panic!("Should have failed"),
    }
}

// =============================================================================
// E2E Test: Identity Verification Flow
// =============================================================================

/// Complete identity verification flow:
/// 1. User builds identity with multiple factors
/// 2. Generates assurance level proof
/// 3. Verifier checks proof meets requirements
#[test]
fn test_identity_verification_flow() {
    // User identity with multiple factors from different categories
    let factors = vec![
        ProofIdentityFactor::new(0.35, 0, true),  // CryptoKey (category 0)
        ProofIdentityFactor::new(0.25, 1, true),  // SocialProof (category 1)
        ProofIdentityFactor::new(0.20, 2, true),  // BiometricHash (category 2)
        ProofIdentityFactor::new(0.15, 3, false), // Inactive factor - shouldn't count
    ];

    let did = "did:mycelix:user_abc123";

    // Generate proof for E2 level (requires 500 score)
    // Active factors: 0.35 + 0.25 + 0.20 = 0.80
    // Diversity bonus: 3 categories * 0.05 = 0.15
    // Total: 0.95 -> 950 scaled score
    let proof = IdentityAssuranceProof::generate(
        did,
        &factors,
        ProofAssuranceLevel::E2,
        test_config(),
    ).expect("Proof generation should succeed");

    // Verify
    let result = proof.verify().expect("Verification should complete");
    assert!(result.valid, "Identity proof should be valid");
    assert!(proof.meets_threshold(), "Should meet E2 threshold");
    assert!(proof.final_score() >= 500, "Score should be >= 500");

    // Try for E4 level (requires 900)
    let proof_e4 = IdentityAssuranceProof::generate(
        did,
        &factors,
        ProofAssuranceLevel::E4,
        test_config(),
    ).expect("Proof generation should succeed");

    // Should meet E4 as well since score is ~950
    assert!(proof_e4.meets_threshold(), "Should meet E4 threshold");

    println!("Identity verification completed");
    println!("  DID: {}", did);
    println!("  Final score: {}", proof.final_score());
    println!("  Meets E2: {}", proof.meets_threshold());
    println!("  Meets E4: {}", proof_e4.meets_threshold());
}

/// Test identity with insufficient assurance
#[test]
fn test_identity_insufficient_assurance() {
    // Low-value factors
    let factors = vec![
        ProofIdentityFactor::new(0.15, 0, true),  // Only 15% contribution
    ];

    // Try for E3 (requires 700)
    // Score: 0.15 + 0.05 diversity = 0.20 -> 200 scaled
    let proof = IdentityAssuranceProof::generate(
        "did:mycelix:low_assurance",
        &factors,
        ProofAssuranceLevel::E3,
        test_config(),
    ).expect("Proof generation should succeed");

    assert!(!proof.meets_threshold(), "Should NOT meet E3 threshold");

    let result = proof.verify().expect("Verification should complete");
    assert!(!result.valid, "Verification should fail for insufficient level");
}

// =============================================================================
// E2E Test: Governance Voting Flow
// =============================================================================

/// Complete governance voting flow:
/// 1. User has voter profile
/// 2. Generates eligibility proof for proposal type
/// 3. Submits vote with proof
/// 4. Vote is accepted/rejected based on proof
#[test]
fn test_governance_voting_flow() {
    // Well-qualified voter
    let voter = ProofVoterProfile {
        did: "did:mycelix:governor_alice".to_string(),
        assurance_level: 3,
        matl_score: 0.85,
        stake: 750.0,
        account_age_days: 120,
        participation_rate: 0.6,
        has_humanity_proof: true,
        fl_contributions: 30,
    };

    // Test eligibility for different proposal types
    let proposal_types = [
        (ProofProposalType::Standard, true),
        (ProofProposalType::Constitutional, true),
        (ProofProposalType::ModelGovernance, true),
        (ProofProposalType::Emergency, true),
        (ProofProposalType::Treasury, true),
    ];

    for (proposal_type, expected_eligible) in proposal_types {
        let proof = VoteEligibilityProof::generate(
            &voter,
            proposal_type,
            test_config(),
        ).expect("Proof generation should succeed");

        let result = proof.verify().expect("Verification should complete");

        assert_eq!(
            proof.is_eligible(), expected_eligible,
            "Eligibility for {:?} should be {}", proposal_type, expected_eligible
        );

        if expected_eligible {
            assert!(result.valid, "Valid eligibility proof should verify");
        }

        println!(
            "  {:?}: eligible={}, requirements_met={}/{}",
            proposal_type,
            proof.is_eligible(),
            proof.requirements_met(),
            proof.active_requirements()
        );
    }

    println!("Governance voting flow completed for {}", voter.did);
}

/// Test voter with partial eligibility
#[test]
fn test_governance_partial_eligibility() {
    // Voter missing some requirements
    let voter = ProofVoterProfile {
        did: "did:mycelix:new_user".to_string(),
        assurance_level: 1,        // Low assurance
        matl_score: 0.4,           // Moderate MATL
        stake: 50.0,               // Low stake
        account_age_days: 10,      // New account
        participation_rate: 0.1,   // Low participation
        has_humanity_proof: false, // No humanity proof
        fl_contributions: 0,       // No FL contributions
    };

    // Should be eligible for Standard
    let standard_proof = VoteEligibilityProof::generate(
        &voter,
        ProofProposalType::Standard,
        test_config(),
    ).expect("Proof generation should succeed");
    assert!(standard_proof.is_eligible(), "Should be eligible for Standard");

    // Should NOT be eligible for Constitutional (needs humanity proof, stake, etc.)
    let constitutional_proof = VoteEligibilityProof::generate(
        &voter,
        ProofProposalType::Constitutional,
        test_config(),
    ).expect("Proof generation should succeed");
    assert!(!constitutional_proof.is_eligible(), "Should NOT be eligible for Constitutional");

    // Should NOT be eligible for ModelGovernance (needs FL contributions)
    let model_proof = VoteEligibilityProof::generate(
        &voter,
        ProofProposalType::ModelGovernance,
        test_config(),
    ).expect("Proof generation should succeed");
    assert!(!model_proof.is_eligible(), "Should NOT be eligible for ModelGovernance");

    println!("Partial eligibility test completed");
    println!("  Standard: {}", standard_proof.is_eligible());
    println!("  Constitutional: {}", constitutional_proof.is_eligible());
    println!("  ModelGovernance: {}", model_proof.is_eligible());
}

// =============================================================================
// E2E Test: Cross-Module Integration
// =============================================================================

/// Test that combines identity, governance, and gradient proofs
/// Simulates: A verified user participates in FL and votes on model governance
#[test]
fn test_full_participant_flow() {
    // Step 1: User establishes identity
    let identity_factors = vec![
        ProofIdentityFactor::new(0.5, 0, true),
        ProofIdentityFactor::new(0.3, 1, true),
    ];

    let did = "did:mycelix:full_participant";

    let identity_proof = IdentityAssuranceProof::generate(
        did,
        &identity_factors,
        ProofAssuranceLevel::E2,
        test_config(),
    ).expect("Identity proof should succeed");

    assert!(identity_proof.meets_threshold(), "Should meet E2");

    // Step 2: User participates in FL with verified gradients (8 elements)
    let gradient_data = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let gradient_proof = GradientIntegrityProof::generate(
        &gradient_data,
        5.0,
        test_config(),
    ).expect("Gradient proof should succeed");

    let gradient_result = gradient_proof.verify().expect("Should verify");
    assert!(gradient_result.valid, "Gradient should be valid");

    // Step 3: After contributing, user can vote on model governance
    let voter = ProofVoterProfile {
        did: did.to_string(),
        assurance_level: 2,
        matl_score: 0.7,
        stake: 200.0,
        account_age_days: 60,
        participation_rate: 0.3,
        has_humanity_proof: true,
        fl_contributions: 15, // Has FL contributions!
    };

    let vote_proof = VoteEligibilityProof::generate(
        &voter,
        ProofProposalType::ModelGovernance,
        test_config(),
    ).expect("Vote proof should succeed");

    assert!(vote_proof.is_eligible(), "Should be eligible for model governance");

    println!("Full participant flow completed:");
    println!("  1. Identity verified at E2 (score: {})", identity_proof.final_score());
    println!("  2. Gradient contribution verified");
    println!("  3. Eligible to vote on model governance");
}

// =============================================================================
// E2E Test: Merkle Membership for Participant Registry
// =============================================================================

/// Test participant registry using Merkle proofs
/// Simulates: Verifying a participant is in the authorized set
#[test]
fn test_participant_registry_membership() {
    // Setup: Registry of authorized participants (as leaf hashes)
    let participants: Vec<[u8; 32]> = (0..8)
        .map(|i| {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            // Simulate hash of participant ID
            hash[1..5].copy_from_slice(b"part");
            hash
        })
        .collect();

    // Build Merkle tree - returns (root, paths_for_each_leaf)
    let (root, all_paths) = build_merkle_tree(&participants);

    // Participant 3 wants to prove membership
    let participant_idx = 3;
    let participant_leaf = participants[participant_idx];
    let path = all_paths[participant_idx].clone();

    // Generate membership proof
    let proof = MembershipProof::generate(
        participant_leaf,
        path,
        root,
        test_config(),
    ).expect("Membership proof should succeed");

    let result = proof.verify().expect("Verification should complete");
    assert!(result.valid, "Membership proof should be valid");

    println!("Participant registry membership verified");
    println!("  Root: {:?}...", &root[..8]);
    println!("  Participant leaf: {:?}...", &participant_leaf[..8]);
}

// =============================================================================
// E2E Test: Proof Size and Performance Characteristics
// =============================================================================

/// Test to document proof sizes across all circuit types
#[test]
fn test_proof_sizes() {
    println!("\n=== Proof Size Report ===\n");

    // RangeProof
    let range_proof = RangeProof::generate(500, 0, 1000, test_config()).unwrap();
    println!("RangeProof: {} bytes", range_proof.size());

    // MembershipProof
    let leaves: Vec<[u8; 32]> = (0..4).map(|i| { let mut h = [0u8; 32]; h[0] = i; h }).collect();
    let (root, all_paths) = build_merkle_tree(&leaves);
    let path = all_paths[1].clone();  // Get actual path for leaf 1
    let membership_proof = MembershipProof::generate(leaves[1], path, root, test_config()).unwrap();
    println!("MembershipProof (depth 2): {} bytes", membership_proof.size());

    // GradientIntegrityProof (small test gradient)
    let gradient: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let gradient_proof = GradientIntegrityProof::generate(&gradient, 5.0, test_config()).unwrap();
    println!("GradientIntegrityProof (8 elements): {} bytes", gradient_proof.size());

    // IdentityAssuranceProof
    let factors = vec![
        ProofIdentityFactor::new(0.4, 0, true),
        ProofIdentityFactor::new(0.3, 1, true),
    ];
    let identity_proof = IdentityAssuranceProof::generate(
        "did:test", &factors, ProofAssuranceLevel::E2, test_config()
    ).unwrap();
    println!("IdentityAssuranceProof: {} bytes", identity_proof.size());

    // VoteEligibilityProof
    let voter = ProofVoterProfile {
        did: "did:test".to_string(),
        assurance_level: 2,
        matl_score: 0.7,
        stake: 500.0,
        account_age_days: 100,
        participation_rate: 0.5,
        has_humanity_proof: true,
        fl_contributions: 20,
    };
    let vote_proof = VoteEligibilityProof::generate(
        &voter, ProofProposalType::Constitutional, test_config()
    ).unwrap();
    println!("VoteEligibilityProof: {} bytes", vote_proof.size());

    println!("\n=========================\n");
}

// =============================================================================
// E2E Test: Proof Serialization and Envelope Round-trips
// =============================================================================

/// Test complete serialization flow: proof -> envelope -> bytes -> envelope -> verify
#[test]
fn test_proof_envelope_roundtrip() {
    // Generate a range proof
    let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();

    // Create envelope
    let envelope = ProofEnvelope::from_range_proof(&proof).unwrap();

    // Serialize to bytes
    let bytes = envelope.to_bytes().unwrap();
    println!("Envelope size: {} bytes", bytes.len());

    // Deserialize
    let restored = ProofEnvelope::from_bytes(&bytes).unwrap();

    // Verify envelope matches
    assert_eq!(restored.proof_type, envelope.proof_type);
    assert_eq!(restored.version, envelope.version);
    assert_eq!(restored.proof_bytes.len(), envelope.proof_bytes.len());

    // Deserialize proof from envelope
    let restored_proof = RangeProof::from_bytes(&restored.proof_bytes).unwrap();

    // Verify restored proof
    let result = restored_proof.verify().unwrap();
    assert!(result.valid, "Restored proof should verify");

    println!("Proof envelope roundtrip successful");
}

/// Test gradient proof envelope roundtrip
#[test]
fn test_gradient_envelope_roundtrip() {
    let gradients = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let proof = GradientIntegrityProof::generate(&gradients, 5.0, test_config()).unwrap();

    let envelope = ProofEnvelope::from_gradient_proof(&proof).unwrap();
    let bytes = envelope.to_bytes().unwrap();
    let restored_envelope = ProofEnvelope::from_bytes(&bytes).unwrap();

    let restored_proof = GradientIntegrityProof::from_bytes(&restored_envelope.proof_bytes).unwrap();
    let result = restored_proof.verify().unwrap();

    assert!(result.valid, "Restored gradient proof should verify");
    assert_eq!(restored_proof.commitment(), proof.commitment());

    println!("Gradient envelope roundtrip successful");
}

// =============================================================================
// E2E Test: Compression (requires proofs-compressed feature)
// =============================================================================

#[cfg(feature = "proofs-compressed")]
#[test]
fn test_envelope_compression() {
    // Generate a proof
    let proof = RangeProof::generate(50, 0, 100, test_config()).unwrap();
    let envelope = ProofEnvelope::from_range_proof(&proof).unwrap();

    // Compress
    let compressed = envelope.compress().unwrap();
    assert!(compressed.compressed);

    // Check compression ratio
    let original_size = envelope.to_bytes().unwrap().len();
    let compressed_size = compressed.to_bytes().unwrap().len();

    println!("Original size: {} bytes", original_size);
    println!("Compressed size: {} bytes", compressed_size);
    println!("Compression ratio: {:.2}%", 100.0 * compressed_size as f64 / original_size as f64);

    // Decompress and verify
    let decompressed = compressed.decompress().unwrap();
    assert!(!decompressed.compressed);
    assert_eq!(decompressed.proof_bytes.len(), envelope.proof_bytes.len());

    // Verify the decompressed proof
    let restored_proof = RangeProof::from_bytes(&decompressed.proof_bytes).unwrap();
    let result = restored_proof.verify().unwrap();
    assert!(result.valid, "Decompressed proof should verify");

    println!("Compression roundtrip successful");
}

// =============================================================================
// E2E Test: Streaming Verification (requires proofs-streaming feature)
// =============================================================================

#[cfg(feature = "proofs-streaming")]
#[test]
fn test_streaming_verification() {
    use fl_aggregator::proofs::integration::streaming::{StreamStats, verify_batch_sync};

    // Create multiple proof envelopes
    let mut envelopes = Vec::new();
    for i in 0..5 {
        let proof = RangeProof::generate(50 + i * 10, 0, 100, test_config()).unwrap();
        let envelope = ProofEnvelope::from_range_proof(&proof).unwrap();
        envelopes.push(envelope);
    }

    println!("Created {} proof envelopes", envelopes.len());

    // Use sync verification for test simplicity
    let results = verify_batch_sync(&envelopes);

    let mut stats = StreamStats::default();
    for result in &results {
        stats.update(result);
        println!("Proof {}: valid={}, duration={:?}", result.index, result.valid, result.duration);
    }

    assert_eq!(stats.total_processed, 5);
    assert_eq!(stats.successful, 5);
    assert_eq!(stats.success_rate(), 1.0);

    println!("Streaming verification completed: {} proofs verified", stats.total_processed);
}

// =============================================================================
// E2E Test: Cross-Chain Anchoring (requires proofs-anchoring feature)
// =============================================================================

#[cfg(feature = "proofs-anchoring")]
#[test]
fn test_merkle_root_anchoring() {
    use fl_aggregator::proofs::integration::anchoring::create_anchor_merkle_root;

    // Create commitments from multiple proofs
    let mut commitments = Vec::new();

    for i in 0..4 {
        let proof = RangeProof::generate(50 + i * 10, 0, 100, test_config()).unwrap();
        let envelope = ProofEnvelope::from_range_proof(&proof).unwrap();

        // Use checksum as commitment
        let mut commitment = [0u8; 32];
        commitment[..4].copy_from_slice(&envelope.checksum);
        commitment[4..8].copy_from_slice(&(i as u32).to_le_bytes());
        // Fill rest with hash of proof bytes
        let hash = blake3::hash(&envelope.proof_bytes);
        commitment[8..].copy_from_slice(&hash.as_bytes()[..24]);

        commitments.push(commitment);
    }

    // Create merkle root
    let root = create_anchor_merkle_root(&commitments);

    // Verify root is non-zero
    assert_ne!(root, [0u8; 32]);

    // Same commitments should give same root
    let root2 = create_anchor_merkle_root(&commitments);
    assert_eq!(root, root2);

    println!("Merkle root for anchoring: {:?}...", &root[..8]);
    println!("Ready to anchor {} proof commitments", commitments.len());
}

// =============================================================================
// E2E Test: Telemetry (requires proofs-telemetry feature)
// =============================================================================

#[cfg(feature = "proofs-telemetry")]
#[test]
fn test_telemetry_collection() {
    use fl_aggregator::proofs::integration::telemetry::ProofTelemetry;
    use std::time::Duration;

    let telemetry = ProofTelemetry::new();

    // Simulate proof operations
    telemetry.record_generation("Range", Duration::from_millis(100), 15000);
    telemetry.record_generation("Range", Duration::from_millis(120), 15500);
    telemetry.record_verification("Range", Duration::from_millis(20), true);
    telemetry.record_verification("Range", Duration::from_millis(25), true);
    telemetry.record_verification("Gradient", Duration::from_millis(30), false);

    telemetry.record_cache_hit();
    telemetry.record_cache_hit();
    telemetry.record_cache_miss();

    let stats = telemetry.stats();

    assert_eq!(stats.proofs_generated, 2);
    assert_eq!(stats.proofs_verified, 3);
    assert_eq!(stats.verification_failures, 1);
    assert_eq!(stats.cache_hits, 2);
    assert_eq!(stats.cache_misses, 1);

    // Check Prometheus output
    let prometheus = stats.to_prometheus();
    assert!(prometheus.contains("proof_generation_total 2"));
    assert!(prometheus.contains("proof_verification_total 3"));

    println!("Telemetry stats:\n{}", prometheus);
}
