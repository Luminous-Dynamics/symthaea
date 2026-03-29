// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Genesis Simulation Tests for the DKG Truth Engine
//!
//! These tests demonstrate the Distributed Knowledge Graph's ability to
//! determine truth through social consensus and reputation-weighted attestations.
//!
//! # The "Sky Color" Scenario
//!
//! The canonical test of the Truth Engine:
//! - Alice (trustworthy) claims: "The sky is blue"
//! - Mallory (untrustworthy) claims: "The sky is green"
//! - Bob endorses Alice's claim
//!
//! Expected outcome: Alice's claim achieves higher confidence due to
//! social proof (Bob's endorsement), defeating Mallory's unattested lie.

use mycelix_sdk::dkg::{
    calculate_confidence, meets_threshold, Attestation, AttestationSet, AttestationType,
    ConfidenceInput, EpistemicType, TripleValue, VerifiableTriple,
};

#[test]
fn test_genesis_simulation_sky_color() {
    // === SETUP: The Participants ===
    // Alice: Honest agent claiming the truth
    // Mallory: Malicious agent spreading misinformation
    // Bob: Witness who endorses Alice's truthful claim

    let current_time = 1700000000u64;

    // === ACT 1: Alice claims "The sky is blue" ===
    let alice_claim = VerifiableTriple::new("sky", "color", TripleValue::String("blue".into()))
        .with_epistemic_type(EpistemicType::Empirical)
        .with_creator("alice")
        .with_timestamp(current_time);

    // === ACT 2: Mallory claims "The sky is green" (lie) ===
    let mallory_claim = VerifiableTriple::new("sky", "color", TripleValue::String("green".into()))
        .with_epistemic_type(EpistemicType::Empirical)
        .with_creator("mallory")
        .with_timestamp(current_time);

    // === ACT 3: Bob endorses Alice's claim ===
    // Create a test signature (in production this would be a real Ed25519 signature)
    let test_signature = vec![0u8; 64]; // Placeholder signature for test

    let mut alice_attestations = AttestationSet::new();
    alice_attestations.add(
        Attestation::endorse("alice_claim_hash", "bob", test_signature)
            .with_reputation(0.5) // Bob has neutral reputation
            .with_evidence("I looked outside and the sky is indeed blue.")
            .with_timestamp(current_time + 5),
    );

    // Mallory's claim has no endorsements (she's untrustworthy)
    let mallory_attestations = AttestationSet::new();

    // === TRUTH ENGINE EVALUATION ===

    // Calculate Alice's confidence
    let alice_input = ConfidenceInput {
        triple: &alice_claim,
        attestation_count: alice_attestations.attestations.len(),
        attester_reputations: &alice_attestations.endorsement_reputations(),
        current_time: current_time + 10,
        contradiction_weights: alice_attestations.contradiction_weight(),
        consciousness: None,
    };
    let alice_score = calculate_confidence(&alice_input);

    // Calculate Mallory's confidence
    let mallory_input = ConfidenceInput {
        triple: &mallory_claim,
        attestation_count: mallory_attestations.attestations.len(),
        attester_reputations: &mallory_attestations.endorsement_reputations(),
        current_time: current_time + 10,
        contradiction_weights: mallory_attestations.contradiction_weight(),
        consciousness: None,
    };
    let mallory_score = calculate_confidence(&mallory_input);

    // === ASSERTIONS: Truth Engine Verdict ===

    println!("\n=== GENESIS SIMULATION RESULTS ===");
    println!("Alice's 'blue' claim confidence: {:.4}", alice_score.score);
    println!(
        "Mallory's 'green' claim confidence: {:.4}",
        mallory_score.score
    );

    // 1. Alice's endorsed claim should beat Mallory's unattested claim
    assert!(
        alice_score.score > mallory_score.score,
        "Truth Engine FAILED: Alice's endorsed claim ({:.4}) should beat Mallory's unattested claim ({:.4})",
        alice_score.score, mallory_score.score
    );

    // 2. Alice's score should be above baseline (0.5)
    assert!(
        alice_score.score > 0.5,
        "Truth Engine FAILED: Endorsed claim ({:.4}) should be above baseline (0.5)",
        alice_score.score
    );

    // 3. Alice's claim should meet "low" verification threshold
    assert!(
        meets_threshold(alice_score.score, "low"),
        "Truth Engine FAILED: Alice's claim should meet 'low' threshold (0.40)"
    );

    // 4. Mallory's claim should NOT be degraded (it still has no attestations)
    assert!(
        mallory_score.degraded,
        "Truth Engine FAILED: Unattested claims should be marked as degraded"
    );

    println!("\n=== TRUTH ENGINE VERDICT ===");
    println!("'The sky is blue' (Alice) WINS over 'The sky is green' (Mallory)");
    println!("Social consensus (Bob's endorsement) elevated truth above lies!");
    println!("=== GENESIS SIMULATION SUCCESS ===\n");
}

#[test]
fn test_genesis_reputation_squared_voting() {
    // Test that high-reputation attesters have more influence than low-reputation ones
    // This verifies the reputation² voting principle from MATL

    let current_time = 1700000000u64;

    let claim = VerifiableTriple::new("test_subject", "test_predicate", TripleValue::Boolean(true))
        .with_epistemic_type(EpistemicType::Empirical)
        .with_timestamp(current_time);

    // Scenario A: One high-rep attester (0.9)
    let high_rep_input = ConfidenceInput {
        triple: &claim,
        attestation_count: 1,
        attester_reputations: &[0.9],
        current_time,
        contradiction_weights: 0.0,
        consciousness: None,
    };
    let high_rep_score = calculate_confidence(&high_rep_input);

    // Scenario B: One low-rep attester (0.2)
    let low_rep_input = ConfidenceInput {
        triple: &claim,
        attestation_count: 1,
        attester_reputations: &[0.2],
        current_time,
        contradiction_weights: 0.0,
        consciousness: None,
    };
    let low_rep_score = calculate_confidence(&low_rep_input);

    // High-rep attestation should produce higher confidence
    assert!(
        high_rep_score.score > low_rep_score.score,
        "High-rep attester ({:.4}) should produce higher confidence than low-rep ({:.4})",
        high_rep_score.score,
        low_rep_score.score
    );

    println!("Reputation² voting verified:");
    println!("  High-rep (0.9) confidence: {:.4}", high_rep_score.score);
    println!("  Low-rep (0.2) confidence: {:.4}", low_rep_score.score);
}

#[test]
fn test_genesis_challenge_reduces_confidence() {
    // Test that challenges from reputable agents reduce confidence

    let current_time = 1700000000u64;

    let claim = VerifiableTriple::new(
        "disputed_fact",
        "status",
        TripleValue::String("true".into()),
    )
    .with_epistemic_type(EpistemicType::Empirical)
    .with_timestamp(current_time);

    // Scenario A: No challenges
    let unchallenged = ConfidenceInput {
        triple: &claim,
        attestation_count: 2,
        attester_reputations: &[0.7, 0.6],
        current_time,
        contradiction_weights: 0.0,
        consciousness: None,
    };
    let unchallenged_score = calculate_confidence(&unchallenged);

    // Scenario B: With challenges
    let challenged = ConfidenceInput {
        triple: &claim,
        attestation_count: 2,
        attester_reputations: &[0.7, 0.6],
        current_time,
        contradiction_weights: 0.8, // High-rep challenge
        consciousness: None,
    };
    let challenged_score = calculate_confidence(&challenged);

    // Challenges should reduce confidence
    assert!(
        unchallenged_score.score > challenged_score.score,
        "Unchallenged claim ({:.4}) should have higher confidence than challenged ({:.4})",
        unchallenged_score.score,
        challenged_score.score
    );

    println!("Challenge mechanism verified:");
    println!("  Unchallenged confidence: {:.4}", unchallenged_score.score);
    println!("  Challenged confidence: {:.4}", challenged_score.score);
}

#[test]
fn test_genesis_time_decay() {
    // Test that older claims decay in confidence

    let current_time = 1700000000u64;
    let one_year_ago = current_time - (365 * 86400);

    let recent_claim = VerifiableTriple::new("fact", "value", TripleValue::Integer(42))
        .with_epistemic_type(EpistemicType::Empirical)
        .with_timestamp(current_time - 86400); // 1 day ago

    let old_claim = VerifiableTriple::new("fact", "value", TripleValue::Integer(42))
        .with_epistemic_type(EpistemicType::Empirical)
        .with_timestamp(one_year_ago);

    let recent_input = ConfidenceInput {
        triple: &recent_claim,
        attestation_count: 1,
        attester_reputations: &[0.7],
        current_time,
        contradiction_weights: 0.0,
        consciousness: None,
    };
    let recent_score = calculate_confidence(&recent_input);

    let old_input = ConfidenceInput {
        triple: &old_claim,
        attestation_count: 1,
        attester_reputations: &[0.7],
        current_time,
        contradiction_weights: 0.0,
        consciousness: None,
    };
    let old_score = calculate_confidence(&old_input);

    // Recent claims should have higher confidence
    assert!(
        recent_score.score > old_score.score,
        "Recent claim ({:.4}) should have higher confidence than old claim ({:.4})",
        recent_score.score,
        old_score.score
    );

    println!("Time decay verified:");
    println!("  Recent (1 day) confidence: {:.4}", recent_score.score);
    println!("  Old (1 year) confidence: {:.4}", old_score.score);
}

#[test]
fn test_genesis_epistemic_classification() {
    // Test that epistemic type affects confidence appropriately

    let current_time = 1700000000u64;

    // Same content, different epistemic types
    let empirical = VerifiableTriple::new("test", "value", TripleValue::Float(1.0))
        .with_epistemic_type(EpistemicType::Empirical)
        .with_timestamp(current_time);

    let normative = VerifiableTriple::new("test", "value", TripleValue::Float(1.0))
        .with_epistemic_type(EpistemicType::Normative)
        .with_timestamp(current_time);

    let metaphysical = VerifiableTriple::new("test", "value", TripleValue::Float(1.0))
        .with_epistemic_type(EpistemicType::Metaphysical)
        .with_timestamp(current_time);

    let reputations = [0.8, 0.7, 0.9, 0.6, 0.85];

    let emp_input = ConfidenceInput {
        triple: &empirical,
        attestation_count: 5,
        attester_reputations: &reputations,
        current_time,
        contradiction_weights: 0.0,
        consciousness: None,
    };
    let emp_score = calculate_confidence(&emp_input);

    let norm_input = ConfidenceInput {
        triple: &normative,
        attestation_count: 5,
        attester_reputations: &reputations,
        current_time,
        contradiction_weights: 0.0,
        consciousness: None,
    };
    let norm_score = calculate_confidence(&norm_input);

    let meta_input = ConfidenceInput {
        triple: &metaphysical,
        attestation_count: 5,
        attester_reputations: &reputations,
        current_time,
        contradiction_weights: 0.0,
        consciousness: None,
    };
    let meta_score = calculate_confidence(&meta_input);

    // Empirical > Normative > Metaphysical
    assert!(
        emp_score.score > norm_score.score,
        "Empirical ({:.4}) should have higher confidence than Normative ({:.4})",
        emp_score.score,
        norm_score.score
    );
    assert!(
        norm_score.score > meta_score.score,
        "Normative ({:.4}) should have higher confidence than Metaphysical ({:.4})",
        norm_score.score,
        meta_score.score
    );

    println!("Epistemic classification verified:");
    println!("  Empirical confidence: {:.4}", emp_score.score);
    println!("  Normative confidence: {:.4}", norm_score.score);
    println!("  Metaphysical confidence: {:.4}", meta_score.score);
}
