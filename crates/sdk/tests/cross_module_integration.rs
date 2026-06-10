// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Module Integration Tests
//!
//! Tests integration between major SDK modules:
//! - MATL (trust scoring)
//! - Agentic (AI agent framework)
//! - FL (federated learning)
//! - Epistemic (classification)

use mycelix_sdk::*;

/// Test that KVector can be created and trust_score computed
#[test]
fn test_kvector_creation_and_trust_score() {
    // Create KVector with all 10 dimensions
    let kv = KVector::new(
        0.8,  // k_r (reputation)
        0.6,  // k_a (activity)
        0.9,  // k_i (integrity)
        0.7,  // k_p (performance)
        0.3,  // k_m (membership)
        0.5,  // k_s (stake)
        0.6,  // k_h (historical)
        0.4,  // k_topo (topology)
        0.7,  // k_v (verification)
        0.65, // k_coherence (coherence)
    );

    let score = kv.trust_score();

    // Trust score should be weighted average in [0,1]
    assert!(score >= 0.0 && score <= 1.0);
    assert!(score > 0.5); // This high-quality vector should score well
}

/// Test that trust score can be converted to KREDIT allocation
#[test]
fn test_calculate_kredit_from_trust() {
    // Create a high-trust K-Vector
    let high_trust = KVector::new(0.9, 0.8, 1.0, 0.9, 0.5, 0.7, 0.8, 0.6, 0.9, 0.85);
    let trust_score = high_trust.trust_score();

    // calculate_kredit_from_trust is exported from agentic module
    let kredit = calculate_kredit_from_trust(trust_score);

    // High trust should yield high KREDIT
    assert!(kredit > 10000);
}

/// Test ProofOfGradientQuality creation and scoring
#[test]
fn test_proof_of_gradient_quality() {
    let pogq = ProofOfGradientQuality::new(0.95, 0.88, 0.12);

    // Check fields are clamped to [0,1]
    assert!(pogq.quality >= 0.0 && pogq.quality <= 1.0);
    assert!(pogq.consistency >= 0.0 && pogq.consistency <= 1.0);
    assert!(pogq.entropy >= 0.0 && pogq.entropy <= 1.0);

    // Test composite score calculation
    let composite = pogq.composite_score(0.75);
    assert!(composite >= 0.0 && composite <= 1.0);
}

/// Test CompositeTrustScore calculation from MATL
#[test]
fn test_composite_trust_score() {
    // Create components
    let pogq = ProofOfGradientQuality::new(0.9, 0.85, 0.1);
    let mut reputation = ReputationScore::new("agent1".to_string(), "test_source".to_string());
    // Give the reputation a high score
    for _ in 0..10 {
        reputation.record_positive();
    }

    // Calculate composite (0.4*pogq + 0.3*consistency + 0.3*reputation)
    let composite = CompositeScore::calculate(&pogq, &reputation);

    assert!(composite.score >= 0.0 && composite.score <= 1.0);
    assert!(composite.score > 0.6); // High quality inputs should yield high composite
}

/// Test epistemic classification levels
#[test]
fn test_epistemic_classification() {
    let claim = EpistemicClaim::new(
        "User verified via WebAuthn",
        EmpiricalLevel::E3Cryptographic,
        NormativeLevel::N2Network,
        MaterialityLevel::M2Persistent,
    );

    assert_eq!(claim.empirical, EmpiricalLevel::E3Cryptographic);
    assert_eq!(claim.normative, NormativeLevel::N2Network);
    assert_eq!(claim.materiality, MaterialityLevel::M2Persistent);

    // Test that it meets lower standards
    assert!(claim.meets_standard(EmpiricalLevel::E2PrivateVerify, NormativeLevel::N1Communal,));
}

/// Test gaming detection configuration
#[test]
fn test_gaming_detection_config() {
    let config = GamingDetectionConfig {
        min_actions_for_analysis: 10,
        suspicious_success_rate: 0.98,
        timing_variance_threshold: 2.5,
        burst_ratio_threshold: 3.0,
        quarantine_threshold: 0.7,
        analysis_window_secs: 3600,
    };

    assert!(config.min_actions_for_analysis > 0);
    assert!(config.suspicious_success_rate > 0.0 && config.suspicious_success_rate <= 1.0);
    assert!(config.quarantine_threshold > 0.0 && config.quarantine_threshold <= 1.0);
}

/// Test FedAvg aggregation method
#[test]
fn test_fedavg_aggregation() {
    // Create FL coordinator with default config
    let mut coordinator = FLCoordinator::new(FLConfig::default());

    // Register participants
    coordinator.register_participant("agent1".to_string());
    coordinator.register_participant("agent2".to_string());
    coordinator.register_participant("agent3".to_string());

    // Start a round
    coordinator.start_round().expect("Should start round");

    // Submit gradient updates
    let update1 = GradientUpdate::new("agent1".to_string(), 1, vec![0.1, 0.2], 100, 0.5);
    let update2 = GradientUpdate::new("agent2".to_string(), 1, vec![0.2, 0.3], 100, 0.4);
    let update3 = GradientUpdate::new("agent3".to_string(), 1, vec![0.15, 0.25], 100, 0.45);

    coordinator.submit_update(update1);
    coordinator.submit_update(update2);
    coordinator.submit_update(update3);

    // Aggregate round
    let result = coordinator
        .aggregate_round()
        .expect("Aggregation should succeed");

    // Should produce a weighted average
    assert_eq!(result.gradients.len(), 2);
    assert!(result.gradients[0] >= 0.0 && result.gradients[0] <= 1.0);
}

/// Test agent K-Vector update from behavior
#[test]
fn test_agent_kvector_update() {
    let agent = InstrumentalActor {
        agent_id: AgentId::generate(),
        sponsor_did: "did:test:sponsor".to_string(),
        agent_class: AgentClass::Supervised,
        kredit_balance: 5000,
        kredit_cap: 10000,
        constraints: AgentConstraints::default(),
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: 1000,
        last_activity: 1000,
        actions_this_hour: 0,
        k_vector: KVector::new_participant(),
        epistemic_stats: EpistemicStats::default(),
        output_history: vec![],
        uncertainty_calibration: UncertaintyCalibration::default(),
        pending_escalations: vec![],
    };

    // Compute trust score from new participant K-Vector
    let trust = agent.k_vector.trust_score();
    assert!(trust >= 0.0 && trust <= 1.0);

    // New participant should have baseline trust
    assert!(trust > 0.3 && trust < 0.7);
}

/// Test cross-module integration: Agent + KREDIT + Trust
#[test]
fn test_agent_kredit_trust_integration() {
    // Create agent with high-trust K-Vector
    let high_trust_kv = KVector::new(0.9, 0.8, 1.0, 0.9, 0.5, 0.7, 0.8, 0.6, 0.85, 0.85);

    let agent = InstrumentalActor {
        agent_id: AgentId::generate(),
        sponsor_did: "did:test:sponsor".to_string(),
        agent_class: AgentClass::Supervised,
        kredit_balance: 1000,
        kredit_cap: 1000,
        constraints: AgentConstraints::default(),
        behavior_log: vec![],
        status: AgentStatus::Active,
        created_at: 0,
        last_activity: 0,
        actions_this_hour: 0,
        k_vector: high_trust_kv,
        epistemic_stats: EpistemicStats::default(),
        output_history: vec![],
        uncertainty_calibration: UncertaintyCalibration::default(),
        pending_escalations: vec![],
    };

    // Compute trust-derived KREDIT cap
    let trust_score = agent.k_vector.trust_score();
    assert!(trust_score > 0.6); // High trust vector

    let kredit_from_trust = calculate_kredit_from_trust(trust_score);
    assert!(kredit_from_trust > agent.kredit_cap); // Should get more than default

    // Test that trust score is high
    assert!(trust_score > 0.6);
    assert!(kredit_from_trust > 10000);
}
